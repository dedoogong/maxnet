#include "convolutional_cpu_layer.h"
#include "utils.h"
#include "batchnorm_cpu_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

image get_convolutional_image(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.output);
}

image get_convolutional_delta(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.delta);
}

size_t get_workspace_size(layer l,const char* mode){
    if(0==strcmp(mode,"cudnn")){
        return get_workspace_size_cudnn(l);
    }  
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(float);
}

void make_convolutional_layer_cpu(convolutional_layer* l, int adam)
{
    if(l->batch_normalize){
        l->scales = calloc(l->n, sizeof(float));
        l->scale_updates = calloc(l->n, sizeof(float));
        for(int i = 0; i < l->n; ++i){
            l->scales[i] = 1;
        }

        l->mean = calloc(l->n, sizeof(float));
        l->variance = calloc(l->n, sizeof(float));

        l->mean_delta = calloc(l->n, sizeof(float));
        l->variance_delta = calloc(l->n, sizeof(float));

        l->rolling_mean = calloc(l->n, sizeof(float));
        l->rolling_variance = calloc(l->n, sizeof(float));
        l->x = calloc(l->batch*l->outputs, sizeof(float));
        l->x_norm = calloc(l->batch*l->outputs, sizeof(float));
    }
    if(adam){
        l->m = calloc(l->nweights, sizeof(float));
        l->v = calloc(l->nweights, sizeof(float));
        l->bias_m = calloc(l->n, sizeof(float));
        l->scale_m = calloc(l->n, sizeof(float));
        l->bias_v = calloc(l->n, sizeof(float));
        l->scale_v = calloc(l->n, sizeof(float));
    }

    l->forward = forward_convolutional_layer;
    l->backward = backward_convolutional_layer;  
}

void forward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output + (i*l.groups + j)*n*m;
            float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            
            if (l.size == 1) {
                b = im;
                //printf("3\n");
            } else {
                //printf("4\n");
                im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            //printf("-----\n");
            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
    }

    activate_array(l.output, l.outputs*l.batch, l.activation);
if(0){
     //l.outputs = l.out_h * l.out_w * l.out_c;
     //l.inputs = l.w * l.h * l.c;
     //l.output = calloc(l.batch*l.outputs, sizeof(float));
     int i,j,k,b;
     for(b = 0; b < l.batch; ++b){
         for(i = 0; i < l.out_c; ++i){
             for(j = 0; j < l.out_h; ++j){
                 for(k = 0; k < l.out_w; ++k){
                     //printf("%f ",l.output[b*l.out_c*l.out_h*l.out_w+i*l.out_h*l.out_w+j*l.out_w+k]);
                 }
                 printf("\n");
             }
             printf("\n");
         }
         printf("\n");
     }
     printf("\n");
     printf("Done\n");
  }
     
}
void resize_convolutional_layer_cpu(convolutional_layer *l, int w, int h,const char*mode)
{
    l->workspace_size = get_workspace_size(*l, mode);
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}


void backward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;
    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.delta + (i*l.groups + j)*m*k;
            float *b = net.workspace;
            float *c = l.weight_updates + j*l.nweights/l.groups;

            float *im  = net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            float *imd = net.delta + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if(l.size == 1){
                b = im;
            } else {
                im2col_cpu(im, l.c/l.groups, l.h, l.w, 
                        l.size, l.stride, l.pad, b);
            }

            gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if (net.delta) {
                a = l.weights + j*l.nweights/l.groups;
                b = l.delta + (i*l.groups + j)*m*k;
                c = net.workspace;
                if (l.size == 1) {
                    c = imd;
                }

                gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (l.size != 1) {
                    col2im_cpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
                }
            }
        }
    }
}

image get_convolutional_weight(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c/l.groups;
    return float_to_image(w,h,c,l.weights+i*h*w*c);
}


image *get_weights(convolutional_layer l)
{
    image *weights = calloc(l.n, sizeof(image));
    int i;
    for(i = 0; i < l.n; ++i){
        weights[i] = copy_image(get_convolutional_weight(l, i));
        normalize_image(weights[i]);
        /*
           char buff[256];
           sprintf(buff, "filter%d", i);
           save_image(weights[i], buff);
         */
    }
    //error("hey");
    return weights;
}
 

void test_convolutional_layer()
{
    //make_convolutional_layer                      (b, h, w, c, n, groups, size, stride, padding, activation, int batch_normalize, int adam)
    int N = 1;
    int size = 3;
    int stride = 1;
    int pad = 1;
    int padding = 0;
    int groups = 1;
    if(pad) padding = size/2;

    convolutional_layer l = {0};
    char *activation_s = "leaky";
    l.activation = get_activation(activation_s);
    
    l.type = CONVOLUTIONAL;
    size_params params;
    
    params.h = 5;
    params.w = 5;
    params.c = 3;
    params.inputs = 75;
    params.batch = 1;
    params.time_steps = 1;
    //params.net = net;

		layer_init(&l,params,size,padding,stride);
    if(!(l.h && l.w && l.c)) error("input should be 3D tensor!");
    l.n = N;
		l.groups = groups;

    int out_w = ((l.w + 2*l.pad - l.size) / l.stride + 1);
    int out_h = ((l.h + 2*l.pad - l.size) / l.stride + 1);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = N;
    l.outputs = l.out_h * l.out_w * l.out_c;

    layer_output_delta_alloc(&l, 0);

    l.batch_normalize = 0;

    l.weights = calloc(l.c/groups*N*size*size, sizeof(float));
    l.weight_updates = calloc(l.c/groups*N*size*size, sizeof(float));

    l.biases = calloc(N, sizeof(float));
    l.bias_updates = calloc(N, sizeof(float));

    l.nweights = l.c/groups*N*size*size;
    l.nbiases = N;

    make_convolutional_layer_cpu(&l, 0);
    l.batch_normalize = 0;
    float data[] = {1,1,1,1,1,
                    1,1,1,1,1,
                    1,1,1,1,1,
                    1,1,1,1,1,
                    1,1,1,1,1,
                    2,2,2,2,2,
                    2,2,2,2,2,
                    2,2,2,2,2,
                    2,2,2,2,2,
                    2,2,2,2,2,
                    3,3,3,3,3,
                    3,3,3,3,3,
                    3,3,3,3,3,
                    3,3,3,3,3,
                    3,3,3,3,3};
    network net;
    net.input = data;
    forward_convolutional_layer(l, net);
    #include "layer.h"
    #include "network.h"
    free_layer(l);
}

