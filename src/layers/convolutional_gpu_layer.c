#include "convolutional_gpu_layer.h"
#include "utils.h"
#include "batchnorm_gpu_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h> 

void make_convolutional_layer_gpu(convolutional_layer *l, int adam)
{
    l->forward_gpu = forward_convolutional_layer_gpu;
    l->backward_gpu = backward_convolutional_layer_gpu;
    l->update_gpu = update_convolutional_layer_gpu;
    if(l->batch_normalize){

        l->scales_gpu = cuda_make_array(l->scales, l->n);
        l->scale_updates_gpu = cuda_make_array(l->scale_updates, l->n);

        l->mean_gpu = cuda_make_array(l->mean, l->n);
        l->variance_gpu = cuda_make_array(l->variance, l->n);

        l->rolling_mean_gpu = cuda_make_array(l->mean, l->n);
        l->rolling_variance_gpu = cuda_make_array(l->variance, l->n);

        l->mean_delta_gpu = cuda_make_array(l->mean, l->n);
        l->variance_delta_gpu = cuda_make_array(l->variance, l->n);

        l->x_gpu = cuda_make_array(l->output, l->batch*l->out_h*l->out_w*l->n);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->out_h*l->out_w*l->n);
    }

    if (adam) {
        l->m_gpu = cuda_make_array(l->m, l->nweights);
        l->v_gpu = cuda_make_array(l->v, l->nweights);
        l->bias_m_gpu = cuda_make_array(l->bias_m, l->n);
        l->bias_v_gpu = cuda_make_array(l->bias_v, l->n);
        l->scale_m_gpu = cuda_make_array(l->scale_m, l->n);
        l->scale_v_gpu = cuda_make_array(l->scale_v, l->n);
    }

    l->weights_gpu = cuda_make_array(l->weights, l->nweights);
    l->weight_updates_gpu = cuda_make_array(l->weight_updates, l->nweights);

    l->biases_gpu = cuda_make_array(l->biases, l->n);
    l->bias_updates_gpu = cuda_make_array(l->bias_updates, l->n);

    l->delta_gpu = cuda_make_array(l->delta, l->batch*l->out_h*l->out_w*l->n);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->out_h*l->out_w*l->n);

}
 

void forward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
 
    int i, j;
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights_gpu + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output_gpu + (i*l.groups + j)*n*m;
            float *im = net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1){
                b = im;
            } else {
                im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    } 

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }

    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void resize_convolutional_layer_gpu(convolutional_layer *l, int w, int h)
{
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

    if(l->batch_normalize){
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }  
}


void backward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    if(l.smooth){
        smooth_layer(l, 5, l.smooth);
    }
    //constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);


    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    float *original_input = net.input_gpu; 

    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    int i, j;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.delta_gpu + (i*l.groups + j)*m*k;
            float *b = net.workspace;
            float *c = l.weight_updates_gpu + j*l.nweights/l.groups;

            float *im  = net.input_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;
            float *imd = net.delta_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;

            im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if (net.delta_gpu) {
                
                a = l.weights_gpu + j*l.nweights/l.groups;
                b = l.delta_gpu + (i*l.groups + j)*m*k;
                c = net.workspace;
                if (l.size == 1) {
                    c = imd;
                }

                gemm_gpu(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (l.size != 1) {
                    col2im_gpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
                }
                
            }
            
        }
     } 
}

void pull_convolutional_layer(convolutional_layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    cuda_pull_array(l.biases_gpu, l.biases, l.n);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.n);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void push_convolutional_layer(convolutional_layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
    cuda_push_array(l.biases_gpu, l.biases, l.n);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.n);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void update_convolutional_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, \
												a.B1, a.B2, a.eps, \
												decay, learning_rate, l.nweights, batch, a.t);

        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, \
												a.B1, a.B2, a.eps, \
												decay, learning_rate, l.n, batch, a.t);

        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, \
														a.B1, a.B2, a.eps, \
														decay, learning_rate, l.n, batch, a.t);
        }
    }else{
        axpy_gpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.nweights, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1);

        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);

        if(l.scales_gpu){
            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
        }
    }
    if(l.clip){
        constrain_gpu(l.nweights, l.clip, l.weights_gpu, 1);
    }
}

