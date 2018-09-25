#include "reorg_layer.h"
#include "cuda.h"
#include "blas.h"

#include <stdio.h>


layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra,const char* mode)
{
    layer l = {0};
    l.type = REORG;
    l.batch = batch;
    l.stride = stride;
    l.extra = extra;
    l.h = h;
    l.w = w;
    l.c = c;
    l.flatten = flatten; 
    l.out_w = w/stride;
    l.out_h = h/stride;
    l.out_c = c*(stride*stride); 
    l.reverse = reverse;

    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    if(l.extra){
        l.out_w = l.out_h = l.out_c = 0;
        l.outputs = l.inputs + l.extra;
    }

    if(extra){
        fprintf(stderr, "reorg              %4d   ->  %4d\n",  l.inputs, l.outputs);
    } else {
        fprintf(stderr, "reorg              /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n",  stride, w, h, c, 
																																												 l.out_w, l.out_h, l.out_c);
    }
    int output_size = l.outputs * batch;
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));

    l.forward = forward_reorg_layer;
    l.backward = backward_reorg_layer;
    l.forward_gpu = forward_reorg_layer_gpu;
    l.backward_gpu = backward_reorg_layer_gpu;
    l.forward_cudnn = forward_reorg_layer_gpu;
    l.backward_cudnn = backward_reorg_layer_gpu;
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    return l;
}

void resize_reorg_layer_gpu(layer *l, int w, int h)
{
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    int output_size = l->outputs * l->batch;
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
}

void forward_reorg_layer(const layer l, network net)
{
    reorg_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 0, l.output);
}

void backward_reorg_layer(const layer l, network net)
{
    reorg_cpu(l.delta, l.w, l.h, l.c, l.batch, l.stride, 1, net.delta);
}
 
void forward_reorg_layer_gpu(layer l, network net)
{
    reorg_gpu(net.input_gpu, l.w, l.h, l.c, l.batch, l.stride, 0, l.output_gpu);
}

void backward_reorg_layer_gpu(layer l, network net)
{
    reorg_gpu(l.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, net.delta_gpu);
} 
