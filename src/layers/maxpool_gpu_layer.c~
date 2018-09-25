#include "maxpool_gpu_layer.h"
#include "cuda.h"

void make_maxpool_layer_gpu(maxpool_layer *l)
{
    l->forward_gpu = forward_maxpool_layer_gpu;
    l->backward_gpu= backward_maxpool_layer_gpu;
    int output_size = l->out_h * l->out_w * l->out_c * l->batch;
    l->indexes_gpu = cuda_make_int_array(0, output_size);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta, output_size);
    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n",
                                                                   l->size, l->size, l->stride, 
																										               l->w, l->h, l->c, 
																										               l->out_w, l->out_h, l->out_c);
    return l;
}

void resize_maxpool_layer_gpu(maxpool_layer *l)
{
    cuda_free((float *)l->indexes_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    int output_size = l->out_h * l->out_w * l->out_c * l->batch;
    l->indexes_gpu = cuda_make_int_array(0, output_size);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
}
