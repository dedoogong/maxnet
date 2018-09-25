#ifndef CONVOLUTIONAL_GPU_LAYER_H
#define CONVOLUTIONAL_GPU_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activation_gpu_layer.h"
#include "layer.h"
#include "network.h"
typedef layer convolutional_layer;
void make_convolutional_layer_gpu(convolutional_layer *l, int adam);
void forward_convolutional_layer_gpu(convolutional_layer layer, network net);
void backward_convolutional_layer_gpu(convolutional_layer layer, network net);

void push_convolutional_layer(convolutional_layer layer);
void pull_convolutional_layer(convolutional_layer layer);

void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
void update_convolutional_layer_gpu(layer l, update_args a); 
#endif

