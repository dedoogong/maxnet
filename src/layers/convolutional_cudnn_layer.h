#ifndef CONVOLUTIONAL_CUDNN_LAYER_H
#define CONVOLUTIONAL_CUDNN_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activation_gpu_layer.h"
#include "layer.h"
#include "network.h"
typedef layer convolutional_layer;
void cudnn_convolutional_setup(layer *l); 
size_t get_workspace_size_cudnn(layer l);
void cudnn_convolutional_setup(layer *l);
void make_convolutional_layer_cudnn(convolutional_layer *l);
void resize_convolutional_layer_cudnn(convolutional_layer *l);
void forward_convolutional_layer_cudnn(convolutional_layer l, network net);
void backward_convolutional_layer_cudnn(convolutional_layer l, network net);
#endif

