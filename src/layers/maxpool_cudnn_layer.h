#ifndef MAXPOOL_CUDNN_LAYER_H
#define MAXPOOL_CUDNN_LAYER_H

#include "cuda.h"
#include "image.h"
#include "layer.h"
#include "network.h"
typedef layer maxpool_layer;
void cudnn_maxpool_setup(layer *l); 
size_t get_workspace_size_cudnn(layer l);
void make_maxpool_layer_cudnn(maxpool_layer *l);
void resize_maxpool_layer_cudnn(maxpool_layer *l);
void forward_maxpool_layer_cudnn(maxpool_layer l, network net);
void backward_maxpool_layer_cudnn(maxpool_layer l, network net);
#endif

