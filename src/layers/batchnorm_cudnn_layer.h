#ifndef BATCHNORM_CUDNN_LAYER_H
#define BATCHNORM_CUDNN_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"
 
void forward_batchnorm_layer_cudnn(layer l, network net);
void backward_batchnorm_layer_cudnn(layer l, network net);
#endif
