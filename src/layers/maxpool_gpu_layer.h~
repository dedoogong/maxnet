#ifndef MAXPOOL_GPU_LAYER_H
#define MAXPOOL_GPU_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"
#include "maxpool_cpu_layer.h"
void make_maxpool_layer_gpu(maxpool_layer *l);
void forward_maxpool_layer_gpu(maxpool_layer l, network net);
void backward_maxpool_layer_gpu(maxpool_layer l, network net);

void resize_maxpool_layer_gpu(maxpool_layer *l);
#endif
