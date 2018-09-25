#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activation_cpu_layer.h"
#include "layer.h"
#include "network.h"

typedef layer convolutional_layer;

void make_convolutional_layer_cpu(convolutional_layer* l, int adam);
void resize_convolutional_layer(convolutional_layer *layer, int w, int h);
void forward_convolutional_layer(const convolutional_layer layer, network net);
image *visualize_convolutional_layer(convolutional_layer layer, char *window, image *prev_weights);

void backward_convolutional_layer(convolutional_layer layer, network net);

void add_bias(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

image get_convolutional_image(convolutional_layer layer);
image get_convolutional_delta(convolutional_layer layer);
image get_convolutional_weight(convolutional_layer layer, int i);
size_t get_workspace_size(layer l,const char* mode);
int convolutional_out_height(convolutional_layer layer);
int convolutional_out_width(convolutional_layer layer);
void test_convolutional_layer();
#define BFLOP_UNIT 1000000000.
#endif

