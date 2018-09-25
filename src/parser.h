#ifndef PARSER_H
#define PARSER_H
#include "maxnet.h"
#include "network.h"

void save_network(network net, char *filename);
void save_weights_double(network net, char *filename);
void layer_init(layer* l, size_params params, int size, int padding, int stride);
void layer_output_delta_alloc(layer* l,int resize);
#endif
