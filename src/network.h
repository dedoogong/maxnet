#ifndef NETWORK_H
#define NETWORK_H
#include "maxnet.h"

#include "image.h"
#include "layer.h"
#include "data.h"

void pull_network_output(network *net);

void compare_networks(network *n1, network *n2, data d, const char* mode);
char *get_layer_string(LAYER_TYPE a);

network *make_network(int n);


float network_accuracy_multi(network *net, data d, int n, const char* mode);
int get_predicted_class_network(network *net);
void print_network(network *net);
void calc_network_cost(network *net);

#endif

