#ifndef ACTIVATION_GPU_LAYER_H
#define ACTIVATION_GPU_LAYER_H

#include "layer.h"
#include "network.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void activate_array_gpu(float *x, int n, ACTIVATION a);
void gradient_array_gpu(float *x, int n, ACTIVATION a, float *delta);
#endif
