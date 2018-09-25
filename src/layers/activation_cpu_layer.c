#include "activation_cpu_layer.h"

 
char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case LOGISTIC:
            return "logistic"; 
        case RELU:
            return "relu"; 
        case TANH:
            return "tanh"; 
        case LEAKY:
            return "leaky"; 
        case LINEAR:
            return "linear";
        default:
            break;
    }
    return "relu";
}

ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "logistic")==0) return LOGISTIC; 
    if (strcmp(s, "relu")==0) return RELU; 
    if (strcmp(s, "leaky")==0) return LEAKY;
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "tanh")==0) return TANH; 
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

float activate(float x, ACTIVATION a)
{
    switch(a){ 
        case LOGISTIC:
            return logistic_activate(x); 
        case RELU:
            return relu_activate(x); 
        case LEAKY:
            return leaky_activate(x);
        case LINEAR:
            return linear_activate(x);
        case TANH:
            return tanh_activate(x); 
    }
    return 0;
}

void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = activate(x[i], a);
    }
}

float gradient(float x, ACTIVATION a)
{
    switch(a){ 
        case LOGISTIC:
            return logistic_gradient(x); 
        case RELU:
            return relu_gradient(x); 
        case LEAKY:
            return leaky_gradient(x);
        case LINEAR:
            return linear_gradient(x);
        case TANH:
            return tanh_gradient(x); 
    }
    return 0;
}

void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[i] *= gradient(x[i], a);
    }
} 
