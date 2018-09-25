#include "maxnet.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "convolutional_cpu_layer.h"
extern void run_detector(int argc, char **argv, const char* mode);

int main(int argc, char **argv)
{
    //test_resize("data/bad.jpg");
    //test_box();
    //test_convolutional_layer();
    char *mode="gpu";
   
    if(argc < 2){
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }
    gpu_index = find_int_arg(argc, argv, "-i", 0);
    if(find_arg(argc, argv, "cpu")) {
        
        gpu_index = -1;
        mode = "cpu";
        printf("RUN IN %d MODE",RUN_CPU);
    }
    else if(find_arg(argc, argv, "gpu")) {
        mode = "gpu";
        printf("RUN IN %d MODE",RUN_CUDA);
    }
    else{
        mode = "cudnn";
        printf("RUN IN %d MODE",RUN_CUDNN);
    }

    if(gpu_index >= 0){
        cuda_set_device(gpu_index);
    }
    run_detector(argc, argv, mode);

    return 0;
}

