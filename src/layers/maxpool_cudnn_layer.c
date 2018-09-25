#include "maxpool_cudnn_layer.h"
#include "maxpool_gpu_layer.h"
#include "utils.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include "cuda.h"
#include <stdio.h>
#include <time.h>

void cudnn_maxpool_setup(layer *l)
{
    cudnnSetTensor4dDescriptor(l->dSrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dDstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
 
 
    cudnnSetPooling2dDescriptor(l->poolDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, l->size, l->size, l->pad, l->pad, l->stride, l->stride); 
}

void make_maxpool_layer_cudnn(maxpool_layer *l)
{
		  l->forward_cudnn  = forward_maxpool_layer_gpu;//forward_maxpool_layer_cudnn;
		  l->backward_cudnn = backward_maxpool_layer_gpu;//backward_maxpool_layer_cudnn;  

      cudnnCreateTensorDescriptor(&l->normTensorDesc);
      cudnnCreateTensorDescriptor(&l->srcTensorDesc);
      cudnnCreateTensorDescriptor(&l->dstTensorDesc);
      cudnnCreateTensorDescriptor(&l->dSrcTensorDesc);
      cudnnCreateTensorDescriptor(&l->dDstTensorDesc);
      cudnnCreatePoolingDescriptor(&l->poolDesc);
      cudnn_maxpool_setup(l); 

} 

void resize_maxpool_layer_cudnn(maxpool_layer *l)
{
    cudnn_maxpool_setup(l);
} 
 
void forward_maxpool_layer_cudnn(maxpool_layer l, network net)
{
    float one = 1.f; 

    cudnnPoolingForward(cudnn_handle(), 
												l.poolDesc, 			// const cudnnPoolingDescriptor_t   poolingDesc,
												&one, 						// const void                      *alpha,
												l.srcTensorDesc,	// const cudnnTensorDescriptor_t    xDesc,
                        net.input_gpu, 		// const void                      *x,
												&one, 						// const void                      *beta,
												l.dstTensorDesc,	// const cudnnTensorDescriptor_t    yDesc,
												l.output_gpu);		// void                            *y
 
}

void backward_maxpool_layer_cudnn(maxpool_layer l, network net){

    float one = 1.f; 
		cudnnPoolingBackward(cudnn_handle(), 
											   l.poolDesc, 
												 &one, 
                         l.dstTensorDesc, 	// const cudnnTensorDescriptor_t yDesc
												 l.output_gpu, 			// const void 									 *y				
												 l.dDstTensorDesc, 	// const cudnnTensorDescriptor_t dyDesc
											   l.delta_gpu,				// const void										 *dy    
				                 l.srcTensorDesc, 	// const cudnnTensorDescriptor_t xDesc
												 net.input_gpu, 		// const void										 *x
												 &one,							// const void										 *beta
											   l.dSrcTensorDesc, 	// const cudnnTensorDescriptor_t dxDesc 
												 net.delta_gpu);    // void 												 *dx 		
 
}



