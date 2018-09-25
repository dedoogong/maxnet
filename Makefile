GPU=1
CUDNN=1
OPENCV=0
OPENMP=0
DEBUG=0

ARCH= -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
      -gencode arch=compute_61,code=[sm_61,compute_61] 

#GeForce GTX 1080	6.1 (CUDA 8)	61
#GeForce GTX 1070	6.1 (CUDA 8)	61
#GeForce GTX 1060	6.1 (CUDA 8)	61
#GeForce GTX TITAN X	5.2				52
#GeForce GTX 980 Ti	5.2				  52
#GeForce GTX 980	5.2					  52 
#GeForce GTX 970	5.2					  52 
#GeForce GTX 960	5.2					  52 
#GeForce GTX 950	5.2					  52 

VPATH=./src/:./examples:./src/layers/
SLIB=libmaxnet.so
ALIB=libmaxnet.a
EXEC=maxnet
OBJDIR=./obj/

CC=gcc
NVCC=nvcc 
AR=ar
ARFLAGS=rcs
OPTS=-Ofast
LDFLAGS= -lm -pthread 
COMMON= -Iinclude/ -Isrc/ -Isrc/layers/
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC

ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` 
COMMON+= `pkg-config --cflags opencv` 
endif
 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand 
COMMON+= -DCUDNN 
CFLAGS+= -DCUDNN
LDFLAGS+= -lcudnn 

OBJ=gemm.o blas.o cuda.o \
		layer.o \
		convolutional_cpu_layer.o convolutional_gpu_layer.o convolutional_cudnn_layer.o \
		im2col.o col2im.o \
		batchnorm_cpu_layer.o batchnorm_gpu_layer.o batchnorm_cudnn_layer.o \
		activation_cpu_layer.o \
		maxpool_cpu_layer.o maxpool_gpu_layer.o maxpool_cudnn_layer.o\
		region_layer.o reorg_layer.o route_layer.o \
		list.o image.o data.o matrix.o network.o parser.o option_list.o box.o utils.o

EXECOBJA=detector.o maxnet.o 
LDFLAGS+= -lstdc++ 

OBJ+=blas_kernels.o \
		 convolutional_kernels.o \
		 im2col_kernels.o col2im_kernels.o \
		 activation_kernels.o \
		 maxpool_layer_kernels.o 

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/maxnet.h src/layers/*.h

all: obj backup results $(SLIB) $(ALIB) $(EXEC)

$(EXEC): $(EXECOBJ) $(ALIB)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj

backup:
	mkdir -p backup

results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) $(OBJDIR)/*

