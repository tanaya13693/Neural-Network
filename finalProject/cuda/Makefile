# C compiler
# CC = gcc
# CC_FLAGS = -g -O2


NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include 
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
# EXE	        = sgemm-tiled
OBJ	        = backprop.o facetrain.o
# backprop_kernel.o


backprop: $(OBJ)
	$(NVCC) $(OBJ) -o backprop $(LD_FLAGS)

# %.o: %.(cu|h)
	# $(NVCC) $(NVCC_FLAGS) $< -c

facetrain.o: facetrain.cu backprop_kernel.cu backprop.h
	$(NVCC) -c -o $@ facetrain.cu $(NVCC_FLAGS)
	
backprop.o: backprop.cu backprop.h
	$(NVCC) -c -o $@ backprop.cu $(NVCC_FLAGS)

# backprop_kernel.o: backprop_kernel.cu backprop.h
# 	$(NVCC) -c -o $@ backprop_kernel.cu $(NVCC_FLAGS)

# imagenet.o: imagenet.cu backprop.h
# 	$(NVCC) -c -o $@ imagenet.cu $(NVCC_FLAGS)


clean:
	rm -f *.o *~ backprop backprop_cuda.linkinfo
