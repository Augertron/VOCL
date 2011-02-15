mpicc -I/usr/local/cuda/include -c gpuv.c
ar rcs libGPUv.a gpuv.o
#mpicc -shared -I/usr/local/cuda/include -fPIC gpuv.o -o libGPUv.a
