mpicc -fPIC -O3 -I/usr/local/cuda/include -c gpuv.c
mpicc -shared -I/usr/local/cuda/include -fPIC gpuv.o -o libGPUv.so
rm gpuv.o
