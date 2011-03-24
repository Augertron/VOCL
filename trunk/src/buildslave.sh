#mpicc -L/home/shucaixiao/ati-stream-sdk-v2.2-lnx64/lib/x86_64 -lOpenCL -I/home/shucaixiao/ati-stream-sdk-v2.2-lnx64/include slaveOpenCL.c -o slave_process
mpicxx -lOpenCL -lpthread -O3 -I/usr/local/cuda/include slaveOpenCL.c -o slave_process
