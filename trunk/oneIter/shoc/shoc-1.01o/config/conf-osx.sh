#!/bin/sh

sh ./configure \
    --with-opencl --with-cuda --with-mpi-includes="-I/opt/local/include/mpich2" \
    --with-mpi-libraries="-L/opt/local/lib -lmpich -lmpicxx -lpmpich" \
    --enable-m64

