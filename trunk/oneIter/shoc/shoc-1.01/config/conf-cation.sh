#!/bin/sh

echo ""
echo "Configuring SHOC assuming OpenMPI bin directory is available in the PATH."
echo "If it is not, this driver script will fail to configure SHOC properly."
echo ""

mpicxx -show > /dev/null || exit 1

# Using OpenMPI
mpi_cxxflags="`mpicxx -showme:compile`" || exit 1
mpi_ldflags="`mpicxx -showme:link`" || exit 1

# Cation has Fedora 10 and CUDA 3.0, and did not create an opencl.h file
# under /usr/include/CL as newer versions of CUDA will do.  Consequently,
# we have to explicitly tell SHOC where to find a usable CL/opencl.h file
# using CPPFLAGS.

# do the actual configuration
sh ./configure \
CPPFLAGS="-I/opt/cuda/include" \
    --with-mpi-includes="$mpi_cxxflags" \
    --with-mpi-libraries="$mpi_ldflags" \
    --disable-stability

# other useful options
#    --enable-m64
#    --disable-m64

