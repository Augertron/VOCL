#!/bin/sh

echo ""
echo "Configuring SHOC assuming OpenMPI bin directory is available in the PATH."
echo "If it is not, this driver script will fail to configure SHOC properly."
echo ""

mpicxx -show > /dev/null || exit 1

CUDA_ROOT=/sw/analysis-x64/cuda/3.0b/sl5.0_binary

# Using OpenMPI
mpi_cxxflags="`mpicxx -showme:compile`"
mpi_ldflags="`mpicxx -showme:link`"


# do the actual configuration
sh ./configure \
CPPFLAGS="-I$CUDA_ROOT/include" \
PATH"=$CUDA_ROOT/bin:$PATH" \
    --with-mpi-includes="$mpi_cxxflags" \
    --with-mpi-libraries="$mpi_ldflags" \
    --disable-stability

# other useful options
#    --enable-m64
#    --disable-m64

