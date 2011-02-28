#!/bin/sh

echo ""
echo "Configuring SHOC assuming OpenMPI bin directory is available in the PATH."
echo "If it is not, this driver script will fail to configure SHOC properly."
echo ""

mpicxx -show > /dev/null || exit 1

# Using OpenMPI
mpi_cxxflags="`mpicxx -showme:compile`"
mpi_ldflags="`mpicxx -showme:link`"


# do the actual configuration
#
# the configure script looks for CUDA using the PATH, but since OpenCL
# is library based, you have to explicitly specify CPPFLAGS to find
# the OpenCL headers.  You may also need to specify LDFLAGS, depending on
# whether the OpenCL libraries are installed in a location searched by
# the linker such as /usr/lib.
#
sh ./configure \
CPPFLAGS="-I/usr/local/cuda/include" \
    --with-mpi-includes="$mpi_cxxflags" \
    --with-mpi-libraries="$mpi_ldflags"

# other useful options
#    --enable-m64
#    --disable-m64

