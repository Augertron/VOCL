#!/bin/sh

echo ""
echo "Configuring SHOC assuming OpenMPI bin directory is available in the PATH."
echo "If it is not, this driver script will fail to configure SHOC properly."
echo ""

mpicxx -show > /dev/null || exit 1


OCL_ROOT=/opt/ati-stream-sdk-v2.2-lnx64

# Using OpenMPI
mpi_cxxflags="`mpicxx -showme:compile`"
mpi_ldflags="`mpicxx -showme:link`"


# do the actual configuration
sh ./configure \
    CPPFLAGS="-I$OCL_ROOT/include" \
    LDFLAGS="-L$OCL_ROOT/lib/x86_64" \
    --with-mpi-includes="$mpi_cxxflags" \
    --with-mpi-libraries="$mpi_ldflags" \
    --without-cuda \
    --disable-stability

# other useful options
#    --enable-m64
#    --disable-m64

