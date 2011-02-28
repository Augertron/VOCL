#!/bin/sh


echo ""
echo "Configuring SHOC assuming MPICH2 bin directory is available in the PATH"
echo "If it is not, this driver script will fail to configure SHOC properly."
echo "This script requires python - be sure it is also in your PATH!"
echo ""

mpicxx -show > /dev/null || exit 1
python -c "import sys" > /dev/null || exit 1


# Using MPICH2
# Much harder than OpenMPI because the output from 'mpicxx -show' does not 
# separate flags for compiling and linking, and because output contains 
# the compiler executable name, and because it may contain rpath specifications
# that are confusing to nvcc.
#
raw_mpi_compile_cmd="`mpicxx -show -c`"
mpi_compile_cmd_noc="`echo $raw_mpi_compile_cmd | sed 's/ -c / /' | sed 's/ -c$/ /'`"
raw_mpi_all_cmd="`mpicxx -show`"
mpi_cxxflags="`mpicxx -show -c | sed 's/^.* -c //'`"


echo "$mpi_compile_cmd_noc" > $$.tmp
echo "$raw_mpi_all_cmd" >> $$.tmp

mpi_ldflags_wrpath=$(cat $$.tmp | python -c "`cat <<EOF 
import sys
c = sys.stdin.readline()
f = sys.stdin.readline()
if f.startswith(c.rstrip()):
    print f[len(c.rstrip()):]
EOF
`"
)

rm $$.tmp

# try to strip rpath
mpi_ldflags=$(echo $mpi_ldflags_wrpath | python -c "`cat <<EOF
import sys
import re

def GenDropBit(l, i):
    m = re.search('^.*rpath$', l[i])
    if m:
        return False

    m = re.search( '^.*rpath=.*$', l[i] )
    if m:
        return False

    if (i > 0):
        m = re.search( '^.*rpath$', l[i-1] )
        if m:
            return False

    return True

f = sys.stdin.readline()
fs = f.split(' ')
fsb = [GenDropBit(fs, i) for i in range(len(fs))]
ff = [fs[i] for i in range(len(fs)) if fsb[i]]
l = ' '.join(ff)
print l
EOF
`"
)



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

