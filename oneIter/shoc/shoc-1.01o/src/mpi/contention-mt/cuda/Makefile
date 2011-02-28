SHOC_ROOT=../../../..
include $(SHOC_ROOT)/config/config.mk

VPATH = ..

BENCHMARKOBJ_CUDA= \
 MTBusCont_mpi.o 

BENCHMARKPROG_CUDA = $(patsubst %_mpi.o,$(SHOC_ROOT)/bin/EP/CUDA/%,$(BENCHMARKOBJ_CUDA))

COMMONOBJ_CUDA= \
 $(SHOC_ROOT)/src/cuda/level0/BusSpeedDownload_cuda.o \
 CUDADriver_cuda.o\
 main_mpi.o

ALLOBJ=$(COMMONOBJ_CUDA) $(BENCHMARKOBJ_CUDA)

LIBS+=-lSHOCCommonMPI -lSHOCCommon
#CUDA_LIBS+=-lSHOCCommonCUDA

CPPFLAGS+=-I${SHOC_ROOT}/src/mpi/common

# We will be using nvcc to drive the compile and link.  Nvcc does not know about
# the 'Wl' option accepted by the underlying g++ compiler, so we have to
# use nvcc's -X option.  If we do the natural thing (-Xlinker), the build
# fails because the -Wl option is not understood by the linker.  Thus
# we have to specify that it is a compiler option (-Xcompiler) even though
# we are linking.
CUDA_CPPFLAGS+=-Xcompiler ""
MPI_LIBS    = -Xcompiler "" -lSHOCCommonMPI -lstdc++

all: $(BENCHMARKPROG_CUDA)

$(SHOC_ROOT)/bin/EP/CUDA/%: %_mpi.o
	$(CUDA_CXX) -o $@ $(CXXFLAGS) $(CUDA_CPPFLAGS) $(CUDA_LDFLAGS) $(MPI_LDFLAGS) $(COMMONOBJ_CUDA) $< $(LIBS) ${MPI_LIBS} $(CUDA_LIBS)

$(BENCHMARKPROG_CUDA): $(COMMONOBJ_CUDA)

clean:
	${RM} $(ALLOBJ) $(BENCHMARKPROG_CUDA)

distclean: clean
	${RM} $(DEP) *~ .buildflags Makefile

include $(SHOC_ROOT)/config/targets.mk
