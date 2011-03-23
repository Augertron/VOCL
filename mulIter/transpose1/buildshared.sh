SDK_DIR=$HOME/NVIDIA_GPU_Computing_SDK
GPUV_DIR=$HOME/workplace/trunk/lib
echo $SDK_DIR
echo $GPUV_DIR
rm tranV
echo "g++ oclTranspose.cpp"
mpicxx oclTranspose.cpp transpose_gold.cpp timeRec.c -o tranV -O3 -I/usr/local/cuda/include -I$SDK_DIR/OpenCL/common/inc -I$SDK_DIR/shared/inc -L$GPUV_DIR -lGPUv -L$SDK_DIR/shared/lib -L$SDK_DIR/OpenCL/common/lib -lshrutil_x86_64 -loclUtil_x86_64
