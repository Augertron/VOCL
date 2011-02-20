rm tranO
echo "g++ oclTranspose.cpp"
g++ oclTranspose.cpp transpose_gold.cpp timeRec.c -o tranO -O3 -I/usr/local/cuda/include -I/home/balaji/software/cuda/sdk/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc -I/home/balaji/software/cuda/sdk/NVIDIA_GPU_Computing_SDK/shared/inc -lOpenCL -L/home/balaji/software/cuda/sdk/NVIDIA_GPU_Computing_SDK/shared/lib -L/home/balaji/software/cuda/sdk/NVIDIA_GPU_Computing_SDK/OpenCL/common/lib -lshrutil_x86_64 -loclUtil_x86_64
