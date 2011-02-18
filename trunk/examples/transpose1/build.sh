rm tranO
#g++ oclTranspose.cpp transpose_gold.cpp -O3 -I/usr/local/cuda/include -I/home/scxiao/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc -I/home/scxiao/NVIDIA_GPU_Computing_SDK/shared/inc -lOpenCL -L/home/scxiao/NVIDIA_GPU_Computing_SDK/shared/lib -L/home/scxiao/NVIDIA_GPU_Computing_SDK/OpenCL/common/lib -lshrutil_x86_64 -loclUtil_x86_64
echo "g++ oclTranspose$1.cpp"
g++ oclTranspose$1.cpp transpose_gold.cpp timeRec.c -o tranO -O3 -I/usr/local/cuda/include -I/home/scxiao/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc -I/home/scxiao/NVIDIA_GPU_Computing_SDK/shared/inc -lOpenCL -L/home/scxiao/NVIDIA_GPU_Computing_SDK/shared/lib -L/home/scxiao/NVIDIA_GPU_Computing_SDK/OpenCL/common/lib -lshrutil_x86_64 -loclUtil_x86_64
