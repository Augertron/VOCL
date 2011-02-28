#ifndef REDUCTION_KERNEL_H_
#define REDUCTION_KERNEL_H_

#include <cuda.h>

template <class T>
__global__ void
reduce(const T* __restrict__ g_idata, T* __restrict__ g_odata,
        const unsigned int n)
{

    const unsigned int tid = threadIdx.x;
    unsigned int i = (blockIdx.x*(blockDim.x*2)) + tid;
    const unsigned int gridSize = blockDim.x*2*gridDim.x;
    const unsigned int blockSize = blockDim.x;

    __shared__ T sdata[256];
    sdata[tid] = 0;

    // Reduce multiple elements per thread
    while (i < n)
    {
        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        i += gridSize;
    }
    __syncthreads();

    for (unsigned int s = blockSize / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}


#endif // REDUCTION_KERNEL_H_
