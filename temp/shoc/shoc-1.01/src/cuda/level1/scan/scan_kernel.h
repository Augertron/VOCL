#ifndef SCAN_KERNEL_H_
#define SCAN_KERNEL_H_

#include <cuda.h>

// This kernel code based in part on CUDPP.  Please see the notice in
// LICENSE_CUDPP.txt.

template <class T>
__global__ void
vectorAddUniform4(T *d_vector, const T *d_uniforms, const int n)
{
    __shared__ T uni[1];

    if (threadIdx.x == 0)
    {
        uni[0] = d_uniforms[blockIdx.x];
    }

    unsigned int address = threadIdx.x + (blockIdx.x *
            blockDim.x * 4);

    __syncthreads();

    // 4 elems per thread
    for (int i = 0; i < 4 && address < n; i++)
    {
        d_vector[address] += uni[0];
        address += blockDim.x;
    }
}

template <class T>
__device__ float scanLocalMem(const T val, T* s_data)
{
    // Shared mem is 512 floats long, set first half to 0
    int idx = threadIdx.x;
    s_data[idx] = 0;
    __syncthreads();

    // Set 2nd half to thread local sum (sum of the 4 elems from global mem)
    idx += blockDim.x; // += 256


    // Some of these __sync's are unnecessary due to warp synchronous
    // execution.  Right now these are left in to be consistent with
    // opencl version, since that has to execute on platforms where
    // thread groups are not synchronous (i.e. CPUs)
    T t;
    s_data[idx] = val;     __syncthreads();
    t = s_data[idx -  1];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  2];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  4];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  8];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 16];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 32];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 64];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 128]; __syncthreads();
    s_data[idx] += t;      __syncthreads();

    return s_data[idx-1];
}

template <class T, class vecT>
__global__ void
scan(T *g_odata, T* g_idata, T* g_blockSums, const int n,
     const bool fullBlock, const bool storeSum)
{
    __shared__ T s_data[512];

    // Load data into shared mem
    vecT tempData;
    vecT threadScanT;
    T res;
    vecT* inData  = (vecT*) g_idata;

    const int gid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int tid = threadIdx.x;
    const int i = gid * 4;

    // If possible, read from global mem in a float4 chunk
    if (fullBlock || i + 3 < n)
    {
        // scan the 4 elems read in from global
        tempData       = inData[gid];
        threadScanT.x = tempData.x;
        threadScanT.y = tempData.y + threadScanT.x;
        threadScanT.z = tempData.z + threadScanT.y;
        threadScanT.w = tempData.w + threadScanT.z;
        res = threadScanT.w;
    }
    else
    {   // if not, read individual floats, scan & store in lmem
        threadScanT.x = (i < n) ? g_idata[i] : 0.0f;
        threadScanT.y = ((i+1 < n) ? g_idata[i+1] : 0.0f) + threadScanT.x;
        threadScanT.z = ((i+2 < n) ? g_idata[i+2] : 0.0f) + threadScanT.y;
        threadScanT.w = ((i+3 < n) ? g_idata[i+3] : 0.0f) + threadScanT.z;
        res = threadScanT.w;
    }

    res = scanLocalMem(res, s_data);
    __syncthreads();

    // If we have to store the sum for the block, have the last work item
    // in the block write it out
    if (storeSum && tid == blockDim.x-1) {
        g_blockSums[blockIdx.x] = res + threadScanT.w;
    }

    // write results to global memory
    vecT* outData = (vecT*) g_odata;

    tempData.x = res;
    tempData.y = res + threadScanT.x;
    tempData.z = res + threadScanT.y;
    tempData.w = res + threadScanT.z;

    if (fullBlock || i + 3 < n)
    {
        outData[gid] = tempData;
    }
    else
    {
        if ( i    < n) { g_odata[i]   = tempData.x;
        if ((i+1) < n) { g_odata[i+1] = tempData.y;
        if ((i+2) < n) { g_odata[i+2] = tempData.z; } } }
    }
}

#endif // SCAN_KERNEL_H_
