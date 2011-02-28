#ifdef SINGLE_PRECISION
#define FPTYPE float
#define FPVECTYPE float4
#elif K_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define FPTYPE double
#define FPVECTYPE double4
#elif AMD_DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#define FPTYPE double
#define FPVECTYPE double4
#endif

// This kernel code based in part on CUDPP.  Please see the notice in
// LICENSE_CUDPP.txt.
__kernel void
addUniform(__global FPTYPE *d_vector, __global const FPTYPE *d_uniforms,
           const int n)
{
    __local FPTYPE uni[1];

    if (get_local_id(0) == 0)
    {
        uni[0] = d_uniforms[get_group_id(0)];
    }

    unsigned int address = get_local_id(0) + (get_group_id(0) *
            get_local_size(0) * 4);

    barrier(CLK_LOCAL_MEM_FENCE);

    // 4 elems per thread
    for (int i = 0; i < 4 && address < n; i++)
    {
        d_vector[address] += uni[0];
        address += get_local_size(0);
    }
}

inline FPTYPE scanLocalMem(const FPTYPE val, __local FPTYPE* s_data)
{
    // Shared mem is 512 floats long, set first half to 0
    int idx = get_local_id(0);
    s_data[idx] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Set 2nd half to thread local sum (sum of the 4 elems from global mem)
    idx += get_local_size(0); // += 256
    FPTYPE t;
    s_data[idx] = val;     barrier(CLK_LOCAL_MEM_FENCE);

    // Do the scan
    t = s_data[idx -  1];  barrier(CLK_LOCAL_MEM_FENCE);
    s_data[idx] += t;      barrier(CLK_LOCAL_MEM_FENCE);
    t = s_data[idx -  2];  barrier(CLK_LOCAL_MEM_FENCE);
    s_data[idx] += t;      barrier(CLK_LOCAL_MEM_FENCE);
    t = s_data[idx -  4];  barrier(CLK_LOCAL_MEM_FENCE);
    s_data[idx] += t;      barrier(CLK_LOCAL_MEM_FENCE);
    t = s_data[idx -  8];  barrier(CLK_LOCAL_MEM_FENCE);
    s_data[idx] += t;      barrier(CLK_LOCAL_MEM_FENCE);
    t = s_data[idx - 16];  barrier(CLK_LOCAL_MEM_FENCE);
    s_data[idx] += t;      barrier(CLK_LOCAL_MEM_FENCE);
    t = s_data[idx - 32];  barrier(CLK_LOCAL_MEM_FENCE);
    s_data[idx] += t;      barrier(CLK_LOCAL_MEM_FENCE);
    t = s_data[idx - 64];  barrier(CLK_LOCAL_MEM_FENCE);
    s_data[idx] += t;      barrier(CLK_LOCAL_MEM_FENCE);
    t = s_data[idx - 128]; barrier(CLK_LOCAL_MEM_FENCE);
    s_data[idx] += t;      barrier(CLK_LOCAL_MEM_FENCE);

    return s_data[idx-1];
}

__kernel void scan(__global FPTYPE *g_odata, __global FPTYPE *g_idata,
        __global FPTYPE *g_blockSums, const int n, const int fullBlock,
        const int storeSum)
{
    __local FPTYPE s_data[512];

    // Load data into shared mem
    FPVECTYPE tempData;
    FPVECTYPE threadScanT;
    FPTYPE res;
    __global FPVECTYPE* inData  = (__global FPVECTYPE*) g_idata;

    const int gid = get_global_id(0);
    const int tid = get_local_id(0);
    const int i = gid * 4;

    // If possible, read from global mem in a FPVECTYPE chunk
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
    barrier(CLK_LOCAL_MEM_FENCE);

    res = scanLocalMem(res, s_data);
    barrier(CLK_LOCAL_MEM_FENCE);

    // If we have to store the sum for the block, have the last work item 
    // in the block write it out
    if (storeSum && tid == get_local_size(0)-1) {
        g_blockSums[get_group_id(0)] = res + threadScanT.w;
    }

    // write results to global memory
    __global FPVECTYPE* outData = (__global FPVECTYPE*) g_odata;
     
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
