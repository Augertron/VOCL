const char *cl_source_scan =
"#ifdef SINGLE_PRECISION\n"
"#define FPTYPE float\n"
"#define FPVECTYPE float4\n"
"#elif K_DOUBLE_PRECISION\n"
"#pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
"#define FPTYPE double\n"
"#define FPVECTYPE double4\n"
"#elif AMD_DOUBLE_PRECISION\n"
"#pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
"#define FPTYPE double\n"
"#define FPVECTYPE double4\n"
"#endif\n"
"\n"
"// This kernel code based in part on CUDPP.  Please see the notice in\n"
"// LICENSE_CUDPP.txt.\n"
"__kernel void\n"
"addUniform(__global FPTYPE *d_vector, __global const FPTYPE *d_uniforms,\n"
"           const int n)\n"
"{\n"
"    __local FPTYPE uni[1];\n"
"\n"
"    if (get_local_id(0) == 0)\n"
"    {\n"
"        uni[0] = d_uniforms[get_group_id(0)];\n"
"    }\n"
"\n"
"    unsigned int address = get_local_id(0) + (get_group_id(0) *\n"
"            get_local_size(0) * 4);\n"
"\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    // 4 elems per thread\n"
"    for (int i = 0; i < 4 && address < n; i++)\n"
"    {\n"
"        d_vector[address] += uni[0];\n"
"        address += get_local_size(0);\n"
"    }\n"
"}\n"
"\n"
"inline FPTYPE scanLocalMem(const FPTYPE val, __local FPTYPE* s_data)\n"
"{\n"
"    // Shared mem is 512 floats long, set first half to 0\n"
"    int idx = get_local_id(0);\n"
"    s_data[idx] = 0.0f;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    \n"
"    // Set 2nd half to thread local sum (sum of the 4 elems from global mem)\n"
"    idx += get_local_size(0); // += 256\n"
"    FPTYPE t;\n"
"    s_data[idx] = val;     barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    // Do the scan\n"
"    t = s_data[idx -  1];  barrier(CLK_LOCAL_MEM_FENCE);\n"
"    s_data[idx] += t;      barrier(CLK_LOCAL_MEM_FENCE);\n"
"    t = s_data[idx -  2];  barrier(CLK_LOCAL_MEM_FENCE);\n"
"    s_data[idx] += t;      barrier(CLK_LOCAL_MEM_FENCE);\n"
"    t = s_data[idx -  4];  barrier(CLK_LOCAL_MEM_FENCE);\n"
"    s_data[idx] += t;      barrier(CLK_LOCAL_MEM_FENCE);\n"
"    t = s_data[idx -  8];  barrier(CLK_LOCAL_MEM_FENCE);\n"
"    s_data[idx] += t;      barrier(CLK_LOCAL_MEM_FENCE);\n"
"    t = s_data[idx - 16];  barrier(CLK_LOCAL_MEM_FENCE);\n"
"    s_data[idx] += t;      barrier(CLK_LOCAL_MEM_FENCE);\n"
"    t = s_data[idx - 32];  barrier(CLK_LOCAL_MEM_FENCE);\n"
"    s_data[idx] += t;      barrier(CLK_LOCAL_MEM_FENCE);\n"
"    t = s_data[idx - 64];  barrier(CLK_LOCAL_MEM_FENCE);\n"
"    s_data[idx] += t;      barrier(CLK_LOCAL_MEM_FENCE);\n"
"    t = s_data[idx - 128]; barrier(CLK_LOCAL_MEM_FENCE);\n"
"    s_data[idx] += t;      barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    return s_data[idx-1];\n"
"}\n"
"\n"
"__kernel void scan(__global FPTYPE *g_odata, __global FPTYPE *g_idata,\n"
"        __global FPTYPE *g_blockSums, const int n, const int fullBlock,\n"
"        const int storeSum)\n"
"{\n"
"    __local FPTYPE s_data[512];\n"
"\n"
"    // Load data into shared mem\n"
"    FPVECTYPE tempData;\n"
"    FPVECTYPE threadScanT;\n"
"    FPTYPE res;\n"
"    __global FPVECTYPE* inData  = (__global FPVECTYPE*) g_idata;\n"
"\n"
"    const int gid = get_global_id(0);\n"
"    const int tid = get_local_id(0);\n"
"    const int i = gid * 4;\n"
"\n"
"    // If possible, read from global mem in a FPVECTYPE chunk\n"
"    if (fullBlock || i + 3 < n)\n"
"    {\n"
"        // scan the 4 elems read in from global\n"
"        tempData       = inData[gid];\n"
"        threadScanT.x = tempData.x;\n"
"        threadScanT.y = tempData.y + threadScanT.x;\n"
"        threadScanT.z = tempData.z + threadScanT.y;\n"
"        threadScanT.w = tempData.w + threadScanT.z;\n"
"        res = threadScanT.w;\n"
"    }\n"
"    else\n"
"    {   // if not, read individual floats, scan & store in lmem\n"
"        threadScanT.x = (i < n) ? g_idata[i] : 0.0f;\n"
"        threadScanT.y = ((i+1 < n) ? g_idata[i+1] : 0.0f) + threadScanT.x;\n"
"        threadScanT.z = ((i+2 < n) ? g_idata[i+2] : 0.0f) + threadScanT.y;\n"
"        threadScanT.w = ((i+3 < n) ? g_idata[i+3] : 0.0f) + threadScanT.z;\n"
"        res = threadScanT.w;\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    res = scanLocalMem(res, s_data);\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    // If we have to store the sum for the block, have the last work item \n"
"    // in the block write it out\n"
"    if (storeSum && tid == get_local_size(0)-1) {\n"
"        g_blockSums[get_group_id(0)] = res + threadScanT.w;\n"
"    }\n"
"\n"
"    // write results to global memory\n"
"    __global FPVECTYPE* outData = (__global FPVECTYPE*) g_odata;\n"
"     \n"
"    tempData.x = res;\n"
"    tempData.y = res + threadScanT.x;\n"
"    tempData.z = res + threadScanT.y;\n"
"    tempData.w = res + threadScanT.z;\n"
"\n"
"    if (fullBlock || i + 3 < n)\n"
"    {\n"
"        outData[gid] = tempData;\n"
"    }\n"
"    else\n"
"    {\n"
"        if ( i    < n) { g_odata[i]   = tempData.x;\n"
"        if ((i+1) < n) { g_odata[i+1] = tempData.y;\n"
"        if ((i+2) < n) { g_odata[i+2] = tempData.z; } } }\n"
"    }\n"
"}\n"
;
