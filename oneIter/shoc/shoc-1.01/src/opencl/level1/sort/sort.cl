// This kernel code based in part on CUDPP.  Please see the notice in
// LICENSE_CUDPP.txt.

inline uint scanLSB(const uint val, __local uint* s_data)
{
    // Local mem is 256 uints long, set first half to 0
    int idx = get_local_id(0);
    s_data[idx] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Set 2nd half to thread local sum (sum of the 4 elems from global mem)
    idx += get_local_size(0); // += 128 in this case

    // Unrolled scan in local memory
    uint t;
    s_data[idx] = val;     barrier(CLK_LOCAL_MEM_FENCE);
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

    return s_data[idx] - val;  // convert inclusive -> exclusive
}

inline uint4 scan4(uint4 idata, __local uint* ptr)
{
    uint4 val4 = idata;
    uint4 sum;

    // Scan the 4 elements in idata within this thread
    sum.x = val4.x;
    sum.y = val4.y + sum.x;
    sum.z = val4.z + sum.y;
    uint val = val4.w + sum.z;

    // Now scan those sums across the local work group
    val = scanLSB(val, ptr);

    val4.x = val;
    val4.y = val + sum.x;
    val4.z = val + sum.y;
    val4.w = val + sum.z;

    return val4;
}

//----------------------------------------------------------------------------
//
// radixSortBlocks sorts all blocks of data independently in shared
// memory.  Each thread block (CTA) sorts one block of 4*CTA_SIZE elements
//
// The radix sort is done in two stages.  This stage calls radixSortBlock
// on each block independently, sorting on the basis of bits
// (startbit) -> (startbit + nbits)
//----------------------------------------------------------------------------

__kernel void radixSortBlocks(uint nbits, uint startbit,
                              __global uint4* keysOut,
                              __global uint4* valuesOut,
                              __global uint4* keysIn,
                              __global uint4* valuesIn,
                              __local uint* sMem)
{
    // Get Indexing information
    uint i = get_global_id(0);
    uint tid = get_local_id(0);
    uint localSize = get_local_size(0);

    // Load keys and vals from global memory
    uint4 key, value;
    key = keysIn[i];
    value = valuesIn[i];

    // For each of the 4 bits
    for(uint shift = startbit; shift < (startbit + nbits); ++shift)
    {
        // Check if the LSB is 0
        uint4 lsb;
        lsb.x = !((key.x >> shift) & 0x1);
        lsb.y = !((key.y >> shift) & 0x1);
        lsb.z = !((key.z >> shift) & 0x1);
        lsb.w = !((key.w >> shift) & 0x1);
        
        // Do an exclusive scan of how many elems have 0's in the LSB
        // When this is finished, address.n will contain the number of
        // elems with 0 in the LSB which precede elem n
        uint4 address = scan4(lsb, sMem);

        __local uint numtrue[1];

        // Store the total number of elems with an LSB of 0
        // to shared mem
        if (tid == localSize - 1)
        {
            numtrue[0] = address.w + lsb.w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Determine rank -- position in the block
        // If your LSB is 0 --> your position is the scan of 0's
        // If your LSB is 1 --> your position is calculated as below
        uint4 rank;
        int idx = tid*4;
        rank.x = lsb.x ? address.x : numtrue[0] + idx     - address.x;
        rank.y = lsb.y ? address.y : numtrue[0] + idx + 1 - address.y;
        rank.z = lsb.z ? address.z : numtrue[0] + idx + 2 - address.z;
        rank.w = lsb.w ? address.w : numtrue[0] + idx + 3 - address.w;

        // Scatter keys into local mem
        sMem[(rank.x & 3) * localSize + (rank.x >> 2)] = key.x;
        sMem[(rank.y & 3) * localSize + (rank.y >> 2)] = key.y;
        sMem[(rank.z & 3) * localSize + (rank.z >> 2)] = key.z;
        sMem[(rank.w & 3) * localSize + (rank.w >> 2)] = key.w;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Read keys out of local mem into registers, in prep for
        // write out to global mem
        key.x = sMem[tid];
        key.y = sMem[tid +     localSize];
        key.z = sMem[tid + 2 * localSize];
        key.w = sMem[tid + 3 * localSize];
        barrier(CLK_LOCAL_MEM_FENCE);

        // Scatter values into local mem
        sMem[(rank.x & 3) * localSize + (rank.x >> 2)] = value.x;
        sMem[(rank.y & 3) * localSize + (rank.y >> 2)] = value.y;
        sMem[(rank.z & 3) * localSize + (rank.z >> 2)] = value.z;
        sMem[(rank.w & 3) * localSize + (rank.w >> 2)] = value.w;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Read keys out of local mem into registers, in prep for
        // write out to global mem
        value.x = sMem[tid];
        value.y = sMem[tid +     localSize];
        value.z = sMem[tid + 2 * localSize];
        value.w = sMem[tid + 3 * localSize];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    keysOut[i]   = key;
    valuesOut[i] = value;
}

//----------------------------------------------------------------------------
// Given an array with blocks sorted according to a 4-bit radix group, each
// block counts the number of keys that fall into each radix in the group, and
// finds the starting offset of each radix in the block.  It then writes the
// radix counts to the counters array, and the starting offsets to the
// blockOffsets array.
//----------------------------------------------------------------------------

__kernel void findRadixOffsets(__global uint2* keys,
                               __global uint* counters,
                               __global uint* blockOffsets,
                               uint startbit,
                               uint numElements,
                               uint totalBlocks,
                               __local uint* sRadix1)
{
    __local uint  sStartPointers[16];
    uint groupId = get_group_id(0);
    uint localId = get_local_id(0);
    uint groupSize = get_local_size(0);

    // Load two keys in from global memory
    uint2 radix2;
    radix2 = keys[get_global_id(0)];

    // Convert those keys to their 4-bit radix
    sRadix1[2 * localId]     = (radix2.x >> startbit) & 0xF;
    sRadix1[2 * localId + 1] = (radix2.y >> startbit) & 0xF;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Finds the position where the sRadix1 entries differ and stores start
    // index for each radix.
    if (localId < 16)
    {
        sStartPointers[localId] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // If this threads radix is different from the previous thread,
    // it has an index where the radix changes, write to shared mem
    if ((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) )
    {
        sStartPointers[sRadix1[localId]] = localId;
    }
    // Same thing, but on second half of elements
    if (sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1])
    {
        sStartPointers[sRadix1[localId + groupSize]] = localId + groupSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId < 16)
    {
        blockOffsets[groupId*16 + localId] = sStartPointers[localId];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute the number of elems with each radix
    if ((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) )
    {
        sStartPointers[sRadix1[localId - 1]] =
            localId - sStartPointers[sRadix1[localId - 1]];
    }
    if (sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1] )
    {
        sStartPointers[sRadix1[localId + groupSize - 1]] =
            localId + groupSize - sStartPointers[sRadix1[localId+groupSize-1]];
    }

    if (localId == groupSize - 1)
    {
        sStartPointers[sRadix1[2 * groupSize - 1]] =
            2 * groupSize - sStartPointers[sRadix1[2 * groupSize - 1]];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(localId < 16)
    {
        counters[localId * totalBlocks + groupId] = sStartPointers[localId];
    }
}
//----------------------------------------------------------------------------
// reorderData shuffles data in the array globally after the radix offsets
// have been found. On compute version 1.1 and earlier GPUs, this code depends
// on RadixSort::CTA_SIZE being 16 * number of radices (i.e. 16 * 2^nbits).
//----------------------------------------------------------------------------
__kernel void reorderData(uint  startbit,
                          __global uint  *outKeys,
                          __global uint  *outValues,
                          __global uint2 *keys,
                          __global uint2 *values,
                          __global uint  *blockOffsets,
                          __global uint  *offsets,
                          __global uint  *sizes,
                          uint   totalBlocks)
{
    __local uint2 sKeys2[256];
    __local uint2 sValues2[256];
    __local uint  sOffsets[16];
    __local uint  sBlockOffsets[16];
    __local uint* sKeys1   = (__local uint*) sKeys2;
    __local uint* sValues1 = (__local uint*) sValues2;
    uint groupSize = get_local_size(0);
    uint blockId = get_group_id(0);

    uint i = blockId * get_local_size(0) + get_local_id(0);

    sKeys2[get_local_id(0)]   = keys[i];
    sValues2[get_local_id(0)] = values[i];

    if(get_local_id(0) < 16)
    {
        sOffsets[get_local_id(0)]      = offsets[get_local_id(0) * totalBlocks
                                                 + blockId];
        sBlockOffsets[get_local_id(0)] = blockOffsets[blockId * 16 +
                                                      get_local_id(0)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint radix = (sKeys1[get_local_id(0)] >> startbit) & 0xF;
    uint globalOffset = sOffsets[radix] + get_local_id(0) -
            sBlockOffsets[radix];

    outKeys[globalOffset]   = sKeys1[get_local_id(0)];
    outValues[globalOffset] = sValues1[get_local_id(0)];

    radix = (sKeys1[get_local_id(0) + groupSize] >> startbit) & 0xF;
    globalOffset = sOffsets[radix] + get_local_id(0) + groupSize -
            sBlockOffsets[radix];

    outKeys[globalOffset]   = sKeys1[get_local_id(0) + groupSize];
    outValues[globalOffset] = sValues1[get_local_id(0) + groupSize];
}

// Scan Kernels
// Duplicated here because Sort uses uints and OpenCL doesn't
// support templates yet

__kernel void
addUniform(__global uint *d_vector, __global const uint *d_uniforms,
           const int n)
{
    __local uint uni[1];

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

inline uint scanLocalMem(const uint val, __local uint* s_data)
{
    // Shared mem is 512 uints long, set first half to 0
    int idx = get_local_id(0);
    s_data[idx] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Set 2nd half to thread local sum (sum of the 4 elems from global mem)
    idx += get_local_size(0); // += 256
    
    uint t;
    s_data[idx] = val;     barrier(CLK_LOCAL_MEM_FENCE);
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

__kernel void scan(__global uint *g_odata, __global uint *g_idata,
        __global uint *g_blockSums, const int n, const int fullBlock,
        const int storeSum)
{
    __local uint s_data[512];

    // Load data into shared mem
    uint4 tempData;
    uint4 threadScanT;
    uint res;
    __global uint4* inData  = (__global uint4*) g_idata;

    const int gid = get_global_id(0);
    const int tid = get_local_id(0);
    const int i = gid * 4;

    // If possible, read from global mem in a uint4 chunk
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
    {   // if not, read individual uints, scan & store in lmem
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
    __global uint4* outData = (__global uint4*) g_odata;
     
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
