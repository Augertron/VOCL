#ifndef _SORT_H
#define _SORT_H

static const int SORT_BLOCK_SIZE = 128;
static const int SCAN_BLOCK_SIZE = 256;
static const int SORT_BITS = 32;

void radixSortStep(uint nbits, uint startbit, cl_mem counters,
        cl_mem countersSum, cl_mem blockOffsets, cl_mem* scanBlockSums,
        uint numElements, cl_kernel sortBlocks, cl_kernel findOffsets,
        cl_kernel reorder, cl_kernel scan, cl_kernel uniformAdd,
        cl_command_queue queue, cl_device_id dev);

void
scanArrayRecursive(cl_mem outArray, cl_mem inArray, int numElements, int level,
        cl_mem* blockSums, cl_command_queue queue, cl_kernel scan,
        cl_kernel uniformAdd);

#endif // _SORT_H
