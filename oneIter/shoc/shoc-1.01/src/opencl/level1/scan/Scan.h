#ifndef _SCAN_H
#define _SCAN_H

//Block or local group size
static const int BLOCK_SIZE = 256;

void
scanArrayRecursive(cl_mem outArray,
                   cl_mem inArray,
                   int numElements,
                   int level,
                   cl_mem* blockSums,
                   cl_command_queue queue,
                   cl_kernel scan,
                   cl_kernel uniformAdd);

#endif // _SCAN_H
