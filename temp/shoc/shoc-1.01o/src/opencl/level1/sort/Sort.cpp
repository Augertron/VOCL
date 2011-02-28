#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include <cassert>
#include <iostream>
#include <vector>

#include "Event.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Sort.h"
#include "support.h"
#include "Timer.h"

using namespace std;

// ****************************************************************************
// Function: verifySort
//
// Purpose:
//   Simple cpu routine to verify device results
//
// Arguments:
//
//
// Returns:  nothing, prints relevant info to stdout
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
bool verifySort(uint *keys, uint* vals, const size_t size)
{
    bool passed = true;

    for (unsigned int i = 0; i < size - 1; i++)
    {
        if (keys[i] > keys[i + 1])
        {
            passed = false;

#ifdef VERBOSE_OUTPUT
            cout << "Idx: " << i;
            cout << " Key: " << keys[i] << " Val: " << vals[i] << "\n";
#endif
        }
    }
    cout << "Test ";
    if (passed)
        cout << "Passed" << endl;
    else
        cout << "---FAILED---" << endl;

    return passed;
}

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("iterations", OPT_INT, "10", "specify sort iterations");
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the radix sort benchmark
//
// Arguments:
//   dev: the opencl device id to use for the benchmark
//   ctx: the opencl context to use for the benchmark
//   queue: the opencl command queue to issue commands to
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing, results are stored in resultDB
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//   Jeremy Meredith, Thu Sep 24 17:30:18 EDT 2009
//   Use implicit include of source file instead of runtime loading.
//
// ****************************************************************************
extern const char *cl_source_sort;

void
RunBenchmark(cl::Device& devcpp,
                  cl::Context& ctxcpp,
                  cl::CommandQueue& queuecpp,
                  ResultDatabase &resultDB,
                  OptionParser &op)
{
    // convert from C++ bindings to C bindings
    // TODO propagate use of C++ bindings
    cl_device_id dev = devcpp();
    cl_context ctx = ctxcpp();
    cl_command_queue queue = queuecpp();

    int err;
    uint waitForEvents = 1;

    // Program Setup
    cl_program prog = clCreateProgramWithSource(ctx, 1,
                        &cl_source_sort, NULL, &err);
    CL_CHECK_ERROR(err);
    err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);	
    CL_CHECK_ERROR(err);

    // If the program doesn't build, display errors and return
    if (err != CL_SUCCESS)
    {
        char log[5000];
        size_t retsize = 0;
        err = clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 5000
                * sizeof(char), log, &retsize);

        CL_CHECK_ERROR(err);
        cout << "Build error." << endl;
        cout << "Log: " << log << endl;
        return;
    }

    // Extract out the kernels
    cl_kernel sortBlocks = clCreateKernel(prog, "radixSortBlocks", &err);
    CL_CHECK_ERROR(err);

    cl_kernel findOffsets = clCreateKernel(prog, "findRadixOffsets", &err);
    CL_CHECK_ERROR(err);

    cl_kernel reorder = clCreateKernel(prog, "reorderData", &err);
    CL_CHECK_ERROR(err);

    cl_kernel scan = clCreateKernel(prog, "scan", &err);
    CL_CHECK_ERROR(err);

    cl_kernel uniformAdd = clCreateKernel(prog, "addUniform", &err);
    CL_CHECK_ERROR(err);

    // If the device doesn't support at least 128 work items,
    // return
    if (getMaxWorkGroupSize(ctx, scan) < 128 ||
            getMaxWorkGroupSize(ctx, uniformAdd)  < 128)
    {
        fprintf(stderr, "Sort requires devices which can support"
                "local work groups sizes of at least 128 items\n");
        err = clReleaseProgram(prog);
        CL_CHECK_ERROR(err);
        err = clReleaseKernel(sortBlocks);
        CL_CHECK_ERROR(err);
        err = clReleaseKernel(findOffsets);
        CL_CHECK_ERROR(err);
        err = clReleaseKernel(reorder);
        CL_CHECK_ERROR(err);
        err = clReleaseKernel(scan);
        CL_CHECK_ERROR(err);
        err = clReleaseKernel(uniformAdd);
        CL_CHECK_ERROR(err);
        return;
    }

    //Number of key-value pairs to sort, must be a multiple of 1024
    int probSizes[4] = { 1, 8, 48, 96 };

    int size = probSizes[op.getOptionInt("size")-1];
    // Convert to MB
    size = (size * 1024 * 1024) / sizeof(uint);

    // Each thread in the sort kernel handles 4 elements
    size_t numSortGroups = size / (4 * SORT_BLOCK_SIZE);

    // Other threads handle 2 elements per thread
    size_t numGroups2 = size / (2 * SCAN_BLOCK_SIZE);

    // Size of the keys & vals buffers in bytes
    uint bytes = size * sizeof(uint);

    // create input data on CPU
    // uint *hKeys = new uint[size];
    // uint *hVals = new uint[size];
    cl_mem h_k = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            bytes, NULL, &err);
    CL_CHECK_ERROR(err);
    uint* hKeys = (uint*)clEnqueueMapBuffer(queue, h_k, true,
            CL_MAP_READ|CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &err);
    CL_CHECK_ERROR(err);
    cl_mem h_v = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            bytes, NULL, &err);
    CL_CHECK_ERROR(err);
    uint* hVals = (uint*)clEnqueueMapBuffer(queue, h_v, true,
            CL_MAP_READ|CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &err);
    CL_CHECK_ERROR(err);

    //Allocate space for block sums in the scan kernel
    uint numLevelsAllocated = 0;
    uint maxNumScanElements = size;
    uint numScanElts = maxNumScanElements;
    uint level = 0;

    do
    {
        // Scan handles 4 elems per work item
        uint numBlocks = max(1, (int) ceil((float) numScanElts / (4.f
                * SCAN_BLOCK_SIZE)));
        if (numBlocks > 1)
        {
            level++;
        }
        numScanElts = numBlocks;
    }
    while (numScanElts > 1);

    cl_mem* scanBlockSums = (cl_mem*) malloc((level + 1) * sizeof(cl_mem));
    assert(scanBlockSums != NULL);
    numLevelsAllocated = level + 1;
    numScanElts = maxNumScanElements;
    level = 0;

    do
    {
        uint numBlocks = max(1, (int) ceil((float) numScanElts / (4.f
                * SCAN_BLOCK_SIZE)));
        if (numBlocks > 1)
        {
            // Malloc device mem for block sums
            scanBlockSums[level] = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                    numBlocks * sizeof(cl_uint), NULL, &err);
            CL_CHECK_ERROR(err);
            level++;
        }
        numScanElts = numBlocks;
    }
    while (numScanElts > 1);

    scanBlockSums[level] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 1
            * sizeof(cl_uint), NULL, &err);
    CL_CHECK_ERROR(err);

    // Allocate device mem for sorting kernels
    cl_mem dKeys = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, NULL,
            &err);
    CL_CHECK_ERROR(err);
    cl_mem dVals = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, NULL,
            &err);
    CL_CHECK_ERROR(err);
    cl_mem dTempKeys =  clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes,
            NULL, &err);
    CL_CHECK_ERROR(err);
    cl_mem dTempVals = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes,
            NULL, &err);
    CL_CHECK_ERROR(err);
    cl_mem dCounters = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 16
            * numSortGroups * sizeof(uint), NULL, &err);
    CL_CHECK_ERROR(err);
    cl_mem dCounterSums = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 16
            * numSortGroups * sizeof(uint), NULL, &err);
    CL_CHECK_ERROR(err);
    cl_mem dBlockOffsets = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 16
            * numSortGroups * sizeof(uint), NULL, &err);
    CL_CHECK_ERROR(err);

    // Threads handle either 4 or two elements each
    const size_t findGlobalWorkSize    = size / 2;
    const size_t reorderGlobalWorkSize = size / 2;
    // Number of blocks (or local work groups)
    const size_t offsetBlocks = findGlobalWorkSize / SCAN_BLOCK_SIZE;
    const size_t reorderBlocks = reorderGlobalWorkSize / SCAN_BLOCK_SIZE;

    // Set static kernel arguments
    // Sort radix blocks kernel
    err = clSetKernelArg(sortBlocks, 2, sizeof(cl_mem), (void*) &dTempKeys);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(sortBlocks, 3, sizeof(cl_mem), (void*) &dTempVals);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(sortBlocks, 4, sizeof(cl_mem), (void*) &dKeys);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(sortBlocks, 5, sizeof(cl_mem), (void*) &dVals);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(sortBlocks, 6, sizeof(cl_float) * 4 * SORT_BLOCK_SIZE,
            NULL);
    CL_CHECK_ERROR(err);
    // Find offsets kernel
    err = clSetKernelArg(findOffsets, 0, sizeof(cl_mem), (void*) &dTempKeys);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(findOffsets, 1, sizeof(cl_mem), (void*) &dCounters);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(findOffsets, 2, sizeof(cl_mem),
            (void*) &dBlockOffsets);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(findOffsets, 4, sizeof(cl_uint),
            (void*) &size);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(findOffsets, 5, sizeof(cl_uint),
            (void*) &offsetBlocks);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(findOffsets, 6, 2 * SCAN_BLOCK_SIZE *
            sizeof(unsigned int), NULL);
    CL_CHECK_ERROR(err);
    // Reorder data kernel
    err = clSetKernelArg(reorder, 1, sizeof(cl_mem), (void*) &dKeys);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(reorder, 2, sizeof(cl_mem), (void*) &dVals);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(reorder, 3, sizeof(cl_mem), (void*) &dTempKeys);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(reorder, 4, sizeof(cl_mem), (void*) &dTempVals);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(reorder, 5, sizeof(cl_mem), (void*) &dBlockOffsets);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(reorder, 6, sizeof(cl_mem), (void*) &dCounterSums);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(reorder, 7, sizeof(cl_mem), (void*) &dCounters);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(reorder, 8, sizeof(cl_uint), (void*) &reorderBlocks);
    CL_CHECK_ERROR(err);

    int iterations = op.getOptionInt("iterations");
    for (int it = 0; it < iterations; it++)
    {
        // Initialize host memory to some pattern
        for (uint i = 0; i < size; i++)
        {
            hKeys[i] = hVals[i] = i % 1024;
        }

        // Copy inputs to GPU
        double transferTime = 0.0;
        // Copy inputs to GPU
        Event evTransfer("PCIe Transfer");
        err = clEnqueueWriteBuffer(queue, dKeys, true, 0, bytes, hKeys, 0,
                NULL, &evTransfer.CLEvent());
        CL_CHECK_ERROR(err);
        evTransfer.FillTimingInfo();
        transferTime += evTransfer.StartEndRuntime() * 1.e-9;
        err = clEnqueueWriteBuffer(queue, dVals, true, 0, bytes, hVals, 0,
                NULL, &evTransfer.CLEvent());
        CL_CHECK_ERROR(err);
        evTransfer.FillTimingInfo();
        transferTime += evTransfer.StartEndRuntime() * 1.e-9;

        // Perform Radix Sort (4 bits at a time)
        int th = Timer::Start();
        for (int i = 0; i < SORT_BITS; i += 4)
        {
            radixSortStep(4, i, dCounters, dCounterSums, dBlockOffsets,
                    scanBlockSums, size, sortBlocks, findOffsets, reorder,
                    scan, uniformAdd, queue, dev);
        }
        err = clFinish(queue);
        CL_CHECK_ERROR(err);
        double totalTime = Timer::Stop(th, "radix sort");
        
        // Readback data from device
        err = clEnqueueReadBuffer(queue, dKeys, true, 0, bytes, hKeys, 0,
                NULL, &evTransfer.CLEvent());
        CL_CHECK_ERROR(err);
        evTransfer.FillTimingInfo();
        transferTime += evTransfer.StartEndRuntime() * 1.e-9;

        err = clEnqueueReadBuffer(queue, dVals, true, 0, bytes, hVals, 0,
                NULL, &evTransfer.CLEvent());
        CL_CHECK_ERROR(err);
        evTransfer.FillTimingInfo();
        transferTime += evTransfer.StartEndRuntime() * 1.e-9;
        
        // Test to make sure data was sorted properly
        if (! verifySort(hKeys, hVals, size))
        {
            return;
        }

        char atts[1024];
        sprintf(atts, "%ld items", size);
        // Count both keys and val's
        double gb = (bytes * 2.) /  (1000. * 1000. * 1000.);
        resultDB.AddResult("Sort-Rate", atts, "GB/s", gb / totalTime);
        resultDB.AddResult("Sort-Rate_PCIe", atts, "GB/s",
                gb / (totalTime + transferTime));
        resultDB.AddResult("Sort-Rate_Parity", atts, "N",
                transferTime / totalTime);
    }
    // Clean up
    for (int i = 0; i < numLevelsAllocated; i++)
    {
        err = clReleaseMemObject(scanBlockSums[i]);
        CL_CHECK_ERROR(err);
    }

    clReleaseMemObject(dKeys);
    CL_CHECK_ERROR(err);
    clReleaseMemObject(dVals);
    CL_CHECK_ERROR(err);
    clReleaseMemObject(dTempKeys);
    CL_CHECK_ERROR(err);
    clReleaseMemObject(dTempVals);
    CL_CHECK_ERROR(err);
    clReleaseMemObject(dCounters);
    CL_CHECK_ERROR(err);
    clReleaseMemObject(dCounterSums);
    CL_CHECK_ERROR(err);
    clReleaseMemObject(dBlockOffsets);
    CL_CHECK_ERROR(err);
    free(scanBlockSums);
    err = clReleaseProgram(prog);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(sortBlocks);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(findOffsets);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(reorder);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(scan);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(uniformAdd);
    CL_CHECK_ERROR(err);
	err = clEnqueueUnmapMemObject(queue, h_k, hKeys, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clEnqueueUnmapMemObject(queue, h_v, hVals, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    clReleaseMemObject(h_k);
    CL_CHECK_ERROR(err);
    clReleaseMemObject(h_v);
    CL_CHECK_ERROR(err);
}

// ****************************************************************************
// Function: radixSortStep
//
// Purpose:
//   This function performs a radix sort, using bits startbit to
//   (startbit + nbits).  It is designed to sort by 4 bits at a time.
//   It also reorders the data in the values array based on the sort.
//
// Arguments:
//      nbits: the number of key bits to use
//      startbit: the bit to start on, 0 = lsb
//      counters: storage for the index counters, used in sort
//      countersSum: storage for the sum of the counters
//      blockOffsets: storage used in sort
//      scanBlockSums: input to Scan, see below
//      numElements: the number of elements to sort
//      sortBlocks: sorting kernel #1
//      findOffsets: sorting kernel #2
//      reorder: sorting kernel #3
//      scan: scan kernel #1
//      uniformAdd: scan kernel #2
//      queue: the opencl command queue to use
//      dev: the device id
//
// Returns: nothing
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
void radixSortStep(uint nbits, uint startbit, cl_mem counters,
        cl_mem countersSum, cl_mem blockOffsets, cl_mem* scanBlockSums,
        uint numElements, cl_kernel sortBlocks, cl_kernel findOffsets,
        cl_kernel reorder, cl_kernel scan, cl_kernel uniformAdd,
        cl_command_queue queue, cl_device_id dev)
{
    int err = 0;

    const size_t sortlocalWorkSize = SORT_BLOCK_SIZE;
    const size_t localWorkSize     = SCAN_BLOCK_SIZE;

    // Threads handle either 4 or two elements each
    const size_t radixGlobalWorkSize   = numElements / 4;
    const size_t findGlobalWorkSize    = numElements / 2;
    const size_t reorderGlobalWorkSize = numElements / 2;

    // Number of blocks
    const size_t radixBlocks = radixGlobalWorkSize / SORT_BLOCK_SIZE;
    const size_t offsetBlocks = findGlobalWorkSize / SCAN_BLOCK_SIZE;
    const size_t reorderBlocks = reorderGlobalWorkSize / SCAN_BLOCK_SIZE;

    err = clSetKernelArg(sortBlocks, 0, sizeof(cl_uint), (void*) &nbits);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(sortBlocks, 1, sizeof(cl_uint), (void*) &startbit);
    CL_CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(queue, sortBlocks, 1, NULL,
            &radixGlobalWorkSize, &sortlocalWorkSize, 0, NULL, NULL);
    CL_CHECK_ERROR(err);

    err = clSetKernelArg(findOffsets, 3, sizeof(cl_uint), (void*) &startbit);
    CL_CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(queue, findOffsets, 1, NULL,
            &findGlobalWorkSize, &localWorkSize, 0, NULL, NULL);

    scanArrayRecursive(countersSum, counters,
            16 * reorderBlocks, 0, scanBlockSums, queue, scan, uniformAdd);

    // Set the arguments of the reorder kernel
    err = clSetKernelArg(reorder, 0, sizeof(cl_uint), (void*) &startbit);
    CL_CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(queue, reorder, 1, NULL,
            &reorderGlobalWorkSize, &localWorkSize, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
}

// ****************************************************************************
// Function: scanArrayRecursive
//
// Purpose:
//   This function recursively scans arbitrary sized arrays, including
//   those which are of a non power of two length, or not evenly divisible
//   by block size
//
// Arguments:
//     outArray: pointer to output memory on the device
//     inArray:  pointer to input memory on the device
//     numElements: the number of elements to scan
//     level: the current level of recursion, starting at 0
//     blockSums: pointer to device memory to store intermediate sums
//     queue: the opencl command queue to execute in
//     scan: pointer to the scan kernel
//     uniformAdd: pointer to the uniform add kernel
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
void
scanArrayRecursive(cl_mem outArray,
                   cl_mem inArray,
                   int numElements,
                   int level,
                   cl_mem* blockSums,
                   cl_command_queue queue,
                   cl_kernel scan,
                   cl_kernel uniformAdd)
{
    int err = 0;

    // Kernels handle 8 elems per thread
    unsigned int numBlocks = (unsigned int)ceil((float)numElements /
            (4.f * SCAN_BLOCK_SIZE));
    if (numBlocks < 1)
    {
       numBlocks = 1;
    }

    const int fullBlock = (numElements == numBlocks * 4 * SCAN_BLOCK_SIZE) ?
        1 : 0;

    const size_t globalWorkSize = numBlocks * SCAN_BLOCK_SIZE;
    const size_t localWorkSize = SCAN_BLOCK_SIZE;

    // Set Scan Kernel Arguments
    const int t=1, f=0;
    err  = clSetKernelArg(scan, 0, sizeof(cl_mem), (void*) &outArray);
    err |= clSetKernelArg(scan, 1, sizeof(cl_mem), (void*) &inArray);
    err |= clSetKernelArg(scan, 2, sizeof(cl_mem), &(blockSums[level]));
    err |= clSetKernelArg(scan, 3, sizeof(cl_int), (void*) &numElements);
    err |= clSetKernelArg(scan, 4, sizeof(cl_int), (void*) &fullBlock);
    CL_CHECK_ERROR(err);

    // execute the scan
    if (numBlocks > 1)
    {
        err = clSetKernelArg(scan, 5, sizeof(cl_int), (void*) &t);
        CL_CHECK_ERROR(err);
    } else
    {
        err = clSetKernelArg(scan, 5, sizeof(cl_int), (void*) &f);
        CL_CHECK_ERROR(err);
    }

    err = clEnqueueNDRangeKernel(queue, scan, 1, NULL, &globalWorkSize,
            &localWorkSize, 0, NULL, NULL);

    if (numBlocks > 1)
    {
        scanArrayRecursive(blockSums[level], blockSums[level], numBlocks,
                           level+1, blockSums, queue, scan, uniformAdd);
        err = clSetKernelArg(uniformAdd, 0, sizeof(cl_mem),
                (void*) &outArray);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(uniformAdd, 1, sizeof(cl_mem),
                (void*) &(blockSums[level]));
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(uniformAdd, 2, sizeof(cl_int),
                (void*)&numElements);
        CL_CHECK_ERROR(err);
        err = clEnqueueNDRangeKernel(queue, uniformAdd, 1, NULL,
                &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    }
}

