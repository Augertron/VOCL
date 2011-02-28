#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <fstream>
#include <vector>

#include "OpenCLDeviceInfo.h"
#include "Event.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "support.h"
#include "Scan.h"
#include "Timer.h"

using namespace std;

template <class T>
void runTest(const string& testName, cl_device_id dev, cl_context ctx,
        cl_command_queue queue, ResultDatabase& resultDB, OptionParser& op,
        const string& compileFlags);


// ****************************************************************************
// Function: scanCPU
//
// Purpose:
//   Simple cpu scan routine to verify device results
//
// Arguments:
//   data : the input data
//   reference : space for the cpu solution
//   dev_result : result from the device
//   size :
//
// Returns:  nothing, prints relevant info to stdout
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
template <class T>
bool
scanCPU(T* data, T* reference, T* dev_result, const size_t size)
{
    reference[0] = 0;
    bool passed = true;

    for (unsigned int i = 1; i < size; ++i)
    {
        reference[i] = data[i - 1] + reference[i - 1];
    }

    for (unsigned int i = 0; i < size; ++i)
    {
        if (reference[i] != dev_result[i])
        {
#ifdef VERBOSE_OUTPUT
            cout << "Mismatch at i: " << i << " ref: " << reference[i] <<
            " dev: " << dev_result[i] << endl;
#endif
            passed = false;
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
void
addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("iterations", OPT_INT, "256", "specify scan iterations");
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the scan (parallel prefix sum) benchmark
//
// Arguments:
//   dev: the opencl device id to use for the benchmark
//   ctx: the opencl context to use for the benchmark
//   queue: the opencl command queue to issue commands to
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//   Jeremy Meredith, Thu Sep 24 17:30:18 EDT 2009
//   Use implicit include of source file instead of runtime loading.
//
// ****************************************************************************
extern const char *cl_source_scan;

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

    // Always run single precision test
    // OpenCL doesn't support templated kernels, so we have to use macros
    string spMacros = "-DSINGLE_PRECISION";
    runTest<float>("Scan", dev, ctx, queue, resultDB, op, spMacros);

    // If double precision is supported, run the DP test
    if (checkExtension(dev, "cl_khr_fp64"))
    {
        cout << "DP Supported\n";
        string dpMacros = "-DK_DOUBLE_PRECISION ";
        runTest<double>
        ("Scan-DP", dev, ctx, queue, resultDB, op, dpMacros);
    }
    else if (checkExtension(dev, "cl_amd_fp64"))
    {
        cout << "DP Supported\n";
        string dpMacros = "-DAMD_DOUBLE_PRECISION ";
        runTest<double>
        ("Scan-DP", dev, ctx, queue, resultDB, op, dpMacros);
    }
    else
    {
        cout << "DP Not Supported\n";
        char atts[1024] = "DP_Not_Supported";
        // resultDB requires neg entry for every possible result
        int passes = op.getOptionInt("passes");
        for (int k = 0; k < passes; k++) {
            resultDB.AddResult("Scan-DP" , atts, "GB/s", FLT_MAX);
            resultDB.AddResult("Scan-DP_PCIe" , atts, "GB/s", FLT_MAX);
            resultDB.AddResult("Scan-DP_Parity" , atts, "GB/s", FLT_MAX);
        }
    }
}

template <class T>
void runTest(const string& testName, cl_device_id dev, cl_context ctx,
        cl_command_queue queue, ResultDatabase& resultDB, OptionParser& op,
        const string& compileFlags)
{
    int err = 0;

    // Program Setup
    cl_program prog = clCreateProgramWithSource(ctx, 1,
                         &cl_source_scan, NULL, &err);
    CL_CHECK_ERROR(err);

    cout << "Compiling scan kernel." << endl;
    err = clBuildProgram(prog, 1, &dev, compileFlags.c_str(), NULL, NULL);
    CL_CHECK_ERROR(err);

    if (err != CL_SUCCESS)
    {
        char log[5000];
        size_t retsize = 0;
        err = clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 5000
                * sizeof(char), log, &retsize);

        CL_CHECK_ERROR(err);
        cout << "Build error." << endl;
        cout << "Retsize: " << retsize << endl;
        cout << "Log: " << log << endl;
        return;
    }

    // Extract out the kernels
    cl_kernel scan = clCreateKernel(prog, "scan", &err);
    CL_CHECK_ERROR(err);

    cl_kernel uniformAdd = clCreateKernel(prog, "addUniform", &err);
    CL_CHECK_ERROR(err);

    // If the device doesn't support at least 256 work items in a
    // group, use a different kernel (TBI)
    if (getMaxWorkGroupSize(dev) < 256)
    {
        cout << "Scan requires work group size of at least 256" << endl;
        char atts[1024] = "GSize_Not_Supported";
        // resultDB requires neg entry for every possible result
        int passes = op.getOptionInt("passes");
        for (int k = 0; k < passes; k++) {
            resultDB.AddResult(testName , atts, "GB/s", FLT_MAX);
            resultDB.AddResult(testName+"_PCIe" , atts, "GB/s", FLT_MAX);
            resultDB.AddResult(testName+"_Parity" , atts, "GB/s", FLT_MAX);
        }
        return;
    }

    int probSizes[4] = { 1, 8, 32, 64 };
    int size = probSizes[op.getOptionInt("size")-1];

    // Convert to MB
    size = (size * 1024 * 1024) / sizeof(T);

    // Create input data on CPU
    unsigned int bytes = size * sizeof(T);
    T* reference = new T[size];

    // h_idata
    cl_mem h_i = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            bytes, NULL, &err);
    CL_CHECK_ERROR(err);
    T* h_idata = (T*)clEnqueueMapBuffer(queue, h_i, true,
            CL_MAP_READ|CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &err);
    CL_CHECK_ERROR(err);
    // h_odata
    cl_mem h_o = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            bytes, NULL, &err);
    CL_CHECK_ERROR(err);
    T* h_odata = (T*)clEnqueueMapBuffer(queue, h_o, true,
            CL_MAP_READ|CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &err);
    CL_CHECK_ERROR(err);

    // Initialize host memory
    cout << "Initializing host memory." << endl;
    for (int i = 0; i < size; i++)
    {
        h_idata[i] = i % 3; //Fill with some pattern
        h_odata[i] = i % 3;
    }

    // Allocate device memory
    cl_mem d_idata = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, NULL, &err);
    CL_CHECK_ERROR(err);

    cl_mem d_odata = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, NULL, &err);
    CL_CHECK_ERROR(err);

    // Copy data to GPU
    cout << "Copying data to device." << endl;
    Event evTransfer("PCIe transfer");
    err = clEnqueueWriteBuffer(queue, d_idata, true, 0, bytes, h_idata, 0,
            NULL, &evTransfer.CLEvent());
    CL_CHECK_ERROR(err);
    evTransfer.FillTimingInfo();
    double inTransferTime = evTransfer.StartEndRuntime();

    //Allocate space for block sums
    int numLevelsAllocated = 0;
    unsigned int blockSize = BLOCK_SIZE;
    unsigned int maxNumElements = size;
    unsigned int numElts = size;
    int level = 0;

    do
    {
        unsigned int numBlocks = max(1, (int) ceil((float) numElts /
                (4.f * blockSize)));
        if (numBlocks > 1)
        {
            level++;
        }
        numElts = numBlocks;
    }
    while (numElts > 1);

    cl_mem *scanBlockSums = (cl_mem *) malloc((level+1) * sizeof(cl_mem));
    numLevelsAllocated = level+1;
    numElts = maxNumElements;
    level = 0;

    do
    {
        unsigned int numBlocks = max(1, (int) ceil((float) numElts / (4.f
                * blockSize)));
        if (numBlocks > 1)
        {
            // Device mem for block sums
            scanBlockSums[level] = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                    numBlocks * sizeof(T), NULL, &err);
            CL_CHECK_ERROR(err);

            level++;
        }
        numElts = numBlocks;
    }
    while (numElts > 1);

    scanBlockSums[level] = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
            sizeof(T), NULL, &err);
    CL_CHECK_ERROR(err);

    // For n passes...
    int passes = op.getOptionInt("passes");
    int iters  = op.getOptionInt("iterations");

    cout << "Running benchmark with size " << size << endl;
    for (int k = 0; k < passes; k++)
    {
        int th = Timer::Start();
        for (int j = 0; j < iters; j++)
        {
            // Make recursive calls to scan kernels
            scanArrayRecursive(d_odata, d_idata, size, 0, scanBlockSums,
                    queue, scan, uniformAdd);
        }
        err = clFinish(queue);
        CL_CHECK_ERROR(err);
        double totalScanTime = Timer::Stop(th, "total scan time");

        err = clEnqueueReadBuffer(queue, d_odata, true, 0, bytes, h_odata,
                0, NULL, &evTransfer.CLEvent());
        CL_CHECK_ERROR(err);
        evTransfer.FillTimingInfo();
        double totalTransfer = inTransferTime + evTransfer.StartEndRuntime();
        totalTransfer /= 1.e9; // Convert to seconds

        // If answer is incorrect, stop test and do not report performance
        if (! scanCPU(h_idata, reference, h_odata, size))
        {
            return;
        }

        char atts[1024];
        double avgTime = totalScanTime / (double) iters;
        double gbs = (double) (size * sizeof(T)) / (1000. * 1000. * 1000.);
        sprintf(atts, "%d items", size);
        resultDB.AddResult(testName, atts, "GB/s", gbs / (avgTime));
        resultDB.AddResult(testName+"_PCIe", atts, "GB/s",
                gbs / (avgTime + totalTransfer));
        resultDB.AddResult(testName+"_Parity", atts, "N",
                totalTransfer / avgTime);
    }

    // Clean up
    err = clReleaseMemObject(d_idata);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_odata);
    CL_CHECK_ERROR(err);
    err = clEnqueueUnmapMemObject(queue, h_i, h_idata, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clEnqueueUnmapMemObject(queue, h_o, h_odata, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(h_i);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(h_o);
    CL_CHECK_ERROR(err);
    delete[] reference;

    for (int i = 0; i < numLevelsAllocated; i++)
    {
        err = clReleaseMemObject(scanBlockSums[i]);
        CL_CHECK_ERROR(err);
    }
    free(scanBlockSums);

    err = clReleaseProgram(prog);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(scan);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(uniformAdd);
    CL_CHECK_ERROR(err);
}

// ****************************************************************************
// Function: scanArrayRecursive
//
// Purpose:
//   Workhorse for the scan benchmark, this function recursively scans
//   arbitrary sized arrays, including those which are of a non power
//   of two length, or not evenly divisible by block size
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
// Returns: device time in nanoseconds
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

    // Kernels handle 4 elems per thread
    unsigned int numBlocks = (unsigned int)ceil((float)numElements /
            (4.f * (float)BLOCK_SIZE));
    if (numBlocks < 1) numBlocks = 1;

    const int fullBlock = (numElements == numBlocks * 4 * BLOCK_SIZE) ? 1 : 0;

    const size_t globalWorkSize = numBlocks * BLOCK_SIZE;
    const size_t localWorkSize = BLOCK_SIZE;

    // Set Scan Kernel Arguments
    const int t=1, f=0;
    err  = clSetKernelArg(scan, 0, sizeof(cl_mem), (void*) &outArray);
    err |= clSetKernelArg(scan, 1, sizeof(cl_mem), (void*) &inArray);
    err |= clSetKernelArg(scan, 2, sizeof(cl_mem), &(blockSums[level]));
    err |= clSetKernelArg(scan, 3, sizeof(cl_int), (void*) &numElements);
    err |= clSetKernelArg(scan, 4, sizeof(cl_int), (void*) &fullBlock);
    CL_CHECK_ERROR(err);

    // Execute the scan
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
    err = clFinish(queue);
    CL_CHECK_ERROR(err);

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
    err = clFinish(queue);
    CL_CHECK_ERROR(err);
    }
}
