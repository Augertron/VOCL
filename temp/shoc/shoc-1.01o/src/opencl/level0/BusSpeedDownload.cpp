#include <iostream>
#include "support.h"
#include "Event.h"
#include "ResultDatabase.h"
#include "OptionParser.h"

using namespace std;

void addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("nopinned", OPT_BOOL, "",
                 "disable usage of pinned (pagelocked) memory");
}

//  Modifications:
//    Jeremy Meredith, Wed Dec  1 17:05:27 EST 2010
//    Added calculation of latency estimate.
//
void RunBenchmark(cl::Device& devcpp,
                  cl::Context& ctxcpp,
                  cl::CommandQueue& queuecpp,
                  ResultDatabase &resultDB,
                  OptionParser &op)
{
    // convert from C++ bindings to C bindings
    // TODO propagate use of C++ bindings
    cl_device_id id = devcpp();
    cl_context ctx = ctxcpp();
    cl_command_queue queue = queuecpp();

    bool verbose = op.getOptionBool("verbose");
    bool pinned = !op.getOptionBool("nopinned");
    int  npasses = op.getOptionInt("passes");
    const bool waitForEvents = true;

    const int nSizes  = 17;
    int sizes[nSizes] = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,
            32768,65536};
    long long numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;

    // Create some host memory pattern
    if (verbose) cout << ">> creating host mem pattern\n";
    int err;
    float *hostMem;
    cl_mem hostMemObj;
    if (pinned)
    {
        hostMemObj = clCreateBuffer(ctx,
                                    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
                                    sizeof(float)*numMaxFloats, NULL, &err);
        CL_CHECK_ERROR(err);
        hostMem = (float*)clEnqueueMapBuffer(queue, hostMemObj, true,
                                             CL_MAP_READ|CL_MAP_WRITE,
                                             0,sizeof(float)*numMaxFloats,0,
                                             NULL,NULL,&err);
        CL_CHECK_ERROR(err);
    }
    else
    {
        hostMem = new float[numMaxFloats];
    }
    for (int i=0; i<numMaxFloats; i++)
        hostMem[i] = i % 77;

    // Allocate some device memory
    if (verbose) cout << ">> allocating device mem\n";
    cl_mem mem1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
                                 sizeof(float)*numMaxFloats, NULL, &err);
    CL_CHECK_ERROR(err);
    if (verbose) cout << ">> filling device mem to force allocation\n";
    Event evDownloadPrime("DownloadPrime");
    err = clEnqueueWriteBuffer(queue, mem1, false, 0,
                               numMaxFloats*sizeof(float), hostMem,
                               0, NULL, &evDownloadPrime.CLEvent());
    CL_CHECK_ERROR(err);
    if (verbose) cout << ">> waiting for download to finish\n";
    err = clWaitForEvents(1, &evDownloadPrime.CLEvent());
    CL_CHECK_ERROR(err);
    
    // Three passes, forward and backward both
    for (int pass = 0; pass < npasses; pass++)
    {
        // store the times temporarily to estimate latency
        float times[nSizes];
        // Step through sizes forward on even passes and backward on odd
        for (int i = 0; i < nSizes; i++)
        {
            int sizeIndex;
            if ((pass%2) == 0)
                sizeIndex = i;
            else
                sizeIndex = (nSizes-1) - i;

            // Copy input memory to the device
            if (verbose) cout << ">> copying to device "<<sizes[sizeIndex]<<"kB\n";
            Event evDownload("Download");
            err = clEnqueueWriteBuffer(queue, mem1, false, 0,
                                       sizes[sizeIndex]*1024, hostMem,
                                       0, NULL, &evDownload.CLEvent());
            CL_CHECK_ERROR(err);

            // Wait for event to finish
            if (verbose) cout << ">> waiting for download to finish\n";
            err = clWaitForEvents(1, &evDownload.CLEvent());
            CL_CHECK_ERROR(err);

            if (verbose) cout << ">> finish!";
            if (verbose) cout << endl;

            // Get timings
            err = clFlush(queue);
            CL_CHECK_ERROR(err);
            evDownload.FillTimingInfo();
            if (verbose) evDownload.Print(cerr);

            double t = evDownload.SubmitEndRuntime() / 1.e6; // in ms
            times[sizeIndex] = t;

            // Add timings to database
            double speed = (double(sizes[sizeIndex] * 1024.) /  (1000.*1000.)) / t;
            char sizeStr[256];
            sprintf(sizeStr, "% 6dkB", sizes[sizeIndex]);
            resultDB.AddResult("DownloadSpeed", sizeStr, "GB/sec", speed);

            // Add timings to database
            double delay = evDownload.SubmitStartDelay() / 1.e6;
            resultDB.AddResult("DownloadDelay", sizeStr, "ms", delay);
            resultDB.AddResult("DownloadTime", sizeStr, "ms", t);
        }
	resultDB.AddResult("DownloadLatencyEstimate", "1-2kb", "ms", times[0]-(times[1]-times[0])/1.);
	resultDB.AddResult("DownloadLatencyEstimate", "1-4kb", "ms", times[0]-(times[2]-times[0])/3.);
	resultDB.AddResult("DownloadLatencyEstimate", "2-4kb", "ms", times[1]-(times[2]-times[1])/1.);
    }

    // Cleanup
    err = clReleaseMemObject(mem1);
    CL_CHECK_ERROR(err);
    if (pinned)
    {
        err = clReleaseMemObject(hostMemObj);
        CL_CHECK_ERROR(err);
    }
    else
    {
        delete[] hostMem;
    }
}
