#include <iostream>
#include "support.h"
#include "Event.h"
#include "ResultDatabase.h"
#include "OptionParser.h"
#include <sys/time.h>

using namespace std;

void addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("nopinned", OPT_BOOL, "",
                 "disable usage of pinned (pagelocked) memory");
}

// Modifications:
//    Jeremy Meredith, Wed Dec  1 17:05:27 EST 2010
//    Added calculation of latency estimate.
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
	//debug---------------------
	pinned = false;
	struct timeval t1, t2;
	double transferTime;
	//-------------------------
    int  npasses = op.getOptionInt("passes");
    const bool waitForEvents = true;

    // 1k through 8M bytes
    const int nSizes  = 17;
    int sizes[nSizes] = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,
            32768,65536};
    long long numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;

    // Create some host memory pattern
    if (verbose) cout << ">> creating host mem pattern\n";
    int err;
    float *hostMem1;
    float *hostMem2;
    cl_mem hostMemObj1;
    cl_mem hostMemObj2;
    if (pinned)
    {
        hostMemObj1 = clCreateBuffer(ctx,
                                     CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
                                     sizeof(float)*numMaxFloats, NULL, &err);
        CL_CHECK_ERROR(err);
        hostMemObj2 = clCreateBuffer(ctx,
                                     CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
                                     sizeof(float)*numMaxFloats, NULL, &err);
        CL_CHECK_ERROR(err);
        hostMem1 = (float*)clEnqueueMapBuffer(queue, hostMemObj1, true,
                                                     CL_MAP_READ|CL_MAP_WRITE,
                                                     0,sizeof(float)*numMaxFloats,0,
                                                     NULL,NULL,&err);

        CL_CHECK_ERROR(err);
        hostMem2 = (float*)clEnqueueMapBuffer(queue, hostMemObj2, true,
                                                     CL_MAP_READ|CL_MAP_WRITE,
                                                     0,sizeof(float)*numMaxFloats,0,
                                                     NULL,NULL,&err);
        CL_CHECK_ERROR(err);
    }
    else
    {
        hostMem1 = new float[numMaxFloats];
        hostMem2 = new float[numMaxFloats];
    }

    for (int i=0; i<numMaxFloats; i++) {
        hostMem1[i] = i % 77;
        hostMem2[i] = -1; 
    }

    // Allocate some device memory
    if (verbose) cout << ">> allocating device mem\n";
    cl_mem mem1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
                                 sizeof(float)*numMaxFloats, NULL, &err);
    CL_CHECK_ERROR(err);
    if (verbose) cout << ">> filling device mem to force allocation\n";
    Event evDownloadPrime("DownloadPrime");
    err = clEnqueueWriteBuffer(queue, mem1, false, 0,
                               numMaxFloats*sizeof(float), hostMem1,
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

            // Read memory back from the device
            if (verbose) cout << ">> reading from device "<<sizes[sizeIndex]<<"kB\n";
            Event evReadback("Readback");
	//debug----------------------------
	gettimeofday(&t1, NULL);
	//-----------------------------------
            err = clEnqueueReadBuffer(queue, mem1, true, 0,
                                       sizes[sizeIndex]*1024, hostMem2,
                                       0, NULL, &evReadback.CLEvent());
	//debug-----------------------------------
	gettimeofday(&t2, NULL);
	transferTime = 1000.0 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
	//printf("time = %.3f\n", transferTime);
	//------------------------------------------
            CL_CHECK_ERROR(err);

            // Wait for event to finish
            if (verbose) cout << ">> waiting for readback to finish\n";
            err = clWaitForEvents(1, &evReadback.CLEvent());
            CL_CHECK_ERROR(err);

            if (verbose) cout << ">> finish!";
            if (verbose) cout << endl;
            
            // Get timings
            err = clFlush(queue);
            CL_CHECK_ERROR(err);
            evReadback.FillTimingInfo();
            if (verbose) evReadback.Print(cerr);

            double t = evReadback.SubmitEndRuntime() / 1.e6; // in ms
			//debug---------------------------
			t = transferTime;
			//--------------------------------
            times[sizeIndex] = t;

            // Add timings to database
            double speed = (double(sizes[sizeIndex] * 1024.) /  (1000.*1000.)) / t;
            char sizeStr[256];
            sprintf(sizeStr, "% 6dkB", sizes[sizeIndex]);
            resultDB.AddResult("ReadbackSpeed", sizeStr, "GB/sec", speed);

            // Add timings to database
            double delay = evReadback.SubmitStartDelay() / 1.e6;
            resultDB.AddResult("ReadbackDelay", sizeStr, "ms", delay);
            resultDB.AddResult("ReadbackTime", sizeStr, "ms", t);
        }
	resultDB.AddResult("ReadbackLatencyEstimate", "1-2kb", "ms", times[0]-(times[1]-times[0])/1.);
	resultDB.AddResult("ReadbackLatencyEstimate", "1-4kb", "ms", times[0]-(times[2]-times[0])/3.);
	resultDB.AddResult("ReadbackLatencyEstimate", "2-4kb", "ms", times[1]-(times[2]-times[1])/1.);
    }

    // Cleanup
    err = clReleaseMemObject(mem1);
    CL_CHECK_ERROR(err);
    if (pinned)
    {
        err = clReleaseMemObject(hostMemObj1);
        CL_CHECK_ERROR(err);
        err = clReleaseMemObject(hostMemObj2);
        CL_CHECK_ERROR(err);
    }
    else
    {
        delete[] hostMem1;
        delete[] hostMem2;
    }
}
