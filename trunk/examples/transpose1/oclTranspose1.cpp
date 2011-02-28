// standard utility and system includes
/*****************************************************
 * 1.  Each time with a different kernel called
 *****************************************************/
#include <oclUtils.h>
#include "timeRec.h"

#define BLOCK_DIM 16
//#define GPU_PROFILING

// max GPU's to manage for multi-GPU parallel compute
const unsigned int MAX_GPU_COUNT = 8;
//#define oclCheckError(a, b) 				\
//	if (a != b) {							\
//		printf("Opencl error, %d!\n", a);	\
//	}										

// global variables
cl_platform_id cpPlatform;
cl_uint uiNumDevices;
cl_device_id* cdDevices;
cl_context cxGPUContext;
cl_kernel ckKernel[MAX_GPU_COUNT];
cl_command_queue commandQueue[MAX_GPU_COUNT];
cl_program cpProgram;

// forward declarations
// *********************************************************************
int runTest( int argc, const char** argv);
extern "C" void computeGold( float* reference, float* idata, 
                         const unsigned int size_x, const unsigned int size_y );

// Main Program
// *********************************************************************
int main( int argc, const char** argv) 
{ 
    // set logfile name and start logs
    shrSetLogFileName ("oclTranspose.txt");
    shrLog("%s Starting...\n\n", argv[0]); 
	
	//initialize timer
	memset(&strTime, 0, sizeof(STRUCT_TIME));

    // run the main test
    int result = runTest(argc, argv);
    oclCheckError(result, 0);

	printTime_toStandardOutput();
    printTime_toFile();

    // finish
    //shrEXIT(argc, argv);
	return 0;
}

double transposeGPU(const char* kernelName, bool useLocalMem,  cl_uint ciDeviceCount, float* h_idata, float* h_odata, unsigned int size_x, unsigned int size_y)
{
    cl_mem d_odata[MAX_GPU_COUNT];
    cl_mem d_idata[MAX_GPU_COUNT];
    cl_kernel ckKernel[MAX_GPU_COUNT];
	double time;

    size_t szGlobalWorkSize[2];
    size_t szLocalWorkSize[2];
    cl_int ciErrNum;
 
    // Create buffers for each GPU
    // Each GPU will compute sizePerGPU rows of the result
    size_t sizePerGPU = shrRoundUp(BLOCK_DIM, (size_x+ciDeviceCount-1) / ciDeviceCount);
    
    // size of memory required to store the matrix
    const size_t mem_size = sizeof(float) * size_x * size_y;

    // execute the kernel numIterations times
    int numIterations = 100;
    shrLog("\nProcessing a %d by %d matrix of floats...\n\n", size_x, size_y);
    for (int k = 0; k < numIterations; ++k)
    {

		for(unsigned int i = 0; i < ciDeviceCount; ++i)
		{
			// create the naive transpose kernel
			timerStart();
			ckKernel[i] = clCreateKernel(cpProgram, kernelName, &ciErrNum);
			timerEnd();
			strTime.createKernel += elapsedTime();
			strTime.numCreateKernel++;
			oclCheckError(ciErrNum, CL_SUCCESS);
	
			// allocate device memory and copy host to device memory
			timerStart();
			//d_idata[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			//                            mem_size, h_idata, &ciErrNum);
			d_idata[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, mem_size, NULL, &ciErrNum);
			oclCheckError(ciErrNum, CL_SUCCESS);

			d_odata[i] = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY ,
										sizePerGPU*size_y*sizeof(float), NULL, &ciErrNum);
			oclCheckError(ciErrNum, CL_SUCCESS);
			timerEnd();
			strTime.createBuffer += elapsedTime();
			strTime.numCreateBuffer += 2;
		}

		for(unsigned int i = 0; i < ciDeviceCount; ++i)
		{
			timerStart();
			ciErrNum = clEnqueueWriteBuffer(commandQueue[i], d_idata[i], CL_TRUE, 0,
									mem_size, h_idata, 0, NULL, NULL);
			timerEnd();
			strTime.enqueueWriteBuffer += elapsedTime();
			strTime.numEnqueueWriteBuffer++;
		
			// set the args values for the naive kernel
			size_t offset = i * sizePerGPU;
			timerStart();
			ciErrNum  = clSetKernelArg(ckKernel[i], 0, sizeof(cl_mem), (void *) &d_odata[i]);
			ciErrNum |= clSetKernelArg(ckKernel[i], 1, sizeof(cl_mem), (void *) &d_idata[0]);
			ciErrNum |= clSetKernelArg(ckKernel[i], 2, sizeof(int), &offset);
			ciErrNum |= clSetKernelArg(ckKernel[i], 3, sizeof(int), &size_x);
			ciErrNum |= clSetKernelArg(ckKernel[i], 4, sizeof(int), &size_y);
			if(useLocalMem)
			{
				ciErrNum |= clSetKernelArg(ckKernel[i], 5, (BLOCK_DIM + 1) * BLOCK_DIM * sizeof(float), 0 );
				strTime.numSetKernelArg++;
			}
			timerEnd();
			strTime.setKernelArg += elapsedTime();
			strTime.numSetKernelArg += 4;
    		oclCheckError(ciErrNum, CL_SUCCESS);
		}

		// set up execution configuration
		szLocalWorkSize[0] = BLOCK_DIM;
		szLocalWorkSize[1] = BLOCK_DIM;
		szGlobalWorkSize[0] = sizePerGPU;
		szGlobalWorkSize[1] = shrRoundUp(BLOCK_DIM, size_y);
    
       // Start time measurement after warmup
        if( k == 0 ) shrDeltaT(0);
		
		timerStart();
        for(unsigned int i=0; i < ciDeviceCount; ++i){
            ciErrNum |= clEnqueueNDRangeKernel(commandQueue[i], ckKernel[i], 2, NULL,                                           
                                szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);
        }
        oclCheckError(ciErrNum, CL_SUCCESS);

		// Block CPU till GPU is done
		for(unsigned int i=0; i < ciDeviceCount; ++i){ 
			ciErrNum |= clFinish(commandQueue[i]);
		}
		timerEnd();
		strTime.kernelExecution += elapsedTime();
		strTime.numKernelExecution += ciDeviceCount;
		time = shrDeltaT(0)/(double)numIterations;
		oclCheckError(ciErrNum, CL_SUCCESS);

		// Copy back to host
		timerStart();
		for(unsigned int i = 0; i < ciDeviceCount; ++i){
			size_t offset = i * sizePerGPU;
			size_t size = MIN(size_x - i * sizePerGPU, sizePerGPU);

			ciErrNum |= clEnqueueReadBuffer(commandQueue[i], d_odata[i], CL_TRUE, 0,
									size * size_y * sizeof(float), &h_odata[offset * size_y], 
									0, NULL, NULL);
		}
		timerEnd();
		strTime.enqueueReadBuffer += elapsedTime();
		strTime.numEnqueueReadBuffer += ciDeviceCount;
		oclCheckError(ciErrNum, CL_SUCCESS);

		timerStart();
		for(unsigned int i = 0; i < ciDeviceCount; ++i){
			ciErrNum |= clReleaseMemObject(d_idata[i]);
			ciErrNum |= clReleaseMemObject(d_odata[i]);
			//ciErrNum |= clReleaseKernel(ckKernel[i]);
		}
		timerEnd();
		strTime.releaseMemObj += elapsedTime();
		strTime.numReleaseMemObj += 2 * ciDeviceCount;
		
		//release kernel
		timerStart();
		for(unsigned int i = 0; i < ciDeviceCount; ++i){
			ciErrNum |= clReleaseKernel(ckKernel[i]);
		}
		timerEnd();
		strTime.releaseKernel += elapsedTime();
		strTime.numReleaseKernel += ciDeviceCount;
	}

    return time;
}

//! Run a simple test for CUDA
// *********************************************************************
int runTest( const int argc, const char** argv) 
{
    cl_int ciErrNum;
    cl_uint ciDeviceCount;
	//int matrixSize = atoi(argv[1]);
    unsigned int size_x = 2048;
    unsigned int size_y = 2048;

    int temp;
    if( shrGetCmdLineArgumenti( argc, argv,"width", &temp) ){
        size_x = temp;
    }

    if( shrGetCmdLineArgumenti( argc, argv,"height", &temp) ){
        size_y = temp;
    }

    // size of memory required to store the matrix
    const size_t mem_size = sizeof(float) * size_x * size_y;

    //Get the NVIDIA platform
    //ciErrNum = oclGetPlatformID(&cpPlatform);
	timerStart();
    ciErrNum = clGetPlatformIDs(1, &cpPlatform, NULL);
	timerEnd();
	strTime.getPlatform += elapsedTime();
	strTime.numGetPlatform++;
    oclCheckError(ciErrNum, CL_SUCCESS);

    //Get the devices
	timerStart();
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiNumDevices);
	timerEnd();
	strTime.getDeviceID += elapsedTime();
	strTime.numGetDeviceID++;

    oclCheckError(ciErrNum, CL_SUCCESS);
    cdDevices = (cl_device_id *)malloc(uiNumDevices * sizeof(cl_device_id) );
	timerStart();
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, uiNumDevices, cdDevices, NULL);
	timerEnd();
	strTime.getDeviceID += elapsedTime();
	strTime.numGetDeviceID++;

    oclCheckError(ciErrNum, CL_SUCCESS);

    //Create the context
	timerStart();
    cxGPUContext = clCreateContext(0, uiNumDevices, cdDevices, NULL, NULL, &ciErrNum);
	timerEnd();
	strTime.createContext += elapsedTime();
	strTime.numCreateContext++;
    oclCheckError(ciErrNum, CL_SUCCESS);
  
    if(shrCheckCmdLineFlag(argc, (const char**)argv, "device"))
    {
        ciDeviceCount = 0;
        // User specified GPUs
        char* deviceList;
        char* deviceStr;

        shrGetCmdLineArgumentstr(argc, (const char**)argv, "device", &deviceList);

        #ifdef WIN32
            char* next_token;
            deviceStr = strtok_s (deviceList," ,.-", &next_token);
        #else
            deviceStr = strtok (deviceList," ,.-");
        #endif   
        ciDeviceCount = 0;
        while(deviceStr != NULL) 
        {
            // get and print the device for this queue
            cl_device_id device = oclGetDev(cxGPUContext, atoi(deviceStr));
		    if( device == (cl_device_id)-1 ) {
                shrLog(" Invalid Device: %s\n\n", deviceStr);
                return -1;
		    }	

            shrLog("Device %d: ", atoi(deviceStr));
            oclPrintDevName(LOGBOTH, device);            
            shrLog("\n");
           
            // create command queue
			timerStart();
            commandQueue[ciDeviceCount] = clCreateCommandQueue(cxGPUContext, device, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
			timerEnd();
			strTime.createCommandQueue += elapsedTime();
			strTime.numCreateCommandQueue++;
            if (ciErrNum != CL_SUCCESS)
            {
                shrLog(" Error %i in clCreateCommandQueue call !!!\n\n", ciErrNum);
                return ciErrNum;
            }

            ++ciDeviceCount;

            #ifdef WIN32
                deviceStr = strtok_s (NULL," ,.-", &next_token);
            #else            
                deviceStr = strtok (NULL," ,.-");
            #endif
        }

        free(deviceList);
    } 
    else 
    {
        // Find out how many GPU's to compute on all available GPUs
        size_t nDeviceBytes;
        ciErrNum |= clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &nDeviceBytes);
        ciDeviceCount = (cl_uint)nDeviceBytes/sizeof(cl_device_id);

        if (ciErrNum != CL_SUCCESS)
        {
            shrLog(" Error %i in clGetDeviceIDs call !!!\n\n", ciErrNum);
            return ciErrNum;
        }
        else if (ciDeviceCount == 0)
        {
            shrLog(" There are no devices supporting OpenCL (return code %i)\n\n", ciErrNum);
            return -1;
        } 

        // create command-queues
        for(unsigned int i = 0; i < ciDeviceCount; ++i) 
        {
            // get and print the device for this queue
            cl_device_id device = oclGetDev(cxGPUContext, i);
            shrLog("Device %d: ", i);
            oclPrintDevName(LOGBOTH, device);            
            shrLog("\n");

            // create command queue
			timerStart();
            commandQueue[i] = clCreateCommandQueue(cxGPUContext, device, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
			timerEnd();
			strTime.createCommandQueue += elapsedTime();
			strTime.numCreateCommandQueue++;
            if (ciErrNum != CL_SUCCESS)
            {
                shrLog(" Error %i in clCreateCommandQueue call !!!\n\n", ciErrNum);
                return ciErrNum;
            }
        }
    }
 
    // allocate and initalize host memory
    float* h_idata = (float*)malloc(mem_size);
    float* h_odata = (float*) malloc(mem_size);
    srand(15235911);
    shrFillArray(h_idata, (size_x * size_y));

    // Program Setup
    size_t program_length;
    char* source_path = shrFindFilePath("transpose.cl", argv[0]);
    oclCheckError(source_path != NULL, shrTRUE);
    char *source = oclLoadProgSource(source_path, "", &program_length);
    oclCheckError(source != NULL, shrTRUE);

    // create the program
	timerStart();
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1,
                      (const char **)&source, &program_length, &ciErrNum);
	timerEnd();
	strTime.createProgramWithSource += elapsedTime();
	strTime.numCreateProgramWithSource++;
    oclCheckError(ciErrNum, CL_SUCCESS);
    
    // build the program
	timerStart();
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
	timerEnd();
	strTime.buildProgram += elapsedTime();
	strTime.numBuildProgram++;
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then return error
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclTranspose.ptx");
        return(EXIT_FAILURE); 
    }

    // Run Naive Kernel
#ifdef GPU_PROFILING
    // Matrix Copy kernel runs to measure reference performance.
    double uncoalescedCopyTime = transposeGPU("uncoalesced_copy", false, ciDeviceCount, h_idata, h_odata, size_x, size_y);
    double simpleCopyTime = transposeGPU("simple_copy", false, ciDeviceCount, h_idata, h_odata, size_x, size_y);
    double sharedCopyTime = transposeGPU("shared_copy", true, ciDeviceCount, h_idata, h_odata, size_x, size_y);
#endif

    double naiveTime = transposeGPU("transpose_naive", false, ciDeviceCount, h_idata, h_odata, size_x, size_y);
    double optimizedTime = transposeGPU("transpose", true, ciDeviceCount, h_idata, h_odata, size_x, size_y);

#ifdef GPU_PROFILING
    // log times

    shrLogEx(LOGBOTH | MASTER, 0, "oclTranspose-Outer-simple copy, Throughput = %.4f GB/s, Time = %.5f s, Size = %u fp32 elements, NumDevsUsed = %u, Workgroup = %u\n", 
           (1.0e-9 * double(size_x * size_y * sizeof(float))/simpleCopyTime), simpleCopyTime, (size_x * size_y), ciDeviceCount, BLOCK_DIM * BLOCK_DIM); 

    shrLogEx(LOGBOTH | MASTER, 0, "oclTranspose-Outer-shared memory copy, Throughput = %.4f GB/s, Time = %.5f s, Size = %u fp32 elements, NumDevsUsed = %u, Workgroup = %u\n", 
           (1.0e-9 * double(size_x * size_y * sizeof(float))/sharedCopyTime), sharedCopyTime, (size_x * size_y), ciDeviceCount, BLOCK_DIM * BLOCK_DIM); 

    shrLogEx(LOGBOTH | MASTER, 0, "oclTranspose-Outer-uncoalesced copy, Throughput = %.4f GB/s, Time = %.5f s, Size = %u fp32 elements, NumDevsUsed = %u, Workgroup = %u\n", 
           (1.0e-9 * double(size_x * size_y * sizeof(float))/uncoalescedCopyTime), uncoalescedCopyTime, (size_x * size_y), ciDeviceCount, BLOCK_DIM * BLOCK_DIM); 

    shrLogEx(LOGBOTH | MASTER, 0, "oclTranspose-Outer-naive, Throughput = %.4f GB/s, Time = %.5f s, Size = %u fp32 elements, NumDevsUsed = %u, Workgroup = %u\n", 
           (1.0e-9 * double(size_x * size_y * sizeof(float))/naiveTime), naiveTime, (size_x * size_y), ciDeviceCount, BLOCK_DIM * BLOCK_DIM); 
    
    shrLogEx(LOGBOTH | MASTER, 0, "oclTranspose-Outer-optimized, Throughput = %.4f GB/s, Time = %.5f s, Size = %u fp32 elements, NumDevsUsed = %u, Workgroup = %u\n", 
          (1.0e-9 * double(size_x * size_y * sizeof(float))/optimizedTime), optimizedTime, (size_x * size_y), ciDeviceCount, BLOCK_DIM * BLOCK_DIM); 

#endif
  
    // compute reference solution and cross check results
    float* reference = (float*)malloc( mem_size);
    computeGold( reference, h_idata, size_x, size_y);
    shrLog("\nComparing results with CPU computation... \n\n");
    shrBOOL res = shrComparef( reference, h_odata, size_x * size_y);
    shrLog( "%s\n\n", (1 == res) ? "PASSED" : "FAILED");

    // cleanup memory
    free(h_idata);
    free(h_odata);
    free(reference);
    free(source);
    free(source_path);

    // cleanup OpenCL
	timerStart();
    ciErrNum = clReleaseProgram(cpProgram);    
	timerEnd();
	strTime.releaseProgram += elapsedTime();
	strTime.numReleaseProgram++;
	timerStart();
    for(unsigned int i = 0; i < ciDeviceCount; ++i) 
    {
        ciErrNum |= clReleaseCommandQueue(commandQueue[i]);
    }
	timerEnd();
	strTime.releaseCmdQueue += elapsedTime();
	strTime.numReleaseCmdQueue += ciDeviceCount;

	timerStart();
    ciErrNum |= clReleaseContext(cxGPUContext);
	timerEnd();
	strTime.releaseContext += elapsedTime();
	strTime.numReleaseContext++;
    oclCheckError(ciErrNum, CL_SUCCESS);
	
    oclCheckError(ciErrNum, CL_SUCCESS);

    return 0;
}