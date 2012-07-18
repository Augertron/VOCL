// standard utility and system includes
#include <oclUtils.h>
#include <sched.h>
#include "mt_timer.h"

#define BLOCK_DIM 16

void fillDoubleArray(double* pfData, int iSize)
{	
	int i;
	const double fScale = 1.0 / (double)RAND_MAX;
	for (i = 0; i < iSize; ++i)
	{
		pfData[i] = fScale * rand();
	}
}

int compareDouble(double *ref, double *data, const unsigned int arraySize)
{
	unsigned int i;
	const double delta = 0.000000001;
	int res = 1;
	for (i = 0; i < arraySize; i++)
	{
		if (ref[i] - data[i] > delta ||
			ref[i] - data[i] < (-1) * delta)
		{
			res = 0;
			printf("ref[%d] = %lf, data[%d] = %lf\n", i, i, ref[i], data[i]);
			break;
		}
	}

	return res;
}

//const unsigned int MAX_GPU_COUNT = 8;

// global variables
cl_platform_id *cpPlatforms;
cl_uint uiNumDevices, platformNum, *deviceNums;
cl_device_id *cdDevices;
cl_context *cxGPUContexts;
cl_command_queue *commandQueues;
cl_program *cpPrograms;
unsigned int numIterations = 20, iterNo;

// forward declarations
// *********************************************************************
int runTest( int argc, const char** argv);
extern "C" void computeGold( double* reference, double* idata, 
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

	return 0;
}

double transposeGPU(const char* kernelName, bool useLocalMem,  cl_uint ciDeviceCount, double* h_idata, double* h_odata, unsigned int size_x, unsigned int size_y)
{
    cl_mem *d_odata;
    cl_mem *d_idata;
    cl_kernel *ckKernels;
	double time = 0.0;

    size_t szGlobalWorkSize[2];
    size_t szLocalWorkSize[2];
    cl_int ciErrNum, i;

	struct timeval t1, t2;
 
    // Create buffers for each GPU
    // Each GPU will compute sizePerGPU rows of the result
    size_t sizePerGPU = shrRoundUp(BLOCK_DIM, size_x);

	d_odata = (cl_mem *) malloc(sizeof(cl_mem) * ciDeviceCount);
	d_idata = (cl_mem *) malloc(sizeof(cl_mem) * ciDeviceCount);
	ckKernels = (cl_kernel *)malloc(sizeof(cl_kernel) * ciDeviceCount);
    // size of memory required to store the matrix
    const size_t mem_size = sizeof(double) * size_x * size_y;

	for(i = 0; i < ciDeviceCount; ++i)
	{
		// create the naive transpose kernel
		timerStart();
		ckKernels[i] = clCreateKernel(cpPrograms[i], kernelName, &ciErrNum);
		oclCheckError(ciErrNum, CL_SUCCESS);
		timerEnd();
		strTime.createKernel += elapsedTime();

		// allocate device memory and copy host to device memory
		timerStart();
		d_idata[i] = clCreateBuffer(cxGPUContexts[i], CL_MEM_READ_ONLY, mem_size, NULL, &ciErrNum);
		oclCheckError(ciErrNum, CL_SUCCESS);

		d_odata[i] = clCreateBuffer(cxGPUContexts[i], CL_MEM_WRITE_ONLY, mem_size, NULL, &ciErrNum);
		oclCheckError(ciErrNum, CL_SUCCESS);
		timerEnd();
		strTime.createBuffer += elapsedTime();
	}

    // execute the kernel numIterations times
    shrLog("\nProcessing a %d by %d matrix of doubles for %d iterations\n\n", size_x, size_y, numIterations);
	timerStart();
	for (iterNo = 0; iterNo < numIterations; iterNo++)
	{
		for(i = 0; i < ciDeviceCount; ++i)
		{
			ciErrNum = clEnqueueWriteBuffer(commandQueues[i], d_idata[i], CL_FALSE, 0,
									mem_size, h_idata, 0, NULL, NULL);
		}

		for(i = 0; i < ciDeviceCount; ++i)
		{
			// set the args values for the naive kernel
			//size_t offset = i * sizePerGPU;
			size_t offset = 0;
			ciErrNum  = clSetKernelArg(ckKernels[i], 0, sizeof(cl_mem), (void *) &d_odata[i]);
			ciErrNum |= clSetKernelArg(ckKernels[i], 1, sizeof(cl_mem), (void *) &d_idata[i]);
			ciErrNum |= clSetKernelArg(ckKernels[i], 2, sizeof(int), &offset);
			ciErrNum |= clSetKernelArg(ckKernels[i], 3, sizeof(int), &size_x);
			ciErrNum |= clSetKernelArg(ckKernels[i], 4, sizeof(int), &size_y);
			if(useLocalMem)
			{
				ciErrNum |= clSetKernelArg(ckKernels[i], 5, (BLOCK_DIM + 1) * BLOCK_DIM * sizeof(double), 0 );
			}
    		oclCheckError(ciErrNum, CL_SUCCESS);
		}

		// set up execution configuration
		szLocalWorkSize[0] = BLOCK_DIM;
		szLocalWorkSize[1] = BLOCK_DIM;
		szGlobalWorkSize[0] = sizePerGPU;
		szGlobalWorkSize[1] = shrRoundUp(BLOCK_DIM, size_y);
    
        for(i=0; i < ciDeviceCount; ++i){
            ciErrNum |= clEnqueueNDRangeKernel(commandQueues[i], ckKernels[i], 2, NULL,                                           
                                szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);
        }
        oclCheckError(ciErrNum, CL_SUCCESS);

		for(i = 0; i < ciDeviceCount; ++i){
			ciErrNum |= clEnqueueReadBuffer(commandQueues[i], d_odata[i], CL_FALSE, 0,
									mem_size, h_odata, 0, NULL, NULL);
		}
	}

	for(i = 0; i < ciDeviceCount; ++i)
	{
		clFinish(commandQueues[i]);
	}

	timerEnd();
	strTime.kernelExecution += elapsedTime();
	oclCheckError(ciErrNum, CL_SUCCESS);

	timerStart();
	for(i = 0; i < ciDeviceCount; ++i){
		ciErrNum |= clReleaseMemObject(d_idata[i]);
		ciErrNum |= clReleaseMemObject(d_odata[i]);
	}
	timerEnd();
	strTime.releaseMemObj += elapsedTime();
	
	//release kernel
	timerStart();
	for(i = 0; i < ciDeviceCount; ++i){
		ciErrNum |= clReleaseKernel(ckKernels[i]);
	}
	timerEnd();
	strTime.releaseKernel += elapsedTime();

	free(ckKernels);
	free(d_idata);
	free(d_odata);

    return time;
}

//! Run a simple test for CUDA
// *********************************************************************
int runTest( const int argc, const char** argv) 
{
    cl_int ciErrNum;
    cl_uint ciDeviceCount;

    unsigned int size_x = 2048;
    unsigned int size_y = 2048;
	int disableCPU = 1;
	cl_uint i, index, deviceNo = 0;
	bool bUseAllDevices = false;
	cpu_set_t set;
	CPU_ZERO(&set);

    int temp;
    if( shrGetCmdLineArgumenti( argc, argv,"width", &temp) ){
        size_x = temp;
    }

    if( shrGetCmdLineArgumenti( argc, argv,"height", &temp) ){
        size_y = temp;
    }

    if( shrGetCmdLineArgumenti( argc, argv,"iter", &temp) ){
        numIterations = temp;
    }

    if( shrGetCmdLineArgumenti( argc, argv,"device", &temp) ){
        deviceNo = temp;
    }

    if( shrGetCmdLineArgumenti( argc, argv,"disablecpu", &temp) ){
        disableCPU = temp;
    }

	bUseAllDevices = (shrTRUE == shrCheckCmdLineFlag(argc, (const char**)argv, "deviceall"));

    // size of memory required to store the matrix
    const size_t mem_size = sizeof(double) * size_x * size_y;

    //Get the NVIDIA platform
	timerStart();
    ciErrNum = clGetPlatformIDs(0, NULL, &platformNum);
	cpPlatforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * platformNum);
	deviceNums = (cl_uint *)malloc(sizeof(cl_uint) * platformNum);
    ciErrNum |= clGetPlatformIDs(platformNum, cpPlatforms, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
	timerEnd();
	strTime.getPlatform += elapsedTime();

	sched_getaffinity(0, sizeof(cpu_set_t), &set);
	shrLog("cpuid = %d\n", set.__bits[0]);

    //Get the devices
	timerStart();
	uiNumDevices = 0;
	for (i = 0; i < platformNum; i++)
	{
    	ciErrNum  = clGetDeviceIDs(cpPlatforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceNums[i]);
    	oclCheckError(ciErrNum, CL_SUCCESS);
		uiNumDevices += deviceNums[i];
	}
	
	cxGPUContexts = (cl_context *) malloc(sizeof(cl_context) * uiNumDevices);
	commandQueues = (cl_command_queue *) malloc(sizeof(cl_command_queue) * uiNumDevices);
	cpPrograms = (cl_program *) malloc(sizeof(cl_program) * uiNumDevices);
	cdDevices = (cl_device_id *) malloc(sizeof(cl_device_id) * uiNumDevices);

	uiNumDevices = 0;
	for (i = 0; i < platformNum; i++)
	{
    	ciErrNum = clGetDeviceIDs(cpPlatforms[i], CL_DEVICE_TYPE_GPU, deviceNums[i], &cdDevices[uiNumDevices], NULL);
    	oclCheckError(ciErrNum, CL_SUCCESS);
		uiNumDevices += deviceNums[i];
	}
	timerEnd();
	strTime.getDeviceID += elapsedTime();

	//set the devices that will be used
	if (bUseAllDevices == true)
	{
		ciDeviceCount = uiNumDevices;
		shrLog("All GPUs are used..., deviceCount = %d\n\n", ciDeviceCount);
		deviceNo = 0;
	}
	else
	{
		shrLog("Device %d is used ...\n\n", deviceNo);
		ciDeviceCount = 1;
	}

    //Create the context
	timerStart();
	for (i = 0; i < ciDeviceCount; i++)
	{
		index = i + deviceNo;
    	cxGPUContexts[i] = clCreateContext(0, 1, &cdDevices[index], NULL, NULL, &ciErrNum);
    	oclCheckError(ciErrNum, CL_SUCCESS);
	}
	timerEnd();
	strTime.createContext += elapsedTime();

	timerStart();
	for (i = 0; i < ciDeviceCount; i++)
	{
		index = i + deviceNo;
		commandQueues[i] = clCreateCommandQueue(cxGPUContexts[i], cdDevices[index], CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
    	oclCheckError(ciErrNum, CL_SUCCESS);
	}
	timerEnd();
	strTime.createCommandQueue += elapsedTime();

    // allocate and initalize host memory
    double* h_idata = (double*)malloc(mem_size);
    double* h_odata = (double*) malloc(mem_size);
    srand(15235911);
    fillDoubleArray(h_idata, (size_x * size_y));

    // Program Setup
    size_t program_length;
    char kernel_source[KERNEL_SOURCE_FILE_LEN];

    snprintf(kernel_source, KERNEL_SOURCE_FILE_LEN,
		"%s/examples/matrixtrans/transpose_dp.cl", ABS_SRCDIR);
    char *source = oclLoadProgSource(kernel_source, "", &program_length);
    oclCheckError(source != NULL, shrTRUE);

    // create the program
	timerStart();
	for (i = 0; i < ciDeviceCount; i++)
	{
    	cpPrograms[i] = clCreateProgramWithSource(cxGPUContexts[i], 1,
                      (const char **)&source, &program_length, &ciErrNum);
    	oclCheckError(ciErrNum, CL_SUCCESS);
	}
	timerEnd();
	strTime.createProgramWithSource += elapsedTime();
    
    // build the program
	timerStart();
	for (i = 0; i < ciDeviceCount; i++)
	{
    	ciErrNum = clBuildProgram(cpPrograms[i], 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
		if (ciErrNum != CL_SUCCESS)
		{
			// write out standard error, Build Log and PTX, then return error
			shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
			oclLogBuildInfo(cpPrograms[i], oclGetFirstDev(cxGPUContexts[i]));
			oclLogPtx(cpPrograms[i], oclGetFirstDev(cxGPUContexts[i]), "oclTranspose.ptx");
			return(EXIT_FAILURE); 
		}
	}
	timerEnd();
	strTime.buildProgram += elapsedTime();

    double naiveTime = transposeGPU("transpose_naive", false, ciDeviceCount, h_idata, h_odata, size_x, size_y);
    double optimizedTime = transposeGPU("transpose", true, ciDeviceCount, h_idata, h_odata, size_x, size_y);
 
    // compute reference solution and cross check results
	if (disableCPU == 0)
	{
		double* reference = (double*)malloc( mem_size);
		computeGold( reference, h_idata, size_x, size_y);
		shrLog("\nComparing results with CPU computation... \n\n");
		int res = compareDouble( reference, h_odata, size_x * size_y);
		shrLog( "%s\n\n", (1 == res) ? "PASSED" : "FAILED");
    	free(reference);
	}

    // cleanup memory
    free(h_idata);
    free(h_odata);
    free(source);

    // cleanup OpenCL
	timerStart();
	for (i = 0; i < ciDeviceCount; i++)
	{
    	ciErrNum = clReleaseProgram(cpPrograms[i]);
    	oclCheckError(ciErrNum, CL_SUCCESS);
	}
	timerEnd();
	strTime.releaseProgram += elapsedTime();
	
	timerStart();
    for(i = 0; i < ciDeviceCount; ++i) 
    {
        ciErrNum = clReleaseCommandQueue(commandQueues[i]);
    	oclCheckError(ciErrNum, CL_SUCCESS);
    }
	timerEnd();
	strTime.releaseCmdQueue += elapsedTime();

	timerStart();
	for (i = 0; i < ciDeviceCount; i++)
	{
    	ciErrNum = clReleaseContext(cxGPUContexts[i]);
    	oclCheckError(ciErrNum, CL_SUCCESS);
	}
	timerEnd();
	strTime.releaseContext += elapsedTime();
	
	free(cpPlatforms);
	free(deviceNums);
	free(cdDevices);
	free(cxGPUContexts);
	free(commandQueues);
	free(cpPrograms);
	
    return 0;
}

