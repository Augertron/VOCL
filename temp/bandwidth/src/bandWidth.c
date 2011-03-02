#include <stdio.h>
#include <CL/opencl.h>
#include "functions.h"
#include "timeRec.h"

/**********************************************************************
 * test the host device and device host memory bandwidth
 *********************************************************************/

#define BLOCK_SIZE 16
#define MAX_BUFF_NUM 20

#define CHECK_ERR(err, str) \
	if (err != CL_SUCCESS)  \
	{ \
		fprintf(stderr, "CL Error %d: %s\n", err, str); \
		exit(1); \
	} \

int main(int argc, char **argv)
{
	if (argc != 3)
	{
		printf("Usage: %s buffSize(1MB) buff#(1)\n", argv[0]);
		return 1;
	}

	double *hostMem[MAX_BUFF_NUM];
	unsigned int buffNum = 1;
	size_t buffSize = 1 << 20;
	int i, megaBytes = 1;

	buffNum = atoi(argv[2]);
	megaBytes = atoi(argv[1]);

	buffSize = buffSize * megaBytes;

	//initialize timer
	memset(&strTime, 0, sizeof(STRUCT_TIME));
	
	cl_int err;
	cl_platform_id platformID;
	cl_device_id deviceID;
	cl_context hContext;
	cl_command_queue hCmdQueue;
	cl_program hProgram;
	cl_mem deviceMem[MAX_BUFF_NUM];
	size_t sourceFileSize;
	char *cSourceCL = NULL;

	//get an opencl platform
	timerStart();
	err = clGetPlatformIDs(1, &platformID, NULL);
	CHECK_ERR(err, "Get platform ID error!");
	timerEnd();
	strTime.getPlatform = elapsedTime();
	strTime.numGetPlatform++;

	timerStart();
	err = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, NULL);
	CHECK_ERR(err, "Get device ID error!");
	timerEnd();
	strTime.getDeviceID = elapsedTime();
	strTime.numGetDeviceID++;

	//create opencl device and context
	timerStart();
	hContext = clCreateContext(0, 1, &deviceID, 0, 0, &err);
	CHECK_ERR(err, "Create context from type error");
	timerEnd();
	strTime.createContext = elapsedTime();
	strTime.numCreateContext++;

	//create a command queue for the first device the context reported
	timerStart();
	hCmdQueue = clCreateCommandQueue(hContext, deviceID, 0, &err);
	CHECK_ERR(err, "Create command queue error");
	timerEnd();
	strTime.createCommandQueue = elapsedTime();
	strTime.numCreateCommandQueue++;

	//create host buffer;
	for (i = 0; i < buffNum; i++)
	{
		hostMem[i] = (double *)malloc(buffSize);
		memset(hostMem[i], 1, buffSize);
	}

	//allocate device memory
	timerStart();
	for (i = 0; i < buffNum; i++)
	{
		deviceMem[i] = clCreateBuffer(hContext,
									 CL_MEM_READ_WRITE,
									 buffSize,
									 0,
									 &err);
		CHECK_ERR(err, "Create deviceMem on device error");
	}
	timerEnd();
	strTime.createBuffer += elapsedTime();
	strTime.numCreateBuffer += buffNum;
	
	//copy the matrix to device memory
	timerStart();
	for (i = 0; i < buffNum; i++)
	{
		err = clEnqueueWriteBuffer(hCmdQueue, deviceMem[i], CL_FALSE, 0, 
								   buffSize,
								   hostMem[i], 0, NULL, NULL);
		CHECK_ERR(err, "Write buffer error!");
	}
	clFinish(hCmdQueue);
	timerEnd();
	strTime.enqueueWriteBuffer += elapsedTime();
	strTime.numEnqueueWriteBuffer += buffNum;

	timerStart();
	for (i = 0; i < buffNum; i++)
	{
		err = clEnqueueReadBuffer(hCmdQueue, deviceMem[i], CL_FALSE, 0,
							buffSize,
							hostMem[i], 0, 0, 0);
		CHECK_ERR(err, "Enqueue read buffer error");
	}
	clFinish(hCmdQueue);
	timerEnd();
	strTime.enqueueReadBuffer += elapsedTime();
	strTime.numEnqueueReadBuffer += buffNum;

	timerStart();
	for (i = 0; i < buffNum; i++)
	{
		clReleaseMemObject(deviceMem[i]);
	}
	timerEnd();
	strTime.releaseMemObj += elapsedTime();
    strTime.numReleaseMemObj += buffNum;

	for (i = 0; i < buffNum; i++)
	{
		free(hostMem[i]);
	}

	timerStart();
	clReleaseContext(hContext);
	timerEnd();
	strTime.releaseContext = elapsedTime();
	strTime.numReleaseContext++;

	timerStart();
	clReleaseCommandQueue(hCmdQueue);
	timerEnd();
	strTime.releaseCmdQueue = elapsedTime();
	strTime.numReleaseCmdQueue++;

	printf("H2D bandwidth = %.3f (GB/s)\n", buffSize * buffNum/strTime.enqueueWriteBuffer/1000000.0);
	printf("D2H bandwidth = %.3f (GB/s)\n", buffSize * buffNum/strTime.enqueueReadBuffer/1000000.0);
	

//	printTime_toStandardOutput();
//	printTime_toFile();

	return 0;
}
