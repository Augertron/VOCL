#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <CL/opencl.h>
#include "timeRec.h"
#include <sched.h>

/**********************************************************************
 * test the host device and device host memory bandwidth
 *********************************************************************/
#define BLOCK_SIZE 16
#define MAX_BUFF_NUM 200

#define CHECK_ERR(err, str) \
	if (err != CL_SUCCESS)  \
	{ \
		fprintf(stderr, "CL Error %d: %s\n", err, str); \
		exit(1); \
	} \

int main(int argc, char **argv)
{
	if (argc != 4)
	{
		printf("Usage: %s buffSize(1MB) buff#(1) devID\n", argv[0]);
		return 1;
	}

	cpu_set_t set;
	CPU_ZERO(&set);
	//CPU_SETSIZE(1024, &set);
	//CPU_SET(0, &set);
	sched_setaffinity(0, sizeof(set), &set);

	double *hostMem[MAX_BUFF_NUM];
	unsigned int buffNum = 1;
	size_t buffSize = 1 << 10;
	int i;
	float megaBytes = 1.0f;
	float issueTime;
	FILE *pfile;
	int deviceNo = 0;

	buffNum = atoi(argv[2]);
	megaBytes = atof(argv[1]);
	deviceNo = atoi(argv[3]);

	buffSize = (size_t)(buffSize * megaBytes);

	//initialize timer
	memset(&strTime, 0, sizeof(STRUCT_TIME));
	
	cl_int err;
	cl_platform_id platformID;
	cl_device_id deviceID[2];
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
	err = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 2, deviceID, NULL);
	CHECK_ERR(err, "Get device ID error!");
	timerEnd();
	strTime.getDeviceID = elapsedTime();
	strTime.numGetDeviceID++;

	//create opencl device and context
	timerStart();
	hContext = clCreateContext(0, 1, &deviceID[deviceNo], 0, 0, &err);
	CHECK_ERR(err, "Create context from type error");
	timerEnd();
	strTime.createContext = elapsedTime();
	strTime.numCreateContext++;

	//create a command queue for the first device the context reported
	timerStart();
	hCmdQueue = clCreateCommandQueue(hContext, deviceID[deviceNo], 0, &err);
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
	
//	for (i = 0; i < 20; i++)
//	{
//		err = clEnqueueWriteBuffer(hCmdQueue, deviceMem[0], CL_FALSE, 0, 
//								   buffSize,
//								   hostMem[0], 0, NULL, NULL);
//		CHECK_ERR(err, "Write buffer error!");
//	}
//	clFinish(hCmdQueue);

	//copy the matrix to device memory
	timerStart();
	for (i = 0; i < buffNum; i++)
	{
		err = clEnqueueWriteBuffer(hCmdQueue, deviceMem[i], CL_FALSE, 0, 
								   buffSize,
								   hostMem[i], 0, NULL, NULL);
		CHECK_ERR(err, "Write buffer error!");
	}
	timerEnd();
	issueTime = elapsedTime();

	timerStart();
	clFinish(hCmdQueue);
	timerEnd();
	strTime.enqueueWriteBuffer += elapsedTime();
	printf("writeTime:%.3f, %.3f\n", issueTime, strTime.enqueueWriteBuffer);
	strTime.enqueueWriteBuffer += issueTime;
	strTime.numEnqueueWriteBuffer += buffNum;
	
//	for (i = 0; i < 20; i++)
//	{
//		err = clEnqueueReadBuffer(hCmdQueue, deviceMem[0], CL_FALSE, 0,
//							buffSize,
//							hostMem[0], 0, 0, 0);
//		CHECK_ERR(err, "Enqueue read buffer error");
//	}
//	clFinish(hCmdQueue);

	timerStart();
	for (i = 0; i < buffNum; i++)
	{
		err = clEnqueueReadBuffer(hCmdQueue, deviceMem[i], CL_FALSE, 0,
							buffSize,
							hostMem[i], 0, 0, 0);
		CHECK_ERR(err, "Enqueue read buffer error");
	}
	timerEnd();
	issueTime = elapsedTime();

	timerStart();
	clFinish(hCmdQueue);
	timerEnd();
	strTime.enqueueReadBuffer += elapsedTime();
	strTime.numEnqueueReadBuffer += buffNum;
	printf("readTime:%.3f, %.3f\n", issueTime, strTime.enqueueReadBuffer);
	strTime.enqueueReadBuffer += issueTime;

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

	pfile = fopen("bandWidth.txt", "at");
	if (pfile == NULL)
	{
		printf("File open error!\n");
		exit (1);
	}

//	printf("H2D bandwidth = %.3f (GB/s)\n", buffSize * buffNum/strTime.enqueueWriteBuffer/1000000.0);
//	printf("D2H bandwidth = %.3f (GB/s)\n", buffSize * buffNum/strTime.enqueueReadBuffer/1000000.0);
	printf("%.1fKB\t\%d\t%.3f\t%.3f\n", megaBytes, buffNum, 
			buffSize * buffNum/strTime.enqueueWriteBuffer/1000000.0,
			buffSize * buffNum/strTime.enqueueReadBuffer/1000000.0);
	fprintf(pfile, "%.1fKB\t\%d\t%.3f\t%.3f\n", megaBytes, buffNum, 
			buffSize * buffNum/strTime.enqueueWriteBuffer/1000000.0,
			buffSize * buffNum/strTime.enqueueReadBuffer/1000000.0);
	fclose(pfile);
//	printTime_toStandardOutput();
//	printTime_toFile();

	return 0;
}
