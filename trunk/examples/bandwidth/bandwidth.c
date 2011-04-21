#define _GNU_SOURCE
#include <stdio.h>
#include <CL/opencl.h>
#include "bw_timer.h"
#include <sched.h>

/**********************************************************************
 * test the host device and device host memory bandwidth
 *********************************************************************/

#define BLOCK_SIZE 16
#define MAX_BUFF_NUM 60
#define MAX_NUM 200

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
		printf("Usage: %s buffSize(1KB) buff#(1) devID\n", argv[0]);
		return 1;
	}

	double *hostMem[MAX_NUM];
	unsigned int buffNum = 1;
	size_t buffSize = 1 << 10;
	int i, index;
	float megaBytes = 1.0f;
	float issueTime;
	FILE *pfile;
	int deviceNo = 0;

	buffNum = atoi(argv[2]);
	megaBytes = atof(argv[1]);
	deviceNo = atoi(argv[3]);

	buffSize = (size_t)(buffSize * megaBytes);


	cl_int err;
	cl_platform_id platformID;
	cl_device_id deviceID[2];
	cl_context hContext;
	cl_command_queue hCmdQueue;
	cl_program hProgram;
	cl_mem deviceMem[MAX_BUFF_NUM];
	cl_uint numPlatforms, deviceCount;
	size_t sourceFileSize;
	char *cSourceCL = NULL;
	cpu_set_t set;

	//get an opencl platform
	timerStart();
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	printf("numPlatforms = %d\n", numPlatforms);
	err = clGetPlatformIDs(1, &platformID, NULL);
	CHECK_ERR(err, "Get platform ID error!");
	timerEnd();
	strTime.getPlatform = elapsedTime();
	strTime.numGetPlatform++;

    sched_getaffinity(0, sizeof(set), &set);
	printf("cpuid = %d\n", set.__bits[0]);

	timerStart();
	err = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
	printf("deviceCount = %d\n", deviceCount);
	err = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, deviceCount, deviceID, NULL);
	CHECK_ERR(err, "Get device ID error!");
	timerEnd();
	strTime.getDeviceID = elapsedTime();
	strTime.numGetDeviceID++;

	//create opencl device and context
	timerStart();
	hContext = clCreateContext(0, 2, deviceID, 0, 0, &err);
	CHECK_ERR(err, "Create context from type error");
	timerEnd();
	strTime.createContext = elapsedTime();
	strTime.numCreateContext++;

	//create a command queue for the first device the context reported
	timerStart();
	hCmdQueue = clCreateCommandQueue(hContext, deviceID[deviceNo], CL_QUEUE_PROFILING_ENABLE, &err);
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
	int tmpNum = MAX_BUFF_NUM < buffNum ? MAX_BUFF_NUM : buffNum;
	for (i = 0; i < tmpNum; i++)
	{
		deviceMem[i] = clCreateBuffer(hContext,
									 CL_MEM_READ_WRITE,
									 buffSize,
									 0,
									 &err);
		CHECK_ERR(err, "Create deviceMem on device error");
	}
	timerEnd();
	err = clFinish(hCmdQueue);
	CHECK_ERR(err, "clFinish, create buffer error!");
	strTime.createBuffer += elapsedTime();
	strTime.numCreateBuffer += buffNum;

	for (i = 0; i < buffNum; i++)
	{
		index = i % MAX_BUFF_NUM;
		err = clEnqueueWriteBuffer(hCmdQueue, deviceMem[index], CL_FALSE, 0, 
								   buffSize,
								   hostMem[i], 0, NULL, NULL);
		CHECK_ERR(err, "Write buffer error!");
	}
	
	err = clFinish(hCmdQueue);
	CHECK_ERR(err, "clFinish, Write buffer error!");

	//copy the matrix to device memory
	timerStart();
	for (i = 0; i < buffNum; i++)
	{
		index = i % MAX_BUFF_NUM;
		err = clEnqueueWriteBuffer(hCmdQueue, deviceMem[index], CL_FALSE, 0, 
								   buffSize,
								   hostMem[i], 0, NULL, NULL);
		CHECK_ERR(err, "Write buffer error!");
	}
	timerEnd();
	issueTime = elapsedTime();

	timerStart();
	err = clFinish(hCmdQueue);
	CHECK_ERR(err, "clFinish, Write buffer error!");
	timerEnd();
	strTime.enqueueWriteBuffer = elapsedTime();
	printf("===>>writeTime:%.3f, %.3f\n", issueTime, strTime.enqueueWriteBuffer);
	strTime.enqueueWriteBuffer += issueTime;
	strTime.numEnqueueWriteBuffer += buffNum;
  
	for (i = 0; i < buffNum; i++)
	{
		index = i % MAX_BUFF_NUM;
		err = clEnqueueReadBuffer(hCmdQueue, deviceMem[index], CL_FALSE, 0,
							buffSize,
							hostMem[i], 0, 0, 0);
		CHECK_ERR(err, "Enqueue read buffer error");
	}
	err = clFinish(hCmdQueue);
	CHECK_ERR(err, "clFinish, Read buffer error!");

	timerStart();
	for (i = 0; i < buffNum; i++)
	{
		index = i % MAX_BUFF_NUM;
		err = clEnqueueReadBuffer(hCmdQueue, deviceMem[index], CL_FALSE, 0,
							buffSize,
							hostMem[i], 0, 0, 0);
		CHECK_ERR(err, "Enqueue read buffer error");
	}
	timerEnd();
	issueTime = elapsedTime();

	timerStart();
	err = clFinish(hCmdQueue);
	CHECK_ERR(err, "clFinish, read buffer error!");
	timerEnd();
	strTime.enqueueReadBuffer += elapsedTime();
	strTime.numEnqueueReadBuffer += buffNum;
	printf("===>>readTime:%.3f, %.3f\n", issueTime, strTime.enqueueReadBuffer);
	strTime.enqueueReadBuffer += issueTime;

	timerStart();
	for (i = 0; i < tmpNum; i++)
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

	printf("%.1fKB\t\%d\t%.3f\t%.3f\n", megaBytes, buffNum, 
			buffSize * buffNum/strTime.enqueueWriteBuffer/1000000.0,
			buffSize * buffNum/strTime.enqueueReadBuffer/1000000.0);
	fprintf(pfile, "%.1fKB\t\%d\t%.3f\t%.3f\n", megaBytes, buffNum, 
			buffSize * buffNum/strTime.enqueueWriteBuffer/1000000.0,
			buffSize * buffNum/strTime.enqueueReadBuffer/1000000.0);
	fclose(pfile);

	return 0;
}
