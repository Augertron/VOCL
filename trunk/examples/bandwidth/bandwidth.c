#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <CL/opencl.h>
#include "bw_timer.h"
#include <sched.h>

/**********************************************************************
 * test the host device and device host memory bandwidth
 *********************************************************************/

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
	cl_uint i, index, deviceIndex;
	float megaBytes = 1.0f;
	float issueTime;
	FILE *pfile;

	buffNum = atoi(argv[2]);
	megaBytes = atof(argv[1]);

	buffSize = (size_t)(buffSize * megaBytes);

	cl_int err;
	cl_uint platformNum, *deviceNums;
	cl_uint totalDeviceNum, usedDeviceNum, deviceNo = 0;
	cl_platform_id *platformIDs;
	cl_device_id *deviceIDs;
	cl_context *hContexts;
	cl_command_queue *hCmdQueues;
	cl_mem *deviceMems;
	cpu_set_t set;
	deviceNo = atoi(argv[3]);
    sched_getaffinity(0, sizeof(set), &set);
	printf("cpuid = %d\n", set.__bits[0]);

	//get an opencl platform
	timerStart();
	err = clGetPlatformIDs(0, NULL, &platformNum);
	CHECK_ERR(err, "Get platform ID error!");
	printf("platformNum = %d\n", platformNum);
	platformIDs = (cl_platform_id *) malloc(sizeof(cl_platform_id) * platformNum);
	deviceNums = (cl_uint *) malloc(sizeof(cl_uint) * platformNum);
	err = clGetPlatformIDs(platformNum, platformIDs, NULL);
	CHECK_ERR(err, "Get platform ID error!");
	timerEnd();
	strTime.getPlatform = elapsedTime();


	timerStart();
	totalDeviceNum = 0;
	for (i = 0; i < platformNum; i++)
	{
		err = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceNums[i]);
		CHECK_ERR(err, "Get device ID error!");
		totalDeviceNum += deviceNums[i];
	}

    deviceIDs = (cl_device_id *) malloc(sizeof(cl_device_id) * totalDeviceNum);
    hContexts = (cl_context *) malloc(sizeof(cl_context) * totalDeviceNum);
    hCmdQueues = (cl_command_queue *) malloc(sizeof(cl_command_queue) * totalDeviceNum);
    deviceMems = (cl_mem *) malloc(sizeof(cl_mem) * (MAX_BUFF_NUM) * totalDeviceNum);

	totalDeviceNum = 0;
	for (i = 0; i < platformNum; i++)
	{
		err = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_GPU, deviceNums[i], &deviceIDs[totalDeviceNum], NULL);
		CHECK_ERR(err, "Get device ID error!");
		totalDeviceNum += deviceNums[i];
	}
	timerEnd();
	strTime.getDeviceID = elapsedTime();

    /* deviceNo is -1 means all virtual GPUs are used */
    if (deviceNo == -1) {
        printf("All GPUs are used..., deviceCount = %d\n", totalDeviceNum);
        usedDeviceNum = totalDeviceNum;
        deviceNo = 0;
    }
    else {
        printf("device %d is used...\n\n", deviceNo);
        usedDeviceNum = 1;
    }

    if (deviceNo >= totalDeviceNum) {
        printf("Device no %d is larger than the total device num %d...\n\n", deviceNo,
               totalDeviceNum);
    }


	//create opencl device and context
	timerStart();
	for (i = 0; i < usedDeviceNum; i++)
	{
		deviceIndex = i + deviceNo; 
		hContexts[i] = clCreateContext(0, 1, &deviceIDs[deviceIndex], 0, 0, &err);
		CHECK_ERR(err, "Create context from type error");
	}
	timerEnd();
	strTime.createContext = elapsedTime();

	//create a command queue for the first device the context reported
	timerStart();
	for (i = 0; i < usedDeviceNum; i++)
	{
		deviceIndex = i + deviceNo;
		hCmdQueues[i] = clCreateCommandQueue(hContexts[i], deviceIDs[deviceIndex], 0, &err);
		CHECK_ERR(err, "Create command queue error");
	}
	timerEnd();
	strTime.createCommandQueue = elapsedTime();

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
		for (deviceIndex = 0; deviceIndex < usedDeviceNum; deviceIndex++)
		{
			deviceMems[i+buffNum*deviceIndex] = clCreateBuffer(hContexts[deviceIndex],
									 CL_MEM_READ_WRITE,
									 buffSize,
									 0,
									 &err);
			CHECK_ERR(err, "Create deviceMem on device error");
		}
	}
	timerEnd();
	CHECK_ERR(err, "clFinish, create buffer error!");
	strTime.createBuffer += elapsedTime();

	for (i = 0; i < buffNum; i++)
	{
		for (deviceIndex = 0; deviceIndex < usedDeviceNum; deviceIndex++)
		{
			err = clEnqueueWriteBuffer(hCmdQueues[deviceIndex], deviceMems[i+buffNum*deviceIndex], CL_FALSE, 0, 
								   buffSize,
								   hostMem[i], 0, NULL, NULL);
			CHECK_ERR(err, "Write buffer error!");
		}
	}
	
	for (deviceIndex = 0; deviceIndex < usedDeviceNum; deviceIndex++)
	{
		err = clFinish(hCmdQueues[deviceIndex]);
		CHECK_ERR(err, "clFinish, Write buffer error!");
	}

	//copy the matrix to device memory
	timerStart();
	for (i = 0; i < buffNum; i++)
	{
		for (deviceIndex = 0; deviceIndex < usedDeviceNum; deviceIndex++)
		{
			err = clEnqueueWriteBuffer(hCmdQueues[deviceIndex], deviceMems[i+buffNum*deviceIndex], CL_FALSE, 0, 
								   buffSize,
								   hostMem[i], 0, NULL, NULL);
			CHECK_ERR(err, "Write buffer error!");
		}
	}
	timerEnd();
	issueTime = elapsedTime();

	timerStart();
	for (deviceIndex = 0; deviceIndex < usedDeviceNum; deviceIndex++)
	{
		err = clFinish(hCmdQueues[deviceIndex]);
		CHECK_ERR(err, "clFinish, Write buffer error!");
	}
	timerEnd();
	strTime.enqueueWriteBuffer = elapsedTime();
	printf("===>>writeTime:%.3f, %.3f\n", issueTime, strTime.enqueueWriteBuffer);
	strTime.enqueueWriteBuffer += issueTime;
  
	for (i = 0; i < buffNum; i++)
	{
		for (deviceIndex = 0; deviceIndex < usedDeviceNum; deviceIndex++)
		{
			err = clEnqueueReadBuffer(hCmdQueues[deviceIndex], deviceMems[i+buffNum*deviceIndex], CL_FALSE, 0,
							buffSize,
							hostMem[i], 0, 0, 0);
			CHECK_ERR(err, "Enqueue read buffer error");
		}
	}


	for (deviceIndex = 0; deviceIndex < usedDeviceNum; deviceIndex++)
	{
		err = clFinish(hCmdQueues[deviceIndex]);
		CHECK_ERR(err, "clFinish, Read buffer error!");
	}

	timerStart();
	for (i = 0; i < buffNum; i++)
	{
		for (deviceIndex = 0; deviceIndex < usedDeviceNum; deviceIndex++)
		{
			err = clEnqueueReadBuffer(hCmdQueues[deviceIndex], deviceMems[i+buffNum*deviceIndex], CL_FALSE, 0,
							buffSize,
							hostMem[i], 0, 0, 0);
			CHECK_ERR(err, "Enqueue read buffer error");
		}
	}
	timerEnd();
	issueTime = elapsedTime();

	timerStart();
	for (deviceIndex = 0; deviceIndex < usedDeviceNum; deviceIndex++)
	{
		err = clFinish(hCmdQueues[deviceIndex]);
		CHECK_ERR(err, "clFinish, read buffer error!");
	}
	timerEnd();
	strTime.enqueueReadBuffer += elapsedTime();

	printf("===>>readTime:%.3f, %.3f\n", issueTime, strTime.enqueueReadBuffer);
	strTime.enqueueReadBuffer += issueTime;

	timerStart();
	for (i = 0; i < buffNum * usedDeviceNum; i++)
	{
		clReleaseMemObject(deviceMems[i]);
	}
	timerEnd();
	strTime.releaseMemObj += elapsedTime();

	for (i = 0; i < buffNum; i++)
	{
		free(hostMem[i]);
	}

	timerStart();
	for (i = 0; i < usedDeviceNum; i++)
	{
		clReleaseCommandQueue(hCmdQueues[i]);
	}
	timerEnd();
	strTime.releaseCmdQueue = elapsedTime();

	timerStart();
	for (i = 0; i < usedDeviceNum; i++)
	{
		clReleaseContext(hContexts[i]);
	}
	timerEnd();
	strTime.releaseContext = elapsedTime();

	free(platformIDs);
	free(deviceNums);
	free(deviceIDs);
	free(hContexts);
	free(hCmdQueues);
	free(deviceMems);

	pfile = fopen("bandWidth.txt", "at");
	if (pfile == NULL)
	{
		printf("File open error!\n");
		exit (1);
	}

	printf("%.1fKB\t\%d\t%.3f\t%.3f\n", megaBytes, buffNum, 
			buffSize * buffNum * usedDeviceNum/strTime.enqueueWriteBuffer/1000000.0,
			buffSize * buffNum * usedDeviceNum/strTime.enqueueReadBuffer/1000000.0);
	fprintf(pfile, "%.1fKB\t\%d\t%.3f\t%.3f\n", megaBytes, buffNum, 
			buffSize * buffNum * usedDeviceNum/strTime.enqueueWriteBuffer/1000000.0,
			buffSize * buffNum * usedDeviceNum/strTime.enqueueReadBuffer/1000000.0);
	fclose(pfile);

	return 0;
}
