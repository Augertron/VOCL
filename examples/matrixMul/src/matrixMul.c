#include <stdio.h>
#include <CL/opencl.h>
#include "functions.h"
#include "timeRec.h"

#define CHECK_ERR(err, str) \
	if (err != CL_SUCCESS)  \
	{ \
		fprintf(stderr, "CL Error %d: %s\n", err, str); \
		exit(1); \
	} \

char * loadSource(char *filePathName, size_t *fileSize)
{
	FILE *pfile;
	size_t tmpFileSize;
	char *fileBuffer;
	pfile = fopen(filePathName, "rb");

	if (pfile == NULL)
	{
		printf("Open file %s open error!\n", filePathName);
		return NULL;
	}

	fseek(pfile, 0, SEEK_END);
	tmpFileSize = ftell(pfile);

	fileBuffer = (char *)malloc(tmpFileSize);

	fseek(pfile, 0, SEEK_SET);
	fread(fileBuffer, sizeof(char), tmpFileSize, pfile);

	fclose(pfile);

	//debug================================
	//for (int i = 0; i < tmpFileSize; i++)
	//{
	//	printf("%c", fileBuffer[i]);
	//}
	//=====================================

	*fileSize = tmpFileSize;
	return fileBuffer;
}

int main(int argc, char **argv)
{
//	if (argc != 4)
//	{
//		fprintf(stderr, "Usage: %s <input_file> <num steps> <output file>\n", argv[0]);
//		exit(EXIT_FAILURE);
//	}

	double *a, *b, *c;
	int hA = 2048;
	int wA = 2048;
	int wB = 2048;
	int sizeA = hA * wA;
	int sizeB = wA * wB;
	int sizeC = hA * wB;
	int i, j;

	size_t blockSize[2] = {256, 1};
	size_t globalSize[2];

	//initialize timer
	memset(&strTime, 0, sizeof(STRUCT_TIME));
	
	cl_int err;
	cl_platform_id platformID;
	cl_device_id deviceID;
	cl_context hContext;
	cl_command_queue hCmdQueue;
	cl_program hProgram;
	cl_mem deviceMem[3];
	size_t sourceFileSize;
	char *cSourceCL = NULL;

	//get an opencl platform
	timerStart();
	err = clGetPlatformIDs(1, &platformID, NULL);
	CHECK_ERR(err, "Get platform ID error!");
	timerEnd();
	strTime.getPlatform = elapsedTime();

	timerStart();
	err = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, NULL);
	CHECK_ERR(err, "Get device ID error!");
	timerEnd();
	strTime.getDeviceID = elapsedTime();

	//create opencl device and context
	timerStart();
	hContext = clCreateContext(0, 1, &deviceID, 0, 0, &err);
	CHECK_ERR(err, "Create context from type error");
	timerEnd();
	strTime.createContext = elapsedTime();

	//create a command queue for the first device the context reported
	timerStart();
	hCmdQueue = clCreateCommandQueue(hContext, deviceID, 0, &err);
	CHECK_ERR(err, "Create command queue error");
	timerEnd();
	strTime.createCommandQueue = elapsedTime();


	//load the source file
	cSourceCL = loadSource("matrixMul.cl", &sourceFileSize);
	
	//Create & compile program
	timerStart();
	hProgram = clCreateProgramWithSource(hContext, 1, (const char **)&cSourceCL, 
				&sourceFileSize, &err);
	CHECK_ERR(err, "Create program with source error");
	timerEnd();
	strTime.createProgramWithSource = elapsedTime();

	timerStart();
	err = clBuildProgram(hProgram, 0, 0, 0, 0, 0);
	//debug================================
    int logSize = 3000;
    size_t retSize;
    char logTxt[3000];
    err = clGetProgramBuildInfo(hProgram, deviceID, CL_PROGRAM_BUILD_LOG, logSize, logTxt, &retSize);
    for (i = 0; i < retSize; i++)
    {
        printf("%c", logTxt[i]);
    }
    //===================================

	CHECK_ERR(err, "Build program error");
	timerEnd();
	strTime.buildProgram = elapsedTime();

	//create kernel
	cl_kernel hKernel;
	timerStart();
	hKernel = clCreateKernel(hProgram, "matrixMul", &err);
	CHECK_ERR(err, "Create kernel error");
	timerEnd();
	strTime.createKernel = elapsedTime();

	//create input matrix
	timerStart();
	a = (double *)malloc(sizeA * sizeof(double));
	b = (double *)malloc(sizeB * sizeof(double));
	c = (double *)malloc(sizeC * sizeof(double));

	for (i = 0; i < hA; i++)
	{
		for (j = 0; j < wA; j++)
		{
			a[i * wA + j] = i + j;
		}
	}
	
	for (i = 0; i < wA; i++)
	{
		for (j = 0; j < wB; j++)
		{
			b[i * wB + j] = 1.0;
		}
	}
	
	for (i = 0; i < hA; i++)
	{
		for (j = 0; j < wB; j++)
		{
			c[i * wB + j] = 0.0;
		}
	}

	globalSize[0] = ((wB - 1) / blockSize[0] + 1) * blockSize[0];
	globalSize[1] = hA;
	timerEnd();
	strTime.readMatrix = elapsedTime();

	//allocate device memory
	timerStart();
	deviceMem[0] = clCreateBuffer(hContext,
								 CL_MEM_READ_ONLY,
								 sizeA * sizeof(cl_double),
								 0,
								 &err);
	CHECK_ERR(err, "Create deviceMem[0] on device error");

	deviceMem[1] = clCreateBuffer(hContext,
								 CL_MEM_READ_ONLY,
								 sizeB * sizeof(cl_double),
								 0,
								 &err);
	CHECK_ERR(err, "Create deviceMem[1] on device error");
	deviceMem[2] = clCreateBuffer(hContext,
								 CL_MEM_READ_WRITE,
								 sizeC * sizeof(cl_double),
								 0,
								 &err);
	CHECK_ERR(err, "Create deviceMem[1] on device error");

	timerEnd();
	strTime.createBuffer += elapsedTime();

	//copy the matrix to device memory
	timerStart();
	cl_event event;
	err = clEnqueueWriteBuffer(hCmdQueue, deviceMem[0], CL_TRUE, 0, 
	//err = clEnqueueWriteBuffer(hCmdQueue, deviceMem[0], CL_FALSE, 0, 
							   sizeA * sizeof(cl_double),
							   a, 0, NULL, &event);
	err = clWaitForEvents(1, &event);
	CHECK_ERR(err, "Write buffer error!");
	err = clEnqueueWriteBuffer(hCmdQueue, deviceMem[1], CL_TRUE, 0, 
	//err = clEnqueueWriteBuffer(hCmdQueue, deviceMem[1], CL_FALSE, 0, 
							   sizeB * sizeof(cl_double),
							   b, 0, NULL, NULL);
	CHECK_ERR(err, "Write buffer error!");
	err = clEnqueueWriteBuffer(hCmdQueue, deviceMem[2], CL_TRUE, 0, 
	//err = clEnqueueWriteBuffer(hCmdQueue, deviceMem[2], CL_FALSE, 0, 
							   sizeC * sizeof(cl_double),
							   c, 0, NULL, NULL);
	CHECK_ERR(err, "Write buffer error!");

	clFlush(hCmdQueue);

	timerEnd();
	strTime.enqueueWriteBuffer = elapsedTime();

	timerStart();
	err  = clSetKernelArg(hKernel, 0, sizeof(cl_mem), (void *)&deviceMem[0]);
	err |= clSetKernelArg(hKernel, 1, sizeof(cl_mem), (void *)&deviceMem[1]);
	err |= clSetKernelArg(hKernel, 2, sizeof(cl_mem), (void *)&deviceMem[2]);
	err |= clSetKernelArg(hKernel, 3, sizeof(cl_int), (void *)&hA);
	err |= clSetKernelArg(hKernel, 4, sizeof(cl_int), (void *)&wA);
	err |= clSetKernelArg(hKernel, 5, sizeof(cl_int), (void *)&wB);
	CHECK_ERR(err, "Set arguments error!");

	//execute kernel
	err = clEnqueueNDRangeKernel(hCmdQueue, hKernel, 2, NULL, globalSize,
						   blockSize, 0, NULL, NULL);
	CHECK_ERR(err, "Launch kernel error");
	clFinish(hCmdQueue);
	timerEnd();
	strTime.kernelExecution = elapsedTime();

	timerStart();
	err = clEnqueueReadBuffer(hCmdQueue, deviceMem[2], CL_TRUE, 0,
						sizeC * sizeof(cl_double),
						c, 0, 0, 0);
	CHECK_ERR(err, "Enqueue read buffer error");
	timerEnd();
	strTime.enqueueReadBuffer = elapsedTime();

	timerStart();
//	for (i = 0; i < hA; i++)
//	{
//		for (j = 0; j < wB; j++)
//		{
//			printf("c[%d][%d] = %lf\n", i, j, c[i * wB + j]);
//		}
//	}
	timerEnd();
	strTime.printMatrix = elapsedTime();
	printTime_toStandardOutput();
	printTime_toFile();

	free(a);
	free(b);
	free(c);
	free(cSourceCL);

	clReleaseKernel(hKernel);
	clReleaseProgram(hProgram);
	clReleaseCommandQueue(hCmdQueue);
	clReleaseContext(hContext);

	clReleaseMemObject(deviceMem[0]);
	clReleaseMemObject(deviceMem[1]);
	clReleaseMemObject(deviceMem[2]);
}
