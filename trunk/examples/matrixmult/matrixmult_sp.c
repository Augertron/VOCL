#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <CL/opencl.h>
#include <sched.h>
#include "mm_timer.h"

/* if enable vocl reablance, include these files */
#if VOCL_BALANCE
#include <dlfcn.h>
#include "vocl.h"
#include "mpi.h"
#endif

/**********************************************************************
 *1. Create the scenario that each time a different kernel is created.
 *2. The scenario that the same kernel is called for multiple times.
 *********************************************************************/

#define BLOCK_SIZE 16

#define CHECK_ERR(err, str) \
	if (err != CL_SUCCESS)  \
	{ \
		fprintf(stderr, "CL Error %d: %s\n", err, str); \
		exit(1); \
	} \

char *loadSource(char *filePathName, size_t * fileSize)
{
    FILE *pfile;
    size_t tmpFileSize;
    char *fileBuffer;
    pfile = fopen(filePathName, "rb");

    if (pfile == NULL) {
        printf("Open file %s open error!\n", filePathName);
        return NULL;
    }

    fseek(pfile, 0, SEEK_END);
    tmpFileSize = ftell(pfile);

    fileBuffer = (char *) malloc(tmpFileSize);

    fseek(pfile, 0, SEEK_SET);
    fread(fileBuffer, sizeof(char), tmpFileSize, pfile);

    fclose(pfile);

    //debug================================
    //for (int i = 0; i < tmpFileSize; i++)
    //{
    //      printf("%c", fileBuffer[i]);
    //}
    //=====================================

    *fileSize = tmpFileSize;
    return fileBuffer;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        printf("Usage: %s matrixSize, numIterations, deviceNo\n", argv[0]);
        printf("       deviceNo = -1: all virtual GPUs are used.\n");
        return 1;
    }

    //cpu_set_t set;
    //CPU_ZERO(&set);
#if VOCL_BALANCE
	MPI_Init(&argc, &argv);
#endif

    float *a, *b, *c;
    int numIterations = 20, iterationNo;
    int matrixSize = atoi(argv[1]);
    numIterations = atoi(argv[2]);
    int hA = matrixSize;
    int wA = matrixSize;
    int wB = matrixSize;
    int sizeA = hA * wA;
    int sizeB = wA * wB;
    int sizeC = hA * wB;
    int i, j, index;

    size_t blockSize[2] = { BLOCK_SIZE, BLOCK_SIZE };
    size_t globalSize[2];

#if VOCL_BALANCE
	int rankNo; 
	void *voclModulePtr;
	const char *error;
	dlVOCLRebalance dlvbPtr;
	MPI_Comm_rank(MPI_COMM_WORLD, &rankNo);
#endif

    //initialize timer
    memset(&strTime, 0, sizeof(STRUCT_TIME));

    cl_int err;
    cl_uint platformNum, *deviceNums;
    cl_uint totalDeviceNum, usedDeviceNum, deviceNo = 0;
    cl_platform_id *platformIDs;
    cl_device_id *deviceIDs;
    cl_context *hContexts;
    cl_command_queue *hCmdQueues;
    cl_program *hPrograms;
    cl_mem *deviceMems;
    cl_kernel *hKernels;
    size_t sourceFileSize;
    char *cSourceCL = NULL;
    char kernel_source[KERNEL_SOURCE_FILE_LEN];
    deviceNo = atoi(argv[3]);

    //get an opencl platform
    timerStart();
    err = clGetPlatformIDs(0, NULL, &platformNum);
    CHECK_ERR(err, "Get platform ID error!");

    platformIDs = (cl_platform_id *) malloc(sizeof(cl_platform_id) * platformNum);
    deviceNums = (cl_uint *) malloc(sizeof(cl_uint) * platformNum);

    err = clGetPlatformIDs(platformNum, platformIDs, NULL);
    CHECK_ERR(err, "Get platform ID error!");
    timerEnd();
    strTime.getPlatform = elapsedTime();

    //sched_getaffinity(0, sizeof(cpu_set_t), &set);
    //printf("cpuid = %d\n", set.__bits[0]);

    timerStart();
    totalDeviceNum = 0;
    for (i = 0; i < platformNum; i++) {
        err = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceNums[i]);
        totalDeviceNum += deviceNums[i];
    }

    deviceIDs = (cl_device_id *) malloc(sizeof(cl_device_id) * totalDeviceNum);
    hContexts = (cl_context *) malloc(sizeof(cl_context) * totalDeviceNum);
    hCmdQueues = (cl_command_queue *) malloc(sizeof(cl_command_queue) * totalDeviceNum);
    hPrograms = (cl_program *) malloc(sizeof(cl_program) * totalDeviceNum);
    hKernels = (cl_kernel *) malloc(sizeof(cl_kernel) * totalDeviceNum);
    deviceMems = (cl_mem *) malloc(sizeof(cl_mem) * 3 * totalDeviceNum);

    totalDeviceNum = 0;
    for (i = 0; i < platformNum; i++) {
        err = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_GPU, deviceNums[i],
                             &deviceIDs[totalDeviceNum], NULL);
        totalDeviceNum += deviceNums[i];
        CHECK_ERR(err, "Get device ID error!");
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
    for (i = 0; i < usedDeviceNum; i++) {
        index = i + deviceNo;
        hContexts[i] = clCreateContext(0, 1, &deviceIDs[index], 0, 0, &err);
        CHECK_ERR(err, "Create context from type error");
    }
    timerEnd();
    strTime.createContext = elapsedTime();

    //create a command queue for the first device the context reported
    timerStart();
    for (i = 0; i < usedDeviceNum; i++) {
        index = i + deviceNo;
        hCmdQueues[i] = clCreateCommandQueue(hContexts[i], deviceIDs[index], 0, &err);
        CHECK_ERR(err, "Create command queue error");
    }
    timerEnd();
    strTime.createCommandQueue = elapsedTime();

    //load the source file
    snprintf(kernel_source, KERNEL_SOURCE_FILE_LEN,
             "%s/examples/matrixmult/matrixmult_sp.cl", ABS_SRCDIR);
    cSourceCL = loadSource(kernel_source, &sourceFileSize);

    //Create & compile program
    timerStart();
    for (i = 0; i < usedDeviceNum; i++) {
        hPrograms[i] = clCreateProgramWithSource(hContexts[i], 1, (const char **) &cSourceCL,
                                                 &sourceFileSize, &err);
        CHECK_ERR(err, "Create program with source error");
    }
    timerEnd();
    strTime.createProgramWithSource = elapsedTime();

    timerStart();
    for (i = 0; i < usedDeviceNum; i++) {
        err = clBuildProgram(hPrograms[i], 0, 0, 0, 0, 0);
        CHECK_ERR(err, "Build program error");
    }
    timerEnd();
    strTime.buildProgram = elapsedTime();
    strTime.numBuildProgram++;

    //create input matrix
    timerStart();
    a = (float *) malloc(sizeA * sizeof(float));
    b = (float *) malloc(sizeB * sizeof(float));
    c = (float *) malloc(sizeC * sizeof(float));

    for (i = 0; i < hA; i++) {
        for (j = 0; j < wA; j++) {
            a[i * wA + j] = i + j;
        }
    }

    for (i = 0; i < wA; i++) {
        for (j = 0; j < wB; j++) {
            b[i * wB + j] = 1.0;
        }
    }

    for (i = 0; i < hA; i++) {
        for (j = 0; j < wB; j++) {
            c[i * wB + j] = 0.0;
        }
    }

    globalSize[0] = ((wB - 1) / blockSize[0] + 1) * blockSize[0];
    globalSize[1] = ((hA - 1) / blockSize[1] + 1) * blockSize[1];
    timerEnd();
    strTime.readMatrix = elapsedTime();

    //allocate device memory
    timerStart();
    for (i = 0; i < usedDeviceNum; i++) {
        deviceMems[3 * i] = clCreateBuffer(hContexts[i],
                                           CL_MEM_READ_WRITE, sizeA * sizeof(cl_float), 0,
                                           &err);
        CHECK_ERR(err, "Create deviceMem[0] on device error");

        deviceMems[3 * i + 1] = clCreateBuffer(hContexts[i],
                                               CL_MEM_READ_WRITE, sizeB * sizeof(cl_float), 0,
                                               &err);
        CHECK_ERR(err, "Create deviceMem[1] on device error");
        deviceMems[3 * i + 2] = clCreateBuffer(hContexts[i],
                                               CL_MEM_READ_WRITE, sizeC * sizeof(cl_float), 0,
                                               &err);
        CHECK_ERR(err, "Create deviceMem[2] on device error");
    }
    timerEnd();
    strTime.createBuffer += elapsedTime();

    //create kernel
    timerStart();
    for (i = 0; i < usedDeviceNum; i++) {
        hKernels[i] = clCreateKernel(hPrograms[i], "matrixMul", &err);
        CHECK_ERR(err, "Create kernel error");
    }
    timerEnd();
    strTime.createKernel += elapsedTime();

#if VOCL_BALANCE
	voclModulePtr = dlopen("libvocl.so", RTLD_NOW);
	if (voclModulePtr == NULL)
	{
		printf("open libvocl.so error, %s\n", dlerror());
		exit (1);
	}

	dlvbPtr = dlsym(voclModulePtr, "voclRebalance");
	if (error = dlerror()) {
		printf("Could find voclRebalance: %s\n", error);
		exit(1);
	}
#endif

    timerStart();
    //copy the matrix to device memory
    for (iterationNo = 0; iterationNo < numIterations; iterationNo++) {
        for (i = 0; i < usedDeviceNum; i++) {
            err = clEnqueueWriteBuffer(hCmdQueues[i], deviceMems[3 * i], CL_FALSE, 0,
                                       sizeA * sizeof(cl_float), a, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(hCmdQueues[i], deviceMems[3 * i + 1], CL_FALSE, 0,
                                        sizeB * sizeof(cl_float), b, 0, NULL, NULL);
            CHECK_ERR(err, "Write buffer error!");
        }

        for (i = 0; i < usedDeviceNum; i++) {
            err = clSetKernelArg(hKernels[i], 0, sizeof(cl_mem), (void *) &deviceMems[3 * i]);
            err |=
                clSetKernelArg(hKernels[i], 1, sizeof(cl_mem),
                               (void *) &deviceMems[3 * i + 1]);
            err |=
                clSetKernelArg(hKernels[i], 2, sizeof(cl_mem),
                               (void *) &deviceMems[3 * i + 2]);
            err |=
                clSetKernelArg(hKernels[i], 3,
                               sizeof(cl_float) * BLOCK_SIZE * (BLOCK_SIZE + 1),
                               (void *) NULL);
            err |=
                clSetKernelArg(hKernels[i], 4,
                               sizeof(cl_float) * BLOCK_SIZE * (BLOCK_SIZE + 1),
                               (void *) NULL);
            err |= clSetKernelArg(hKernels[i], 5, sizeof(cl_int), (void *) &hA);
            err |= clSetKernelArg(hKernels[i], 6, sizeof(cl_int), (void *) &wA);
            err |= clSetKernelArg(hKernels[i], 7, sizeof(cl_int), (void *) &wB);
            err |= clEnqueueNDRangeKernel(hCmdQueues[i], hKernels[i], 2, NULL, globalSize,
                                          blockSize, 0, NULL, NULL);
        }

        for (i = 0; i < usedDeviceNum; i++) {
            err = clEnqueueReadBuffer(hCmdQueues[i], deviceMems[3 * i + 2], CL_FALSE, 0,
                                      sizeC * sizeof(cl_float), c, 0, 0, 0);
            CHECK_ERR(err, "Enqueue read buffer error");
        }

#if VOCL_BALANCE
		if (rankNo == 0 && iterationNo == 10)
		{
			for (i = 0; i  < usedDeviceNum; i++)
			{
				(*dlvbPtr)(hCmdQueues[i]);
			}
		}
#endif
    }

    for (i = 0; i < usedDeviceNum; i++) {
        clFinish(hCmdQueues[i]);
    }
    timerEnd();
    strTime.kernelExecution += elapsedTime();

    timerStart();
    for (i = 0; i < usedDeviceNum; i++) {
        clReleaseKernel(hKernels[i]);
    }
    timerEnd();
    strTime.releaseKernel += elapsedTime();

    timerStart();
    for (i = 0; i < usedDeviceNum; i++) {
        clReleaseMemObject(deviceMems[3 * i]);
        clReleaseMemObject(deviceMems[3 * i + 1]);
        clReleaseMemObject(deviceMems[3 * i + 2]);
    }
    timerEnd();
    strTime.releaseMemObj += elapsedTime();

    timerStart();
	for (i = hA-5; i < hA; i++)
	{
		for (j = 0; j < 1; j++)
		{
			printf("c[%d][%d] = %lf\n", i, j, c[i * wB + j]);
		}
	}
    timerEnd();
    strTime.printMatrix = elapsedTime();

    free(a);
    free(b);
    free(c);
    free(cSourceCL);

    timerStart();
    for (i = 0; i < usedDeviceNum; i++) {
        clReleaseProgram(hPrograms[i]);
    }
    timerEnd();
    strTime.releaseProgram = elapsedTime();

    timerStart();
    for (i = 0; i < usedDeviceNum; i++) {
        clReleaseCommandQueue(hCmdQueues[i]);
    }
    timerEnd();
    strTime.releaseCmdQueue = elapsedTime();

    timerStart();
    for (i = 0; i < usedDeviceNum; i++) {
        clReleaseContext(hContexts[i]);
    }
    timerEnd();

    free(platformIDs);
    free(deviceIDs);
    free(hContexts);
    free(hCmdQueues);
    free(hPrograms);
    free(hKernels);
    free(deviceMems);

    printTime_toStandardOutput();
    printTime_toFile();

#if VOCL_BALANCE
	dlclose(voclModulePtr);
	MPI_Finalize();
#endif

    return 0;
}
