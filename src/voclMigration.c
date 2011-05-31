#include <stdio.h>
#include "voclStructures.h"
#include "voclKernelArgProc.h"
#include "voclMigrationDeviceProc.h"

#define VOCL_MIG_CHECK_ERR(err, str) \
	if (err != CL_SUCCESS)  \
	{ \
		fprintf(stderr, "CL Error %d: %s\n", err, str); \
		exit(1); \
	}

extern int voclMigIssueGPUMemoryWrite(MPI_Comm oldComm, MPI_Comm oldCommData, int oldRank,
                                      MPI_Comm newComm, int newRank, int isFromLocal,
                                      int isToLocal, cl_command_queue command_queue,
                                      cl_mem mem, size_t size);

extern int voclMigIssueGPUMemoryRead(MPI_Comm oldComm, int oldRank,
                                     MPI_Comm newComm, MPI_Comm newCommData, int newRank,
                                     int isFromLocal, int isToLocal,
                                     cl_command_queue command_queue, cl_mem mem, size_t size);

extern void voclMigFinishDataTransfer(MPI_Comm oldComm, int oldRank,
                                      cl_command_queue oldCmdQueue, MPI_Comm newComm,
                                      int newRank, cl_command_queue newCmdQueu,
                                      int proxySourceRank, int proxyDestRank, int isFromLocal,
                                      int isToLocal);

extern vocl_device_id voclGetCommandQueueDeviceID(vocl_command_queue cmdQueue);
extern vocl_context voclGetCommandQueueContext(vocl_command_queue cmdQueue);
extern cl_command_queue voclVOCLCommandQueue2CLCommandQueueComm(vocl_command_queue
                                                                command_queue, int *proxyRank,
                                                                int *proxyIndex,
                                                                MPI_Comm * proxyComm,
                                                                MPI_Comm * proxyCommData);
extern void voclUpdateVOCLCommandQueue(vocl_command_queue voclCmdQueue, int proxyRank,
                                       int proxyIndex, MPI_Comm comm, MPI_Comm commData,
                                       vocl_context context, vocl_device_id device);
extern cl_command_queue voclVOCLCommandQueue2CLCommandQueue(vocl_command_queue command_queue);
extern cl_int clMigReleaseOldCommandQueue(vocl_command_queue command_queue);

extern kernel_info *getKernelPtr(cl_kernel kernel);
extern vocl_program voclGetProgramFromKernel(vocl_kernel kernel);
extern void voclUpdateVOCLKernel(vocl_kernel voclKernel, int proxyRank, int proxyIndex,
                                 MPI_Comm proxyComm, MPI_Comm proxyCommData,
                                 vocl_program program);

extern cl_device_id voclVOCLDeviceID2CLDeviceIDComm(vocl_device_id device, int *proxyRank,
                                                    int *proxyIndex, MPI_Comm * proxyComm,
                                                    MPI_Comm * proxyCommData);

extern void voclUpdateVOCLContext(vocl_context voclContext, int proxyRank, int proxyIndex,
                                  MPI_Comm proxyComm, MPI_Comm proxyCommData,
                                  vocl_device_id deviceID);

extern void voclUpdateVOCLProgram(vocl_program voclProgram, int proxyRank, int proxyIndex,
                                  MPI_Comm proxyComm, MPI_Comm proxyCommData,
                                  vocl_context context);


extern size_t voclGetVOCLMemorySize(vocl_mem memory);
extern cl_mem voclVOCLMemory2CLMemory(vocl_mem memory);
extern void voclUpdateVOCLMemory(vocl_mem voclMemory, int proxyRank, int proxyIndex,
                                 MPI_Comm proxyComm, MPI_Comm proxyCommData,
                                 vocl_context context);
extern cl_int clMigReleaseOldMemObject(vocl_mem memobj);

vocl_device_id voclSearchTargetGPU(size_t size)
{
    int i, j, index, err;
    cl_platform_id *platformID;
    cl_device_id *deviceID;
    vocl_device_id targetDeviceID;
    int numPlatforms, *numDevices, totalDeviceNum;
    char cBuffer[256];
    cl_ulong mem_size;

    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    VOCL_MIG_CHECK_ERR(err, "migration, clGetPlatformIDs");
    platformID = (cl_platform_id *) malloc(sizeof(cl_platform_id) * numPlatforms);
    numDevices = (int *) malloc(sizeof(int) * numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platformID, NULL);

    /* retrieve the total number of devices on all platforms */
    totalDeviceNum = 0;
    for (i = 0; i < numPlatforms; i++) {
        err = clGetDeviceIDs(platformID[i], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices[i]);
        VOCL_MIG_CHECK_ERR(err, "migration, clGetDeviceIDs");
        totalDeviceNum += numDevices[i];
    }

    deviceID = (cl_device_id *) malloc(sizeof(cl_device_id) * totalDeviceNum);

    /* get all device ids */
    totalDeviceNum = 0;
    for (i = 0; i < numPlatforms; i++) {
        err =
            clGetDeviceIDs(platformID[i], CL_DEVICE_TYPE_GPU, numDevices[i],
                           &deviceID[totalDeviceNum], NULL);
        VOCL_MIG_CHECK_ERR(err, "migration, clGetDeviceIDs");
        totalDeviceNum += numDevices[i];
    }

    /* get global memory size of each GPU */
    for (i = 2; i < totalDeviceNum; i++) {
        err =
            clGetDeviceInfo(deviceID[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size),
                            &mem_size, NULL);
        VOCL_MIG_CHECK_ERR(err, "migration, clGetDeviceInfo");
        if (mem_size >= size) {
            targetDeviceID = deviceID[i];
            break;
        }
    }

    free(platformID);
    free(numDevices);
    free(deviceID);

    return targetDeviceID;

}

int voclCheckIsMigrationNeeded(vocl_command_queue cmdQueue, kernel_args *argsPtr, int argsNum)
{
	VOCL_LIB_DEVICE *devicePtr;
	int isMigrationNeeded = 0;
	size_t sizeForKernel = 0;
	cl_mem memory;
	int i;

	devicePtr = voclLibGetDeviceIDFromCmdQueue(cmdQueue);
	for (i = 0; i < argsNum; i++)
	{
		/* if it is glboal memory. check if device memory is enough */
		if (argsPtr[i].isGlobalMemory == 1)
		{
			/* if the current global memory is not bind on the device */
			/* check whether the global memory size is enougth */
			memory = *((cl_mem *)argsPtr[i].arg_value);
			if (voclLibIsMemoryOnDevice(devicePtr, memory) == 0)
			{
				/* global memory size is not enough */
				sizeForKernel += argsPtr[i].globalSize;
				if (devicePtr->usedSize + sizeForKernel > devicePtr->globalSize)
				{
					isMigrationNeeded = 1;
					break;
				}
			}
		}
	}

	/* kernel will be launched on this device */
	if (isMigrationNeeded == 0)
	{
		for (i = 0; i < argsNum; i++)
		{
			if (argsPtr[i].isGlobalMemory == 1)
			{
				/* add new memory to the device */
				memory = *((cl_mem *)argsPtr[i].arg_value);
				voclLibUpdateMemoryOnDevice(devicePtr, memory, argsPtr[i].globalSize);
			}
		}
	}

	printf("isMigrationNeeded = %d\n", isMigrationNeeded);

//	return 1;

	return isMigrationNeeded;
}

void voclLibUpdateGlobalMemUsage(cl_command_queue cmdQueue, kernel_args *argsPtr, int argsNum)
{
	int i;
	cl_mem memory;
	VOCL_LIB_DEVICE *devicePtr;
	devicePtr = voclLibGetDeviceIDFromCmdQueue(cmdQueue);

	for (i = 0; i < argsNum; i++)
	{
		if (argsPtr[i].isGlobalMemory == 1)
		{
			/* add new memory to the device */
			memory = *((cl_mem *)argsPtr[i].arg_value);
			voclLibUpdateMemoryOnDevice(devicePtr, memory, argsPtr[i].globalSize);
		}
	}

	return;
}

void voclTaskMigration(vocl_kernel kernel, vocl_command_queue command_queue)
{
    kernel_info *kernelPtr;
    vocl_device_id newDeviceID, oldDeviceID;
    cl_device_id clDeviceID;
    vocl_context context;
    vocl_program program;
    cl_mem oldMem, newMem;
    cl_command_queue oldCmdQueue, newCmdQueue;
    cl_kernel clKernel;
    int newRank, newIndex, oldRank, oldIndex;
    int proxySourceRank, proxyDestRank;
    int isFromLocal, isToLocal;
    MPI_Comm newComm, newCommData, oldComm, oldCommData;
    int i, err, memWrittenFlag, flag;
    size_t size;
    char *tmpBuf;

    /* finish previous tasks in the command queue */
    clFinish(command_queue);

    kernelPtr = getKernelPtr(kernel);
    oldDeviceID = voclGetCommandQueueDeviceID(command_queue);
    newDeviceID = voclSearchTargetGPU(kernelPtr->globalMemSize);
    printf("newDeviceID = %ld\n", newDeviceID);
    context = voclGetCommandQueueContext(command_queue);
    program = voclGetProgramFromKernel(kernel);

    clDeviceID =
        voclVOCLDeviceID2CLDeviceIDComm(newDeviceID, &newRank, &newIndex, &newComm,
                                        &newCommData);
    /* re-create context */
    voclUpdateVOCLContext(context, newRank, newIndex, newComm, newCommData, newDeviceID);

    /* re-create command queue */
    voclUpdateVOCLCommandQueue(command_queue, newRank, newIndex,
                               newComm, newCommData, context, newDeviceID);
    oldCmdQueue =
        voclVOCLCommandQueue2OldCLCommandQueueComm(command_queue, &oldRank, &oldIndex,
                                                   &oldComm, &oldCommData);
    newCmdQueue = voclVOCLCommandQueue2CLCommandQueue(command_queue);

    isFromLocal = voclIsOnLocalNode(oldIndex);
    isToLocal = voclIsOnLocalNode(newIndex);

    printf("isFromLocal = %d, isToLocal = %d\n", isFromLocal, isToLocal);

    /* go throught all argument of the kernel */
    memWrittenFlag = 0;
    for (i = 0; i < kernelPtr->args_num; i++) {
        if (kernelPtr->args_flag[i] == 1) {     /* it is global memory */
            size = voclGetVOCLMemorySize(kernelPtr->args_ptr[i].memory);
			flag = voclGetMemWrittenFlag(kernelPtr->args_ptr[i].memory);
		printf("i = %d, size = %ld, flag = %d\n", i, size, flag);

			/* write to gpu memory is completed */
            if (flag == 1) {
                if (isFromLocal == 0 || isToLocal == 0) {
                    /* send a message to the source proxy process for migration data transfer */
                    oldMem = voclVOCLMemory2CLMemory(kernelPtr->args_ptr[i].memory);
                    proxyDestRank =
                        voclMigIssueGPUMemoryRead(oldComm, oldRank, newComm, newCommData,
                                                  newRank, isFromLocal, isToLocal, oldCmdQueue,
                                                  oldMem, size);
                    /* update the memory to the new device */
                    voclUpdateVOCLMemory(kernelPtr->args_ptr[i].memory, newRank, newIndex,
                                         newComm, newCommData, context);
                    /* send a message to the dest proxy process for migration data transfer */
                    newMem = voclVOCLMemory2CLMemory(kernelPtr->args_ptr[i].memory);
                    proxySourceRank =
                        voclMigIssueGPUMemoryWrite(oldComm, oldCommData, oldRank, newComm,
                                                   newRank, isFromLocal, isToLocal,
                                                   newCmdQueue, newMem, size);
                }
                else {
                    /* send a message to the source proxy process for migration data transfer */
                    oldMem = voclVOCLMemory2CLMemory(kernelPtr->args_ptr[i].memory);
                    /* update the memory to the new device */
                    voclUpdateVOCLMemory(kernelPtr->args_ptr[i].memory, newRank, newIndex,
                                         newComm, newCommData, context);
                    /* send a message to the dest proxy process for migration data transfer */
                    newMem = voclVOCLMemory2CLMemory(kernelPtr->args_ptr[i].memory);
                    //free(tmp);
                    voclMigLocalToLocal(oldCmdQueue, oldMem, newCmdQueue, newMem, size);
                }
                memWrittenFlag = 1;
            }
			/*either not written at all, or written is incomplete */
            else {
                voclUpdateVOCLMemory(kernelPtr->args_ptr[i].memory, newRank, newIndex,
                                     newComm, newCommData, context);
				/* memory written is incomplete, just write to the new memory from */
				/* host memory, directory */
//				if (flag == 1) 
//				{
//					clEnqueueWriteBuffer(command_queue, kernelPtr->args_ptr[i].memory,
//										 CL_TRUE, 0, size, 
//										 voclGetMemHostPtr(kernelPtr->args_ptr[i].memory),
//										 0, NULL, NULL);
//				}
            }
            newMem = voclVOCLMemory2CLMemory(kernelPtr->args_ptr[i].memory);
            memcpy(kernelPtr->args_ptr[i].arg_value, (void *) &newMem, sizeof(cl_mem));
        }
    }

    /* if there are memory transfer from one proxy process  to another */
    /* tell the proxy process to complete the data transfer */
    if (memWrittenFlag == 1) {
        voclMigFinishDataTransfer(oldComm, oldRank, oldCmdQueue, newComm, newRank,
                                  newCmdQueue, proxySourceRank, proxyDestRank, isFromLocal,
                                  isToLocal);
    }

    /* data transfer is completed, release old command queue and gpu memory */
    clMigReleaseOldCommandQueue(command_queue);
    for (i = 0; i < kernelPtr->args_num; i++) {
        if (kernelPtr->args_flag[i] == 1) {     /* it is global memory */
            clMigReleaseOldMemObject(kernelPtr->args_ptr[i].memory);
        }
    }

    /* re-create program */
    voclUpdateVOCLProgram(program, newRank, newIndex, newComm, newCommData, context);

    /* build the program, only one target device is searched */
    /* the last two arguments are ignored currently */
    err = clBuildProgram(program, 1, &newDeviceID, voclGetProgramBuildOptions(program), 0, 0);

    voclUpdateVOCLKernel(kernel, newRank, newIndex, newComm, newCommData, program);

    if (isToLocal == 1) {
        clKernel = voclVOCLKernel2CLKernel(kernel);
        for (i = 0; i < kernelPtr->args_num; i++) {
            if (kernelPtr->args_ptr[i].arg_null_flag == 0) {
                err = dlCLSetKernelArg(clKernel, kernelPtr->args_ptr[i].arg_index,
                                       kernelPtr->args_ptr[i].arg_size,
                                       kernelPtr->args_ptr[i].arg_value);
            }
            else {
                err = dlCLSetKernelArg(clKernel, kernelPtr->args_ptr[i].arg_index,
                                       kernelPtr->args_ptr[i].arg_size, NULL);
            }
            if (err != CL_SUCCESS) {
                printf("argsetting %d, err = %d,\n", i, err);
            }
        }
    }

    return;
}

