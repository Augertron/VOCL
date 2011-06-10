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

/* whether migration is needed. no migration by default */
static int voclTaskMigrationCheckCondition = 0;

extern int voclMigIssueGPUMemoryWrite(MPI_Comm oldComm, MPI_Comm oldCommData, int oldRank, int oldIndex,
                                      MPI_Comm newComm, int newRank, int isFromLocal,
                                      int isToLocal, cl_command_queue command_queue,
                                      cl_mem mem, size_t size);

extern int voclMigIssueGPUMemoryRead(MPI_Comm oldComm, int oldRank,
                                     MPI_Comm newComm, MPI_Comm newCommData, int newRank, int newIndex,
                                     int isFromLocal, int isToLocal,
                                     cl_command_queue command_queue, cl_mem mem, size_t size);

extern void voclMigFinishDataTransfer(MPI_Comm oldComm, int oldRank, int oldIndex,
                                      cl_command_queue oldCmdQueue, MPI_Comm newComm,
                                      int newRank, int newIndex, cl_command_queue newCmdQueu,
                                      int proxySourceRank, int proxyDestRank, int isFromLocal,
                                      int isToLocal);

extern void voclMigrationOnSameRemoteNode(MPI_Comm comm, int rank, cl_command_queue oldCmdQueue,
        cl_mem oldMem, cl_command_queue newCmdQueue, cl_mem newMem, size_t size);
extern void voclMigrationOnSameRemoteNodeCmpd(MPI_Comm comm, int rank);

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
extern int voclCommandQueueGetMigrationStatus(vocl_command_queue cmdQueue);

extern vocl_program voclGetProgramFromKernel(vocl_kernel kernel);
extern void voclUpdateVOCLKernel(vocl_kernel voclKernel, int proxyRank, int proxyIndex,
                                 MPI_Comm proxyComm, MPI_Comm proxyCommData,
                                 vocl_program program);
extern cl_kernel voclVOCLKernel2CLKernel(vocl_kernel kernel);
extern int voclKernelGetMigrationStatus(vocl_kernel kernel);

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
extern int voclMemGetMigrationStatus(vocl_mem mem);
extern VOCL_LIB_DEVICE *voclLibGetDeviceIDFromCmdQueue(cl_command_queue cmdQueue);
extern void voclLibUpdateMemoryOnDevice(VOCL_LIB_DEVICE *devicePtr, cl_mem mem, size_t size);
extern cl_command_queue voclVOCLCommandQueue2OldCLCommandQueueComm(vocl_command_queue command_queue,
                                                            int *proxyRank, int *proxyIndex,
                                                            MPI_Comm * proxyComm,
                                                            MPI_Comm * proxyCommData);
extern int voclIsOnLocalNode(int index);
extern int voclGetMemWrittenFlag(vocl_mem memory);
extern void voclMigLocalToLocal(cl_command_queue oldCmdQueue, cl_mem oldMem,
                         cl_command_queue newCmdQueue, cl_mem newMem, size_t size);
extern char *voclGetProgramBuildOptions(vocl_program program);
extern int voclProgramGetMigrationStatus(vocl_program program);
extern cl_kernel voclVOCLKernel2CLKernelComm(vocl_kernel kernel, int *proxyRank,
                                      int *proxyIndex, MPI_Comm * proxyComm,
                                      MPI_Comm * proxyCommData);
extern kernel_info *getKernelPtr(cl_kernel kernel);
extern cl_int dlCLSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value);


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

    for (i = 2; i < totalDeviceNum; i++) {
        err =
            clGetDeviceInfo(deviceID[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size),
                            &mem_size, NULL);
        VOCL_MIG_CHECK_ERR(err, "migration, clGetDeviceInfo");
        if (mem_size >= size) {
            targetDeviceID = (vocl_device_id)deviceID[i];
            break;
        }
    }

    free(platformID);
    free(numDevices);
    free(deviceID);

    return targetDeviceID;

}

/* set whether migration is needed according to an environment variable */
void voclSetTaskMigrationCondition()
{
	char *migrationConditionsPtr, *tmpFlagPtr;
	char migrationConditions[][20] = {{"MEMORY_FULL"}};
	char *conditionList;
	size_t len;

	migrationConditionsPtr = getenv("VOCL_MIGRATION_CONDITION");
	if (migrationConditionsPtr == NULL)
	{
		voclTaskMigrationCheckCondition = 0;
	}
	else
	{
		len = strlen(migrationConditionsPtr) + 1;
		conditionList = (char *)malloc(sizeof(char) * len);
		strcpy(conditionList, migrationConditionsPtr);

		tmpFlagPtr = strtok(conditionList, ",");
		while (tmpFlagPtr != NULL)
		{
			if (strcmp(tmpFlagPtr, migrationConditions[0]) == 0)
			{
				voclTaskMigrationCheckCondition = 1;
			}
			else //more conditions are added later
			{
				
			}
			tmpFlagPtr = strtok(NULL, ",");
		}

		free(conditionList);
	}

	return;
}

int voclGetTaskMigrationCondition()
{
	return voclTaskMigrationCheckCondition;
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
	int cmdQueueMigStatus, memMigStatus, programMigStatus, kernelMigStatus;
    MPI_Comm newComm, newCommData, oldComm, oldCommData;
    int i, err, memWrittenFlag, flag;
    size_t size;

	/* if the command queue is already migrated */
	kernelPtr = getKernelPtr(kernel);
	cmdQueueMigStatus = voclCommandQueueGetMigrationStatus(command_queue);
	context = voclGetCommandQueueContext(command_queue);
	if (cmdQueueMigStatus == 0)
	{
		/* finish previous tasks in the command queue */
		clFinish(command_queue);

		oldDeviceID = voclGetCommandQueueDeviceID(command_queue);
		newDeviceID = voclSearchTargetGPU(kernelPtr->globalMemSize);

		clDeviceID =
			voclVOCLDeviceID2CLDeviceIDComm(newDeviceID, &newRank, &newIndex, &newComm,
											&newCommData);
		/* re-create context */
		voclUpdateVOCLContext(context, newRank, newIndex, newComm, newCommData, newDeviceID);
		
		/* re-create command queue */
		voclUpdateVOCLCommandQueue(command_queue, newRank, newIndex,
								   newComm, newCommData, context, newDeviceID);
	}

	oldCmdQueue = voclVOCLCommandQueue2OldCLCommandQueueComm(command_queue, &oldRank, 
					&oldIndex, &oldComm, &oldCommData);
	newCmdQueue = voclVOCLCommandQueue2CLCommandQueueComm(command_queue, &newRank,
					&newIndex, &newComm, &newCommData);

    isFromLocal = voclIsOnLocalNode(oldIndex);
    isToLocal = voclIsOnLocalNode(newIndex);
	newDeviceID = voclGetCommandQueueDeviceID(command_queue);
	printf("newDeviceID = %ld\n", newDeviceID);

    printf("kernelLaunch, isFromLocal = %d, isToLocal = %d\n", isFromLocal, isToLocal);

	cmdQueueMigStatus = voclCommandQueueGetMigrationStatus(command_queue);
    /* go throught all argument of the kernel */
    memWrittenFlag = 0;
    for (i = 0; i < kernelPtr->args_num; i++) {
        if (kernelPtr->args_flag[i] == 1) {     /* it is global memory */
            size = voclGetVOCLMemorySize(kernelPtr->args_ptr[i].memory);
			flag = voclGetMemWrittenFlag(kernelPtr->args_ptr[i].memory);
			memMigStatus = voclMemGetMigrationStatus(kernelPtr->args_ptr[i].memory);
			printf("i = %d, size = %ld, flag = %d, memMigStatus = %d, cmdMigStatus = %d\n", 
					i, size, flag, memMigStatus, cmdQueueMigStatus);
			
			/* write to gpu memory is completed */
			if (memMigStatus < cmdQueueMigStatus)
			{
				if (flag == 1) {
						/* send a message to the source proxy process for migration data transfer */
					oldMem = voclVOCLMemory2CLMemory(kernelPtr->args_ptr[i].memory);
					/* update the memory to the new device */
					voclUpdateVOCLMemory(kernelPtr->args_ptr[i].memory, newRank, newIndex,
										 newComm, newCommData, context);
					/* send a message to the dest proxy process for migration data transfer */
					newMem = voclVOCLMemory2CLMemory(kernelPtr->args_ptr[i].memory);

					if (isFromLocal == 0 || isToLocal == 0) {
						/* if migration is on the same remote node */
						if (isFromLocal == 0 && isToLocal == 0 && oldIndex == newIndex)
						{
							voclMigrationOnSameRemoteNode(oldComm, oldRank, oldCmdQueue, oldMem,
								newCmdQueue, newMem, size);
						}

						proxyDestRank =
							voclMigIssueGPUMemoryRead(oldComm, oldRank, newComm, newCommData,
													  newRank, newIndex, isFromLocal, isToLocal, oldCmdQueue,
													  oldMem, size);
						proxySourceRank =
							voclMigIssueGPUMemoryWrite(oldComm, oldCommData, oldRank, oldIndex, newComm,
													   newRank, isFromLocal, isToLocal,
													   newCmdQueue, newMem, size);
					}
					else {
						voclMigLocalToLocal(oldCmdQueue, oldMem, newCmdQueue, newMem, size);
					}
					memWrittenFlag = 1;
				}
				/*either not written at all, or written is incomplete */
				else {
					voclUpdateVOCLMemory(kernelPtr->args_ptr[i].memory, newRank, newIndex,
										 newComm, newCommData, context);
				}
				newMem = voclVOCLMemory2CLMemory(kernelPtr->args_ptr[i].memory);
				memcpy(kernelPtr->args_ptr[i].arg_value, (void *) &newMem, sizeof(cl_mem));
			}
        }
    }

    /* if there are memory transfer from one proxy process  to another */
    /* tell the proxy process to complete the data transfer */
    if (memWrittenFlag == 1) {
        voclMigFinishDataTransfer(oldComm, oldRank, oldIndex, oldCmdQueue, newComm, newRank,
                                  newIndex, newCmdQueue, proxySourceRank, proxyDestRank, isFromLocal,
                                  isToLocal);
    }

    /* data transfer is completed, release old command queue and gpu memory */
    clMigReleaseOldCommandQueue(command_queue);
    for (i = 0; i < kernelPtr->args_num; i++) {
        if (kernelPtr->args_flag[i] == 1) {     /* it is global memory */
            clMigReleaseOldMemObject(kernelPtr->args_ptr[i].memory);
        }
    }

	program = voclGetProgramFromKernel(kernel);
    /* cmdQueue is migrated, but kernel is not yet */
	programMigStatus = voclProgramGetMigrationStatus((vocl_program)program);
	if (programMigStatus < cmdQueueMigStatus)
	{
		voclUpdateVOCLProgram(program, newRank, newIndex, newComm, newCommData, context);
		/* build the program, only one target device is searched */
		/* the last two arguments are ignored currently */
		err = clBuildProgram(program, 1, &newDeviceID, voclGetProgramBuildOptions(program), 0, 0);
	}

	kernelMigStatus = voclKernelGetMigrationStatus((vocl_kernel)kernel);
	if (kernelMigStatus < cmdQueueMigStatus)
	{
		voclUpdateVOCLKernel(kernel, newRank, newIndex, newComm, newCommData, program);
	}

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

