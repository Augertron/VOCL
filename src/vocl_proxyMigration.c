#include <stdio.h>
#include <string.h>
#include "vocl_proxy.h"
#include "vocl_proxyWinProc.h"
#include "vocl_proxyBufferProc.h"
#include "vocl_proxyStructures.h"

extern vocl_virtual_gpu *voclProxyGetVirtualGPUPtr(int appIndex, cl_device_id deviceID);
extern void voclProxyAddVirtualGPU(int appIndex, int proxyRank, cl_device_id deviceID);
extern char voclProxyUpdateVGPUMigStatus(int appIndex, cl_device_id deviceID);
extern void voclProxyAddContextToVGPU(int appIndex, cl_device_id deviceID, vocl_proxy_context *context);
extern void voclProxyAddCommandQueueToVGPU(int appIndex, cl_device_id deviceID, vocl_proxy_command_queue *command_queue);
extern void voclProxyReleaseVirtualGPU(int appIndex, cl_device_id deviceID);
extern void vocl_proxyGetKernelNumsOnGPUs(struct strKernelNumOnDevice *gpuKernelNum);
extern void voclProxyGetMessageSizeForVGPU(vocl_virtual_gpu *vgpuPtr, vocl_vgpu_msg *msgPtr);
extern void voclProxyPackMessageForVGPU(vocl_virtual_gpu *vgpuPtr, char *bufPtr);

extern void voclProxyAddContext(cl_context context, cl_uint deviceNum, cl_device_id *deviceIDs);
extern vocl_proxy_context *voclProxyGetContextPtr(cl_context context);
extern void voclProxyStoreOldContextValue(cl_context context, cl_context oldContext);
extern void voclProxyAddProgramToContext(cl_context context, vocl_proxy_program *program);
extern void voclProxyAddCommandQueueToContext(cl_context context, vocl_proxy_command_queue *command_queue);
extern void voclProxyAddMemToContext(cl_context context, vocl_proxy_mem *mem);
extern void voclProxyReleaseContext(cl_context context);
extern char voclProxyUpdateContextMigStatus(cl_context context);
extern void voclProxySetContextMigStatus(cl_context context, char migStatus);

extern void voclProxyAddProgram(cl_program program, char *sourceString, size_t sourceSize, int stringNum, size_t *stringSizeArray, cl_context context);
extern vocl_proxy_program *voclProxyGetProgramPtr(cl_program program);
extern void voclProxySetProgramBuildOptions(cl_program program, cl_uint deviceNum, cl_device_id *device_list, char *buildOptions);
extern void voclProxySetProgramMigStatus(cl_program program, char migStatus);
extern void voclProxyStoreOldProgramValue(cl_program program, cl_program oldProgram);
extern void voclProxyAddKernelToProgram(cl_program program, vocl_proxy_kernel *kernel);
extern void voclProxyReleaseProgram(cl_program program);

extern void voclProxyAddKernel(cl_kernel kernel, char *kernelName, cl_program program);
extern vocl_proxy_kernel *voclProxyGetKernelPtr(cl_kernel kernel);
extern void voclProxySetKernelMigStatus(cl_kernel kernel, char migStatus);
extern void voclProxyStoreOldKernelValue(cl_kernel kernel, cl_kernel oldKernel);
extern void voclProxySetKernelArgFlag(cl_kernel kernel, int argNum, char *argFlag);
extern void voclProxyStoreKernelArgs(cl_kernel kernel, int argNum, kernel_args *args);
extern void voclProxySetKernelArgs(cl_kernel kernel);
extern void voclProxyReleaseKernel(cl_kernel kernel);

extern void voclProxyAddCmdQueue(cl_command_queue command_queue, cl_command_queue_properties properties, cl_context context, cl_device_id deviceID);
extern vocl_proxy_command_queue *voclProxyGetCmdQueuePtr(cl_command_queue command_queue);
extern void voclProxySetCommandQueueMigStatus(cl_command_queue command_queue, char migStatus);
extern void voclProxyStoreOldCommandQueueValue(cl_command_queue command_queue, cl_command_queue oldCommand_queue);
extern void voclProxyReleaseCommandQueue(cl_command_queue command_queue);

extern void voclProxyAddMem(cl_mem mem, cl_mem_flags flags, size_t size, cl_context context);
extern vocl_proxy_mem *voclProxyGetMemPtr(cl_mem mem);
extern void voclProxySetMemMigStatus(cl_mem mem, char migStatus);
extern void voclProxyStoreOldMemValue(cl_mem mem, cl_mem oldMem);
extern void voclProxySetMemWritten(cl_mem mem, int isWritten);
extern void voclProxyReleaseMem(cl_mem mem);

extern MPI_Win *voclProxyGetWinPtr(int index);

extern int voclMigGetNextReadBufferIndex(int rank);
extern struct strMigReadBufferInfo *voclMigGetReadBufferPtr(int rank, int index);
extern cl_event *voclMigGetReadEventPtr(int rank, int index);
extern void voclMigSetReadBufferFlag(int rank, int index, int flag);
extern int voclMigFinishDataRead(int rank);

extern int voclMigGetNextWriteBufferIndex(int rank);
extern struct strMigWriteBufferInfo *voclMigGetWriteBufferPtr(int rank, int index);
extern MPI_Request *voclMigGetWriteRequestPtr(int rank, int index);
extern void voclMigSetWriteBufferFlag(int rank, int index, int flag);
extern int voclMigFinishDataWrite(int rank);

extern int voclMigRWGetNextBufferIndex(int rank);
extern struct strMigRWBufferSameNode *voclMigRWGetBufferInfoPtr(int rank, int index);
extern void voclMigSetRWBufferFlag(int rank, int index, int flag);
extern int voclMigFinishDataRWOnSameNode(int rank);
extern void voclProxyUpdateMigStatus(int appIndex, int destProxyIndex);
extern void voclProxySetMigStatus(int appIndex, char migStatus);

extern void voclProxyMigrationMutexLock(int appIndex);
extern void voclProxyMigrationMutexUnlock(int appIndex);

void voclProxyMigSendDeviceMemoryData(vocl_virtual_gpu *vgpuPtr, int destRankNo, MPI_Comm migComm, MPI_Comm migCommData);
void voclProxyMigSendRecvDeviceMemoryData(vocl_virtual_gpu *sourceVGPUPtr, vocl_virtual_gpu *destVGPUPtr);

extern MPI_Comm *appComm, *appCommData;

void voclProxyMigCreateVirtualGPU(int appIndex, int proxyRank, cl_device_id deviceID, cl_uint contextNum, char *bufPtr)
{
	int i, j, k, stringIndex;
	size_t offset;
	size_t startLoc;
	char **strings, *sourceString;
	char *argFlag;
	kernel_args *args;
	int argNum;
	char migStatus;
	cl_int retCode;
	cl_context context;
	cl_command_queue cmdQueue;
	cl_program program;
	cl_kernel kernel;
	cl_mem mem;
	vocl_proxy_context *contextPtr, *ctxPtr;
	vocl_proxy_command_queue *cmdQueuePtr, *cqPtr;
	vocl_proxy_program *programPtr, *pgPtr;
	vocl_proxy_kernel *kernelPtr, *knPtr;
	vocl_proxy_mem *memPtr, *mmPtr;
	
	/* add the new virtual to the target proxy process */
	voclProxyAddVirtualGPU(appIndex, proxyRank, deviceID);
	migStatus = voclProxyUpdateVGPUMigStatus(appIndex, deviceID);
	voclProxySetMigStatus(appIndex, migStatus);


	/* unpack the received message and create corresponding */
	/* resources on the new virtual GPU */
	offset = 0;
	startLoc = 0;

	for (i = 0; i < contextNum; i++)
	{
		/* obtain the context pointer to unpack the migration message */
		contextPtr = (vocl_proxy_context *)(bufPtr+offset);
		offset += sizeof(vocl_proxy_context);
	
		context = clCreateContext(NULL, 1, &deviceID, NULL, NULL, &retCode);

		voclProxyAddContext(context, 1, &deviceID);
		/* set the new migration status to that of the virtual GPU */
		voclProxySetContextMigStatus(context, migStatus);
		/* store the old context corresponding to new context */
		voclProxyStoreOldContextValue(context, contextPtr->context);

		/* get vocl_proxy_context pointer */
		ctxPtr = voclProxyGetContextPtr(context);
		voclProxyAddContextToVGPU(appIndex, deviceID, ctxPtr);
	
		/* unpack the program */
		for (j = 0; j < contextPtr->programNo; j++)
		{
			/* decode the program structure */
			programPtr = (vocl_proxy_program *)(bufPtr + offset);
			offset += sizeof(vocl_proxy_program);
			
			/* decode the string string */
			programPtr->sourceString = (char *)(bufPtr+offset);
			offset += programPtr->sourceSize;

			/* decode the string size array */
			programPtr->stringSizeArray = (size_t *)(bufPtr+offset);
			offset += programPtr->stringNum * sizeof(size_t);

			/* decode the build option */
			if (programPtr->buildOptionLen > 0)
			{
				programPtr->buildOptions = (char *)(bufPtr+offset);
				offset += programPtr->buildOptionLen;
			}

			/* decode the devices */
			if (programPtr->deviceNum > 0)
			{
				programPtr->device_list = (cl_device_id *)(bufPtr+offset);
				offset += sizeof(cl_device_id) * programPtr->deviceNum;
			}

			/* divide the source string into different strings */
			strings = (char **)malloc(programPtr->stringNum * sizeof(char *));
			startLoc = 0;
			for (stringIndex = 0; stringIndex < programPtr->stringNum; stringIndex++)
			{
				strings[stringIndex] = (char *)malloc(programPtr->stringSizeArray[stringIndex] + 1);
				memcpy(strings[stringIndex], &programPtr->sourceString[startLoc], programPtr->stringSizeArray[stringIndex]);
				strings[stringIndex][programPtr->stringSizeArray[stringIndex]] = '\0';
				startLoc += programPtr->stringSizeArray[stringIndex];
			}

			/* create opencl program */
			program = clCreateProgramWithSource(context, 
												programPtr->stringNum,
												(const char **)strings,
												programPtr->stringSizeArray,
												&retCode);
			/* build the program */
			retCode = clBuildProgram(program, 1, &deviceID, programPtr->buildOptions, NULL, NULL);

			/* add create program */
			voclProxyAddProgram(program, 
								programPtr->sourceString, 
								programPtr->sourceSize, 
								programPtr->stringNum, 
								programPtr->stringSizeArray,
								context);
			/* set program migration status */
			voclProxySetProgramMigStatus(program, migStatus);

			/* store the program value before migration */
			voclProxyStoreOldProgramValue(program, programPtr->program);

			pgPtr = voclProxyGetProgramPtr(program);
			voclProxyAddProgramToContext(context, pgPtr);

			/* store program build options */
			voclProxySetProgramBuildOptions(program, 1, &deviceID, programPtr->buildOptions);

			/* release string buffer */
			for (stringIndex = 0; stringIndex < programPtr->stringNum; stringIndex++)
			{
				free(strings[stringIndex]);
			}
			free(strings);

			for (k = 0; k < programPtr->kernelNo; k++)
			{
				kernelPtr = (vocl_proxy_kernel *)(bufPtr + offset);
				offset += sizeof(vocl_proxy_kernel);
				kernelPtr->kernelName = (char *)(bufPtr + offset);
				offset += kernelPtr->nameLen;
				kernel = clCreateKernel(program, kernelPtr->kernelName, &retCode);

				argNum = kernelPtr->argNum;
				/*unpack the argument flag */
				argFlag = (char *)(bufPtr + offset);
				offset += argNum * sizeof(char);
				
				/* unpack the arguments */
				args = (kernel_args *)(bufPtr + offset);
				offset += argNum * sizeof(kernel_args);

				/* store the kernel */
				voclProxyAddKernel(kernel, kernelPtr->kernelName, program);

				/* set the kernel migration status */
				voclProxySetKernelMigStatus(kernel, migStatus);

				/* store the kernel value before migration */
				voclProxyStoreOldKernelValue(kernel, kernelPtr->kernel);

				knPtr = voclProxyGetKernelPtr(kernel);
				voclProxyAddKernelToProgram(program, knPtr);

				/* set the argument flag and arguments */
				voclProxySetKernelArgFlag(kernel, argNum, argFlag);
				voclProxyStoreKernelArgs(kernel, argNum, args);

				/* set the arguments for the kernel in the new virtual GPU */
				voclProxySetKernelArgs(kernel);
			}
		}

		for (j = 0; j < contextPtr->cmdQueueNo; j++)
		{
			/* decode command queue */
			cmdQueuePtr = (vocl_proxy_command_queue *)(bufPtr + offset);
			offset += sizeof(vocl_proxy_command_queue);
			cmdQueue = clCreateCommandQueue(context, deviceID, cmdQueuePtr->properties, &retCode);

			/* store the command queue */
			voclProxyAddCmdQueue(cmdQueue, cmdQueuePtr->properties, context, deviceID);
			
			/* set the migration status of command queue */
			voclProxySetCommandQueueMigStatus(cmdQueue, migStatus);

			/* store the command queue value before migration */
			voclProxyStoreOldCommandQueueValue(cmdQueue, cmdQueuePtr->command_queue);

			cqPtr = voclProxyGetCmdQueuePtr(cmdQueue);
			voclProxyAddCommandQueueToContext(context, cqPtr);
			voclProxyAddCommandQueueToVGPU(appIndex, deviceID, cqPtr);
		}

		for (j = 0; j < contextPtr->memNo; j++)
		{
			/* decoce the device memory */
			memPtr = (vocl_proxy_mem *)(bufPtr + offset);
			offset += sizeof(vocl_proxy_mem);
			mem = clCreateBuffer(context, memPtr->flags, memPtr->size,
								 NULL, &retCode);
			/* store the memory */
			voclProxyAddMem(mem, memPtr->flags, memPtr->size, context);
			voclProxySetMemMigStatus(mem, migStatus);

			/* store the memory value before migration */
			voclProxyStoreOldMemValue(mem, memPtr->mem);

			voclProxySetMemWritten(mem, memPtr->isWritten);
			mmPtr = voclProxyGetMemPtr(mem);
			voclProxyAddMemToContext(context, mmPtr);
		}
	}

	return;
}

void voclProxyMigReleaseVirtualGPU(vocl_virtual_gpu *vgpuPtr)
{
	int i, j, k;
	vocl_proxy_context **contextPtr;
	vocl_proxy_program **programPtr;
	vocl_proxy_kernel **kernelPtr;
	vocl_proxy_command_queue **cmdQueuePtr;
	vocl_proxy_mem **memPtr;

	cl_context context;
	cl_program program;
	cl_kernel kernel;
	cl_command_queue cmdQueue;
	cl_mem mem;

	
	contextPtr = vgpuPtr->contextPtr;
	for (i = 0; i < vgpuPtr->contextNo; i++)
	{
		programPtr = contextPtr[i]->programPtr;
		for (j = 0; j < contextPtr[i]->programNo; j++)
		{
			kernelPtr = programPtr[j]->kernelPtr;
			for (k = 0; k < programPtr[j]->kernelNo; k++)
			{
				/* release kernel */
				kernel = kernelPtr[i]->kernel;
				voclProxyReleaseKernel(kernel);
				clReleaseKernel(kernel);
			}

			/* release program */
			program = programPtr[j]->program;
			voclProxyReleaseProgram(program);
			clReleaseProgram(program);
		}

		/* release command queue */
		cmdQueuePtr = contextPtr[i]->cmdQueuePtr;
		for (j = 0; j < contextPtr[i]->cmdQueueNo; j++)
		{
			cmdQueue = cmdQueuePtr[j]->command_queue;
			voclProxyReleaseCommandQueue(cmdQueue);
			clReleaseCommandQueue(cmdQueue);
		}

		/* release memory */
		memPtr = contextPtr[i]->memPtr;
		for (j = 0; j < contextPtr[i]->memNo; j++)
		{
			mem = memPtr[j]->mem;
			voclProxyReleaseMem(mem);
			clReleaseMemObject(mem);
		}

		context = contextPtr[i]->context;
		voclProxyReleaseContext(context);
		clReleaseContext(context);
	}

	/* release the virtual gpu itself */
	voclProxyReleaseVirtualGPU(vgpuPtr->appIndex, vgpuPtr->deviceID);

	return;
}

void voclProxyMigReleaseVirtualGPUOverload(int appIndex, cl_device_id deviceID)
{
	vocl_virtual_gpu *vgpuPtr;
	vgpuPtr = voclProxyGetVirtualGPUPtr(appIndex, deviceID);

	voclProxyMigReleaseVirtualGPU(vgpuPtr);
}

/* go through each proxy process and obtain the */
/* number of kernels in waiting */
struct strKernelNumOnDevice *voclProxyMigQueryLoadOnGPUs(int appIndex, int *proxyNum)
{
	int myRank, proxyRank, j;
	MPI_Request *request;
	MPI_Status *status;
	int requestNo = 0;
	struct strKernelNumOnDevice *loadPtr;
	MPI_Comm comm, commData;
	MPI_Win *winPtr;
	vocl_proxy_wins *winBufPtr;

	comm = appComm[0]; /* for communicator is for migration */
	commData = appCommData[0];

	MPI_Comm_rank(comm, &myRank);

	/* get the number of kernels in waiting in each proxy process */
	winPtr = voclProxyGetWinPtr(appIndex);
	winBufPtr = (vocl_proxy_wins *)malloc(sizeof(vocl_proxy_wins));

	MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, *winPtr);
	MPI_Get(winBufPtr, sizeof(vocl_proxy_wins), MPI_BYTE, 0, 0,
			sizeof(vocl_proxy_wins), MPI_BYTE, *winPtr);
	MPI_Win_unlock(0, *winPtr);
	loadPtr = (struct strKernelNumOnDevice *)malloc(sizeof(struct strKernelNumOnDevice) * winBufPtr->proxyNum);
	request = (MPI_Request *)malloc(sizeof(MPI_Request) * winBufPtr->proxyNum * 2);
	status  = (MPI_Status  *)malloc(sizeof(MPI_Status) * winBufPtr->proxyNum * 2);

	for (j = 0; j < winBufPtr->proxyNum; j++)
	{	
		if (winBufPtr->wins[j].proxyRank != myRank)
		{
			MPI_Isend(&loadPtr[j], sizeof(struct strKernelNumOnDevice), MPI_BYTE, 
					  winBufPtr->wins[j].proxyRank, LB_GET_KERNEL_NUM, comm, 
					  request+(requestNo++));
			MPI_Irecv(&loadPtr[j], sizeof(struct strKernelNumOnDevice), MPI_BYTE,
					  winBufPtr->wins[j].proxyRank, LB_GET_KERNEL_NUM, commData, 
					  request+(requestNo++));
		}
		else
		{
			vocl_proxyGetKernelNumsOnGPUs(&loadPtr[j]);
		}
	}
	if (requestNo > 0)
	{
		MPI_Waitall(requestNo, request, status);
	}

	for (j = 0; j < winBufPtr->proxyNum; j++)
	{
		loadPtr[j].appIndex = winBufPtr->wins[j].appIndex;
	}

	*proxyNum = winBufPtr->proxyNum;

	free(winBufPtr);
	free(request);
	free(status);
	return loadPtr;
}

/* find the physical gpu that has the least load. If all gpus have */
/* the same amount of load, select the first device */
cl_device_id voclProxyMigFindTargetGPU(struct strKernelNumOnDevice *gpuKernelNum, int proxyNum, int *rankNo, int *appIndex)
/* return the rank number and the device id of the corresponding gpu */
{
	int i, j;
	int minKernelNumInWaiting = 999999;
	int rankNoOfMinKernelNum = -1;
	int indexOfMinKernelNum = -1;

	cl_device_id deviceID;
	for (i = 1; i < proxyNum; i++)
	{
		for (j = 0; j < gpuKernelNum[i].deviceNum; j++)
		{
			if (minKernelNumInWaiting > gpuKernelNum[i].kernelNums[j])
			{
				minKernelNumInWaiting = gpuKernelNum[i].kernelNums[j];
				rankNoOfMinKernelNum = gpuKernelNum[i].rankNo;
				indexOfMinKernelNum = gpuKernelNum[i].appIndex;
				deviceID = gpuKernelNum[i].deviceIDs[j];
			}
		}
	}

	*rankNo = rankNoOfMinKernelNum;
	*appIndex = indexOfMinKernelNum;
	return deviceID;
}

cl_int voclProxyMigrationOneVGPU(vocl_virtual_gpu *vgpuPtr)
{
	struct strKernelNumOnDevice *gpuKernelNum;
	struct strVGPUMigration vgpuMigrationMsg;
	vocl_vgpu_msg vgpuMsg;
	vocl_virtual_gpu *newVGPUPtr;
	size_t msgSize;
	char *msgBuf;
	int proxyNum, destRankNo, myRankNo, destAppIndex;
	MPI_Comm comm, commData;
	MPI_Request request[3];
	MPI_Status status[3];
	int requestNo = 0;
	cl_device_id deviceID;

	/* acquire the locker */
//	voclProxyMigrationMutexLock(vgpuPtr->appIndex);

	comm = appComm[0];
	commData = appCommData[0];
	gpuKernelNum = voclProxyMigQueryLoadOnGPUs(vgpuPtr->appIndex, &proxyNum);
	deviceID = voclProxyMigFindTargetGPU(gpuKernelNum, proxyNum, &destRankNo, &destAppIndex);
	free(gpuKernelNum);

	/* send the migration message to target proxy process */
	voclProxyGetMessageSizeForVGPU(vgpuPtr, &vgpuMsg);
	/* pack the message for the virtual GPU */
	msgBuf = (char *)malloc(vgpuMsg.size);
	voclProxyPackMessageForVGPU(vgpuPtr, msgBuf);

	/* in different proxy process */
	if (vgpuPtr->proxyRank != destRankNo)
	{
		vgpuMigrationMsg.migMsgSize = vgpuMsg.size;
		vgpuMigrationMsg.deviceID = deviceID;
		vgpuMigrationMsg.contextNum = vgpuMsg.contextNum;
		vgpuMigrationMsg.appIndex = destAppIndex;

		MPI_Isend(&vgpuMigrationMsg, sizeof(struct strVGPUMigration), MPI_BYTE, 
				  destRankNo, VOCL_MIGRATION, comm, request+(requestNo++));
		MPI_Isend(msgBuf, vgpuMsg.size, MPI_BYTE, destRankNo, VOCL_MIGRATION,
				  commData, request+(requestNo++));
		MPI_Irecv(&vgpuMigrationMsg, sizeof(struct strVGPUMigration), MPI_BYTE, 
				  destRankNo, VOCL_MIGRATION, commData, request+(requestNo++));
		MPI_Waitall(requestNo, request, status);
		/* finish data transfer for migration */
		voclProxyMigSendDeviceMemoryData(vgpuPtr, destRankNo, comm, commData);

		/* update the mapping information in the library size */
		voclProxyUpdateMigStatus(vgpuPtr->appIndex, destRankNo);
	}
	else
	{
		/* in the same proxy process, but in different devices */
		if (vgpuPtr->deviceID != deviceID)
		{
			voclProxyMigCreateVirtualGPU(vgpuPtr->appIndex, destRankNo, deviceID, 
				vgpuMsg.contextNum, msgBuf);
			newVGPUPtr = voclProxyGetVirtualGPUPtr(vgpuPtr->appIndex, deviceID);
			voclProxyMigSendRecvDeviceMemoryData(vgpuPtr, newVGPUPtr);
		}
		else {   } /* same device, no migration is needed */
	}

	/* After data transfer and kernel launch are migrated, this function */
	/* will be called to release the virtual GPU on the original proxy */
//	voclProxyMigReleaseVirtualGPU(vgpuPtr);

	/* acquire the locker */
//	voclProxyMigrationMutexUnlock(vgpuPtr->appIndex);

	return vgpuMigrationMsg.retCode;
}

cl_int voclProxyMigrationOneVGPUOverload(int appIndex, cl_device_id deviceID)
{
	vocl_virtual_gpu *vgpuPtr;
	vgpuPtr = voclProxyGetVirtualGPUPtr(appIndex, deviceID);

	return voclProxyMigrationOneVGPU(vgpuPtr);
}

void voclProxyMigSendDeviceMemoryData(vocl_virtual_gpu *vgpuPtr, int destRankNo, MPI_Comm migComm, MPI_Comm migCommData)
{
	cl_int err;
	int i, j, k, appIndex;
	int bufferNum, bufferIndex;
	size_t bufferSize, remainingSize;
	vocl_proxy_context **contextPtr;
	vocl_proxy_mem **memPtr;
    struct strMigReadBufferInfo *migReadBufferInfoPtr;
	
	appIndex = vgpuPtr->appIndex;
	contextPtr = vgpuPtr->contextPtr;
	for (i = 0; i < vgpuPtr->contextNo; i++)
	{
		memPtr = contextPtr[i]->memPtr;
		for (j = 0; j < contextPtr[i]->memNo; j++)
		{
			/* memory is written */
			if (memPtr[j]->isWritten == 1)
			{
				bufferSize = VOCL_MIG_BUF_SIZE;
				bufferNum = (memPtr[j]->size - 1) / VOCL_MIG_BUF_SIZE;
				remainingSize = memPtr[j]->size - bufferNum * VOCL_MIG_BUF_SIZE;
				for (k = 0; k <= bufferNum; k++)
				{
					if (k == bufferNum)
					{
						bufferSize = remainingSize;
					}
					bufferIndex = voclMigGetNextReadBufferIndex(appIndex);
					migReadBufferInfoPtr = voclMigGetReadBufferPtr(appIndex, bufferIndex);
					err = clEnqueueReadBuffer(memPtr[k]->cmdQueue,
											  memPtr[k]->mem,
											  CL_FALSE,
											  k * VOCL_MIG_BUF_SIZE,
											  bufferSize,
											  migReadBufferInfoPtr->ptr,
											  0, NULL, voclMigGetReadEventPtr(appIndex,
																			  bufferIndex));
					migReadBufferInfoPtr->size = bufferSize;
					migReadBufferInfoPtr->offset = i * VOCL_MIG_BUF_SIZE;
					migReadBufferInfoPtr->comm = migComm;
                    migReadBufferInfoPtr->commData = migCommData;
                    migReadBufferInfoPtr->dest = destRankNo;
					migReadBufferInfoPtr->tag = VOCL_PROXY_MIG_TAG;
					voclMigSetReadBufferFlag(appIndex, bufferIndex, MIG_READ_RDGPU);
				}
			}
		}
	}

	voclMigFinishDataRead(appIndex);

	/* update the virtual GPU mapping */

	return;
}

void voclProxyMigRecvDeviceMemoryData(int appIndex, cl_device_id deviceID, int sourceRankNo)
{
	int err;
	int i, j, k;
	int bufferNum, bufferIndex;
	size_t bufferSize, remainingSize;
	vocl_virtual_gpu *vgpuPtr;
	vocl_proxy_context **contextPtr;
	vocl_proxy_mem **memPtr;
	MPI_Comm migComm, migCommData;
	cl_command_queue cmdQueue;
    struct strMigWriteBufferInfo *migWriteBufferInfoPtr;

	vgpuPtr = voclProxyGetVirtualGPUPtr(appIndex, deviceID);

	migComm = appComm[0];
	migCommData = appCommData[0];

	appIndex = vgpuPtr->appIndex;
	contextPtr = vgpuPtr->contextPtr;
	for (i = 0; i < vgpuPtr->contextNo; i++)
	{
		memPtr = contextPtr[i]->memPtr;
		cmdQueue = contextPtr[i]->cmdQueuePtr[0]->command_queue;
		for (j = 0; j < contextPtr[i]->memNo; j++)
		{
			/* memory is written */
			if (memPtr[j]->isWritten == 1)
			{
				bufferSize = VOCL_MIG_BUF_SIZE;
				bufferNum = (memPtr[j]->size - 1) / VOCL_MIG_BUF_SIZE;
				remainingSize = memPtr[j]->size - bufferNum * VOCL_MIG_BUF_SIZE;
				for (k = 0; k <= bufferNum; k++)
				{
					if (k == bufferNum)
					{
						bufferSize = remainingSize;
					}
					bufferIndex = voclMigGetNextWriteBufferIndex(appIndex);
					migWriteBufferInfoPtr = voclMigGetWriteBufferPtr(appIndex, bufferIndex);
					
					err = MPI_Irecv(migWriteBufferInfoPtr->ptr, bufferSize, MPI_BYTE,
									sourceRankNo, VOCL_PROXY_MIG_TAG,
									migCommData, voclMigGetWriteRequestPtr(appIndex,
									bufferIndex));
					migWriteBufferInfoPtr->cmdQueue = cmdQueue;
					migWriteBufferInfoPtr->memory = memPtr[j]->mem;
					migWriteBufferInfoPtr->source = sourceRankNo;
					migWriteBufferInfoPtr->comm = migComm;
					migWriteBufferInfoPtr->size = bufferSize;
					migWriteBufferInfoPtr->offset = i * VOCL_MIG_BUF_SIZE;
					voclMigSetWriteBufferFlag(appIndex, bufferIndex, MIG_WRT_MPIRECV);
				}
			}
		}
	}

	voclMigFinishDataWrite(appIndex);

	return;
}

void voclProxyMigSendRecvDeviceMemoryData(vocl_virtual_gpu *sourceVGPUPtr, vocl_virtual_gpu *destVGPUPtr)
{
	int err;
	int i, j, k, appIndex;
	int bufferNum, bufferIndex;
	size_t bufferSize, remainingSize;
	vocl_proxy_context **oldContextPtr, **newContextPtr;
	vocl_proxy_mem **oldMemPtr, **newMemPtr;
	cl_command_queue oldCmdQueue, newCmdQueue;
	struct strMigRWBufferSameNode *migRWBufferInfoPtr;

	appIndex = sourceVGPUPtr->appIndex;
	oldContextPtr = sourceVGPUPtr->contextPtr;
	newContextPtr = destVGPUPtr->contextPtr;
	for (i = 0; i < destVGPUPtr->contextNo; i++)
	{
		oldMemPtr = oldContextPtr[i]->memPtr;
		newMemPtr = newContextPtr[i]->memPtr;
		for (j = 0; j < oldContextPtr[i]->memNo; j++)
		{
			newCmdQueue = newContextPtr[i]->cmdQueuePtr[0]->command_queue;
			/* memory is written */
			if (oldMemPtr[j]->isWritten == 1)
			{
				bufferSize = VOCL_MIG_BUF_SIZE;
				bufferNum = (oldMemPtr[j]->size - 1) / VOCL_MIG_BUF_SIZE;
				remainingSize = oldMemPtr[j]->size - bufferNum * VOCL_MIG_BUF_SIZE;
				for (k = 0; k <= bufferNum; k++)
				{
					if (k == bufferNum)
					{
						bufferSize = remainingSize;
					}
					bufferIndex = voclMigRWGetNextBufferIndex(appIndex);
					migRWBufferInfoPtr = voclMigRWGetBufferInfoPtr(appIndex, bufferIndex);
					err = clEnqueueReadBuffer(oldMemPtr[i]->cmdQueue,
											  oldMemPtr[i]->mem,
											  CL_FALSE,
											  i * VOCL_MIG_BUF_SIZE,
											  bufferSize,
											  migRWBufferInfoPtr->ptr,
											  0, NULL, &migRWBufferInfoPtr->rdEvent);
					migRWBufferInfoPtr->wtCmdQueue = newCmdQueue;
					migRWBufferInfoPtr->wtMem = newMemPtr[i]->mem;
					migRWBufferInfoPtr->size = bufferSize;
					migRWBufferInfoPtr->offset = i * VOCL_MIG_BUF_SIZE;
					voclMigSetRWBufferFlag(appIndex, bufferIndex, MIG_RW_SAME_NODE_RDMEM);
				}
			}
		}
	}

	voclMigFinishDataRWOnSameNode(appIndex);

	return;
}

