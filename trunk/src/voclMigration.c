//#include "voclMigration.h"
#include <stdio.h>
#include "vocl_structures.h"
#include "voclKernelArgProc.h"

#define VOCL_MIG_CHECK_ERR(err, str) \
	if (err != CL_SUCCESS)  \
	{ \
		fprintf(stderr, "CL Error %d: %s\n", err, str); \
		exit(1); \
	}

extern int voclMigIssueGPUMemoryWrite(MPI_Comm oldComm, int oldRank,
	MPI_Comm newComm, int newRank, cl_command_queue command_queue,
	cl_mem mem, size_t size);

extern int voclMigIssueGPUMemoryRead(MPI_Comm oldComm, int oldRank, 
	MPI_Comm newComm, int newRank, cl_command_queue command_queue,
	cl_mem mem, size_t size);

extern vocl_device_id voclGetCommandQueueDeviceID(vocl_command_queue cmdQueue);
extern vocl_context voclGetCommandQueueContext(vocl_command_queue cmdQueue);
extern cl_command_queue voclVOCLCommandQueue2CLCommandQueueComm(vocl_command_queue command_queue,
		int *proxyRank, int *proxyIndex, MPI_Comm *proxyComm, MPI_Comm *proxyCommData);
extern void voclUpdateVOCLCommandQueue(vocl_command_queue voclCmdQueue, int proxyRank, int proxyIndex,
		MPI_Comm comm, MPI_Comm commData, vocl_context context, vocl_device_id device);
extern cl_command_queue voclVOCLCommandQueue2CLCommandQueue(vocl_command_queue command_queue);

extern kernel_info *getKernelPtr(cl_kernel kernel);
extern vocl_program voclGetProgramFromKernel(vocl_kernel kernel);
extern void voclUpdateVOCLKernel(vocl_kernel voclKernel, int proxyRank, int proxyIndex,
		MPI_Comm proxyComm, MPI_Comm proxyCommData, vocl_program program);

extern cl_device_id voclVOCLDeviceID2CLDeviceIDComm(vocl_device_id device, int *proxyRank,
		int *proxyIndex, MPI_Comm *proxyComm, MPI_Comm *proxyCommData);

extern void voclUpdateVOCLContext(vocl_context voclContext, int proxyRank, int proxyIndex,
		MPI_Comm proxyComm, MPI_Comm proxyCommData, vocl_device_id deviceID);

extern void voclUpdateVOCLProgram(vocl_program voclProgram, int proxyRank, int proxyIndex,
		MPI_Comm proxyComm, MPI_Comm proxyCommData, vocl_context context);


extern size_t voclGetVOCLMemorySize(vocl_mem memory);
extern cl_mem voclVOCLMemory2CLMemory(vocl_mem memory);
extern void voclUpdateVOCLMemory(vocl_mem voclMemory, int proxyRank, int proxyIndex,
		MPI_Comm proxyComm, MPI_Comm proxyCommData, vocl_context context);


vocl_device_id voclSearchTargetGPU(size_t size)
{
	int i, j, index, err;
	cl_platform_id *platformID;
	cl_device_id   *deviceID;
	vocl_device_id targetDeviceID;
	int numPlatforms, *numDevices, totalDeviceNum;
	char cBuffer[256];
	cl_ulong mem_size;

	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	VOCL_MIG_CHECK_ERR(err, "migration, clGetPlatformIDs");
	platformID = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);
	numDevices = (int *)malloc(sizeof(int) * numPlatforms);
	err = clGetPlatformIDs(numPlatforms, platformID, NULL);

	/* retrieve the total number of devices on all platforms */
	totalDeviceNum = 0;
	for (i = 0; i < numPlatforms; i++)
	{
		err = clGetDeviceIDs(platformID[i], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices[i]);
		VOCL_MIG_CHECK_ERR(err, "migration, clGetDeviceIDs");
		printf("getDeviceID = %d, numDevices = %d\n", i, numDevices[i]);
		totalDeviceNum += numDevices[i];
	}

	deviceID = (cl_device_id *)malloc(sizeof(cl_device_id) * totalDeviceNum);

	/* get all device ids */
	totalDeviceNum = 0;
	for (i = 0; i < numPlatforms; i++)
	{
		err = clGetDeviceIDs(platformID[i], CL_DEVICE_TYPE_GPU, numDevices[i], &deviceID[totalDeviceNum], NULL);
		VOCL_MIG_CHECK_ERR(err, "migration, clGetDeviceIDs");
		totalDeviceNum += numDevices[i];
	}

	/* get global memory size of each GPU */
	for (i = 1; i < totalDeviceNum; i++)
	{
		err = clGetDeviceInfo(deviceID[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
		VOCL_MIG_CHECK_ERR(err, "migration, clGetDeviceInfo");
		if (mem_size >= size)
		{
			targetDeviceID = deviceID[i];
			break;
		}
	}

	free(platformID);
	free(numDevices);
	free(deviceID);

	return targetDeviceID;

}

int voclCheckTaskMigration(vocl_kernel kernel, vocl_command_queue command_queue)
{
	kernel_info *kernelPtr;
	vocl_device_id deviceID;
	cl_ulong mem_size;
	int err;
	
	kernelPtr = getKernelPtr(kernel);
	deviceID = voclGetCommandQueueDeviceID(command_queue);
	/* get the global memory size */
	err = clGetDeviceInfo(deviceID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
	VOCL_MIG_CHECK_ERR(err, "Migration, get device memory size error");
	printf("neededSize = %ld, availableSize = %ld\n", kernelPtr->globalMemSize, mem_size);
	if (mem_size < kernelPtr->globalMemSize)
	{
		return 1;  /* gpu memory is not enough for the current kernel */
	}
	else
	{
		return 0; /* gpu memory is enougth */
	}
}

void voclTaskMigration(vocl_kernel kernel, vocl_command_queue command_queue)
{
	kernel_info *kernelPtr;
	vocl_device_id newDeviceID, oldDeviceID;
	cl_device_id clDeviceID;
	vocl_context context;
	vocl_program program;
	cl_mem clMem;
	cl_command_queue oldCmdQueue, newCmdQueue;
	int newRank, newIndex, oldRank, oldIndex;
	int proxySourceRank, proxyDestRank;
	MPI_Comm newComm, newCommData, oldComm, oldCommData;
	int i, err, memWrittenFlag;
	size_t size;
	char *tmpBuf;

	/* finish previous tasks */
	clFinish(command_queue);

	kernelPtr = getKernelPtr(kernel);
	oldDeviceID = voclGetCommandQueueDeviceID(command_queue);
	newDeviceID = voclSearchTargetGPU(kernelPtr->globalMemSize);
	context = voclGetCommandQueueContext(command_queue);
	program = voclGetProgramFromKernel(kernel);

	clDeviceID = voclVOCLDeviceID2CLDeviceIDComm(newDeviceID, &newRank, &newIndex, &newComm, &newCommData);
	/* re-create context */
	voclUpdateVOCLContext(context, newRank, newIndex, newComm, newCommData, newDeviceID);

	/* re-create command queue */
	oldCmdQueue = voclVOCLCommandQueue2CLCommandQueueComm(command_queue, &oldRank, &oldIndex, 
					&oldComm, &oldCommData);
	
	voclUpdateVOCLCommandQueue(command_queue, newRank, newIndex, 
		newComm, newCommData, context, newDeviceID);
	newCmdQueue = voclVOCLCommandQueue2CLCommandQueue(command_queue);

	/* go throught all argument of the kernel */
	memWrittenFlag = 0;
	for (i = 0; i < kernelPtr->args_num; i++)
	{
		printf("kernelPtr->args_flag[%d] = %d\n", i, kernelPtr->args_flag[i]);
		if (kernelPtr->args_flag[i] == 1) /* it is global memory */
		{
			size = voclGetVOCLMemorySize(kernelPtr->args_ptr[i].memory);
			if (voclGetMemWrittenFlag(kernelPtr->args_ptr[i].memory))
			{
				/* send a message to the source proxy process for migration data transfer */
				clMem = voclVOCLMemory2CLMemory(kernelPtr->args_ptr[i].memory);
				proxyDestRank = voclMigIssueGPUMemoryRead(oldComm, oldRank, newComm, newRank, 
						oldCmdQueue, clMem, size);
				/* update the memory to the new device */
				voclUpdateVOCLMemory(kernelPtr->args_ptr[i].memory, newRank, newIndex,
								 newComm, newCommData, context);
				/* send a message to the dest proxy process for migration data transfer */
				clMem = voclVOCLMemory2CLMemory(kernelPtr->args_ptr[i].memory);
				proxySourceRank = voclMigIssueGPUMemoryWrite(oldComm, oldRank, newComm, newRank, 
						newCmdQueue, clMem, size);
				memWrittenFlag = 1;
			}
			else
			{
				voclUpdateVOCLMemory(kernelPtr->args_ptr[i].memory, newRank, newIndex,
						newComm, newCommData, context);
			}
			clMem = voclVOCLMemory2CLMemory(kernelPtr->args_ptr[i].memory);
			memcpy(kernelPtr->args_ptr[i].arg_value, (void*)&clMem, sizeof(cl_mem));
		}
	}

	/* if there are memory transfer from one proxy process  to another */
	/* tell the proxy process to complete the data transfer */
	if (memWrittenFlag == 1)
	{
		voclMigFinishDataTransfer(oldComm, oldRank, newComm, newRank, 
			proxySourceRank, proxyDestRank);
	}

	/* re-create program */
	voclUpdateVOCLProgram(program, newRank, newIndex, newComm, 
		newCommData, context);
		
	/* build the program, only one target device is searched */
	/* the last two arguments are ignored currently */
	err = clBuildProgram(program, 1, &newDeviceID, voclGetProgramBuildOptions(program), 0, 0);

	voclUpdateVOCLKernel(kernel, newRank, newIndex, newComm, 
			newCommData, program);
	return;
}

