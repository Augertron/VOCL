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

//extern int voclIsOnLocalNode(int i);
//
//static struct strVoclGPUMemInfo *voclGPUInfo = NULL;
//static int voclTotalDeviceNum = 0;
//
//struct strVoclAppInfo voclAppInfo;
//
//void voclMigGetGPUMemorySizes()
//{
//	int i, j, index, err;
//	cl_platform_id *platformID;
//	cl_device_id   *deviceID;
//	int numPlatforms, *numDevices, totalDeviceNum;
//	char cBuffer[256];
//	cl_ulong mem_size;
//
//	err = clGetPlatformIDs(0, NULL, &numPlatforms);
//	VOCL_MIG_CHECK_ERR(err, "migration, clGetPlatformIDs");
//	platformID = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);
//	numDevices = (int *)malloc(sizeof(int) * numPlatforms);
//	err = clGetPlatformIDs(numPlatforms, platformID, NULL);
//
//	/* retrieve the total number of devices on all platforms */
//	totalDeviceNum = 0;
//	for (i = 0; i < numPlatforms; i++)
//	{
//		err = clGetDeviceIDs(platformID[i], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices[i]);
//		VOCL_MIG_CHECK_ERR(err, "migration, clGetDeviceIDs");
//		totalDeviceNum += numDevices[i];
//	}
//
//	voclTotalDeviceNum = totalDeviceNum;
//	deviceID = (cl_device_id *)malloc(sizeof(cl_device_id) * totalDeviceNum);
//	voclGPUInfo = (struct strVoclGPUMemInfo *)malloc(sizeof(struct strVoclGPUMemInfo) * totalDeviceNum);
//
//	/* get all device ids */
//	totalDeviceNum = 0;
//	for (i = 0; i < numPlatforms; i++)
//	{
//		err = clGetDeviceIDs(platformID[i], CL_DEVICE_TYPE_GPU, numDevices[i], &deviceID[totalDeviceNum], NULL);
//		VOCL_MIG_CHECK_ERR(err, "migration, clGetDeviceIDs");
//		totalDeviceNum += numDevices[i];
//	}
//
//	/* get global memory size of each GPU */
//	totalDeviceNum = 0;
//	for (i = 0; i < numPlatforms; i++)
//	{
//		for (j = 0; j < numDevices[i]; j++)
//		{
//			index = j + totalDeviceNum;
//			voclGPUInfo[index].gpuIndex = index;
//			err = clGetDeviceInfo(deviceID[index], CL_DEVICE_NAME, sizeof(cBuffer), cBuffer, NULL);
//			sprintf(voclGPUInfo[index].gpuName, "%s", cBuffer);
//			err |= clGetDeviceInfo(deviceID[index], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
//			VOCL_MIG_CHECK_ERR(err, "migration, clGetDeviceInfo");
//			voclGPUInfo[index].globalSize = mem_size;
//			voclGPUInfo[index].allocatedSize = 0;
//			voclGPUInfo[index].deviceID = (vocl_device_id)deviceID[index];
//
//			/* whether it is a local gpu */
//			voclGPUInfo[index].isLocalGPU = voclIsOnLocalNode(i);
//		}
//	}
//
//	free(platformID);
//	free(numDevices);
//	free(deviceID);
//
//	return;
//}
//
//void voclMigFinalize()
//{
//	if (voclGPUInfo != NULL)
//	{
//		free(voclGPUInfo);
//		voclGPUInfo = NULL;
//	}
//	voclTotalDeviceNum = 0;
//}
//
//
//
//void voclCreateAppInfo(int proxyRank, int proxyIndex, MPI_Comm comm, MPI_Comm commData)
//{
//	voclAppInfo.proxyRank = proxyRank;
//	voclAppInfo.proxyIndex = proxyIndex;
//	voclAppInfo.comm = comm;
//	voclAppInfo.commData = commData;
//	voclAppInfo.memorySizeNeeded = 0;
//
//	voclAppInfo.memNum = VOCL_MIG_MEM_NUM;
//	voclAppInfo.memNo = 0;
//	voclAppInfo.memPtr = (vocl_mem *)malloc(sizeof(vocl_mem) * voclAppInfo.memNum);
//
//	voclAppInfo.platformNum = VOCL_MIG_PLATFORM_NUM;
//	voclAppInfo.platformNo = 0;
//	voclAppInfo.platformPtr = (vocl_platform_id *)malloc(sizeof(vocl_platform_id) * voclAppInfo.platformNum);
//
//	voclAppInfo.deviceNum = VOCL_MIG_DEVICE_NUM;
//	voclAppInfo.deviceNo = 0;
//	voclAppInfo.devicePtr = (vocl_device_id *)malloc(sizeof(vocl_device_id) * voclAppInfo.deviceNum);
//
//	voclAppInfo.contextNum = VOCL_MIG_CONTEXT_NUM;
//	voclAppInfo.contextNo = 0;
//	voclAppInfo.contextPtr = (vocl_context *)malloc(sizeof(vocl_context) * voclAppInfo.contextNum);
//
//	voclAppInfo.cmdQueueNum = VOCL_MIG_CMDQ_NUM;
//	voclAppInfo.cmdQueueNo = 0;
//	voclAppInfo.cmdQueuePtr = (vocl_command_queue)malloc(sizeof(vocl_command_queue) * voclAppInfo.cmdQueueNum);
//	voclAppInfo.oldCmdQueuePtr = (vocl_command_queue)malloc(sizeof(vocl_command_queue) * voclAppInfo.cmdQueueNum);
//
//
//	voclAppInfo.programNum = VOCL_MIG_PROGRAM_NUM;
//	voclAppInfo.programNo = 0;
//	voclAppInfo.programPtr = (vocl_program *)malloc(sizeof(vocl_program) * voclAppInfo.programNum);
//
//	voclAppInfo.kernelNum = VOCL_MIG_KERNEL_NUM;
//	voclAppInfo.kernelNo = 0;
//	voclAppInfo.kernelPtr = (vocl_kernel *)malloc(sizeof(vocl_kernel) * voclAppInfo.kernelNum);
//
//	voclAppInfo.eventNum = VOCL_MIG_EVENT_NUM;
//	voclAppInfo.eventNo = 0;
//	voclAppInfo.eventPtr = (vocl_event *)malloc(sizeof(vocl_event) * voclAppInfo.eventNum);
//
//	voclAppInfo.samplerNum = VOCL_MIG_SAMPLER_NUM;
//	voclAppInfo.samplerNo = 0;
//	voclAppInfo.samplerPtr = (vocl_sampler *)malloc(sizeof(vocl_sampler) * voclAppInfo.samplerNum);
//}
//
//void voclMigAddPlatform(vocl_platform_id platform)
//{
//	if (voclAppInfo.platformNo >= voclAppInfo.platformNum)
//	{
//		voclAppInfo.platformNum *= 2;
//		voclAppInfo.platformPtr = (vocl_platform_id *)realloc(voclAppInfo.platformPtr, 
//				sizeof(vocl_platform_id) * voclAppInfo.platformNum);
//	}
//	voclAppInfo.platformPtr[voclAppInfo.platformNo++] = platform;
//	
//	return;
//}
//
//void voclMigAddDevice(vocl_device_id device)
//{
//	if (voclAppInfo.deviceNo >= voclAppInfo.deviceNum)
//	{
//		voclAppInfo.deviceNum *= 2;
//		voclAppInfo.devicePtr = (vocl_device_id *)realloc(voclAppInfo.devicePtr, sizeof(vocl_device_id) * voclAppInfo.deviceNum);
//	}
//	voclAppInfo.devicePtr[voclAppInfo.deviceNo++] = device;
//	return;
//}
//
//void voclMigAddContext(vocl_context context)
//{
//	if (voclAppInfo.contextNo >= voclAppInfo.contextNum)
//	{
//		voclAppInfo.contextNum *= 2;
//		voclAppInfo.contextPtr = (vocl_context *)realloc(voclAppInfo.contextPtr, sizeof(vocl_context) * voclAppInfo.contextNum);
//	}
//	voclAppInfo.contextPtr[voclAppInfo.contextNo++] = context;
//	return;
//}
//
//void voclMigAddCmdQueue(vocl_command_queue cmdQueue)
//{
//	if (voclAppInfo.cmdQueueNo >= voclAppInfo.cmdQueueNum)
//	{
//		voclAppInfo.cmdQueueNum *= 2;
//		voclAppInfo.cmdQueuePtr = (vocl_command_queue *)realloc(voclAppInfo.cmdQueuePtr, 
//				sizeof(vocl_command_queue) * voclAppInfo.cmdQueueNum);
//		voclAppInfo.oldCmdQueuePtr = (vocl_command_queue *)realloc(voclAppInfo.oldCmdQueuePtr, 
//				sizeof(vocl_command_queue) * voclAppInfo.cmdQueueNum);
//	}
//	voclAppInfo.cmdQueuePtr[voclAppInfo.cmdQueueNo++] = cmdQueue;
//	return;
//}
//
//void voclMigAddProgram(vocl_program program)
//{
//	if (voclAppInfo.programNo >= voclAppInfo.programNum)
//	{
//		voclAppInfo.programNum *= 2;
//		voclAppInfo.programPtr = (vocl_program *)realloc(voclAppInfo.programPtr, sizeof(vocl_program) * voclAppInfo.programNum);
//	}
//	voclAppInfo.programPtr[voclAppInfo.programNo++] = program;
//	return;
//}
//
//void voclMigAddMemory(vocl_mem memory, size_t size)
//{
//	if (voclAppInfo.memNo >= voclAppInfo.memNum)
//	{
//		voclAppInfo.memNum *= 2;
//		voclAppInfo.memPtr = (vocl_mem *)realloc(voclAppInfo.memPtr, sizeof(vocl_mem) * voclAppInfo.memNum);
//	}
//	voclAppInfo.memPtr[voclAppInfo.memNo++] = memory;
//	voclAppInfo.memorySizeNeeded += size;
//	return;
//}
//
//void voclMigAddKernel(vocl_kernel kernel)
//{
//	if (voclAppInfo.kernelNo >= voclAppInfo.kernelNum)
//	{
//		voclAppInfo.kernelNum *= 2;
//		voclAppInfo.kernelPtr = (vocl_kernel *)realloc(voclAppInfo.kernelPtr, sizeof(vocl_kernel) * voclAppInfo.kernelNum);
//	}
//	voclAppInfo.kernelPtr[voclAppInfo.kernelNo++] = kernel;
//
//	return;
//}
//
//void voclMigAddEvent(vocl_event event)
//{
//	if (voclAppInfo.eventNo >= voclAppInfo.eventNum)
//	{
//		voclAppInfo.eventNum *= 2;
//		voclAppInfo.eventPtr = (vocl_event *)realloc(voclAppInfo.eventPtr, sizeof(vocl_event) * voclAppInfo.eventNum);
//	}
//	voclAppInfo.eventPtr[voclAppInfo.eventNo++] = event;
//
//	return;
//}
//
//void voclMigAddSampler(vocl_sampler sampler)
//{
//	if (voclAppInfo.samplerNo >= voclAppInfo.samplerNum)
//	{
//		voclAppInfo.samplerNum *= 2;
//		voclAppInfo.samplerPtr = (vocl_sampler *)realloc(voclAppInfo.samplerPtr, sizeof(vocl_sampler) * voclAppInfo.samplerNum);
//	}
//	voclAppInfo.samplerPtr[voclAppInfo.samplerNo++] = sampler;
//
//	return;
//}
//
//int voclMigIsGPUMemoryEnough(size_t memorySize)
//{
//	int i, index = -1;
//
//	/* search the corresponding gpu */
//	for (i = 0; i < voclTotalDeviceNum; i++)
//	{
//		if (voclGPUInfo[i].deviceID == voclAppInfo.devicePtr[0])
//		{
//			index = i;
//			break;
//		}
//	}
//
//	if (memorySize + voclGPUInfo[index].allocatedSize > voclGPUInfo[index].globalSize)
//	{
//		return 0;
//	}
//	else
//	{
//		return 1;
//	}
//}
//
//void voclMigUpdateGPUMemorySize(int gpuIndex, size_t memorySize)
//{
//	int i, index = -1;
//	for (i = 0; i < voclTotalDeviceNum;i++)
//	{
//		if (proxyIndex == voclGPUInfo[i].gpuIndex)
//		{
//			index = i;
//			break;
//		}
//	}
//
//	if (index == -1)
//	{
//		printf("Input gpu index %d is invalid!\n", gpuInde);
//	}
//
//	voclGPUInfo[index].allocatedSize += memorySize;
//	return;
//}

//void voclMigrationResources(vocl_device_id oldDevice, cl_device_id newDevice)
//{
//	int err, i;
//	MPI_Comm proxyComm, proxyCommData;
//	int proxyRank, proxyIndex;
//	cl_device_id deviceID;
//	cl_command_queue cmdQueue;
//
//	deviceID = voclVOCLDeviceID2CLDeviceIDComm(newDevice, &proxyRank, &proxyIndex, &proxyComm, &proxyCommData);
//	/* re-create context */
//	voclUpdateVOCLContext(voclAppInfo.contextPtr[0], proxyRank, proxyIndex, proxyComm, proxyCommData, newDevice);
//
//	/* re-create command queue */
//	cmdQueue = voclVOCLCommandQueue2CLCommandQueueComm(voclAppInfo.cmdQueuePtr[0], &proxyRank, &proxyIndex, 
//					&proxyComm, &proxyCommData);
//	voclAppInfo.oldCmdQueuePtr[0] = voclCLCommandQueue2VOCLCommandQueue(cmdQueue, proxyRank, proxyIndex, 
//										proxyComm, proxyCommData);
//
//	voclUpdateVOCLCommandQueue(voclAppInfo.cmdQueuePtr[0], proxyRank, proxyIndex, 
//		proxyComm, proxyCommData, voclAppInfo.contextPtr[0]);
//
//	/* re-create program */
//	voclUpdateVOCLProgram(voclAppInfo.programPtr[0], proxyRank, proxyIndex, proxyComm, 
//		proxyCommData, voclAppInfo.contextPtr[0]);
//		
//	/* build the program */
//	err = clBuildProgram(voclAppInfo.programPtr[0], 0, 0, 0, 0, 0);
//
//	/* re-create the kernel */
//	for (i = 0; i < voclAppInfo.kernelNo; i++)
//	{
//		voclUpdateVOCLKernel(voclAppInfo.kernelPtr[i], proxyRank, proxyIndex, proxyComm, 
//			proxyCommData, voclAppInfo.programPtr[0]);
//	}
//
//	return;
//}
//
///* current gpu memory is not enough, migrate to another gpu */
//void voclMigrationMemory(vocl_device_id oldDevice, vocl_device_id newDevice)
//{
//	int i, err;
//	cl_device_id deviceID;
//	int proxyRank, proxyIndex;
//	char *tmpBuf;
//	size_t size;
//	MPI_Comm proxyComm, proxyCommData;
//
//	deviceID = voclVOCLDeviceID2CLDeviceIDComm(newDevice, &proxyRank, &proxyIndex, &proxyComm, &proxyCommData);
//
//	for (i = 0; i < voclAppInfo.memNo; i++)
//	{
//		size = voclGetVOCLMeorySize(voclAppInfo.memPtr[i]);
//		tmpBuf = (char *)malloc(size);
//		/* read contents in previous memory */
//		clEnqueueReadBuffer(voclAppInfo.oldCmdQueuePtr[0], voclAppInfo.memPtr[i], CL_TRUE,
//							0, size, tmpBuf, NULL, 0, NULL, NULL);
//		voclUpdateVOCLMemory(voclAppInfo.memPtr[i], proxyRank, proxyIndex,
//							 proxyComm, proxyCommData, voclAppInfo.contextPtr[0]);
//		/* write to newly allocated memory */
//		clEnqueueWriteBuffer(voclAppInfo.cmdQueuePtr[0], voclAppInfo.memPtr[i], CL_TRUE,
//							0, size, tmpBuf, NULL, 0, NULL, NULL);
//		free(tmpBuf);
//	}
//
//	return;
//}

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
	cl_command_queue cmdQueue;
	int proxyRank, proxyIndex, tmpRank, tmpIndex;
	MPI_Comm proxyComm, proxyCommData, tmpComm, tmpCommData;
	int i, err;
	size_t size;
	char *tmpBuf;

	kernelPtr = getKernelPtr(kernel);
	oldDeviceID = voclGetCommandQueueDeviceID(command_queue);
	printf("oclDeviceID = %ld\n", oldDeviceID);
	newDeviceID = voclSearchTargetGPU(kernelPtr->globalMemSize);
	printf("newDeviceID = %ld\n", newDeviceID);
	context = voclGetCommandQueueContext(command_queue);
	program = voclGetProgramFromKernel(kernel);

	clDeviceID = voclVOCLDeviceID2CLDeviceIDComm(newDeviceID, &proxyRank, &proxyIndex, &proxyComm, &proxyCommData);
	/* re-create context */
	voclUpdateVOCLContext(context, proxyRank, proxyIndex, proxyComm, proxyCommData, newDeviceID);

	/* re-create command queue */
	clFinish(command_queue);
	cmdQueue = voclVOCLCommandQueue2CLCommandQueueComm(command_queue, &tmpRank, &tmpIndex, 
					&tmpComm, &tmpCommData);
	
	voclUpdateVOCLCommandQueue(command_queue, proxyRank, proxyIndex, 
		proxyComm, proxyCommData, context, newDeviceID);

	/* go throught all argument of the kernel */
	printf("args_num = %d\n", kernelPtr->args_num);
	for (i = 0; i < kernelPtr->args_num; i++)
	{
		printf("kernelPtr->args_flag[%d] = %d\n", i, kernelPtr->args_flag[i]);
		if (kernelPtr->args_flag[i] == 1) /* it is global memory */
		{
			size = voclGetVOCLMemorySize(kernelPtr->args_ptr[i].memory);
			printf("size = %ld\n", size);
			if (voclGetMemWrittenFlag(kernelPtr->args_ptr[i].memory))
			{
				tmpBuf = (char *)malloc(size);
				/* read contents in previous memory */
				printf("cmdQueue = %ld, memory = %d, i = %d\n", cmdQueue, kernelPtr->args_ptr[i].memory, i);
				clMigEnqueueReadBuffer(cmdQueue, kernelPtr->args_ptr[i].memory, CL_TRUE,
								0, size, tmpBuf, NULL, 0, NULL);
				printf("a = %.1f\n", ((float *)tmpBuf)[10]);
				voclUpdateVOCLMemory(kernelPtr->args_ptr[i].memory, proxyRank, proxyIndex,
								 proxyComm, proxyCommData, context);
				/* write to newly allocated memory */
				clEnqueueWriteBuffer((cl_command_queue)command_queue, kernelPtr->args_ptr[i].memory, 
					CL_TRUE, 0, size, tmpBuf, NULL, 0, NULL);
				free(tmpBuf);
			}
			else
			{
				voclUpdateVOCLMemory(kernelPtr->args_ptr[i].memory, proxyRank, proxyIndex,
						proxyComm, proxyCommData, context);
			}
			clMem = voclVOCLMemory2CLMemoryComm(kernelPtr->args_ptr[i].memory, &tmpRank, &tmpIndex,
						&tmpComm, &tmpCommData);
			memcpy(kernelPtr->args_ptr[i].arg_value, (void*)&clMem, sizeof(vocl_mem));
		}
	}

	/* re-create program */
	voclUpdateVOCLProgram(program, proxyRank, proxyIndex, proxyComm, 
		proxyCommData, context);
		
	/* build the program, only one target device is searched */
	/* the last two arguments are ignored currently */
	err = clBuildProgram(program, 1, &newDeviceID, voclGetProgramBuildOptions(program), 0, 0);

	voclUpdateVOCLKernel(kernel, proxyRank, proxyIndex, proxyComm, 
			proxyCommData, program);
}

