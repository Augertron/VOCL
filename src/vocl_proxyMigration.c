#include <stdio.h>
#include <string.h>
#include "vocl_proxy.h"
#include "vocl_proxyWinProc.h"
#include "vocl_proxyStructures.h"

extern vocl_virtual_gpu *voclProxyGetVirtualGPUPtr(int appIndex, cl_device_id deviceID);
extern void voclProxyAddVirtualGPU(int appIndex, cl_device_id deviceID);
extern void voclProxyAddContextToVGPU(int appIndex, cl_device_id deviceID, vocl_proxy_context *context);
extern void voclProxyAddCommandQueueToVGPU(int appIndex, cl_device_id deviceID, vocl_proxy_command_queue *command_queue);
extern void voclProxyReleaseVirtualGPU(int appIndex, cl_device_id deviceID);
extern void vocl_proxyGetKernelNumsOnGPUs(struct strKernelNumOnDevice *gpuKernelNum);

extern void voclProxyAddContext(cl_context context, cl_uint deviceNum, cl_device_id *deviceIDs);
extern vocl_proxy_context *voclProxyGetContextPtr(cl_context context);
extern void voclProxyAddProgramToContext(cl_context context, vocl_proxy_program *program);
extern void voclProxyAddCommandQueueToContext(cl_context context, vocl_proxy_command_queue *command_queue);
extern void voclProxyAddMemToContext(cl_context context, vocl_proxy_mem *mem);
extern void voclProxyReleaseContext(cl_context context);

extern void voclProxyAddProgram(cl_program program, char *sourceString, size_t sourceSize, int stringNum, size_t *stringSizeArray, cl_context context);
extern vocl_proxy_program *voclProxyGetProgramPtr(cl_program program);
extern void voclProxySetProgramBuildOptions(cl_program program, cl_uint deviceNum, cl_device_id *device_list, char *buildOptions);
extern void voclProxyAddKernelToProgram(cl_program program, vocl_proxy_kernel *kernel);
extern void voclProxyReleaseProgram(cl_program program);

extern void voclProxyAddKernel(cl_kernel kernel, char *kernelName, cl_program program);
extern vocl_proxy_kernel *voclProxyGetKernelPtr(cl_kernel kernel);
extern void voclProxyReleaseKernel(cl_kernel kernel);

extern void voclProxyAddCmdQueue(cl_command_queue command_queue, cl_command_queue_properties properties, cl_context context, cl_device_id deviceID);
extern vocl_proxy_command_queue *voclProxyGetCmdQueuePtr(cl_command_queue command_queue);
extern void voclProxyReleaseCommandQueue(cl_command_queue command_queue);

extern void voclProxyAddMem(cl_mem mem, cl_mem_flags flags, size_t size, cl_context context);
extern vocl_proxy_mem *voclProxyGetMemPtr(cl_mem mem);
extern void voclProxyReleaseMem(cl_mem mem);

extern MPI_Win *voclProxyGetWinPtr(int index);


void voclProxyMigCreateVirtualGPU(int appIndex, cl_device_id deviceID, vocl_vgpu_msg *msgPtr, char *bufPtr)
{
	int i, j, k, stringIndex;
	size_t offset;
	size_t startLoc;
	char **strings, *sourceString;
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
	voclProxyAddVirtualGPU(appIndex, deviceID);

	/* unpack the received message and create corresponding */
	/* resources on the new virtual GPU */
	offset = 0;
	startLoc = 0;
	for (i = 0; i < msgPtr->contextNum; i++)
	{
		/* obtain the context pointer to unpack the migration message */
		contextPtr = (vocl_proxy_context *)(bufPtr+offset);
		offset += sizeof(vocl_proxy_context);
	
		context = clCreateContext(NULL, 1, &deviceID, NULL, NULL, &retCode);
		voclProxyAddContext(context, 1, &deviceID);
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
				strings[k][programPtr->stringSizeArray[stringIndex]] = '\0';
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
				offset += sizeof(kernelPtr->nameLen);

				kernel = clCreateKernel(program, kernelPtr->kernelName, &retCode);

				/* store the kernel */
				voclProxyAddKernel(kernel, kernelPtr->kernelName, program);
				knPtr = voclProxyGetKernelPtr(kernel);
				voclProxyAddKernelToProgram(program, knPtr);
			}

			/* decode command queue */
			cmdQueuePtr = (vocl_proxy_command_queue *)(bufPtr + offset);
			offset += sizeof(vocl_proxy_command_queue);
			cmdQueue = clCreateCommandQueue(context, deviceID, cmdQueuePtr->properties, &retCode);

			/* store the command queue */
			voclProxyAddCmdQueue(cmdQueue, cmdQueuePtr->properties, context, deviceID);
			cqPtr = voclProxyGetCmdQueuePtr(cmdQueue);
			voclProxyAddCommandQueueToContext(context, cqPtr);
			voclProxyAddCommandQueueToVGPU(appIndex, deviceID, cqPtr);

			/* decoce the device memory */
			memPtr = (vocl_proxy_mem *)(bufPtr + offset);
			offset += sizeof(vocl_proxy_mem);

			mem = clCreateBuffer(context, memPtr->flags, memPtr->size,
								 NULL, &retCode);
			/* store the memory */
			voclProxyAddMem(mem, memPtr->flags, memPtr->size, context);
			mmPtr = voclProxyGetMemPtr(mem);
			voclProxyAddMemToContext(context, mmPtr);
		}
	}

	return;
}

void voclProxyMigReleaseVirtualGPU(int appIndex, cl_device_id deviceID)
{
	int i, j, k;
	vocl_virtual_gpu *vgpuPtr;
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

	vgpuPtr = voclProxyGetVirtualGPUPtr(appIndex, deviceID);
	
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
	voclProxyReleaseVirtualGPU(appIndex, deviceID);

	return;
}

/* go through each proxy process and obtain the */
/* number of kernels in waiting */
struct strKernelNumOnDevice *voclProxyMigQueryLoadOnGPU(int appIndex, int *proxyNum)
{
	int myRank, proxyRank, j;
	MPI_Request *request;
	MPI_Status *status;
	int requestNo = 0;
	struct strKernelNumOnDevice *loadPtr;
	MPI_Comm comm;
	MPI_Win *winPtr;
	vocl_proxy_wins *winBufPtr;

	comm = MPI_COMM_WORLD;
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
					  winBufPtr->wins[j].proxyRank, LB_GET_KERNEL_NUM, comm, 
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

	free(winBufPtr);
	free(request);
	free(status);
	return loadPtr;
}


