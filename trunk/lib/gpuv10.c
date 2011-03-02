#include <stdio.h>
#include <string.h>
#include "gpuv.h"

/****************************************************************************
 * 4.   This is the first working version, with all functions in matrixMul and S-W are mapped
 * 7.   The function clGetDeviceIDs is modified to support multiple devices,
 * 		second work version, some Opencl API functions are added to support matrix transpose
 * 8.   Modify the clEnqueWrite/ReadBuffer API to support events, a few other API functions
 *      are added.
 * 9.   Change the way of setKernelArg processing to save argument transmission overhead
 * 10.  Process the asynchronous data transfer between the host and device memory
 ****************************************************************************/
//for slave process
int slaveComm;
int slaveCreated = 0;
int np = 1;
int errCodes[MAX_NPS];
cl_uint readBufferTag = PROGRAM_END + 1;
//for storing kernel arguments
kernel_info *kernelInfo = NULL;

//get the kernel structure whether argument
//pointer is stored
cl_int createKernel(cl_kernel kernel)
{
	kernel_info *kernelPtr;
	kernelPtr = (kernel_info *)malloc(sizeof(kernel_info));
	kernelPtr->kernel = kernel;
	kernelPtr->args_num = 0;
	kernelPtr->args_allocated = 0;
	kernelPtr->args_ptr = NULL;
	kernelPtr->next = kernelInfo;

	kernelInfo = kernelPtr;
}

kernel_info *getKernelPtr(cl_kernel kernel)
{
	kernel_info *kernelPtr = NULL;

	kernel_info *nextKernel = kernelInfo;
	while (nextKernel != NULL)
	{
		if (kernel == nextKernel->kernel)
		{
			kernelPtr = nextKernel;
			break;
		}
		nextKernel = nextKernel->next;
	}

	if (kernelPtr == NULL)
	{
		printf("Error, kernel does not exist. In getKernelPtr!\n");
		exit (1);
	}

	return kernelPtr;
}

//release the structure corresponing to the current kernel
cl_int releaseKernelPtr(cl_kernel kernel)
{
	kernel_info *kernelPtr, *curKernel, *preKernel;
	if (kernel == kernelInfo->kernel)
	{
		kernelPtr = kernelInfo;
		kernelInfo = kernelInfo->next;
		if (kernelPtr->args_allocated == 1)
		{
			free(kernelPtr->args_ptr);
		}
		free(kernelPtr);
		return 0;
	}

	kernelPtr = NULL;
	curKernel = kernelInfo->next;
	preKernel = kernelInfo;
	while (curKernel != NULL)
	{
		if (kernel == curKernel->kernel)
		{
			kernelPtr = curKernel;
			break;
		}
		preKernel = curKernel;
		curKernel = curKernel->next;
	}

	if (kernelPtr == NULL)
	{
		printf("Kernel does not exist!\n");
		exit (1);
	}

	preKernel->next = curKernel->next;
	if (kernelPtr->args_allocated == 1)
	{
		free(kernelPtr->args_ptr);
	}
	free(kernelPtr);

	return 0;
}
//end set arguments function

//--------------------------------------------------------------------------------------
//process asynchronous data read
typedef struct strDataRead {
	cl_event           event;
	MPI_Request        request;
	struct strDataRead *next;
} DATA_READ;

typedef struct cmdQueue {
	cl_command_queue command_queue;
	DATA_READ        *dataReadPtr;
	DATA_READ        *dataReadPtrTail;
	struct cmdQueue  *next;
} CMD_QUEUE;

CMD_QUEUE *hCmdQueueHead = NULL;

void createCommandQueue(cl_command_queue command_queue)
{
	CMD_QUEUE *cmdQueuePtr = (CMD_QUEUE *)malloc(sizeof(CMD_QUEUE));
	cmdQueuePtr->command_queue = command_queue;
	cmdQueuePtr->dataReadPtr = NULL;
	cmdQueuePtr->dataReadPtrTail = NULL;
	cmdQueuePtr->next = hCmdQueueHead;
	hCmdQueueHead = cmdQueuePtr;

	return;
}

CMD_QUEUE* getCommandQueue(cl_command_queue command_queue)
{
	CMD_QUEUE *hCmdQueues = hCmdQueueHead;
	while (hCmdQueues != NULL)
	{
		if (hCmdQueues->command_queue == command_queue)
		{
			break;
		}
	}

	if (hCmdQueues == NULL)
	{
		printf("Error in lib, command queue does not exist. In getCommandQueue!\n");
		exit(1);
	}

	return hCmdQueues;
}

void releaseCommandQueue(cl_command_queue command_queue)
{
	CMD_QUEUE *preCmdQueue, *curCmdQueue, *nextCmdQueue;
	DATA_READ *curDataReadPtr, *nextDataReadPtr;
	if (command_queue == hCmdQueueHead->command_queue)
	{
		curCmdQueue = hCmdQueueHead;
		hCmdQueueHead = hCmdQueueHead->next;
		curDataReadPtr = curCmdQueue->dataReadPtr;
		while (curDataReadPtr != NULL)
		{
			nextDataReadPtr = curDataReadPtr->next;
			free(curDataReadPtr);
			curDataReadPtr = nextDataReadPtr;
		}
		free(curCmdQueue);
		return;
	}

	preCmdQueue = hCmdQueueHead;
	curCmdQueue = preCmdQueue->next;
	while (curCmdQueue != NULL)
	{
		if (command_queue == curCmdQueue->command_queue)
		{
			curDataReadPtr = curCmdQueue->dataReadPtr;
			while (curDataReadPtr != NULL)
			{
				nextDataReadPtr = curDataReadPtr->next;
				free(curDataReadPtr);
				curDataReadPtr = nextDataReadPtr;
			}
			preCmdQueue->next = curCmdQueue->next;
			free(curCmdQueue);
			break;
		}
		preCmdQueue = curCmdQueue;
		curCmdQueue = curCmdQueue->next;
	}
	
	return;
}

DATA_READ *createDataRead(cl_command_queue command_queue,
						  cl_event         event)
{
	CMD_QUEUE *cmdQueue = getCommandQueue(command_queue);
	DATA_READ *dataReadPtr = (DATA_READ *)malloc(sizeof(DATA_READ));
	dataReadPtr->event = event;
	dataReadPtr->next = NULL;

	if (cmdQueue->dataReadPtr == NULL &&
		cmdQueue->dataReadPtrTail == NULL)
	{
		cmdQueue->dataReadPtr = dataReadPtr;
		cmdQueue->dataReadPtrTail = dataReadPtr;
	}
	else
	{
		cmdQueue->dataReadPtrTail->next = dataReadPtr;
		cmdQueue->dataReadPtrTail = dataReadPtr;
	}

	return dataReadPtr;
}

DATA_READ *getDataRead(cl_command_queue command_queue,
                               cl_event         event)
{
	CMD_QUEUE *cmdQueue = getCommandQueue(command_queue);
	DATA_READ *dataReadPtr = cmdQueue->dataReadPtr;
	while (dataReadPtr != NULL)
	{
		if (dataReadPtr->event == event)
		{
			break;
		}
		dataReadPtr = dataReadPtr->next;
	}

	if (dataReadPtr == NULL)
	{
		printf("Lib, In function getDataRead() error, the corresponding event is not there!\n");
		exit (0);
	}

	return dataReadPtr;
}

DATA_READ *getDataReadAll(cl_event event)
{
	CMD_QUEUE *cmdQueue = hCmdQueueHead;
	DATA_READ *curDataReadPtr, *dataReadPtr = NULL;
	while (cmdQueue != NULL)
	{
		curDataReadPtr = cmdQueue->dataReadPtr;
		while (curDataReadPtr != NULL)
		{
			if (curDataReadPtr->event == event)
			{
				dataReadPtr = curDataReadPtr;
				break;
			}
			curDataReadPtr = curDataReadPtr->next;
		}

		if (dataReadPtr != NULL)
		{
			break;
		}

		cmdQueue = cmdQueue->next;
	}

	if (dataReadPtr == NULL)
	{
		printf("Lib, In getDataReadAll, event does not exist!\n");
		exit (1);
	}

	return dataReadPtr;
}

void releaseDataRead(cl_command_queue command_queue,
                     cl_event         event)
{
	CMD_QUEUE *cmdQueue = getCommandQueue(command_queue);
	DATA_READ *curDataReadPtr = cmdQueue->dataReadPtr;
	DATA_READ *preDataReadPtr;
	if (curDataReadPtr->event == event)
	{
		cmdQueue->dataReadPtr = curDataReadPtr->next;
		free(curDataReadPtr);

		if (cmdQueue->dataReadPtr == NULL)
		{
			cmdQueue->dataReadPtrTail = NULL;
		}

		return;
	}

	preDataReadPtr = curDataReadPtr;
	curDataReadPtr = curDataReadPtr->next;
	while (curDataReadPtr != NULL)
	{
		if (curDataReadPtr->event == event)
		{
			preDataReadPtr->next = curDataReadPtr->next;
			free(curDataReadPtr);
			if (preDataReadPtr->next = NULL)
			{
				cmdQueue->dataReadPtrTail = preDataReadPtr;
			}
			break;
		}
		preDataReadPtr = curDataReadPtr;
		curDataReadPtr = curDataReadPtr->next;
	}

	return;
}

void releaseDataReadAll(cl_event event)
{
	CMD_QUEUE *cmdQueue = hCmdQueueHead;
	DATA_READ *curDataReadPtr, *preDataReadPtr;
	while (cmdQueue != NULL)
	{
		curDataReadPtr = cmdQueue->dataReadPtr;
		if (curDataReadPtr->event == event)
		{
			cmdQueue->dataReadPtr = curDataReadPtr->next;
			free(curDataReadPtr);
			if (cmdQueue->dataReadPtr == NULL)
			{
				cmdQueue->dataReadPtrTail = NULL;
			}
			return;
		}

		preDataReadPtr = curDataReadPtr;
		curDataReadPtr = preDataReadPtr->next;
		while (curDataReadPtr != NULL)
		{
			if (curDataReadPtr->event == event)
			{
				preDataReadPtr->next = curDataReadPtr->next;
				free(curDataReadPtr);
				if (preDataReadPtr->next == NULL)
				{
					cmdQueue->dataReadPtrTail = preDataReadPtr;
				}
				return;
			}
			preDataReadPtr = curDataReadPtr;
			curDataReadPtr = preDataReadPtr->next;
		}

		cmdQueue = cmdQueue->next;
	}

	return;
}

void processEvent(cl_event event)
{
	DATA_READ *dataReadPtr;
	MPI_Status status;
	dataReadPtr = getDataReadAll(event);
	if (dataReadPtr != NULL)
	{
		MPI_Wait(&dataReadPtr->request, &status);
	}

	//release the event
	releaseDataReadAll(event);
}

void processCommandQueue(cl_command_queue command_queue)
{
	MPI_Status status;
	CMD_QUEUE *cmdQueue = getCommandQueue(command_queue);
	DATA_READ *dataReadPtr = cmdQueue->dataReadPtr;
	DATA_READ *nextDataReadPtr;
	while (dataReadPtr != NULL)
	{
		MPI_Wait(&dataReadPtr->request, &status);
		nextDataReadPtr = dataReadPtr->next;
		free(dataReadPtr);
		dataReadPtr = nextDataReadPtr;
	}
	cmdQueue->dataReadPtr = NULL;
	cmdQueue->dataReadPtrTail = NULL;

	return;
}


//--------------------------------------------------------------------------------------

void checkSlaveProc()
{
	if (slaveCreated == 0)
	{
		MPI_Init(NULL, NULL);
		MPI_Comm_spawn("/home/scxiao/workplace/trunk/lib/slave_process", MPI_ARGV_NULL, np, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &slaveComm, errCodes);
		slaveCreated = 1;
		char hostName[200];
		int  len;
		MPI_Get_processor_name(hostName, &len);
		hostName[len] = '\0';
		printf("masterHostName = %s\n", hostName);

		if (atexit(mpiFinalize) != 0)
		{
			printf("register Finalize error!\n");
			exit(1);
		}
	}
}

void mpiFinalize()
{
	MPI_Send(NULL, 0, MPI_BYTE, 0, PROGRAM_END, slaveComm);
	MPI_Comm_free(&slaveComm);
	MPI_Finalize();
}

cl_int
clGetPlatformIDs(cl_uint          num_entries,
				 cl_platform_id * platforms,
				 cl_uint *        num_platforms)
{
	MPI_Status status;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strGetPlatformIDs tmpGetPlatform;

	//initialize structure
	tmpGetPlatform.num_entries = num_entries;
	tmpGetPlatform.platforms = platforms;
	tmpGetPlatform.num_platforms = 0;
	if (num_platforms != NULL)
	{
		tmpGetPlatform.num_platforms = 1;
	}

	//send parameters to remote node
	MPI_Send(&tmpGetPlatform, sizeof(tmpGetPlatform), MPI_BYTE, 0, 
			 GET_PLATFORM_ID_FUNC, slaveComm);

	MPI_Recv(&tmpGetPlatform, sizeof(tmpGetPlatform), MPI_BYTE, 0,
			 GET_PLATFORM_ID_FUNC, slaveComm, &status);
	if (num_platforms != NULL)
	{
		*num_platforms = tmpGetPlatform.num_platforms;
	}

	if (platforms != NULL && num_entries > 0)
	{
		MPI_Recv(platforms, sizeof(cl_platform_id) * num_entries, MPI_BYTE, 0,
				 GET_PLATFORM_ID_FUNC1, slaveComm, &status);
	}
	
	return tmpGetPlatform.res;
}

/* Device APIs */
cl_int
clGetDeviceIDs(cl_platform_id   platform,
               cl_device_type   device_type,
               cl_uint          num_entries,
               cl_device_id    *devices,
               cl_uint *        num_devices)
{
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strGetDeviceIDs tmpGetDeviceIDs;
	MPI_Status status;
	
	//initialize structure
	tmpGetDeviceIDs.platform = platform;
	tmpGetDeviceIDs.device_type = device_type;
	tmpGetDeviceIDs.num_entries = num_entries;
	tmpGetDeviceIDs.devices = devices;

	//indicate num_device be NOT NULL
	tmpGetDeviceIDs.num_devices = 1;
	if (num_devices == NULL)
	{
		tmpGetDeviceIDs.num_devices = 0;
	}
	//send parameters to remote node
	MPI_Send(&tmpGetDeviceIDs, sizeof(tmpGetDeviceIDs), MPI_BYTE, 0, 
			 GET_DEVICE_ID_FUNC, slaveComm);

	MPI_Recv(&tmpGetDeviceIDs, sizeof(tmpGetDeviceIDs), MPI_BYTE, 0,
			 GET_DEVICE_ID_FUNC, slaveComm, &status);
	if (num_entries > 0 && devices != NULL)
	{
		MPI_Recv(devices, sizeof(cl_device_id) * num_entries, MPI_BYTE, 0,
			     GET_DEVICE_ID_FUNC1, slaveComm, &status);
	}
	if (num_devices != NULL)
	{
		*num_devices = tmpGetDeviceIDs.num_devices;
	}
	return tmpGetDeviceIDs.res;
}

cl_context
clCreateContext(const cl_context_properties    *properties,
                cl_uint                        num_devices,
                const cl_device_id            *devices,
                void (CL_CALLBACK * pfn_notify)(const char *, const void *, size_t, void *),
                void *                         user_data,
                cl_int *                       errcode_ret)
{
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strCreateContext tmpCreateContext;
	MPI_Status status;
	int res;

	//initialize structure
	//tmpCreateContext.properties = *properties;
	tmpCreateContext.num_devices = num_devices;
	tmpCreateContext.devices = (cl_device_id *)devices;
	tmpCreateContext.user_data = user_data;
	
	//send parameters to remote node
	res = MPI_Send(&tmpCreateContext, sizeof(tmpCreateContext), MPI_BYTE, 0, 
			 CREATE_CONTEXT_FUNC, slaveComm);

	if (devices != NULL)
	{
		MPI_Send((void *)devices, sizeof(cl_device_id) * num_devices, MPI_BYTE, 0,
				 CREATE_CONTEXT_FUNC1, slaveComm);
	}

	MPI_Recv(&tmpCreateContext, sizeof(tmpCreateContext), MPI_BYTE, 0, 
			 CREATE_CONTEXT_FUNC, slaveComm, &status);
	*errcode_ret = tmpCreateContext.errcode_ret;
	
	return tmpCreateContext.hContext;
}

/* Command Queue APIs */
cl_command_queue
clCreateCommandQueue(cl_context                     context,
                     cl_device_id                   device,
                     cl_command_queue_properties    properties,
                     cl_int *                       errcode_ret)
{
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strCreateCommandQueue tmpCreateCommandQueue;
	MPI_Status status;

	tmpCreateCommandQueue.context = context;
	tmpCreateCommandQueue.device = device;
	tmpCreateCommandQueue.properties = properties;

	//send parameters to remote node
	MPI_Send(&tmpCreateCommandQueue, sizeof(tmpCreateCommandQueue), MPI_BYTE, 0, 
			 CREATE_COMMAND_QUEUE_FUNC, slaveComm);
	MPI_Recv(&tmpCreateCommandQueue, sizeof(tmpCreateCommandQueue), MPI_BYTE, 0, 
			 CREATE_COMMAND_QUEUE_FUNC, slaveComm, &status);
	*errcode_ret = tmpCreateCommandQueue.errcode_ret;

	//create local command queue
	createCommandQueue(tmpCreateCommandQueue.clCommand);

	return tmpCreateCommandQueue.clCommand;
}

cl_program
clCreateProgramWithSource(cl_context        context,
                          cl_uint           count,
                          const char **     strings,
                          const size_t *    lengths,
                          cl_int *          errcode_ret)
{
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strCreateProgramWithSource tmpCreateProgramWithSource;
	MPI_Status status;

	//initialize structure
	tmpCreateProgramWithSource.context = context;
	tmpCreateProgramWithSource.count = count;
	size_t totalLength, *lengthsArray, strStartLoc;
	cl_uint strIndex;
	char *allStrings;

	lengthsArray = (size_t *)malloc(count * sizeof(size_t));

	totalLength = 0;
	if (lengths == NULL) //all strings are null-terminated
	{
		for (strIndex = 0; strIndex < count; strIndex++)
		{
			lengthsArray[strIndex] = strlen(strings[strIndex]);
			totalLength += lengthsArray[strIndex];
		}
	}
	else
	{
		for (strIndex = 0; strIndex < count; strIndex++)
		{
			if (lengths[strIndex] == 0)
			{
				lengthsArray[strIndex] = strlen(strings[strIndex]);
				totalLength += lengthsArray[strIndex];
			}
			else
			{
				lengthsArray[strIndex] = lengths[strIndex];
				totalLength += lengthsArray[strIndex];
			}
		}
	}
	allStrings = (char *)malloc(totalLength * sizeof(char));

	strStartLoc = 0;
	for (strIndex = 0; strIndex < count; strIndex++)
	{
		memcpy(&allStrings[strStartLoc], strings[strIndex], sizeof(char) * lengthsArray[strIndex]);
		strStartLoc += lengthsArray[strIndex];
	}

	tmpCreateProgramWithSource.lengths = totalLength;

	//send parameters to remote node
	MPI_Send(&tmpCreateProgramWithSource,
			 sizeof(tmpCreateProgramWithSource), 
			 MPI_BYTE, 0, CREATE_PROGRMA_WITH_SOURCE, 
			 slaveComm);
	MPI_Send(lengthsArray, sizeof(size_t) * count, MPI_BYTE, 0,
			 CREATE_PROGRMA_WITH_SOURCE1, slaveComm);
	MPI_Send((void *)allStrings, totalLength * sizeof(char), MPI_BYTE, 0, 
			 CREATE_PROGRMA_WITH_SOURCE2, slaveComm);
	MPI_Recv(&tmpCreateProgramWithSource,
			 sizeof(tmpCreateProgramWithSource), 
			 MPI_BYTE, 0, CREATE_PROGRMA_WITH_SOURCE, 
			 slaveComm, &status);
	if (errcode_ret != NULL)
	{
		*errcode_ret = tmpCreateProgramWithSource.errcode_ret;
	}

	free(allStrings);
	free(lengthsArray);

	return tmpCreateProgramWithSource.clProgram;
}

cl_int
clBuildProgram(cl_program           program,
               cl_uint              num_devices,
               const cl_device_id * device_list,
               const char *         options, 
               void (CL_CALLBACK *  pfn_notify)(cl_program  program, void * user_data),
               void *               user_data)
{
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	//printf("shared lib, build program, message sent!\n");
	int optionsLen = 0;
	if (options != NULL)
	{
		optionsLen = strlen(options);
	}

	struct strBuildProgram tmpBuildProgram;
	MPI_Status status;

	//initialize structure
	tmpBuildProgram.program = program;
	tmpBuildProgram.num_devices = num_devices;
	tmpBuildProgram.device_list = (cl_device_id*)device_list;
	tmpBuildProgram.optionLen = optionsLen;

	//send parameters to remote node
	MPI_Send(&tmpBuildProgram, sizeof(tmpBuildProgram), MPI_BYTE, 0, 
			 BUILD_PROGRAM, slaveComm);
	if (optionsLen > 0)
	{
		MPI_Send((void *)options, optionsLen, MPI_BYTE, 0, BUILD_PROGRAM1, slaveComm);
	}
	if (device_list != NULL)
	{
		MPI_Send((void *)device_list, sizeof(cl_device_id) * num_devices, MPI_BYTE, 0,
				 BUILD_PROGRAM, slaveComm);
	}

	MPI_Recv(&tmpBuildProgram, sizeof(tmpBuildProgram), MPI_BYTE, 0, 
			 BUILD_PROGRAM, slaveComm, &status);
	
	return tmpBuildProgram.res;
}

cl_kernel
clCreateKernel(cl_program      program,
               const char *    kernel_name,
               cl_int *        errcode_ret)
{
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	int kernelNameSize = strlen(kernel_name);
	MPI_Status status;

	struct strCreateKernel tmpCreateKernel;
	tmpCreateKernel.program = program;
	tmpCreateKernel.kernelNameSize = kernelNameSize;
	
	//send input parameters to remote node
	MPI_Send(&tmpCreateKernel, sizeof(tmpCreateKernel), MPI_BYTE, 0,
			 CREATE_KERNEL, slaveComm);
	MPI_Send((void *)kernel_name, kernelNameSize, MPI_CHAR, 0, CREATE_KERNEL1, slaveComm);
	MPI_Recv(&tmpCreateKernel, sizeof(tmpCreateKernel), MPI_BYTE, 0,
			 CREATE_KERNEL, slaveComm, &status);
	*errcode_ret = tmpCreateKernel.errcode_ret;

	//create kernel info on the local node
	createKernel(tmpCreateKernel.kernel);
	
	return tmpCreateKernel.kernel;
}

/* Memory Object APIs */
cl_mem
clCreateBuffer(cl_context   context,
               cl_mem_flags flags,
               size_t       size,
               void *       host_ptr,
               cl_int *     errcode_ret)
{
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strCreateBuffer tmpCreateBuffer;
	MPI_Status status;

	//initialize structure
	tmpCreateBuffer.context = context;
	tmpCreateBuffer.flags = flags;
	tmpCreateBuffer.size = size;
	tmpCreateBuffer.host_ptr_flag = 0;
	if (host_ptr != NULL)
	{
		tmpCreateBuffer.host_ptr_flag = 1;
	}

	//send parameters to remote node
	MPI_Send(&tmpCreateBuffer, sizeof(tmpCreateBuffer), MPI_BYTE, 0, 
			 CREATE_BUFFER_FUNC, slaveComm);
	if (tmpCreateBuffer.host_ptr_flag == 1)
	{
		MPI_Send(host_ptr, size, MPI_BYTE, 0, CREATE_BUFFER_FUNC1, slaveComm);
	}
	MPI_Recv(&tmpCreateBuffer, sizeof(tmpCreateBuffer), MPI_BYTE, 0, 
			 CREATE_BUFFER_FUNC, slaveComm, &status);
	if (errcode_ret != NULL)
	{
		*errcode_ret = tmpCreateBuffer.errcode_ret;
	}

	return tmpCreateBuffer.deviceMem;
}

cl_int
clEnqueueWriteBuffer(cl_command_queue   command_queue, 
                     cl_mem             buffer, 
                     cl_bool            blocking_write, 
                     size_t             offset, 
                     size_t             cb, 
                     const void *       ptr, 
                     cl_uint            num_events_in_wait_list, 
                     const cl_event *   event_wait_list, 
                     cl_event *         event)
{
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strEnqueueWriteBuffer tmpEnqueueWriteBuffer;
	MPI_Status status;
	MPI_Request request;
	

	//initialize structure
	tmpEnqueueWriteBuffer.command_queue = command_queue;
	tmpEnqueueWriteBuffer.buffer = buffer;
	tmpEnqueueWriteBuffer.blocking_write = blocking_write;
	tmpEnqueueWriteBuffer.offset = offset;
	tmpEnqueueWriteBuffer.cb = cb;
	tmpEnqueueWriteBuffer.num_events_in_wait_list = num_events_in_wait_list;
	if (event == NULL)
	{
		tmpEnqueueWriteBuffer.event_null_flag = 1;
	}
	else
	{
		tmpEnqueueWriteBuffer.event_null_flag = 0;
	}

	//send parameters to remote node
	MPI_Send(&tmpEnqueueWriteBuffer, sizeof(tmpEnqueueWriteBuffer), MPI_BYTE, 0, 
			 ENQUEUE_WRITE_BUFFER, slaveComm);
	
	//blocking or non-blocking data transfer
	if (blocking_write == CL_FALSE)
	{
		MPI_Isend((void *)ptr, cb, MPI_BYTE, 0, ENQUEUE_WRITE_BUFFER1, slaveComm, &request);
	}
	else
	{
		MPI_Send((void *)ptr, cb, MPI_BYTE, 0, ENQUEUE_WRITE_BUFFER1, slaveComm);
	}

	if (num_events_in_wait_list > 0)
	{
		MPI_Send((void *)event_wait_list, sizeof(cl_event) * num_events_in_wait_list, MPI_BYTE, 0,
				 ENQUEUE_WRITE_BUFFER2, slaveComm);
	}

	MPI_Recv(&tmpEnqueueWriteBuffer, sizeof(tmpEnqueueWriteBuffer), MPI_BYTE, 0, 
			 ENQUEUE_WRITE_BUFFER, slaveComm, &status);
	if (event != NULL)
	{
		*event = tmpEnqueueWriteBuffer.event;
	}

	if (blocking_write == CL_TRUE)
	{
		//for a blocking write, process all previous non-blocking ones
		processCommandQueue(command_queue);
	}

	return tmpEnqueueWriteBuffer.res;
}

//cl_int
//clSetKernelArg(cl_kernel    kernel,
//               cl_uint      arg_index,
//               size_t       arg_size,
//               const void * arg_value)
//{
//	//check whether the slave process is created. If not, create one.
//	checkSlaveProc();
//
//	struct strSetKernelArg tmpSetKernelArg;
//	MPI_Status status;
//
//	//initialize structure
//	tmpSetKernelArg.kernel = kernel;
//	tmpSetKernelArg.arg_index = arg_index;
//	tmpSetKernelArg.arg_size = arg_size;
//	tmpSetKernelArg.arg_value = arg_value;
//
//	//send parameters to remote node
//	MPI_Send(&tmpSetKernelArg, sizeof(tmpSetKernelArg), MPI_BYTE, 0, 
//			 SET_KERNEL_ARG, slaveComm);
//	if (arg_value != NULL)
//	{
//		MPI_Send((void *)arg_value, arg_size, MPI_BYTE, 0, SET_KERNEL_ARG1,
//			 	  slaveComm);
//	}
//
//	MPI_Recv(&tmpSetKernelArg, sizeof(tmpSetKernelArg), MPI_BYTE, 0, 
//			 SET_KERNEL_ARG, slaveComm, &status);
//	return tmpSetKernelArg.res;
//}

cl_int
clSetKernelArg(cl_kernel    kernel,
               cl_uint      arg_index,
               size_t       arg_size,
               const void * arg_value)
{
	kernel_info *kernelPtr = getKernelPtr(kernel);
	if (kernelPtr->args_allocated == 0)
	{
		kernelPtr->args_ptr = (kernel_args *) malloc(sizeof(kernel_args) * MAX_ARGS);
		kernelPtr->args_allocated = 1;
	}
	kernelPtr->args_ptr[kernelPtr->args_num].arg_index = arg_index;
	kernelPtr->args_ptr[kernelPtr->args_num].arg_size  = arg_size;
	kernelPtr->args_ptr[kernelPtr->args_num].arg_null_flag = 1;
	if (arg_value != NULL)
	{
		kernelPtr->args_ptr[kernelPtr->args_num].arg_null_flag = 0;
		memcpy(kernelPtr->args_ptr[kernelPtr->args_num].arg_value, arg_value, arg_size);
	}
	kernelPtr->args_num++;

//	struct strSetKernelArg tmpSetKernelArg;
//	MPI_Status status;

//	//initialize structure
//	tmpSetKernelArg.kernel = kernel;
//	tmpSetKernelArg.arg_index = arg_index;
//	tmpSetKernelArg.arg_size = arg_size;
//	tmpSetKernelArg.arg_value = arg_value;

//	//send parameters to remote node
//	MPI_Send(&tmpSetKernelArg, sizeof(tmpSetKernelArg), MPI_BYTE, 0, 
//			 SET_KERNEL_ARG, slaveComm);
//	if (arg_value != NULL)
//	{
//		MPI_Send((void *)arg_value, arg_size, MPI_BYTE, 0, SET_KERNEL_ARG1,
//			 	  slaveComm);
//	}
//
//	MPI_Recv(&tmpSetKernelArg, sizeof(tmpSetKernelArg), MPI_BYTE, 0, 
//			 SET_KERNEL_ARG, slaveComm, &status);
	return 0;
}

cl_int
clEnqueueNDRangeKernel(cl_command_queue command_queue,
                       cl_kernel        kernel,
                       cl_uint          work_dim,
                       const size_t *   global_work_offset,
                       const size_t *   global_work_size,
                       const size_t *   local_work_size,
                       cl_uint          num_events_in_wait_list,
                       const cl_event * event_wait_list,
                       cl_event *       event)
{
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strEnqueueNDRangeKernel tmpEnqueueNDRangeKernel;
	MPI_Status status;

	//initialize structure
	tmpEnqueueNDRangeKernel.command_queue = command_queue;
	tmpEnqueueNDRangeKernel.kernel = kernel;
	tmpEnqueueNDRangeKernel.work_dim = work_dim;
	tmpEnqueueNDRangeKernel.num_events_in_wait_list = num_events_in_wait_list;
	tmpEnqueueNDRangeKernel.global_work_offset_flag = 0;
	tmpEnqueueNDRangeKernel.global_work_size_flag = 0;
	tmpEnqueueNDRangeKernel.local_work_size_flag = 0;
	if (global_work_offset != NULL)
	{
		tmpEnqueueNDRangeKernel.global_work_offset_flag = 1;
	}
	if (global_work_size != NULL)
	{
		tmpEnqueueNDRangeKernel.global_work_size_flag = 1;
	}
	if (local_work_size != NULL)
	{
		tmpEnqueueNDRangeKernel.local_work_size_flag = 1;
	}
	kernel_info *kernelPtr = getKernelPtr(kernel);
	tmpEnqueueNDRangeKernel.args_num = kernelPtr->args_num;
	if (event == NULL)
	{
		tmpEnqueueNDRangeKernel.event_null_flag = 1;
	}
	else
	{
		tmpEnqueueNDRangeKernel.event_null_flag = 0;
	}
	
	//tmpEnqueueNDRangeKernel.event_wait_list = event_wait_list;
	//tmpEnqueueNDRangeKernel.event = event;

	//send parameters to remote node
	MPI_Send(&tmpEnqueueNDRangeKernel, sizeof(tmpEnqueueNDRangeKernel), MPI_BYTE, 0, 
			 ENQUEUE_ND_RANGE_KERNEL, slaveComm);

	//printf("%d, %d, %d\n",
	//		tmpEnqueueNDRangeKernel.global_work_offset_flag,
	//		tmpEnqueueNDRangeKernel.global_work_size_flag,
	//		tmpEnqueueNDRangeKernel.local_work_size_flag);

	if (tmpEnqueueNDRangeKernel.global_work_offset_flag == 1)
	{
		MPI_Send((void *)global_work_offset, sizeof(size_t) * work_dim, MPI_BYTE, 0,
				 ENQUEUE_ND_RANGE_KERNEL1, slaveComm);
	}

	if (tmpEnqueueNDRangeKernel.global_work_size_flag == 1)
	{
		MPI_Send((void *)global_work_size, sizeof(size_t) * work_dim, MPI_BYTE, 0,
				 ENQUEUE_ND_RANGE_KERNEL2, slaveComm);
	}

	if (tmpEnqueueNDRangeKernel.local_work_size_flag == 1)
	{
		MPI_Send((void *)local_work_size, sizeof(size_t) * work_dim, MPI_BYTE, 0,
				 ENQUEUE_ND_RANGE_KERNEL3, slaveComm);
	}

	if (kernelPtr->args_num > 0)
	{
		MPI_Send((void *)kernelPtr->args_ptr, sizeof(kernel_args) * kernelPtr->args_num, MPI_BYTE, 0,
				 ENQUEUE_ND_RANGE_KERNEL4, slaveComm);
	}
	//arguments for current call are processed
	kernelPtr->args_num = 0;

	MPI_Recv(&tmpEnqueueNDRangeKernel, sizeof(tmpEnqueueNDRangeKernel), MPI_BYTE, 0, 
			 ENQUEUE_ND_RANGE_KERNEL, slaveComm, &status);
	
	if (event != NULL)
	{
		*event = tmpEnqueueNDRangeKernel.event;
	}
	
	return tmpEnqueueNDRangeKernel.res;
}

/* Enqueued Commands APIs */
cl_int
clEnqueueReadBuffer(cl_command_queue    command_queue,
                    cl_mem              buffer,
                    cl_bool             blocking_read,
                    size_t              offset,
                    size_t              cb,
                    void *              ptr,
                    cl_uint             num_events_in_wait_list,
                    const cl_event *    event_wait_list,
                    cl_event *          event)
{
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	
	MPI_Request request;
	MPI_Status  status;
	struct strEnqueueReadBuffer tmpEnqueueReadBuffer;


	//initialize structure
	tmpEnqueueReadBuffer.command_queue = command_queue;
	tmpEnqueueReadBuffer.buffer = buffer;
	tmpEnqueueReadBuffer.blocking_read = blocking_read;
	tmpEnqueueReadBuffer.readBufferTag = readBufferTag;
	tmpEnqueueReadBuffer.offset = offset;
	tmpEnqueueReadBuffer.cb = cb;
	tmpEnqueueReadBuffer.num_events_in_wait_list = num_events_in_wait_list;
	if (event == NULL)
	{
		tmpEnqueueReadBuffer.event_null_flag = 1;
	}
	else
	{
		tmpEnqueueReadBuffer.event_null_flag = 0;
	}

	//send parameters to remote node
	MPI_Send(&tmpEnqueueReadBuffer, sizeof(tmpEnqueueReadBuffer), MPI_BYTE, 0, 
			 ENQUEUE_READ_BUFFER, slaveComm);
	if (num_events_in_wait_list > 0)
	{
		MPI_Send((void *)event_wait_list, sizeof(cl_event) * num_events_in_wait_list, MPI_BYTE, 0,
				 ENQUEUE_READ_BUFFER1, slaveComm);
	}

	MPI_Recv(&tmpEnqueueReadBuffer, sizeof(tmpEnqueueReadBuffer), MPI_BYTE, 0, 
			 ENQUEUE_READ_BUFFER, slaveComm, &status);
	if (blocking_read == CL_TRUE)
	{
		MPI_Recv(ptr, cb, MPI_CHAR, 0, ENQUEUE_READ_BUFFER1, slaveComm, &status);
		//for a blocking read, process all previous non-blocking ones
		processCommandQueue(command_queue);
	}
	else //non blocking read
	{
		MPI_Irecv(ptr, cb, MPI_CHAR, 0, readBufferTag, slaveComm, &request);
		if (++readBufferTag > MAX_TAG)
		{
			readBufferTag == PROGRAM_END + 1;
		}

		DATA_READ *dataReadPtr = createDataRead(command_queue,
									tmpEnqueueReadBuffer.event);
		dataReadPtr->request = request;
	}

	if (event != NULL)
	{
		*event = tmpEnqueueReadBuffer.event;
	}
	
	return tmpEnqueueReadBuffer.res;
}

cl_int
clReleaseMemObject(cl_mem memobj)
{
	MPI_Status status;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	
	struct strReleaseMemObject tmpReleaseMemObject;
	tmpReleaseMemObject.memobj = memobj;

	MPI_Send(&tmpReleaseMemObject, sizeof(tmpReleaseMemObject), MPI_BYTE, 
			 0, RELEASE_MEM_OBJ, slaveComm);
	MPI_Recv(&tmpReleaseMemObject, sizeof(tmpReleaseMemObject), MPI_BYTE, 
			 0, RELEASE_MEM_OBJ, slaveComm, &status);

	return tmpReleaseMemObject.res;
}

cl_int
clReleaseKernel(cl_kernel kernel)
{
	MPI_Status status;
	//release kernel and parameter buffers related
	//to the kernel;
	releaseKernelPtr(kernel);
	
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	//release kernel on the remote node
	struct strReleaseKernel tmpReleaseKernel;
	tmpReleaseKernel.kernel = kernel;
	MPI_Send(&tmpReleaseKernel, sizeof(tmpReleaseKernel), MPI_BYTE,
			 0, CL_RELEASE_KERNEL_FUNC, slaveComm);
	MPI_Recv(&tmpReleaseKernel, sizeof(tmpReleaseKernel), MPI_BYTE,
			 0, CL_RELEASE_KERNEL_FUNC, slaveComm, &status);
	return tmpReleaseKernel.res;
}

cl_int
clFinish(cl_command_queue hInCmdQueue)
{
	MPI_Status status;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	
	struct strFinish tmpFinish;
	tmpFinish.command_queue = hInCmdQueue;
	MPI_Send(&tmpFinish, sizeof(tmpFinish), MPI_BYTE, 0,
			 FINISH_FUNC, slaveComm);
	MPI_Recv(&tmpFinish, sizeof(tmpFinish), MPI_BYTE, 0,
			 FINISH_FUNC, slaveComm, &status);

	//preocess the local command queue
	processCommandQueue(hInCmdQueue);
	return tmpFinish.res;
}

//16
cl_int
clGetContextInfo(cl_context         context, 
                 cl_context_info    param_name, 
                 size_t             param_value_size, 
                 void *             param_value, 
                 size_t *           param_value_size_ret)
{
	MPI_Status status;
	struct strGetContextInfo tmpGetContextInfo;
	tmpGetContextInfo.context = context;
	tmpGetContextInfo.param_name = param_name;
	tmpGetContextInfo.param_value_size = param_value_size;
	tmpGetContextInfo.param_value = param_value;
	tmpGetContextInfo.param_value_size_ret = 1;
	if (param_value_size_ret == NULL)
	{
		tmpGetContextInfo.param_value_size_ret = 0;
	}

	MPI_Send(&tmpGetContextInfo, sizeof(tmpGetContextInfo), MPI_BYTE, 0,
			 GET_CONTEXT_INFO_FUNC, slaveComm);
	MPI_Recv(&tmpGetContextInfo, sizeof(tmpGetContextInfo), MPI_BYTE, 0,
			 GET_CONTEXT_INFO_FUNC, slaveComm, &status);

	if (param_value != NULL)
	{
		MPI_Recv(param_value, param_value_size, MPI_BYTE, 0,
				 GET_CONTEXT_INFO_FUNC1, slaveComm, &status);
	}

	if (param_value_size_ret != NULL)
	{
		*param_value_size_ret = tmpGetContextInfo.param_value_size_ret;
	}

	return tmpGetContextInfo.res;
}

//17
cl_int
clGetProgramBuildInfo(cl_program            program,
                      cl_device_id          device,
                      cl_program_build_info param_name,
                      size_t                param_value_size,
                      void *                param_value,
                      size_t *              param_value_size_ret)
{
	MPI_Status status;
	struct strGetProgramBuildInfo tmpGetProgramBuildInfo;
	tmpGetProgramBuildInfo.program = program;
	tmpGetProgramBuildInfo.device  = device;
	tmpGetProgramBuildInfo.param_name = param_name;
	tmpGetProgramBuildInfo.param_value_size = param_value_size;
	tmpGetProgramBuildInfo.param_value = param_value;
	tmpGetProgramBuildInfo.param_value_size_ret = 1;
	if (param_value_size_ret == NULL)
	{
		tmpGetProgramBuildInfo.param_value_size_ret = 0;
	}

	MPI_Send(&tmpGetProgramBuildInfo, sizeof(tmpGetProgramBuildInfo), MPI_BYTE, 0,
			 GET_BUILD_INFO_FUNC, slaveComm);
	MPI_Recv(&tmpGetProgramBuildInfo, sizeof(tmpGetProgramBuildInfo), MPI_BYTE, 0,
			 GET_BUILD_INFO_FUNC, slaveComm, &status);

	if (param_value != NULL)
	{
		MPI_Recv(param_value, param_value_size, MPI_BYTE, 0,
				 GET_BUILD_INFO_FUNC1, slaveComm, &status);
	}

	if (param_value_size_ret != NULL)
	{
		*param_value_size_ret = tmpGetProgramBuildInfo.param_value_size_ret;
	}
	return tmpGetProgramBuildInfo.res;
}

//18
cl_int
clGetProgramInfo(cl_program         program,
                 cl_program_info    param_name,
                 size_t             param_value_size,
                 void *             param_value,
                 size_t *           param_value_size_ret)
{
	MPI_Status status;
	struct strGetProgramInfo tmpGetProgramInfo;
	tmpGetProgramInfo.program = program;
	tmpGetProgramInfo.param_name = param_name;
	tmpGetProgramInfo.param_value_size = param_value_size;
	tmpGetProgramInfo.param_value = param_value;
	tmpGetProgramInfo.param_value_size_ret = 1;
	if (param_value_size_ret == NULL)
	{
		tmpGetProgramInfo.param_value_size_ret = 0;
	}

	MPI_Send(&tmpGetProgramInfo, sizeof(tmpGetProgramInfo), MPI_BYTE, 0,
			 GET_PROGRAM_INFO_FUNC, slaveComm);
	MPI_Recv(&tmpGetProgramInfo, sizeof(tmpGetProgramInfo), MPI_BYTE, 0,
			 GET_PROGRAM_INFO_FUNC, slaveComm, &status);

	if (param_value != NULL)
	{
		MPI_Recv(param_value, param_value_size, MPI_BYTE, 0,
				 GET_PROGRAM_INFO_FUNC1, slaveComm, &status);
	}

	if (param_value_size_ret != NULL)
	{
		*param_value_size_ret = tmpGetProgramInfo.param_value_size_ret;
	}

	return tmpGetProgramInfo.res;
}

//19
cl_int
clReleaseProgram(cl_program program)
{
	MPI_Status status;
	struct strReleaseProgram tmpReleaseProgram;
	tmpReleaseProgram.program = program;
	MPI_Send(&tmpReleaseProgram, sizeof(tmpReleaseProgram), MPI_BYTE, 0,
			 REL_PROGRAM_FUNC, slaveComm);
	MPI_Recv(&tmpReleaseProgram, sizeof(tmpReleaseProgram), MPI_BYTE, 0,
			 REL_PROGRAM_FUNC, slaveComm, &status);
	return tmpReleaseProgram.res;
}

//20
cl_int
clReleaseCommandQueue(cl_command_queue command_queue)
{
	MPI_Status status;
	struct strReleaseCommandQueue tmpReleaseCommandQueue;
	tmpReleaseCommandQueue.command_queue = command_queue;

	//release the local command queue
	releaseCommandQueue(command_queue);

	MPI_Send(&tmpReleaseCommandQueue, sizeof(tmpReleaseCommandQueue), MPI_BYTE, 0,
			 REL_COMMAND_QUEUE_FUNC, slaveComm);
	MPI_Recv(&tmpReleaseCommandQueue, sizeof(tmpReleaseCommandQueue), MPI_BYTE, 0,
			 REL_COMMAND_QUEUE_FUNC, slaveComm, &status);
	return tmpReleaseCommandQueue.res;
}

//21
cl_int
clReleaseContext(cl_context context)
{
	MPI_Status status;
	struct strReleaseContext tmpReleaseContext;
	tmpReleaseContext.context = context;
	MPI_Send(&tmpReleaseContext, sizeof(tmpReleaseContext), MPI_BYTE, 0,
			 REL_CONTEXT_FUNC, slaveComm);
	MPI_Recv(&tmpReleaseContext, sizeof(tmpReleaseContext), MPI_BYTE, 0,
			 REL_CONTEXT_FUNC, slaveComm, &status);
	return tmpReleaseContext.res;
}

//22
cl_int
clGetDeviceInfo(cl_device_id    device,
                cl_device_info  param_name, 
                size_t          param_value_size, 
                void *          param_value,
                size_t *        param_value_size_ret)
{
	MPI_Status status;
	struct strGetDeviceInfo tmpGetDeviceInfo;
	tmpGetDeviceInfo.device = device;
	tmpGetDeviceInfo.param_name = param_name;
	tmpGetDeviceInfo.param_value_size = param_value_size;
	tmpGetDeviceInfo.param_value = param_value;
	tmpGetDeviceInfo.param_value_size_ret = 1;
	if (param_value_size_ret == NULL)
	{
		tmpGetDeviceInfo.param_value_size_ret = 0;
	}

	MPI_Send(&tmpGetDeviceInfo, sizeof(tmpGetDeviceInfo), MPI_BYTE, 0,
			 GET_DEVICE_INFO_FUNC, slaveComm);
	MPI_Recv(&tmpGetDeviceInfo, sizeof(tmpGetDeviceInfo), MPI_BYTE, 0,
			 GET_DEVICE_INFO_FUNC, slaveComm, &status);

	if (param_value != NULL)
	{
		MPI_Recv(param_value, param_value_size, MPI_BYTE, 0,
				 GET_DEVICE_INFO_FUNC1, slaveComm, &status);
	}

	if (param_value_size_ret != NULL)
	{
		*param_value_size_ret = tmpGetDeviceInfo.param_value_size_ret;
	}

	return tmpGetDeviceInfo.res;
}

//23
cl_int
clGetPlatformInfo(cl_platform_id    platform,
                  cl_platform_info  param_name, 
                  size_t            param_value_size, 
                  void *            param_value,
                  size_t *          param_value_size_ret)
{
	MPI_Status status;
	struct strGetPlatformInfo tmpGetPlatformInfo;
	tmpGetPlatformInfo.platform = platform;
	tmpGetPlatformInfo.param_name = param_name;
	tmpGetPlatformInfo.param_value_size = param_value_size;
	tmpGetPlatformInfo.param_value = param_value;
	tmpGetPlatformInfo.param_value_size_ret = 1;
	if (param_value_size_ret == NULL)
	{
		tmpGetPlatformInfo.param_value_size_ret = 0;
	}

	MPI_Send(&tmpGetPlatformInfo, sizeof(tmpGetPlatformInfo), MPI_BYTE, 0,
			 GET_PLATFORM_INFO_FUNC, slaveComm);
	MPI_Recv(&tmpGetPlatformInfo, sizeof(tmpGetPlatformInfo), MPI_BYTE, 0,
			 GET_PLATFORM_INFO_FUNC, slaveComm, &status);

	if (param_value != NULL)
	{
		MPI_Recv(param_value, param_value_size, MPI_BYTE, 0,
				 GET_PLATFORM_INFO_FUNC1, slaveComm, &status);
	}

	if (param_value_size_ret != NULL)
	{
		*param_value_size_ret = tmpGetPlatformInfo.param_value_size_ret;
	}

	return tmpGetPlatformInfo.res;
}

//24
cl_int
clFlush(cl_command_queue hInCmdQueue)
{
	MPI_Status status;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	
	struct strFlush tmpFlush;
	tmpFlush.command_queue = hInCmdQueue;
	MPI_Send(&tmpFlush, sizeof(tmpFlush), MPI_BYTE, 0,
			 FLUSH_FUNC, slaveComm);
	MPI_Recv(&tmpFlush, sizeof(tmpFlush), MPI_BYTE, 0,
			 FLUSH_FUNC, slaveComm, &status);
	return tmpFlush.res;
}

//25
cl_int
clWaitForEvents(cl_uint           num_events,
				const cl_event   *event_list)
{
	MPI_Status status;
	int i;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	
	struct strWaitForEvents tmpWaitForEvents;
	tmpWaitForEvents.num_events = num_events;
	MPI_Send(&tmpWaitForEvents, sizeof(tmpWaitForEvents), MPI_BYTE, 0,
			 WAIT_FOR_EVENT_FUNC, slaveComm);
	MPI_Send((void *)event_list, sizeof(cl_event) * num_events, MPI_BYTE, 0,
			 WAIT_FOR_EVENT_FUNC1, slaveComm);
	MPI_Recv(&tmpWaitForEvents, sizeof(tmpWaitForEvents), MPI_BYTE, 0,
			 WAIT_FOR_EVENT_FUNC, slaveComm, &status);
	for (i = 0; i < num_events; i++)
	{
		processEvent(event_list[i]);
	}

	return tmpWaitForEvents.res;
}

//26
cl_sampler
clCreateSampler(cl_context          context,
				cl_bool             normalized_coords,
				cl_addressing_mode  addressing_mode,
				cl_filter_mode      filter_mode,
				cl_int 			   *errcode_ret)
{
	MPI_Status status;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	
	struct strCreateSampler tmpCreateSampler;
	tmpCreateSampler.context = context;
	tmpCreateSampler.normalized_coords = normalized_coords;
	tmpCreateSampler.addressing_mode = addressing_mode;
	tmpCreateSampler.filter_mode = filter_mode;
	tmpCreateSampler.errcode_ret = 0;
	if (errcode_ret != NULL)
	{
		tmpCreateSampler.errcode_ret = 1;
	}

	MPI_Send(&tmpCreateSampler, sizeof(tmpCreateSampler), MPI_BYTE, 0,
			 CREATE_SAMPLER_FUNC, slaveComm);
	MPI_Recv(&tmpCreateSampler, sizeof(tmpCreateSampler), MPI_BYTE, 0,
			 CREATE_SAMPLER_FUNC, slaveComm, &status);
	
	if (errcode_ret != NULL)
	{
		*errcode_ret = tmpCreateSampler.errcode_ret;
	}

	return tmpCreateSampler.sampler;
}

//27
cl_int
clGetCommandQueueInfo(cl_command_queue       command_queue,
                      cl_command_queue_info  param_name, 
                      size_t                 param_value_size, 
                      void *                 param_value,
                      size_t *               param_value_size_ret)
{
	MPI_Status status;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	struct strGetCommandQueueInfo tmpGetCommandQueueInfo;
	tmpGetCommandQueueInfo.command_queue = command_queue;
	tmpGetCommandQueueInfo.param_name = param_name;
	tmpGetCommandQueueInfo.param_value_size = param_value_size;
	tmpGetCommandQueueInfo.param_value = param_value;
	tmpGetCommandQueueInfo.param_value_size_ret = 1;
	if (param_value_size_ret == NULL)
	{
		tmpGetCommandQueueInfo.param_value_size_ret = 0;
	}

	MPI_Send(&tmpGetCommandQueueInfo, sizeof(tmpGetCommandQueueInfo), MPI_BYTE, 0,
			 GET_CMD_QUEUE_INFO_FUNC, slaveComm);
	MPI_Recv(&tmpGetCommandQueueInfo, sizeof(tmpGetCommandQueueInfo), MPI_BYTE, 0,
			 GET_CMD_QUEUE_INFO_FUNC, slaveComm, &status);

	if (param_value != NULL)
	{
		MPI_Recv(param_value, param_value_size, MPI_BYTE, 0,
				 GET_CMD_QUEUE_INFO_FUNC1, slaveComm, &status);
	}

	if (param_value_size_ret != NULL)
	{
		*param_value_size_ret = tmpGetCommandQueueInfo.param_value_size_ret;
	}

	return tmpGetCommandQueueInfo.res;
}

//28
void *
clEnqueueMapBuffer(cl_command_queue command_queue,
                   cl_mem           buffer,
                   cl_bool          blocking_map, 
                   cl_map_flags     map_flags,
                   size_t           offset,
                   size_t           cb,
                   cl_uint          num_events_in_wait_list,
                   const cl_event * event_wait_list,
                   cl_event *       event,
                   cl_int *         errcode_ret)
{
	MPI_Status status;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	struct strEnqueueMapBuffer tmpEnqueueMapBuffer;
	tmpEnqueueMapBuffer.command_queue = command_queue;
	tmpEnqueueMapBuffer.buffer = buffer;
	tmpEnqueueMapBuffer.blocking_map = blocking_map;
	tmpEnqueueMapBuffer.map_flags = map_flags;
	tmpEnqueueMapBuffer.offset = offset;
	tmpEnqueueMapBuffer.cb = cb;
	tmpEnqueueMapBuffer.num_events_in_wait_list = num_events_in_wait_list;
	if (event == NULL)
	{
		tmpEnqueueMapBuffer.event_null_flag = 1;
	}
	else
	{ 
		tmpEnqueueMapBuffer.event_null_flag = 0;
	}

	//0, NOT NULL, 1: NULL
	tmpEnqueueMapBuffer.errcode_ret = 0;
	if (errcode_ret == NULL)
	{
		tmpEnqueueMapBuffer.errcode_ret = 1;
	}
	MPI_Send(&tmpEnqueueMapBuffer, sizeof(tmpEnqueueMapBuffer), MPI_BYTE, 0,
			 ENQUEUE_MAP_BUFF_FUNC, slaveComm);
	if (num_events_in_wait_list > 0)
	{
		MPI_Send((void *)event_wait_list, sizeof(cl_event) * num_events_in_wait_list, MPI_BYTE, 0,
				 ENQUEUE_MAP_BUFF_FUNC1, slaveComm);
	}
	MPI_Recv(&tmpEnqueueMapBuffer, sizeof(tmpEnqueueMapBuffer), MPI_BYTE, 0,
			 ENQUEUE_MAP_BUFF_FUNC, slaveComm, &status);
	if (event != NULL)
	{
		*event = tmpEnqueueMapBuffer.event;
	}

	if (errcode_ret != NULL)
	{
		*errcode_ret = tmpEnqueueMapBuffer.errcode_ret;
	}

	return tmpEnqueueMapBuffer.ret_ptr;
}

//29
cl_int
clReleaseEvent(cl_event event)
{
	MPI_Status status;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	struct strReleaseEvent tmpReleaseEvent;
	tmpReleaseEvent.event = event;
	MPI_Send(&tmpReleaseEvent, sizeof(tmpReleaseEvent), MPI_BYTE, 0,
			 RELEASE_EVENT_FUNC, slaveComm);
	MPI_Recv(&tmpReleaseEvent, sizeof(tmpReleaseEvent), MPI_BYTE, 0,
			 RELEASE_EVENT_FUNC, slaveComm, &status);
	return tmpReleaseEvent.res;
}

//30
cl_int
clGetEventProfilingInfo(cl_event           event,
                        cl_profiling_info  param_name, 
                        size_t             param_value_size, 
                        void *             param_value,
                        size_t *           param_value_size_ret)
{
	MPI_Status status;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	struct strGetEventProfilingInfo tmpGetEventProfilingInfo;
	tmpGetEventProfilingInfo.event = event;
	tmpGetEventProfilingInfo.param_name = param_name;
	tmpGetEventProfilingInfo.param_value_size = param_value_size;
	tmpGetEventProfilingInfo.param_value = param_value;
	tmpGetEventProfilingInfo.param_value_size_ret = 1;
	if (param_value_size_ret == NULL)
	{
		tmpGetEventProfilingInfo.param_value_size_ret = 0;
	}

	MPI_Send(&tmpGetEventProfilingInfo, sizeof(tmpGetEventProfilingInfo), MPI_BYTE, 0,
			 GET_EVENT_PROF_INFO_FUNC, slaveComm);
	MPI_Recv(&tmpGetEventProfilingInfo, sizeof(tmpGetEventProfilingInfo), MPI_BYTE, 0,
			 GET_EVENT_PROF_INFO_FUNC, slaveComm, &status);

	if (param_value != NULL)
	{
		MPI_Recv(param_value, param_value_size, MPI_BYTE, 0,
				 GET_EVENT_PROF_INFO_FUNC1, slaveComm, &status);
	}

	if (param_value_size_ret != NULL)
	{
		*param_value_size_ret = tmpGetEventProfilingInfo.param_value_size_ret;
	}

	return tmpGetEventProfilingInfo.res;
}

//31
cl_int
clReleaseSampler(cl_sampler sampler)
{
	MPI_Status status;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	struct strReleaseSampler tmpReleaseSampler;
	tmpReleaseSampler.sampler = sampler;
	MPI_Send(&tmpReleaseSampler, sizeof(tmpReleaseSampler), MPI_BYTE, 0,
			 RELEASE_SAMPLER_FUNC, slaveComm);
	MPI_Recv(&tmpReleaseSampler, sizeof(tmpReleaseSampler), MPI_BYTE, 0,
			 RELEASE_SAMPLER_FUNC, slaveComm, &status);
	return tmpReleaseSampler.res;
}

//32
cl_int
clGetKernelWorkGroupInfo(cl_kernel                  kernel,
                         cl_device_id               device,
						 cl_kernel_work_group_info  param_name,
						 size_t                     param_value_size,
						 void *                     param_value,
						 size_t *                   param_value_size_ret)
{
	MPI_Status status;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	struct strGetKernelWorkGroupInfo tmpGetKernelWorkGroupInfo;
	tmpGetKernelWorkGroupInfo.kernel = kernel;
	tmpGetKernelWorkGroupInfo.device = device;
	tmpGetKernelWorkGroupInfo.param_name = param_name;
	tmpGetKernelWorkGroupInfo.param_value_size = param_value_size;
	tmpGetKernelWorkGroupInfo.param_value = param_value;
	tmpGetKernelWorkGroupInfo.param_value_size_ret = 1;
	if (param_value_size_ret == NULL)
	{
		tmpGetKernelWorkGroupInfo.param_value_size_ret = 0;
	}

	MPI_Send(&tmpGetKernelWorkGroupInfo, sizeof(tmpGetKernelWorkGroupInfo), MPI_BYTE, 0,
			 GET_KERNEL_WGP_INFO_FUNC, slaveComm);
	MPI_Recv(&tmpGetKernelWorkGroupInfo, sizeof(tmpGetKernelWorkGroupInfo), MPI_BYTE, 0,
			 GET_KERNEL_WGP_INFO_FUNC, slaveComm, &status);

	if (param_value != NULL)
	{
		MPI_Recv(param_value, param_value_size, MPI_BYTE, 0,
				 GET_KERNEL_WGP_INFO_FUNC1, slaveComm, &status);
	}

	if (param_value_size_ret != NULL)
	{
		*param_value_size_ret = tmpGetKernelWorkGroupInfo.param_value_size_ret;
	}

	return tmpGetKernelWorkGroupInfo.res;
}

//33
cl_mem
clCreateImage2D(cl_context              context,
                cl_mem_flags            flags,                                
                const cl_image_format * image_format,                                             
                size_t                  image_width,
                size_t                  image_height,
                size_t                  image_row_pitch,
                void *                  host_ptr, 
				cl_int *                errcode_ret)
{
	MPI_Status status;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	struct strCreateImage2D tmpCreateImage2D;
	tmpCreateImage2D.context = context;
	tmpCreateImage2D.flags   = flags;
	tmpCreateImage2D.img_format.image_channel_order = image_format->image_channel_order;
	tmpCreateImage2D.img_format.image_channel_data_type = image_format->image_channel_data_type;
	tmpCreateImage2D.image_width = image_width;
	tmpCreateImage2D.image_height = image_height;
	tmpCreateImage2D.image_row_pitch = image_row_pitch;
	tmpCreateImage2D.host_buff_size = 0;
	if (host_ptr != NULL)
	{
		if (image_row_pitch == 0)
		{
			tmpCreateImage2D.host_buff_size = image_width * sizeof(cl_image_format) * image_height * 2;
		}
		else
		{
			tmpCreateImage2D.host_buff_size = image_row_pitch * image_height * 2;
		}
	}
	//default errcode 
	tmpCreateImage2D.errcode_ret = 0;
	if (errcode_ret == NULL)
	{
		tmpCreateImage2D.errcode_ret = 1;
	}
	MPI_Send(&tmpCreateImage2D, sizeof(tmpCreateImage2D), MPI_BYTE, 0,
			 CREATE_IMAGE_2D_FUNC, slaveComm);
	if (host_ptr != NULL)
	{
		MPI_Send(host_ptr, tmpCreateImage2D.host_buff_size, MPI_BYTE, 0,
				 CREATE_IMAGE_2D_FUNC1, slaveComm);
	}
	MPI_Recv(&tmpCreateImage2D, sizeof(tmpCreateImage2D), MPI_BYTE, 0,
			 CREATE_IMAGE_2D_FUNC, slaveComm, &status);
	if (errcode_ret != NULL)
	{
		*errcode_ret = tmpCreateImage2D.errcode_ret;
	}

	return tmpCreateImage2D.mem_obj;
}

//34
cl_int
clEnqueueCopyBuffer(cl_command_queue    command_queue,
                    cl_mem              src_buffer,                                       
                    cl_mem              dst_buffer,                                                           
                    size_t              src_offset,                                                                               
                    size_t              dst_offset,
                    size_t              cb, 
                    cl_uint             num_events_in_wait_list,
                    const cl_event *    event_wait_list,
                    cl_event *          event)
{
	MPI_Status status;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strEnqueueCopyBuffer tmpEnqueueCopyBuffer;
	tmpEnqueueCopyBuffer.command_queue = command_queue;
	tmpEnqueueCopyBuffer.src_buffer = src_buffer;
	tmpEnqueueCopyBuffer.dst_buffer = dst_buffer;
	tmpEnqueueCopyBuffer.src_offset = src_offset;
	tmpEnqueueCopyBuffer.dst_offset = dst_offset;
	tmpEnqueueCopyBuffer.cb = cb;
	tmpEnqueueCopyBuffer.num_events_in_wait_list = num_events_in_wait_list;
	tmpEnqueueCopyBuffer.event_null_flag = 0;
	if (event == NULL)
	{
		tmpEnqueueCopyBuffer.event_null_flag = 1;
	}

	MPI_Send(&tmpEnqueueCopyBuffer, sizeof(tmpEnqueueCopyBuffer), MPI_BYTE, 0,
			 ENQ_COPY_BUFF_FUNC, slaveComm);
	if (num_events_in_wait_list > 0)
	{
		MPI_Send((void *)event_wait_list, sizeof(cl_event) * num_events_in_wait_list, MPI_BYTE, 0,
				 ENQ_COPY_BUFF_FUNC1, slaveComm);
	}
	MPI_Recv(&tmpEnqueueCopyBuffer, sizeof(tmpEnqueueCopyBuffer), MPI_BYTE, 0,
			 ENQ_COPY_BUFF_FUNC, slaveComm, &status);

	if (event != NULL)
	{
		*event = tmpEnqueueCopyBuffer.event;
	}
	
	return tmpEnqueueCopyBuffer.res;
}

//35
cl_int
clRetainEvent(cl_event event)
{
	MPI_Status status;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strRetainEvent tmpRetainEvent;
	tmpRetainEvent.event = event;
	MPI_Send(&tmpRetainEvent, sizeof(tmpRetainEvent), MPI_BYTE, 0,
			 RETAIN_EVENT_FUNC, slaveComm);
	MPI_Recv(&tmpRetainEvent, sizeof(tmpRetainEvent), MPI_BYTE, 0,
			 RETAIN_EVENT_FUNC, slaveComm, &status);
	return tmpRetainEvent.res;
}

//36
cl_int
clRetainMemObject(cl_mem memobj)
{
	MPI_Status status;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strRetainMemObject tmpRetainMemObject;
	tmpRetainMemObject.memobj = memobj;
	MPI_Send(&tmpRetainMemObject, sizeof(tmpRetainMemObject), MPI_BYTE, 0,
			 RETAIN_MEMOBJ_FUNC, slaveComm);
	MPI_Recv(&tmpRetainMemObject, sizeof(tmpRetainMemObject), MPI_BYTE, 0,
			 RETAIN_MEMOBJ_FUNC, slaveComm, &status);
	return tmpRetainMemObject.res;
}

//37
cl_int
clRetainKernel(cl_kernel kernel)
{
	MPI_Status status;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strRetainKernel tmpRetainKernel;
	tmpRetainKernel.kernel = kernel;
	MPI_Send(&tmpRetainKernel, sizeof(tmpRetainKernel), MPI_BYTE, 0,
			 RETAIN_KERNEL_FUNC, slaveComm);
	MPI_Recv(&tmpRetainKernel, sizeof(tmpRetainKernel), MPI_BYTE, 0,
			 RETAIN_KERNEL_FUNC, slaveComm, &status);
	return tmpRetainKernel.res;
}

//38
cl_int
clRetainCommandQueue(cl_command_queue command_queue)
{
	MPI_Status status;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strRetainCommandQueue tmpRetainCommandQueue;
	tmpRetainCommandQueue.command_queue = command_queue;
	MPI_Send(&tmpRetainCommandQueue, sizeof(tmpRetainCommandQueue), MPI_BYTE, 0,
			 RETAIN_CMDQUE_FUNC, slaveComm);
	MPI_Recv(&tmpRetainCommandQueue, sizeof(tmpRetainCommandQueue), MPI_BYTE, 0,
			 RETAIN_CMDQUE_FUNC, slaveComm, &status);
	return tmpRetainCommandQueue.res;
}

//39
cl_int
clEnqueueUnmapMemObject(cl_command_queue command_queue,
                        cl_mem           memobj,
                        void *           mapped_ptr,
                        cl_uint          num_events_in_wait_list,
                        const cl_event * event_wait_list,
						cl_event *       event)
{
	MPI_Status status;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	
	struct strEnqueueUnmapMemObject tmpEnqueueUnmapMemObject;
	tmpEnqueueUnmapMemObject.command_queue = command_queue;
	tmpEnqueueUnmapMemObject.memobj = memobj;
	tmpEnqueueUnmapMemObject.mapped_ptr = mapped_ptr;
	tmpEnqueueUnmapMemObject.num_events_in_wait_list = num_events_in_wait_list;
	tmpEnqueueUnmapMemObject.event_null_flag = 0;
	if (event == NULL)
	{
		tmpEnqueueUnmapMemObject.event_null_flag = 1;
	}
	MPI_Send(&tmpEnqueueUnmapMemObject, sizeof(tmpEnqueueUnmapMemObject), MPI_BYTE, 0,
			 ENQ_UNMAP_MEMOBJ_FUNC, slaveComm);
	if (num_events_in_wait_list > 0)
	{
		MPI_Send((void *)event_wait_list, sizeof(cl_event) * num_events_in_wait_list, MPI_BYTE, 0,
				 ENQ_UNMAP_MEMOBJ_FUNC1, slaveComm);
	}
	MPI_Recv(&tmpEnqueueUnmapMemObject, sizeof(tmpEnqueueUnmapMemObject), MPI_BYTE, 0,
			 ENQ_UNMAP_MEMOBJ_FUNC, slaveComm, &status);
	if (event != NULL)
	{
		*event = tmpEnqueueUnmapMemObject.event;
	}

	return tmpEnqueueUnmapMemObject.res;
}

