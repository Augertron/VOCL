#include <stdio.h>
#include <string.h>
#include "gpuv.h"

/****************************************************************************
 * 4.   This is the first working version, with all functions in matrixMul and S-W are mapped
 ****************************************************************************/
int slaveComm;
int slaveCreated = 0;
int np = 1;
int errCodes[MAX_NPS];

void checkSlaveProc()
{
	if (slaveCreated == 0)
	{
		MPI_Init(NULL, NULL);
		MPI_Comm_spawn("/home/scxiao/workplace/anl/gpuvirtualization/slave_process", MPI_ARGV_NULL, np, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &slaveComm, errCodes);
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

	//debug===================================
	//printf("In get platform ID function, shared library\n");
	//========================================

	struct strGetPlatformIDs tmpGetPlatform;

	//initialize structure
	tmpGetPlatform.num_entries = num_entries;
	//tmpGetPlatform.platforms = platforms;
	//tmpGetPlatform.num_platforms = num_platforms;

	//send parameters to remote node
	MPI_Send(&tmpGetPlatform, sizeof(tmpGetPlatform), MPI_BYTE, 0, 
			 GET_PLATFORM_ID_FUNC, slaveComm);

	MPI_Recv(&tmpGetPlatform, sizeof(tmpGetPlatform), MPI_BYTE, 0,
			 GET_PLATFORM_ID_FUNC, slaveComm, &status);
	*platforms = tmpGetPlatform.platforms;
	num_platforms = tmpGetPlatform.num_platforms;
	
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
	
	//debug===================================
	//printf("In get device ID function, shared library\n");
	//========================================

	//initialize structure
	tmpGetDeviceIDs.platform = platform;
	tmpGetDeviceIDs.device_type = device_type;
	tmpGetDeviceIDs.num_entries = num_entries;

	//send parameters to remote node
	MPI_Send(&tmpGetDeviceIDs, sizeof(tmpGetDeviceIDs), MPI_BYTE, 0, 
			 GET_DEVICE_ID_FUNC, slaveComm);
	MPI_Recv(&tmpGetDeviceIDs, sizeof(tmpGetDeviceIDs), MPI_BYTE, 0,
			 GET_DEVICE_ID_FUNC, slaveComm, &status);
	*devices = tmpGetDeviceIDs.devices;
	num_devices = tmpGetDeviceIDs.num_devices;
	
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

	//debug===================================
	//printf("In create Context function, shared library\n");
	//========================================

	//initialize structure
	//tmpCreateContext.properties = *properties;
	tmpCreateContext.num_devices = num_devices;
	tmpCreateContext.devices = *devices;
	tmpCreateContext.user_data = user_data;

	
	//send parameters to remote node
	res = MPI_Send(&tmpCreateContext, sizeof(tmpCreateContext), MPI_BYTE, 0, 
			 CREATE_CONTEXT_FUNC, slaveComm);
	if (res != MPI_SUCCESS)
	{
		printf("In create context, send message error!\n");
		exit(1);
	}

	MPI_Recv(&tmpCreateContext, sizeof(tmpCreateContext), MPI_BYTE, 0, 
			 CREATE_CONTEXT_FUNC, slaveComm, &status);
	*errcode_ret = tmpCreateContext.errcode_ret;
	
	//debug===================================
	//printf("In create Context function, shared library, message received.\n");
	//========================================
	
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
	tmpCreateProgramWithSource.lengths = *lengths;

	//send parameters to remote node
	MPI_Send(&tmpCreateProgramWithSource,
			 sizeof(tmpCreateProgramWithSource), 
			 MPI_BYTE, 0, CREATE_PROGRMA_WITH_SOURCE, 
			 slaveComm);
	MPI_Send((void *)*strings, *lengths, MPI_BYTE, 0, 
			 CREATE_PROGRMA_WITH_SOURCE1, slaveComm);
	MPI_Recv(&tmpCreateProgramWithSource,
			 sizeof(tmpCreateProgramWithSource), 
			 MPI_BYTE, 0, CREATE_PROGRMA_WITH_SOURCE, 
			 slaveComm, &status);
	*errcode_ret = tmpCreateProgramWithSource.errcode_ret;

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
		optionsLen = strlen(options) + 1;
	}

	struct strBuildProgram tmpBuildProgram;
	MPI_Status status;

	//initialize structure
	tmpBuildProgram.program = program;
	tmpBuildProgram.num_devices = num_devices;
	tmpBuildProgram.device_list = device_list;
	tmpBuildProgram.optionLen = optionsLen;

	//send parameters to remote node
	MPI_Send(&tmpBuildProgram, sizeof(tmpBuildProgram), MPI_BYTE, 0, 
			 BUILD_PROGRAM, slaveComm);
	if (optionsLen > 0)
	{
		MPI_Send((void *)options, optionsLen, MPI_CHAR, 0, BUILD_PROGRAM1, slaveComm);
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
	int kernelNameSize = strlen(kernel_name) + 1;
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
	//printf("host_ptr = %p\n", host_ptr);
	if (host_ptr != NULL)
	{
		tmpCreateBuffer.host_ptr_flag = 1;
	}
	//printf("host_ptr_flag = %d\n", tmpCreateBuffer.host_ptr_flag);

	//send parameters to remote node
	MPI_Send(&tmpCreateBuffer, sizeof(tmpCreateBuffer), MPI_BYTE, 0, 
			 CREATE_BUFFER_FUNC, slaveComm);
	if (tmpCreateBuffer.host_ptr_flag == 1)
	{
		//printf("Here!\n");
		MPI_Send(host_ptr, size, MPI_BYTE, 0, CREATE_BUFFER_FUNC1, slaveComm);
	}
	MPI_Recv(&tmpCreateBuffer, sizeof(tmpCreateBuffer), MPI_BYTE, 0, 
			 CREATE_BUFFER_FUNC, slaveComm, &status);
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

	//initialize structure
	tmpEnqueueWriteBuffer.command_queue = command_queue;
	tmpEnqueueWriteBuffer.buffer = buffer;
	tmpEnqueueWriteBuffer.blocking_write = blocking_write;
	tmpEnqueueWriteBuffer.offset = offset;
	tmpEnqueueWriteBuffer.cb = cb;
	tmpEnqueueWriteBuffer.num_events_in_wait_list = num_events_in_wait_list;
	//tmpEnqueueWriteBuffer.event_wait_list = event_wait_list;
	//tmpEnqueueWriteBuffer.event = event;

	//send parameters to remote node
	MPI_Send(&tmpEnqueueWriteBuffer, sizeof(tmpEnqueueWriteBuffer), MPI_BYTE, 0, 
			 ENQUEUE_WRITE_BUFFER, slaveComm);
	MPI_Send((void *)ptr, cb, MPI_BYTE, 0, ENQUEUE_WRITE_BUFFER1, slaveComm);
	MPI_Recv(&tmpEnqueueWriteBuffer, sizeof(tmpEnqueueWriteBuffer), MPI_BYTE, 0, 
			 ENQUEUE_WRITE_BUFFER, slaveComm, &status);
	return tmpEnqueueWriteBuffer.res;
}

cl_int
clSetKernelArg(cl_kernel    kernel,
               cl_uint      arg_index,
               size_t       arg_size,
               const void * arg_value)
{
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strSetKernelArg tmpSetKernelArg;
	MPI_Status status;

	//initialize structure
	tmpSetKernelArg.kernel = kernel;
	tmpSetKernelArg.arg_index = arg_index;
	tmpSetKernelArg.arg_size = arg_size;

	//send parameters to remote node
	MPI_Send(&tmpSetKernelArg, sizeof(tmpSetKernelArg), MPI_BYTE, 0, 
			 SET_KERNEL_ARG, slaveComm);
	MPI_Send((void *)arg_value, arg_size, MPI_BYTE, 0, SET_KERNEL_ARG1,
			 slaveComm);
	MPI_Recv(&tmpSetKernelArg, sizeof(tmpSetKernelArg), MPI_BYTE, 0, 
			 SET_KERNEL_ARG, slaveComm, &status);
	return tmpSetKernelArg.res;
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

	MPI_Recv(&tmpEnqueueNDRangeKernel, sizeof(tmpEnqueueNDRangeKernel), MPI_BYTE, 0, 
			 ENQUEUE_ND_RANGE_KERNEL, slaveComm, &status);
	
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
	
	MPI_Status status;
	struct strEnqueueReadBuffer tmpEnqueueReadBuffer;

	//initialize structure
	tmpEnqueueReadBuffer.command_queue = command_queue;
	tmpEnqueueReadBuffer.buffer = buffer;
	tmpEnqueueReadBuffer.blocking_read = blocking_read;
	tmpEnqueueReadBuffer.offset = offset;
	tmpEnqueueReadBuffer.cb = cb;
	tmpEnqueueReadBuffer.num_events_in_wait_list = num_events_in_wait_list;
	//tmpEnqueueReadBuffer.event_wait_list;
	//tmpEnqueueReadBuffer.event = event;

	//send parameters to remote node
	MPI_Send(&tmpEnqueueReadBuffer, sizeof(tmpEnqueueReadBuffer), MPI_BYTE, 0, 
			 ENQUEUE_READ_BUFFER, slaveComm);
	MPI_Recv(&tmpEnqueueReadBuffer, sizeof(tmpEnqueueReadBuffer), MPI_BYTE, 0, 
			 ENQUEUE_READ_BUFFER, slaveComm, &status);
	MPI_Recv(ptr, cb, MPI_CHAR, 0, ENQUEUE_READ_BUFFER1, slaveComm, &status);
	
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
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
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
	return tmpFinish.res;
}

