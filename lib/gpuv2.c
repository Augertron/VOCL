#include "gpuv.h"

void checkSlaveProc()
{
	if (slaveCreated == 0)
	{
		MPI_Init(NULL, NULL);
		MPI_Comm_spawn("./slave_process", MPI_ARGV_NULL, np, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &slaveComm, errCodes);
		slaveCreated = 1;
	}
}

cl_int
clGetPlatformIDs(cl_uint          num_entries,
				 cl_platform_id * platforms,
				 cl_uint *        num_platforms)
{
	int res = 0;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strGetPlatformIDs tmpGetPlatform;

	//create struct type
	MPI_Datatype getPlatformType;
	MPI_Datatype type[GET_PLAT_FORM_ELEM_NUM] = {MPI_UINT};
	int blocklen[GET_PLAT_FORM_ELEM_NUM] = {1};
	MPI_Aint disp[GET_PLAT_FORM_ELEM_NUM];
	int base, i;

	MPI_Address(&tmpGetPlatform, disp);
	base = disp[0];

	for (i = 0; i < GET_PLAT_FORM_ELEM_NUM; i++)
	{
		disp[i] = disp[i] - base;
	}

	MPI_Type_create_struct(GET_PLAT_FORM_ELEM_NUM, blocklen, disp, type,
						   &getPlatformType);
	MPI_Type_commit(&getPlatformType);

	//initialize structure
	tmpGetPlatform.num_entries = num_entries;

	//send parameters to remote node
	MPI_Send(&tmpGetPlatform, 1, getPlatformType, 0, 
			 GET_PLATFORM_ID_FUNC, slaveComm);
	
	//make sure all parameters can be received.
	//MPI_Barrier(slaveComm);
	MPI_Type_free(&getPlatformType);

	return res;
}


/* Device APIs */
cl_int
clGetDeviceIDs(cl_platform_id   platform,
               cl_device_type   device_type,
               cl_uint          num_entries,
               cl_device_id *   devices,
               cl_uint *        num_devices)
{
	int res = 0;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strGetDeviceIDs tmpGetDeviceIDs;

	//create struct type
	MPI_Datatype getDeviceIDsType;
	MPI_Datatype type[GET_DEVICE_IDS_ELEM_NUM] = {MPI_UINT, MPI_UINT};
	int blocklen[GET_DEVICE_IDS_ELEM_NUM] = {1, 1};
	MPI_Aint disp[GET_DEVICE_IDS_ELEM_NUM];
	int base, i;

	MPI_Address(&tmpGetDeviceIDs, disp);
	MPI_Address(&tmpGetDeviceIDs.num_entries, disp + 1);
	base = disp[0];

	for (i = 0; i < GET_DEVICE_IDS_ELEM_NUM; i++)
	{
		disp[i] = disp[i] - base;
	}

	MPI_Type_create_struct(GET_DEVICE_IDS_ELEM_NUM, blocklen, disp, type,
						   &getDeviceIDsType);
	MPI_Type_commit(&getDeviceIDsType);

	//initialize structure
	tmpGetPlatform.device_type = device_type;
	tmpGetPlatform.num_entries = num_entries;

	//send parameters to remote node
	MPI_Send(&tmpGetPlatform, 1, getDeviceIDsType, 0, 
			 GET_DEVICE_ID_FUNC, slaveComm);
	
	//make sure all parameters can be received.
	//MPI_Barrier(slaveComm);
	MPI_Type_free(&getDeviceIDsType);

	return res;
}

/* Context APIs  */
cl_context
clCreateContext(const cl_context_properties *  properties,
                cl_uint                        num_devices,
                const cl_device_id *           devices,
                void (CL_CALLBACK * pfn_notify)(const char *, const void *, size_t, void *),
                void *                         user_data,
                cl_int *                       errcode_ret)
{
	cl_context res;

	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strCreateContext tmpCreateContext;

	//create struct type
	MPI_Datatype createContextType;
	MPI_Datatype type[CREATE_CONTEXT_ELEM_NUM] = {MPI_UINT};
	int blocklen[CREATE_CONTEXT_ELEM_NUM] = {1};
	MPI_Aint disp[CREATE_CONTEXT_ELEM_NUM];
	int base, i;

	MPI_Address(&tmpCreateContext, disp);
	base = disp[0];

	for (i = 0; i < CREATE_CONTEXT_ELEM_NUM; i++)
	{
		disp[i] = disp[i] - base;
	}

	MPI_Type_create_struct(CREATE_CONTEXT_ELEM_NUM, blocklen, disp, type,
						   &createContextType);
	MPI_Type_commit(&createContextType);

	//initialize structure
	tmpCreateContext.num_devices = num_devices;

	//send parameters to remote node
	MPI_Send(&tmpGetPlatform, 1, createContextType, 0, 
			 CREATE_CONTEXT_FUNC, slaveComm);
	
	//make sure all parameters can be received.
	//MPI_Barrier(slaveComm);
	MPI_Type_free(&createContextType);

	return res;
}

/* Command Queue APIs */
cl_command_queue
clCreateCommandQueue(cl_context                     context,
                     cl_device_id                   device,
                     cl_command_queue_properties    properties,
                     cl_int *                       errcode_ret)
{
	cl_command_queue res = 0;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strCreateCommandQueue tmpCreateCommandQueue;

	//create struct type
	MPI_Datatype createCommandQueueType;
	MPI_Datatype type[CREATE_COMMAND_QUEUE_ELEM_NUM] = {MPI_UINT};
	int blocklen[CREATE_COMMAND_QUEUE_ELEM_NUM] = {1};
	MPI_Aint disp[CREATE_COMMAND_QUEUE_ELEM_NUM];
	int base, i;

	MPI_Address(&tmpCreateCommandQueue, disp);
	base = disp[0];

	for (i = 0; i < CREATE_COMMAND_QUEUE_ELEM_NUM; i++)
	{
		disp[i] = disp[i] - base;
	}

	MPI_Type_create_struct(CREATE_COMMAND_QUEUE_ELEM_NUM, blocklen, disp, type,
						   &createCommandQueueType);
	MPI_Type_commit(&createCommandQueueType);

	//initialize structure
	tmpCreateCommandQueue.cl_command_queue_properties = num_entries;

	//send parameters to remote node
	MPI_Send(&tmpCreateCommandQueue, 1, createCommandQueueType, 0, 
			 CREATE_COMMAND_QUEUE_FUNC, slaveComm);
	
	//make sure all parameters can be received.
	//MPI_Barrier(slaveComm);
	MPI_Type_free(&createCommandQueueType);

	return res;
}

//char * loadSource(char *filePathName, size_t *fileSize)
//{
//	//check whether the slave process is created. If not, create one.
//	checkSlaveProc();
//
//	FILE *pfile;
//	size_t tmpFileSize;
//	char *fileBuffer;
//	pfile = fopen(filePathName, "rb");
//
//	if (pfile == NULL)
//	{
//		printf("Open file %s open error!\n", filePathName);
//		return NULL;
//	}
//	fseek(pfile, 0, SEEK_END);
//	tmpFileSize = ftell(pfile);
//
//	fileBuffer = (char *)malloc(tmpFileSize);
//
//	fseek(pfile, 0, SEEK_SET);
//	fread(fileBuffer, sizeof(char), tmpFileSize, pfile);
//	fclose(pfile);
//
//	//send a message to indicate the function
//	MPI_Send(NULL, 0, MPI_INT, 0, LOAD_SOURCE_FUNC, slaveComm);
//
//	//source file size to remote node
//	MPI_Send(&tmpFileSize, 1, MPI_INT, 0, LOAD_SOURCE_FUNC, slaveComm);
//	MPI_Send(fileBuffer, tmpFileSize, MPI_CHAR, 0, LOAD_SOURCE_FUNC1, slaveComm);
//
//	free(fileBuffer);
//
//	return NULL;
//}

cl_program
clCreateProgramWithSource(cl_context        context,
                          cl_uint           count,
                          const char **     strings,
                          const size_t *    lengths,
                          cl_int *          errcode_ret)
{
	cl_program res;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strCreateProgramWithSource tmpCreateProgramWithSource;

	//create struct type
	MPI_Datatype createProgramWithSourceType;
	MPI_Datatype type[CREATE_PROGRAM_WITH_SOURCE_ELEM_NUM] = {MPI_UINT, MPI_UINT};
	int blocklen[CREATE_PROGRAM_WITH_SOURCE_ELEM_NUM] = {1, 1};
	MPI_Aint disp[CREATE_PROGRAM_WITH_SOURCE_ELEM_NUM];
	int base, i;

	MPI_Address(&tmpCreateProgramWithSource, disp);
	base = disp[0];

	for (i = 0; i < CREATE_PROGRAM_WITH_SOURCE_ELEM_NUM; i++)
	{
		disp[i] = disp[i] - base;
	}

	MPI_Type_create_struct(CREATE_PROGRAM_WITH_SOURCE_ELEM_NUM, blocklen, disp, type,
						   &createProgramWithSourceType);
	MPI_Type_commit(&createProgramWithSourceType);

	//initialize structure
	tmpCreateProgramWithSource.count = count;
	tmpCreateProgramWithSource.length = *length;

	//send parameters to remote node
	MPI_Send(&tmpCreateProgramWithSource, 1, createProgramWithSourceType, 0, 
			 CREATE_PROGRMA_WITH_SOURCE, slaveComm);
	MPI_Send(*strings, *length, MPI_CHAR, 0, CREATE_PROGRMA_WITH_SOURCE1, slaveComm);
	
	//make sure all parameters can be received.
	//MPI_Barrier(slaveComm);
	MPI_Type_free(&createProgramWithSourceType);

	return res;
}

cl_int
clBuildProgram(cl_program           program,
               cl_uint              num_devices,
               const cl_device_id * device_list,
               const char *         options, 
               void (CL_CALLBACK *  pfn_notify)(cl_program  program, void * user_data),
               void *               user_data)
{
	cl_int res;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	int optionsLen = str_len(options);

	struct strBuildProgram tmpBuildProgram;

	//create struct type
	MPI_Datatype buildProgramType;
	MPI_Datatype type[BUILD_PROGRAM_ELEM_NUM] = {MPI_UINT, MPI_UINT};
	int blocklen[BUILD_PROGRAM_ELEM_NUM] = {1, 1};
	MPI_Aint disp[BUILD_PROGRAM_ELEM_NUM];
	int base, i;

	MPI_Address(&tmpBuildProgram, disp);
	base = disp[0];

	for (i = 0; i < BUILD_PROGRAM_ELEM_NUM; i++)
	{
		disp[i] = disp[i] - base;
	}

	MPI_Type_create_struct(BUILD_PROGRAM_ELEM_NUM, blocklen, disp, type,
						   &buildProgramType);
	MPI_Type_commit(&buildProgramType);

	//initialize structure
	tmpBuildProgram.num_devices = count;
	tmpBuildProgram.length = optionsLen;

	//send parameters to remote node
	MPI_Send(&tmpBuildProgram, 1, buildProgramType, 0, 
			 BUILD_PROGRAM, slaveComm);
	MPI_Send(options, optionsLen, MPI_CHAR, 0, BUILD_PROGRAM1, slaveComm);
	
	//make sure all parameters can be received.
	//MPI_Barrier(slaveComm);
	MPI_Type_free(&buildProgramType);

	return res;
}

cl_kernel
clCreateKernel(cl_program      program,
               const char *    kernel_name,
               cl_int *        errcode_ret)
{
	cl_kernel res;
	int kernelNameSize = str_len(kernel_name);
	
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	
	//send input parameters to remote node
	MPI_Send(&kernelNameSize, 1, MPI_INT, 0, CREATE_KERNEL, slaveComm);
	MPI_Send(&kernel_name, kernelNameSize, MPI_CHAR, 0, CREATE_KERNEL1, slaveComm);

	return res;
}

/* Memory Object APIs */
cl_mem
clCreateBuffer(cl_context   context,
               cl_mem_flags flags,
               size_t       size,
               void *       host_ptr,
               cl_int *     errcode_ret)
{
	cl_mem res = 0;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strCreateBuffer tmpCreateBuffer;

	//create struct type
	MPI_Datatype createBufferType;
	MPI_Datatype type[CREATE_BUFFER_ELEM_NUM] = {MPI_UINT, MPI_UINT};
	int blocklen[CREATE_BUFFER_ELEM_NUM] = {1, 1};
	MPI_Aint disp[CREATE_BUFFER_ELEM_NUM];
	int base, i;

	MPI_Address(&tmpCreateBuffer, disp);
	MPI_Address(&tmpCreateBuffer.size, disp + 1);
	base = disp[0];

	for (i = 0; i < CREATE_BUFFER_ELEM_NUM; i++)
	{
		disp[i] = disp[i] - base;
	}

	MPI_Type_create_struct(CREATE_BUFFER_ELEM_NUM, blocklen, disp, type,
						   &createBufferType);
	MPI_Type_commit(&createBufferType);

	//initialize structure
	tmpCreateBuffer.flags = flags;
	tmpCreateBuffer.size = size;

	//send parameters to remote node
	MPI_Send(&tmpCreateBuffer, 1, createBufferType, 0, 
			 CREATE_BUFFER_FUNC, slaveComm);
	
	//make sure all parameters can be received.
	//MPI_Barrier(slaveComm);
	MPI_Type_free(&createBufferType);

	return res;
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
	cl_int res;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strEnqueueWriteBuffer tmpEnqueueWriteBuffer;

	//create struct type
	MPI_Datatype enqueueWriteBufferType;
	MPI_Datatype type[ENQUEUE_WRITE_BUFFER_ELEM_NUM] = {MPI_BOOL, MPI_UINT, MPI_UINT,
														MPI_UINT};
	int blocklen[ENQUEUE_WRITE_BUFFER_ELEM_NUM] = {1, 1, 1, 1};
	MPI_Aint disp[ENQUEUE_WRITE_BUFFER_ELEM_NUM];
	int base, i;

	MPI_Address(&tmpEnqueueWriteBuffer, disp);
	MPI_Address(&tmpEnqueueWriteBuffer.offset, disp+1);
	MPI_Address(&tmpEnqueueWriteBuffer.cb, disp+2);
	MPI_Address(&tmpEnqueueWriteBuffer.num_events_in_wait_list, disp+3);
	base = disp[0];

	for (i = 0; i < ENQUEUE_WRITE_BUFFER_ELEM_NUM; i++)
	{
		disp[i] = disp[i] - base;
	}

	MPI_Type_create_struct(ENQUEUE_WRITE_BUFFER_ELEM_NUM, blocklen, disp, type,
						   &enqueueWriteBufferType);
	MPI_Type_commit(&enqueueWriteBufferType);

	//initialize structure
	tmpEnqueueWriteBuffer.blocking_write = blocking_write;
	tmpEnqueueWriteBuffer.offset = offset;
	tmpEnqueueWriteBuffer.cb = cb;
	tmpEnqueueWriteBuffer.num_events_in_wait_list = num_events_in_wait_list;

	//send parameters to remote node
	MPI_Send(&tmpEnqueueWriteBuffer, 1, enqueueWriteBufferType, 0, 
			 ENQUEUE_WRITE_BUFFER, slaveComm);
	MPI_Send(ptr, cb, MPI_CHAR, 0, ENQUEUE_WRITE_BUFFER1, slaveComm);
	
	//make sure all parameters can be received.
	//MPI_Barrier(slaveComm);
	MPI_Type_free(&enqueueWriteBufferType);

	return res;
}

cl_int
clSetKernelArg(cl_kernel    kernel,
               cl_uint      arg_index,
               size_t       arg_size,
               const void * arg_value)
{
	cl_int res;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strSetKernelArg tmpSetKernelArg;

	//create struct type
	MPI_Datatype setKernelArgType;
	MPI_Datatype type[SET_KERNEL_ARG_ELEM_NUM] = {MPI_UINT, MPI_UINT, MPI_UINT};
	int blocklen[SET_KERNEL_ARG_ELEM_NUM] = {1, 1, 1};
	MPI_Aint disp[SET_KERNEL_ARG_ELEM_NUM];
	int base, i;

	MPI_Address(&tmpSetKernelArg, disp);
	MPI_Address(&tmpSetKernelArg.arg_size, disp+1);
	MPI_Address(&tmpSetKernelArg.globalIndex, disp+2);
	base = disp[0];

	for (i = 0; i < SET_KERNEL_ARG_ELEM_NUM; i++)
	{
		disp[i] = disp[i] - base;
	}

	MPI_Type_create_struct(SET_KERNEL_ARG_ELEM_NUM, blocklen, disp, type,
						   &setKernelArgType);
	MPI_Type_commit(&setKernelArgType);

	//initialize structure
	tmpSetKernelArg.arg_index = arg_index;
	tmpSetKernelArg.arg_size = arg_size;
	tmpSetKernelArg.arg_globalIndex = arg_globalIndex;

	//send parameters to remote node
	MPI_Send(&tmpSetKernelArg, 1, setKernelArgType, 0, 
			 SET_KERNEL_ARG, slaveComm);
	
	//make sure all parameters can be received.
	//MPI_Barrier(slaveComm);
	MPI_Type_free(&setKernelArgType);

	return res;
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
	cl_int res;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	struct strEnqueueNDRangeKernel tmpEnqueueNDRangeKernel;

	//create struct type
	MPI_Datatype enqueueNDRangeKernelType;
	MPI_Datatype type[ENQUEUE_ND_RANGE_KERNEL_ELEM_NUM] = {MPI_UINT, MPI_UINT, MPI_UINT,
														MPI_UINT};
	int blocklen[ENQUEUE_ND_RANGE_KERNEL_ELEM_NUM] = {1, 3, 3, 3, 1};
	MPI_Aint disp[ENQUEUE_ND_RANGE_KERNEL_ELEM_NUM];
	int base, i;

	MPI_Address(&tmpEnqueueNDRangeKernel, disp);
	MPI_Address(&tmpEnqueueNDRangeKernel.global_work_offset, disp+1);
	MPI_Address(&tmpEnqueueNDRangeKernel.global_work_size, disp+2);
	MPI_Address(&tmpEnqueueNDRangeKernel.local_work_size, disp+3);
	MPI_Address(&tmpEnqueueNDRangeKernel.num_events_in_wait_list, disp+4);
	base = disp[0];

	for (i = 0; i < ENQUEUE_ND_RANGE_KERNEL_ELEM_NUM; i++)
	{
		disp[i] = disp[i] - base;
	}

	MPI_Type_create_struct(ENQUEUE_ND_RANGE_KERNEL_ELEM_NUM, blocklen, disp, type,
						   &enqueueNDRangeKernelType);
	MPI_Type_commit(&enqueueNDRangeKernelType);

	//initialize structure
	tmptmpEnqueueNDRangeKernel.work_dim = work_dim;
	tmptmpEnqueueNDRangeKernel.global_work_offset[0] = global_work_offset[0];
	tmptmpEnqueueNDRangeKernel.global_work_offset[1] = global_work_offset[1];
	tmptmpEnqueueNDRangeKernel.global_work_offset[2] = global_work_offset[2];
	tmptmpEnqueueNDRangeKernel.global_work_size[0] = global_work_size[0];
	tmptmpEnqueueNDRangeKernel.global_work_size[1] = global_work_size[1];
	tmptmpEnqueueNDRangeKernel.global_work_size[2] = global_work_size[2];
	tmptmpEnqueueNDRangeKernel.local_work_size[0] = local_work_size[0];
	tmptmpEnqueueNDRangeKernel.local_work_size[1] = local_work_size[1];
	tmptmpEnqueueNDRangeKernel.local_work_size[2] = local_work_size[2];
	tmptmpEnqueueNDRangeKernel.num_events_in_wait_list = num_events_in_wait_list;

	//send parameters to remote node
	MPI_Send(&tmpEnqueueNDRangeKernel, 1, enqueueNDRangeKernelType, 0, 
			 ENQUEUE_ND_RANGE_KERNEL, slaveComm);
	
	//make sure all parameters can be received.
	//MPI_Barrier(slaveComm);
	MPI_Type_free(&enqueueNDRangeKernelType);

	return res;
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
	cl_int res;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	
	MPI_Status status;
	struct strEnqueueReadBuffer tmpEnqueueReadBuffer;

	//create struct type
	MPI_Datatype enqueueReadBufferType;
	MPI_Datatype type[ENQUEUE_READ_BUFFER_ELEM_NUM] = {MPI_BOOL, MPI_UINT, MPI_UINT,
														MPI_UINT};
	int blocklen[ENQUEUE_READ_BUFFER_ELEM_NUM] = {1, 1, 1, 1};
	MPI_Aint disp[ENQUEUE_READ_BUFFER_ELEM_NUM];
	int base, i;

	MPI_Address(&tmpEnqueueReadBuffer, disp);
	MPI_Address(&tmpEnqueueReadBuffer.offset, disp+1);
	MPI_Address(&tmpEnqueueReadBuffer.cb, disp+2);
	MPI_Address(&tmpEnqueueReadBuffer.num_events_in_wait_list, disp+3);
	base = disp[0];

	for (i = 0; i < ENQUEUE_READ_BUFFER_ELEM_NUM; i++)
	{
		disp[i] = disp[i] - base;
	}

	MPI_Type_create_struct(ENQUEUE_READ_BUFFER_ELEM_NUM, blocklen, disp, type,
						   &enqueueReadBufferType);
	MPI_Type_commit(&enqueueReadBufferType);

	//initialize structure
	tmpEnqueueReadBuffer.blocking_read = blocking_read;
	tmpEnqueueReadBuffer.offset = offset;
	tmpEnqueueReadBuffer.cb = cb;
	tmpEnqueueReadBuffer.num_events_in_wait_list = num_events_in_wait_list;

	//send parameters to remote node
	MPI_Send(&tmpEnqueueReadBuffer, 1, enqueueReadBufferType, 0, 
			 ENQUEUE_READ_BUFFER, slaveComm);
	MPI_Recv(ptr, cb, MPI_CHAR, 0, ENQUEUE_READ_BUFFER, slaveComm, &status);
	
	//make sure all parameters can be received.
	//MPI_Barrier(slaveComm);
	MPI_Type_free(&enqueueReadBufferType);

	return res;


}

cl_int
clReleaseMemObject(cl_mem memobj)
{
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	
	//send a message to indicate the function
	MPI_Send(NULL, 0, MPI_INT, 0, RELEASE_MEM_OBJ, slaveComm);

	return 0;
}

