#include "gpuv.h"

void checkSlaveProc()
{
	if (slaveCreated == 0)
	{
		MPI_Init(NULL, NULL);
		MPI_Comm_spawn("./slave_process", MPI_ARGV_NULL, np, MPI_INFO_NULL, 0, MPI_COMM_WORLD,
						 &slaveComm, errCodes);
		slaveCreated = 1;
	}
}

cl_int
clGetPlatformIDs(cl_uint          num_entries,
				 cl_platform_id * platforms,
				 cl_uint *        num_platforms)
{
	int res;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	//send a message to indicate the function
	MPI_Send(NULL, 0, MPI_INT, 0, GET_PLATFORM_ID_FUNC, slaveComm);
	//send the number of cl_platform_id entries that can be added to platforms.
	MPI_Send(&num_entries, 1, MPI_INT, 0, 0, slaveComm);
	//make sure all parameters can be received.
	MPI_Barrier(slaveComm);

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
	int res;
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	//send a message to indicate the function
	MPI_Send(NULL, 0, MPI_INT, 0, GET_DEVICE_ID_FUNC, slaveComm);

	//send input parameters to remote node
	MPI_Send(&platform, 1, MPI_INT, 0, 0, slaveComm);
	MPI_Send(&device_type, 1, MPI_INT, 0, 1, slaveComm);
	MPI_Send(&num_entries, 1, MPI_INT, 0, 2, slaveComm);

	//barrier
	MPI_Barrier(slaveComm);

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
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	//send a message to indicate the function
	MPI_Send(NULL, 0, MPI_INT, 0, CREATE_CONTEXT_FUNC, slaveComm);

	//send input parameters to remote node
	MPI_Send(&num_devices, 1, MPI_INT, 0, 0, slaveComm);

	//barrier
	MPI_Barrier(slaveComm);

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

	//send a message to indicate the function
	MPI_Send(NULL, 0, MPI_INT, 0, CREATE_COMMAND_QUEUE_FUNC, slaveComm);

	//send input parameters to remote node
	MPI_Send(&properties, 1, MPI_INT, 0, 0, slaveComm);

	//barrier
	MPI_Barrier(slaveComm);
}

char * loadSource(char *filePathName, size_t *fileSize)
{
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

    FILE *pfile;
    size_t tmpFileSize;
    char *fileBuffer;
    pfile = fopen(filePathName, "rb");

    if (pfile == NULL)
    {
        printf("Open file %s open error!\n", filePathName);
        return NULL;
    }
    fseek(pfile, 0, SEEK_END);
    tmpFileSize = ftell(pfile);

    fileBuffer = (char *)malloc(tmpFileSize);

    fseek(pfile, 0, SEEK_SET);
    fread(fileBuffer, sizeof(char), tmpFileSize, pfile);
    fclose(pfile);

	//send a message to indicate the function
	MPI_Send(NULL, 0, MPI_INT, 0, LOAD_SOURCE_FUNC, slaveComm);

	//source file size to remote node
	MPI_Send(&tmpFileSize, 1, MPI_INT, 0, 0, slaveComm);
	MPI_Send(fileBuffer, tmpFileSize, MPI_CHAR, 0, 1, slaveComm);

	//barrier
	MPI_Barrier(slaveComm);
	*fileSize = tmpFileSize;

	return fileBuffer;
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
	
	//send a message to indicate the function
	MPI_Send(NULL, 0, MPI_INT, 0, CREATE_PROGRMA_WITH_SOURCE, slaveComm);

	//send input parameters to remote node
	MPI_Send(&count, 1, MPI_INT, 0, 0, slaveComm);

	//barrier
	MPI_Barrier(slaveComm);
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
	
	//send a message to indicate the function
	MPI_Send(NULL, 0, MPI_INT, 0, BUILD_PROGRAM, slaveComm);

	//send input parameters to remote node
	MPI_Send(&num_devices, 1, MPI_INT, 0, 0, slaveComm);

	//barrier
	MPI_Barrier(slaveComm);
}

cl_kernel
clCreateKernel(cl_program      program,
               const char *    kernel_name,
               cl_int *        errcode_ret)
{
	int kernelNameSize = str_len(kernel_name);
	
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	
	//send a message to indicate the function
	MPI_Send(NULL, 0, MPI_INT, 0, CREATE_KERNEL, slaveComm);

	//send input parameters to remote node
	MPI_Send(&kernelNameSize, 1, MPI_INT, 0, 0, slaveComm);
	MPI_Send(&kernel_name, kernelNameSize, MPI_CHAR, 0, 1, slaveComm);

	//barrier
	MPI_Barrier(slaveComm);
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
	
	//send a message to indicate the function
	MPI_Send(NULL, 0, MPI_INT, 0, CREATE_BUFFER, slaveComm);

	//send input parameters to remote node
	MPI_Send(&flags, 1, MPI_INT, 0, 0, slaveComm);
	MPI_Send(&size, 1, MPI_INT, 0, 1, slaveComm);

	//barrier
	MPI_Barrier(slaveComm);
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
	
	//send a message to indicate the function
	MPI_Send(NULL, 0, MPI_INT, 0, ENQUEUE_WRITE_BUFFER, slaveComm);

	//send input parameters to remote node
	MPI_Send(&blocking_write, 1, MPI_BOOL, 0, 0, slaveComm);
	MPI_Send(&offset, 1, MPI_INT, 0, 1, slaveComm);
	MPI_Send(&cb, 1, MPI_INT, 0, 2, slaveComm);
	MPI_Send(ptr, cb, MPI_CHAR, 0, 3, slaveComm);
	MPI_Send(&num_events_in_wait_list, 1, MPI_UINT, 0, 4, slaveComm);

	//barrier
	MPI_Barrier(slaveComm);
}

cl_int
clSetKernelArg(cl_kernel    kernel,
               cl_uint      arg_index,
               size_t       arg_size,
               const void * arg_value)
{
	//check whether the slave process is created. If not, create one.
	checkSlaveProc();
	
	//send a message to indicate the function
	MPI_Send(NULL, 0, MPI_INT, 0, SET_KERNEL_ARG, slaveComm);

	//send input parameters to remote node
	MPI_Send(&arg_index, 1, MPI_UINT, 0, 0, slaveComm);
	MPI_Send(&arg_size, 1, MPI_INT, 0, 1, slaveComm);
	MPI_Send(arg_value, arg_size, MPI_CHAR, 0, 2, slaveComm);

	//barrier
	MPI_Barrier(slaveComm);

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
	
	//send a message to indicate the function
	MPI_Send(NULL, 0, MPI_INT, 0, ENQUEUE_ND_RANGE_KERNEL, slaveComm);

	//send parameters to remote node
	MPI_Send(&work_dim, 1, MPI_UINT, 0, 0, slaveComm);
	MPI_Send(global_work_offset, work_dim, MPI_INT, 0, 1, slaveComm);
	MPI_Send(global_work_size, work_dim, MPI_INT, 0, 2, slaveComm);
	MPI_Send(local_work_size, work_dim, MPI_INT, 0, 3, slaveComm);

	//barrier
	MPI_Barrier(slaveComm);
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
	MPI_Status status;

	//check whether the slave process is created. If not, create one.
	checkSlaveProc();

	//send a message to indicate the function
	MPI_Send(NULL, 0, MPI_INT, 0, ENQUEUE_READ_BUFFER, slaveComm);

	MPI_Send(&blocking_read, 1, MPI_BOOL, 0, 0, slaveComm);
	MPI_Send(&offset, 1, MPI_INT, 0, 1, slaveComm);
	MPI_Send(&cb, 1, MPI_INT, 0, 2, slaveComm);

	//receive the output
	MPI_Recv(ptr, cb, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, slaveComm, &status);

	return 0;
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

