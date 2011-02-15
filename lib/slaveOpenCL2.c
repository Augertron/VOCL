#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <CL/opencl.h>

/**************************************************************************
 *1. First version of the slave process
 *2. Use non-blocking message send instead of the blocking ones
 **************************************************************************/

#define MPI_MYSEND MPI_Isend

#define GET_PLATFORM_ID_FUNC       10000
#define GET_DEVICE_ID_FUNC         10001
#define CREATE_CONTEXT_FUNC        10002
#define CREATE_COMMAND_QUEUE_FUNC  10003
#define LOAD_SOURCE_FUNC           10004
#define LOAD_SOURCE_FUNC1          10005
#define CREATE_PROGRMA_WITH_SOURCE 10006
#define CREATE_PROGRMA_WITH_SOURCE1 10007
#define BUILD_PROGRAM              10008
#define BUILD_PROGRAM1             10009
#define CREATE_KERNEL              10010
#define CREATE_KERNEL1             10011
#define CREATE_BUFFER_FUNC         10012
#define CREATE_BUFFER_FUNC1        10013
#define ENQUEUE_WRITE_BUFFER       10014
#define ENQUEUE_WRITE_BUFFER1      10015
#define SET_KERNEL_ARG             10016
#define SET_KERNEL_ARG1            10017
#define ENQUEUE_ND_RANGE_KERNEL    10018
#define ENQUEUE_ND_RANGE_KERNEL1   10019
#define ENQUEUE_ND_RANGE_KERNEL2   10020
#define ENQUEUE_ND_RANGE_KERNEL3   10021
#define ENQUEUE_READ_BUFFER        10022
#define ENQUEUE_READ_BUFFER1       10023
#define RELEASE_MEM_OBJ            10024
#define FINISH_FUNC                10025
#define CL_RELEASE_KERNEL_FUNC     10026
#define PROGRAM_END                11111


cl_int err;
cl_platform_id platformID;
cl_device_id deviceID;
cl_context hContext;
cl_command_queue hCmdQueue;
cl_program hProgram;
cl_kernel hKernel;
cl_mem deviceMem[3];

struct strGetPlatformIDs {
	cl_uint          num_entries;
	cl_platform_id   platforms;
	cl_uint *        num_platforms;
	cl_int           res;
};


struct strGetPlatformIDs getPlatformIDStr;
void mpiOpenCLGetPlatformIDs(struct strGetPlatformIDs *tmpGetPlatform)
{
	err = clGetPlatformIDs(tmpGetPlatform->num_entries, &platformID, NULL);
	tmpGetPlatform->platforms = platformID;
	tmpGetPlatform->res = err;

	//printf("In mpiOpenCLGetPlatformIDs function!\n");

	return;
}

struct strGetDeviceIDs
{
	cl_platform_id   platform;
	cl_device_type   device_type;
	cl_uint          num_entries;
	cl_device_id     devices;
	cl_uint *        num_devices;
	cl_int           res;
};

struct strGetDeviceIDs tmpGetDeviceIDs;
void mpiOpenCLGetDeviceIDs(struct strGetDeviceIDs *tmpGetDeviceIDs)
{
	//printf("In mpiOpenCLGetDeviceIDs function1!\n");
	cl_platform_id platform       = tmpGetDeviceIDs->platform;
	cl_device_type device_type = tmpGetDeviceIDs->device_type;
	cl_uint num_entries        = tmpGetDeviceIDs->num_entries;
	err = clGetDeviceIDs(platform, 
						 device_type, 
						 num_entries, 
						 &deviceID, 
						 NULL);
	tmpGetDeviceIDs->devices = deviceID;
	tmpGetDeviceIDs->res     = err;
	
	//printf("In mpiOpenCLGetDeviceIDs function2!\n");

	return;
}

struct strCreateContext {
	cl_context_properties   properties;
	cl_uint                       num_devices;
	cl_device_id                  devices;
	//CL_CALLBACK *                 pfn_notify;
	void *                        user_data;
	cl_int                        errcode_ret;
	cl_context                    hContext;
};

struct strCreateContext tmpCreateContext;
void mpiOpenCLCreateContext(struct strCreateContext *tmpCreateContext)
{
	cl_uint num_devices = tmpCreateContext->num_devices;
	//const cl_context_properties properties = tmpCreateContext->properties;
	const cl_device_id devices = tmpCreateContext->devices;
	hContext = clCreateContext(0, num_devices, &devices, 0, 0, &err);
	tmpCreateContext->hContext = hContext;
	tmpCreateContext->errcode_ret = err;
}

struct strCreateCommandQueue {
	cl_context                     context;
	cl_device_id                   device;
	cl_command_queue_properties    properties;
	cl_command_queue               clCommand;
	cl_int                         errcode_ret;
};

struct strCreateCommandQueue tmpCreateCommandQueue;
void mpiOpenCLCreateCommandQueue(struct strCreateCommandQueue *tmpCreateCommandQueue)
{
	cl_command_queue_properties properties = tmpCreateCommandQueue->properties;
	cl_device_id device = tmpCreateCommandQueue->device;
	cl_context hInContext = tmpCreateCommandQueue->context;

	hCmdQueue = clCreateCommandQueue(hInContext, device, properties, &err);

	tmpCreateCommandQueue->errcode_ret = err;

	tmpCreateCommandQueue->clCommand = hCmdQueue;
}

struct strCreateProgramWithSource {
	cl_context        context;
	cl_uint           count;
	size_t            lengths;
	cl_program        clProgram;
	cl_int            errcode_ret;
};

struct strCreateProgramWithSource tmpCreateProgramWithSource;
void mpiOpenCLCreateProgramWithSource(struct strCreateProgramWithSource *tmpCreateProgramWithSource, 
									  const char *cSourceCL)
{
	size_t sourceFileSize = tmpCreateProgramWithSource->lengths;
	int count = tmpCreateProgramWithSource->count;
	cl_context hInContext = tmpCreateProgramWithSource->context;
    hProgram = clCreateProgramWithSource(hInContext, count, (const char **)&cSourceCL, 
										 &sourceFileSize, &err);
	tmpCreateProgramWithSource->clProgram = hProgram;
	tmpCreateProgramWithSource->errcode_ret = err;
}

struct strBuildProgram {
	cl_program           program;
	cl_uint              num_devices;
	const cl_device_id   *device_list;
	cl_uint              optionLen;
	//CL_CALLBACK *        pfn_notify;
	void *               user_data;
	cl_int               res;
};

struct strBuildProgram tmpBuildProgram;
void mpiOpenCLBuildProgram(struct strBuildProgram *tmpBuildProgram, char *options)
{
	cl_program hInProgram = tmpBuildProgram->program;
	cl_uint num_devices = tmpBuildProgram->num_devices;
	err = clBuildProgram(hInProgram, num_devices, 0, options, 0, 0);
	tmpBuildProgram->res = err;
}

struct strCreateKernel {
	cl_program      program;
	size_t          kernelNameSize;
	cl_int          errcode_ret;
	cl_kernel       kernel;
};

struct strCreateKernel tmpCreateKernel;
void mpiOpenCLCreateKernel(struct strCreateKernel *tmpCreateKernel, char *kernel_name)
{
	cl_program hInProgram = tmpCreateKernel->program;
	hKernel = clCreateKernel(hInProgram, kernel_name, &err);
	tmpCreateKernel->kernel = hKernel;
	tmpCreateKernel->errcode_ret = err;
}
	
struct strCreateBuffer {
	cl_context   context;
	cl_mem_flags flags;
	size_t       size;
	//void *       host_ptr;
	cl_int       host_ptr_flag;
	cl_int       errcode_ret;
	cl_mem       deviceMem;
};

struct strCreateBuffer tmpCreateBuffer;
void mpiOpenCLCreateBuffer(struct strCreateBuffer *tmpCreateBuffer, void *host_ptr)
{
	cl_context hInContext = tmpCreateBuffer->context;
	cl_mem_flags flags = tmpCreateBuffer->flags;
	size_t size = tmpCreateBuffer->size;
	cl_mem deviceMem;
	deviceMem = clCreateBuffer(hInContext,
							   flags,
							   size,
							   host_ptr,
							   &err);
	tmpCreateBuffer->errcode_ret = err;
	tmpCreateBuffer->deviceMem = deviceMem;
}

struct strEnqueueWriteBuffer {
	cl_command_queue   command_queue;
	cl_mem             buffer;
	cl_bool            blocking_write;
	size_t             offset;
	size_t             cb;
	//const void *       ptr;
	cl_uint            num_events_in_wait_list;
	const cl_event *   event_wait_list;
	cl_event *         event;
	cl_int             res;
};

struct strEnqueueWriteBuffer tmpEnqueueWriteBuffer;
void mpiOpenCLEnqueueWriteBuffer(struct strEnqueueWriteBuffer *tmpEnqueueWriteBuffer,
								 void *ptr)
{
	cl_command_queue hInCmdQueue = tmpEnqueueWriteBuffer->command_queue;
	cl_mem deviceMem = tmpEnqueueWriteBuffer->buffer;
	cl_bool blocking_write = tmpEnqueueWriteBuffer->blocking_write;
	size_t offset = tmpEnqueueWriteBuffer->offset;
	size_t cb = tmpEnqueueWriteBuffer->cb;
	cl_uint num_events_in_wait_list = tmpEnqueueWriteBuffer->num_events_in_wait_list;

	err = clEnqueueWriteBuffer(hInCmdQueue, deviceMem, blocking_write, offset,
							   cb, ptr, num_events_in_wait_list, NULL, NULL);
	tmpEnqueueWriteBuffer->res = err; 
}

struct strSetKernelArg {
	cl_kernel    kernel;
	cl_uint      arg_index;
	size_t       arg_size;
	const void * arg_value;
	cl_int       res;
};

struct strSetKernelArg tmpSetKernelArg;
void mpiOpenCLSetKernelArg(struct strSetKernelArg *tmpSetKernelArg, void *arg_value)
{
	cl_kernel hInKernel = tmpSetKernelArg->kernel;
	cl_uint   arg_index = tmpSetKernelArg->arg_index;
	size_t    arg_size  = tmpSetKernelArg->arg_size;

	err = clSetKernelArg(hInKernel, arg_index, arg_size, arg_value);

	tmpSetKernelArg->res = err;
}

struct strEnqueueNDRangeKernel {
	cl_command_queue command_queue;
	cl_kernel        kernel;
	cl_uint          work_dim;
	cl_int           global_work_offset_flag;
	cl_int           global_work_size_flag;
	cl_int           local_work_size_flag;
	//const size_t     *global_work_offset;
	//const size_t     *global_work_size;
	//const size_t     *local_work_size;
	cl_uint          num_events_in_wait_list;
	//const cl_event * event_wait_list;
	cl_event *       event;
	cl_int           res;
};

struct strEnqueueNDRangeKernel tmpEnqueueNDRangeKernel;
void mpiOpenCLEnqueueNDRangeKernel(struct strEnqueueNDRangeKernel *tmpEnqueueNDRangeKernel,
							  size_t     *global_work_offset,
							  size_t     *global_work_size,
							  size_t     *local_work_size)
{
	cl_command_queue hInCommand = tmpEnqueueNDRangeKernel->command_queue;
	cl_kernel hInKernel = tmpEnqueueNDRangeKernel->kernel;
	cl_uint work_dim = tmpEnqueueNDRangeKernel->work_dim;
	cl_uint num_events_in_wait_list = tmpEnqueueNDRangeKernel->num_events_in_wait_list;
	err = clEnqueueNDRangeKernel(hInCommand,
								 hInKernel,
								 work_dim,
								 global_work_offset,
								 global_work_size,
								 local_work_size,
								 0,
								 //num_events_in_wait_list,
								 NULL,
								 NULL);
	//printf("after kernel execution!\n");
	if (err != CL_SUCCESS)
	{
		printf("Kernel launch error!\n");
	}
	tmpEnqueueNDRangeKernel->res = err;
}

struct strEnqueueReadBuffer {
	cl_command_queue    command_queue;
	cl_mem              buffer;
	cl_bool             blocking_read;
	size_t              offset;
	size_t              cb;
	//void *            ptr;
	cl_uint             num_events_in_wait_list;
	//const cl_event *    event_wait_list;
	//cl_event *          event;
	cl_int              res;
};

struct strEnqueueReadBuffer tmpEnqueueReadBuffer;
void mpiOpenCLEnqueueReadBuffer(struct strEnqueueReadBuffer *tmpEnqueueReadBuffer, void *ptr)
{
	cl_command_queue hInCmdQueue = tmpEnqueueReadBuffer->command_queue;
	cl_mem deviceMem = tmpEnqueueReadBuffer->buffer;
	cl_bool read_flag = tmpEnqueueReadBuffer->blocking_read;
	size_t  offset = tmpEnqueueReadBuffer->offset;
	size_t  cb = tmpEnqueueReadBuffer->cb;
	cl_uint num_events_in_wait_list = tmpEnqueueReadBuffer->num_events_in_wait_list;

	err = clEnqueueReadBuffer(hInCmdQueue,
							deviceMem,
							read_flag,
							offset,
							cb,
							ptr,
							num_events_in_wait_list,
							NULL,
							NULL);
	tmpEnqueueReadBuffer->res = err;
}

struct strReleaseMemObject {
	cl_mem memobj;
	cl_int res;
};

struct strReleaseMemObject tmpReleaseMemObject;
void mpiOpenCLReleaseMemObject(struct strReleaseMemObject *tmpReleaseMemObject)
{
	cl_mem deviceMem = tmpReleaseMemObject->memobj;
	err = clReleaseMemObject(deviceMem);
	tmpReleaseMemObject->res = err;
}

struct strReleaseKernel {
	cl_kernel kernel;
	cl_int    res;
};

struct strReleaseKernel tmpReleaseKernel;
void mpiOpenCLReleaseKernel(struct strReleaseKernel *tmpReleaseKernel)
{
	cl_kernel hInKernel = tmpReleaseKernel->kernel;
	err = clReleaseKernel(hInKernel);
	tmpReleaseKernel->res = err;
}

struct strFinish {
	cl_command_queue command_queue;
	cl_int res;
};

struct strFinish tmpFinish;
void mpiOpenCLFinish(struct strFinish *tmpFinish)
{
	cl_command_queue hInCmdQueue = tmpFinish->command_queue;
	err = clFinish(hInCmdQueue);
	tmpFinish->res = err;
}

int main(int argc, char *argv[])
{
	int rank, i;
	MPI_Status status;
	MPI_Request request1, request2;
	MPI_Comm parentComm;
	MPI_Init(&argc, &argv);
	MPI_Comm_get_parent(&parentComm);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	char hostName[200];
	int  len;
	MPI_Get_processor_name(hostName, &len);
	hostName[len] = '\0';
	printf("slaveHostName = %s\n", hostName);

	while (1)
	{
		MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, parentComm, &status);
		if (status.MPI_TAG == GET_PLATFORM_ID_FUNC)
		{
			MPI_Recv(&getPlatformIDStr, sizeof(getPlatformIDStr), MPI_BYTE, 0,
					 GET_PLATFORM_ID_FUNC, parentComm, &status);
			mpiOpenCLGetPlatformIDs(&getPlatformIDStr);
			MPI_MYSEND(&getPlatformIDStr, sizeof(getPlatformIDStr), MPI_BYTE, 0,
					 GET_PLATFORM_ID_FUNC, parentComm, &request1);
			//printf("In slave process, in function mpiOpenCLGetPlatformIDs!\n");
		}

		if (status.MPI_TAG == GET_DEVICE_ID_FUNC)
		{
			//printf("In slave process, before GET_DEVICE_ID_FUNC received!\n");
			MPI_Recv(&tmpGetDeviceIDs, sizeof(tmpGetDeviceIDs), MPI_BYTE, 0,
					GET_DEVICE_ID_FUNC, parentComm, &status);
			mpiOpenCLGetDeviceIDs(&tmpGetDeviceIDs);
			MPI_MYSEND(&tmpGetDeviceIDs, sizeof(tmpGetDeviceIDs), MPI_BYTE, 0,
					GET_DEVICE_ID_FUNC, parentComm, &request1);
		}

		if (status.MPI_TAG == CREATE_CONTEXT_FUNC)
		{
			//printf("In slave process, before CREATE_CONTEXT_FUNC received!\n");
			MPI_Recv(&tmpCreateContext, sizeof(tmpCreateContext), MPI_BYTE, 0,
					CREATE_CONTEXT_FUNC, parentComm, &status);
			mpiOpenCLCreateContext(&tmpCreateContext);
			MPI_MYSEND(&tmpCreateContext, sizeof(tmpCreateContext), MPI_BYTE, 0,
					CREATE_CONTEXT_FUNC, parentComm, &request1);
		}

		if (status.MPI_TAG == CREATE_COMMAND_QUEUE_FUNC)
		{
			MPI_Recv(&tmpCreateCommandQueue, sizeof(tmpCreateCommandQueue), MPI_BYTE, 0,
					 CREATE_COMMAND_QUEUE_FUNC, parentComm, &status);
			mpiOpenCLCreateCommandQueue(&tmpCreateCommandQueue);
			MPI_MYSEND(&tmpCreateCommandQueue, sizeof(tmpCreateCommandQueue), MPI_BYTE, 0,
					 CREATE_COMMAND_QUEUE_FUNC, parentComm, &request1);	
		}

		if (status.MPI_TAG == CREATE_PROGRMA_WITH_SOURCE)
		{
			MPI_Recv(&tmpCreateProgramWithSource, sizeof(tmpCreateProgramWithSource), MPI_BYTE, 0,
					 CREATE_PROGRMA_WITH_SOURCE, parentComm, &status);
			int fileSize = tmpCreateProgramWithSource.lengths;
			char *fileBuffer = (char *)malloc(fileSize);
			MPI_Recv(fileBuffer, fileSize, MPI_BYTE, 0,
					 CREATE_PROGRMA_WITH_SOURCE1, parentComm, &status);

			mpiOpenCLCreateProgramWithSource(&tmpCreateProgramWithSource, fileBuffer);
			MPI_MYSEND(&tmpCreateProgramWithSource, sizeof(tmpCreateProgramWithSource), MPI_BYTE, 0,
					 CREATE_PROGRMA_WITH_SOURCE, parentComm, &request1);
			free(fileBuffer);
		}

		if (status.MPI_TAG == BUILD_PROGRAM)
		{
			MPI_Recv(&tmpBuildProgram, sizeof(tmpBuildProgram), MPI_BYTE, 0,
					 BUILD_PROGRAM, parentComm, &status);
			//printf("slave, build program, message received!\n");
			char *tmpBuf = NULL;
			if (tmpBuildProgram.optionLen > 0)
			{
				char *tmpBuf = (char *)malloc(tmpBuildProgram.optionLen);
				MPI_Recv(tmpBuf, tmpBuildProgram.optionLen, MPI_CHAR, 0,
					     BUILD_PROGRAM1, parentComm, &status);
			}

			mpiOpenCLBuildProgram(&tmpBuildProgram, tmpBuf);
			//printf("slave, build program completed!\n");
			MPI_MYSEND(&tmpBuildProgram, sizeof(tmpBuildProgram), MPI_BYTE, 0,
					 BUILD_PROGRAM, parentComm, &request1);
			if (tmpBuildProgram.optionLen > 0)
			{
				free(tmpBuf);
			}
		}

		if (status.MPI_TAG == CREATE_KERNEL)
		{
			MPI_Recv(&tmpCreateKernel, sizeof(tmpCreateKernel), MPI_BYTE, 0,
					 CREATE_KERNEL, parentComm, &status);
			char *kernelName = (char *)malloc(tmpCreateKernel.kernelNameSize * sizeof(char));
			MPI_Recv(kernelName, tmpCreateKernel.kernelNameSize, MPI_CHAR, 0,
					 CREATE_KERNEL1, parentComm, &status);
			mpiOpenCLCreateKernel(&tmpCreateKernel, kernelName);
			MPI_MYSEND(&tmpCreateKernel, sizeof(tmpCreateKernel), MPI_BYTE, 0,
					 CREATE_KERNEL, parentComm, &request1);
			free(kernelName);
		}

		if (status.MPI_TAG == CREATE_BUFFER_FUNC)
		{
			MPI_Recv(&tmpCreateBuffer, sizeof(tmpCreateBuffer), MPI_BYTE, 0,
					 CREATE_BUFFER_FUNC, parentComm, &status);
			char *host_ptr = NULL;
			if (tmpCreateBuffer.host_ptr_flag == 1)
			{
				host_ptr = (char *)malloc(sizeof(char) * tmpCreateBuffer.size);
				MPI_Recv(host_ptr, tmpCreateBuffer.size, MPI_BYTE, 0, 
						 CREATE_BUFFER_FUNC1, parentComm, &status);
			}
			mpiOpenCLCreateBuffer(&tmpCreateBuffer, host_ptr);
			MPI_MYSEND(&tmpCreateBuffer, sizeof(tmpCreateBuffer), MPI_BYTE, 0,
					 CREATE_BUFFER_FUNC, parentComm, &request1);
			if (tmpCreateBuffer.host_ptr_flag == 1)
			{
				free(host_ptr);
			}
		}

		if (status.MPI_TAG == ENQUEUE_WRITE_BUFFER)
		{
			MPI_Recv(&tmpEnqueueWriteBuffer, sizeof(tmpEnqueueWriteBuffer), MPI_BYTE, 0,
					 ENQUEUE_WRITE_BUFFER, parentComm, &status);
			char *host_ptr = (char *)malloc(tmpEnqueueWriteBuffer.cb * sizeof(char));
			MPI_Recv(host_ptr, tmpEnqueueWriteBuffer.cb, MPI_BYTE, 0,
					 ENQUEUE_WRITE_BUFFER1, parentComm, &status);
			mpiOpenCLEnqueueWriteBuffer(&tmpEnqueueWriteBuffer, host_ptr);
			MPI_MYSEND(&tmpEnqueueWriteBuffer, sizeof(tmpEnqueueWriteBuffer), MPI_BYTE, 0,
					 ENQUEUE_WRITE_BUFFER, parentComm, &request1);
			free(host_ptr);
		}
		
		if (status.MPI_TAG == SET_KERNEL_ARG)
		{
			MPI_Recv(&tmpSetKernelArg, sizeof(tmpSetKernelArg), MPI_BYTE, 0,
					 SET_KERNEL_ARG, parentComm, &status);
			char *arg_value = (char *)malloc(tmpSetKernelArg.arg_size);
			MPI_Recv(arg_value, tmpSetKernelArg.arg_size, MPI_BYTE, 0,
					 SET_KERNEL_ARG1, parentComm, &status);
			mpiOpenCLSetKernelArg(&tmpSetKernelArg, arg_value);
			MPI_MYSEND(&tmpSetKernelArg, sizeof(tmpSetKernelArg), MPI_BYTE, 0,
					 SET_KERNEL_ARG, parentComm, &request1);
		}

		if (status.MPI_TAG == ENQUEUE_ND_RANGE_KERNEL)
		{
			MPI_Recv(&tmpEnqueueNDRangeKernel, sizeof(tmpEnqueueNDRangeKernel), MPI_BYTE, 0,
					 ENQUEUE_ND_RANGE_KERNEL, parentComm, &status);
			int work_dim = tmpEnqueueNDRangeKernel.work_dim;
			size_t *global_work_offset, *global_work_size, *local_work_size;
			global_work_offset = NULL;
			global_work_size   = NULL;
			local_work_size    = NULL;

			if (tmpEnqueueNDRangeKernel.global_work_offset_flag == 1)
			{
				//printf("global work offset\n");
				global_work_offset = (size_t *)malloc(work_dim * sizeof(size_t));
				MPI_Recv(global_work_offset, work_dim * sizeof(size_t), MPI_BYTE, 0,
						 ENQUEUE_ND_RANGE_KERNEL1, parentComm, &status);
				//for (i = 0; i < work_dim; i++)
				//{
				//	printf("globalWorkOffset[%d] = %ld\n", i, global_work_offset[i]);
				//}
			}

			if (tmpEnqueueNDRangeKernel.global_work_size_flag == 1)
			{
				//printf("global work size\n");
				global_work_size   = (size_t *)malloc(work_dim * sizeof(size_t));
				MPI_Recv(global_work_size, work_dim * sizeof(size_t), MPI_BYTE, 0,
						 ENQUEUE_ND_RANGE_KERNEL2, parentComm, &status);
				//for (i = 0; i < work_dim; i++)
				//{
				//	printf("globalWorkSize[%d] = %ld\n", i, global_work_size[i]);
				//}
			}

			if (tmpEnqueueNDRangeKernel.local_work_size_flag == 1)
			{
				//printf("local work size\n");
				local_work_size    = (size_t *)malloc(work_dim * sizeof(size_t));
				MPI_Recv(local_work_size, work_dim * sizeof(size_t), MPI_BYTE, 0,
						 ENQUEUE_ND_RANGE_KERNEL3, parentComm, &status);
				//for (i = 0; i < work_dim; i++)
				//{
				//	printf("localWorkSize[%d] = %ld\n", i, local_work_size[i]);
				//}
			}

			mpiOpenCLEnqueueNDRangeKernel(&tmpEnqueueNDRangeKernel,
										  global_work_offset,
										  global_work_size,
										  local_work_size);
			//printf("after kernel execution!\n");

			MPI_MYSEND(&tmpEnqueueNDRangeKernel, sizeof(tmpEnqueueNDRangeKernel), MPI_BYTE, 0,
					 ENQUEUE_ND_RANGE_KERNEL, parentComm, &request1);
			if (tmpEnqueueNDRangeKernel.global_work_offset_flag == 1)
			{
				free(global_work_offset);
			}

			if (tmpEnqueueNDRangeKernel.global_work_size_flag == 1)
			{
				free(global_work_size);
			}

			if (tmpEnqueueNDRangeKernel.local_work_size_flag == 1)
			{
				free(local_work_size);
			}
		}

		if (status.MPI_TAG == ENQUEUE_READ_BUFFER)
		{
			MPI_Recv(&tmpEnqueueReadBuffer, sizeof(tmpEnqueueReadBuffer), MPI_BYTE, 0,
					 ENQUEUE_READ_BUFFER, parentComm, &status);
			int bufSize = tmpEnqueueReadBuffer.cb;
			char *host_ptr = (char *)malloc(bufSize);
			mpiOpenCLEnqueueReadBuffer(&tmpEnqueueReadBuffer, host_ptr);
			MPI_MYSEND(&tmpEnqueueReadBuffer, sizeof(tmpEnqueueReadBuffer), MPI_BYTE, 0,
					 ENQUEUE_READ_BUFFER, parentComm, &request1);
			MPI_MYSEND(host_ptr, bufSize, MPI_BYTE, 0,
					 ENQUEUE_READ_BUFFER1, parentComm, &request2);
		}

		if (status.MPI_TAG == RELEASE_MEM_OBJ)
		{
			MPI_Recv(&tmpReleaseMemObject, sizeof(tmpReleaseMemObject), MPI_BYTE, 0,
					 RELEASE_MEM_OBJ, parentComm, &status);
			mpiOpenCLReleaseMemObject(&tmpReleaseMemObject);
			MPI_MYSEND(&tmpReleaseMemObject, sizeof(tmpReleaseMemObject), MPI_BYTE, 0,
					 RELEASE_MEM_OBJ, parentComm, &request1);
		}

		if (status.MPI_TAG == CL_RELEASE_KERNEL_FUNC)
		{
			MPI_Recv(&tmpReleaseKernel, sizeof(tmpReleaseKernel), MPI_BYTE, 0,
					 CL_RELEASE_KERNEL_FUNC, parentComm, &status);
			mpiOpenCLReleaseKernel(&tmpReleaseKernel);
			MPI_MYSEND(&tmpReleaseKernel, sizeof(tmpReleaseKernel), MPI_BYTE, 0,
					 CL_RELEASE_KERNEL_FUNC, parentComm, &request1);
		}

		if (status.MPI_TAG == FINISH_FUNC)
		{
			MPI_Recv(&tmpFinish, sizeof(tmpFinish), MPI_BYTE, 0,
					 FINISH_FUNC, parentComm, &status);
			mpiOpenCLFinish(&tmpFinish);
			MPI_MYSEND(&tmpFinish, sizeof(tmpFinish), MPI_BYTE, 0,
					 FINISH_FUNC, parentComm, &request1);
		}

		if (status.MPI_TAG == PROGRAM_END)
		{
			break;
		}
	}

	MPI_Comm_free(&parentComm);
	MPI_Finalize();

	return 0;
}

