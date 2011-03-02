#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <CL/opencl.h>
#include "slaveOpenCL.h"

/**************************************************************************
 *1. First version of the slave process
 *4. The function clGetDeviceIDs is modified
 *8. Enqueue read/write buffer is modified and other API functions are added.
 *9. change the way of set arguments to save data communication time
 *10.support asynchronous data transfer
 **************************************************************************/

#define GET_PLATFORM_ID_FUNC        10001
#define GET_PLATFORM_ID_FUNC1	    10002
#define GET_DEVICE_ID_FUNC          10003
#define GET_DEVICE_ID_FUNC1         10004
#define CREATE_CONTEXT_FUNC         10005
#define CREATE_CONTEXT_FUNC1        10006
#define CREATE_COMMAND_QUEUE_FUNC   10007
#define LOAD_SOURCE_FUNC            10008
#define LOAD_SOURCE_FUNC1           10009
#define CREATE_PROGRMA_WITH_SOURCE  10010
#define CREATE_PROGRMA_WITH_SOURCE1 10011
#define CREATE_PROGRMA_WITH_SOURCE2 10070
#define BUILD_PROGRAM               10012
#define BUILD_PROGRAM1              10013
#define CREATE_KERNEL               10014
#define CREATE_KERNEL1              10015
#define CREATE_BUFFER_FUNC          10016
#define CREATE_BUFFER_FUNC1         10017
#define ENQUEUE_WRITE_BUFFER        10018
#define ENQUEUE_WRITE_BUFFER1       10019
#define ENQUEUE_WRITE_BUFFER2       10020
#define SET_KERNEL_ARG              10021
#define SET_KERNEL_ARG1             10022
#define ENQUEUE_ND_RANGE_KERNEL     10023
#define ENQUEUE_ND_RANGE_KERNEL1    10024
#define ENQUEUE_ND_RANGE_KERNEL2    10025
#define ENQUEUE_ND_RANGE_KERNEL3    10026
#define ENQUEUE_ND_RANGE_KERNEL4    10069
#define ENQUEUE_READ_BUFFER         10027
#define ENQUEUE_READ_BUFFER1        10028
#define RELEASE_MEM_OBJ             10029
#define FINISH_FUNC                 10030
#define CL_RELEASE_KERNEL_FUNC      10031
#define GET_CONTEXT_INFO_FUNC	    10032
#define GET_CONTEXT_INFO_FUNC1	    10033
#define GET_BUILD_INFO_FUNC         10034
#define GET_BUILD_INFO_FUNC1        10035
#define GET_PROGRAM_INFO_FUNC       10036
#define GET_PROGRAM_INFO_FUNC1      10037
#define REL_PROGRAM_FUNC            10038
#define REL_COMMAND_QUEUE_FUNC      10039
#define REL_CONTEXT_FUNC            10040
#define GET_DEVICE_INFO_FUNC        10041
#define GET_DEVICE_INFO_FUNC1       10042
#define GET_PLATFORM_INFO_FUNC      10043
#define GET_PLATFORM_INFO_FUNC1     10044
#define FLUSH_FUNC				    10045
#define WAIT_FOR_EVENT_FUNC         10046
#define WAIT_FOR_EVENT_FUNC1        10047
#define CREATE_SAMPLER_FUNC         10048
#define GET_CMD_QUEUE_INFO_FUNC 	10049
#define GET_CMD_QUEUE_INFO_FUNC1 	10050
#define ENQUEUE_MAP_BUFF_FUNC       10051
#define ENQUEUE_MAP_BUFF_FUNC1      10052
#define RELEASE_EVENT_FUNC          10053
#define GET_EVENT_PROF_INFO_FUNC    10054
#define GET_EVENT_PROF_INFO_FUNC1   10055
#define RELEASE_SAMPLER_FUNC        10056
#define GET_KERNEL_WGP_INFO_FUNC    10057
#define GET_KERNEL_WGP_INFO_FUNC1   10058
#define CREATE_IMAGE_2D_FUNC        10059
#define CREATE_IMAGE_2D_FUNC1       10060
#define ENQ_COPY_BUFF_FUNC          10061
#define ENQ_COPY_BUFF_FUNC1         10062
#define RETAIN_EVENT_FUNC           10063
#define RETAIN_MEMOBJ_FUNC          10064
#define RETAIN_KERNEL_FUNC          10065
#define RETAIN_CMDQUE_FUNC          10066
#define ENQ_UNMAP_MEMOBJ_FUNC       10067
#define ENQ_UNMAP_MEMOBJ_FUNC1      10068
#define PROGRAM_END                 11111



cl_int err;
typedef struct strKernelArgs {
	cl_uint arg_index;
	size_t  arg_size;
	char    arg_value[64];
	cl_char arg_null_flag;
} kernel_args;

//1-----------------------------------------------------------------------
struct strGetPlatformIDs {
	cl_uint          num_entries;
	cl_platform_id   *platforms;
	cl_uint          num_platforms;
	cl_int           res;
};

struct strGetPlatformIDs getPlatformIDStr;
void mpiOpenCLGetPlatformIDs(struct strGetPlatformIDs *tmpGetPlatform, cl_platform_id *platforms)
{
	cl_uint *num_platforms = NULL;
	if (tmpGetPlatform->num_platforms == 1)
	{
		num_platforms = &tmpGetPlatform->num_platforms;
	}

	err = clGetPlatformIDs(tmpGetPlatform->num_entries, platforms, num_platforms);
	tmpGetPlatform->res = err;

	//printf("In mpiOpenCLGetPlatformIDs function!\n");
	return;
}

//2-----------------------------------------------------------------------
struct strGetDeviceIDs
{
	cl_platform_id   platform;
	cl_device_type   device_type;
	cl_uint          num_entries;
	cl_device_id     *devices;
	cl_uint          num_devices;
	cl_int           res;
};

struct strGetDeviceIDs tmpGetDeviceIDs;
void mpiOpenCLGetDeviceIDs(struct strGetDeviceIDs *tmpGetDeviceIDs, 
						   cl_device_id *devices)
{
	//printf("In mpiOpenCLGetDeviceIDs function1!\n");
	cl_platform_id platform    = tmpGetDeviceIDs->platform;
	cl_device_type device_type = tmpGetDeviceIDs->device_type;
	cl_uint        num_entries = tmpGetDeviceIDs->num_entries;

	cl_uint *num_device_ptr = NULL;
	if (tmpGetDeviceIDs->num_devices == 1)
	{
		num_device_ptr = &tmpGetDeviceIDs->num_devices;
	}

	err = clGetDeviceIDs(platform, 
						 device_type, 
						 num_entries, 
						 devices, 
						 num_device_ptr);
	tmpGetDeviceIDs->res     = err;
	
	//printf("In mpiOpenCLGetDeviceIDs function2!\n");

	return;
}

//3-----------------------------------------------------------------------
struct strCreateContext {
	cl_context_properties   properties;
	cl_uint                       num_devices;
	cl_device_id                  *devices;
	//CL_CALLBACK *                 pfn_notify;
	void *                        user_data;
	cl_int                        errcode_ret;
	cl_context                    hContext;
};

struct strCreateContext tmpCreateContext;
void mpiOpenCLCreateContext(struct strCreateContext *tmpCreateContext, cl_device_id *devices)
{
	cl_uint num_devices = tmpCreateContext->num_devices;
	//const cl_context_properties properties = tmpCreateContext->properties;
	//const cl_device_id devices = tmpCreateContext->devices;
	cl_context hContext = clCreateContext(0, num_devices, devices, 0, 0, &err);
	tmpCreateContext->hContext = hContext;
	tmpCreateContext->errcode_ret = err;
}

//4-----------------------------------------------------------------------
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

	cl_command_queue hCmdQueue = clCreateCommandQueue(hInContext, device, properties, &err);

	//store the command queue locally
	createCommandQueue(hCmdQueue);

	tmpCreateCommandQueue->errcode_ret = err;

	tmpCreateCommandQueue->clCommand = hCmdQueue;
}

//5-----------------------------------------------------------------------
struct strCreateProgramWithSource {
	cl_context        context;
	cl_uint           count;
	size_t            lengths;
	cl_program        clProgram;
	cl_int            errcode_ret;
};

struct strCreateProgramWithSource tmpCreateProgramWithSource;
void mpiOpenCLCreateProgramWithSource(struct strCreateProgramWithSource *tmpCreateProgramWithSource, 
									  char *cSourceCL, size_t *lengthsArray)
{
	size_t sourceFileSize = tmpCreateProgramWithSource->lengths;
	int count = tmpCreateProgramWithSource->count;
	cl_context hInContext = tmpCreateProgramWithSource->context;
	cl_uint strIndex;
	size_t strStartLoc;
	//transform the whole string buffer to multiple strings
	char **strings = (char **)malloc(count * sizeof(char *));
	strStartLoc = 0;
	for (strIndex = 0; strIndex < count; strIndex++)
	{
		strings[strIndex] = (char *)malloc(lengthsArray[strIndex] + 1);
		memcpy(strings[strIndex], &cSourceCL[strStartLoc], lengthsArray[strIndex] * sizeof(char));
		strStartLoc += lengthsArray[strIndex];
		strings[strIndex][lengthsArray[strIndex]] = '\0';
	}

    cl_program hProgram = clCreateProgramWithSource(hInContext, count, (const char **)strings, 
										 lengthsArray, &err);
	tmpCreateProgramWithSource->clProgram = hProgram;
	tmpCreateProgramWithSource->errcode_ret = err;
	for (strIndex = 0; strIndex < count; strIndex++)
	{
		free(strings[strIndex]);
	}
	free(strings);
}

//6-----------------------------------------------------------------------
struct strBuildProgram {
	cl_program           program;
	cl_uint              num_devices;
	cl_device_id        *device_list;
	cl_uint              optionLen;
	//CL_CALLBACK *      pfn_notify;
	void *               user_data;
	cl_int               res;
};

struct strBuildProgram tmpBuildProgram;
void mpiOpenCLBuildProgram(struct strBuildProgram *tmpBuildProgram, 
						   char *options, cl_device_id *device_list)
{
	cl_program hInProgram = tmpBuildProgram->program;
	cl_uint num_devices = tmpBuildProgram->num_devices;
	err = clBuildProgram(hInProgram, num_devices, device_list, options, 0, 0);

	tmpBuildProgram->res = err;
}

//7-----------------------------------------------------------------------
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
	cl_kernel hKernel = clCreateKernel(hInProgram, kernel_name, &err);
	tmpCreateKernel->kernel = hKernel;
	tmpCreateKernel->errcode_ret = err;
}
	
//8-----------------------------------------------------------------------
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
	size_t bufferSize = tmpCreateBuffer->size;
	cl_mem deviceMem;
	deviceMem = clCreateBuffer(hInContext,
							   flags,
							   bufferSize,
							   host_ptr,
							   &err);
	if (err != CL_SUCCESS)
	{
		printf("In slave, create buffer error!\n");
	}

	tmpCreateBuffer->errcode_ret = err;
	tmpCreateBuffer->deviceMem = deviceMem;
}

//9-----------------------------------------------------------------------
struct strEnqueueWriteBuffer {
	cl_command_queue   command_queue;
	cl_mem             buffer;
	cl_bool            blocking_write;
	size_t             offset;
	size_t             cb;
	cl_uint            num_events_in_wait_list;
	cl_int             event_null_flag;   //1, flag is NULL, 0, is NOT NULL
	cl_event           event;
	cl_int             res;
};

struct strEnqueueWriteBuffer tmpEnqueueWriteBuffer;
void mpiOpenCLEnqueueWriteBuffer(struct strEnqueueWriteBuffer *tmpEnqueueWriteBuffer,
								 void *ptr, cl_event *event_wait_list)
{
	cl_command_queue hInCmdQueue = tmpEnqueueWriteBuffer->command_queue;
	cl_mem deviceMem = tmpEnqueueWriteBuffer->buffer;
	cl_bool blocking_write = tmpEnqueueWriteBuffer->blocking_write;
	size_t offset = tmpEnqueueWriteBuffer->offset;
	size_t cb = tmpEnqueueWriteBuffer->cb;
	cl_uint num_events_in_wait_list = tmpEnqueueWriteBuffer->num_events_in_wait_list;
//	cl_event *event_ret = NULL;
//	if (tmpEnqueueWriteBuffer->event_null_flag == 0)
//	{
//		event_ret = &tmpEnqueueWriteBuffer->event;
//	}
	cl_event *event_ret = &tmpEnqueueWriteBuffer->event;

	err = clEnqueueWriteBuffer(hInCmdQueue, deviceMem, blocking_write, offset,
							   cb, ptr, num_events_in_wait_list, event_wait_list, event_ret);
	tmpEnqueueWriteBuffer->res = err; 
}

//10-----------------------------------------------------------------------
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

//11-----------------------------------------------------------------------
struct strEnqueueNDRangeKernel {
	cl_command_queue command_queue;
	cl_kernel        kernel;
	cl_uint          work_dim;
	cl_int           global_work_offset_flag;
	cl_int           global_work_size_flag;
	cl_int           local_work_size_flag;
	cl_uint          args_num;
	cl_uint          num_events_in_wait_list;
	cl_int           event_null_flag;
	cl_event         event;
	cl_int           res;
};

struct strEnqueueNDRangeKernel tmpEnqueueNDRangeKernel;
void mpiOpenCLEnqueueNDRangeKernel(struct strEnqueueNDRangeKernel *tmpEnqueueNDRangeKernel,
							  size_t     *global_work_offset,
							  size_t     *global_work_size,
							  size_t     *local_work_size,
							  kernel_args *args_ptr)
{
	cl_command_queue hInCommand = tmpEnqueueNDRangeKernel->command_queue;
	cl_kernel hInKernel = tmpEnqueueNDRangeKernel->kernel;
	cl_uint work_dim = tmpEnqueueNDRangeKernel->work_dim;
	cl_uint num_events_in_wait_list = tmpEnqueueNDRangeKernel->num_events_in_wait_list;
	cl_uint args_num = tmpEnqueueNDRangeKernel->args_num;
	cl_uint args_index;

	//call real opencl functions to set arguments
	//void *arg_value;
	for (args_index = 0; args_index < args_num; args_index++)
	{
		if (args_ptr[args_index].arg_null_flag == 1)
		{
			err = clSetKernelArg(hInKernel, 
								 args_ptr[args_index].arg_index,
								 args_ptr[args_index].arg_size,
								 NULL);
		}
		else
		{
			err = clSetKernelArg(hInKernel, 
								 args_ptr[args_index].arg_index,
								 args_ptr[args_index].arg_size,
								 (const void *)args_ptr[args_index].arg_value);
		}

	}

	cl_event *event_ptr = NULL;
	if (tmpEnqueueNDRangeKernel->event_null_flag == 0)
	{
		event_ptr = &tmpEnqueueNDRangeKernel->event;
	}

	err = clEnqueueNDRangeKernel(hInCommand,
								 hInKernel,
								 work_dim,
								 global_work_offset,
								 global_work_size,
								 local_work_size,
								 0,
								 //num_events_in_wait_list,
								 NULL,
								 event_ptr);

	tmpEnqueueNDRangeKernel->res = err;
}

//12-----------------------------------------------------------------------
struct strEnqueueReadBuffer {
	cl_command_queue    command_queue;
	cl_mem              buffer;
	cl_bool             blocking_read;
	cl_uint             readBufferTag;
	size_t              offset;
	size_t              cb;
	cl_uint             num_events_in_wait_list;
	cl_int              event_null_flag;  //1: the event point is NULL. 0: the event point is NOT NULL
	cl_event            event;
	cl_int              res;
};

struct strEnqueueReadBuffer tmpEnqueueReadBuffer;
void mpiOpenCLEnqueueReadBuffer(struct strEnqueueReadBuffer *tmpEnqueueReadBuffer, 
								void *ptr, cl_event *event_wait_list)
{
	cl_command_queue hInCmdQueue = tmpEnqueueReadBuffer->command_queue;
	cl_mem deviceMem = tmpEnqueueReadBuffer->buffer;
	cl_bool read_flag = tmpEnqueueReadBuffer->blocking_read;
	size_t  offset = tmpEnqueueReadBuffer->offset;
	size_t  cb = tmpEnqueueReadBuffer->cb;
	cl_uint num_events_in_wait_list = tmpEnqueueReadBuffer->num_events_in_wait_list;
//	cl_event *event_ret = NULL;
//	if (tmpEnqueueReadBuffer->event_null_flag == 0)
//	{
//		event_ret = &tmpEnqueueReadBuffer->event;
//	}

	cl_event *event_ret = &tmpEnqueueReadBuffer->event;

	err = clEnqueueReadBuffer(hInCmdQueue,
							deviceMem,
							read_flag,
							offset,
							cb,
							ptr,
							num_events_in_wait_list,
							event_wait_list,
							event_ret);
	tmpEnqueueReadBuffer->res = err;
}

//13-----------------------------------------------------------------------
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

//14-----------------------------------------------------------------------
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

//15-----------------------------------------------------------------------
struct strGetContextInfo {
	cl_context         context;
	cl_context_info    param_name;
	size_t             param_value_size;
	void *             param_value;  
	size_t             param_value_size_ret;
	cl_int             res;
};

struct strGetContextInfo tmpGetContextInfo;
void mpiOpenCLGetContextInfo(struct strGetContextInfo *tmpGetContextInfo, void *param_value)
{
	cl_int errcode;
	cl_context context = tmpGetContextInfo->context;
	cl_context_info param_name = tmpGetContextInfo->param_name;
	size_t param_value_size = tmpGetContextInfo->param_value_size;
	size_t *value_size_ptr = NULL;
	if (tmpGetContextInfo->param_value_size_ret == 1)
	{
		value_size_ptr = &tmpGetContextInfo->param_value_size_ret;
	}
	errcode = clGetContextInfo(context, 
							   param_name, 
							   param_value_size, 
							   param_value, 
							   value_size_ptr);

	tmpGetContextInfo->res = errcode;
}

//16-----------------------------------------------------------------------
struct strGetProgramBuildInfo {
	cl_program            program;
	cl_device_id          device;
	cl_program_build_info param_name;
	size_t                param_value_size;
	void *                param_value;
	size_t                param_value_size_ret;
	cl_int                res;
};

struct strGetProgramBuildInfo tmpGetProgramBuildInfo;
void mpiOpenCLGetProgramBuildInfo(struct strGetProgramBuildInfo *tmpGetProgramBuildInfo, void *param_value)
{
	cl_int errcode;
	cl_program program = tmpGetProgramBuildInfo->program;
	cl_device_id device = tmpGetProgramBuildInfo->device;
	cl_program_build_info param_name = tmpGetProgramBuildInfo->param_name;
	size_t param_value_size = tmpGetProgramBuildInfo->param_value_size;
	size_t *value_size_ptr = NULL;
	if (tmpGetProgramBuildInfo->param_value_size_ret == 1)
	{
		value_size_ptr = &tmpGetProgramBuildInfo->param_value_size_ret;
	}
	errcode = clGetProgramBuildInfo(program,
									device,
									param_name,
									param_value_size,
									param_value,
									value_size_ptr);
	tmpGetProgramBuildInfo->res = errcode;
}

//17-----------------------------------------------------------------------
struct strGetProgramInfo {	
	cl_program         program;
	cl_program_info    param_name;
	size_t             param_value_size;
	void *             param_value;
	size_t             param_value_size_ret;
	cl_int             res;
};

struct strGetProgramInfo tmpGetProgramInfo;
void mpiOpenCLGetProgramInfo(struct strGetProgramInfo *tmpGetProgramInfo, void *param_value)
{
	cl_int errcode;
	cl_program program = tmpGetProgramInfo->program;
	cl_program_info param_name = tmpGetProgramInfo->param_name;
	size_t param_value_size = tmpGetProgramInfo->param_value_size;
	size_t *value_size_ptr = NULL;
	if (tmpGetProgramInfo->param_value_size_ret == 1)
	{
		value_size_ptr = &tmpGetProgramInfo->param_value_size_ret;
	}
	errcode = clGetProgramInfo(program,
							   param_name,
							   param_value_size,
							   param_value,
							   value_size_ptr);
	tmpGetProgramInfo->res = errcode;
}

//18-----------------------------------------------------------------------
struct strReleaseProgram {
	cl_program  program;
	cl_int      res;
};

struct strReleaseProgram tmpReleaseProgram;
void mpiOpenCLReleaseProgram(struct strReleaseProgram *tmpReleaseProgram)
{
	cl_int errcode;
	cl_program program = tmpReleaseProgram->program;
	errcode = clReleaseProgram(program);
	tmpReleaseProgram->res = errcode;
}

//19-----------------------------------------------------------------------
struct strReleaseCommandQueue {
	cl_command_queue command_queue;
	cl_int           res;
};

struct strReleaseCommandQueue tmpReleaseCommandQueue;
void mpiOpenCLReleaseCommandQueue(struct strReleaseCommandQueue *tmpReleaseCommandQueue)
{
	cl_int errcode;
	cl_command_queue command_queue = tmpReleaseCommandQueue->command_queue;

	//release command information for data copy
	releaseCommandQueue(command_queue);

	errcode = clReleaseCommandQueue(command_queue);
	tmpReleaseCommandQueue->res = errcode;
}

//20-----------------------------------------------------------------------
struct strReleaseContext {
	cl_context context;
	cl_int     res;
};

struct strReleaseContext tmpReleaseContext;
void mpiOpenCLReleaseContext(struct strReleaseContext *tmpReleaseContext)
{
	cl_int errcode;
	cl_context context = tmpReleaseContext->context;
	errcode = clReleaseContext(context);
	tmpReleaseContext->res = errcode;
}

//21-----------------------------------------------------------------------
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

//22-------------------------------------------------------------------------
struct strGetDeviceInfo {
	cl_device_id    device;
	cl_device_info  param_name;
	size_t          param_value_size;
	void *          param_value;
	size_t          param_value_size_ret;
	cl_int          res;
};

struct strGetDeviceInfo tmpGetDeviceInfo;
void mpiOpenCLGetDeviceInfo(struct strGetDeviceInfo *tmpGetDeviceInfo, void *param_value)
{
	cl_int errcode;
	cl_device_id device = tmpGetDeviceInfo->device;
	cl_device_info param_name = tmpGetDeviceInfo->param_name;
	size_t param_value_size = tmpGetDeviceInfo->param_value_size;
	size_t *value_size_ptr = NULL;
	if (tmpGetDeviceInfo->param_value_size_ret == 1)
	{
		value_size_ptr = &tmpGetDeviceInfo->param_value_size_ret;
	}
	errcode = clGetDeviceInfo(device,
							  param_name,
							  param_value_size,
							  param_value,
							  value_size_ptr);
	tmpGetDeviceInfo->res = errcode;
}

//23-------------------------------------------------------------------------
struct strGetPlatformInfo {
	cl_platform_id    platform;
	cl_platform_info  param_name;
	size_t            param_value_size;
	void *            param_value;
	size_t            param_value_size_ret;
	cl_int            res;
};

struct strGetPlatformInfo tmpGetPlatformInfo;
void mpiOpenCLGetPlatformInfo(struct strGetPlatformInfo *tmpGetPlatformInfo, void *param_value)
{
	cl_int errcode;
	cl_platform_id platform = tmpGetPlatformInfo->platform;
	cl_platform_info param_name = tmpGetPlatformInfo->param_name;
	size_t param_value_size = tmpGetPlatformInfo->param_value_size;
	size_t *value_size_ptr = NULL;
	if (tmpGetPlatformInfo->param_value_size_ret == 1)
	{
		value_size_ptr = &tmpGetPlatformInfo->param_value_size_ret;
	}
	errcode = clGetPlatformInfo(platform,
							    param_name,
							    param_value_size,
							    param_value,
							    value_size_ptr);
	tmpGetPlatformInfo->res = errcode;
}

//24-----------------------------------------------------------------------
struct strFlush {
	cl_command_queue command_queue;
	cl_int res;
};

struct strFlush tmpFlush;
void mpiOpenCLFlush(struct strFlush *tmpFlush)
{
	cl_command_queue hInCmdQueue = tmpFlush->command_queue;
	err = clFlush(hInCmdQueue);
	tmpFlush->res = err;
}

//25-----------------------------------------------------------------------
struct strWaitForEvents {
	cl_uint  num_events;
	cl_int   res;
};

void mpiOpenCLWaitForEvents(struct strWaitForEvents *tmpWaitForEvents, cl_event *event_list)
{
	cl_uint num_events = tmpWaitForEvents->num_events;
	err = clWaitForEvents(num_events, event_list);
	tmpWaitForEvents->res = err;
}


//26-----------------------------------------------------------------------
struct strCreateSampler {
	cl_context          context;
	cl_bool             normalized_coords;
	cl_addressing_mode  addressing_mode;
	cl_filter_mode      filter_mode;
	cl_int              errcode_ret;
	cl_sampler          sampler;
};

struct strCreateSampler tmpCreateSampler;
void mpiOpenCLCreateSampler(struct strCreateSampler *tmpCreateSampler)
{
	cl_context context = tmpCreateSampler->context;
	cl_bool    normalized_coords = tmpCreateSampler->normalized_coords;
	cl_addressing_mode addressing_mode = tmpCreateSampler->addressing_mode;
	cl_filter_mode filter_mode = tmpCreateSampler->filter_mode;
	cl_int *errcode = NULL;
	if (tmpCreateSampler->errcode_ret == 1)
	{
		errcode = &tmpCreateSampler->errcode_ret;
	}

	tmpCreateSampler->sampler = 
	clCreateSampler(context, 
					normalized_coords, 
					addressing_mode, 
					filter_mode, 
					errcode);
}

//27-----------------------------------------------------------------------
struct strGetCommandQueueInfo {
	cl_command_queue      command_queue;
	cl_command_queue_info param_name;
	size_t                param_value_size;
	void *                param_value;
	size_t                param_value_size_ret;
	cl_int                res;
};

struct strGetCommandQueueInfo tmpGetCommandQueueInfo;
void mpiOpenCLGetCommandQueueInfo(struct strGetCommandQueueInfo *tmpGetCommandQueueInfo,
								  void *param_value)
{
	cl_int errcode;
	cl_command_queue command_queue = tmpGetCommandQueueInfo->command_queue;
	cl_command_queue_info param_name = tmpGetCommandQueueInfo->param_name;
	size_t param_value_size = tmpGetCommandQueueInfo->param_value_size;
	size_t *value_size_ptr = NULL;
	if (tmpGetCommandQueueInfo->param_value_size_ret == 1)
	{
		value_size_ptr = &tmpGetCommandQueueInfo->param_value_size_ret;
	}
	
	errcode = clGetCommandQueueInfo(command_queue,
							        param_name,
							        param_value_size,
							        param_value,
							        value_size_ptr);
	tmpGetCommandQueueInfo->res = errcode;
}

//28-----------------------------------------------------------------------
struct strEnqueueMapBuffer {
	cl_command_queue command_queue;
	cl_mem           buffer;
	cl_bool          blocking_map;
	cl_map_flags     map_flags;
	size_t           offset;
	size_t           cb;
	cl_uint          num_events_in_wait_list;
	cl_int           event_null_flag; //1: NULL, 0: NOT NULL
	cl_event         event;
	cl_int           errcode_ret;
	void             *ret_ptr;
};

struct strEnqueueMapBuffer tmpEnqueueMapBuffer;
void mpiOpenCLEnqueueMapBuffer(struct strEnqueueMapBuffer *tmpEnqueueMapBuffer,
							   cl_event * event_wait_list)
{
	cl_command_queue command_queue = tmpEnqueueMapBuffer->command_queue;
	cl_mem buffer = tmpEnqueueMapBuffer->buffer;
	cl_bool blocking_map = tmpEnqueueMapBuffer->blocking_map;
	cl_map_flags map_flags = tmpEnqueueMapBuffer->map_flags;
	size_t offset = tmpEnqueueMapBuffer->offset;
	size_t cb = tmpEnqueueMapBuffer->cb;
	cl_uint num_events_in_wait_list = tmpEnqueueMapBuffer->num_events_in_wait_list;
	cl_event *event = NULL;
	if (tmpEnqueueMapBuffer->event_null_flag == 0)
	{
		event = &tmpEnqueueMapBuffer->event;
	}
	cl_int *errcode_ptr = NULL;
	if (tmpEnqueueMapBuffer->errcode_ret == 0)
	{
		errcode_ptr = &tmpEnqueueMapBuffer->errcode_ret;
	}

	tmpEnqueueMapBuffer->ret_ptr = 
		clEnqueueMapBuffer(command_queue,
						   buffer,
						   blocking_map,
						   map_flags,
						   offset,
						   cb,
						   num_events_in_wait_list,
						   event_wait_list,
						   event,
						   errcode_ptr);

}

//29--------------------------------------------------------------------
struct strReleaseEvent {
	cl_event         event;
	cl_int           res;
};

struct strReleaseEvent tmpReleaseEvent;
void mpiOpenCLReleaseEvent(struct strReleaseEvent *tmpReleaseEvent)
{
	cl_event event = tmpReleaseEvent->event;
	err = clReleaseEvent(event);
	tmpReleaseEvent->res = err;
}

//30----------------------------------------------------------------------
struct strGetEventProfilingInfo {
	cl_event          event;
	cl_profiling_info param_name;
	size_t            param_value_size;
	void *            param_value;
	size_t            param_value_size_ret;
	cl_int            res;
};

struct strGetEventProfilingInfo tmpGetEventProfilingInfo;
void mpiOpenCLGetEventProfilingInfo(struct strGetEventProfilingInfo *tmpGetEventProfilingInfo,
								  void *param_value)
{
	cl_int errcode;
	cl_event event = tmpGetEventProfilingInfo->event;
	cl_profiling_info param_name = tmpGetEventProfilingInfo->param_name;
	size_t param_value_size = tmpGetEventProfilingInfo->param_value_size;
	size_t *value_size_ptr = NULL;
	if (tmpGetEventProfilingInfo->param_value_size_ret == 1)
	{
		value_size_ptr = &tmpGetEventProfilingInfo->param_value_size_ret;
	}
	errcode = clGetEventProfilingInfo(event,
							          param_name,
							          param_value_size,
							          param_value,
							          value_size_ptr);
	tmpGetEventProfilingInfo->res = errcode;
}

//31--------------------------------------------------------------------
struct strReleaseSampler {
	cl_sampler       sampler;
	cl_int           res;
};

struct strReleaseSampler tmpReleaseSampler;
void mpiOpenCLReleaseSampler(struct strReleaseSampler *tmpReleaseSampler)
{
	cl_sampler sampler = tmpReleaseSampler->sampler;
	err = clReleaseSampler(sampler);
	tmpReleaseSampler->res = err;
}

//32--------------------------------------------------------------------
struct strGetKernelWorkGroupInfo {
   cl_kernel                  kernel;
   cl_device_id               device;
   cl_kernel_work_group_info  param_name;
   size_t                     param_value_size;
   void *                     param_value;
   size_t                     param_value_size_ret;
   cl_int                     res;
};

struct strGetKernelWorkGroupInfo tmpGetKernelWorkGroupInfo;
void mpiOpenCLGetKernelWorkGroupInfo(struct strGetKernelWorkGroupInfo *tmpGetKernelWorkGroupInfo,
								  void *param_value)
{
	cl_int errcode;
	cl_kernel kernel = tmpGetKernelWorkGroupInfo->kernel;
	cl_device_id device = tmpGetKernelWorkGroupInfo->device;
	cl_kernel_work_group_info param_name = tmpGetKernelWorkGroupInfo->param_name;
	size_t param_value_size = tmpGetKernelWorkGroupInfo->param_value_size;
	size_t *value_size_ptr = NULL;
	if (tmpGetKernelWorkGroupInfo->param_value_size_ret == 1)
	{
		value_size_ptr = &tmpGetKernelWorkGroupInfo->param_value_size_ret;
	}
	errcode = clGetKernelWorkGroupInfo(kernel,
									   device,
							           param_name,
							           param_value_size,
							           param_value,
							           value_size_ptr);
	tmpGetKernelWorkGroupInfo->res = errcode;
}

//33---------------------------------------------------------------------
struct strCreateImage2D {
   cl_context              context;
   cl_mem_flags            flags;
   cl_image_format         img_format;
   size_t                  image_width;
   size_t                  image_height;
   size_t                  image_row_pitch;
   size_t                  host_buff_size;
   cl_int                  errcode_ret;
   cl_mem                  mem_obj;
};

struct strCreateImage2D tmpCreateImage2D;
void mpiOpenCLCreateImage2D(struct strCreateImage2D *tmpCreateImage2D, void *host_ptr)
{
	cl_context context = tmpCreateImage2D->context;
	cl_mem_flags flags = tmpCreateImage2D->flags;
	cl_image_format img_format;
	img_format.image_channel_order = tmpCreateImage2D->img_format.image_channel_order;
	img_format.image_channel_data_type = tmpCreateImage2D->img_format.image_channel_data_type;
	size_t image_width = tmpCreateImage2D->image_width;
	size_t image_height = tmpCreateImage2D->image_height;
	size_t image_row_pitch = tmpCreateImage2D->image_row_pitch;
	cl_int *errcode_ptr = NULL;
	if (tmpCreateImage2D->errcode_ret == 0)
	{
		errcode_ptr = &tmpCreateImage2D->errcode_ret;
	}

	tmpCreateImage2D->mem_obj = 
		clCreateImage2D(context,
						flags,
						&img_format,
						image_width,
						image_height,
						image_row_pitch,
						host_ptr,
						errcode_ptr);
}

//34-----------------------------------------------------------------------
struct strEnqueueCopyBuffer {
   cl_command_queue    command_queue;
   cl_mem              src_buffer;
   cl_mem              dst_buffer;
   size_t              src_offset;
   size_t              dst_offset;
   size_t              cb;
   cl_uint             num_events_in_wait_list;
   cl_int              event_null_flag;
   cl_event            event;
   cl_int              res;
};

struct strEnqueueCopyBuffer tmpEnqueueCopyBuffer;
void mpiOpenCLEnqueueCopyBuffer(struct strEnqueueCopyBuffer *tmpEnqueueCopyBuffer, 
								cl_event * event_wait_list)
{
	cl_int errcode;
	cl_command_queue command_queue = tmpEnqueueCopyBuffer->command_queue;
	cl_mem src_buffer = tmpEnqueueCopyBuffer->src_buffer;
	cl_mem dst_buffer = tmpEnqueueCopyBuffer->dst_buffer;
	size_t src_offset = tmpEnqueueCopyBuffer->src_offset;
	size_t dst_offset = tmpEnqueueCopyBuffer->dst_offset;
	size_t cb = tmpEnqueueCopyBuffer->cb;
	cl_uint num_events_in_wait_list = tmpEnqueueCopyBuffer->num_events_in_wait_list;
	cl_event *event_ptr = NULL;
	if (tmpEnqueueCopyBuffer->event_null_flag == 0)
	{
		event_ptr = &tmpEnqueueCopyBuffer->event;
	}

	errcode = clEnqueueCopyBuffer(command_queue,
								  src_buffer,
								  dst_buffer,
								  src_offset,
								  dst_offset,
								  cb,
								  num_events_in_wait_list,
								  event_wait_list,
								  event_ptr);
	tmpEnqueueCopyBuffer->res = errcode;
}

//35-----------------------------------------------------------------------
struct strRetainEvent {
	cl_event event;
	cl_int   res;
};

struct strRetainEvent tmpRetainEvent;
void mpiOpenCLRetainEvent(struct strRetainEvent *tmpRetainEvent)
{
	cl_int errcode;
	cl_event event = tmpRetainEvent->event;
	errcode = clRetainEvent(event);
	tmpRetainEvent->res = errcode;
}

//36-----------------------------------------------------------------------
struct strRetainMemObject {
	cl_mem   memobj; 
	cl_int   res;
};

struct strRetainMemObject tmpRetainMemObject;
void mpiOpenCLRetainMemObject(struct strRetainMemObject *tmpRetainMemObject)
{
	cl_int errcode;
	cl_mem memobj = tmpRetainMemObject->memobj;
	errcode = clRetainMemObject(memobj);
	tmpRetainMemObject->res = errcode;
}


//37-----------------------------------------------------------------------
struct strRetainKernel {
	cl_kernel    kernel;
	cl_int       res;
};

struct strRetainKernel tmpRetainKernel;
void mpiOpenCLRetainKernel(struct strRetainKernel *tmpRetainKernel)
{
	cl_int errcode;
	cl_kernel kernel = tmpRetainKernel->kernel;
	errcode = clRetainKernel(kernel);
	tmpRetainKernel->res = errcode;
}

//38-----------------------------------------------------------------------
struct strRetainCommandQueue {
	cl_command_queue command_queue;
	cl_int           res;
};

struct strRetainCommandQueue tmpRetainCommandQueue;
void mpiOpenCLRetainCommandQueue(struct strRetainCommandQueue *tmpRetainCommandQueue)
{
	cl_int errcode;
	cl_command_queue command_queue = tmpRetainCommandQueue->command_queue;
	errcode = clRetainCommandQueue(command_queue);
	tmpRetainCommandQueue->res = errcode;
}

//39-----------------------------------------------------------------------
struct strEnqueueUnmapMemObject {
   cl_command_queue  command_queue;
   cl_mem            memobj;
   void *            mapped_ptr;
   cl_uint           num_events_in_wait_list;
   cl_int            event_null_flag;
   cl_event          event;
   cl_int            res;
};

struct strEnqueueUnmapMemObject tmpEnqueueUnmapMemObject;
void mpiOpenCLEnqueueUnmapMemObject(struct strEnqueueUnmapMemObject *tmpEnqueueUnmapMemObject, 
									cl_event *event_wait_list)
{
	cl_int errcode;
	cl_command_queue command_queue = tmpEnqueueUnmapMemObject->command_queue;
	cl_mem memobj = tmpEnqueueUnmapMemObject->memobj;
	void *mapped_ptr = tmpEnqueueUnmapMemObject->mapped_ptr;
	cl_uint num_events_in_wait_list= tmpEnqueueUnmapMemObject->num_events_in_wait_list;
	cl_event *event_ptr = NULL;
	if (tmpEnqueueUnmapMemObject->event_null_flag == 0)
	{
		event_ptr = &tmpEnqueueUnmapMemObject->event;
	}
	errcode = clEnqueueUnmapMemObject(command_queue,
									  memobj,
									  mapped_ptr,
									  num_events_in_wait_list,
									  event_wait_list,
									  event_ptr);
	tmpEnqueueUnmapMemObject->res = errcode;
}

//=========================================================================
int main(int argc, char *argv[])
{
	int rank, i;
	MPI_Status status;
	MPI_Request request;
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
		//MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, parentComm, &status);
		MPI_Probe(0, MPI_ANY_TAG, parentComm, &status);
		if (status.MPI_TAG == GET_PLATFORM_ID_FUNC)
		{
			MPI_Recv(&getPlatformIDStr, sizeof(getPlatformIDStr), MPI_BYTE, 0,
					 GET_PLATFORM_ID_FUNC, parentComm, &status);
			//printf("In slave process, in function mpiOpenCLGetPlatformIDs!\n");
			cl_platform_id *platforms = NULL;
			if (getPlatformIDStr.platforms != NULL)
			{
				platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * getPlatformIDStr.num_entries);
			}

			mpiOpenCLGetPlatformIDs(&getPlatformIDStr, platforms);
			MPI_Send(&getPlatformIDStr, sizeof(getPlatformIDStr), MPI_BYTE, 0,
					 GET_PLATFORM_ID_FUNC, parentComm);
			if (getPlatformIDStr.platforms != NULL && getPlatformIDStr.num_entries > 0)
			{
				MPI_Send((void *)platforms, sizeof(cl_platform_id) * getPlatformIDStr.num_entries, MPI_BYTE, 0,
						 GET_PLATFORM_ID_FUNC1, parentComm);
				free(platforms);
			}

		}

		if (status.MPI_TAG == GET_DEVICE_ID_FUNC)
		{
			//printf("In slave process, before GET_DEVICE_ID_FUNC received!\n");
			MPI_Recv(&tmpGetDeviceIDs, sizeof(tmpGetDeviceIDs), MPI_BYTE, 0,
					GET_DEVICE_ID_FUNC, parentComm, &status);
			cl_device_id *devices = NULL;
			cl_uint num_entries = tmpGetDeviceIDs.num_entries;
			if (num_entries > 0 && tmpGetDeviceIDs.devices != NULL)
			{
				devices = (cl_device_id *)malloc(num_entries * sizeof(cl_device_id));
			}
			mpiOpenCLGetDeviceIDs(&tmpGetDeviceIDs, devices);
			MPI_Send(&tmpGetDeviceIDs, sizeof(tmpGetDeviceIDs), MPI_BYTE, 0,
					GET_DEVICE_ID_FUNC, parentComm);
			if (num_entries > 0 && tmpGetDeviceIDs.devices != NULL)
			{
				MPI_Send(devices, sizeof(cl_device_id) * num_entries, MPI_BYTE, 0,
					GET_DEVICE_ID_FUNC1, parentComm);
				free(devices);
			}
		}

		if (status.MPI_TAG == CREATE_CONTEXT_FUNC)
		{
			//printf("In slave process, before CREATE_CONTEXT_FUNC received!\n");
			MPI_Recv(&tmpCreateContext, sizeof(tmpCreateContext), MPI_BYTE, 0,
					CREATE_CONTEXT_FUNC, parentComm, &status);
			cl_device_id *devices = NULL;
			if (tmpCreateContext.devices != NULL)
			{
				devices = (cl_device_id *)malloc(sizeof(cl_device_id) * tmpCreateContext.num_devices);
				MPI_Recv(devices, sizeof(cl_device_id) * tmpCreateContext.num_devices, MPI_BYTE, 0,
						 CREATE_CONTEXT_FUNC1, parentComm, &status);
			}

			mpiOpenCLCreateContext(&tmpCreateContext, devices);
			MPI_Send(&tmpCreateContext, sizeof(tmpCreateContext), MPI_BYTE, 0,
					CREATE_CONTEXT_FUNC, parentComm);
			if (devices != NULL)
			{
				free(devices);
			}
		}

		if (status.MPI_TAG == CREATE_COMMAND_QUEUE_FUNC)
		{
			//printf("In slave process, before CREATE_CMD_QUEUE received!\n");
			MPI_Recv(&tmpCreateCommandQueue, sizeof(tmpCreateCommandQueue), MPI_BYTE, 0,
					 CREATE_COMMAND_QUEUE_FUNC, parentComm, &status);
			mpiOpenCLCreateCommandQueue(&tmpCreateCommandQueue);
			//printf("In slave process, after CMD_QUEUE is created!\n");

			MPI_Send(&tmpCreateCommandQueue, sizeof(tmpCreateCommandQueue), MPI_BYTE, 0,
					 CREATE_COMMAND_QUEUE_FUNC, parentComm);

		}

		if (status.MPI_TAG == CREATE_PROGRMA_WITH_SOURCE)
		{
			MPI_Recv(&tmpCreateProgramWithSource, sizeof(tmpCreateProgramWithSource), MPI_BYTE, 0,
					 CREATE_PROGRMA_WITH_SOURCE, parentComm, &status);
			cl_uint count = tmpCreateProgramWithSource.count;
			size_t *lengthsArray = (size_t *)malloc(count * sizeof(size_t));
			
			size_t fileSize = tmpCreateProgramWithSource.lengths;
			char *fileBuffer = (char *)malloc(fileSize * sizeof(char));

			MPI_Recv(lengthsArray, count * sizeof(size_t), MPI_BYTE, 0,
					 CREATE_PROGRMA_WITH_SOURCE1, parentComm, &status);
			MPI_Recv(fileBuffer, fileSize, MPI_BYTE, 0,
					 CREATE_PROGRMA_WITH_SOURCE2, parentComm, &status);

			mpiOpenCLCreateProgramWithSource(&tmpCreateProgramWithSource, fileBuffer, lengthsArray);
			MPI_Send(&tmpCreateProgramWithSource, sizeof(tmpCreateProgramWithSource), MPI_BYTE, 0,
					 CREATE_PROGRMA_WITH_SOURCE, parentComm);
			free(fileBuffer);
			free(lengthsArray);
		}

		if (status.MPI_TAG == BUILD_PROGRAM)
		{
			MPI_Recv(&tmpBuildProgram, sizeof(tmpBuildProgram), MPI_BYTE, 0,
					 BUILD_PROGRAM, parentComm, &status);
			char *tmpBuf = NULL;
			if (tmpBuildProgram.optionLen > 0)
			{
				tmpBuf = (char *)malloc((tmpBuildProgram.optionLen + 1) * sizeof(char));
				MPI_Recv(tmpBuf, tmpBuildProgram.optionLen, MPI_BYTE, 0,
					     BUILD_PROGRAM1, parentComm, &status);
				tmpBuf[tmpBuildProgram.optionLen] = '\0';
			}

			cl_device_id *device_list = NULL;
			if (tmpBuildProgram.device_list != NULL)
			{
				device_list = (cl_device_id *)malloc(sizeof(cl_device_id) * tmpBuildProgram.num_devices);
				MPI_Recv(device_list, sizeof(cl_device_id) * tmpBuildProgram.num_devices, MPI_BYTE, 0,
						 BUILD_PROGRAM, parentComm, &status);
			}

			mpiOpenCLBuildProgram(&tmpBuildProgram, tmpBuf, device_list);
			MPI_Send(&tmpBuildProgram, sizeof(tmpBuildProgram), MPI_BYTE, 0,
					 BUILD_PROGRAM, parentComm);
			if (tmpBuildProgram.optionLen > 0)
			{
				free(tmpBuf);
			}
			if (tmpBuildProgram.device_list != NULL)
			{
				free(device_list);
			}
		}

		if (status.MPI_TAG == CREATE_KERNEL)
		{
			MPI_Recv(&tmpCreateKernel, sizeof(tmpCreateKernel), MPI_BYTE, 0,
					 CREATE_KERNEL, parentComm, &status);
			char *kernelName = (char *)malloc((tmpCreateKernel.kernelNameSize + 1)* sizeof(char));
			MPI_Recv(kernelName, tmpCreateKernel.kernelNameSize, MPI_CHAR, 0,
					 CREATE_KERNEL1, parentComm, &status);
			kernelName[tmpCreateKernel.kernelNameSize] = '\0';
			mpiOpenCLCreateKernel(&tmpCreateKernel, kernelName);
			MPI_Send(&tmpCreateKernel, sizeof(tmpCreateKernel), MPI_BYTE, 0,
					 CREATE_KERNEL, parentComm);
			free(kernelName);
		}

		if (status.MPI_TAG == CREATE_BUFFER_FUNC)
		{
			MPI_Recv(&tmpCreateBuffer, sizeof(tmpCreateBuffer), MPI_BYTE, 0,
					 CREATE_BUFFER_FUNC, parentComm, &status);
			void *host_ptr = NULL;
			if (tmpCreateBuffer.host_ptr_flag == 1)
			{
				host_ptr = malloc(tmpCreateBuffer.size);
				MPI_Recv(host_ptr, tmpCreateBuffer.size, MPI_BYTE, 0, 
						 CREATE_BUFFER_FUNC1, parentComm, &status);
			}
			mpiOpenCLCreateBuffer(&tmpCreateBuffer, host_ptr);
			MPI_Send(&tmpCreateBuffer, sizeof(tmpCreateBuffer), MPI_BYTE, 0,
					 CREATE_BUFFER_FUNC, parentComm);
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
			cl_event *event_wait_list = NULL;
			cl_uint num_events_in_wait_list = tmpEnqueueWriteBuffer.num_events_in_wait_list;
			if (num_events_in_wait_list > 0)
			{
				event_wait_list = (cl_event *)malloc(sizeof(cl_event) * num_events_in_wait_list);
				MPI_Recv(event_wait_list, sizeof(cl_event) * num_events_in_wait_list, MPI_BYTE, 0,
						 ENQUEUE_WRITE_BUFFER2, parentComm, &status);
			}

			mpiOpenCLEnqueueWriteBuffer(&tmpEnqueueWriteBuffer, host_ptr, event_wait_list);
			MPI_Send(&tmpEnqueueWriteBuffer, sizeof(tmpEnqueueWriteBuffer), MPI_BYTE, 0,
					 ENQUEUE_WRITE_BUFFER, parentComm);
			if (num_events_in_wait_list > 0)
			{
				free(event_wait_list);
			}

			//store the host pointer for later release
			if (tmpEnqueueWriteBuffer.blocking_write == CL_FALSE)
			{
				DATA_TRANSFER *dataTransferPtr = createDataTransfer(tmpEnqueueWriteBuffer.command_queue,
																	tmpEnqueueWriteBuffer.event);
				dataTransferPtr->host_ptr = host_ptr;
				dataTransferPtr->readOrWrite = CL_GPUV_WRITE;
			}
			else //if it is blocking write, write is complete.
			{
				free(host_ptr);

				//process all the events in the command queue
				processCommandQueue(tmpEnqueueWriteBuffer.command_queue);
			}
		}
		
		if (status.MPI_TAG == SET_KERNEL_ARG)
		{
			MPI_Recv(&tmpSetKernelArg, sizeof(tmpSetKernelArg), MPI_BYTE, 0,
					 SET_KERNEL_ARG, parentComm, &status);
			void *arg_value = NULL;
			if (tmpSetKernelArg.arg_value != NULL)
			{
				arg_value = (char *)malloc(tmpSetKernelArg.arg_size);
				MPI_Recv(arg_value, tmpSetKernelArg.arg_size, MPI_BYTE, 0,
						 SET_KERNEL_ARG1, parentComm, &status);
			}
			mpiOpenCLSetKernelArg(&tmpSetKernelArg, arg_value);
			MPI_Send(&tmpSetKernelArg, sizeof(tmpSetKernelArg), MPI_BYTE, 0,
					 SET_KERNEL_ARG, parentComm);
			if (tmpSetKernelArg.arg_value != NULL)
			{
				free(arg_value);
			}
		}

		if (status.MPI_TAG == ENQUEUE_ND_RANGE_KERNEL)
		{
			MPI_Recv(&tmpEnqueueNDRangeKernel, sizeof(tmpEnqueueNDRangeKernel), MPI_BYTE, 0,
					 ENQUEUE_ND_RANGE_KERNEL, parentComm, &status);
			int work_dim = tmpEnqueueNDRangeKernel.work_dim;
			size_t *global_work_offset, *global_work_size, *local_work_size;
			kernel_args *args_ptr;
			args_ptr = NULL;
			global_work_offset = NULL;
			global_work_size   = NULL;
			local_work_size    = NULL;

			if (tmpEnqueueNDRangeKernel.global_work_offset_flag == 1)
			{
				//printf("global work offset\n");
				global_work_offset = (size_t *)malloc(work_dim * sizeof(size_t));
				MPI_Recv(global_work_offset, work_dim * sizeof(size_t), MPI_BYTE, 0,
						 ENQUEUE_ND_RANGE_KERNEL1, parentComm, &status);
			}

			if (tmpEnqueueNDRangeKernel.global_work_size_flag == 1)
			{
				//printf("global work size\n");
				global_work_size   = (size_t *)malloc(work_dim * sizeof(size_t));
				MPI_Recv(global_work_size, work_dim * sizeof(size_t), MPI_BYTE, 0,
						 ENQUEUE_ND_RANGE_KERNEL2, parentComm, &status);
			}

			if (tmpEnqueueNDRangeKernel.local_work_size_flag == 1)
			{
				//printf("local work size\n");
				local_work_size    = (size_t *)malloc(work_dim * sizeof(size_t));
				MPI_Recv(local_work_size, work_dim * sizeof(size_t), MPI_BYTE, 0,
						 ENQUEUE_ND_RANGE_KERNEL3, parentComm, &status);
			}

			if (tmpEnqueueNDRangeKernel.args_num > 0)
			{
				args_ptr = (kernel_args *)malloc(tmpEnqueueNDRangeKernel.args_num * sizeof(kernel_args));
				MPI_Recv(args_ptr, tmpEnqueueNDRangeKernel.args_num * sizeof(kernel_args), MPI_BYTE, 0,
						 ENQUEUE_ND_RANGE_KERNEL4, parentComm, &status);
			}

			mpiOpenCLEnqueueNDRangeKernel(&tmpEnqueueNDRangeKernel,
										  global_work_offset,
										  global_work_size,
										  local_work_size,
										  args_ptr);

			MPI_Send(&tmpEnqueueNDRangeKernel, sizeof(tmpEnqueueNDRangeKernel), MPI_BYTE, 0,
					 ENQUEUE_ND_RANGE_KERNEL, parentComm);
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

			if (tmpEnqueueNDRangeKernel.args_num > 0)
			{
				free(args_ptr);
			}
		}

		if (status.MPI_TAG == ENQUEUE_READ_BUFFER)
		{
			MPI_Recv(&tmpEnqueueReadBuffer, sizeof(tmpEnqueueReadBuffer), MPI_BYTE, 0,
					 ENQUEUE_READ_BUFFER, parentComm, &status);
			int num_events_in_wait_list = tmpEnqueueReadBuffer.num_events_in_wait_list;
			cl_event *event_wait_list = NULL;
			if (num_events_in_wait_list > 0)
			{
				event_wait_list = (cl_event *)malloc(num_events_in_wait_list * sizeof(cl_event));
				MPI_Recv(event_wait_list, num_events_in_wait_list * sizeof(cl_event), MPI_BYTE, 0,
						 ENQUEUE_READ_BUFFER1, parentComm, &status);
			}

			int bufSize = tmpEnqueueReadBuffer.cb;
			char *host_ptr = (char *)malloc(bufSize);
			mpiOpenCLEnqueueReadBuffer(&tmpEnqueueReadBuffer, host_ptr, event_wait_list);
			MPI_Send(&tmpEnqueueReadBuffer, sizeof(tmpEnqueueReadBuffer), MPI_BYTE, 0,
					 ENQUEUE_READ_BUFFER, parentComm);
			if (num_events_in_wait_list > 0)
			{
				free(event_wait_list);
			}

			if (tmpEnqueueReadBuffer.blocking_read == CL_FALSE)
			{
				DATA_TRANSFER *dataTransfer = createDataTransfer(tmpEnqueueReadBuffer.command_queue,
												   tmpEnqueueReadBuffer.event);
				dataTransfer->host_ptr = host_ptr;
				dataTransfer->tag = tmpEnqueueReadBuffer.readBufferTag;
				dataTransfer->msgSize = bufSize;
				dataTransfer->readOrWrite = CL_GPUV_READ;
				dataTransfer->comm = parentComm;
			}
			else //blocking, reading is complete, send data to local node
			{
				MPI_Send(host_ptr, bufSize, MPI_BYTE, 0,
						 ENQUEUE_READ_BUFFER1, parentComm);
				free(host_ptr);
				//process all the events in the command queue
				processCommandQueue(tmpEnqueueReadBuffer.command_queue);
			}
		}

		if (status.MPI_TAG == RELEASE_MEM_OBJ)
		{
			MPI_Recv(&tmpReleaseMemObject, sizeof(tmpReleaseMemObject), MPI_BYTE, 0,
					 RELEASE_MEM_OBJ, parentComm, &status);
			mpiOpenCLReleaseMemObject(&tmpReleaseMemObject);
			MPI_Send(&tmpReleaseMemObject, sizeof(tmpReleaseMemObject), MPI_BYTE, 0,
					 RELEASE_MEM_OBJ, parentComm);
		}

		if (status.MPI_TAG == CL_RELEASE_KERNEL_FUNC)
		{
			MPI_Recv(&tmpReleaseKernel, sizeof(tmpReleaseKernel), MPI_BYTE, 0,
					 CL_RELEASE_KERNEL_FUNC, parentComm, &status);
			mpiOpenCLReleaseKernel(&tmpReleaseKernel);
			MPI_Send(&tmpReleaseKernel, sizeof(tmpReleaseKernel), MPI_BYTE, 0,
					 CL_RELEASE_KERNEL_FUNC, parentComm);
		}

		if (status.MPI_TAG == GET_CONTEXT_INFO_FUNC)
		{
			MPI_Recv(&tmpGetContextInfo, sizeof(tmpGetContextInfo), MPI_BYTE, 0,
					 GET_CONTEXT_INFO_FUNC, parentComm, &status);
			size_t param_value_size = tmpGetContextInfo.param_value_size;
			void *param_value = NULL;
			if (param_value_size > 0 && tmpGetContextInfo.param_value != NULL)
			{
				param_value = malloc(param_value_size);
			}
			mpiOpenCLGetContextInfo(&tmpGetContextInfo, param_value);
			MPI_Send(&tmpGetContextInfo, sizeof(tmpGetContextInfo), MPI_BYTE, 0,
					 GET_CONTEXT_INFO_FUNC, parentComm);

			if (param_value_size > 0 && tmpGetContextInfo.param_value != NULL)
			{
				MPI_Send(param_value, param_value_size, MPI_BYTE, 0,
						 GET_CONTEXT_INFO_FUNC1, parentComm);
				free(param_value);
			}
		}

		if (status.MPI_TAG == GET_BUILD_INFO_FUNC)
		{
			MPI_Recv(&tmpGetProgramBuildInfo, sizeof(tmpGetProgramBuildInfo), MPI_BYTE, 0,
					 GET_BUILD_INFO_FUNC, parentComm, &status);
			size_t param_value_size = tmpGetProgramBuildInfo.param_value_size;
			void *param_value = NULL;
			if (param_value_size > 0 && tmpGetProgramBuildInfo.param_value != NULL)
			{
				param_value = malloc(param_value_size);
			}
			mpiOpenCLGetProgramBuildInfo(&tmpGetProgramBuildInfo, param_value);
			MPI_Send(&tmpGetProgramBuildInfo, sizeof(tmpGetProgramBuildInfo), MPI_BYTE, 0,
					 GET_BUILD_INFO_FUNC, parentComm);

			if (param_value_size > 0 && tmpGetProgramBuildInfo.param_value != NULL)
			{
				MPI_Send(param_value, param_value_size, MPI_BYTE, 0,
						 GET_BUILD_INFO_FUNC1, parentComm);
				free(param_value);
			}
		}

		if (status.MPI_TAG == GET_PROGRAM_INFO_FUNC)
		{
			MPI_Recv(&tmpGetProgramInfo, sizeof(tmpGetProgramInfo), MPI_BYTE, 0,
					 GET_PROGRAM_INFO_FUNC, parentComm, &status);
			size_t param_value_size = tmpGetProgramInfo.param_value_size;
			void *param_value = NULL;
			if (param_value_size > 0 && tmpGetProgramInfo.param_value != NULL)
			{
				param_value = malloc(param_value_size);
			}
			mpiOpenCLGetProgramInfo(&tmpGetProgramInfo, param_value);
			MPI_Send(&tmpGetProgramInfo, sizeof(tmpGetProgramInfo), MPI_BYTE, 0,
					 GET_PROGRAM_INFO_FUNC, parentComm);

			if (param_value_size > 0 && tmpGetProgramInfo.param_value != NULL)
			{
				MPI_Send(param_value, param_value_size, MPI_BYTE, 0,
						 GET_PROGRAM_INFO_FUNC1, parentComm);
				free(param_value);
			}
		}

		if (status.MPI_TAG == REL_PROGRAM_FUNC)
		{
			MPI_Recv(&tmpReleaseProgram, sizeof(tmpReleaseProgram), MPI_BYTE, 0,
					 REL_PROGRAM_FUNC, parentComm, &status);
			mpiOpenCLReleaseProgram(&tmpReleaseProgram);
			MPI_Send(&tmpReleaseProgram, sizeof(tmpReleaseProgram), MPI_BYTE, 0,
					 REL_PROGRAM_FUNC, parentComm);
		}

		if (status.MPI_TAG == REL_COMMAND_QUEUE_FUNC)
		{
			MPI_Recv(&tmpReleaseCommandQueue, sizeof(tmpReleaseCommandQueue), MPI_BYTE, 0,
					 REL_COMMAND_QUEUE_FUNC, parentComm, &status);

			mpiOpenCLReleaseCommandQueue(&tmpReleaseCommandQueue);
			MPI_Send(&tmpReleaseCommandQueue, sizeof(tmpReleaseCommandQueue), MPI_BYTE, 0,
					 REL_COMMAND_QUEUE_FUNC, parentComm);
		}

		if (status.MPI_TAG == REL_CONTEXT_FUNC)
		{
			MPI_Recv(&tmpReleaseContext, sizeof(tmpReleaseContext), MPI_BYTE, 0,
					REL_CONTEXT_FUNC, parentComm, &status);
			mpiOpenCLReleaseContext(&tmpReleaseContext);
			MPI_Send(&tmpReleaseContext, sizeof(tmpReleaseContext), MPI_BYTE, 0,
					REL_CONTEXT_FUNC, parentComm);

		}

		if (status.MPI_TAG == GET_DEVICE_INFO_FUNC)
		{
			MPI_Recv(&tmpGetDeviceInfo, sizeof(tmpGetDeviceInfo), MPI_BYTE, 0,
					 GET_DEVICE_INFO_FUNC, parentComm, &status);
			size_t param_value_size = tmpGetDeviceInfo.param_value_size;
			void *param_value = NULL;
			if (param_value_size > 0 && tmpGetDeviceInfo.param_value != NULL)
			{
				param_value = malloc(param_value_size);
			}
			mpiOpenCLGetDeviceInfo(&tmpGetDeviceInfo, param_value);
			MPI_Send(&tmpGetDeviceInfo, sizeof(tmpGetDeviceInfo), MPI_BYTE, 0,
					 GET_DEVICE_INFO_FUNC, parentComm);

			if (param_value_size > 0 && tmpGetDeviceInfo.param_value != NULL)
			{
				MPI_Send(param_value, param_value_size, MPI_BYTE, 0,
						 GET_DEVICE_INFO_FUNC1, parentComm);
				free(param_value);
			}
		}

		if (status.MPI_TAG == GET_PLATFORM_INFO_FUNC)
		{
			MPI_Recv(&tmpGetPlatformInfo, sizeof(tmpGetPlatformInfo), MPI_BYTE, 0,
					 GET_PLATFORM_INFO_FUNC, parentComm, &status);
			size_t param_value_size = tmpGetPlatformInfo.param_value_size;
			void *param_value = NULL;
			if (param_value_size > 0 && tmpGetPlatformInfo.param_value != NULL)
			{
				param_value = malloc(param_value_size);
			}
			mpiOpenCLGetPlatformInfo(&tmpGetPlatformInfo, param_value);
			MPI_Send(&tmpGetPlatformInfo, sizeof(tmpGetPlatformInfo), MPI_BYTE, 0,
					 GET_PLATFORM_INFO_FUNC, parentComm);

			if (param_value_size > 0 && tmpGetPlatformInfo.param_value != NULL)
			{
				MPI_Send(param_value, param_value_size, MPI_BYTE, 0,
						 GET_PLATFORM_INFO_FUNC1, parentComm);
				free(param_value);
			}
		}

		if (status.MPI_TAG == FINISH_FUNC)
		{
			MPI_Recv(&tmpFinish, sizeof(tmpFinish), MPI_BYTE, 0,
					 FINISH_FUNC, parentComm, &status);
			mpiOpenCLFinish(&tmpFinish);

			//process all the events in the command queue
			processCommandQueue(tmpFinish.command_queue);
			MPI_Isend(&tmpFinish, sizeof(tmpFinish), MPI_BYTE, 0,
					 FINISH_FUNC, parentComm, &request);
		}

		if (status.MPI_TAG == FLUSH_FUNC)
		{
			MPI_Recv(&tmpFlush, sizeof(tmpFlush), MPI_BYTE, 0,
					 FLUSH_FUNC, parentComm, &status);
			mpiOpenCLFlush(&tmpFlush);
			MPI_Send(&tmpFlush, sizeof(tmpFlush), MPI_BYTE, 0,
					 FLUSH_FUNC, parentComm);
		}

		if (status.MPI_TAG == WAIT_FOR_EVENT_FUNC)
		{
			struct strWaitForEvents tmpWaitForEvents;
			MPI_Recv(&tmpWaitForEvents, sizeof(tmpWaitForEvents), MPI_BYTE, 0,
					 WAIT_FOR_EVENT_FUNC, parentComm, &status);
			cl_uint num_events = tmpWaitForEvents.num_events;
			cl_event *event_list = (cl_event *)malloc(sizeof(cl_event) * num_events);
			MPI_Recv(event_list, sizeof(cl_event) * num_events, MPI_BYTE, 0,
					 WAIT_FOR_EVENT_FUNC1, parentComm, &status);
			mpiOpenCLWaitForEvents(&tmpWaitForEvents, event_list);
			//get the pointer structures corresponding to the events
			processEvents(event_list, num_events);
//			for (i = 0; i < num_events; i++)
//			{
//				processEvent(event_list[i]);
//			}

			MPI_Send(&tmpWaitForEvents, sizeof(tmpWaitForEvents), MPI_BYTE, 0,
					 WAIT_FOR_EVENT_FUNC, parentComm);
			free(event_list);
		}

		if (status.MPI_TAG == CREATE_SAMPLER_FUNC)
		{
			MPI_Recv(&tmpCreateSampler, sizeof(tmpCreateSampler), MPI_BYTE, 0,
					 CREATE_SAMPLER_FUNC, parentComm, &status);
			mpiOpenCLCreateSampler(&tmpCreateSampler);
			MPI_Send(&tmpCreateSampler, sizeof(tmpCreateSampler), MPI_BYTE, 0,
					 CREATE_SAMPLER_FUNC, parentComm);
		}

		if (status.MPI_TAG == GET_CMD_QUEUE_INFO_FUNC)
		{
			MPI_Recv(&tmpGetCommandQueueInfo, sizeof(tmpGetCommandQueueInfo), MPI_BYTE, 0,
					 GET_CMD_QUEUE_INFO_FUNC, parentComm, &status);
			size_t param_value_size = tmpGetCommandQueueInfo.param_value_size;
			void *param_value = NULL;
			if (param_value_size > 0 && tmpGetCommandQueueInfo.param_value != NULL)
			{
				param_value = malloc(param_value_size);
			}

			mpiOpenCLGetCommandQueueInfo(&tmpGetCommandQueueInfo, param_value);
			MPI_Send(&tmpGetCommandQueueInfo, sizeof(tmpGetCommandQueueInfo), MPI_BYTE, 0,
					 GET_CMD_QUEUE_INFO_FUNC, parentComm);

			if (param_value_size > 0 && tmpGetCommandQueueInfo.param_value != NULL)
			{
				MPI_Send(param_value, param_value_size, MPI_BYTE, 0,
						 GET_CMD_QUEUE_INFO_FUNC1, parentComm);
				free(param_value);
			}
		}

		if (status.MPI_TAG == ENQUEUE_MAP_BUFF_FUNC)
		{
			MPI_Recv(&tmpEnqueueMapBuffer, sizeof(tmpEnqueueMapBuffer), MPI_BYTE, 0,
					 ENQUEUE_MAP_BUFF_FUNC, parentComm, &status);
			cl_uint num_events_in_wait_list = tmpEnqueueMapBuffer.num_events_in_wait_list;
			cl_event *event_wait_list = NULL;
			if (num_events_in_wait_list > 0)
			{
				event_wait_list = (cl_event *)malloc(sizeof(cl_event) * num_events_in_wait_list);
				MPI_Recv(event_wait_list, sizeof(cl_event) * num_events_in_wait_list, MPI_BYTE, 0,
						 ENQUEUE_MAP_BUFF_FUNC1, parentComm, &status);
			}
			mpiOpenCLEnqueueMapBuffer(&tmpEnqueueMapBuffer, event_wait_list);
			MPI_Send(&tmpEnqueueMapBuffer, sizeof(tmpEnqueueMapBuffer), MPI_BYTE, 0,
					 ENQUEUE_MAP_BUFF_FUNC, parentComm);
			if (num_events_in_wait_list > 0)
			{
				free(event_wait_list);
			}
		}

		if (status.MPI_TAG == RELEASE_EVENT_FUNC)
		{
			MPI_Recv(&tmpReleaseEvent, sizeof(tmpReleaseEvent), MPI_BYTE, 0,
					 RELEASE_EVENT_FUNC, parentComm, &status);
			mpiOpenCLReleaseEvent(&tmpReleaseEvent);
			MPI_Send(&tmpReleaseEvent, sizeof(tmpReleaseEvent), MPI_BYTE, 0,
					 RELEASE_EVENT_FUNC, parentComm);
		}

		if (status.MPI_TAG == GET_EVENT_PROF_INFO_FUNC)
		{
			MPI_Recv(&tmpGetEventProfilingInfo, sizeof(tmpGetEventProfilingInfo), MPI_BYTE, 0,
					 GET_EVENT_PROF_INFO_FUNC, parentComm, &status);
			size_t param_value_size = tmpGetEventProfilingInfo.param_value_size;
			void *param_value = NULL;
			if (param_value_size > 0 && tmpGetEventProfilingInfo.param_value != NULL)
			{
				param_value = malloc(param_value_size);
			}
			mpiOpenCLGetEventProfilingInfo(&tmpGetEventProfilingInfo, param_value);
			MPI_Send(&tmpGetEventProfilingInfo, sizeof(tmpGetEventProfilingInfo), MPI_BYTE, 0,
					 GET_EVENT_PROF_INFO_FUNC, parentComm);

			if (param_value_size > 0 && tmpGetEventProfilingInfo.param_value != NULL)
			{
				MPI_Send(param_value, param_value_size, MPI_BYTE, 0,
						 GET_EVENT_PROF_INFO_FUNC1, parentComm);
				free(param_value);
			}
		}

		if (status.MPI_TAG == RELEASE_SAMPLER_FUNC)
		{
			MPI_Recv(&tmpReleaseSampler, sizeof(tmpReleaseSampler), MPI_BYTE, 0,
					 RELEASE_SAMPLER_FUNC, parentComm, &status);
			mpiOpenCLReleaseSampler(&tmpReleaseSampler);
			MPI_Send(&tmpReleaseSampler, sizeof(tmpReleaseSampler), MPI_BYTE, 0,
					 RELEASE_SAMPLER_FUNC, parentComm);
		}

		if (status.MPI_TAG == GET_KERNEL_WGP_INFO_FUNC)
		{
			MPI_Recv(&tmpGetKernelWorkGroupInfo, sizeof(tmpGetKernelWorkGroupInfo), MPI_BYTE, 0,
					 GET_KERNEL_WGP_INFO_FUNC, parentComm, &status);
			size_t param_value_size = tmpGetKernelWorkGroupInfo.param_value_size;
			void *param_value = NULL;
			if (param_value_size > 0 && tmpGetKernelWorkGroupInfo.param_value != NULL)
			{
				param_value = malloc(param_value_size);
			}
			mpiOpenCLGetKernelWorkGroupInfo(&tmpGetKernelWorkGroupInfo, param_value);
			MPI_Send(&tmpGetKernelWorkGroupInfo, sizeof(tmpGetKernelWorkGroupInfo), MPI_BYTE, 0,
					 GET_KERNEL_WGP_INFO_FUNC, parentComm);

			if (param_value_size > 0 && tmpGetKernelWorkGroupInfo.param_value != NULL)
			{
				MPI_Send(param_value, param_value_size, MPI_BYTE, 0,
						 GET_KERNEL_WGP_INFO_FUNC1, parentComm);
				free(param_value);
			}
		}

		if (status.MPI_TAG == CREATE_IMAGE_2D_FUNC)
		{
			MPI_Recv(&tmpCreateImage2D, sizeof(tmpCreateImage2D), MPI_BYTE, 0,
					 CREATE_IMAGE_2D_FUNC, parentComm, &status);
			size_t host_buff_size = tmpCreateImage2D.host_buff_size;
			void *host_ptr = NULL;
			if (host_buff_size > 0)
			{
				host_ptr = malloc(host_buff_size);
				MPI_Recv(host_ptr, host_buff_size, MPI_BYTE, 0,
						 CREATE_IMAGE_2D_FUNC1, parentComm, &status);
			}
			mpiOpenCLCreateImage2D(&tmpCreateImage2D, host_ptr);
			MPI_Send(&tmpCreateImage2D, sizeof(tmpCreateImage2D), MPI_BYTE, 0,
					 CREATE_IMAGE_2D_FUNC, parentComm);
			if (host_buff_size > 0)
			{
				free(host_ptr);
			}
		}

		if (status.MPI_TAG == ENQ_COPY_BUFF_FUNC)
		{
			MPI_Recv(&tmpEnqueueCopyBuffer, sizeof(tmpEnqueueCopyBuffer), MPI_BYTE, 0,
					 ENQ_COPY_BUFF_FUNC, parentComm, &status);
			cl_uint num_events_in_wait_list = tmpEnqueueCopyBuffer.num_events_in_wait_list;
			cl_event *event_wait_list = NULL;
			if (num_events_in_wait_list > 0)
			{
				event_wait_list = (cl_event *)malloc(sizeof(cl_event) * num_events_in_wait_list);
				MPI_Recv(event_wait_list, sizeof(cl_event) * num_events_in_wait_list, MPI_BYTE, 0,
						 ENQ_COPY_BUFF_FUNC1, parentComm, &status);
			}
			mpiOpenCLEnqueueCopyBuffer(&tmpEnqueueCopyBuffer, event_wait_list);
			MPI_Send(&tmpEnqueueCopyBuffer, sizeof(tmpEnqueueCopyBuffer), MPI_BYTE, 0,
					 ENQ_COPY_BUFF_FUNC, parentComm);
			if (num_events_in_wait_list > 0)
			{
				free(event_wait_list);
			}
		}

		if (status.MPI_TAG == RETAIN_EVENT_FUNC)
		{
			MPI_Recv(&tmpRetainEvent, sizeof(tmpRetainEvent), MPI_BYTE, 0,
					 RETAIN_EVENT_FUNC, parentComm, &status);
			mpiOpenCLRetainEvent(&tmpRetainEvent);
			MPI_Send(&tmpRetainEvent, sizeof(tmpRetainEvent), MPI_BYTE, 0,
					 RETAIN_EVENT_FUNC, parentComm);
		}

		if (status.MPI_TAG == RETAIN_MEMOBJ_FUNC)
		{
			MPI_Recv(&tmpRetainMemObject, sizeof(tmpRetainMemObject), MPI_BYTE, 0,
					 RETAIN_MEMOBJ_FUNC, parentComm, &status);
			mpiOpenCLRetainMemObject(&tmpRetainMemObject);
			MPI_Send(&tmpRetainMemObject, sizeof(tmpRetainMemObject), MPI_BYTE, 0,
					 RETAIN_MEMOBJ_FUNC, parentComm);
		}

		if (status.MPI_TAG == RETAIN_KERNEL_FUNC)
		{
			MPI_Recv(&tmpRetainKernel, sizeof(tmpRetainKernel), MPI_BYTE, 0,
					 RETAIN_KERNEL_FUNC, parentComm, &status);
			mpiOpenCLRetainKernel(&tmpRetainKernel);
			MPI_Send(&tmpRetainKernel, sizeof(tmpRetainKernel), MPI_BYTE, 0,
					 RETAIN_KERNEL_FUNC, parentComm);
		}

		if (status.MPI_TAG == RETAIN_CMDQUE_FUNC)
		{
			MPI_Recv(&tmpRetainCommandQueue, sizeof(tmpRetainCommandQueue), MPI_BYTE, 0,
					 RETAIN_CMDQUE_FUNC, parentComm, &status);
			mpiOpenCLRetainCommandQueue(&tmpRetainCommandQueue);
			MPI_Send(&tmpRetainCommandQueue, sizeof(tmpRetainCommandQueue), MPI_BYTE, 0,
					 RETAIN_CMDQUE_FUNC, parentComm);
		}

		if (status.MPI_TAG == ENQ_UNMAP_MEMOBJ_FUNC)
		{
			MPI_Recv(&tmpEnqueueUnmapMemObject, sizeof(tmpEnqueueUnmapMemObject), MPI_BYTE, 0,
					 ENQ_UNMAP_MEMOBJ_FUNC, parentComm, &status);
			cl_uint num_events_in_wait_list = tmpEnqueueUnmapMemObject.num_events_in_wait_list;
			cl_event *event_wait_list = NULL;
			if (num_events_in_wait_list > 0)
			{
				event_wait_list = (cl_event *)malloc(sizeof(cl_event) * num_events_in_wait_list);
				MPI_Recv(event_wait_list, sizeof(cl_event) * num_events_in_wait_list, MPI_BYTE, 0,
						 ENQ_UNMAP_MEMOBJ_FUNC1, parentComm, &status);
			}
			mpiOpenCLEnqueueUnmapMemObject(&tmpEnqueueUnmapMemObject, event_wait_list);
			MPI_Send(&tmpEnqueueUnmapMemObject, sizeof(tmpEnqueueUnmapMemObject), MPI_BYTE, 0,
					 ENQ_UNMAP_MEMOBJ_FUNC, parentComm);
			if (num_events_in_wait_list > 0)
			{
				free(event_wait_list);
			}
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

