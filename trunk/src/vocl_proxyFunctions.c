#include <sys/time.h>
#include "vocl_proxy.h"
#include "vocl_proxyKernelArgProc.h"

extern void voclProxySetMemWritten(cl_mem mem, int isWritten);
extern void voclProxySetMemWriteCmdQueue(cl_mem mem, cl_command_queue cmdQueue);

/* record number of objects accocated in the current proxy process */
static int voclProxyObjCount = 0;

void voclProxyObjCountInitialize()
{
	voclProxyObjCount = 0;
	return;
}

void voclProxyObjCountFinalize()
{
	voclProxyObjCount = 0;
	return;
}

void voclProxyObjCountIncrease()
{
	voclProxyObjCount++;
	return;
}


void voclProxyObjCountDecrease()
{
	voclProxyObjCount--;
	return;
}

/* functions to call native opencl functions */
void mpiOpenCLGetPlatformIDs(struct strGetPlatformIDs *tmpGetPlatform,
                             cl_platform_id * platforms)
{
    cl_uint *num_platforms = NULL;
    cl_int err_code;
    if (tmpGetPlatform->num_platforms == 1) {
        num_platforms = &tmpGetPlatform->num_platforms;
    }
    err_code = clGetPlatformIDs(tmpGetPlatform->num_entries, platforms, num_platforms);
    tmpGetPlatform->res = err_code;

    return;
}

void mpiOpenCLGetDeviceIDs(struct strGetDeviceIDs *tmpGetDeviceIDs, cl_device_id * devices)
{
    cl_int err_code;
    cl_platform_id platform = tmpGetDeviceIDs->platform;
    cl_device_type device_type = tmpGetDeviceIDs->device_type;
    cl_uint num_entries = tmpGetDeviceIDs->num_entries;

    cl_uint *num_device_ptr = NULL;
    if (tmpGetDeviceIDs->num_devices == 1) {
        num_device_ptr = &tmpGetDeviceIDs->num_devices;
    }

    err_code = clGetDeviceIDs(platform, device_type, num_entries, devices, num_device_ptr);
    tmpGetDeviceIDs->res = err_code;
	
    return;
}

void mpiOpenCLCreateContext(struct strCreateContext *tmpCreateContext, cl_device_id * devices)
{
    cl_int err_code;
    cl_uint num_devices = tmpCreateContext->num_devices;
    /* const cl_context_properties properties = tmpCreateContext->properties; */
    /* const cl_device_id devices = tmpCreateContext->devices; */
    cl_context hContext = clCreateContext(0, num_devices, devices, 0, 0, &err_code);
    tmpCreateContext->hContext = hContext;
    tmpCreateContext->errcode_ret = err_code;
	voclProxyObjCountIncrease();
}

void mpiOpenCLCreateCommandQueue(struct strCreateCommandQueue *tmpCreateCommandQueue)
{
    cl_int err_code;
    cl_command_queue_properties properties = tmpCreateCommandQueue->properties;
    cl_device_id device = tmpCreateCommandQueue->device;
    cl_context hInContext = tmpCreateCommandQueue->context;

    cl_command_queue hCmdQueue =
        clCreateCommandQueue(hInContext, device, properties, &err_code);

    tmpCreateCommandQueue->errcode_ret = err_code;
    tmpCreateCommandQueue->clCommand = hCmdQueue;

	voclProxyObjCountIncrease();
}

void mpiOpenCLCreateProgramWithSource(struct strCreateProgramWithSource
                                      *tmpCreateProgramWithSource, char *cSourceCL,
                                      size_t * lengthsArray)
{
    cl_int err_code;
    size_t sourceFileSize = tmpCreateProgramWithSource->lengths;
    int count = tmpCreateProgramWithSource->count;
    cl_context hInContext = tmpCreateProgramWithSource->context;
    cl_uint strIndex;
    size_t strStartLoc;
    /* transform the whole string buffer to multiple strings */
    char **strings = (char **) malloc(count * sizeof(char *));
    strStartLoc = 0;
    for (strIndex = 0; strIndex < count; strIndex++) {
        strings[strIndex] = (char *) malloc(lengthsArray[strIndex] + 1);
        memcpy(strings[strIndex], &cSourceCL[strStartLoc],
               lengthsArray[strIndex] * sizeof(char));
        strStartLoc += lengthsArray[strIndex];
        strings[strIndex][lengthsArray[strIndex]] = '\0';
    }

    cl_program hProgram = clCreateProgramWithSource(hInContext, count, (const char **) strings,
                                                    lengthsArray, &err_code);
    tmpCreateProgramWithSource->clProgram = hProgram;
    tmpCreateProgramWithSource->errcode_ret = err_code;
    for (strIndex = 0; strIndex < count; strIndex++) {
        free(strings[strIndex]);
    }

	voclProxyObjCountIncrease();
    free(strings);
}

void mpiOpenCLBuildProgram(struct strBuildProgram *tmpBuildProgram,
                           char *options, cl_device_id * device_list)
{
    cl_int err_code;
    cl_program hInProgram = tmpBuildProgram->program;
    cl_uint num_devices = tmpBuildProgram->num_devices;
    err_code = clBuildProgram(hInProgram, num_devices, device_list, options, 0, 0);

    tmpBuildProgram->res = err_code;
}

void mpiOpenCLCreateKernel(struct strCreateKernel *tmpCreateKernel, char *kernel_name)
{
    cl_int err_code;
    cl_program hInProgram = tmpCreateKernel->program;
    cl_kernel hKernel = clCreateKernel(hInProgram, kernel_name, &err_code);
    tmpCreateKernel->kernel = hKernel;
    tmpCreateKernel->errcode_ret = err_code;

	voclProxyObjCountIncrease();

	return;
}

void mpiOpenCLCreateBuffer(struct strCreateBuffer *tmpCreateBuffer, void *host_ptr)
{
    cl_int err_code;
    cl_context hInContext = tmpCreateBuffer->context;
    cl_mem_flags flags = tmpCreateBuffer->flags;
    size_t bufferSize = tmpCreateBuffer->size;
    cl_mem deviceMem;
    deviceMem = clCreateBuffer(hInContext, flags, bufferSize, host_ptr, &err_code);
    if (err_code != CL_SUCCESS) {
        printf("In slave, create buffer error!\n");
    }

    tmpCreateBuffer->errcode_ret = err_code;
    tmpCreateBuffer->deviceMem = deviceMem;

	voclProxyObjCountIncrease();
}

void mpiOpenCLEnqueueWriteBuffer(struct strEnqueueWriteBuffer *tmpEnqueueWriteBuffer,
                                 void *ptr, cl_event * event_wait_list)
{
    cl_int err_code;
    cl_command_queue hInCmdQueue = tmpEnqueueWriteBuffer->command_queue;
    cl_mem deviceMem = tmpEnqueueWriteBuffer->buffer;
    cl_bool blocking_write = tmpEnqueueWriteBuffer->blocking_write;
    size_t offset = tmpEnqueueWriteBuffer->offset;
    size_t cb = tmpEnqueueWriteBuffer->cb;
    cl_uint num_events_in_wait_list = tmpEnqueueWriteBuffer->num_events_in_wait_list;
    cl_event *event_ret = &tmpEnqueueWriteBuffer->event;

    err_code = clEnqueueWriteBuffer(hInCmdQueue, deviceMem, blocking_write, offset,
                                    cb, ptr, num_events_in_wait_list, event_wait_list,
                                    event_ret);
	//voclProxyUpdateMemoryOnCmdQueue(hInCmdQueue, deviceMem, cb);
    tmpEnqueueWriteBuffer->res = err_code;
}

void mpiOpenCLSetKernelArg(struct strSetKernelArg *tmpSetKernelArg, void *arg_value)
{
    cl_int err_code;
    cl_kernel hInKernel = tmpSetKernelArg->kernel;
    cl_uint arg_index = tmpSetKernelArg->arg_index;
    size_t arg_size = tmpSetKernelArg->arg_size;

    err_code = clSetKernelArg(hInKernel, arg_index, arg_size, arg_value);

    tmpSetKernelArg->res = err_code;
}

static float qTime, wTime, eTime;

void eventProfiling(cl_event event)
{   
    cl_ulong queuedTime, submitTime, startTime, endTime;
    cl_int err;
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED,
                                  sizeof(cl_ulong), &queuedTime, NULL);
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT,
                                  sizeof(cl_ulong), &submitTime, NULL);
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                  sizeof(cl_ulong), &startTime, NULL);
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                  sizeof(cl_ulong), &endTime, NULL);
    qTime += (submitTime - queuedTime) / 1000000.0;
    wTime += (startTime - submitTime) / 1000000.0;
    eTime += (endTime - startTime) / 1000000.0;

    printf("qTime = %.3f, wTime = %.3f, eTime = %.3f\n", qTime, wTime, eTime);

    return;
}


void mpiOpenCLEnqueueNDRangeKernel(struct strEnqueueNDRangeKernel *tmpEnqueueNDRangeKernel,
								   struct strEnqueueNDRangeKernelReply *kernelLaunchReply,
                                   cl_event * event_wait_list,
                                   size_t * global_work_offset,
                                   size_t * global_work_size,
                                   size_t * local_work_size, kernel_args * args_ptr)
{
    cl_int err_code;
    cl_command_queue hInCommand = tmpEnqueueNDRangeKernel->command_queue;
    cl_kernel hInKernel = tmpEnqueueNDRangeKernel->kernel;
    cl_uint work_dim = tmpEnqueueNDRangeKernel->work_dim;
    cl_uint num_events_in_wait_list = tmpEnqueueNDRangeKernel->num_events_in_wait_list;
    cl_uint args_num = tmpEnqueueNDRangeKernel->args_num;
    cl_uint args_index;
	cl_mem mem;

    /* call real opencl functions to set kernel arguments */
    for (args_index = 0; args_index < args_num; args_index++) {
        if (args_ptr[args_index].arg_null_flag == 1) {
            err_code = clSetKernelArg(hInKernel,
                                      args_ptr[args_index].arg_index,
                                      args_ptr[args_index].arg_size, NULL);
        }
        else {
            err_code = clSetKernelArg(hInKernel,
                                      args_ptr[args_index].arg_index,
                                      args_ptr[args_index].arg_size,
                                      (const void *) args_ptr[args_index].arg_value);
        }

		/* set memory write flag */
		if (args_ptr[args_index].isGlobalMemory == 1)
		{
			mem =  *((cl_mem *)args_ptr[args_index].arg_value);
			voclProxySetMemWritten(mem, 1);
			voclProxySetMemWriteCmdQueue(mem, hInCommand);
		}
    }

    cl_event *event_ptr = NULL;
    if (tmpEnqueueNDRangeKernel->event_null_flag == 0) {
        event_ptr = &kernelLaunchReply->event;
    }

    err_code = clEnqueueNDRangeKernel(hInCommand,
                                      hInKernel,
                                      work_dim,
                                      global_work_offset,
                                      global_work_size,
                                      local_work_size,
                                      num_events_in_wait_list, event_wait_list, event_ptr);
    kernelLaunchReply->res = err_code;
}

void mpiOpenCLEnqueueReadBuffer(struct strEnqueueReadBuffer *tmpEnqueueReadBuffer,
                                void *ptr, cl_event * event_wait_list)
{
    cl_int err_code;
    cl_command_queue hInCmdQueue = tmpEnqueueReadBuffer->command_queue;
    cl_mem deviceMem = tmpEnqueueReadBuffer->buffer;
    cl_bool read_flag = tmpEnqueueReadBuffer->blocking_read;
    size_t offset = tmpEnqueueReadBuffer->offset;
    size_t cb = tmpEnqueueReadBuffer->cb;
    cl_uint num_events_in_wait_list = tmpEnqueueReadBuffer->num_events_in_wait_list;
    cl_event *event_ret = &tmpEnqueueReadBuffer->event;

    err_code = clEnqueueReadBuffer(hInCmdQueue,
                                   deviceMem,
                                   read_flag,
                                   offset,
                                   cb,
                                   ptr, num_events_in_wait_list, event_wait_list, event_ret);
    tmpEnqueueReadBuffer->res = err_code;
}

void mpiOpenCLReleaseMemObject(struct strReleaseMemObject *tmpReleaseMemObject)
{
    cl_int err_code;
    cl_mem deviceMem = tmpReleaseMemObject->memobj;

    err_code = clReleaseMemObject(deviceMem);
    tmpReleaseMemObject->res = err_code;
	voclProxyObjCountDecrease();
}

void mpiOpenCLReleaseKernel(struct strReleaseKernel *tmpReleaseKernel)
{
    cl_int err_code;
    cl_kernel hInKernel = tmpReleaseKernel->kernel;

    err_code = clReleaseKernel(hInKernel);
    tmpReleaseKernel->res = err_code;
	voclProxyObjCountDecrease();
}

void mpiOpenCLGetContextInfo(struct strGetContextInfo *tmpGetContextInfo, void *param_value)
{
    cl_int errcode;
    cl_context context = tmpGetContextInfo->context;
    cl_context_info param_name = tmpGetContextInfo->param_name;
    size_t param_value_size = tmpGetContextInfo->param_value_size;
    size_t *value_size_ptr = NULL;
    if (tmpGetContextInfo->param_value_size_ret == 1) {
        value_size_ptr = &tmpGetContextInfo->param_value_size_ret;
    }
    errcode = clGetContextInfo(context,
                               param_name, param_value_size, param_value, value_size_ptr);

    tmpGetContextInfo->res = errcode;
}

void mpiOpenCLGetProgramBuildInfo(struct strGetProgramBuildInfo *tmpGetProgramBuildInfo,
                                  void *param_value)
{
    cl_int errcode;
    cl_program program = tmpGetProgramBuildInfo->program;
    cl_device_id device = tmpGetProgramBuildInfo->device;
    cl_program_build_info param_name = tmpGetProgramBuildInfo->param_name;
    size_t param_value_size = tmpGetProgramBuildInfo->param_value_size;
    size_t *value_size_ptr = NULL;
    if (tmpGetProgramBuildInfo->param_value_size_ret == 1) {
        value_size_ptr = &tmpGetProgramBuildInfo->param_value_size_ret;
    }

    errcode = clGetProgramBuildInfo(program,
                                    device,
                                    param_name, param_value_size, param_value, value_size_ptr);
    tmpGetProgramBuildInfo->res = errcode;
}

void mpiOpenCLGetProgramInfo(struct strGetProgramInfo *tmpGetProgramInfo, void *param_value)
{
    cl_int errcode;
    cl_program program = tmpGetProgramInfo->program;
    cl_program_info param_name = tmpGetProgramInfo->param_name;
    size_t param_value_size = tmpGetProgramInfo->param_value_size;
    size_t *value_size_ptr = NULL;
    if (tmpGetProgramInfo->param_value_size_ret == 1) {
        value_size_ptr = &tmpGetProgramInfo->param_value_size_ret;
    }
    errcode = clGetProgramInfo(program,
                               param_name, param_value_size, param_value, value_size_ptr);
    tmpGetProgramInfo->res = errcode;
}

void mpiOpenCLReleaseProgram(struct strReleaseProgram *tmpReleaseProgram)
{
    cl_int errcode;
    cl_program program = tmpReleaseProgram->program;

    errcode = clReleaseProgram(program);
    tmpReleaseProgram->res = errcode;
	voclProxyObjCountDecrease();
}

void mpiOpenCLReleaseCommandQueue(struct strReleaseCommandQueue *tmpReleaseCommandQueue)
{
    cl_int errcode;
    cl_command_queue command_queue = tmpReleaseCommandQueue->command_queue;

    errcode = clReleaseCommandQueue(command_queue);
    tmpReleaseCommandQueue->res = errcode;
	voclProxyObjCountDecrease();
}

void mpiOpenCLReleaseContext(struct strReleaseContext *tmpReleaseContext)
{
    cl_int errcode;
    cl_context context = tmpReleaseContext->context;
    errcode = clReleaseContext(context);
    tmpReleaseContext->res = errcode;
	voclProxyObjCountDecrease();
}

void mpiOpenCLFinish(struct strFinish *tmpFinish)
{
    cl_int err_code;
    cl_command_queue hInCmdQueue = tmpFinish->command_queue;
    err_code = clFinish(hInCmdQueue);
    tmpFinish->res = err_code;
}

void mpiOpenCLGetDeviceInfo(struct strGetDeviceInfo *tmpGetDeviceInfo, void *param_value)
{
    cl_int errcode;
    cl_device_id device = tmpGetDeviceInfo->device;
    cl_device_info param_name = tmpGetDeviceInfo->param_name;
    size_t param_value_size = tmpGetDeviceInfo->param_value_size;
    size_t *value_size_ptr = NULL;
    if (tmpGetDeviceInfo->param_value_size_ret == 1) {
        value_size_ptr = &tmpGetDeviceInfo->param_value_size_ret;
    }
    errcode = clGetDeviceInfo(device,
                              param_name, param_value_size, param_value, value_size_ptr);
    tmpGetDeviceInfo->res = errcode;
}

void mpiOpenCLGetPlatformInfo(struct strGetPlatformInfo *tmpGetPlatformInfo, void *param_value)
{
    cl_int errcode;
    cl_platform_id platform = tmpGetPlatformInfo->platform;
    cl_platform_info param_name = tmpGetPlatformInfo->param_name;
    size_t param_value_size = tmpGetPlatformInfo->param_value_size;
    size_t *value_size_ptr = NULL;
    if (tmpGetPlatformInfo->param_value_size_ret == 1) {
        value_size_ptr = &tmpGetPlatformInfo->param_value_size_ret;
    }
    errcode = clGetPlatformInfo(platform,
                                param_name, param_value_size, param_value, value_size_ptr);
    tmpGetPlatformInfo->res = errcode;
}

void mpiOpenCLFlush(struct strFlush *tmpFlush)
{
    cl_int err_code;
    cl_command_queue hInCmdQueue = tmpFlush->command_queue;
    err_code = clFlush(hInCmdQueue);
    tmpFlush->res = err_code;
}

void mpiOpenCLWaitForEvents(struct strWaitForEvents *tmpWaitForEvents, cl_event * event_list)
{
    cl_int err_code;
    cl_uint num_events = tmpWaitForEvents->num_events;
    err_code = clWaitForEvents(num_events, event_list);
    tmpWaitForEvents->res = err_code;
}

void mpiOpenCLCreateSampler(struct strCreateSampler *tmpCreateSampler)
{
    cl_context context = tmpCreateSampler->context;
    cl_bool normalized_coords = tmpCreateSampler->normalized_coords;
    cl_addressing_mode addressing_mode = tmpCreateSampler->addressing_mode;
    cl_filter_mode filter_mode = tmpCreateSampler->filter_mode;
    cl_int *errcode = NULL;
    if (tmpCreateSampler->errcode_ret == 1) {
        errcode = &tmpCreateSampler->errcode_ret;
    }

    tmpCreateSampler->sampler =
        clCreateSampler(context, normalized_coords, addressing_mode, filter_mode, errcode);
	voclProxyObjCountIncrease();
}

void mpiOpenCLGetCommandQueueInfo(struct strGetCommandQueueInfo *tmpGetCommandQueueInfo,
                                  void *param_value)
{
    cl_int errcode;
    cl_command_queue command_queue = tmpGetCommandQueueInfo->command_queue;
    cl_command_queue_info param_name = tmpGetCommandQueueInfo->param_name;
    size_t param_value_size = tmpGetCommandQueueInfo->param_value_size;
    size_t *value_size_ptr = NULL;
    if (tmpGetCommandQueueInfo->param_value_size_ret == 1) {
        value_size_ptr = &tmpGetCommandQueueInfo->param_value_size_ret;
    }

    errcode = clGetCommandQueueInfo(command_queue,
                                    param_name, param_value_size, param_value, value_size_ptr);
    tmpGetCommandQueueInfo->res = errcode;
}

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
    if (tmpEnqueueMapBuffer->event_null_flag == 0) {
        event = &tmpEnqueueMapBuffer->event;
    }
    cl_int *errcode_ptr = NULL;
    if (tmpEnqueueMapBuffer->errcode_ret == 0) {
        errcode_ptr = &tmpEnqueueMapBuffer->errcode_ret;
    }

    tmpEnqueueMapBuffer->ret_ptr =
        clEnqueueMapBuffer(command_queue,
                           buffer,
                           blocking_map,
                           map_flags,
                           offset,
                           cb, num_events_in_wait_list, event_wait_list, event, errcode_ptr);

}

void mpiOpenCLReleaseEvent(struct strReleaseEvent *tmpReleaseEvent)
{
    cl_int err_code;
    cl_event event = tmpReleaseEvent->event;
    err_code = clReleaseEvent(event);
    tmpReleaseEvent->res = err_code;
}

void mpiOpenCLGetEventProfilingInfo(struct strGetEventProfilingInfo *tmpGetEventProfilingInfo,
                                    void *param_value)
{
    cl_int errcode;
    cl_event event = tmpGetEventProfilingInfo->event;
    cl_profiling_info param_name = tmpGetEventProfilingInfo->param_name;
    size_t param_value_size = tmpGetEventProfilingInfo->param_value_size;
    size_t *value_size_ptr = NULL;
    if (tmpGetEventProfilingInfo->param_value_size_ret == 1) {
        value_size_ptr = &tmpGetEventProfilingInfo->param_value_size_ret;
    }
    errcode = clGetEventProfilingInfo(event,
                                      param_name,
                                      param_value_size, param_value, value_size_ptr);
    tmpGetEventProfilingInfo->res = errcode;
}

void mpiOpenCLReleaseSampler(struct strReleaseSampler *tmpReleaseSampler)
{
    cl_int err_code;
    cl_sampler sampler = tmpReleaseSampler->sampler;
    err_code = clReleaseSampler(sampler);
    tmpReleaseSampler->res = err_code;
	voclProxyObjCountDecrease();
}

void mpiOpenCLGetKernelWorkGroupInfo(struct strGetKernelWorkGroupInfo
                                     *tmpGetKernelWorkGroupInfo, void *param_value)
{
    cl_int errcode;
    cl_kernel kernel = tmpGetKernelWorkGroupInfo->kernel;
    cl_device_id device = tmpGetKernelWorkGroupInfo->device;
    cl_kernel_work_group_info param_name = tmpGetKernelWorkGroupInfo->param_name;
    size_t param_value_size = tmpGetKernelWorkGroupInfo->param_value_size;
    size_t *value_size_ptr = NULL;
    if (tmpGetKernelWorkGroupInfo->param_value_size_ret == 1) {
        value_size_ptr = &tmpGetKernelWorkGroupInfo->param_value_size_ret;
    }
    errcode = clGetKernelWorkGroupInfo(kernel,
                                       device,
                                       param_name,
                                       param_value_size, param_value, value_size_ptr);
    tmpGetKernelWorkGroupInfo->res = errcode;
}

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
    if (tmpCreateImage2D->errcode_ret == 0) {
        errcode_ptr = &tmpCreateImage2D->errcode_ret;
    }

    tmpCreateImage2D->mem_obj =
        clCreateImage2D(context,
                        flags,
                        &img_format,
                        image_width, image_height, image_row_pitch, host_ptr, errcode_ptr);
	voclProxyObjCountIncrease();
}

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
    if (tmpEnqueueCopyBuffer->event_null_flag == 0) {
        event_ptr = &tmpEnqueueCopyBuffer->event;
    }

    errcode = clEnqueueCopyBuffer(command_queue,
                                  src_buffer,
                                  dst_buffer,
                                  src_offset,
                                  dst_offset,
                                  cb, num_events_in_wait_list, event_wait_list, event_ptr);
    tmpEnqueueCopyBuffer->res = errcode;
}

void mpiOpenCLRetainEvent(struct strRetainEvent *tmpRetainEvent)
{
    cl_int errcode;
    cl_event event = tmpRetainEvent->event;
    errcode = clRetainEvent(event);
    tmpRetainEvent->res = errcode;
}

void mpiOpenCLRetainMemObject(struct strRetainMemObject *tmpRetainMemObject)
{
    cl_int errcode;
    cl_mem memobj = tmpRetainMemObject->memobj;
    errcode = clRetainMemObject(memobj);
    tmpRetainMemObject->res = errcode;
}

void mpiOpenCLRetainKernel(struct strRetainKernel *tmpRetainKernel)
{
    cl_int errcode;
    cl_kernel kernel = tmpRetainKernel->kernel;
    errcode = clRetainKernel(kernel);
    tmpRetainKernel->res = errcode;
}

void mpiOpenCLRetainCommandQueue(struct strRetainCommandQueue *tmpRetainCommandQueue)
{
    cl_int errcode;
    cl_command_queue command_queue = tmpRetainCommandQueue->command_queue;
    errcode = clRetainCommandQueue(command_queue);
    tmpRetainCommandQueue->res = errcode;
}

void mpiOpenCLEnqueueUnmapMemObject(struct strEnqueueUnmapMemObject *tmpEnqueueUnmapMemObject,
                                    cl_event * event_wait_list)
{
    cl_int errcode;
    cl_command_queue command_queue = tmpEnqueueUnmapMemObject->command_queue;
    cl_mem memobj = tmpEnqueueUnmapMemObject->memobj;
    void *mapped_ptr = tmpEnqueueUnmapMemObject->mapped_ptr;
    cl_uint num_events_in_wait_list = tmpEnqueueUnmapMemObject->num_events_in_wait_list;
    cl_event *event_ptr = NULL;
    if (tmpEnqueueUnmapMemObject->event_null_flag == 0) {
        event_ptr = &tmpEnqueueUnmapMemObject->event;
    }
    errcode = clEnqueueUnmapMemObject(command_queue,
                                      memobj,
                                      mapped_ptr,
                                      num_events_in_wait_list, event_wait_list, event_ptr);
    tmpEnqueueUnmapMemObject->res = errcode;
}


