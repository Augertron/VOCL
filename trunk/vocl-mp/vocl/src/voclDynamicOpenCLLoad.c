#include <stdio.h>
#include <dlfcn.h>
#include <CL/opencl.h>
#include "voclDynamicOpenCLLoad.h"

/* functions for module processing */
static void *voclOpenclModulePtr = NULL;

void voclOpenclModuleInitialize()
{
    voclOpenclModulePtr = dlopen("libOpenCL.so", RTLD_LAZY);
    if (voclOpenclModulePtr == NULL) {
        printf("Could not open libOpenCL:%s\n", dlerror());
        exit(1);
    }

    return;
}

static void *voclGetOpenclModulePtr()
{
    return voclOpenclModulePtr;
}

void voclOpenclModuleRelease()
{
    if (voclOpenclModulePtr != NULL) {
        dlclose(voclOpenclModulePtr);
    }

    return;
}

/*call of native opencl funtions via the dynamic load */
cl_int
dlCLGetPlatformIDs(cl_uint num_entries, cl_platform_id * platforms, cl_uint * num_platforms)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clGetPlatformIDsLocal funcPtr;
    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clGetPlatformIDs");
    if ((error = dlerror())) {
        printf("Could find clGetPlatformIDs: %s\n", error);
        exit(1);
    }
    errCode = (*funcPtr) (num_entries, platforms, num_platforms);

    return errCode;
}

cl_int
dlCLGetDeviceIDs(cl_platform_id platform,
                 cl_device_type device_type,
                 cl_uint num_entries, cl_device_id * devices, cl_uint * num_devices)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clGetDeviceIDsLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clGetDeviceIDs");
    if ((error = dlerror())) {
        printf("Could find clGetDeviceIDs: %s\n", error);
        exit(1);
    }
    errCode = (*funcPtr) (platform, device_type, num_entries, devices, num_devices);

    return errCode;
}

/* The created context handler is outputed as a pointer */
/* If we return it as a return value and the address value */
/* is larger than 4G, some parts will be truncated */
void dlCLCreateContext(const cl_context_properties * properties,
                       cl_uint num_devices,
                       const cl_device_id * devices,
                       void (CL_CALLBACK * pfn_notify) (const char *, const void *, size_t,
                                                        void *), void *user_data,
                       cl_int * errcode_ret, cl_context * contextPtr)
{
    const char *error;
    void *modulePtr;
    cl_context context;
    clCreateContextLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clCreateContext");
    if ((error = dlerror())) {
        printf("Could find clCreateContext: %s\n", error);
        exit(1);
    }

    *contextPtr =
        (*funcPtr) (properties, num_devices, devices, pfn_notify, user_data, errcode_ret);

    return;
}

/* Command Queue APIs */
void dlCLCreateCommandQueue(cl_context context,
                            cl_device_id device,
                            cl_command_queue_properties properties,
                            cl_int * errcode_ret, cl_command_queue * cmdQueuePtr)
{
    const char *error;
    void *modulePtr;
    clCreateCommandQueueLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clCreateCommandQueue");
    if ((error = dlerror())) {
        printf("Could find clCreateContext: %s\n", error);
        exit(1);
    }
    *cmdQueuePtr = (*funcPtr) (context, device, properties, errcode_ret);

    return;
}

void dlCLCreateProgramWithSource(cl_context context,
                                 cl_uint count,
                                 const char **strings, const size_t * lengths,
                                 cl_int * errcode_ret, cl_program * programPtr)
{
    const char *error;
    void *modulePtr;
    clCreateProgramWithSourceLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clCreateProgramWithSource");
    if ((error = dlerror())) {
        printf("Could find clCreateContext: %s\n", error);
        exit(1);
    }
    *programPtr = (*funcPtr) (context, count, strings, lengths, errcode_ret);

    return;
}

cl_int
dlCLBuildProgram(cl_program program,
                 cl_uint num_devices,
                 const cl_device_id * device_list,
                 const char *options,
                 void (CL_CALLBACK * pfn_notify) (cl_program program, void *user_data),
                 void *user_data)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clBuildProgramLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clBuildProgram");
    if ((error = dlerror())) {
        printf("Could find clBuildProgram: %s\n", error);
        exit(1);
    }
    errCode = (*funcPtr) (program, num_devices, device_list, options, pfn_notify, user_data);

    return errCode;
}

void dlCLCreateKernel(cl_program program, const char *kernel_name, cl_int * errcode_ret,
                      cl_kernel * kernelPtr)
{
    const char *error;
    void *modulePtr;
    clCreateKernelLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clCreateKernel");
    if ((error = dlerror())) {
        printf("Could find clCreateKernel: %s\n", error);
        exit(1);
    }
    *kernelPtr = (*funcPtr) (program, kernel_name, errcode_ret);

    return;
}

/* Memory Object APIs */
void dlCLCreateBuffer(cl_context context,
                      cl_mem_flags flags, size_t size, void *host_ptr, cl_int * errcode_ret,
                      cl_mem * memPtr)
{
    const char *error;
    void *modulePtr;
    clCreateBufferLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    *(void **) (&funcPtr) = dlsym(modulePtr, "clCreateBuffer");
    if ((error = dlerror())) {
        printf("Could find clCreateBuffer: %s\n", error);
        exit(1);
    }

    *memPtr = (*funcPtr) (context, flags, size, host_ptr, errcode_ret);

    return;
}

cl_int
dlCLEnqueueWriteBuffer(cl_command_queue command_queue,
                       cl_mem buffer,
                       cl_bool blocking_write,
                       size_t offset,
                       size_t cb,
                       const void *ptr,
                       cl_uint num_events_in_wait_list,
                       const cl_event * event_wait_list, cl_event * event)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clEnqueueWriteBufferLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clEnqueueWriteBuffer");
    if ((error = dlerror())) {
        printf("Could find clEnqueueWriteBuffer: %s\n", error);
        exit(1);
    }
    errCode = (*funcPtr) (command_queue, buffer, blocking_write, offset,
                          cb, ptr, num_events_in_wait_list, event_wait_list, event);

    return errCode;
}

cl_int
dlCLSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clSetKernelArgLocal funcPtr;
    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clSetKernelArg");
    if ((error = dlerror())) {
        printf("Could find clSetKernelArg: %s\n", error);
        exit(1);
    }
    errCode = (*funcPtr) (kernel, arg_index, arg_size, arg_value);

    return errCode;
}

cl_int
dlCLEnqueueNDRangeKernel(cl_command_queue command_queue,
                         cl_kernel kernel,
                         cl_uint work_dim,
                         const size_t * global_work_offset,
                         const size_t * global_work_size,
                         const size_t * local_work_size,
                         cl_uint num_events_in_wait_list,
                         const cl_event * event_wait_list, cl_event * event)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clEnqueueNDRangeKernelLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clEnqueueNDRangeKernel");
    if ((error = dlerror())) {
        printf("Could find clEnqueueNDRangeKernel: %s\n", error);
        exit(1);
    }

    errCode = (*funcPtr) (command_queue, kernel, work_dim, global_work_offset,
                          global_work_size, local_work_size, num_events_in_wait_list,
                          event_wait_list, event);

    return errCode;
}

/* Enqueued Commands for GPU memory read */
cl_int
dlCLEnqueueReadBuffer(cl_command_queue command_queue,
                      cl_mem buffer,
                      cl_bool blocking_read,
                      size_t offset,
                      size_t cb,
                      void *ptr,
                      cl_uint num_events_in_wait_list,
                      const cl_event * event_wait_list, cl_event * event)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clEnqueueReadBufferLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clEnqueueReadBuffer");
    if ((error = dlerror())) {
        printf("Could find clEnqueueReadBuffer: %s\n", error);
        exit(1);
    }
    errCode = (*funcPtr) (command_queue, buffer, blocking_read, offset,
                          cb, ptr, num_events_in_wait_list, event_wait_list, event);

    return errCode;
}

cl_int dlCLReleaseMemObject(cl_mem memobj)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clReleaseMemObjectLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clReleaseMemObject");
    if ((error = dlerror())) {
        printf("Could find clReleaseMemObject: %s\n", error);
        exit(1);
    }
    errCode = (*funcPtr) (memobj);

    return errCode;
}

cl_int dlCLReleaseKernel(cl_kernel kernel)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clReleaseKernelLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clReleaseKernel");
    if ((error = dlerror())) {
        printf("Could find clReleaseKernel: %s\n", error);
        exit(1);
    }
    errCode = (*funcPtr) (kernel);

    return errCode;
}

cl_int dlCLFinish(cl_command_queue command_queue)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clFinishLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clFinish");
    if ((error = dlerror())) {
        printf("Could find clFinish: %s\n", error);
        exit(1);
    }
    errCode = (*funcPtr) (command_queue);

    return errCode;
}

cl_int
dlCLGetContextInfo(cl_context context,
                   cl_context_info param_name,
                   size_t param_value_size, void *param_value, size_t * param_value_size_ret)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clGetContextInfoLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clGetContextInfo");
    if ((error = dlerror())) {
        printf("Could find clGetContextInfo: %s\n", error);
        exit(1);
    }
    errCode = (*funcPtr) (context, param_name, param_value_size,
                          param_value, param_value_size_ret);

    return errCode;
}

cl_int
dlCLGetProgramBuildInfo(cl_program program,
                        cl_device_id device,
                        cl_program_build_info param_name,
                        size_t param_value_size,
                        void *param_value, size_t * param_value_size_ret)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clGetProgramBuildInfoLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clGetProgramBuildInfo");
    if ((error = dlerror())) {
        printf("Could find clGetProgramBuildInfo: %s\n", error);
        exit(1);
    }
    errCode = (*funcPtr) (program, device, param_name, param_value_size,
                          param_value, param_value_size_ret);

    return errCode;
}

cl_int
dlCLGetProgramInfo(cl_program program,
                   cl_program_info param_name,
                   size_t param_value_size, void *param_value, size_t * param_value_size_ret)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clGetProgramInfoLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clGetProgramInfo");
    if ((error = dlerror())) {
        printf("Could find clGetProgramInfo: %s\n", error);
        exit(1);
    }
    errCode = (*funcPtr) (program, param_name, param_value_size,
                          param_value, param_value_size_ret);

    return errCode;
}

cl_int dlCLReleaseProgram(cl_program program)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clReleaseProgramLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clReleaseProgram");
    if ((error = dlerror())) {
        printf("Could find clReleaseProgram: %s\n", error);
        exit(1);
    }
    errCode = (*funcPtr) (program);

    return errCode;
}

cl_int dlCLReleaseCommandQueue(cl_command_queue command_queue)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clReleaseCommandQueueLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clReleaseCommandQueue");
    if ((error = dlerror())) {
        printf("Could find clReleaseCommandQueue: %s\n", error);
        exit(1);
    }
    errCode = (*funcPtr) (command_queue);

    return errCode;
}

cl_int dlCLReleaseContext(cl_context context)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clReleaseContextLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clReleaseContext");
    if ((error = dlerror())) {
        printf("Could find clReleaseContext: %s\n", error);
        exit(1);
    }
    errCode = (*funcPtr) (context);

    return errCode;
}

cl_int
dlCLGetDeviceInfo(cl_device_id device,
                  cl_device_info param_name,
                  size_t param_value_size, void *param_value, size_t * param_value_size_ret)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clGetDeviceInfoLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clGetDeviceInfo");
    if ((error = dlerror())) {
        printf("Could find clGetDeviceInfo: %s\n", error);
        exit(1);
    }
    errCode = (*funcPtr) (device, param_name, param_value_size,
                          param_value, param_value_size_ret);

    return errCode;
}

cl_int
dlCLGetPlatformInfo(cl_platform_id platform,
                    cl_platform_info param_name,
                    size_t param_value_size, void *param_value, size_t * param_value_size_ret)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clGetPlatformInfoLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clGetPlatformInfo");
    if ((error = dlerror())) {
        printf("Could find clGetPlatformInfo: %s\n", error);
        exit(1);
    }
    errCode = (*funcPtr) (platform, param_name, param_value_size,
                          param_value, param_value_size_ret);

    return errCode;
}

cl_int dlCLFlush(cl_command_queue command_queue)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clFlushLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clFlush");
    if ((error = dlerror())) {
        printf("Could find clFlush: %s\n", error);
        exit(1);
    }
    errCode = (*funcPtr) (command_queue);

    return errCode;
}

cl_int dlCLWaitForEvents(cl_uint num_events, const cl_event * event_list)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clWaitForEventsLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clWaitForEvents");
    if ((error = dlerror())) {
        printf("Could find clWaitForEvents: %s\n", error);
        exit(1);
    }
    errCode = (*funcPtr) (num_events, event_list);

    return errCode;
}

void dlCLCreateSampler(cl_context context,
                       cl_bool normalized_coords,
                       cl_addressing_mode addressing_mode,
                       cl_filter_mode filter_mode, cl_int * errcode_ret,
                       cl_sampler * samplerPtr)
{
    const char *error;
    void *modulePtr;
    clCreateSamplerLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clCreateSampler");
    if ((error = dlerror())) {
        printf("Could find clCreateSampler: %s\n", error);
        exit(1);
    }
    *samplerPtr = (*funcPtr) (context, normalized_coords, addressing_mode,
                              filter_mode, errcode_ret);

    return;
}

cl_int
dlCLGetCommandQueueInfo(cl_command_queue command_queue,
                        cl_command_queue_info param_name,
                        size_t param_value_size,
                        void *param_value, size_t * param_value_size_ret)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clGetCommandQueueInfoLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clGetCommandQueueInfo");
    if ((error = dlerror())) {
        printf("Could find clGetCommandQueueInfo: %s\n", error);
        exit(1);
    }

    errCode = (*funcPtr) (command_queue, param_name, param_value_size,
                          param_value, param_value_size_ret);

    return errCode;
}

void *dlCLEnqueueMapBuffer(cl_command_queue command_queue,
                           cl_mem buffer,
                           cl_bool blocking_map,
                           cl_map_flags map_flags,
                           size_t offset,
                           size_t cb,
                           cl_uint num_events_in_wait_list,
                           const cl_event * event_wait_list,
                           cl_event * event, cl_int * errcode_ret)
{
    const char *error;
    void *modulePtr;
    void *retPtr;
    clEnqueueMapBufferLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clEnqueueMapBuffer");
    if ((error = dlerror())) {
        printf("Could find clEnqueueMapBuffer: %s\n", error);
        exit(1);
    }

    retPtr = (*funcPtr) (command_queue, buffer, blocking_map, map_flags,
                         offset, cb, num_events_in_wait_list, event_wait_list,
                         event, errcode_ret);

    return retPtr;
}

cl_int dlCLReleaseEvent(cl_event event)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clReleaseEventLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clReleaseEvent");
    if ((error = dlerror())) {
        printf("Could find clReleaseEvent: %s\n", error);
        exit(1);
    }

    errCode = (*funcPtr) (event);

    return errCode;
}

cl_int
dlCLGetEventProfilingInfo(cl_event event,
                          cl_profiling_info param_name,
                          size_t param_value_size,
                          void *param_value, size_t * param_value_size_ret)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clGetEventProfilingInfoLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clGetEventProfilingInfo");
    if ((error = dlerror())) {
        printf("Could find clGetEventProfilingInfo: %s\n", error);
        exit(1);
    }

    errCode = (*funcPtr) (event, param_name, param_value_size,
                          param_value, param_value_size_ret);

    return errCode;
}

cl_int dlCLReleaseSampler(cl_sampler sampler)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clReleaseSamplerLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clReleaseSampler");
    if ((error = dlerror())) {
        printf("Could find clReleaseSampler: %s\n", error);
        exit(1);
    }

    errCode = (*funcPtr) (sampler);

    return errCode;
}

cl_int
dlCLGetKernelWorkGroupInfo(cl_kernel kernel,
                           cl_device_id device,
                           cl_kernel_work_group_info param_name,
                           size_t param_value_size,
                           void *param_value, size_t * param_value_size_ret)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clGetKernelWorkGroupInfoLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clGetKernelWorkGroupInfo");
    if ((error = dlerror())) {
        printf("Could find clGetKernelWorkGroupInfo: %s\n", error);
        exit(1);
    }

    errCode = (*funcPtr) (kernel, device, param_name, param_value_size,
                          param_value, param_value_size_ret);

    return errCode;
}

void dlCLCreateImage2D(cl_context context,
                       cl_mem_flags flags,
                       const cl_image_format * image_format,
                       size_t image_width,
                       size_t image_height,
                       size_t image_row_pitch, void *host_ptr, cl_int * errcode_ret,
                       cl_mem * memPtr)
{
    const char *error;
    void *modulePtr;
    clCreateImage2DLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clCreateImage2D");
    if ((error = dlerror())) {
        printf("Could find clCreateImage2D: %s\n", error);
        exit(1);
    }

    *memPtr = (*funcPtr) (context, flags, image_format, image_width, image_height,
                          image_row_pitch, host_ptr, errcode_ret);

    return;
}

cl_int
dlCLEnqueueCopyBuffer(cl_command_queue command_queue,
                      cl_mem src_buffer,
                      cl_mem dst_buffer,
                      size_t src_offset,
                      size_t dst_offset,
                      size_t cb,
                      cl_uint num_events_in_wait_list,
                      const cl_event * event_wait_list, cl_event * event)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clEnqueueCopyBufferLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clEnqueueCopyBuffer");
    if ((error = dlerror())) {
        printf("Could find clEnqueueCopyBuffer: %s\n", error);
        exit(1);
    }

    errCode = (*funcPtr) (command_queue, src_buffer, dst_buffer, src_offset, dst_offset,
                          cb, num_events_in_wait_list, event_wait_list, event);

    return errCode;
}

cl_int dlCLRetainEvent(cl_event event)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clRetainEventLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clRetainEvent");
    if ((error = dlerror())) {
        printf("Could find clRetainEvent: %s\n", error);
        exit(1);
    }

    errCode = (*funcPtr) (event);

    return errCode;

}

cl_int dlCLRetainMemObject(cl_mem memobj)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clRetainMemObjectLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clRetainMemObject");
    if ((error = dlerror())) {
        printf("Could find clRetainMemObject: %s\n", error);
        exit(1);
    }

    errCode = (*funcPtr) (memobj);

    return errCode;
}

cl_int dlCLRetainKernel(cl_kernel kernel)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clRetainKernelLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clRetainKernel");
    if ((error = dlerror())) {
        printf("Could find clRetainKernel: %s\n", error);
        exit(1);
    }

    errCode = (*funcPtr) (kernel);

    return errCode;
}

cl_int dlCLRetainCommandQueue(cl_command_queue command_queue)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clRetainCommandQueueLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clRetainCommandQueue");
    if ((error = dlerror())) {
        printf("Could find clRetainCommandQueue: %s\n", error);
        exit(1);
    }

    errCode = (*funcPtr) (command_queue);

    return errCode;
}

cl_int
dlCLEnqueueUnmapMemObject(cl_command_queue command_queue,
                          cl_mem memobj,
                          void *mapped_ptr,
                          cl_uint num_events_in_wait_list,
                          const cl_event * event_wait_list, cl_event * event)
{
    const char *error;
    void *modulePtr;
    int errCode;
    clEnqueueUnmapMemObjectLocal funcPtr;

    modulePtr = voclGetOpenclModulePtr();
    funcPtr = dlsym(modulePtr, "clEnqueueUnmapObject");
    if ((error = dlerror())) {
        printf("Could find clEnqueueUnmapObject: %s\n", error);
        exit(1);
    }

    errCode = (*funcPtr) (command_queue, memobj, mapped_ptr, num_events_in_wait_list,
                          event_wait_list, event);

    return errCode;
}
