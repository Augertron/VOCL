#ifndef __VOCL_LIB_H__
#define __VOCL_LIB_H__
#include <CL/opencl.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

	struct strGetProxyCommInfo {
		int proxyIndexInApp;
		int proxyRank;
		int appIndex;
		MPI_Comm comm;
		MPI_Comm commData;
	};

    struct strGetPlatformIDs {
        cl_uint num_entries;
        cl_platform_id *platforms;
        cl_uint num_platforms;
        cl_int res;
    };

    struct strGetDeviceIDs {
        cl_platform_id platform;
        cl_device_type device_type;
        cl_uint num_entries;
        cl_device_id *devices;
        cl_uint num_devices;
        cl_int res;
    };

    struct strCreateContext {
        cl_context_properties properties;
        cl_uint num_devices;
        cl_device_id *devices;
        /* CL_CALLBACK *              pfn_notify; */
        void *user_data;
        cl_int errcode_ret;
        cl_context hContext;
    };

    struct strCreateCommandQueue {
        cl_context context;
        cl_device_id device;
        cl_command_queue_properties properties;
        cl_command_queue clCommand;
        cl_int errcode_ret;
    };

    struct strCreateProgramWithSource {
        cl_context context;
        cl_uint count;
        size_t lengths;
        cl_program clProgram;
        cl_int errcode_ret;
    };

    struct strBuildProgram {
        cl_program program;
        cl_uint num_devices;
        cl_device_id *device_list;
        cl_uint optionLen;
        /* CL_CALLBACK *     pfn_notify; */
        void *user_data;
        cl_int res;
    };

    struct strCreateKernel {
        cl_program program;
        size_t kernelNameSize;
        cl_int errcode_ret;
        cl_kernel kernel;
    };

/* Memory Object APIs */
    struct strCreateBuffer {
        cl_context context;
        cl_mem_flags flags;
        size_t size;
        cl_int host_ptr_flag;
        cl_int errcode_ret;
        cl_mem deviceMem;
    };

    struct strEnqueueWriteBuffer {
        cl_command_queue command_queue;
        cl_mem buffer;
        cl_bool blocking_write;
        cl_int tag;
        size_t offset;
        size_t cb;
        cl_uint num_events_in_wait_list;
        cl_int event_null_flag; /* 1, flag is NULL, 0, is NOT NULL */
        cl_event event;
        cl_int res;
    };

    struct strSetKernelArg {
        cl_kernel kernel;
        cl_uint arg_index;
        size_t arg_size;
        const void *arg_value;
        cl_int res;
    };

	struct strMigrationCheck {
		cl_command_queue command_queue;
		int              rankNo;
		int              checkLocation;
		int              argsNum;
		size_t           memSize;
		int              isMigrationNeeded;
	};

    struct strEnqueueNDRangeKernel {
        cl_command_queue command_queue;
        cl_kernel kernel;
        cl_uint work_dim;
        cl_int global_work_offset_flag;
        cl_int global_work_size_flag;
        cl_int local_work_size_flag;
        cl_uint args_num;
		size_t dataSize;
        cl_uint num_events_in_wait_list;
        cl_int event_null_flag;
    };

	struct strEnqueueNDRangeKernelReply {
		cl_event event;
		cl_int res;
	};

    struct strEnqueueReadBuffer {
        cl_command_queue command_queue;
        cl_mem buffer;
        cl_bool blocking_read;
        cl_uint readBufferTag;
        size_t offset;
        size_t cb;
        cl_uint num_events_in_wait_list;
        cl_int event_null_flag; /* 1: the event point is NULL. 0: the event point is NOT NULL */
        cl_event event;
        cl_int res;
    };

    struct strReleaseMemObject {
        cl_mem memobj;
        cl_int res;
    };

    struct strReleaseKernel {
        cl_kernel kernel;
        cl_int res;
    };

    struct strFinish {
        cl_command_queue command_queue;
        cl_int res;
    };

    struct strGetContextInfo {
        cl_context context;
        cl_context_info param_name;
        size_t param_value_size;
        void *param_value;
        size_t param_value_size_ret;
        cl_int res;
    };

    struct strGetProgramBuildInfo {
        cl_program program;
        cl_device_id device;
        cl_program_build_info param_name;
        size_t param_value_size;
        void *param_value;
        size_t param_value_size_ret;
        cl_int res;
    };

    struct strGetProgramInfo {
        cl_program program;
        cl_program_info param_name;
        size_t param_value_size;
        void *param_value;
        size_t param_value_size_ret;
        cl_int res;
    };

    struct strReleaseProgram {
        cl_program program;
        cl_int res;
    };

    struct strReleaseCommandQueue {
        cl_command_queue command_queue;
        cl_int res;
    };

    struct strReleaseContext {
        cl_context context;
        cl_int res;
    };

    struct strGetDeviceInfo {
        cl_device_id device;
        cl_device_info param_name;
        size_t param_value_size;
        void *param_value;
        size_t param_value_size_ret;
        cl_int res;
    };

    struct strGetPlatformInfo {
        cl_platform_id platform;
        cl_platform_info param_name;
        size_t param_value_size;
        void *param_value;
        size_t param_value_size_ret;
        cl_int res;
    };

    struct strFlush {
        cl_command_queue command_queue;
        cl_int res;
    };

    struct strWaitForEvents {
        cl_uint num_events;
        cl_int res;
    };

    struct strCreateSampler {
        cl_context context;
        cl_bool normalized_coords;
        cl_addressing_mode addressing_mode;
        cl_filter_mode filter_mode;
        cl_int errcode_ret;
        cl_sampler sampler;
    };

    struct strGetCommandQueueInfo {
        cl_command_queue command_queue;
        cl_command_queue_info param_name;
        size_t param_value_size;
        void *param_value;
        size_t param_value_size_ret;
        cl_int res;
    };

    struct strEnqueueMapBuffer {
        cl_command_queue command_queue;
        cl_mem buffer;
        cl_bool blocking_map;
        cl_map_flags map_flags;
        size_t offset;
        size_t cb;
        cl_uint num_events_in_wait_list;
        cl_int event_null_flag; /* 1: NULL, 0: NOT NULL */
        cl_event event;
        cl_int errcode_ret;
        void *ret_ptr;
    };

    struct strReleaseEvent {
        cl_event event;
        cl_int res;
    };

    struct strGetEventProfilingInfo {
        cl_event event;
        cl_profiling_info param_name;
        size_t param_value_size;
        void *param_value;
        size_t param_value_size_ret;
        cl_int res;
    };

    struct strReleaseSampler {
        cl_sampler sampler;
        cl_int res;
    };

    struct strGetKernelWorkGroupInfo {
        cl_kernel kernel;
        cl_device_id device;
        cl_kernel_work_group_info param_name;
        size_t param_value_size;
        void *param_value;
        size_t param_value_size_ret;
        cl_int res;
    };

    struct strCreateImage2D {
        cl_context context;
        cl_mem_flags flags;
        cl_image_format img_format;
        size_t image_width;
        size_t image_height;
        size_t image_row_pitch;
        size_t host_buff_size;
        cl_int errcode_ret;
        cl_mem mem_obj;
    };

    struct strEnqueueCopyBuffer {
        cl_command_queue command_queue;
        cl_mem src_buffer;
        cl_mem dst_buffer;
        size_t src_offset;
        size_t dst_offset;
        size_t cb;
        cl_uint num_events_in_wait_list;
        cl_int event_null_flag;
        cl_event event;
        cl_int res;
    };

    struct strRetainEvent {
        cl_event event;
        cl_int res;
    };

    struct strRetainMemObject {
        cl_mem memobj;
        cl_int res;
    };

    struct strRetainKernel {
        cl_kernel kernel;
        cl_int res;
    };

    struct strRetainCommandQueue {
        cl_command_queue command_queue;
        cl_int res;
    };

    struct strEnqueueUnmapMemObject {
        cl_command_queue command_queue;
        cl_mem memobj;
        void *mapped_ptr;
        cl_uint num_events_in_wait_list;
        cl_int event_null_flag;
        cl_event event;
        cl_int res;
    };

/* define pointer to each opencl function to be used */
/* to process the opencl function call in the local node */
    typedef cl_int
        (*clGetPlatformIDsLocal) (cl_uint num_entries, cl_platform_id * platforms,
                                  cl_uint * num_platforms);

    typedef cl_int
        (*clGetDeviceIDsLocal) (cl_platform_id platform,
                                cl_device_type device_type,
                                cl_uint num_entries, cl_device_id * devices,
                                cl_uint * num_devices);

    typedef cl_context
        (*clCreateContextLocal) (const cl_context_properties * properties,
                                 cl_uint num_devices,
                                 const cl_device_id * devices,
                                 void (CL_CALLBACK * pfn_notify) (const char *, const void *,
                                                                  size_t, void *),
                                 void *user_data, cl_int * errcode_ret);

    typedef cl_command_queue
        (*clCreateCommandQueueLocal) (cl_context context,
                                      cl_device_id device,
                                      cl_command_queue_properties properties,
                                      cl_int * errcode_ret);

    typedef cl_program
        (*clCreateProgramWithSourceLocal) (cl_context context,
                                           cl_uint count,
                                           const char **strings, const size_t * lengths,
                                           cl_int * errcode_ret);

    typedef cl_int
        (*clBuildProgramLocal) (cl_program program,
                                cl_uint num_devices,
                                const cl_device_id * device_list,
                                const char *options,
                                void (CL_CALLBACK * pfn_notify) (cl_program program,
                                                                 void *user_data),
                                void *user_data);

    typedef cl_kernel
        (*clCreateKernelLocal) (cl_program program, const char *kernel_name,
                                cl_int * errcode_ret);

    typedef cl_mem
        (*clCreateBufferLocal) (cl_context context,
                                cl_mem_flags flags, size_t size, void *host_ptr,
                                cl_int * errcode_ret);

    typedef cl_int
        (*clEnqueueWriteBufferLocal) (cl_command_queue command_queue,
                                      cl_mem buffer,
                                      cl_bool blocking_write,
                                      size_t offset,
                                      size_t cb,
                                      const void *ptr,
                                      cl_uint num_events_in_wait_list,
                                      const cl_event * event_wait_list, cl_event * event);

    typedef cl_int
        (*clSetKernelArgLocal) (cl_kernel kernel, cl_uint arg_index, size_t arg_size,
                                const void *arg_value);

    typedef cl_int
        (*clEnqueueNDRangeKernelLocal) (cl_command_queue command_queue,
                                        cl_kernel kernel,
                                        cl_uint work_dim,
                                        const size_t * global_work_offset,
                                        const size_t * global_work_size,
                                        const size_t * local_work_size,
                                        cl_uint num_events_in_wait_list,
                                        const cl_event * event_wait_list, cl_event * event);

/* Enqueued Commands for GPU memory read */
    typedef cl_int
        (*clEnqueueReadBufferLocal) (cl_command_queue command_queue,
                                     cl_mem buffer,
                                     cl_bool blocking_read,
                                     size_t offset,
                                     size_t cb,
                                     void *ptr,
                                     cl_uint num_events_in_wait_list,
                                     const cl_event * event_wait_list, cl_event * event);

    typedef cl_int(*clReleaseMemObjectLocal) (cl_mem memobj);
    typedef cl_int(*clReleaseKernelLocal) (cl_kernel kernel);
    typedef cl_int(*clFinishLocal) (cl_command_queue hInCmdQueue);

    typedef cl_int
        (*clGetContextInfoLocal) (cl_context context,
                                  cl_context_info param_name,
                                  size_t param_value_size, void *param_value,
                                  size_t * param_value_size_ret);

    typedef cl_int
        (*clGetProgramBuildInfoLocal) (cl_program program,
                                       cl_device_id device,
                                       cl_program_build_info param_name,
                                       size_t param_value_size,
                                       void *param_value, size_t * param_value_size_ret);

    typedef cl_int
        (*clGetProgramInfoLocal) (cl_program program,
                                  cl_program_info param_name,
                                  size_t param_value_size, void *param_value,
                                  size_t * param_value_size_ret);

    typedef cl_int(*clReleaseProgramLocal) (cl_program program);
    typedef cl_int(*clReleaseCommandQueueLocal) (cl_command_queue command_queue);
    typedef cl_int(*clReleaseContextLocal) (cl_context context);

    typedef cl_int
        (*clGetDeviceInfoLocal) (cl_device_id device,
                                 cl_device_info param_name,
                                 size_t param_value_size, void *param_value,
                                 size_t * param_value_size_ret);

    typedef cl_int
        (*clGetPlatformInfoLocal) (cl_platform_id platform,
                                   cl_platform_info param_name,
                                   size_t param_value_size, void *param_value,
                                   size_t * param_value_size_ret);

    typedef cl_int(*clFlushLocal) (cl_command_queue hInCmdQueue);

    typedef cl_int(*clWaitForEventsLocal) (cl_uint num_events, const cl_event * event_list);

    typedef cl_sampler
        (*clCreateSamplerLocal) (cl_context context,
                                 cl_bool normalized_coords,
                                 cl_addressing_mode addressing_mode,
                                 cl_filter_mode filter_mode, cl_int * errcode_ret);

    typedef cl_int
        (*clGetCommandQueueInfoLocal) (cl_command_queue command_queue,
                                       cl_command_queue_info param_name,
                                       size_t param_value_size,
                                       void *param_value, size_t * param_value_size_ret);

    typedef void *(*clEnqueueMapBufferLocal) (cl_command_queue command_queue,
                                              cl_mem buffer,
                                              cl_bool blocking_map,
                                              cl_map_flags map_flags,
                                              size_t offset,
                                              size_t cb,
                                              cl_uint num_events_in_wait_list,
                                              const cl_event * event_wait_list,
                                              cl_event * event, cl_int * errcode_ret);

    typedef cl_int(*clReleaseEventLocal) (cl_event event);
    typedef cl_int
        (*clGetEventProfilingInfoLocal) (cl_event event,
                                         cl_profiling_info param_name,
                                         size_t param_value_size,
                                         void *param_value, size_t * param_value_size_ret);

    typedef cl_int(*clReleaseSamplerLocal) (cl_sampler sampler);

    typedef cl_int
        (*clGetKernelWorkGroupInfoLocal) (cl_kernel kernel,
                                          cl_device_id device,
                                          cl_kernel_work_group_info param_name,
                                          size_t param_value_size,
                                          void *param_value, size_t * param_value_size_ret);
    typedef cl_mem
        (*clCreateImage2DLocal) (cl_context context,
                                 cl_mem_flags flags,
                                 const cl_image_format * image_format,
                                 size_t image_width,
                                 size_t image_height,
                                 size_t image_row_pitch, void *host_ptr, cl_int * errcode_ret);

    typedef cl_int
        (*clEnqueueCopyBufferLocal) (cl_command_queue command_queue,
                                     cl_mem src_buffer,
                                     cl_mem dst_buffer,
                                     size_t src_offset,
                                     size_t dst_offset,
                                     size_t cb,
                                     cl_uint num_events_in_wait_list,
                                     const cl_event * event_wait_list, cl_event * event);
    typedef cl_int(*clRetainEventLocal) (cl_event event);
    typedef cl_int(*clRetainMemObjectLocal) (cl_mem memobj);
    typedef cl_int(*clRetainKernelLocal) (cl_kernel kernel);
    typedef cl_int(*clRetainCommandQueueLocal) (cl_command_queue command_queue);
    typedef cl_int
        (*clEnqueueUnmapMemObjectLocal) (cl_command_queue command_queue,
                                         cl_mem memobj,
                                         void *mapped_ptr,
                                         cl_uint num_events_in_wait_list,
                                         const cl_event * event_wait_list, cl_event * event);

#ifdef __cplusplus
}
#endif
#endif
