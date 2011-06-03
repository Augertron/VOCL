#ifndef __VOCL_LIB_H__
#define __VOCL_LIB_H__
#include <CL/opencl.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

/* number of slave process */
#define MAX_NPS 100

/* for kernel arguments */
#define MAX_TAG 65535

/* vocl true and false */
#define VOCL_TRUE  1
#define VOCL_FALSE 0

/* parameters of buffer pool */
/* write buffer */
#define VOCL_WRITE_BUFFER_NUM 8
//#define VOCL_WRITE_BUFFER_SIZE 64108864 /* 64MB 1024 X 1024 X 64 */
#define VOCL_WRITE_BUFFER_SIZE 33554432 /* 64MB 1024 X 1024 X 64 */
#define VOCL_WRITE_TAG 5000

/* read buffer */
#define VOCL_READ_BUFFER_NUM 16
//#define VOCL_READ_BUFFER_SIZE 64108864  /* 64MB 1024 X 1024 X 64 */
#define VOCL_READ_BUFFER_SIZE 33554432  /* 64MB 1024 X 1024 X 64 */
#define VOCL_READ_TAG 4000

/* macros for migration */
#define VOCL_MIG_BUF_NUM 8
#define VOCL_MIG_POOL_NUM 4
//#define VOCL_MIG_BUF_SIZE 64108864
#define VOCL_MIG_BUF_SIZE 33554432
#define VOCL_MIG_TAG 6000

#define VOCL_MIG_LOCAL_WT_BUF_AVALB     0
#define VOCL_MIG_LOCAL_WT_BUF_WAITDATA  1
#define VOCL_MIG_LOCAL_WT_BUF_WTGPUMEM  2

#define VOCL_MIG_LOCAL_RD_BUF_AVALB     3
#define VOCL_MIG_LOCAL_RD_BUF_RDGPUMEM  4
#define VOCL_MIG_LOCAL_RD_BUF_MPISEND   5

/* flag use for local to local migration */
#define VOCL_MIG_LOCAL_RW_BUF_AVALB     6
#define VOCL_MIG_LOCAL_RW_BUF_RDGPUMEM  7
#define VOCL_MIG_LOCAL_RW_BUF_WTGPUMEM  8

/* for api function identification */
#define GET_PLATFORM_ID_FUNC        10
#define GET_DEVICE_ID_FUNC          11
#define CREATE_CONTEXT_FUNC         12
#define LOAD_SOURCE_FUNC            13
#define CREATE_PROGRMA_WITH_SOURCE  14
#define CREATE_COMMAND_QUEUE_FUNC   15
#define BUILD_PROGRAM               16
#define CREATE_KERNEL               17
#define CREATE_BUFFER_FUNC          18
#define ENQUEUE_WRITE_BUFFER        19
#define SET_KERNEL_ARG              20
#define ENQUEUE_ND_RANGE_KERNEL     21
#define ENQUEUE_READ_BUFFER         22
#define RELEASE_MEM_OBJ             23
#define FINISH_FUNC                 24
#define GET_CONTEXT_INFO_FUNC       25
#define CL_RELEASE_KERNEL_FUNC      26
#define GET_BUILD_INFO_FUNC         27
#define GET_PROGRAM_INFO_FUNC       28
#define REL_PROGRAM_FUNC            29
#define REL_COMMAND_QUEUE_FUNC      30
#define REL_CONTEXT_FUNC            31
#define GET_DEVICE_INFO_FUNC        32
#define GET_PLATFORM_INFO_FUNC      33
#define FLUSH_FUNC                  34
#define WAIT_FOR_EVENT_FUNC         35
#define GET_CMD_QUEUE_INFO_FUNC     36
#define CREATE_SAMPLER_FUNC         37
#define ENQUEUE_MAP_BUFF_FUNC       38
#define RELEASE_EVENT_FUNC          39
#define RELEASE_SAMPLER_FUNC        40
#define GET_EVENT_PROF_INFO_FUNC    41
#define GET_KERNEL_WGP_INFO_FUNC    42
#define CREATE_IMAGE_2D_FUNC        43
#define ENQ_COPY_BUFF_FUNC          44
#define RETAIN_EVENT_FUNC           45
#define RETAIN_MEMOBJ_FUNC          46
#define RETAIN_KERNEL_FUNC          47
#define RETAIN_CMDQUE_FUNC          48
#define ENQ_UNMAP_MEMOBJ_FUNC       49
#define MIG_MEM_WRITE_REQUEST       50
#define MIG_MEM_READ_REQUEST        51
#define MIG_MEM_WRITE_CMPLD         52
#define MIG_MEM_READ_CMPLD          53
#define MIG_GET_PROXY_RANK          54
#define MIG_SAME_REMOTE_NODE        55
#define MIG_SAME_REMOTE_NODE_CMPLD  56
#define MIGRATION_CHECK             57
#define PROGRAM_END                 58

#define CMSG_NUM                    60
#define DATAMSG_NUM                 100
#define TOTAL_MSG_NUM               (CMSG_NUM + DATAMSG_NUM)

#define GET_PLATFORM_ID_FUNC1       10000
#define GET_DEVICE_ID_FUNC1         10001
#define CREATE_CONTEXT_FUNC1        10002
#define LOAD_SOURCE_FUNC1           10003
#define CREATE_PROGRMA_WITH_SOURCE1 10004
#define CREATE_PROGRMA_WITH_SOURCE2 10005
#define BUILD_PROGRAM1              10006
#define CREATE_KERNEL1              10007
#define CREATE_BUFFER_FUNC1         10008
#define ENQUEUE_WRITE_BUFFER1       10009
#define ENQUEUE_WRITE_BUFFER2       10010
#define SET_KERNEL_ARG1             10011
#define ENQUEUE_ND_RANGE_KERNEL1    10012
#define ENQUEUE_ND_RANGE_KERNEL2    10013
#define ENQUEUE_ND_RANGE_KERNEL3    10014
#define ENQUEUE_ND_RANGE_KERNEL4    10015
#define ENQUEUE_READ_BUFFER1        10016
#define GET_CONTEXT_INFO_FUNC1      10017
#define GET_BUILD_INFO_FUNC1        10018
#define GET_PROGRAM_INFO_FUNC1      10019
#define GET_DEVICE_INFO_FUNC1       10020
#define GET_PLATFORM_INFO_FUNC1     10021
#define WAIT_FOR_EVENT_FUNC1        10022
#define GET_CMD_QUEUE_INFO_FUNC1    10023
#define ENQUEUE_MAP_BUFF_FUNC1      10024
#define GET_EVENT_PROF_INFO_FUNC1   10025
#define GET_KERNEL_WGP_INFO_FUNC1   10026
#define CREATE_IMAGE_2D_FUNC1       10027
#define ENQ_COPY_BUFF_FUNC1         10028
#define ENQ_UNMAP_MEMOBJ_FUNC1      10029

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
		int              argsNum;
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
        cl_uint num_events_in_wait_list;
        cl_int event_null_flag;
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
