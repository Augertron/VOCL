#ifndef __OPENCL_REMOTE_ACCESS_H__
#define __OPENCL_REMOTE_ACCESS_H__
#include <CL/opencl.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct strKernelArgs {
        cl_uint arg_index;
        size_t arg_size;
        char arg_value[64];
        cl_char arg_null_flag;
        //const void  *arg_value;
    } kernel_args;

    typedef struct strKernelInfo {
        cl_kernel kernel;
        cl_uint args_num;
        cl_bool args_allocated;
        kernel_args *args_ptr;
        struct strKernelInfo *next;
    } kernel_info;

//for slave process
#define MAX_NPS 10
    extern int slaveComm;
    extern int slaveCreated;
    extern int np;
    extern int errCodes[MAX_NPS];
    extern cl_uint readBufferTag;

//for storing kernel arguments
#define MAX_ARGS 100
#define MAX_TAG 65535

#define MIN_WRITE_TAG 1000
#define MAX_WRITE_TAG 2000
#define MIN_READ_TAG  3000
#define MAX_READ_TAG  4000

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
#define PROGRAM_END                 50

#define CMSG_NUM                    (PROGRAM_END-OFFSET+1)
#define DATAMSG_NUM                 500
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


    extern kernel_info *kernelInfo;

//1
#define GET_PLAT_FORM_ELEM_NUM 1
    struct strGetPlatformIDs {
        cl_uint num_entries;
        cl_platform_id *platforms;
        cl_uint num_platforms;
        cl_int res;
    };

//2
#define GET_DEVICE_IDS_ELEM_NUM 2
    struct strGetDeviceIDs {
        cl_platform_id platform;
        cl_device_type device_type;
        cl_uint num_entries;
        cl_device_id *devices;
        cl_uint num_devices;
        cl_int res;
    };

//3
#define CREATE_CONTEXT_ELEM_NUM 1
    struct strCreateContext {
        cl_context_properties properties;
        cl_uint num_devices;
        cl_device_id *devices;
        //CL_CALLBACK *                 pfn_notify;
        void *user_data;
        cl_int errcode_ret;
        cl_context hContext;
    };

//4
#define CREATE_COMMAND_QUEUE_ELEM_NUM 1
    struct strCreateCommandQueue {
        cl_context context;
        cl_device_id device;
        cl_command_queue_properties properties;
        cl_command_queue clCommand;
        cl_int errcode_ret;
    };

//5
#define CREATE_PROGRAM_WITH_SOURCE_ELEM_NUM 2
    struct strCreateProgramWithSource {
        cl_context context;
        cl_uint count;
        size_t lengths;
        cl_program clProgram;
        cl_int errcode_ret;
    };

//6
#define BUILD_PROGRAM_ELEM_NUM 2
    struct strBuildProgram {
        cl_program program;
        cl_uint num_devices;
        cl_device_id *device_list;
        cl_uint optionLen;
        //CL_CALLBACK *        pfn_notify;
        void *user_data;
        cl_int res;
    };

//7
    struct strCreateKernel {
        cl_program program;
        size_t kernelNameSize;
        cl_int errcode_ret;
        cl_kernel kernel;
    };

//8
#define CREATE_BUFFER_ELEM_NUM 2
/* Memory Object APIs */
    struct strCreateBuffer {
        cl_context context;
        cl_mem_flags flags;
        size_t size;
        //void *       host_ptr;
        cl_int host_ptr_flag;
        cl_int errcode_ret;
        cl_mem deviceMem;
    };

//9
#define ENQUEUE_WRITE_BUFFER_ELEM_NUM 4
    struct strEnqueueWriteBuffer {
        cl_command_queue command_queue;
        cl_mem buffer;
        cl_bool blocking_write;
        cl_int tag;
        size_t offset;
        size_t cb;
        cl_uint num_events_in_wait_list;
        cl_int event_null_flag; //1, flag is NULL, 0, is NOT NULL
        cl_event event;
        cl_int res;
    };

//10
#define SET_KERNEL_ARG_ELEM_NUM 3
    struct strSetKernelArg {
        cl_kernel kernel;
        cl_uint arg_index;
        size_t arg_size;
        const void *arg_value;
        cl_int res;
    };

//11
#define ENQUEUE_ND_RANGE_KERNEL_ELEM_NUM 5
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

//12
#define ENQUEUE_READ_BUFFER_ELEM_NUM
    struct strEnqueueReadBuffer {
        cl_command_queue command_queue;
        cl_mem buffer;
        cl_bool blocking_read;
        cl_uint readBufferTag;
        size_t offset;
        size_t cb;
        cl_uint num_events_in_wait_list;
        cl_int event_null_flag; //1: the event point is NULL. 0: the event point is NOT NULL
        cl_event event;
        cl_int res;
    };

//13
    struct strReleaseMemObject {
        cl_mem memobj;
        cl_int res;
    };

//14
    struct strReleaseKernel {
        cl_kernel kernel;
        cl_int res;
    };

//15
    struct strFinish {
        cl_command_queue command_queue;
        cl_int res;
    };

//16
    struct strGetContextInfo {
        cl_context context;
        cl_context_info param_name;
        size_t param_value_size;
        void *param_value;
        size_t param_value_size_ret;
        cl_int res;
    };

//17
    struct strGetProgramBuildInfo {
        cl_program program;
        cl_device_id device;
        cl_program_build_info param_name;
        size_t param_value_size;
        void *param_value;
        size_t param_value_size_ret;
        cl_int res;
    };

//18
    struct strGetProgramInfo {
        cl_program program;
        cl_program_info param_name;
        size_t param_value_size;
        void *param_value;
        size_t param_value_size_ret;
        cl_int res;
    };

//19
    struct strReleaseProgram {
        cl_program program;
        cl_int res;
    };

//20
    struct strReleaseCommandQueue {
        cl_command_queue command_queue;
        cl_int res;
    };

//21
    struct strReleaseContext {
        cl_context context;
        cl_int res;
    };

//22
    struct strGetDeviceInfo {
        cl_device_id device;
        cl_device_info param_name;
        size_t param_value_size;
        void *param_value;
        size_t param_value_size_ret;
        cl_int res;
    };

//23
    struct strGetPlatformInfo {
        cl_platform_id platform;
        cl_platform_info param_name;
        size_t param_value_size;
        void *param_value;
        size_t param_value_size_ret;
        cl_int res;
    };

//24
    struct strFlush {
        cl_command_queue command_queue;
        cl_int res;
    };

//25
    struct strWaitForEvents {
        cl_uint num_events;
        cl_int res;
    };

//26
    struct strCreateSampler {
        cl_context context;
        cl_bool normalized_coords;
        cl_addressing_mode addressing_mode;
        cl_filter_mode filter_mode;
        cl_int errcode_ret;
        cl_sampler sampler;
    };

//27
    struct strGetCommandQueueInfo {
        cl_command_queue command_queue;
        cl_command_queue_info param_name;
        size_t param_value_size;
        void *param_value;
        size_t param_value_size_ret;
        cl_int res;
    };

//28
    struct strEnqueueMapBuffer {
        cl_command_queue command_queue;
        cl_mem buffer;
        cl_bool blocking_map;
        cl_map_flags map_flags;
        size_t offset;
        size_t cb;
        cl_uint num_events_in_wait_list;
        cl_int event_null_flag; //1: NULL, 0: NOT NULL
        cl_event event;
        cl_int errcode_ret;
        void *ret_ptr;
    };

//29
    struct strReleaseEvent {
        cl_event event;
        cl_int res;
    };

//30
    struct strGetEventProfilingInfo {
        cl_event event;
        cl_profiling_info param_name;
        size_t param_value_size;
        void *param_value;
        size_t param_value_size_ret;
        cl_int res;
    };

//31
    struct strReleaseSampler {
        cl_sampler sampler;
        cl_int res;
    };

//32
    struct strGetKernelWorkGroupInfo {
        cl_kernel kernel;
        cl_device_id device;
        cl_kernel_work_group_info param_name;
        size_t param_value_size;
        void *param_value;
        size_t param_value_size_ret;
        cl_int res;
    };

//33
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

//34
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

//35
    struct strRetainEvent {
        cl_event event;
        cl_int res;
    };

//36
    struct strRetainMemObject {
        cl_mem memobj;
        cl_int res;
    };

//37
    struct strRetainKernel {
        cl_kernel kernel;
        cl_int res;
    };

//38
    struct strRetainCommandQueue {
        cl_command_queue command_queue;
        cl_int res;
    };

//39
    struct strEnqueueUnmapMemObject {
        cl_command_queue command_queue;
        cl_mem memobj;
        void *mapped_ptr;
        cl_uint num_events_in_wait_list;
        cl_int event_null_flag;
        cl_event event;
        cl_int res;
    };

    void mpiFinalize();

#ifdef __cplusplus
}
#endif
#endif
