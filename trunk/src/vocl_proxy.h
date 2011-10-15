#ifndef __VOCL_PROXY_H__
#define __VOCL_PROXY_H__

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <CL/opencl.h>
#include <sched.h>
#include "vocl_proxy_macro.h"

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
    cl_context hContext;
    cl_int errcode_ret;
	char migStatus;
	char padding[3];
};

struct strCreateCommandQueue {
    cl_context context;
    cl_device_id device;
    cl_command_queue_properties properties;
    cl_command_queue clCommand;
    cl_int errcode_ret;
	char migStatus;
	char padding[3];
};

struct strCreateProgramWithSource {
    cl_context context;
    cl_uint count;
    size_t lengths;
    cl_program clProgram;
    cl_int errcode_ret;
	char migStatus;
	char padding[3];
};

struct strBuildProgram {
    cl_program program;
    cl_uint num_devices;
    cl_device_id *device_list;
    cl_uint optionLen;
    /*CL_CALLBACK *      pfn_notify; */
    void *user_data;
    cl_int res;
};

struct strCreateKernel {
    cl_program program;
	int argNum;
    size_t kernelNameSize;
    cl_kernel kernel;
    cl_int errcode_ret;
	char migStatus;
	char padding[3];
};

struct strCreateBuffer {
    cl_context context;
    cl_mem_flags flags;
    size_t size;
    cl_int host_ptr_flag;
    cl_mem deviceMem;
    cl_int errcode_ret;
	char migStatus;
	char padding[3];
};

struct strEnqueueWriteBuffer {
    cl_command_queue command_queue;
    cl_mem buffer;
    cl_bool blocking_write;
    cl_int tag;
    size_t offset;
    size_t cb;
    cl_uint num_events_in_wait_list;
    cl_int event_null_flag;     /* 1: event is NULL, 0: NOT NULL */
    cl_event event;
    cl_int res;
//	char cmdQueueMigStatus;
//	char memMigStatus;
//	char padding[2];
};

struct strSetKernelArg {
    cl_kernel kernel;
    cl_uint arg_index;
    size_t arg_size;
    const void *arg_value;
    cl_int res;
};

struct strVGPUMigration {
	size_t migMsgSize;
	cl_device_id deviceID;
	int   appIndex;
	cl_uint contextNum;
	cl_int retCode;
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
//	char cmdQueueMigStatus;
//	char kernelMigStatus;
//	char padding[2];
};

struct strEnqueueNDRangeKernelReply {
	cl_command_queue command_queue;
	cl_kernel kernel;
	cl_event event;
	cl_int res;
//	char cmdQueueMigStatus;
//	char kernelMigStatus;
//	char padding[2];
};

struct strEnqueueReadBuffer {
    cl_command_queue command_queue;
    cl_mem buffer;
    cl_bool blocking_read;
    cl_uint readBufferTag;
    size_t offset;
    size_t cb;
    cl_uint num_events_in_wait_list;
    cl_int event_null_flag;     /* 1: event is NULL, 0: NOT NULL */
    cl_event event;
    cl_int res;
//	char cmdQueueMigStatus;
//	char memMigStatus;
//	char padding[2];
};

struct strReleaseMemObject {
    cl_mem memobj;
    cl_int res;
	char migStatus;
	char padding[3];
};

struct strReleaseKernel {
    cl_kernel kernel;
    cl_int res;
	char migStatus;
	char padding[3];
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
	char migStatus;
	char padding[3];
};

struct strReleaseCommandQueue {
    cl_command_queue command_queue;
    cl_int res;
	char migStatus;
	char padding[3];
};

struct strReleaseContext {
    cl_context context;
    cl_int res;
	char migStatus;
	char padding[3];
};

struct strFinish {
    cl_command_queue command_queue;
    cl_int res;
//	char migStatus;
//	char padding[3];
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
    cl_int event_null_flag;     /* 1: NULL, 0: NOT NULL */
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

/* for task migration between GPUs */
struct strMigGPUMemoryWrite {
    cl_command_queue cmdQueue;
    cl_mem memory;
    size_t size;
    int source;
    int isFromLocal;
    MPI_Comm comm;
	int res;
};

struct strMigGPUMemoryRead {
    cl_command_queue cmdQueue;
    cl_mem memory;
    size_t size;
    int dest;
    int isToLocal;
    MPI_Comm comm;
	int res;
};

struct strMigGPUMemoryWriteCmpd {
    int source;
    int isFromLocal;
    MPI_Comm comm;
    int retCode;
};

struct strMigGPUMemoryReadCmpd {
    int dest;
    int isToLocal;
    MPI_Comm comm;
    int retCode;
};

struct strMigRemoteGPUMemoryRW {
	cl_command_queue oldCmdQueue;
	cl_command_queue newCmdQueue;
	cl_mem           oldMem;
	cl_mem           newMem;
	size_t           size;
	int res;
};

struct strMigUpdateVGPU {
	size_t msgSize;
};

struct strMigRemoteGPURWCmpd {
	int    res;
};

struct strForcedMigration {
	int status;
//	int rankThreshold;
	int res;
};

struct strKernelNumOnDevice {
	int deviceNum;
	int rankNo;
	int appIndex;
	cl_device_id deviceIDs[MAX_DEVICE_NUM_PER_NODE];
	int kernelNums[MAX_DEVICE_NUM_PER_NODE];
};

union CMSG_UNION {
	struct strGetProxyCommInfo tmpGetProxyCommInfo;
    struct strGetPlatformIDs tmpGetPlatformID;
    struct strGetDeviceIDs tmpGetDeviceIDs;
    struct strCreateContext tmpCreateContext;
    struct strCreateCommandQueue tmpCreateCommandQueue;
    struct strCreateProgramWithSource tmpCreateProgramWithSource;
    struct strBuildProgram tmpBuildProgram;
    struct strCreateKernel tmpCreateKernel;
    struct strCreateBuffer tmpCreateBuffer;
    struct strEnqueueWriteBuffer tmpEnqueueWriteBuffer;
    struct strSetKernelArg tmpSetKernelArg;
	struct strVGPUMigration tmpVGPUMigration;
    struct strEnqueueNDRangeKernel tmpEnqueueNDRangeKernel;
    struct strEnqueueReadBuffer tmpEnqueueReadBuffer;
    struct strReleaseMemObject tmpReleaseMemObject;
    struct strReleaseKernel tmpReleaseKernel;
    struct strGetContextInfo tmpGetContextInfo;
    struct strGetProgramBuildInfo tmpGetProgramBuildInfo;
    struct strGetProgramInfo tmpGetProgramInfo;
    struct strReleaseProgram tmpReleaseProgram;
    struct strReleaseCommandQueue tmpReleaseCommandQueue;
    struct strReleaseContext tmpReleaseContext;
    struct strFinish tmpFinish;
    struct strGetDeviceInfo tmpGetDeviceInfo;
    struct strGetPlatformInfo tmpGetPlatformInfo;
    struct strFlush tmpFlush;
    struct strWaitForEvents tmpWaitForEvents;
    struct strCreateSampler tmpCreateSampler;
    struct strGetCommandQueueInfo tmpGetCommandQueueInfo;
    struct strEnqueueMapBuffer tmpEnqueueMapBuffer;
    struct strReleaseEvent tmpReleaseEvent;
    struct strGetEventProfilingInfo tmpGetEventProfilingInfo;
    struct strReleaseSampler tmpReleaseSampler;
    struct strGetKernelWorkGroupInfo tmpGetKernelWorkGroupInfo;
    struct strCreateImage2D tmpCreateImage2D;
    struct strEnqueueCopyBuffer tmpEnqueueCopyBuffer;
    struct strRetainEvent tmpRetainEvent;
    struct strRetainMemObject tmpRetainMemObject;
    struct strRetainKernel tmpRetainKernel;
    struct strRetainCommandQueue tmpRetainCommandQueue;
    struct strEnqueueUnmapMemObject tmpEnqueueUnmapMemObject;
    struct strMigGPUMemoryWrite tmpMigGPUMemWrite;
    struct strMigGPUMemoryRead tmpMigGPUMemRead;
    struct strMigGPUMemoryWriteCmpd tmpMigWriteCmpd;
    struct strMigGPUMemoryReadCmpd tmpMigReadCmpd;
	struct strMigRemoteGPUMemoryRW tmpMigGPUMemRW;
	struct strMigUpdateVGPU tmpMigUpdateVGPU;
	struct strMigRemoteGPURWCmpd tmpMigGPUMemRWCmpd;
	struct strForcedMigration tmpForcedMigration;
	struct strKernelNumOnDevice tmpKernelNumOnDevice;
} CONTROL_MSG_UNION;

#endif
