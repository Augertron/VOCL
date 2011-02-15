#ifndef __OPENCL_REMOTE_ACCESS_H__
#define __OPENCL_REMOTE_ACCESS_H__
#include <CL/opencl.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_NPS 10
extern int slaveComm;
extern int slaveCreated;
extern int np;
extern int errCodes[MAX_NPS];

#define GET_PLATFORM_ID_FUNC       10000
#define GET_PLATFORM_ID_FUNC1      10042
#define GET_DEVICE_ID_FUNC         10001
#define GET_DEVICE_ID_FUNC1        10027
#define CREATE_CONTEXT_FUNC        10002
#define CREATE_CONTEXT_FUNC1       10041
#define CREATE_COMMAND_QUEUE_FUNC  10003
#define LOAD_SOURCE_FUNC           10004
#define LOAD_SOURCE_FUNC1          10005
#define CREATE_PROGRMA_WITH_SOURCE 10006
#define CREATE_PROGRMA_WITH_SOURCE1 10007
#define BUILD_PROGRAM			   10008
#define BUILD_PROGRAM1			   10009
#define CREATE_KERNEL			   10010
#define CREATE_KERNEL1			   10011
#define CREATE_BUFFER_FUNC		   10012
#define CREATE_BUFFER_FUNC1		   10013
#define ENQUEUE_WRITE_BUFFER	   10014
#define ENQUEUE_WRITE_BUFFER1	   10015
#define SET_KERNEL_ARG			   10016
#define SET_KERNEL_ARG1			   10017
#define ENQUEUE_ND_RANGE_KERNEL    10018
#define ENQUEUE_ND_RANGE_KERNEL1   10019
#define ENQUEUE_ND_RANGE_KERNEL2   10020
#define ENQUEUE_ND_RANGE_KERNEL3   10021
#define ENQUEUE_READ_BUFFER        10022
#define ENQUEUE_READ_BUFFER1       10023
#define RELEASE_MEM_OBJ			   10024
#define FINISH_FUNC				   10025
#define CL_RELEASE_KERNEL_FUNC     10026
#define GET_CONTEXT_INFO_FUNC      10028
#define GET_CONTEXT_INFO_FUNC1     10029
#define GET_BUILD_INFO_FUNC        10030
#define GET_BUILD_INFO_FUNC1       10031
#define GET_PROGRAM_INFO_FUNC      10032
#define GET_PROGRAM_INFO_FUNC1     10033
#define REL_PROGRAM_FUNC		   10034
#define REL_COMMAND_QUEUE_FUNC	   10035
#define REL_CONTEXT_FUNC		   10036
#define GET_DEVICE_INFO_FUNC	   10037
#define GET_DEVICE_INFO_FUNC1	   10038
#define GET_PLATFORM_INFO_FUNC	   10039
#define GET_PLATFORM_INFO_FUNC1	   10040
#define PROGRAM_END				   11111

//1
#define GET_PLAT_FORM_ELEM_NUM 1
struct strGetPlatformIDs {
	cl_uint          num_entries;
	cl_platform_id  *platforms;
	cl_uint          num_platforms;
	cl_int           res;
};

//2
#define GET_DEVICE_IDS_ELEM_NUM 2
struct strGetDeviceIDs
{
	cl_platform_id   platform;
	cl_device_type   device_type;
	cl_uint          num_entries;
	cl_device_id     *devices;
	cl_uint          num_devices;
	cl_int			 res;
};

//3
#define CREATE_CONTEXT_ELEM_NUM 1
struct strCreateContext {
	cl_context_properties         properties;
	cl_uint                       num_devices;
	cl_device_id                  *devices;
	//CL_CALLBACK *                 pfn_notify;
	void *                        user_data;
	cl_int                        errcode_ret;
	cl_context					  hContext;
};

//4
#define CREATE_COMMAND_QUEUE_ELEM_NUM 1
struct strCreateCommandQueue {
	cl_context                     context;
	cl_device_id                   device;
	cl_command_queue_properties    properties;
	cl_command_queue			   clCommand;
	cl_int                         errcode_ret;
};

//5
#define CREATE_PROGRAM_WITH_SOURCE_ELEM_NUM 2
struct strCreateProgramWithSource {
	cl_context        context;
	cl_uint           count;
	size_t            lengths;
	cl_program		  clProgram;
	cl_int            errcode_ret;
};

//6
#define BUILD_PROGRAM_ELEM_NUM 2
struct strBuildProgram {
	cl_program           program;
	cl_uint              num_devices;
	cl_device_id        *device_list;
	cl_uint				 optionLen;
	//CL_CALLBACK *        pfn_notify;
	void *               user_data;
	cl_int				 res;
};

//7
struct strCreateKernel {
	cl_program      program;
	size_t			kernelNameSize;
	cl_int          errcode_ret;
	cl_kernel		kernel;
};

//8
#define CREATE_BUFFER_ELEM_NUM 2
/* Memory Object APIs */
struct strCreateBuffer {
	cl_context   context;
	cl_mem_flags flags;
	size_t       size;
	//void *       host_ptr;
	cl_int		 host_ptr_flag;
	cl_int       errcode_ret;
	cl_mem		 deviceMem;
};

//9
#define ENQUEUE_WRITE_BUFFER_ELEM_NUM 4
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
	cl_int 			   res;
};

//10
#define SET_KERNEL_ARG_ELEM_NUM 3
struct strSetKernelArg {
	cl_kernel    kernel;
	cl_uint      arg_index;
	size_t       arg_size;
	const void * arg_value;
	cl_int		 res;
};

//11
#define ENQUEUE_ND_RANGE_KERNEL_ELEM_NUM 5
struct strEnqueueNDRangeKernel {
	cl_command_queue command_queue;
	cl_kernel        kernel;
	cl_uint          work_dim;
	cl_int			 global_work_offset_flag;
	cl_int			 global_work_size_flag;
	cl_int			 local_work_size_flag;
	cl_uint          num_events_in_wait_list;
	//const cl_event * event_wait_list;
	cl_event *       event;
	cl_int			 res;
};

//12
#define ENQUEUE_READ_BUFFER_ELEM_NUM 
struct strEnqueueReadBuffer {
	cl_command_queue    command_queue;
	cl_mem              buffer;
	cl_bool             blocking_read;
	size_t              offset;
	size_t              cb;
	cl_uint             num_events_in_wait_list;
	//const cl_event *    event_wait_list;
	//cl_event *          event;
	cl_int				res;
};

//13
struct strReleaseMemObject {
	cl_mem memobj;
	cl_int res;
};

//14
struct strReleaseKernel {
	cl_kernel kernel;
	cl_int    res;
};

//15
struct strFinish {
	cl_command_queue command_queue;
	cl_int res;
}; 

//16
struct strGetContextInfo {
	cl_context         context; 
	cl_context_info    param_name; 
	size_t             param_value_size; 
	void *             param_value; 
	size_t             param_value_size_ret;
	cl_int             res;
};

//17
struct strGetProgramBuildInfo {
	cl_program            program;
	cl_device_id          device;
	cl_program_build_info param_name;
	size_t                param_value_size;
	void *                param_value;
	size_t                param_value_size_ret;
	cl_int				  res;
};

//18
struct strGetProgramInfo {
	cl_program         program;
	cl_program_info    param_name;
	size_t             param_value_size;
	void *             param_value;
	size_t             param_value_size_ret;
	cl_int			   res;
};

//19
struct strReleaseProgram {
	cl_program  program;
	cl_int		res;
};

//20
struct strReleaseCommandQueue {
	cl_command_queue command_queue;
	cl_int			 res;
};

//21
struct strReleaseContext {
	cl_context context;
	cl_int	   res;
};

//22
struct strGetDeviceInfo {
	cl_device_id    device;
	cl_device_info  param_name;
	size_t          param_value_size;
	void *          param_value;
	size_t          param_value_size_ret;
	cl_int			res;
};

//23
struct strGetPlatformInfo {
	cl_platform_id    platform;
	cl_platform_info  param_name;
	size_t            param_value_size;
	void *            param_value;
	size_t            param_value_size_ret;
	cl_int			  res;
};


void mpiFinalize();

#ifdef __cplusplus
}
#endif

#endif

