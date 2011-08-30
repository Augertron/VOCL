#ifndef __VOCL_PROXY_STRUCTURE_H__
#define __VOCL_PROXY_STRUCTURE_H__

#include <CL/opencl.h>

typedef struct strVoclProxyMemory {
	cl_mem         mem;
	cl_context     context;
	cl_mem_flags   flags;
	size_t         size;
	int            isWritten;

	struct strVoclProxyMemory *next;
} vocl_proxy_mem;

typedef struct strVoclKernel {
	cl_kernel      kernel;
	cl_program     program;
	size_t         nameLen;
	char           *kernelName;

	struct strVoclKernel *next;
} vocl_proxy_kernel;

typedef struct strVoclProgram {
	cl_program    program;
	cl_context    context;
	char          *sourceString;
	size_t        sourceSize;
	int           stringNum;
	size_t        *stringSizeArray;

	size_t        buildOptionLen;
	char          *buildOptions;

	cl_uint       deviceNum;
	cl_device_id  *device_list;

	cl_uint       kernelNo, kernelNum;
	vocl_proxy_kernel **kernelPtr;

	struct strVoclProgram *next;
} vocl_proxy_program;

typedef struct strVoclProxyCommandQueue {
	cl_command_queue command_queue;
	cl_context       context;
	cl_device_id     deviceID;
	cl_command_queue_properties properties;
	
	cl_uint  memNum, memNo;
	vocl_proxy_mem **memPtr;

	cl_uint  kernelNum, kernelNo;
	vocl_proxy_kernel **kernelPtr;

	/* number of kernels launched, but not finished execution */
	cl_uint  kernelNumInCmdQueue;

	struct strVoclProxyCommandQueue *next;
} vocl_proxy_command_queue;

typedef struct strVoclProxyContext {
	cl_context       context;
	cl_uint          deviceNum;
	cl_device_id     *devices;
	
	cl_uint cmdQueueNum, cmdQueueNo;
	vocl_proxy_command_queue **cmdQueuePtr;

	cl_uint  memNum, memNo;
	vocl_proxy_mem **memPtr;

	cl_uint programNum, programNo;
	vocl_proxy_program **programPtr;

	struct strVoclProxyContext *next;
} vocl_proxy_context;

typedef struct strVoclVirtualGPU {
    int appIndex;
	cl_device_id deviceID;

    cl_uint contextNum, contextNo;  /* buffer size and number of contexts created */
	vocl_proxy_context **contextPtr;

    cl_uint cmdQueueNum, cmdQueueNo; /*buffer size and number of cmdQueue created */
    vocl_proxy_command_queue **cmdQueuePtr;

	struct strVoclVirtualGPU *next;
} vocl_virtual_gpu;

typedef struct voclProxyVirtualGPUMig {
	cl_uint contextNum;
	cl_uint programNum;
	cl_uint kernelNum;
	cl_uint cmdQueueNum;
	cl_uint memNum;
	size_t size;
} vocl_vgpu_msg;

#endif

