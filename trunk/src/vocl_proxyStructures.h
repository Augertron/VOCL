#ifndef __VOCL_PROXY_STRUCTURE_H__
#define __VOCL_PROXY_STRUCTURE_H__

#include <CL/opencl.h>

typedef struct strVoclProxyMemory {
	cl_mem         mem;
	cl_context     context;
	size_t         size;
	int            isWritten;

	struct strVoclProxyBuffer *next;
} str_vocl_proxy_mem;

typedef struct strVoclProgram {
	cl_program    program;
	cl_context    context;
	char          *sourceString;
	size_t        sourceSize;
	int           stringNum;
	size_t        *stringSizeArray;
	char          *buildOptions;

	cl_uint       deviceNum;
	cl_device_id  *device_list;

	struct strVoclProgram *next;
} str_vocl_proxy_program;

typedef struct strVoclKernel {
	cl_kernel      kernel;
	cl_program     program;
	char           *kernelName;

	struct strVoclKernel *next;
} str_vocl_proxy_kernel;

typedef struct strVoclProxyCommandQueue {
	cl_command_queue command_queue;
	cl_context       context;
	cl_device_id     deviceID;
	cl_command_queue_properties properties;
	
	cl_uint  memNum, memNo;
	str_vocl_proxy_mem **memPtr;

	cl_uint  kernelNum, kernelNo;
	str_vocl_proxy_kernel **kernelPtr;

	/* number of kernels launched, but not finished execution */
	cl_uint  kernelNumInCmdQueue;

	struct strVoclProxyCommandQueue *next;
} str_vocl_proxy_command_queue;

typedef struct strVoclProxyContext {
	cl_context       context;
	cl_uint          deviceNum;
	cl_device_id     *devices
	
	cl_uint cmdQueueNum, cmdQueueNo;
	str_vocl_proxy_command_queue **cmdQueuePtrs;

	cl_uint  memNum, memNo;
	str_vocl_proxy_mem **memPtr;

	cl_uint programNum, programNo;
	str_vocl_proxy_program **programPtr;

	struct strVoclProxyContext *next;
} str_vocl_proxy_context;

typedef struct strVoclVirtualGPU {
    int appIndex;
	cl_device_id deviceID;

    cl_uint contextNum, contextNo;  /* buffer size and number of contexts created */
    cl_context contexts;

    cl_uint cmdQueueNum, cmdQueueNo; /*buffer size and number of cmdQueue created */
    cl_command_queue *cmdQueues;

	struct strVOCLVirtualGPU *next;
} vocl_virtual_gpu;

#endif

