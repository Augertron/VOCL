#ifndef __VOCL_PROXY_STRUCTURE_H__
#define __VOCL_PROXY_STRUCTURE_H__

#include <CL/opencl.h>

typedef struct strVoclProxyMemory {
	cl_mem         mem;
	cl_context     context;
	size_t         size;

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

#endif

