#ifndef __VOCL_PROXY_STRUCTURE_H__
#define __VOCL_PROXY_STRUCTURE_H__

#include <CL/opencl.h>
#include "vocl_proxyKernelArgProc.h"

typedef struct strVoclProxyMemory {
	cl_mem         mem, oldMem;
	cl_context     context;
	cl_mem_flags   flags;
	size_t         size;

	/* for migration */
	int            isWritten;
	cl_command_queue cmdQueue;
	char migStatus;

	struct strVoclProxyMemory *next;
} vocl_proxy_mem;

typedef struct strVoclKernel {
	cl_kernel      kernel, oldKernel;
	size_t         nameLen;
	char           *kernelName;
	/* indicate whether an argument is device memory*/
	int            argNum;
	char           *argFlag; 
	kernel_args    *args;
	char migStatus;

	cl_program     program;
	struct strVoclKernel *next;
} vocl_proxy_kernel;

typedef struct strVoclProgram {
	cl_program    program, oldProgram;
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

	char migStatus;

	struct strVoclProgram *next;
} vocl_proxy_program;

typedef struct strVoclProxyCommandQueue {
	cl_command_queue command_queue;
	cl_command_queue oldCommand_queue;
	cl_context       context;
	cl_device_id     deviceID;
	cl_command_queue_properties properties;
	
	cl_uint  memNum, memNo;
	vocl_proxy_mem **memPtr;

	cl_uint  kernelNum, kernelNo;
	vocl_proxy_kernel **kernelPtr;

	/* number of kernels launched, but not finished execution */
	cl_uint  kernelNumInCmdQueue;

	char migStatus;

	struct strVoclProxyCommandQueue *next;
} vocl_proxy_command_queue;

typedef struct strVoclProxyContext {
	cl_context       context, oldContext;
	cl_uint          deviceNum;
	cl_device_id     *devices;
	
	cl_uint cmdQueueNum, cmdQueueNo;
	vocl_proxy_command_queue **cmdQueuePtr;

	cl_uint  memNum, memNo;
	vocl_proxy_mem **memPtr;

	cl_uint programNum, programNo;
	vocl_proxy_program **programPtr;

	char migStatus;

	struct strVoclProxyContext *next;
} vocl_proxy_context;

typedef struct strVoclVirtualGPU {
    int appIndex;
	cl_device_id deviceID;
	cl_device_id oldDeviceID;
	int proxyRank;

    cl_uint contextNum, contextNo;  /* buffer size and number of contexts created */
	vocl_proxy_context **contextPtr;

    cl_uint cmdQueueNum, cmdQueueNo; /*buffer size and number of cmdQueue created */
    vocl_proxy_command_queue **cmdQueuePtr;

	char migStatus;
	char padding[3];

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

/* structures used for update virtual gpu info on the local side */
/* vocl structures used for update virtual gpu info */
typedef struct strVOCLMigMem {
    cl_mem mem;
    char migStatus;
    char padding[3];
} vocl_mig_mem;

typedef struct strVOCLMigCommandQueue {
    cl_command_queue command_queue;
    char migStatus;
    char padding[3];
} vocl_mig_command_queue;

typedef struct strVOCLMigSampler {
    cl_sampler sampler;
    char migStatus;
    char padding[3];
} vocl_mig_sampler;

typedef struct strVOCLMigKernel {
    cl_kernel kernel;
    char migStatus;
    char padding[3];
} vocl_mig_kernel;

typedef struct strVOCLMigProgram {
    cl_program program;
    char migStatus;
    char padding[3];

    cl_uint kernelNo;
    cl_kernel *kernelPtr;
} vocl_mig_program;

typedef struct strVOCLMigContext {
    cl_context context;
    char migStatus;
    char padding[3];

    cl_uint programNo;
    cl_program *programPtr;

    cl_uint memNo;
    cl_mem *memPtr;

    cl_uint cmdQueueNo;
    cl_command_queue *cmdQueuePtr;

    cl_uint samplerNo;
    cl_sampler *samplerPtr;
} vocl_mig_context;

typedef struct strVOCLMigVGPU {
    cl_device_id deviceID;
    int proxyIndex, proxyRank;
    char migStatus;
    char padding[3];

    cl_uint contextNo;
    cl_context *contextPtr;
} vocl_mig_vgpu;

#endif

