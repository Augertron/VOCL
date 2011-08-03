#ifndef __VOCL_PROXY_MIGRATION_H__
#define __VOCL_PROXY_MIGRATION_H__
#include <CL/opencl.h>

typedef struct voclProxyMemory {
	cl_mem mem;
	size_t size;
	struct voclProxyMemory *next;
} PROXY_MEM;

typedef struct voclProxyCmdQueue {
	cl_command_queue cmdQueue;
	cl_uint kernelNumInCmdQueue;
	struct voclProxyCmdQueue *next;
} PROXY_CMD_QUEUE;

typedef struct strVoclProxyDevice {
	cl_device_id     device;
	size_t           globalSize;
	size_t           usedSize;
	int              cmdQueueNum;
	PROXY_CMD_QUEUE  *cmdQueuePtr;
	PROXY_MEM        *memPtr;
	struct strVoclProxyDevice *next;
} VOCL_PROXY_DEVICE;

#endif
