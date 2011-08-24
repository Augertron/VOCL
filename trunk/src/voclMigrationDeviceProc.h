#ifndef __VOCL_LIB_MIGRATION_H__
#define __VOCL_LIB_MIGRATION_H__
#include <CL/opencl.h>

typedef struct voclLibMemory {
	cl_mem mem;
	size_t size;
	struct voclLibMemory *next;
} LIB_MEM;

typedef struct voclLibCmdQueue {
	cl_command_queue cmdQueue;
	cl_uint kernelNumInCmdQueue;
	struct voclLibCmdQueue *next;
} LIB_CMD_QUEUE;

typedef struct strVoclLibDevice {
	cl_device_id     device;
	size_t           globalSize;
	size_t           usedSize;
	cl_uint          cmdQueueNum;
	LIB_CMD_QUEUE    *cmdQueuePtr;
	LIB_MEM          *memPtr;
	struct strVoclLibDevice *next;
} VOCL_LIB_DEVICE;



#endif
