#ifndef __VOCL_MIGRATION_H__
#define __VOCL_MIGRATION_H__
#include <stdio.h>
#include <stdlib.h>
#include "vocl_structures.h"

struct strVoclGPUMemInfo {
	int     gpuIndex;
	char    gpuName[256];
	size_t  globalSize;
	size_t  allocatedSize;
	int     isLocalGPU;
	int     isInUse;
	vocl_device_id deviceID;
};

//struct strVoclDeviceMemInfo 
	/* memory info */
//	cl_mem       clMem;
//	vocl_mem     voclMem;
//	size_t       size;
//	cl_mem_flags flags;
//	int          writtenFlag;

//	/* opencl resources the memory depend on */
//	int          platformIndex;
//	int          deviceIndex;
//	int          contextIndex;
//	int          cmdQueueIndex;
//	int          programIndex;
//	int          kernelIndex;
//};

#define VOCL_MIG_MEM_NUM      50
#define VOCL_MIG_PLATFORM_NUM 10
#define VOCL_MIG_DEVICE_NUM   50
#define VOCL_MIG_CONTEXT_NUM  50
#define VOCL_MIG_CMDQ_NUM     50
#define VOCL_MIG_PROGRAM_NUM  50
#define VOCL_MIG_KERNEL_NUM   50
#define VOCL_MIG_EVENT_NUM    400
#define VOCL_MIG_SAMPLER_NUM  20

struct strVoclAppInfo {
	/* for data communication */
	int     proxyIndex;
	int     proxyRank;
	MPI_Comm comm;
	MPI_Comm commData;

	int     memorySizeNeeded;

	vocl_mem *memPtr;
	int    memNum, memNo;

	vocl_platform_id   *platformPtr;
	int                platformNum, platformNo;
	vocl_device_id     *devicePtr;
	int                deviceNum, deviceNo;
	vocl_context       *contextPtr;
	int                contextNum, contextNo;
	vocl_command_queue *cmdQueuePtr, *oldCmdQueuePtr;
	int                cmdQueueNum, cmdQueueNo;

	vocl_program       *programPtr;
	int                programNum, programNo;
	vocl_kernel        *kernelPtr;
	int                kernelNum, kernelNo;
	vocl_event         *eventPtr;
	int                eventNum, eventNo;
	vocl_sampler       *samplerPtr;
	int                samplerNum, samplerNo;
};


#endif

