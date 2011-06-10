#include <stdio.h>
#include <stdlib.h>
#include "vocl_proxyKernelArgProc.h"
#include "vocl_proxyMigrationDeviceProc.h"

extern VOCL_PROXY_DEVICE *voclProxyGetDeviceIDFromCmdQueue(cl_command_queue cmdQueue);
extern int voclProxyIsMemoryOnDevice(VOCL_PROXY_DEVICE *devicePtr, cl_mem mem);
extern void voclProxyUpdateMemoryOnDevice(VOCL_PROXY_DEVICE *devicePtr, cl_mem mem, size_t size);

int voclProxyMigrationCheckKernelLaunch(cl_command_queue cmdQueue, kernel_args *argsPtr, int argsNum)
{
	VOCL_PROXY_DEVICE *devicePtr;
	int isMigrationNeeded = 0;
	size_t sizeForKernel = 0;
	cl_mem memory;
	int i;
	
	devicePtr = voclProxyGetDeviceIDFromCmdQueue(cmdQueue);
	for (i = 0; i < argsNum; i++)
	{
		/* if it is glboal memory. check if device memory is enough */
		if (argsPtr[i].isGlobalMemory == 1)
		{
			/* if the current global memory is not bind on the device */
			/* check whether the global memory size is enougth */
			memory = *((cl_mem *)argsPtr[i].arg_value);
			voclProxyUpdateMemoryOnDevice(devicePtr, memory, argsPtr[i].globalSize);
			/* global memory size is not enough */
			if (devicePtr->usedSize > devicePtr->globalSize)
			{
				isMigrationNeeded = 1;
			}
		}
	}

	printf("proxy, usedSize = %ld, available = %ld, isMigrationNeeded = %d\n", 
			devicePtr->usedSize, devicePtr->globalSize, isMigrationNeeded);
	return isMigrationNeeded;
}

int voclProxyMigrationCheckWriteBuffer(cl_command_queue cmdQueue, size_t size)
{
	VOCL_PROXY_DEVICE *devicePtr;
	devicePtr = voclProxyGetDeviceIDFromCmdQueue(cmdQueue);
	if (devicePtr->usedSize + size > devicePtr->globalSize)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

void voclProxyUpdateGlobalMemUsage(cl_command_queue cmdQueue, kernel_args *argsPtr, int argsNum)
{
	int i;
	cl_mem memory;
	VOCL_PROXY_DEVICE *devicePtr;
	devicePtr = voclProxyGetDeviceIDFromCmdQueue(cmdQueue);

	for (i = 0; i < argsNum; i++)
	{
		if (argsPtr[i].isGlobalMemory == 1)
		{
			/* add new memory to the device */
			memory = *((cl_mem *)argsPtr[i].arg_value);
			voclProxyUpdateMemoryOnDevice(devicePtr, memory, argsPtr[i].globalSize);
		}
	}

	return;
}

