#include "vocl_proxyKernelArgProc.h"
#include "vocl_proxyMigrationDeviceProc.h"

extern cl_device_id voclProxyGetDeviceIDFromCmdQueue(cl_command_queue cmdQueue);
extern int voclProxyIsMemoryOnDevice(VOCL_PROXY_DEVICE *devicePtr, cl_mem mem);
extern void voclProxyUpdateMemoryOnDevice(VOCL_PROXY_DEVICE *devicePtr, cl_mem mem, size_t size);

int voclProxyIsMigrationNeeded(cl_command_queue cmdQueue, kernel_args *argsPtr, int argsNum)
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
			if (voclProxyIsMemoryOnDevice(devicePtr, memory) == 0)
			{
				/* not added to the device yet, add it now */
				voclProxyUpdateMemoryOnDevice(devicePtr, memory, argsPtr[i].globalSize);
				/* global memory size is not enough */
				sizeForKernel += argsPtr[i].globalSize;
				printf("i = %d, usedSize = %ld, sizeofKernel = %ld, argSize = %ld, globalSize = %ld\n",
						i, devicePtr->usedSize, sizeForKernel, argsPtr[i].globalSize, devicePtr->globalSize);
				if (devicePtr->usedSize + sizeForKernel > devicePtr->globalSize)
				{
					isMigrationNeeded = 1;
					break;
				}
			}
		}
	}
//	return 1;

	printf("proxy, isMigrationNeeded = %d\n", isMigrationNeeded);

	return isMigrationNeeded;
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
