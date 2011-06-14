#include <stdio.h>
#include <stdlib.h>
#include "vocl_proxyMigrationDeviceProc.h"

VOCL_PROXY_DEVICE *voclProxyDevicePtr = NULL;

/* device operations */
void voclProxyCreateDevice(cl_device_id device, size_t globalSize)
{
	VOCL_PROXY_DEVICE *devicePtr = (VOCL_PROXY_DEVICE *)malloc(sizeof(VOCL_PROXY_DEVICE));
	devicePtr->device = device;
	devicePtr->globalSize = globalSize;
	//devicePtr->globalSize = 3000000000;

	devicePtr->usedSize = 0;
	devicePtr->cmdQueuePtr = NULL;
	devicePtr->memPtr = NULL;
	devicePtr->next = voclProxyDevicePtr;
	voclProxyDevicePtr = devicePtr;

	return;
}

void voclProxyReleaseDevice(cl_device_id device)
{
	VOCL_PROXY_DEVICE *devicePtr, *curDevice, *preDevice;
	PROXY_CMD_QUEUE *cmdQueuePtr, *curCmdQueuePtr;
	PROXY_MEM *memPtr, *curMemPtr;
	if (voclProxyDevicePtr->device == device)
	{
		devicePtr = voclProxyDevicePtr;
		voclProxyDevicePtr = voclProxyDevicePtr->next;
		cmdQueuePtr = devicePtr->cmdQueuePtr;
		while(cmdQueuePtr != NULL)
		{
			curCmdQueuePtr = cmdQueuePtr;
			cmdQueuePtr = cmdQueuePtr->next;
			free(curCmdQueuePtr);
		}

		memPtr = devicePtr->memPtr;
		while(memPtr != NULL)
		{
			curMemPtr = memPtr;
			memPtr = memPtr->next;
			free(curMemPtr);
		}

		free(devicePtr);

		return;
	}

	devicePtr = NULL;
	preDevice = voclProxyDevicePtr;
	curDevice = voclProxyDevicePtr->next;
	while (curDevice != NULL)
	{
		if (curDevice->device == device)
		{
			devicePtr = curDevice;
			break;
		}
		preDevice = curDevice;
		curDevice = curDevice->next;
	}

	if (devicePtr == NULL)
	{
		printf("voclProxyReleaseDevice, device does not exist!\n");
		exit (1);
	}

	preDevice->next = curDevice->next;
	cmdQueuePtr = devicePtr->cmdQueuePtr;
	while(cmdQueuePtr != NULL)
	{
		curCmdQueuePtr = cmdQueuePtr;
		cmdQueuePtr = cmdQueuePtr->next;
		free(curCmdQueuePtr);
	}

	memPtr = devicePtr->memPtr;
	while(memPtr != NULL)
	{
		curMemPtr = memPtr;
		memPtr = memPtr->next;
		free(curMemPtr);
	}

	free(devicePtr);

	return;
}

void voclProxyReleaseAllDevices()
{
	VOCL_PROXY_DEVICE *devicePtr, *curDevicePtr;
	PROXY_CMD_QUEUE *cmdQueuePtr, *curCmdQueuePtr;
	PROXY_MEM *memPtr, *curMemPtr;

	curDevicePtr = voclProxyDevicePtr;
	while (curDevicePtr != NULL)
	{
		devicePtr = curDevicePtr;
		curDevicePtr = curDevicePtr->next;
		/* delete the command queue */
		curCmdQueuePtr = devicePtr->cmdQueuePtr;
		while (curCmdQueuePtr != NULL)
		{
			cmdQueuePtr = curCmdQueuePtr;
			curCmdQueuePtr = curCmdQueuePtr->next;
			free(cmdQueuePtr);
		}
		/* delete memory  */
		curMemPtr = devicePtr->memPtr;
		while (curMemPtr != NULL)
		{
			memPtr = curMemPtr;
			curMemPtr = curMemPtr->next;
			free(memPtr);
		}

		/* delete device info */
		free(devicePtr);
	}

	return;
}

VOCL_PROXY_DEVICE *voclProxyGetDevicePtr(cl_device_id device)
{
	VOCL_PROXY_DEVICE *devicePtr = voclProxyDevicePtr;
	while (devicePtr != NULL)
	{
		if (devicePtr->device == device)
		{
			break;
		}
		devicePtr = devicePtr->next;
	}

	if (devicePtr == NULL)
	{
		printf("voclProxyGetDevicePtr, device does not exist!\n");
		exit (1);
	}

	return devicePtr;
}

VOCL_PROXY_DEVICE *voclProxyIsDeviceExist(cl_device_id device)
{
	VOCL_PROXY_DEVICE *devicePtr = voclProxyDevicePtr;
	while (devicePtr != NULL)
	{
		if (devicePtr->device == device)
		{
			break;
		}
		devicePtr = devicePtr->next;
	}

	if (devicePtr == NULL)
	{
		return 0;
	}
	else
	{
		return 1;
	}
}

/* command queue operations */
void voclProxyUpdateCmdQueueOnDevicePtr(VOCL_PROXY_DEVICE *devicePtr, cl_command_queue cmdQueue)
{
	PROXY_CMD_QUEUE *cmdQueuePtr, *curCmdQueuePtr;
	curCmdQueuePtr = devicePtr->cmdQueuePtr;
	while (curCmdQueuePtr != NULL)
	{
		if (cmdQueue == curCmdQueuePtr->cmdQueue)
		{
			break;
		}
		curCmdQueuePtr = curCmdQueuePtr->next;
	}

	/* if the command queue is not added yet, add it here */
	/* otherwise, do nothing */
	if (curCmdQueuePtr == NULL)
	{
		cmdQueuePtr = (PROXY_CMD_QUEUE *)malloc(sizeof(PROXY_CMD_QUEUE));
		cmdQueuePtr->cmdQueue = cmdQueue;
		cmdQueuePtr->next = devicePtr->cmdQueuePtr;
		devicePtr->cmdQueuePtr = cmdQueuePtr;
	}

	return;
}

void voclProxyUpdateCmdQueueOnDeviceID(cl_device_id device, cl_command_queue cmdQueue)
{
	VOCL_PROXY_DEVICE *devicePtr = voclProxyGetDevicePtr(device);
	voclProxyUpdateCmdQueueOnDevicePtr(devicePtr, cmdQueue);
}

VOCL_PROXY_DEVICE *voclProxyGetDeviceIDFromCmdQueue(cl_command_queue cmdQueue)
{
	PROXY_CMD_QUEUE *cmdQueuePtr;
	VOCL_PROXY_DEVICE *devicePtr = voclProxyDevicePtr;
	while (devicePtr != NULL)
	{
		cmdQueuePtr = devicePtr->cmdQueuePtr;
		while (cmdQueuePtr != NULL)
		{
			if (cmdQueue == cmdQueuePtr->cmdQueue)
			{
				break;
			}
			cmdQueuePtr = cmdQueuePtr->next;
		}

		if (cmdQueuePtr != NULL)
		{
			break;
		}
		else
		{
			devicePtr = devicePtr->next;
		}
	}

	if (devicePtr == NULL)
	{
		printf("voclProxyGetDeviceIDFromCmdQueue, command queue does not exist!\n");
		exit (1);
	}

	return devicePtr;
}

void voclProxyReleaseCommandQueue(cl_command_queue cmdQueue)
{
	PROXY_CMD_QUEUE *preCmdQueuePtr, *curCmdQueuePtr;
	VOCL_PROXY_DEVICE *devicePtr = voclProxyDevicePtr;
	while (devicePtr != NULL)
	{
		curCmdQueuePtr = devicePtr->cmdQueuePtr;
		if (curCmdQueuePtr != NULL)
		{
			if (curCmdQueuePtr->cmdQueue == cmdQueue)
			{
				curCmdQueuePtr = devicePtr->cmdQueuePtr;
				devicePtr->cmdQueuePtr = curCmdQueuePtr->next;
				free(curCmdQueuePtr);
				return;
			}

			preCmdQueuePtr = curCmdQueuePtr;
			curCmdQueuePtr = curCmdQueuePtr->next;
			while (curCmdQueuePtr != NULL)
			{
				if (cmdQueue == curCmdQueuePtr->cmdQueue)
				{
					break;
				}
				preCmdQueuePtr = curCmdQueuePtr;
				curCmdQueuePtr = curCmdQueuePtr->next;
			}

			if (curCmdQueuePtr != NULL)
			{
				break;
			}
		}
		devicePtr = devicePtr->next;
	}

	if (curCmdQueuePtr == NULL)
	{
		printf("voclProxyReleaseCommandQueue, command queue does not exist!\n");
		exit (1);
	}

	preCmdQueuePtr->next = curCmdQueuePtr->next;
	free(curCmdQueuePtr);

	return;
}

/* memory operations */
void voclProxyUpdateMemoryOnDevice(VOCL_PROXY_DEVICE *devicePtr, cl_mem mem, size_t size)
{
	PROXY_MEM *memPtr, *curMemPtr;
	curMemPtr = devicePtr->memPtr;
	while (curMemPtr != NULL)
	{
		if (mem == curMemPtr->mem)
		{
			break;
		}
		curMemPtr = curMemPtr->next;
	}

	/* if the mem is not added yet, add it here */
	/* otherwise, do nothing */
	if (curMemPtr == NULL)
	{
		memPtr = (PROXY_MEM *)malloc(sizeof(PROXY_MEM));
		memPtr->size = size;
		memPtr->mem = mem;
		memPtr->next = devicePtr->memPtr;
		devicePtr->memPtr = memPtr;
		devicePtr->usedSize += size;
	}

	return;
}

void voclProxyUpdateMemoryOnCmdQueue(cl_command_queue cmdQueue, cl_mem mem, size_t size)
{
	VOCL_PROXY_DEVICE *devicePtr = voclProxyGetDeviceIDFromCmdQueue(cmdQueue);
	voclProxyUpdateMemoryOnDevice(devicePtr, mem, size);
}

int voclProxyIsMemoryOnDevice(VOCL_PROXY_DEVICE *devicePtr, cl_mem mem)
/* is already on the device, return 1; otherwise, return 0 */
{
	PROXY_MEM *memPtr, *curMemPtr;
	curMemPtr = devicePtr->memPtr;
	while (curMemPtr != NULL)
	{
		if (mem == curMemPtr->mem)
		{
			break;
		}
		curMemPtr = curMemPtr->next;
	}

	/* if the mem is not on the device yet*/
	if (curMemPtr == NULL)
	{
		return 0;
	}
	else
	{
		return 1;
	}
}

//int voclProxyIsMemoryIsExisted(cl_command_queue cmdQueue, cl_mem)
//{
//	VOCL_PROXY_DEVICE devicePtr = voclProxyGetDeviceIDFromCmdQueue(cmdQueue);
//	return voclProxyIsMemoryOnDevice(devicePtr, mem);
//}

void voclProxyReleaseMem(cl_mem mem)
{
	PROXY_MEM *preMemPtr, *curMemPtr;
	VOCL_PROXY_DEVICE *devicePtr = voclProxyDevicePtr;
	while (devicePtr != NULL)
	{
		curMemPtr = devicePtr->memPtr;
		if (curMemPtr != NULL)
		{
			if (curMemPtr->mem == mem)
			{
				curMemPtr = devicePtr->memPtr;
				devicePtr->memPtr = curMemPtr->next;
				devicePtr->usedSize -= curMemPtr->size;
				free(curMemPtr);
			}
			else
			{
				preMemPtr = curMemPtr;
				curMemPtr = curMemPtr->next;
				while (curMemPtr != NULL)
				{
					if (mem == curMemPtr->mem)
					{
						preMemPtr->next = curMemPtr->next;
						devicePtr->usedSize -= curMemPtr->size;
						free(curMemPtr);
						break;
					}
					preMemPtr = curMemPtr;
					curMemPtr = curMemPtr->next;
				}
			}
		}
		devicePtr = devicePtr->next;
	}

	return;
}

