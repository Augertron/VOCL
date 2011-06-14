#include <stdio.h>
#include "voclStructures.h"
#include "voclKernelArgProc.h"
#include "voclMigrationDeviceProc.h"

extern cl_int dlCLGetPlatformIDs(cl_uint num_entries, cl_platform_id * platforms, cl_uint * num_platforms);
extern cl_int dlCLGetDeviceIDs(cl_platform_id platform,
                 cl_device_type device_type,
                 cl_uint num_entries, cl_device_id * devices, cl_uint * num_devices);
extern cl_int dlCLGetDeviceInfo(cl_device_id device,
                  cl_device_info param_name,
                  size_t param_value_size, void *param_value, size_t * param_value_size_ret);
extern size_t voclCommandQueueGetDeviceMemorySize(vocl_command_queue cmdQueue);
extern cl_command_queue voclVOCLCommandQueue2CLCommandQueue(vocl_command_queue command_queue);
extern kernel_info *getKernelPtr(cl_kernel kernel);

VOCL_LIB_DEVICE *voclLibDevicePtr = NULL;

/* device operations */
void voclLibCreateDevice(cl_device_id device, size_t globalSize)
{
	VOCL_LIB_DEVICE *devicePtr = (VOCL_LIB_DEVICE *)malloc(sizeof(VOCL_LIB_DEVICE));
	devicePtr->device = device;
	devicePtr->globalSize = globalSize;
//	devicePtr->globalSize = 1000000000;
	devicePtr->usedSize = 0;
	devicePtr->cmdQueuePtr = NULL;
	devicePtr->memPtr = NULL;
	devicePtr->next = voclLibDevicePtr;
	voclLibDevicePtr = devicePtr;

	return;
}

void voclLibReleaseDevice(cl_device_id device)
{
	VOCL_LIB_DEVICE *devicePtr, *curDevice, *preDevice;
	LIB_CMD_QUEUE *cmdQueuePtr, *curCmdQueuePtr;
	LIB_MEM *memPtr, *curMemPtr;
	if (voclLibDevicePtr->device == device)
	{
		devicePtr = voclLibDevicePtr;
		voclLibDevicePtr = voclLibDevicePtr->next;
		curCmdQueuePtr = devicePtr->cmdQueuePtr;
		while(curCmdQueuePtr != NULL)
		{
			cmdQueuePtr = curCmdQueuePtr;
			curCmdQueuePtr = curCmdQueuePtr->next;
			free(cmdQueuePtr);
		}

		curMemPtr = devicePtr->memPtr;
		while(curMemPtr != NULL)
		{
			memPtr = curMemPtr;
			curMemPtr = curMemPtr->next;
			free(memPtr);
		}

		free(devicePtr);

		return;
	}

	devicePtr = NULL;
	preDevice = voclLibDevicePtr;
	curDevice = voclLibDevicePtr->next;
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
		printf("voclLibReleaseDevice, device does not exist!\n");
		exit (1);
	}

	preDevice->next = curDevice->next;
	curCmdQueuePtr = devicePtr->cmdQueuePtr;
	while(curCmdQueuePtr != NULL)
	{
		cmdQueuePtr = curCmdQueuePtr;
		curCmdQueuePtr = curCmdQueuePtr->next;
		free(cmdQueuePtr);
	}

	curMemPtr = devicePtr->memPtr;
	while(curMemPtr != NULL)
	{
		memPtr = curMemPtr;
		curMemPtr = curMemPtr->next;
		free(memPtr);
	}

	free(devicePtr);

	return;
}

void voclLibReleaseAllDevices()
{
	VOCL_LIB_DEVICE *devicePtr, *curDevicePtr;
	LIB_CMD_QUEUE *cmdQueuePtr, *curCmdQueuePtr;
	LIB_MEM *memPtr, *curMemPtr;

	curDevicePtr = voclLibDevicePtr;
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

VOCL_LIB_DEVICE *voclLibGetDevicePtr(cl_device_id device)
{
	VOCL_LIB_DEVICE *devicePtr = voclLibDevicePtr;
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
		printf("voclLibGetDevicePtr, device does not exist!\n");
		exit (1);
	}

	return devicePtr;
}

/* command queue operations */
void voclLibUpdateCmdQueueOnDevicePtr(VOCL_LIB_DEVICE *devicePtr, cl_command_queue cmdQueue)
{
	LIB_CMD_QUEUE *cmdQueuePtr, *curCmdQueuePtr;
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
		cmdQueuePtr = (LIB_CMD_QUEUE *)malloc(sizeof(LIB_CMD_QUEUE));
		cmdQueuePtr->cmdQueue = cmdQueue;
		cmdQueuePtr->next = devicePtr->cmdQueuePtr;
		devicePtr->cmdQueuePtr = cmdQueuePtr;
	}

	return;
}

void voclLibUpdateCmdQueueOnDeviceID(cl_device_id device, cl_command_queue cmdQueue)
{
	VOCL_LIB_DEVICE *devicePtr = voclLibGetDevicePtr(device);
	voclLibUpdateCmdQueueOnDevicePtr(devicePtr, cmdQueue);
}

VOCL_LIB_DEVICE *voclLibGetDeviceIDFromCmdQueue(cl_command_queue cmdQueue)
{
	LIB_CMD_QUEUE *cmdQueuePtr;
	VOCL_LIB_DEVICE *devicePtr = voclLibDevicePtr;
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
		printf("voclLibGetDeviceIDFromCmdQueue, command queue does not exist!\n");
		exit (1);
	}

	return devicePtr;
}

void voclLibReleaseCommandQueue(cl_command_queue cmdQueue)
{
	LIB_CMD_QUEUE *preCmdQueuePtr, *curCmdQueuePtr;
	VOCL_LIB_DEVICE *devicePtr = voclLibDevicePtr;
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
		printf("voclLibReleaseCommandQueue, command queue does not exist!\n");
		exit (1);
	}

	preCmdQueuePtr->next = curCmdQueuePtr->next;
	free(curCmdQueuePtr);

	return;
}

/* memory operations */
void voclLibUpdateMemoryOnDevice(VOCL_LIB_DEVICE *devicePtr, cl_mem mem, size_t size)
{
	LIB_MEM *memPtr, *curMemPtr;
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
		memPtr = (LIB_MEM *)malloc(sizeof(LIB_MEM));
		memPtr->size = size;
		memPtr->mem = mem;
		memPtr->next = devicePtr->memPtr;
		devicePtr->memPtr = memPtr;
		devicePtr->usedSize += size;
	}

	return;
}

int voclLibIsMemoryOnDevice(VOCL_LIB_DEVICE *devicePtr, cl_mem mem)
/* is already on the device, return 1; otherwise, return 0 */
{
	LIB_MEM *memPtr, *curMemPtr;
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

void voclLibReleaseMem(cl_mem mem)
{
	int isMemoryFound = 0;
	LIB_MEM *preMemPtr, *curMemPtr;
	VOCL_LIB_DEVICE *devicePtr = voclLibDevicePtr;
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
				isMemoryFound = 1;
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
						isMemoryFound = 1;
						break;
					}
					preMemPtr = curMemPtr;
					curMemPtr = curMemPtr->next;
				}
			}
		}
		devicePtr = devicePtr->next;
	}


	if (isMemoryFound == 0)
	{
		printf("voclLibReleaseMem, mem does not exist!\n");
		exit (1);
	}

	return;
}

/* get local device info */
void voclGetLocalDeviceInfo()
{
	cl_platform_id *platformIDs;
	cl_device_id *deviceIDs;
	cl_uint numPlatforms, numDevices;
	cl_ulong globalSize;
	int err, i, j;
	
	err = dlCLGetPlatformIDs(0, NULL, &numPlatforms);
	platformIDs = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);
	err = dlCLGetPlatformIDs(numPlatforms, platformIDs, NULL);
	for (i = 0; i < numPlatforms; i++)
	{
		err = dlCLGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
		/* it is possible there is no gpu on the local node */
		if (err == CL_SUCCESS)
		{
			deviceIDs = (cl_device_id *)malloc(sizeof(cl_device_id) * numDevices);
			err = dlCLGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_GPU, numDevices, deviceIDs, NULL);
			for (j = 0; j < numDevices; j++)
			{
				dlCLGetDeviceInfo(deviceIDs[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &globalSize, NULL);
				voclLibCreateDevice(deviceIDs[j], globalSize);
			}
			free(deviceIDs);
		}
	}
	free(platformIDs);

	return;
}

void voclLibUpdateGlobalMemUsage(cl_command_queue cmdQueue, kernel_args *argsPtr, int argsNum)
{
	int i;
	cl_mem memory;
	VOCL_LIB_DEVICE *devicePtr;
	devicePtr = voclLibGetDeviceIDFromCmdQueue(cmdQueue);

	for (i = 0; i < argsNum; i++)
	{
		if (argsPtr[i].isGlobalMemory == 1)
		{
			/* add new memory to the device */
			memory = *((cl_mem *)argsPtr[i].arg_value);
			voclLibUpdateMemoryOnDevice(devicePtr, memory, argsPtr[i].globalSize);
		}
	}

	return;
}

int voclCheckMigrationInKernelLaunch(vocl_command_queue cmdQueue, kernel_args *argsPtr, int argsNum)
{
	VOCL_LIB_DEVICE *devicePtr;
	int isMigrationNeeded = 0;
	size_t sizeForKernel = 0;
	cl_mem memory;
	int i;

	devicePtr = voclLibGetDeviceIDFromCmdQueue((cl_command_queue)cmdQueue);
	for (i = 0; i < argsNum; i++)
	{
		/* if it is glboal memory. check if device memory is enough */
		if (argsPtr[i].isGlobalMemory == 1)
		{
			/* if the current global memory is not bind on the device */
			/* check whether the global memory size is enougth */
			memory = *((cl_mem *)argsPtr[i].arg_value);
			voclLibUpdateMemoryOnDevice(devicePtr, memory, argsPtr[i].globalSize);
			if (devicePtr->usedSize > devicePtr->globalSize)
			{
				isMigrationNeeded = 1;
			}
		}
	}
	printf("local, isMigrationNeeded = %d\n", isMigrationNeeded);
	return isMigrationNeeded;
}

int voclCheckMigrationInWriteBuffer(cl_command_queue cmdQueue, size_t size)
{
	size_t cmdQueueMemSize;
	cl_command_queue clCmdQueue;
	VOCL_LIB_DEVICE *devicePtr;

	cmdQueueMemSize = voclCommandQueueGetDeviceMemorySize(cmdQueue);
	clCmdQueue = voclVOCLCommandQueue2CLCommandQueue(cmdQueue);
	devicePtr = voclLibGetDeviceIDFromCmdQueue(clCmdQueue);

	if (devicePtr->usedSize + cmdQueueMemSize + size > devicePtr->globalSize)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}
