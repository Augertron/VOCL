#include <stdio.h>
#include <string.h>
#include "vocl_proxy.h"
#include "vocl_proxyStructures.h"

static str_vocl_proxy_command_queue *voclProxyCmdQueuePtr = NULL;

void voclProxyAddCmdQueue(cl_command_queue command_queue, cl_command_queue_properties properties, cl_context context, cl_device_id deviceID)
{
	str_vocl_proxy_command_queue *cmdQueuePtr;
	cmdQueuePtr = (str_vocl_proxy_command_queue *)malloc(sizeof(str_vocl_proxy_command_queue));
	cmdQueuePtr->command_queue = command_queue;
	cmdQueuePtr->properties = properties;
	cmdQueuePtr->context = context;
	cmdQueuePtr->deviceID = deviceID;

	cmdQueuePtr->memNo = 0;
	cmdQueuePtr->memNum = 100;
	cmdQueuePtr->memPtr = (str_vocl_proxy_mem **)malloc(sizeof(str_vocl_proxy_mem *) * cmdQueuePtr->memNum);
	memset(cmdQueuePtr->memPtr, 0, sizeof(str_vocl_proxy_mem *) * cmdQueuePtr->memNum);

	cmdQueuePtr->kernelNo = 0;
	cmdQueuePtr->kernelNum = 50;
	cmdQueuePtr->kernelPtr = (str_vocl_proxy_kernel **)malloc(sizeof(str_vocl_proxy_mem *) * cmdQueuePtr->kernelNum);
	memset(cmdQueuePtr->kernelPtr, 0, sizeof(str_vocl_proxy_kernel *) * cmdQueuePtr->kernelNum);

	cmdQueuePtr->kernelNumInCmdQueue = 0;

	cmdQueuePtr->next = voclProxyCmdQueuePtr;
	voclProxyCmdQueuePtr = cmdQueuePtr;

	return;
}

str_vocl_proxy_command_queue *voclProxyGetCmdQueuePtr(cl_command_queue command_queue)
{
	str_vocl_proxy_command_queue *cmdQueuePtr;
	cmdQueuePtr = voclProxyCmdQueuePtr;
	while (cmdQueuePtr != NULL)
	{
		if (cmdQueuePtr->command_queue == command_queue)
		{
			break;
		}
	}

	if (cmdQueuePtr == NULL)
	{
		printf("voclProxyGetCmdQueuePtr, command queue %p does not exist!\n", command_queue);
		exit (1);
	}

	return cmdQueuePtr;
}

void voclProxyReleaseCommandQueue(cl_command_queue command_queue)
{
	str_vocl_proxy_command_queue *cmdQueuePtr, *preCmdQueuePtr;
	/* if the cmdQueue is in the first node */
	cmdQueuePtr = voclProxyCmdQueuePtr;
	if (cmdQueuePtr != NULL)
	{
		if (voclProxyCmdQueuePtr->command_queue == command_queue)
		{
			cmdQueuePtr = voclProxyCmdQueuePtr;
			voclProxyCmdQueuePtr = cmdQueuePtr->next;
			free(cmdQueuePtr->memPtr);
			free(cmdQueuePtr->kernelPtr);
			free(cmdQueuePtr);
			return;
		}

		preCmdQueuePtr = voclProxyCmdQueuePtr;
		cmdQueuePtr = preCmdQueuePtr->next;
		while (cmdQueuePtr != NULL)
		{
			if (cmdQueuePtr->command_queue == command_queue)
			{
				break;
			}

			preCmdQueuePtr = cmdQueuePtr;
			cmdQueuePtr = cmdQueuePtr->next;
		}
	}

	if (cmdQueuePtr == NULL)
	{
		printf("voclProxyReleaseCommandQueue, command queue %p does not exist!\n", command_queue);
		exit (1);
	}

	preCmdQueuePtr->next = cmdQueuePtr->next;
	free(cmdQueuePtr->memPtr);
	free(cmdQueuePtr->kernelPtr);
	free(cmdQueuePtr);

	return;
}

void voclProxyReleaseAllCommandQueues()
{
	str_vocl_proxy_command_queue *cmdQueuePtr, *nextCmdQueuePtr;

	cmdQueuePtr = voclProxyCmdQueuePtr;
	while (cmdQueuePtr != NULL)
	{
		nextCmdQueuePtr = cmdQueuePtr->next;
		free(cmdQueuePtr->memPtr);
		free(cmdQueuePtr->kernelPtr);
		free(cmdQueuePtr);
		cmdQueuePtr = nextCmdQueuePtr;
	}

	voclProxyCmdQueuePtr = NULL;

	return;
}

void voclProxyAddMemToCmdQueue(cl_command_queue command_queue, str_vocl_proxy_mem *mem)
{
	int i;
	str_vocl_proxy_command_queue *cmdQueuePtr;
	cmdQueuePtr = voclProxyGetCmdQueuePtr(command_queue);

	for (i = 0; i < cmdQueuePtr->memNo; i++)
	{
		if (cmdQueuePtr->memPtr[i] == mem)
		{
			break;
		}
	}

	if (i == cmdQueuePtr->memNo)
	{
		cmdQueuePtr->memPtr[i] = mem;
		cmdQueuePtr->memNo++;

		/* check whether memptr buffer is enough */
		if (cmdQueuePtr->memNo == cmdQueuePtr->memNum)
		{
			cmdQueuePtr->memPtr = (str_vocl_proxy_mem **)realloc(cmdQueuePtr->memPtr, sizeof(str_vocl_proxy_mem *) * cmdQueuePtr->memNum * 2);
			memset(&cmdQueuePtr->memPtr[cmdQueuePtr->memNum], 0, sizeof(str_vocl_proxy_mem *) * cmdQueuePtr->memNum);
			cmdQueuePtr->memNum *= 2;
		}
	}

	return;
}

void voclProxyAddKernelToCmdQueue(cl_command_queue command_queue, str_vocl_proxy_kernel *kernel)
{
	int i;
	str_vocl_proxy_command_queue *cmdQueuePtr;
	cmdQueuePtr = voclProxyGetCmdQueuePtr(command_queue);

	for (i = 0; i < cmdQueuePtr->kernelNo; i++)
	{
		if (cmdQueuePtr->kernelPtr[i] == kernel)
		{
			break;
		}
	}

	if (i == cmdQueuePtr->kernelNo)
	{
		cmdQueuePtr->kernelPtr[i] = kernel;
		cmdQueuePtr->kernelNo++;

		/* check whether memptr buffer is enough */
		if (cmdQueuePtr->kernelNo == cmdQueuePtr->kernelNum)
		{
			cmdQueuePtr->kernelPtr = (str_vocl_proxy_kernel **)realloc(cmdQueuePtr->kernelPtr, sizeof(str_vocl_proxy_kernel *) * cmdQueuePtr->kernelNum * 2);
			memset(&cmdQueuePtr->memPtr[cmdQueuePtr->kernelNum], 0, sizeof(str_vocl_proxy_kernel *) * cmdQueuePtr->kernelNum);
			cmdQueuePtr->kernelNum *= 2;
		}
	}

	return;
}

/*increase the number of kernels in the command queue */
void voclProxyIncreaseKernelNumInCmdQueue(cl_command_queue command_queue, int kernelNum)
{
    str_vocl_proxy_command_queue *cmdQueuePtr;
    cmdQueuePtr = voclProxyGetCmdQueuePtr(command_queue);
    cmdQueuePtr->kernelNumInCmdQueue += kernelNum;
}

void voclProxyDecreaseKernelNumInCmdQueue(cl_command_queue command_queue, int kernelNum)
{
    str_vocl_proxy_command_queue *cmdQueuePtr;
    cmdQueuePtr = voclProxyGetCmdQueuePtr(command_queue);
    cmdQueuePtr->kernelNumInCmdQueue -= kernelNum;
}

void voclProxyResetKernelNumInCmdQueue(cl_command_queue command_queue)
{
    str_vocl_proxy_command_queue *cmdQueuePtr;
    cmdQueuePtr = voclProxyGetCmdQueuePtr(command_queue);
    cmdQueuePtr->kernelNumInCmdQueue = 0;
}

void voclProxyGetKernelNumsOnDevice(struct strKernelNumOnDevice *kernelNums)
{
	int newDeviceFound, i;
	str_vocl_proxy_command_queue *cmdQueuePtr, *tmpCmdQueuePtr;
	cl_device_id deviceID;

	/* obtain the devices used on the node */
	kernelNums->deviceNum = 0;
	cmdQueuePtr = voclProxyCmdQueuePtr;
	while (cmdQueuePtr != NULL)
	{
		newDeviceFound = 1;
		deviceID = cmdQueuePtr->deviceID;
		tmpCmdQueuePtr = voclProxyCmdQueuePtr;
		while (tmpCmdQueuePtr != cmdQueuePtr)
		{
			if (deviceID == tmpCmdQueuePtr->deviceID)
			{
				newDeviceFound = 0;
				break;
			}
			tmpCmdQueuePtr = tmpCmdQueuePtr->next;
		}

		if (newDeviceFound == 1)
		{
			kernelNums->deviceIDs[kernelNums->deviceNum] = deviceID;
			kernelNums->kernelNums[kernelNums->deviceNum] = 0;
			kernelNums->deviceNum++;
		}

		cmdQueuePtr = cmdQueuePtr->next;
	}

	/* get the kernels queued on each device */
	cmdQueuePtr = voclProxyCmdQueuePtr;
    while (cmdQueuePtr != NULL)
    {
		for (i = 0; i < kernelNums->deviceNum; i++)
		{
			if (kernelNums->deviceIDs[i] == cmdQueuePtr->deviceID)
			{
				break;
			}
		}
		kernelNums->kernelNums[i] += cmdQueuePtr->kernelNumInCmdQueue;
		cmdQueuePtr = cmdQueuePtr->next;
    }

    return;
}

