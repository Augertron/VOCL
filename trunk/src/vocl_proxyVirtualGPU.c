#include <stdio.h>
#include <string.h>
#include "vocl_proxyStructures.h"

static vocl_virtual_gpu *virtualGPUPtr = NULL;

void voclProxyAddVirtualGPU(int appIndex, cl_device_id deviceID)
{
	vocl_virtual_gpu *vgpuPtr;
	vgpuPtr = (vocl_virtual_gpu *)malloc(sizeof(vocl_virtual_gpu));
	vgpuPtr->appIndex = appIndex;
	vgpuPtr->deviceID = deviceID;

	vgpuPtr->contextNum = 100;
	vgpuPtr->contextNo = 0;
	vgpuPtr->contexts = (cl_context *)malloc(sizeof(cl_context) * vgpuPtr->contextNum);

	vgpuPtr->cmdQueueNum = 100;
	vgpuPtr->cmdQueueNo = 0;
	vgpuPtr->cmdQueues = (cl_command_queue *)malloc(sizeof(cl_command_queue) * vgpuPtr->cmdQueueNum);

	vgpuPtr->next = virtualGPUPtr;
	virtualGPUPtr = vgpuPtr;

	return;
}

void voclProxyPrintVirtualGPUs()
{
	int i, j;
	vocl_virtual_gpu *vgpuPtr;
	printf("In voclProxyPrintVirtualGPUs>>>>\n");
	
	i = 0;
	vgpuPtr = virtualGPUPtr;
	while (vgpuPtr != NULL)
	{
		printf("virtual GPU %d:\n", i++);
		printf("\tappIndex = %d, deviceID = %p\n", vgpuPtr->appIndex, vgpuPtr->deviceID);
		
		for (j = 0; j < vgpuPtr->contextNo; j++)
		{
			printf("\t\tcontext = %p\n", vgpuPtr->contexts[j]);
		}

		for (j = 0; j < vgpuPtr->cmdQueueNo; j++)
		{
			printf("\t\tcommand queue = %p\n", vgpuPtr->cmdQueues[j]);
		}
		printf("\n");
		vgpuPtr = vgpuPtr->next;
	}

	return;
}

vocl_virtual_gpu *voclProxyGetVirtualGPUPtr(int appIndex, cl_device_id deviceID)
{
	vocl_virtual_gpu *vgpuPtr;
	
	vgpuPtr = virtualGPUPtr;
	while (vgpuPtr != NULL)
	{
		if (vgpuPtr->appIndex == appIndex && vgpuPtr->deviceID == deviceID)
		{
			break;
		}
		vgpuPtr = vgpuPtr->next;
	}

	if (vgpuPtr == NULL)
	{
		printf("voclProxyGetVirtualGPUPtr, virtual GPU with appRank %d and deviceID %p does not exist!\n",
				appIndex, deviceID);
		exit (1);
	}

	return vgpuPtr;
}

void voclProxyRemoveVirtualGPU(int appIndex, cl_device_id deviceID)
{
	vocl_virtual_gpu *vgpuPtr, *preVgpuPtr;

	vgpuPtr = virtualGPUPtr;
	if (vgpuPtr != NULL)
	{
		if (vgpuPtr->appIndex == appIndex && vgpuPtr->deviceID == deviceID)
		{
			virtualGPUPtr = vgpuPtr->next;
			free(vgpuPtr->contexts);
			free(vgpuPtr->cmdQueues);
			free(vgpuPtr);

			return;
		}
		
		preVgpuPtr = vgpuPtr;
		vgpuPtr = vgpuPtr->next;
		while (vgpuPtr != NULL)
		{
			if (vgpuPtr->appIndex == appIndex && vgpuPtr->deviceID == deviceID)
			{
				break;
			}
			preVgpuPtr = vgpuPtr;
			vgpuPtr = vgpuPtr->next;
		}
	}

	if (vgpuPtr == NULL)
	{
		printf("voclProxyRemoveVirtualGPU, virtual GPU with appRank %d and deviceID %p does not exist!\n",
				appIndex, deviceID);
		exit(1);
	}

	preVgpuPtr->next = vgpuPtr->next;
	free(vgpuPtr->contexts);
	free(vgpuPtr->cmdQueues);
	free(vgpuPtr);

	return;
}

void voclProxyReleaseAllVirtualGPU()
{
	vocl_virtual_gpu *vgpuPtr, *nextVgpuPtr;
	vgpuPtr = virtualGPUPtr;
	while (vgpuPtr != NULL)
	{
		nextVgpuPtr = vgpuPtr->next;
		free(vgpuPtr->contexts);
		free(vgpuPtr->cmdQueues);
		free(vgpuPtr);
		vgpuPtr = nextVgpuPtr;
	}

	return;
}

void voclProxyAddContextToVGPU(int appIndex, cl_device_id deviceID, cl_context context)
{
	vocl_virtual_gpu *vgpuPtr;
	vgpuPtr = voclProxyGetVirtualGPUPtr(appIndex, deviceID);
	if (vgpuPtr->contextNo >= vgpuPtr->contextNum)
	{
		vgpuPtr->contextNum *= 2;
		vgpuPtr->contexts = (cl_context *)realloc(vgpuPtr->contexts, sizeof(cl_context) * vgpuPtr->contextNum);
	}
	vgpuPtr->contexts[vgpuPtr->contextNo] = context;
	vgpuPtr->contextNo++;

	return;
}

void voclProxyRemoveContextFromVGPU(int appIndex, cl_context context)
{
	vocl_virtual_gpu *vgpuPtr;
	int i, j;
	int contextFound = 0;
	vgpuPtr = virtualGPUPtr;
	while (vgpuPtr != NULL)
	{
		for (i = 0; i < vgpuPtr->contextNo; i++)
		{
			if (vgpuPtr->contexts[i] == context)
			{
				contextFound = 1;
				break;
			}
		}

		if (i < vgpuPtr->contextNo)
		{
			for (j = i; j < vgpuPtr->contextNo - 1; j++)
			{
				vgpuPtr->contexts[j] = vgpuPtr->contexts[j+1];
			}
		}
		
		vgpuPtr = vgpuPtr->next;
	}

	/* no context found at all */
	if (contextFound == 0)
	{
		printf("voclProxyRemoveContextFromVGPU, context %p from app %d does not exist!\n",
				context, appIndex);
		exit(1);
	}

	return;
}

void voclProxyAddCommandQueueToVGPU(int appIndex, cl_device_id deviceID, cl_command_queue command_queue)
{
	vocl_virtual_gpu *vgpuPtr;
	vgpuPtr = voclProxyGetVirtualGPUPtr(appIndex, deviceID);
	if (vgpuPtr->cmdQueueNo >= vgpuPtr->cmdQueueNum)
	{
		vgpuPtr->cmdQueueNum *= 2;
		vgpuPtr->cmdQueues = (cl_command_queue *)realloc(vgpuPtr->cmdQueues, sizeof(cl_command_queue) * vgpuPtr->cmdQueueNum);
	}
	vgpuPtr->cmdQueues[vgpuPtr->cmdQueueNo] = command_queue;
	vgpuPtr->cmdQueueNo++;

	return;
}

void voclProxyRemoveCommandQueueFromVGPU(int appIndex, cl_command_queue command_queue)
{
	vocl_virtual_gpu *vgpuPtr;
	int i, j;
	int cmdQueueFound = 0;
	vgpuPtr = virtualGPUPtr;
	while (vgpuPtr != NULL)
	{
		for (i = 0; i < vgpuPtr->cmdQueueNo; i++)
		{
			if (vgpuPtr->cmdQueues[i] == command_queue)
			{
				cmdQueueFound = 1;
				break;
			}
		}

		if (i < vgpuPtr->cmdQueueNo)
		{
			for (j = i; j < vgpuPtr->cmdQueueNo - 1; j++)
			{
				vgpuPtr->cmdQueues[j] = vgpuPtr->cmdQueues[j+1];
			}
		}
		
		vgpuPtr = vgpuPtr->next;
	}

	/* no context found at all */
	if (cmdQueueFound == 0)
	{
		printf("voclProxyRemoveCommandQueueFromVGPU, command queue %p from app %d does not exist!\n",
				command_queue, appIndex);
		exit(1);
	}

	return;
}


