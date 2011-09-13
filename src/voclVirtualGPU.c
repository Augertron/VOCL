#include <stdio.h>
#include <string.h>
#include <CL/opencl.h>
#include "voclStructures.h"

static vocl_gpu_str *voclVGPUPtr = NULL;

void voclAddVirtualGPU(int proxyRank, int proxyIndex, vocl_device_id deviceID)
{
	vocl_gpu_str *vgpuPtr;
	vgpuPtr = (vocl_gpu_str *)malloc(sizeof(vocl_gpu_str));

	vgpuPtr->proxyRank = proxyRank;
	vgpuPtr->proxyIndex = proxyIndex;
	vgpuPtr->deviceID = deviceID;

	vgpuPtr->contextNum = 20;
	vgpuPtr->contextNo = 0;
	vgpuPtr->contextPtr = (vocl_context_str **)malloc(sizeof(vocl_context *) * vgpuPtr->contextNum);
	memset(vgpuPtr->contextPtr, 0, sizeof(vocl_context_str*) * vgpuPtr->contextNum);

	vgpuPtr->cmdQueueNum = 50;
	vgpuPtr->cmdQueueNo = 0;
	vgpuPtr->cmdQueuePtr = (vocl_command_queue_str **)malloc(sizeof(vocl_command_queue_str*) * vgpuPtr->cmdQueueNum);
	memset(vgpuPtr->cmdQueuePtr, 0, sizeof(vocl_command_queue_str *) * vgpuPtr->cmdQueueNum);

	vgpuPtr->migStatus = 0;

	vgpuPtr->next = voclVGPUPtr;
	voclVGPUPtr = vgpuPtr;

	return;
}

void voclPrintVirtualGPUs()
{
    int i, j;
    vocl_gpu_str *vgpuPtr;
    printf("In voclPrintVirtualGPUs>>>>\n");
    
    i = 0;
    vgpuPtr = voclVGPUPtr;
    while (vgpuPtr != NULL)
    {
        printf("virtual GPU %d:\n", i++);
        printf("\tappIndex = %d, deviceID = %p\n", vgpuPtr->proxyIndex, vgpuPtr->deviceID);
        
        for (j = 0; j < vgpuPtr->contextNo; j++)
        {
            printf("\t\tcontext = %d\n", vgpuPtr->contextPtr[j]->voclContext);
        }

        for (j = 0; j < vgpuPtr->cmdQueueNo; j++)
        {
            printf("\t\tcommand queue = %d\n", vgpuPtr->cmdQueuePtr[j]->voclCommandQueue);
        }
        printf("\n");
        vgpuPtr = vgpuPtr->next;
    }

    return;
}

vocl_gpu_str *voclGetVirtualGPUPtr(int proxyIndex, vocl_device_id deviceID)
{
    vocl_gpu_str *vgpuPtr;

    vgpuPtr = voclVGPUPtr;
    while (vgpuPtr != NULL)
    {
        if (vgpuPtr->proxyIndex == proxyIndex && vgpuPtr->deviceID == deviceID)
        {
            break;
        }
        vgpuPtr = vgpuPtr->next;
    }

    if (vgpuPtr == NULL)
    {
        printf("voclGetVirtualGPUPtr, virtual GPU with proxyIndex %d and deviceID %p does not exist!\n",
                proxyIndex, deviceID);
        exit (1);
    }

    return vgpuPtr;
}

void voclSetVGPUMigStatus(int proxyIndex, vocl_device_id deviceID, char migStatus)
{
    vocl_gpu_str *vgpuPtr;
    vgpuPtr = voclGetVirtualGPUPtr(proxyIndex, deviceID);
    vgpuPtr->migStatus = migStatus;
}

char voclProxyUpdateVGPUMigStatus(int proxyIndex, vocl_device_id deviceID)
{
    vocl_gpu_str *vgpuPtr;
    vgpuPtr = voclGetVirtualGPUPtr(proxyIndex, deviceID);
    vgpuPtr->migStatus++;

    return vgpuPtr->migStatus;
}

char voclGetVGPUMigStatus(int proxyIndex, vocl_device_id deviceID)
{
    vocl_gpu_str *vgpuPtr;
    vgpuPtr = voclGetVirtualGPUPtr(proxyIndex, deviceID);
    return vgpuPtr->migStatus;
}

void voclReleaseVirtualGPU(int proxyIndex, vocl_device_id deviceID)
{
    vocl_gpu_str *vgpuPtr, *preVgpuPtr;

    vgpuPtr = voclVGPUPtr;
    if (vgpuPtr != NULL)
    {
        if (vgpuPtr->proxyIndex == proxyIndex && vgpuPtr->deviceID == deviceID)
        {
            voclVGPUPtr = vgpuPtr->next;
            free(vgpuPtr->contextPtr);
            free(vgpuPtr->cmdQueuePtr);
            free(vgpuPtr);

            return;
        }

        preVgpuPtr = vgpuPtr;
        vgpuPtr = vgpuPtr->next;
        while (vgpuPtr != NULL)
        {
            if (vgpuPtr->proxyIndex == proxyIndex && vgpuPtr->deviceID == deviceID)
            {
                break;
            }
            preVgpuPtr = vgpuPtr;
            vgpuPtr = vgpuPtr->next;
        }
    }

    if (vgpuPtr == NULL)
    {
        printf("voclReleaseVirtualGPU, virtual GPU with proxyIndex %d and deviceID %p does not exist!\n",
                proxyIndex, deviceID);
        exit(1);
    }

    preVgpuPtr->next = vgpuPtr->next;
    free(vgpuPtr->contextPtr);
    free(vgpuPtr->cmdQueuePtr);
    free(vgpuPtr);

    return;
}

void voclReleaseAllVirtualGPU()
{
    vocl_gpu_str *vgpuPtr, *nextVgpuPtr;
    vgpuPtr = voclVGPUPtr;
    while (vgpuPtr != NULL)
    {
        nextVgpuPtr = vgpuPtr->next;
        free(vgpuPtr->contextPtr);
        free(vgpuPtr->cmdQueuePtr);
        free(vgpuPtr);
        vgpuPtr = nextVgpuPtr;
    }

    return;
}

void voclAddContextToVGPU(int proxyIndex, vocl_device_id deviceID, vocl_context_str *contextPtr)
{
	int i;
    vocl_gpu_str *vgpuPtr;
    vgpuPtr = voclGetVirtualGPUPtr(proxyIndex, deviceID);

	for (i = 0; i < vgpuPtr->contextNo; i++)
	{
		if (vgpuPtr->contextPtr[i] == contextPtr)
		{
			break;
		}
	}

	if (i == vgpuPtr->contextNo)
	{
		vgpuPtr->contextPtr[vgpuPtr->contextNo] = contextPtr;
		vgpuPtr->contextNo++;

		if (vgpuPtr->contextNo >= vgpuPtr->contextNum)
		{
			vgpuPtr->contextPtr = (vocl_context_str **)realloc(vgpuPtr->contextPtr, sizeof(vocl_context_str*) * vgpuPtr->contextNum * 2);
			memset(&vgpuPtr->contextPtr[vgpuPtr->contextNum], 0, vgpuPtr->contextNum * sizeof(vocl_context_str*));
			vgpuPtr->contextNum *= 2;
		}
	}

    return;
}

void voclRemoveContextFromVGPU(int proxyIndex, vocl_context_str *contextPtr)
{
    vocl_gpu_str *vgpuPtr;
    int i, j;
    int contextFound = 0;
    vgpuPtr = voclVGPUPtr;
    while (vgpuPtr != NULL)
    {
        for (i = 0; i < vgpuPtr->contextNo; i++)
        {
            if (vgpuPtr->contextPtr[i] == contextPtr)
            {
                contextFound = 1;
                break;
            }
        }

        if (i < vgpuPtr->contextNo)
        {
            for (j = i; j < vgpuPtr->contextNo - 1; j++)
            {
                vgpuPtr->contextPtr[j] = vgpuPtr->contextPtr[j+1];
            }
        }

        vgpuPtr = vgpuPtr->next;
    }

    /* no context found at all */
    if (contextFound == 0)
    {
        printf("voclRemoveContextFromVGPU, context %d from proxy %d does not exist!\n",
                contextPtr->voclContext, proxyIndex);
        exit(1);
    }

    return;
}

void voclAddCommandQueueToVGPU(int proxyIndex, vocl_device_id deviceID, vocl_command_queue_str *cmdQueuePtr)
{
	int i;
    vocl_gpu_str *vgpuPtr;
    vgpuPtr = voclGetVirtualGPUPtr(proxyIndex, deviceID);

	for (i = 0; i < vgpuPtr->cmdQueueNo; i++)
	{
		if (vgpuPtr->cmdQueuePtr[i] == cmdQueuePtr)
		{
			break;
		}
	}

	if (i == vgpuPtr->cmdQueueNo)
	{
		vgpuPtr->cmdQueuePtr[vgpuPtr->cmdQueueNo] = cmdQueuePtr;
		vgpuPtr->cmdQueueNo++;

		if (vgpuPtr->cmdQueueNo >= vgpuPtr->cmdQueueNum)
		{
			vgpuPtr->cmdQueuePtr = (vocl_command_queue_str **)realloc(vgpuPtr->cmdQueuePtr,
					sizeof(vocl_command_queue_str*) * vgpuPtr->cmdQueueNum * 2);
			memset(&vgpuPtr->cmdQueuePtr[vgpuPtr->cmdQueueNum], 0, 
				   vgpuPtr->cmdQueueNum * sizeof(vocl_command_queue_str *));
			vgpuPtr->cmdQueueNum *= 2;
		}
	}

    return;
}

void voclRemoveCommandQueueFromVGPU(int proxyIndex, vocl_command_queue_str *cmdQueuePtr)
{
    vocl_gpu_str *vgpuPtr;
    int i, j;
    int cmdQueueFound = 0;
    vgpuPtr = voclVGPUPtr;
    while (vgpuPtr != NULL)
    {
        for (i = 0; i < vgpuPtr->cmdQueueNo; i++)
        {
            if (vgpuPtr->cmdQueuePtr[i] == cmdQueuePtr)
            {
                cmdQueueFound = 1;
                break;
            }
        }

        if (i < vgpuPtr->cmdQueueNo)
        {
            for (j = i; j < vgpuPtr->cmdQueueNo - 1; j++)
            {
                vgpuPtr->cmdQueuePtr[j] = vgpuPtr->cmdQueuePtr[j+1];
            }
        }

        vgpuPtr = vgpuPtr->next;
    }

    /* no context found at all */
    if (cmdQueueFound == 0)
    {
        printf("voclRemoveCommandQueueFromVGPU, command queue %d from proxy %d does not exist!\n",
                cmdQueuePtr->voclCommandQueue, proxyIndex);
        exit(1);
    }

    return;
}

