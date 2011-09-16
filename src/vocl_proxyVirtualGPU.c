#include <stdio.h>
#include <string.h>
#include "vocl_proxy.h"
#include "vocl_proxyStructures.h"

static vocl_virtual_gpu *virtualGPUPtr = NULL;
static cl_uint voclProxyDeviceNum = 0;
static cl_device_id *voclProxyDeviceIDs = NULL;

void voclProxyAddVirtualGPU(int appIndex, int proxyRank, cl_device_id deviceID)
{
	vocl_virtual_gpu *vgpuPtr;
	vgpuPtr = (vocl_virtual_gpu *)malloc(sizeof(vocl_virtual_gpu));
	vgpuPtr->appIndex = appIndex;
	vgpuPtr->deviceID = deviceID;
	vgpuPtr->oldDeviceID = NULL;
	vgpuPtr->proxyRank = proxyRank;

	vgpuPtr->contextNum = 100;
	vgpuPtr->contextNo = 0;
	vgpuPtr->contextPtr = (vocl_proxy_context **)malloc(sizeof(vocl_proxy_context*) * vgpuPtr->contextNum);

	vgpuPtr->cmdQueueNum = 100;
	vgpuPtr->cmdQueueNo = 0;
	vgpuPtr->cmdQueuePtr = (vocl_proxy_command_queue **)malloc(sizeof(vocl_proxy_command_queue*) * vgpuPtr->cmdQueueNum);
	vgpuPtr->migStatus = 0;  /* 0 means no migration, 1: migration once, 2: migratoin twice, and so on ... */

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
			printf("\t\tcontext = %p\n", vgpuPtr->contextPtr[j]->context);
		}

		for (j = 0; j < vgpuPtr->cmdQueueNo; j++)
		{
			printf("\t\tcommand queue = %p\n", vgpuPtr->cmdQueuePtr[j]->command_queue);
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

void voclProxySetVGPUMigStatus(int appIndex, cl_device_id deviceID, char migStatus)
{
	vocl_virtual_gpu *vgpuPtr;
	vgpuPtr = voclProxyGetVirtualGPUPtr(appIndex, deviceID);
	vgpuPtr->migStatus = migStatus;

	return;
}

cl_device_id voclProxyNewVGPUDeviceID(int appIndex, cl_device_id oldDeviceID)
{
	vocl_virtual_gpu *vgpuPtr;
	
	vgpuPtr = virtualGPUPtr;
	while (vgpuPtr != NULL)
	{
		if (vgpuPtr->appIndex == appIndex && vgpuPtr->oldDeviceID == oldDeviceID)
		{
			break;
		}
		vgpuPtr = vgpuPtr->next;
	}

	if (vgpuPtr == NULL)
	{
		printf("voclProxyGetNewVGPUDeviceID, virtual GPU with appRank %d and old device id %p does not exist!\n",
				appIndex, oldDeviceID);
		exit (1);
	}

	return vgpuPtr->deviceID;

}

char voclProxyUpdateVGPUMigStatus(int appIndex, cl_device_id deviceID)
{
	vocl_virtual_gpu *vgpuPtr;
	vgpuPtr = voclProxyGetVirtualGPUPtr(appIndex, deviceID);
	vgpuPtr->migStatus++;

	return vgpuPtr->migStatus;
}

void voclProxyStoreVGPUOldDeviceID(int appIndex, cl_device_id deviceID, cl_device_id oldDeviceID)
{
    vocl_virtual_gpu *vgpuPtr;
    vgpuPtr = voclProxyGetVirtualGPUPtr(appIndex, deviceID);
    vgpuPtr->oldDeviceID = oldDeviceID;

	return;
}

char voclProxyGetVGPUMigStatus(int appIndex, cl_device_id deviceID)
{
	vocl_virtual_gpu *vgpuPtr;
	vgpuPtr = voclProxyGetVirtualGPUPtr(appIndex, deviceID);
	return vgpuPtr->migStatus;
}

void voclProxyGetDeviceIDs()
{
	cl_platform_id *platformIDs;
	cl_uint *deviceNums, platformNum, totalDeviceNum, i;
	cl_int retCode;
	retCode = clGetPlatformIDs(0, NULL, &platformNum);
	if (retCode != CL_SUCCESS)
	{
		printf("voclProxyGetDeviceIDs, error %d\n", retCode);
		exit(1);
	}

	platformIDs = (cl_platform_id *)malloc(sizeof(cl_platform_id) * platformNum);
	deviceNums = (cl_uint *)malloc(sizeof(cl_uint) * platformNum);
	retCode = clGetPlatformIDs(platformNum, platformIDs, NULL);
	if (retCode != CL_SUCCESS)
	{
		printf("voclProxyGetDeviceIDs, clGetPlatformIDs error %d\n", retCode);
		exit(1);
	}

	totalDeviceNum = 0;
	retCode = CL_SUCCESS;
	for (i = 0; i < platformNum; i++)
	{
		retCode |= clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceNums[i]);
		totalDeviceNum += deviceNums[i];
	}
	if (retCode != CL_SUCCESS)
	{
		printf("voclProxyGetDeviceIDs, clGetDeviceIDs error %d\n", retCode);
		exit(1);
	}

	voclProxyDeviceIDs = (cl_device_id *)malloc(sizeof(cl_device_id) * totalDeviceNum);
	voclProxyDeviceNum = totalDeviceNum;

	totalDeviceNum = 0;
	for (i = 0; i < platformNum; i++)
	{
		retCode |= clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_GPU, 
				deviceNums[i], &voclProxyDeviceIDs[totalDeviceNum], NULL);
		totalDeviceNum += deviceNums[i];
	}
	if (retCode != CL_SUCCESS)
	{
		printf("voclProxyGetDeviceIDs, clGetDeviceIDs error %d\n", retCode);
		exit(1);
	}

	free(platformIDs);
	free(deviceNums);

	return;
}

/* pack the message for migration of virtual GPU */
void voclProxyGetMessageSizeForVGPU(vocl_virtual_gpu *vgpuPtr, vocl_vgpu_msg *msgPtr)
{
    size_t msgSize;
    int i, j, k;
    vocl_proxy_context **contextPtr;
    vocl_proxy_command_queue **cmdQueuePtr;
    vocl_proxy_program **programPtr;
    vocl_proxy_mem **memPtr;
    vocl_proxy_kernel **kernelPtr;

    msgPtr->contextNum = vgpuPtr->contextNo;
    msgPtr->cmdQueueNum = 0;
    msgPtr->programNum = 0;
    msgPtr->memNum = 0;
    msgPtr->kernelNum = 0;

    msgSize = 0;
    contextPtr = vgpuPtr->contextPtr;
    for (i = 0; i < vgpuPtr->contextNo; i++)
    {
        /* calculate size of program */
        msgPtr->programNum += contextPtr[i]->programNo;
        programPtr = contextPtr[i]->programPtr;
        for (j = 0; j < contextPtr[i]->programNo; j++)
        {
            /* add space for various program info */
            msgSize += programPtr[j]->sourceSize;
            msgSize += programPtr[j]->stringNum * sizeof(size_t);
            msgSize += programPtr[j]->buildOptionLen;
            msgSize += programPtr[j]->deviceNum * sizeof(cl_device_id);

            msgPtr->kernelNum += programPtr[j]->kernelNo;
            kernelPtr = programPtr[j]->kernelPtr;
            /* add space for various program info */
            for (k = 0; k < programPtr[j]->kernelNo; k++)
            {
                msgSize += kernelPtr[k]->nameLen;
				/* add the size of kernel arg flag and kernel args */
				msgSize += kernelPtr[k]->argNum * sizeof(char);
				msgSize += kernelPtr[k]->argNum * sizeof(kernel_args);
            }
        }

        msgPtr->cmdQueueNum += contextPtr[i]->cmdQueueNo;
        msgPtr->memNum += contextPtr[i]->memNo;
    }
	
	msgSize += sizeof(vocl_virtual_gpu);
    msgSize += msgPtr->contextNum * sizeof(vocl_proxy_context);
    msgSize += msgPtr->programNum * sizeof(vocl_proxy_program);
    msgSize += msgPtr->kernelNum * sizeof(vocl_proxy_kernel);
    msgSize += msgPtr->cmdQueueNum * sizeof(vocl_proxy_command_queue);
    msgSize += msgPtr->memNum * sizeof(vocl_proxy_mem);

    msgPtr->size = msgSize;

    return;
}

/* pack all contents in the virtual GPU in a message */
//void voclProxyPackMessageForVGPU(int appIndex, cl_device_id deviceID, char *bufPtr)
void voclProxyPackMessageForVGPU(vocl_virtual_gpu *vgpuPtr, char *bufPtr)
{
    size_t offset;
    int i, j, k;
//  vocl_virtual_gpu *vgpuPtr;
    vocl_proxy_context **contextPtr;
    vocl_proxy_command_queue **cmdQueuePtr;
    vocl_proxy_program **programPtr;
    vocl_proxy_kernel **kernelPtr;
    vocl_proxy_mem **memPtr;

    offset = 0;
	memcpy(bufPtr+offset, vgpuPtr, sizeof(vocl_virtual_gpu));
	offset += sizeof(vocl_virtual_gpu);

    contextPtr = vgpuPtr->contextPtr;
    for (i = 0; i < vgpuPtr->contextNo; i++)
    {
        /* pack the context structure */
        memcpy(bufPtr+offset, contextPtr[i], sizeof(vocl_proxy_context));
        offset += sizeof(vocl_proxy_context);

        /*pack the program based on the context */
        programPtr = contextPtr[i]->programPtr;
        for (j = 0; j < contextPtr[i]->programNo; j++)
        {
            memcpy(bufPtr+offset, programPtr[j], sizeof(vocl_proxy_program));
            offset += sizeof(vocl_proxy_program);

            /* pack the program source in the message */
            memcpy(bufPtr+offset, programPtr[j]->sourceString, programPtr[j]->sourceSize);
            offset += programPtr[j]->sourceSize;

            /* pack the string info */
            memcpy(bufPtr+offset, programPtr[j]->stringSizeArray, sizeof(size_t) * programPtr[j]->stringNum);
            offset += sizeof(size_t) * programPtr[j]->stringNum;

            /* pack the build options */
            if (programPtr[j]->buildOptionLen > 0)
            {
                memcpy(bufPtr+offset, programPtr[j]->buildOptions, programPtr[j]->buildOptionLen);
                offset += programPtr[j]->buildOptionLen;
            }

            /* pack the devices corresponding to the program */
            if (programPtr[j]->deviceNum > 0)
            {
                memcpy(bufPtr+offset, programPtr[j]->device_list, sizeof(cl_device_id) * programPtr[j]->deviceNum);
                offset += sizeof(cl_device_id) * programPtr[j]->deviceNum;
            }

            kernelPtr = programPtr[j]->kernelPtr;
            for (k = 0; k < programPtr[j]->kernelNo; k++)
            {
                /* pack the kernel structure */
                memcpy(bufPtr+offset, kernelPtr[k], sizeof(vocl_proxy_kernel));
                offset += sizeof(vocl_proxy_kernel);

                /* pack the kernel name */
                memcpy(bufPtr+offset, kernelPtr[k]->kernelName, kernelPtr[k]->nameLen);
                offset += kernelPtr[k]->nameLen;

				/* pack the kernel arg flag */
				memcpy(bufPtr+offset, kernelPtr[k]->argFlag, kernelPtr[k]->argNum * sizeof(char));
				offset += kernelPtr[k]->argNum * sizeof(char);

				/* pack the kernel argments */
				memcpy(bufPtr+offset, kernelPtr[k]->args, kernelPtr[k]->argNum * sizeof(kernel_args));
				offset += kernelPtr[k]->argNum * sizeof(kernel_args);
            }
        }

        /* pack the command queue based on the context */
        cmdQueuePtr = contextPtr[i]->cmdQueuePtr;
        for (j = 0; j < contextPtr[i]->cmdQueueNo; j++)
        {
            memcpy(bufPtr+offset, cmdQueuePtr[j], sizeof(vocl_proxy_command_queue));
            offset += sizeof(vocl_proxy_command_queue);
        }

        /* pack the mem based on the context */
        memPtr = contextPtr[i]->memPtr;
        for (j = 0; j < contextPtr[i]->memNo; j++)
        {
            memcpy(bufPtr+offset, memPtr[j], sizeof(vocl_proxy_mem));
            offset += sizeof(vocl_proxy_mem);
        }
    }

    return;
}

void voclProxyReleaseVirtualGPU(int appIndex, cl_device_id deviceID)
{
	vocl_virtual_gpu *vgpuPtr, *preVgpuPtr;

	vgpuPtr = virtualGPUPtr;
	if (vgpuPtr != NULL)
	{
		if (vgpuPtr->appIndex == appIndex && vgpuPtr->deviceID == deviceID)
		{
			virtualGPUPtr = vgpuPtr->next;
			free(vgpuPtr->contextPtr);
			free(vgpuPtr->cmdQueuePtr);
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
		printf("voclProxyReleaseVirtualGPU, virtual GPU with appRank %d and deviceID %p does not exist!\n",
				appIndex, deviceID);
		exit(1);
	}

	preVgpuPtr->next = vgpuPtr->next;
	free(vgpuPtr->contextPtr);
	free(vgpuPtr->cmdQueuePtr);
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
		free(vgpuPtr->contextPtr);
		free(vgpuPtr->cmdQueuePtr);
		free(vgpuPtr);
		vgpuPtr = nextVgpuPtr;
	}

	if (voclProxyDeviceIDs != NULL)
	{
		free(voclProxyDeviceIDs);
	}

	return;
}

void voclProxyAddContextToVGPU(int appIndex, cl_device_id deviceID, vocl_proxy_context *context)
{
	int i;
	vocl_virtual_gpu *vgpuPtr;
	vgpuPtr = voclProxyGetVirtualGPUPtr(appIndex, deviceID);

	for (i = 0; i < vgpuPtr->contextNo; i++)
	{
		if (vgpuPtr->contextPtr[i] == context)
		{
			break;
		}
	}

	if (i == vgpuPtr->contextNo)
	{
		vgpuPtr->contextPtr[vgpuPtr->contextNo] = context;
		vgpuPtr->contextNo++;

		if (vgpuPtr->contextNo >= vgpuPtr->contextNum)
		{
			vgpuPtr->contextNum *= 2;
			vgpuPtr->contextPtr = (vocl_proxy_context **)realloc(vgpuPtr->contextPtr, sizeof(vocl_proxy_context*) * vgpuPtr->contextNum);
		}
	}

	return;
}

void voclProxyRemoveContextFromVGPU(int appIndex, vocl_proxy_context *context)
{
	vocl_virtual_gpu *vgpuPtr;
	int i, j;
	int contextFound = 0;
	vgpuPtr = virtualGPUPtr;
	while (vgpuPtr != NULL)
	{
		for (i = 0; i < vgpuPtr->contextNo; i++)
		{
			if (vgpuPtr->contextPtr[i] == context)
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
		printf("voclProxyRemoveContextFromVGPU, context %p from app %d does not exist!\n",
				context->context, appIndex);
		exit(1);
	}

	return;
}

void voclProxyAddCommandQueueToVGPU(int appIndex, cl_device_id deviceID, vocl_proxy_command_queue *command_queue)
{
	int i;
	vocl_virtual_gpu *vgpuPtr;
	vgpuPtr = voclProxyGetVirtualGPUPtr(appIndex, deviceID);

	for (i = 0; i < vgpuPtr->cmdQueueNo; i++)
	{
		if (vgpuPtr->cmdQueuePtr[i] == command_queue)
		{
			break;
		}
	}

	if (i == vgpuPtr->cmdQueueNo)
	{
		vgpuPtr->cmdQueuePtr[vgpuPtr->cmdQueueNo] = command_queue;
		vgpuPtr->cmdQueueNo++;

		if (vgpuPtr->cmdQueueNo >= vgpuPtr->cmdQueueNum)
		{
			vgpuPtr->cmdQueueNum *= 2;
			vgpuPtr->cmdQueuePtr = (vocl_proxy_command_queue **)realloc(vgpuPtr->cmdQueuePtr, 
					sizeof(vocl_proxy_command_queue*) * vgpuPtr->cmdQueueNum);
		}
	}

	return;
}

void voclProxyRemoveCommandQueueFromVGPU(int appIndex, vocl_proxy_command_queue *command_queue)
{
	vocl_virtual_gpu *vgpuPtr;
	int i, j;
	int cmdQueueFound = 0;
	vgpuPtr = virtualGPUPtr;
	while (vgpuPtr != NULL)
	{
		for (i = 0; i < vgpuPtr->cmdQueueNo; i++)
		{
			if (vgpuPtr->cmdQueuePtr[i] == command_queue)
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
		printf("voclProxyRemoveCommandQueueFromVGPU, command queue %p from app %d does not exist!\n",
				command_queue->command_queue, appIndex);
		exit(1);
	}

	return;
}

void vocl_proxyGetKernelNumsOnGPUs(struct strKernelNumOnDevice *gpuKernelNum)
{
	/* go through each gpu and add the numbers of all kernels in the waiting state together */
	int i, j, k;
	int rankNo;
	vocl_virtual_gpu *vgpuPtr;
	vocl_proxy_context **contextPtr;
	vocl_proxy_command_queue **cmdQueuePtr;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rankNo);

	gpuKernelNum->deviceNum = voclProxyDeviceNum;
	for (i = 0; i < voclProxyDeviceNum; i++)
	{
		gpuKernelNum->kernelNums[i] = 0;
		gpuKernelNum->deviceIDs[i] = voclProxyDeviceIDs[i];

		vgpuPtr = virtualGPUPtr;
		while (vgpuPtr != NULL)
		{
			if (vgpuPtr->deviceID == gpuKernelNum->deviceIDs[i])
			{
				contextPtr = vgpuPtr->contextPtr;
				for (j = 0; j < vgpuPtr->contextNo; j++)
				{
					cmdQueuePtr = contextPtr[j]->cmdQueuePtr;
					for (k = 0; k < contextPtr[j]->cmdQueueNo; k++)
					{
						gpuKernelNum->kernelNums[i] += cmdQueuePtr[k]->kernelNumInCmdQueue;
					}
				}
			}

			vgpuPtr = vgpuPtr->next;
		}
	}

	gpuKernelNum->rankNo = rankNo;

	return;
}


