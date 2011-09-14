#include <stdio.h>
#include <string.h>
#include <CL/opencl.h>
#include "voclStructures.h"

extern vocl_device_id voclCLDeviceID2VOCLDeviceID(cl_device_id device, int proxyRank,
                                           int proxyIndex, MPI_Comm proxyComm,
                                           MPI_Comm proxyCommData);
extern void voclUpdateVOCLContext(vocl_context voclContext, cl_context newContext, int proxyRank,
                           int proxyIndex, MPI_Comm proxyComm, MPI_Comm proxyCommData);
extern void voclContextSetDevices(vocl_context context, cl_uint deviceNum, vocl_device_id *devices);
extern void voclContextSetMigrationStatus(vocl_context context, char status);

extern void voclUpdateVOCLProgram(vocl_program voclProgram, cl_program newProgram, int proxyRank, int proxyIndex,
                           MPI_Comm proxyComm, MPI_Comm proxyCommData);
extern void voclProgramSetMigrationStatus(vocl_program program, char status);

extern void voclUpdateVOCLKernel(vocl_kernel voclKernel, cl_kernel newKernel, int proxyRank,
                          int proxyIndex, MPI_Comm proxyComm, MPI_Comm proxyCommData);
extern void voclKernelSetMigrationStatus(vocl_kernel kernel, char status);

extern void voclUpdateVOCLCommandQueue(vocl_command_queue voclCmdQueue, cl_command_queue newCmdQueue,
                                int proxyRank, int proxyIndex, MPI_Comm comm, MPI_Comm commData);
extern void voclCommandQueueSetMigrationStatus(vocl_command_queue cmdQueue, char status);

extern void voclUpdateVOCLMemory(vocl_mem voclMemory, cl_mem newMem, int proxyRank, int proxyIndex,
                          MPI_Comm proxyComm, MPI_Comm proxyCommData);
extern void voclMemSetMigrationStatus(vocl_mem mem, char status);

extern void voclUpdateVOCLSampler(vocl_sampler voclSampler, cl_sampler newSampler, int proxyRank, int proxyIndex,
                           MPI_Comm proxyComm, MPI_Comm proxyCommData);
extern void voclSamplerSetMigrationStatus(vocl_sampler sampler, char status);

static vocl_gpu_str *voclVGPUPtr = NULL;
//static char *voclVGPUResourceBuffer = NULL;
//static size_t voclVGPUResourceSize = 0;

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

size_t voclGetVGPUMsgSize(int proxyIndex, vocl_device_id device)
{
	cl_uint i, j, k;
	vocl_gpu_str *vgpuPtr;
	vocl_context_str **contextPtr;
	vocl_program_str **programPtr;
	size_t voclBufSize, vgpuMsgSize;

	vgpuPtr = voclGetVirtualGPUPtr(proxyIndex, device);
	voclBufSize = 0;
	vgpuMsgSize = sizeof(vocl_mig_vgpu);

	voclBufSize += vgpuPtr->contextNo * sizeof(vocl_context);
	vgpuMsgSize += vgpuPtr->contextNo * sizeof(vocl_mig_context);

	contextPtr = vgpuPtr->contextPtr;
	for (i = 0; i < vgpuPtr->contextNo; i++)
	{
		/* size needed by program */
		voclBufSize += contextPtr[i]->programNo * sizeof(vocl_program);
		vgpuMsgSize += contextPtr[i]->programNo * sizeof(vocl_mig_program);

		/* size needed by kernels */
		programPtr = contextPtr[i]->programPtr;
		for (j = 0; j < contextPtr[i]->programNo; j++)
		{
			voclBufSize += programPtr[j]->kernelNo * sizeof(vocl_kernel);
			vgpuMsgSize += programPtr[j]->kernelNo * sizeof(vocl_mig_kernel);
		}

		/* size needed by command queue */
		voclBufSize += contextPtr[i]->cmdQueueNo * sizeof(vocl_command_queue);
		vgpuMsgSize += contextPtr[i]->cmdQueueNo * sizeof(vocl_mig_command_queue);

		/* size needed by memory */
		voclBufSize += contextPtr[i]->memNo * sizeof(vocl_mem);
		vgpuMsgSize += contextPtr[i]->memNo * sizeof(vocl_mig_mem);

		/* size needed by sampler */
		voclBufSize += contextPtr[i]->samplerNo * sizeof(vocl_sampler);
		vgpuMsgSize += contextPtr[i]->samplerNo * sizeof(vocl_mig_sampler);
	}

	return vgpuMsgSize;
}

void voclPackVGPUMsg(int proxyIndex, vocl_device_id device, 
					 int newProxyIndex, int newProxyRank,
					 MPI_Comm newComm, MPI_Comm newCommData,
					 char *msgBuf)
{
	cl_uint i, j, k;
	vocl_gpu_str *vgpuPtr;
	vocl_context_str **contextPtr;
	vocl_program_str **programPtr;
	vocl_kernel_str **kernelPtr;
	vocl_mem_str **memPtr;
	vocl_command_queue_str **cmdQueuePtr;
	vocl_sampler_str **samplerPtr;
	vocl_mig_context *ctxPtr;
	vocl_mig_program *pgPtr;
	vocl_mig_kernel *knPtr;
	vocl_mig_mem *mmPtr;
	vocl_mig_command_queue *cqPtr;
	vocl_mig_sampler *spPtr;
	vocl_mig_vgpu *vgPtr;
//	vocl_context *ctx;
//	vocl_program *pg;
//	vocl_kernel *kn;
//	vocl_mem *mm;
//	vocl_command_queue *cq;
//	vocl_sampler *sp;
//	size_t bufOffset, msgOffset;
	size_t msgOffset;

//	voclVGPUResourceBuffer = (char *)malloc(voclVGPUResourceSize);
	vgpuPtr = voclGetVirtualGPUPtr(proxyIndex, device);
	
//	bufOffset = 0;
	msgOffset = 0;

	/* pack the virtual gpu info */
	vgPtr = (vocl_mig_vgpu *)(msgBuf + msgOffset);
	msgOffset += sizeof(vocl_mig_vgpu);
	vgPtr->contextNo = vgpuPtr->contextNo;

	contextPtr = vgpuPtr->contextPtr;
	for (i = 0; i < vgpuPtr->contextNo; i++)
	{
		/* pack the context info for msg */
		ctxPtr = (vocl_mig_context *)(msgBuf + msgOffset);
		msgOffset += sizeof(vocl_mig_context);
		ctxPtr->context = contextPtr[i]->clContext;
		ctxPtr->migStatus = contextPtr[i]->migrationStatus;
		ctxPtr->programNo = contextPtr[i]->programNo;
		ctxPtr->cmdQueueNo = contextPtr[i]->cmdQueueNo;
		ctxPtr->memNo = contextPtr[i]->memNo;
		ctxPtr->samplerNo = contextPtr[i]->samplerNo;
		
		/* pack vocl context for update */
//		ctx = (vocl_context *)(voclVGPUResourceBuffer + bufOffset);
//		bufOffset += sizeof(vocl_context);
//		*ctx = contextPtr[i]->voclContext;

		/* pack the program info */
		programPtr = contextPtr[i]->programPtr;
		for (j = 0; j < contextPtr[i]->programNo; j++)
		{
			pgPtr = (vocl_mig_program *)(msgBuf + msgOffset);
			msgOffset += sizeof(vocl_mig_program);
			pgPtr->program = programPtr[j]->clProgram;
			pgPtr->migStatus = programPtr[j]->migrationStatus;
			pgPtr->kernelNo = programPtr[j]->kernelNo;

//			pg = (vocl_program *)(voclVGPUResourceBuffer + bufOffset);
//			bufOffset += sizeof(vocl_program);
//			*pg = programPtr[j]->voclProgram;

			/* pack the kernel info */
			kernelPtr = programPtr[j]->kernelPtr;
			for (k = 0; k < programPtr[j]->kernelNo; k++)
			{
				knPtr = (vocl_mig_kernel *)(msgBuf + msgOffset);
				msgOffset += sizeof(vocl_mig_kernel);

				knPtr->kernel = kernelPtr[k]->clKernel;
				knPtr->migStatus = kernelPtr[k]->migrationStatus;

//				kn = (vocl_kernel *)(voclVGPUResourceBuffer + bufOffset);
//				bufOffset += sizeof(vocl_kernel);
//				*kn = kernelPtr[k]->voclKernel;
			}
		}

		/* pack command queue info */
		cmdQueuePtr = contextPtr[i]->cmdQueuePtr;
		for (j = 0; j < contextPtr[i]->cmdQueueNo; j++)
		{
			cqPtr = (vocl_mig_command_queue *)(msgBuf + msgOffset);
			msgOffset += sizeof(vocl_mig_command_queue);
			cqPtr->command_queue = cmdQueuePtr[j]->clCommandQueue;
			cqPtr->migStatus = cmdQueuePtr[j]->migrationStatus;

//			cq = (vocl_command_queue *)(voclVGPUResourceBuffer + bufOffset);
//			bufOffset += sizeof(vocl_command_queue);
//			*cq = cmdQueuePtr[j]->voclCommandQueue;
		}
			
		/* pack mem info */
		memPtr = contextPtr[i]->memPtr;
		for (j = 0; j < contextPtr[i]->memNo; j++)
		{
			mmPtr = (vocl_mig_mem *)(msgBuf + msgOffset);
			msgOffset += sizeof(vocl_mig_mem);
			mmPtr->mem = memPtr[j]->clMemory;
			mmPtr->migStatus = memPtr[j]->migrationStatus;

//			mm = (vocl_mem *)(voclVGPUResourceBuffer + bufOffset);
//			bufOffset += sizeof(vocl_mem);
//			*mm = mmPtr[j]->voclMemory;
		}

		/* pack sampler info (not used currently) */
		samplerPtr = contextPtr[i]->samplerPtr;
		for (j = 0; j < contextPtr[i]->samplerNo; j++)
		{
			spPtr = (vocl_mig_sampler *)(msgBuf + msgOffset);
			msgOffset += sizeof(vocl_mig_sampler);
			spPtr->sampler = samplerPtr[j]->clSampler;
			spPtr->migStatus = samplerPtr[j]->migrationStatus;

//			sp = (vocl_sampler *)(voclVGPUResourceBuffer + bufOffset);
//			bufOffset += sizeof(vocl_sampler);
//			*sp = samplerPtr[j]->voclSampler;
		}
	}

	return;
}

void voclUpdateVirtualGPU(int proxyIndex, vocl_device_id device, 
						  int newProxyIndex, int newProxyRank,
						  MPI_Comm newProxyComm, MPI_Comm newProxyCommData,
						  char *msgBuf)
{
	cl_uint i, j, k;
	vocl_device_id voclDeviceID;
	vocl_gpu_str *vgpuPtr;
	vocl_context_str **contextPtr;
	vocl_program_str **programPtr;
	vocl_kernel_str **kernelPtr;
	vocl_mem_str **memPtr;
	vocl_command_queue_str **cmdQueuePtr;
	vocl_sampler_str **samplerPtr;
	vocl_mig_context *ctxPtr;
	vocl_mig_program *pgPtr;
	vocl_mig_kernel *knPtr;
	vocl_mig_mem *mmPtr;
	vocl_mig_command_queue *cqPtr;
	vocl_mig_sampler *spPtr;
	vocl_mig_vgpu *vgPtr;
//	vocl_context *ctx;
//	vocl_program *pg;
//	vocl_kernel *kn;
//	vocl_mem *mm;
//	vocl_command_queue *cq;
//	vocl_sampler *sp;
//	size_t bufOffset, msgOffset;
	size_t msgOffset;

	vgpuPtr = voclGetVirtualGPUPtr(proxyIndex, device);
	
	/* decode the received message */
//	bufOffset = 0;
	msgOffset = 0;
	vgPtr = (vocl_mig_vgpu *)(msgBuf + msgOffset);
	msgOffset += sizeof(vocl_mig_vgpu);

	/* create a new vocl device id */
	voclDeviceID = voclCLDeviceID2VOCLDeviceID(vgPtr->deviceID, newProxyRank,
                                           newProxyIndex, newProxyComm,
                                           newProxyCommData);

	/* update the virtual GPU info */
	vgpuPtr->proxyIndex = newProxyIndex;
	vgpuPtr->proxyRank = newProxyRank;
	vgpuPtr->deviceID = voclDeviceID;
	
	/* update context */
	contextPtr = vgpuPtr->contextPtr;
	for (i = 0; i < vgPtr->contextNo; i++)
	{
		ctxPtr = (vocl_mig_context *)(msgBuf + msgOffset);
		msgOffset += sizeof(vocl_mig_context);

		/* update context info */
		voclUpdateVOCLContext(contextPtr[i]->voclContext,
				ctxPtr->context, newProxyRank, newProxyIndex,
				newProxyComm, newProxyCommData);
		voclContextSetDevices(contextPtr[i]->voclContext,
				1, &voclDeviceID);
		voclContextSetMigrationStatus(contextPtr[i]->voclContext,
				ctxPtr->migStatus);

		/* update program info */
		programPtr = contextPtr[i]->programPtr;
		for (j = 0; j < contextPtr[i]->programNo; j++)
		{
			pgPtr = (vocl_mig_program *)(msgBuf + msgOffset);
			msgOffset += sizeof(vocl_mig_program);

			voclUpdateVOCLProgram(programPtr[j]->voclProgram, 
					pgPtr->program, newProxyRank, newProxyIndex,
					newProxyComm, newProxyCommData);
			voclProgramSetMigrationStatus(programPtr[j]->voclProgram,
					pgPtr->migStatus);

			/* update kernel info */
			kernelPtr = programPtr[j]->kernelPtr;
			for (k = 0; k < programPtr[j]->kernelNo; k++)
			{
				knPtr = (vocl_mig_kernel *)(msgBuf + msgOffset);
				msgOffset += sizeof(vocl_mig_kernel);

				voclUpdateVOCLKernel(kernelPtr[k]->voclKernel,
						knPtr->kernel, newProxyRank, newProxyIndex,
						newProxyComm, newProxyCommData);
				voclKernelSetMigrationStatus(kernelPtr[k]->voclKernel,
						knPtr->migStatus);
			}
		}

		/* update command queue info */
		cmdQueuePtr = contextPtr[i]->cmdQueuePtr;
		for (j = 0; j < contextPtr[i]->cmdQueueNo; j++)
		{
			cqPtr = (vocl_mig_command_queue *)(msgBuf + msgOffset);
			msgOffset += sizeof(vocl_mig_command_queue);

			voclUpdateVOCLCommandQueue(cmdQueuePtr[j]->voclCommandQueue,
					cqPtr->command_queue, newProxyRank, newProxyIndex,
					newProxyComm, newProxyCommData);
			voclCommandQueueSetMigrationStatus(cmdQueuePtr[j]->voclCommandQueue,
					cqPtr->migStatus);
		}

		/* update mem info */
		memPtr = contextPtr[i]->memPtr;
		for (j = 0; j < contextPtr[j]->memNo; j++)
		{
			mmPtr = (vocl_mig_mem *)(msgBuf + msgOffset);
			msgOffset += sizeof(vocl_mig_mem);

			voclUpdateVOCLMemory(memPtr[j]->voclMemory,
					mmPtr->mem, newProxyRank, newProxyIndex,
					newProxyComm, newProxyCommData);
			voclMemSetMigrationStatus(memPtr[j]->voclMemory,
					mmPtr->migStatus);
		}

		/* update sampler info */
		samplerPtr = contextPtr[i]->samplerPtr;
		for (j = 0; j < contextPtr[i]->samplerNo; j++)
		{
			spPtr = (vocl_mig_sampler *)(msgBuf + msgOffset);
			msgOffset += sizeof(vocl_mig_sampler);

			voclUpdateVOCLSampler(samplerPtr[j]->voclSampler,
					spPtr->sampler, newProxyRank, newProxyIndex,
					newProxyComm, newProxyCommData);
			voclSamplerSetMigrationStatus(samplerPtr[j]->voclSampler,
					spPtr->migStatus);
		}
	}
	
	return;
}
