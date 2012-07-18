#include <stdio.h>
#include <string.h>
#include "voclStructures.h"

static struct strVOCLContext *voclContextPtr = NULL;
static vocl_context voclContext;
static int voclContextNo;

static vocl_context getVOCLContextValue()
{
    vocl_context context = voclContext;
    voclContext++;

    return context;
}

static struct strVOCLContext *createVOCLContext()
{
    struct strVOCLContext *contextPtr;
    contextPtr = (struct strVOCLContext *) malloc(sizeof(struct strVOCLContext));
    contextPtr->next = voclContextPtr;
    voclContextPtr = contextPtr;

    return contextPtr;
}

struct strVOCLContext *voclGetContextPtr(vocl_context context)
{
    struct strVOCLContext *contextPtr;
    contextPtr = voclContextPtr;
    while (contextPtr != NULL) {
        if (contextPtr->voclContext == context) {
            break;
        }
        contextPtr = contextPtr->next;
    }

    if (contextPtr == NULL) {
        printf("Error, context does not exist!\n");
        exit(1);
    }

    return contextPtr;
}

void voclContextStoreDevices(vocl_context context, cl_uint deviceNum, vocl_device_id *devices)
{
	vocl_context_str *contextPtr;
	contextPtr = voclGetContextPtr(context);

	contextPtr->deviceNum = deviceNum;
	if (contextPtr->voclDevices != NULL)
	{
		free(contextPtr->voclDevices);
	}
	contextPtr->voclDevices = (vocl_device_id *)malloc(sizeof(vocl_device_id) * deviceNum);
	memcpy(contextPtr->voclDevices, devices, sizeof(vocl_device_id) * deviceNum);

	return;
}

vocl_device_id * voclContextGetDevices(vocl_context context, cl_uint *deviceNum)
{
	vocl_context_str *contextPtr;
	contextPtr = voclGetContextPtr(context);
	*deviceNum = contextPtr->deviceNum;

	return contextPtr->voclDevices;
}

void voclContextInitialize()
{
    voclContextPtr = NULL;
    voclContextNo = 0;
    voclContext = 0;
}

void voclContextFinalize()
{
    struct strVOCLContext *contextPtr, *tmpcontextPtr;
    contextPtr = voclContextPtr;
    while (contextPtr != NULL) {
        tmpcontextPtr = contextPtr->next;

		if (contextPtr->deviceNum > 0)
		{
			free(contextPtr->voclDevices);
		}
		free(contextPtr->cmdQueuePtr);
		free(contextPtr->memPtr);
		free(contextPtr->programPtr);
		free(contextPtr->samplerPtr);
        free(contextPtr);
        contextPtr = tmpcontextPtr;
    }

    voclContextPtr = NULL;
    voclContextNo = 0;
    voclContext = 0;
}

void voclContextSetMigrationStatus(vocl_context context, char status)
{
	struct strVOCLContext *contextPtr = voclGetContextPtr(context);
	contextPtr->migrationStatus = status;
	return;
}

char voclContextGetMigrationStatus(vocl_context context)
{
	struct strVOCLContext *contextPtr = voclGetContextPtr(context);
	return contextPtr->migrationStatus;
}

vocl_context voclCLContext2VOCLContext(cl_context context, int proxyRank,
                                       int proxyIndex, MPI_Comm proxyComm,
                                       MPI_Comm proxyCommData)
{
    struct strVOCLContext *contextPtr = createVOCLContext();
    contextPtr->clContext = context;
    contextPtr->proxyRank = proxyRank;
    contextPtr->proxyIndex = proxyIndex;
    contextPtr->proxyComm = proxyComm;
    contextPtr->proxyCommData = proxyCommData;
    contextPtr->voclContext = getVOCLContextValue();

	contextPtr->deviceNum = 0;
	contextPtr->voclDevices = NULL;

	contextPtr->cmdQueueNum = 20;
	contextPtr->cmdQueueNo = 0;
	contextPtr->cmdQueuePtr = (vocl_command_queue_str **)malloc(sizeof(vocl_command_queue_str *) * contextPtr->cmdQueueNum);
	memset(contextPtr->cmdQueuePtr, 0, sizeof(vocl_command_queue_str *) * contextPtr->cmdQueueNum);

	contextPtr->memNum = 100;
	contextPtr->memNo = 0;
	contextPtr->memPtr = (vocl_mem_str**)malloc(sizeof(vocl_mem_str*) * contextPtr->memNum);
	memset(contextPtr->memPtr, 0, sizeof(vocl_mem_str*) * contextPtr->memNum);

	contextPtr->programNum = 20;
	contextPtr->programNo = 0;
	contextPtr->programPtr = (vocl_program_str **)malloc(sizeof(vocl_program_str *) * contextPtr->programNum);
	memset(contextPtr->programPtr, 0, sizeof(vocl_program_str *) * contextPtr->programNum);

	contextPtr->samplerNum = 20;
	contextPtr->samplerNo = 0;
	contextPtr->samplerPtr = (vocl_sampler_str **)malloc(sizeof(vocl_sampler_str *) * contextPtr->samplerNum);
	memset(contextPtr->samplerPtr, 0, sizeof(vocl_sampler_str *) * contextPtr->samplerNum);

	contextPtr->migrationStatus = 0;

    return contextPtr->voclContext;
}

cl_context voclVOCLContext2CLContextComm(vocl_context context, int *proxyRank,
                                         int *proxyIndex, MPI_Comm * proxyComm,
                                         MPI_Comm * proxyCommData)
{
    struct strVOCLContext *contextPtr = voclGetContextPtr(context);
    *proxyRank = contextPtr->proxyRank;
    *proxyIndex = contextPtr->proxyIndex;
    *proxyComm = contextPtr->proxyComm;
    *proxyCommData = contextPtr->proxyCommData;

    return contextPtr->clContext;
}

cl_context voclVOCLContext2CLContext(vocl_context context)
{
    struct strVOCLContext *contextPtr = voclGetContextPtr(context);
    return contextPtr->clContext;
}

void voclUpdateVOCLContext(vocl_context voclContext, cl_context newContext, int proxyRank, 
						   int proxyIndex, MPI_Comm proxyComm, MPI_Comm proxyCommData)
{
    struct strVOCLContext *contextPtr = voclGetContextPtr(voclContext);
    int err;

    contextPtr->proxyRank = proxyRank;
    contextPtr->proxyIndex = proxyIndex;
    contextPtr->proxyComm = proxyComm;
    contextPtr->proxyCommData = proxyCommData;

    contextPtr->clContext = newContext;

    return;
}

int voclReleaseContext(vocl_context context)
{
    struct strVOCLContext *contextPtr, *preContextPtr, *curContextPtr;
    /* the first node in the link list */
    if (context == voclContextPtr->voclContext) {
        contextPtr = voclContextPtr;
        voclContextPtr = voclContextPtr->next;
		if (contextPtr->deviceNum > 0)
		{
			free(contextPtr->voclDevices);
		}
		free(contextPtr->cmdQueuePtr);
		free(contextPtr->memPtr);
		free(contextPtr->programPtr);
		free(contextPtr->samplerPtr);
        free(contextPtr);

        return 0;
    }

    contextPtr = NULL;
    preContextPtr = voclContextPtr;
    curContextPtr = voclContextPtr->next;
    while (curContextPtr != NULL) {
        if (context == curContextPtr->voclContext) {
            contextPtr = curContextPtr;
            break;
        }
        preContextPtr = curContextPtr;
        curContextPtr = curContextPtr->next;
    }

    if (contextPtr == NULL) {
        printf("context does not exist!\n");
        exit(1);
    }

    /* remote the current node from link list */
    preContextPtr->next = curContextPtr->next;
	if (curContextPtr->deviceNum > 0)
	{
		free(curContextPtr->voclDevices);
	}
	free(curContextPtr->cmdQueuePtr);
	free(curContextPtr->memPtr);
	free(curContextPtr->programPtr);
	free(curContextPtr->samplerPtr);
    free(curContextPtr);

    return 0;
}

void voclAddMemToContext(vocl_context context, vocl_mem_str *memPtr)
{
    int i;
    vocl_context_str *contextPtr;
    contextPtr = voclGetContextPtr(context);

    for (i = 0; i < contextPtr->memNo; i++)
    {
        if (contextPtr->memPtr[i] == memPtr)
        {
            break;
        }
    }

    if (i == contextPtr->memNo)
    {
        contextPtr->memPtr[contextPtr->memNo] = memPtr;
        contextPtr->memNo++;

        if (contextPtr->memNo >= contextPtr->memNum)
        {
            contextPtr->memPtr = (vocl_mem_str **)realloc(contextPtr->memPtr, sizeof(vocl_mem_str *) * contextPtr->memNum * 2);
            memset(&contextPtr->memPtr[contextPtr->memNum], 0, sizeof(vocl_mem_str*) * contextPtr->memNum);
            contextPtr->memNum *= 2;
        }
    }

    return;
}

void voclRemoveMemFromContext(vocl_mem_str *memPtr)
{
    int i, j;
    int memFound = 0;
    vocl_context_str *contextPtr;

    contextPtr = voclContextPtr;
    while (contextPtr != NULL)
    {
        for (i = 0; i < contextPtr->memNo; i++)
        {
            if (contextPtr->memPtr[i] == memPtr)
            {
                memFound = 1;
                break;
            }
        }

        if (i < contextPtr->memNo)
        {
            for (j = i; j < contextPtr->memNo - 1; j++)
            {
                contextPtr->memPtr[j] = contextPtr->memPtr[j+1];
            }
            contextPtr->memNo--;
        }

        contextPtr = contextPtr->next;
    }

    if (memFound == 0)
    {
        printf("voclProxyRemoveMemFromContext, vocl mem %p does not exist!\n", memPtr->voclMemory);
        exit (1);
    }

    return;
}

void voclRemoveMemFromContextSimple(vocl_context context, vocl_mem_str *memPtr)
{
    int i, j;
    vocl_context_str *contextPtr;
    contextPtr = voclGetContextPtr(context);

    for (i = 0; i < contextPtr->memNo; i++)
    {
        if (contextPtr->memPtr[i] == memPtr)
        {
            break;
        }
    }

    if (i == contextPtr->memNo)
    {
        printf("voclRemoveMemFromContext, mem %d does not exist!\n", memPtr->clMemory);
        exit(1);
    }
    else
    {
        for (j = i; j < contextPtr->memNo - 1; j++)
        {
            contextPtr->memPtr[j] = contextPtr->memPtr[j+1];
        }
        contextPtr->memNo--;
    }

    return;
}

void voclAddProgramToContext(vocl_context context, vocl_program_str *programPtr)
{
    int i;
    vocl_context_str *contextPtr;
    contextPtr = voclGetContextPtr(context);

    for (i = 0; i < contextPtr->programNo; i++)
    {
        if (contextPtr->programPtr[i] == programPtr)
        {
            break;
        }
    }

    if (i == contextPtr->programNo)
    {
        contextPtr->programPtr[i] = programPtr;
        contextPtr->programNo++;

        /* check whether memptr buffer is enough */
        if (contextPtr->programNo >= contextPtr->programNum)
        {
            contextPtr->programPtr = (vocl_program_str **)realloc(contextPtr->programPtr, sizeof(vocl_program_str *) * contextPtr->programNum * 2);
            memset(&contextPtr->programPtr[contextPtr->programNum], 0, sizeof(vocl_program_str *) * contextPtr->programNum);
            contextPtr->programNum *= 2;
        }
    }

    return;
}

void voclRemoveProgramFromContext(vocl_program_str *programPtr)
{
    int i, j;
    int programFound = 0;
    vocl_context_str *contextPtr;

    contextPtr = voclContextPtr;
    while (contextPtr != NULL)
    {
        for (i = 0; i < contextPtr->programNo; i++)
        {
            if (contextPtr->programPtr[i] == programPtr)
            {
                programFound = 1;
                break;
            }
        }

        if (i < contextPtr->programNo)
        {
            for (j = i; j < contextPtr->programNo - 1; j++)
            {
                contextPtr->programPtr[j] = contextPtr->programPtr[j+1];
            }
        }

        contextPtr = contextPtr->next;
    }

    if (programFound == 0)
    {
        printf("voclRemoveProgramFromContext, vocl program %d does not exist!\n", programPtr->voclProgram);
        exit(1);
    }

    return;
}

void voclRemoveProgramFromContextSimple(vocl_context context, vocl_program_str *programPtr)
{
    int i, j;
    vocl_context_str *contextPtr;
    contextPtr = voclGetContextPtr(context);

    for (i = 0; i < contextPtr->programNo; i++)
    {
        if (contextPtr->programPtr[i] == programPtr)
        {
            break;
        }
    }

    if (i == contextPtr->programNo)
    {
        printf("voclRemoveProgramFromContext, vocl_program %d does not exist!\n", programPtr->voclProgram);
        exit(1);
    }
    else
    {
        for (j = i; j < contextPtr->programNo - 1; j++)
        {
            contextPtr->programPtr[j] = contextPtr->programPtr[j+1];
        }
        contextPtr->programNo--;
    }

    return;
}

void voclAddCommandQueueToContext(vocl_context context, vocl_command_queue_str *cmdQueuePtr)
{
    int i;
    vocl_context_str *contextPtr;
    contextPtr = voclGetContextPtr(context);

    for (i = 0; i < contextPtr->cmdQueueNo; i++)
    {
        if (contextPtr->cmdQueuePtr[i] == cmdQueuePtr)
        {
            break;
        }
    }

    if (i == contextPtr->cmdQueueNo)
    {
        contextPtr->cmdQueuePtr[i] = cmdQueuePtr;
        contextPtr->cmdQueueNo++;

        /* check whether memptr buffer is enough */
        if (contextPtr->cmdQueueNo >= contextPtr->cmdQueueNum)
        {
            contextPtr->cmdQueuePtr = (vocl_command_queue_str **)realloc(contextPtr->cmdQueuePtr,
                    sizeof(vocl_command_queue_str *) * contextPtr->cmdQueueNum * 2);
            memset(&contextPtr->cmdQueuePtr[contextPtr->cmdQueueNum], 0,
                    sizeof(vocl_command_queue_str *) * contextPtr->cmdQueueNum);
            contextPtr->cmdQueueNum *= 2;
        }
    }

    return;
}

void voclRemoveCommandQueueFromContext(vocl_command_queue_str *cmdQueuePtr)
{
    int i, j;
    int cmdQueueFound = 0;
    vocl_context_str *contextPtr;
    contextPtr = voclContextPtr;
    while (contextPtr != NULL)
    {
        for (i = 0; i < contextPtr->cmdQueueNo; i++)
        {
            if (contextPtr->cmdQueuePtr[i] == cmdQueuePtr)
            {
                cmdQueueFound = 1;
                break;
            }
        }

        if (i < contextPtr->cmdQueueNo)
        {
            for (j = i; j < contextPtr->cmdQueueNo; j++)
            {
                contextPtr->cmdQueuePtr[j] = contextPtr->cmdQueuePtr[j+1];
            }
            contextPtr->cmdQueueNo--;
        }
        contextPtr = contextPtr->next;
    }

    if (cmdQueueFound == 0)
    {
        printf("voclRemoveCommandQueueFromContext, command queue %d does not exist!\n", cmdQueuePtr->voclCommandQueue);
        exit(1);
    }

    return;
}

void voclRemoveCommandQueueFromContextSimple(vocl_context context, vocl_command_queue_str *cmdQueuePtr)
{
    int i, j;
    vocl_context_str *contextPtr;
    contextPtr = voclGetContextPtr(context);

    for (i = 0; i < contextPtr->cmdQueueNo; i++)
    {
        if (contextPtr->cmdQueuePtr[i] == cmdQueuePtr)
        {
            break;
        }
    }

    if (i == contextPtr->cmdQueueNo)
    {
        printf("voclRemoveCommandQueueFromContext, command queue %d does not exist!\n", cmdQueuePtr->voclCommandQueue);
        exit(1);
    }
    else
    {
        for (j = i; j < contextPtr->cmdQueueNo - 1; j++)
        {
            contextPtr->cmdQueuePtr[j] = contextPtr->cmdQueuePtr[j+1];
        }
        contextPtr->cmdQueueNo--;
    }

    return;
}

void voclAddSamplerToContext(vocl_context context, vocl_sampler_str *samplerPtr)
{
    int i;
    vocl_context_str *contextPtr;
    contextPtr = voclGetContextPtr(context);

    for (i = 0; i < contextPtr->samplerNo; i++)
    {
        if (contextPtr->samplerPtr[i] == samplerPtr)
        {
            break;
        }
    }

    if (i == contextPtr->samplerNo)
    {
        contextPtr->samplerPtr[i] = samplerPtr;
        contextPtr->samplerNo++;

        /* check whether memptr buffer is enough */
        if (contextPtr->samplerNo >= contextPtr->samplerNum)
        {
            contextPtr->samplerPtr = (vocl_sampler_str **)realloc(contextPtr->samplerPtr, sizeof(vocl_sampler_str *) * contextPtr->samplerNum * 2);
            memset(&contextPtr->samplerPtr[contextPtr->samplerNum], 0, sizeof(vocl_sampler_str *) * contextPtr->samplerNum);
            contextPtr->samplerNum *= 2;
        }
    }

    return;
}

void voclRemoveSamplerFromContext(vocl_sampler_str *samplerPtr)
{
    int i, j;
    int samplerFound = 0;
    vocl_context_str *contextPtr;

    contextPtr = voclContextPtr;
    while (contextPtr != NULL)
    {
        for (i = 0; i < contextPtr->samplerNo; i++)
        {
            if (contextPtr->samplerPtr[i] == samplerPtr)
            {
                samplerFound = 1;
                break;
            }
        }

        if (i < contextPtr->samplerNo)
        {
            for (j = i; j < contextPtr->samplerNo - 1; j++)
            {
                contextPtr->samplerPtr[j] = contextPtr->samplerPtr[j+1];
            }
        }

        contextPtr = contextPtr->next;
    }

    if (samplerFound == 0)
    {
        printf("voclRemoveSamplerFromContext, vocl sampler %d does not exist!\n", samplerPtr->voclSampler);
        exit(1);
    }

    return;
}

void voclRemoveSamplerFromContextSimple(vocl_context context, vocl_sampler_str *samplerPtr)
{
    int i, j;
    vocl_context_str *contextPtr;
    contextPtr = voclGetContextPtr(context);

    for (i = 0; i < contextPtr->samplerNo; i++)
    {
        if (contextPtr->samplerPtr[i] == samplerPtr)
        {
            break;
        }
    }

    if (i == contextPtr->samplerNo)
    {
        printf("voclRemoveSamplerFromContext, vocl_sampler %d does not exist!\n", samplerPtr->voclSampler);
        exit(1);
    }
    else
    {
        for (j = i; j < contextPtr->samplerNo - 1; j++)
        {
            contextPtr->samplerPtr[j] = contextPtr->samplerPtr[j+1];
        }
        contextPtr->samplerNo--;
    }

    return;
}

