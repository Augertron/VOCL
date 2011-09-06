#include <stdio.h>
#include <string.h>
#include "vocl_proxyStructures.h"

static vocl_proxy_context *voclProxyContextPtr = NULL;

void voclProxyAddContext(cl_context context, cl_uint deviceNum, cl_device_id *deviceIDs)
{
	vocl_proxy_context *contextPtr;
	contextPtr = (vocl_proxy_context *)malloc(sizeof(vocl_proxy_context));
	contextPtr->context = context;
	contextPtr->deviceNum = deviceNum;
	contextPtr->devices = NULL;
	if (deviceNum > 0)
	{
		contextPtr->devices = (cl_device_id *)malloc(sizeof(cl_device_id) * deviceNum);
		memcpy(contextPtr->devices, deviceIDs, sizeof(cl_device_id) * deviceNum);
	}

	contextPtr->cmdQueueNo = 0;
	contextPtr->cmdQueueNum = 50;
	contextPtr->cmdQueuePtr = (vocl_proxy_command_queue **)malloc(sizeof(vocl_proxy_command_queue *) * contextPtr->cmdQueueNum);
	memset(contextPtr->cmdQueuePtr, 0, sizeof(vocl_proxy_command_queue *) * contextPtr->cmdQueueNum);

	contextPtr->memNo = 0;
	contextPtr->memNum = 100;
	contextPtr->memPtr = (vocl_proxy_mem **)malloc(sizeof(vocl_proxy_mem *) * contextPtr->memNum);
	memset(contextPtr->memPtr, 0, sizeof(vocl_proxy_mem *) * contextPtr->memNum);

	contextPtr->programNo = 0;
	contextPtr->programNum = 50;
	contextPtr->programPtr = (vocl_proxy_program **)malloc(sizeof(vocl_proxy_program *) * contextPtr->programNum);
	memset(contextPtr->programPtr, 0, sizeof(vocl_proxy_program *) * contextPtr->programNum);

	contextPtr->migStatus = 0;

	contextPtr->next = voclProxyContextPtr;
	voclProxyContextPtr = contextPtr;

	return;
}

vocl_proxy_context *voclProxyGetContextPtr(cl_context context)
{
	vocl_proxy_context *contextPtr;
	contextPtr = voclProxyContextPtr;
	while (contextPtr != NULL)
	{
		if (contextPtr->context == context)
		{
			break;
		}
		contextPtr = contextPtr->next;
	}

	if (contextPtr == NULL)
	{
		printf("voclProxyGetContextPtr, context %p does not exist!\n", context);
		exit (1);
	}

	return contextPtr;
}

cl_device_id *voclProxyGetDeviceIDsFromContext(cl_context context, cl_uint *deviceNum)
{
	vocl_proxy_context *contextPtr;
	contextPtr = voclProxyGetContextPtr(context);
	
	*deviceNum = contextPtr->deviceNum;
	return contextPtr->devices;
}

void voclProxyReleaseContext(cl_context context)
{
	vocl_proxy_context *contextPtr, *preContextPtr;
	/* if the cmdQueue is in the first node */
	contextPtr = voclProxyContextPtr;
	if (contextPtr != NULL)
	{
		if (voclProxyContextPtr->context == context)
		{
			contextPtr = voclProxyContextPtr;
			voclProxyContextPtr = contextPtr->next;
			free(contextPtr->memPtr);
			free(contextPtr->cmdQueuePtr);
			free(contextPtr->programPtr);
			free(contextPtr->devices);
			free(contextPtr);
			return;
		}

		preContextPtr = voclProxyContextPtr;
		contextPtr = preContextPtr->next;
		while (contextPtr != NULL)
		{
			if (contextPtr->context == context)
			{
				break;
			}

			preContextPtr = contextPtr;
			contextPtr = contextPtr->next;
		}
	}

	if (contextPtr == NULL)
	{
		printf("voclProxyReleaseContxt, context %p does not exist!\n", context);
		exit (1);
	}

	preContextPtr->next = contextPtr->next;
	free(contextPtr->memPtr);
	free(contextPtr->cmdQueuePtr);
	free(contextPtr->programPtr);
	free(contextPtr->devices);
	free(contextPtr);

	return;
}

void voclProxyReleaseAllContexts()
{
	vocl_proxy_context *contextPtr, *nextContextPtr;

	contextPtr = voclProxyContextPtr;
	while (contextPtr != NULL)
	{
		nextContextPtr = contextPtr->next;
		free(contextPtr->memPtr);
		free(contextPtr->cmdQueuePtr);
		free(contextPtr->programPtr);
		free(contextPtr->devices);
		free(contextPtr);
		contextPtr = nextContextPtr;
	}

	voclProxyContextPtr = NULL;

	return;
}

void voclProxyAddMemToContext(cl_context context, vocl_proxy_mem *mem)
{
	int i;
	vocl_proxy_context *contextPtr;
	contextPtr = voclProxyGetContextPtr(context);

	for (i = 0; i < contextPtr->memNo; i++)
	{
		if (contextPtr->memPtr[i] == mem)
		{
			break;
		}
	}

	if (i == contextPtr->memNo)
	{
		contextPtr->memPtr[contextPtr->memNo] = mem;
		contextPtr->memNo++;

		if (contextPtr->memNo >= contextPtr->memNum)
		{
			contextPtr->memPtr = (vocl_proxy_mem **)realloc(contextPtr->memPtr, sizeof(vocl_proxy_mem *) * contextPtr->memNum * 2);
			memset(&contextPtr->memPtr[contextPtr->memNum], 0, sizeof(vocl_proxy_mem*) * contextPtr->memNum);
			contextPtr->memNum *= 2;
		}
	}

	return;
}

void voclProxyRemoveMemFromContext(vocl_proxy_mem *mem)
{
	int i, j;
	int memFound = 0;
	vocl_proxy_context *contextPtr;

	contextPtr = voclProxyContextPtr;
	while (contextPtr != NULL)
	{
		for (i = 0; i < contextPtr->memNo; i++)
		{
			if (contextPtr->memPtr[i] == mem)
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
		printf("voclProxyRemoveMemFromContext, mem %p does not exist!\n", mem->mem);
		exit (1);
	}

	return;
}

void voclProxyRemoveMemFromContextSimple(cl_context context, vocl_proxy_mem *mem)
{
	int i, j;
	vocl_proxy_context *contextPtr;
	contextPtr = voclProxyGetContextPtr(context);

	for (i = 0; i < contextPtr->memNo; i++)
	{
		if (contextPtr->memPtr[i] == mem)
		{
			break;
		}
	}

	if (i == contextPtr->memNo)
	{
		printf("voclProxyRemoveMemFromContext, mem %p does not exist!\n", mem->mem);
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



void voclProxyAddProgramToContext(cl_context context, vocl_proxy_program *program)
{
	int i;
	vocl_proxy_context *contextPtr;
	contextPtr = voclProxyGetContextPtr(context);

	for (i = 0; i < contextPtr->programNo; i++)
	{
		if (contextPtr->programPtr[i] == program)
		{
			break;
		}
	}

	if (i == contextPtr->programNo)
	{
		contextPtr->programPtr[i] = program;
		contextPtr->programNo++;

		/* check whether memptr buffer is enough */
		if (contextPtr->programNo >= contextPtr->programNum)
		{
			contextPtr->programPtr = (vocl_proxy_program **)realloc(contextPtr->programPtr, sizeof(vocl_proxy_program *) * contextPtr->programNum * 2);
			memset(&contextPtr->programPtr[contextPtr->programNum], 0, sizeof(vocl_proxy_program *) * contextPtr->programNum);
			contextPtr->programNum *= 2;
		}
	}

	return;
}

void voclProxyRemoveProgramFromContext(vocl_proxy_program *program)
{
	int i, j;
	int programFound = 0;
	vocl_proxy_context *contextPtr;

	contextPtr = voclProxyContextPtr;
	while (contextPtr != NULL)
	{
		for (i = 0; i < contextPtr->programNo; i++)
		{
			if (contextPtr->programPtr[i] == program)
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
		printf("voclProxyRemoveProgramFromContext, cl_program %p does not exist!\n", program->program);
		exit(1);
	}

	return;
}

void voclProxyRemoveProgramFromContextSimple(cl_context context, vocl_proxy_program *program)
{
	int i, j;
	vocl_proxy_context *contextPtr;
	contextPtr = voclProxyGetContextPtr(context);

	for (i = 0; i < contextPtr->programNo; i++)
	{
		if (contextPtr->programPtr[i] == program)
		{
			break;
		}
	}

	if (i == contextPtr->programNo)
	{
		printf("voclProxyRemoveProgramFromContext, cl_program %p does not exist!\n", program->program);
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

void voclProxyAddCommandQueueToContext(cl_context context, vocl_proxy_command_queue *command_queue)
{
	int i;
	vocl_proxy_context *contextPtr;
	contextPtr = voclProxyGetContextPtr(context);

	for (i = 0; i < contextPtr->cmdQueueNo; i++)
	{
		if (contextPtr->cmdQueuePtr[i] == command_queue)
		{
			break;
		}
	}

	if (i == contextPtr->cmdQueueNo)
	{
		contextPtr->cmdQueuePtr[i] = command_queue;
		contextPtr->cmdQueueNo++;

		/* check whether memptr buffer is enough */
		if (contextPtr->cmdQueueNo >= contextPtr->cmdQueueNum)
		{
			contextPtr->cmdQueuePtr = (vocl_proxy_command_queue **)realloc(contextPtr->cmdQueuePtr, 
					sizeof(vocl_proxy_command_queue *) * contextPtr->cmdQueueNum * 2);
			memset(&contextPtr->cmdQueuePtr[contextPtr->cmdQueueNum], 0, 
					sizeof(vocl_proxy_command_queue *) * contextPtr->cmdQueueNum);
			contextPtr->cmdQueueNum *= 2;
		}
	}

	return;
}

void voclProxyRemoveCommandQueueFromContext(vocl_proxy_command_queue *command_queue)
{
	int i, j;
	int cmdQueueFound = 0;
	vocl_proxy_context *contextPtr;
	contextPtr = voclProxyContextPtr;
	while (contextPtr != NULL)
	{
		for (i = 0; i < contextPtr->cmdQueueNo; i++)
		{
			if (contextPtr->cmdQueuePtr[i] == command_queue)
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
		printf("voclProxyRemoveCommandQueueFromContext, command queue %p does not exist!\n", command_queue->command_queue);
		exit(1);
	}

	return;
}

void voclProxyRemoveCommandQueueFromContextSimple(cl_context context, vocl_proxy_command_queue *command_queue)
{
	int i, j;
	vocl_proxy_context *contextPtr;
	contextPtr = voclProxyGetContextPtr(context);

	for (i = 0; i < contextPtr->cmdQueueNo; i++)
	{
		if (contextPtr->cmdQueuePtr[i] == command_queue)
		{
			break;
		}
	}

	if (i == contextPtr->cmdQueueNo)
	{
		printf("voclProxyRemoveCommandQueueFromContext, command queue %p does not exist!\n", command_queue->command_queue);
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

void voclProxySetContextMigStatus(cl_context context, char migStatus)
{
	vocl_proxy_context *contextPtr;
	contextPtr = voclProxyGetContextPtr(context);
	contextPtr->migStatus = migStatus;
	return;
}

/* add the migration status by one and return the new migration status */
char voclProxyUpdateContextMigStatus(cl_context context)
{
	vocl_proxy_context *contextPtr;
	contextPtr = voclProxyGetContextPtr(context);
	contextPtr->migStatus++;

	return contextPtr->migStatus;
}

char voclProxyGetContextMigStatus(cl_context context)
{
	vocl_proxy_context *contextPtr;
	contextPtr = voclProxyGetContextPtr(context);
	return contextPtr->migStatus;
}

