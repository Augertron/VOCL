#include <stdio.h>
#include <string.h>
#include "vocl_proxyStructures.h"

static vocl_proxy_mem *voclProxyMemPtr = NULL;

void voclProxyAddMem(cl_mem mem, cl_mem_flags flags, size_t size, cl_context context)
{
    vocl_proxy_mem *memPtr;

    memPtr = (vocl_proxy_mem *)malloc(sizeof(vocl_proxy_mem));
	memPtr->mem = mem;
	memPtr->oldMem = NULL;
	memPtr->size = size;
	memPtr->flags = flags;
	memPtr->context = context;
	memPtr->migStatus = 0;
	memPtr->isWritten = 0;

    memPtr->next = voclProxyMemPtr;
    voclProxyMemPtr = memPtr;

    return;
}

vocl_proxy_mem *voclProxyGetMemPtr(cl_mem mem)
{
    vocl_proxy_mem *memPtr;
    memPtr = voclProxyMemPtr;
    while (memPtr != NULL)
    {
        if (memPtr->mem == mem)
        {
            break;
        }

		memPtr = memPtr->next;
    }

    if (memPtr == NULL)
    {
        printf("voclProxyGetMemPtr, mem %p does not exist!\n", mem);
        exit (1);
    }

    return memPtr;
}

void voclProxySetMemMigStatus(cl_mem mem, char migStatus)
{
	vocl_proxy_mem *memPtr;
	memPtr = voclProxyGetMemPtr(mem);
	memPtr->migStatus = migStatus;

	return;
}

char voclProxyGetMemMigStatus(cl_mem mem)
{
	vocl_proxy_mem *memPtr;
	memPtr = voclProxyGetMemPtr(mem);
	return memPtr->migStatus;
}

void voclProxyStoreOldMemValue(cl_mem mem, cl_mem oldMem)
{
	vocl_proxy_mem *memPtr;
	memPtr = voclProxyGetMemPtr(mem);
	memPtr->oldMem = oldMem;

	return;
}

cl_mem voclProxyGetOldMemValue(cl_mem mem)
{
	vocl_proxy_mem *memPtr;
	memPtr = voclProxyGetMemPtr(mem);
	return memPtr->oldMem;
}

cl_mem voclProxyGetNewMemValue(cl_mem oldMem)
{
	vocl_proxy_mem *memPtr;
	memPtr = voclProxyMemPtr;
	while (memPtr != NULL)
	{
		if (memPtr->oldMem == oldMem)
		{
			break;
		}
		memPtr = memPtr->next;
	}

	if (memPtr == NULL)
	{
		printf("voclProxyGetNewMemValue, old mem %p does not exist!\n", oldMem);
		exit (1);
	}

	return memPtr->mem;
}

void voclProxySetMemWritten(cl_mem mem, int isWritten)
{
	vocl_proxy_mem *memPtr;
	memPtr = voclProxyGetMemPtr(mem);
	memPtr->isWritten = isWritten;

	return;
}

void voclProxySetMemWriteCmdQueue(cl_mem mem, cl_command_queue cmdQueue)
{
	vocl_proxy_mem *memPtr;
	memPtr = voclProxyGetMemPtr(mem);
	memPtr->cmdQueue = cmdQueue;
	
	return;
}

void voclProxyReleaseMem(cl_mem mem)
{
    vocl_proxy_mem *memPtr, *preMemPtr;

    /* if the cmdQueue is in the first node */
	memPtr = voclProxyMemPtr;

	if (memPtr != NULL)
	{
		if (memPtr->mem == mem)
		{
			memPtr = voclProxyMemPtr;
			voclProxyMemPtr = memPtr->next;
			free(memPtr);
			return;
		}

		preMemPtr = voclProxyMemPtr;
		memPtr = preMemPtr->next;
		while (memPtr != NULL)
		{
			if (memPtr->mem == mem)
			{
				break;
			}

			preMemPtr = memPtr;
			memPtr = memPtr->next;
		}
	}

    if (memPtr == NULL)
    {
        printf("voclProxyReleaseMem, Mem %p does not exist!\n", mem);
        exit (1);
    }

    preMemPtr->next = memPtr->next;
    free(memPtr);

    return;
}

void voclProxyReleaseAllMems()
{
    vocl_proxy_mem *memPtr, *nextMemPtr;

    memPtr = voclProxyMemPtr;
    while (memPtr != NULL)
    {
        nextMemPtr = memPtr->next;
        free(memPtr);
        memPtr = nextMemPtr;
    }

    voclProxyMemPtr = NULL;

    return;
}

