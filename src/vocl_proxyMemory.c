#include <stdio.h>
#include <string.h>
#include "vocl_proxyStructures.h"

static vocl_proxy_mem *voclProxyMemPtr = NULL;

void voclProxyAddMem(cl_mem mem, size_t size, cl_context context)
{
    vocl_proxy_mem *memPtr;

    memPtr = (vocl_proxy_mem *)malloc(sizeof(vocl_proxy_mem));
	memPtr->mem = mem;
	memPtr->size = size;
	memPtr->context = context;

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
    }

    if (memPtr == NULL)
    {
        printf("voclProxyGetMemPtr, mem %p does not exist!\n", mem);
        exit (1);
    }

    return memPtr;
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

