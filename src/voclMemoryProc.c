#include <stdio.h>
#include "vocl_structures.h"

static struct strVOCLMemory *voclMemoryPtr = NULL;
static vocl_mem voclMemory;
static int voclMemoryNo;

static vocl_mem getVOCLMemoryValue()
{
    vocl_mem memory = voclMemory;
	voclMemory++;

    return memory;
}

static struct strVOCLMemory *createVOCLMemory()
{
	struct strVOCLMemory *memPtr;
	memPtr = (struct strVOCLMemory *)malloc(sizeof(struct strVOCLMemory));
	memPtr->next = voclMemoryPtr;
	voclMemoryPtr = memPtr;

	return memPtr;
}

static struct strVOCLMemory *getVOCLMemoryPtr(vocl_mem memory)
{
	struct strVOCLMemory *memPtr;
	memPtr = voclMemoryPtr;
	while (memPtr != NULL)
	{
		if (memPtr->voclMemory == memory)
		{
			break;
		}
		memPtr = memPtr->next;
	}

	if (memPtr == NULL)
	{
		printf("Error, memory does not exist!\n");
		exit (1);
	}

	return memPtr;
}

void voclMemoryInitialize()
{
    voclMemoryPtr = NULL;
    voclMemoryNo = 0;
    voclMemory = 0;
}

void voclMemoryFinalize()
{
	struct strVOCLMemory *memPtr, *tmpMemPtr;
	memPtr = voclMemoryPtr;
	while (memPtr != NULL)
	{
		tmpMemPtr = memPtr->next;
		free(memPtr);
		memPtr = tmpMemPtr;
	}

    voclMemoryPtr = NULL;
    voclMemoryNo = 0;
    voclMemory = 0;
}

vocl_mem voclCLMemory2VOCLMemory(cl_mem memory, int proxyID,
             int proxyIndex, MPI_Comm proxyComm, MPI_Comm proxyCommData)
{
    struct strVOCLMemory *memoryPtr = createVOCLMemory();
    memoryPtr->clMemory = memory;
	memoryPtr->proxyID = proxyID;
	memoryPtr->proxyIndex = proxyIndex;
	memoryPtr->proxyComm = proxyComm;
	memoryPtr->proxyCommData = proxyCommData;
    memoryPtr->voclMemory = getVOCLMemoryValue();

    return memoryPtr->voclMemory;
}

cl_mem voclVOCLMemory2CLMemoryComm(vocl_mem memory, int *proxyID,
           int *proxyIndex, MPI_Comm *proxyComm, MPI_Comm *proxyCommData)
{
	struct strVOCLMemory *memoryPtr = getVOCLMemoryPtr(memory);
	*proxyID = memoryPtr->proxyID;
	*proxyIndex = memoryPtr->proxyIndex;
	*proxyComm = memoryPtr->proxyComm;
	*proxyCommData = memoryPtr->proxyCommData;

    return memoryPtr->clMemory;
}

int voclReleaseMemory(vocl_mem memory)
{
	struct strVOCLMemory *memoryPtr, *preMemoryPtr, *curMemoryPtr;
	/* the first node in the link list */
	if (memory == voclMemoryPtr->voclMemory)
	{
		memoryPtr = voclMemoryPtr;
		voclMemoryPtr = voclMemoryPtr->next;
		free(memoryPtr);

		return 0;
	}

	memoryPtr = NULL;
	preMemoryPtr = voclMemoryPtr;
	curMemoryPtr = voclMemoryPtr->next;
	while (curMemoryPtr != NULL)
	{
		if (memory == curMemoryPtr->voclMemory)
		{
			memoryPtr = curMemoryPtr;
			break;
		}
		preMemoryPtr = curMemoryPtr;
		curMemoryPtr = curMemoryPtr->next;
	}

	if (memoryPtr == NULL)
	{
		printf("memory does not exist!\n");
		exit (1);
	}

	/* remote the current node from link list */
	preMemoryPtr->next = curMemoryPtr->next;
	free(curMemoryPtr);
	
	return 0;
}
