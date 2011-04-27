#include "vocl_structures.h"

static struct strVOCLMemory *voclMemoryPtr = NULL;
static vocl_mem voclMemory;
static int voclMemoryNum;
static int voclMemoryNo;

static vocl_mem getVOCLMemoryValue()
{
    vocl_mem memory = voclMemory;
	voclMemory++;

    return memory;
}

static struct strVOCLMemory *getVOCLMemoryPtr()
{
    if (voclMemoryNo >= voclMemoryNum) {
        voclMemoryNum *= 2;
        voclMemoryPtr = (struct strVOCLMemory *) realloc(voclMemoryPtr,
                                                   voclMemoryNum *
                                                   sizeof(struct strVOCLMemory));
    }
    return &voclMemoryPtr[voclMemoryNo++];
}


void voclMemoryInitialize()
{
    voclMemoryNum = VOCL_MEM_OBJ_NUM;
    voclMemoryPtr =
        (struct strVOCLMemory *) malloc(voclMemoryNum * sizeof(struct strVOCLMemory));
    voclMemoryNo = 0;
    voclMemory = 0;
}

void voclMemoryFinalize()
{
    if (voclMemoryPtr != NULL) {
        free(voclMemoryPtr);
        voclMemoryPtr = NULL;
    }
    voclMemoryNo = 0;
    voclMemory = 0;
    voclMemoryNum = 0;
}

vocl_mem voclCLMemory2VOCLMemory(cl_mem memory, int proxyID)
{
    struct strVOCLMemory *memoryPtr = getVOCLMemoryPtr();
    memoryPtr->clMemory = memory;
	memoryPtr->proxyID = proxyID;
    //memoryPtr->voclMemory = (vocl_mem)memory;
    memoryPtr->voclMemory = getVOCLMemoryValue();

    return memoryPtr->voclMemory;
}

cl_mem voclVOCLMemory2CLMemoryComm(vocl_mem memory, int *proxyID)
{
    int i = (int)memory;
	*proxyID = voclMemoryPtr[i].proxyID;

    return voclMemoryPtr[i].clMemory;
}

