#include <stdio.h>
#include "voclStructures.h"

extern cl_mem voclMigCreateBuffer(vocl_context context,
        cl_mem_flags flags, size_t size, 
		void *host_ptr, cl_int * errcode_ret);
extern char voclContextGetMigrationStatus(vocl_context context);
extern cl_context voclVOCLContext2CLContextComm(vocl_context context, int *proxyRank,
                                         int *proxyIndex, MPI_Comm * proxyComm,
										 MPI_Comm * proxyCommData);

vocl_context voclMemGetContext(vocl_mem mem);
void voclUpdateVOCLMemory(vocl_mem voclMemory, int proxyRank, int proxyIndex,
                          MPI_Comm proxyComm, MPI_Comm proxyCommData, vocl_context context);

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
    memPtr = (struct strVOCLMemory *) malloc(sizeof(struct strVOCLMemory));
    memPtr->isWritten = 0;
    memPtr->isOldValid = 0;
    memPtr->next = voclMemoryPtr;
    voclMemoryPtr = memPtr;

    return memPtr;
}

static struct strVOCLMemory *getVOCLMemoryPtr(vocl_mem memory)
{
    struct strVOCLMemory *memPtr;
    memPtr = voclMemoryPtr;
    while (memPtr != NULL) {
        if (memPtr->voclMemory == memory) {
            break;
        }
        memPtr = memPtr->next;
    }

    if (memPtr == NULL) {
        printf("Error, memory does not exist!\n");
        exit(1);
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
    while (memPtr != NULL) {
        tmpMemPtr = memPtr->next;
        free(memPtr);
        memPtr = tmpMemPtr;
    }

    voclMemoryPtr = NULL;
    voclMemoryNo = 0;
    voclMemory = 0;
}

vocl_mem voclCLMemory2VOCLMemory(cl_mem memory, int proxyRank,
                                 int proxyIndex, MPI_Comm proxyComm, MPI_Comm proxyCommData)
{
    struct strVOCLMemory *memoryPtr = createVOCLMemory();
    memoryPtr->clMemory = memory;
    memoryPtr->proxyRank = proxyRank;
    memoryPtr->proxyIndex = proxyIndex;
    memoryPtr->proxyComm = proxyComm;
    memoryPtr->proxyCommData = proxyCommData;
    memoryPtr->voclMemory = getVOCLMemoryValue();

    return memoryPtr->voclMemory;
}

void voclStoreMemoryParameters(vocl_mem memory, cl_mem_flags flags,
                               size_t size, vocl_context context)
{
    struct strVOCLMemory *memoryPtr = getVOCLMemoryPtr(memory);
    memoryPtr->flags = flags;
    memoryPtr->size = size;
    memoryPtr->context = context;

    return;
}

size_t voclGetVOCLMemorySize(vocl_mem memory)
{
    struct strVOCLMemory *memoryPtr = getVOCLMemoryPtr(memory);
    return memoryPtr->size;
}

cl_mem voclVOCLMemory2CLMemoryComm(vocl_mem memory, int *proxyRank,
                                   int *proxyIndex, MPI_Comm * proxyComm,
                                   MPI_Comm * proxyCommData)
{
    struct strVOCLMemory *memoryPtr = getVOCLMemoryPtr(memory);
    *proxyRank = memoryPtr->proxyRank;
    *proxyIndex = memoryPtr->proxyIndex;
    *proxyComm = memoryPtr->proxyComm;
    *proxyCommData = memoryPtr->proxyCommData;

    return memoryPtr->clMemory;
}

cl_mem voclVOCLMemory2OldCLMemoryComm(vocl_mem memory, int *proxyRank,
                                      int *proxyIndex, MPI_Comm * proxyComm,
                                      MPI_Comm * proxyCommData)
{
    struct strVOCLMemory *memoryPtr = getVOCLMemoryPtr(memory);
    *proxyRank = memoryPtr->oldProxyRank;
    *proxyIndex = memoryPtr->oldProxyIndex;
    *proxyComm = memoryPtr->oldProxyComm;
    *proxyCommData = memoryPtr->oldProxyCommData;

    return memoryPtr->oldMemory;
}

void voclSetOldMemoryReleased(vocl_mem memory)
{
    struct strVOCLMemory *memoryPtr = getVOCLMemoryPtr(memory);
    memoryPtr->isOldValid = 0;
    return;
}

int voclIsOldMemoryValid(vocl_mem memory)
{
    struct strVOCLMemory *memoryPtr = getVOCLMemoryPtr(memory);
    return memoryPtr->isOldValid;
}

cl_mem voclVOCLMemory2CLMemory(vocl_mem memory)
{
    struct strVOCLMemory *memoryPtr = getVOCLMemoryPtr(memory);
    return memoryPtr->clMemory;
}

void voclUpdateSingleMemory(vocl_mem mem)
{
	int proxyRank, proxyIndex;
	MPI_Comm proxyComm, proxyCommData;
	vocl_context voclContext;
	cl_context clContext;
	int err;

	voclContext = voclMemGetContext(mem);
	clContext = voclVOCLContext2CLContextComm(voclContext, &proxyRank,
				&proxyIndex, &proxyComm, &proxyCommData);
	
	voclUpdateVOCLMemory(mem, proxyRank, proxyIndex, proxyComm, proxyCommData,
		voclContext);

	return;
}

void voclUpdateVOCLMemory(vocl_mem voclMemory, int proxyRank, int proxyIndex,
                          MPI_Comm proxyComm, MPI_Comm proxyCommData, vocl_context context)
{
    struct strVOCLMemory *memoryPtr = getVOCLMemoryPtr(voclMemory);
    int err;

    /* store old cl memory info */
    memoryPtr->oldMemory = memoryPtr->clMemory;
    memoryPtr->oldProxyRank = memoryPtr->proxyRank;
    memoryPtr->oldProxyIndex = memoryPtr->proxyIndex;
    memoryPtr->oldProxyComm = memoryPtr->proxyComm;
    memoryPtr->oldProxyCommData = memoryPtr->proxyCommData;
    memoryPtr->isOldValid = 1;

    /* update the cl_mem corresponding to the vocl_mem */
    memoryPtr->proxyRank = proxyRank;
    memoryPtr->proxyIndex = proxyIndex;
    memoryPtr->proxyComm = proxyComm;
    memoryPtr->proxyCommData = proxyCommData;

    memoryPtr->clMemory = voclMigCreateBuffer(context, memoryPtr->flags,
                                              memoryPtr->size, NULL, &err);
	memoryPtr->migrationStatus = voclContextGetMigrationStatus(context);

    return;
}

void voclSetMemWrittenFlag(vocl_mem memory, int flag)
{
    struct strVOCLMemory *memoryPtr = getVOCLMemoryPtr(memory);
    memoryPtr->isWritten = flag;
    return;
}

int voclGetMemWrittenFlag(vocl_mem memory)
{
    struct strVOCLMemory *memoryPtr = getVOCLMemoryPtr(memory);
    return memoryPtr->isWritten;
}

void voclSetMemHostPtr(vocl_mem memory, void *ptr)
{
    struct strVOCLMemory *memoryPtr = getVOCLMemoryPtr(memory);
    memoryPtr->hostPtr = ptr;
	return;
}

void * voclGetMemHostPtr(vocl_mem memory)
{
    struct strVOCLMemory *memoryPtr = getVOCLMemoryPtr(memory);
    return memoryPtr->hostPtr;
}

void voclMemSetMigrationStatus(vocl_mem mem, char status)
{
	struct strVOCLMemory *memoryPtr = getVOCLMemoryPtr(mem);
	memoryPtr->migrationStatus = status;
	return;
}

char voclMemGetMigrationStatus(vocl_mem mem)
{
	struct strVOCLMemory *memoryPtr = getVOCLMemoryPtr(mem);
	return memoryPtr->migrationStatus;
}

vocl_context voclMemGetContext(vocl_mem mem)
{
	struct strVOCLMemory *memoryPtr = getVOCLMemoryPtr(mem);
	return memoryPtr->context;
}

int voclReleaseMemory(vocl_mem memory)
{
    struct strVOCLMemory *memoryPtr, *preMemoryPtr, *curMemoryPtr;
    /* the first node in the link list */
    if (memory == voclMemoryPtr->voclMemory) {
        memoryPtr = voclMemoryPtr;
        voclMemoryPtr = voclMemoryPtr->next;
        free(memoryPtr);

        return 0;
    }

    memoryPtr = NULL;
    preMemoryPtr = voclMemoryPtr;
    curMemoryPtr = voclMemoryPtr->next;
    while (curMemoryPtr != NULL) {
        if (memory == curMemoryPtr->voclMemory) {
            memoryPtr = curMemoryPtr;
            break;
        }
        preMemoryPtr = curMemoryPtr;
        curMemoryPtr = curMemoryPtr->next;
    }

    if (memoryPtr == NULL) {
        printf("memory does not exist!\n");
        exit(1);
    }

    /* remote the current node from link list */
    preMemoryPtr->next = curMemoryPtr->next;
    free(curMemoryPtr);

    return 0;
}
