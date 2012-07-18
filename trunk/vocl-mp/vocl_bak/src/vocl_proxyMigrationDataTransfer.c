#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>
#include "vocl_proxy_macro.h"
#include "vocl_proxyBufferProc.h"
#include "vocl_proxy.h"

/*-------------------Write GPU memory operations --------------- */
static struct strProxyMigWriteBufferAll *proxyMigWriteBufferPtr = NULL;
static int proxyMigWriteBufferPoolNum  = 0;

void voclMigWriteBufferInitialize(int index)
{
    int i;

    for (i = 0; i < VOCL_MIG_BUF_NUM; i++) {
        proxyMigWriteBufferPtr[index].buffers[i].cmdQueue = NULL;
        proxyMigWriteBufferPtr[index].buffers[i].memory = NULL;
        proxyMigWriteBufferPtr[index].buffers[i].size = 0;
        proxyMigWriteBufferPtr[index].buffers[i].offset = 0;
        proxyMigWriteBufferPtr[index].buffers[i].event = NULL;
        proxyMigWriteBufferPtr[index].buffers[i].source = -1;
        proxyMigWriteBufferPtr[index].buffers[i].comm = -1;
        proxyMigWriteBufferPtr[index].buffers[i].tag = -1;
        proxyMigWriteBufferPtr[index].buffers[i].useFlag = MIG_WRT_AVAILABLE;
        proxyMigWriteBufferPtr[index].buffers[i].ptr = (char *) malloc(VOCL_MIG_BUF_SIZE * sizeof(char));
    }
    proxyMigWriteBufferPtr[index].voclMigWriteBufferIndex = 0;
    proxyMigWriteBufferPtr[index].voclMigWriteBufferRequestNum = 0;
}

void voclMigWriteBufferInitializeAll()
{
	int i;
	proxyMigWriteBufferPoolNum = VOCL_MIG_BUF_POOL;
	proxyMigWriteBufferPtr = 
		(struct strProxyMigWriteBufferAll*)malloc(sizeof(struct strProxyMigWriteBufferAll) * 
										   proxyMigWriteBufferPoolNum);

	for (i = 0; i < proxyMigWriteBufferPoolNum; i++)
	{
		voclMigWriteBufferInitialize(i);
	}
	return;
}

static void voclReallocMigWriteBuffer(int origBufferNum, int newBufferNum)
{
	int i;
	proxyMigWriteBufferPtr = (struct strProxyMigWriteBufferAll*)realloc(proxyMigWriteBufferPtr,
		sizeof(struct strProxyMigWriteBufferAll) * newBufferNum);
	
	for (i = origBufferNum; i < newBufferNum; i++)
	{
		voclMigWriteBufferInitialize(i);
	}
	return;
}

void voclMigWriteBufferFinalize()
{
    int i, rank;
	for (rank = 0; rank < proxyMigWriteBufferPoolNum; rank ++)
	{
		for (i = 0; i < VOCL_MIG_BUF_NUM; i++) {
			proxyMigWriteBufferPtr[rank].buffers[i].useFlag = MIG_WRT_AVAILABLE;
			free(proxyMigWriteBufferPtr[rank].buffers[i].ptr);
			proxyMigWriteBufferPtr[rank].buffers[i].ptr = NULL;
		}
		proxyMigWriteBufferPtr[rank].voclMigWriteBufferIndex = 0;
		proxyMigWriteBufferPtr[rank].voclMigWriteBufferRequestNum = 0;
	}
}

void voclMigSetWriteBufferFlag(int rank, int index, int flag)
{
    proxyMigWriteBufferPtr[rank].buffers[index].useFlag = flag;
    return;
}

struct strMigWriteBufferInfo *voclMigGetWriteBufferPtr(int rank, int index)
{
    return &proxyMigWriteBufferPtr[rank].buffers[index];
}

MPI_Request *voclMigGetWriteRequestPtr(int rank, int index)
{
    return &proxyMigWriteBufferPtr[rank].buffers[index].request;
}

static int voclMigWriteToGPUMemory(int rank, int index)
{
    int err;
    err = clEnqueueWriteBuffer(proxyMigWriteBufferPtr[rank].buffers[index].cmdQueue,
                               proxyMigWriteBufferPtr[rank].buffers[index].memory,
                               CL_FALSE,
                               proxyMigWriteBufferPtr[rank].buffers[index].offset,
                               proxyMigWriteBufferPtr[rank].buffers[index].size,
                               proxyMigWriteBufferPtr[rank].buffers[index].ptr,
                               0, NULL, &proxyMigWriteBufferPtr[rank].buffers[index].event);
    return err;
}

int voclMigGetNextWriteBufferIndex(int rank)
{
    int index;
    MPI_Status status;

	if (rank >= proxyMigWriteBufferPoolNum)
	{
		voclReallocMigWriteBuffer(proxyMigWriteBufferPoolNum, 
				2 * proxyMigWriteBufferPoolNum);
		proxyMigWriteBufferPoolNum *= 2;
	}

    index = proxyMigWriteBufferPtr[rank].voclMigWriteBufferIndex;
    if (proxyMigWriteBufferPtr[rank].buffers[index].useFlag == MIG_WRT_MPIRECV) {
        MPI_Wait(&proxyMigWriteBufferPtr[rank].buffers[index].request, &status);
        voclMigWriteToGPUMemory(rank, index);
        clWaitForEvents(1, &proxyMigWriteBufferPtr[rank].buffers[index].event);
    }
    else if (proxyMigWriteBufferPtr[rank].buffers[index].useFlag == MIG_WRT_WRTGPU) {
        clWaitForEvents(1, &proxyMigWriteBufferPtr[rank].buffers[index].event);
    }
    voclMigSetWriteBufferFlag(rank, index, MIG_WRT_AVAILABLE);

    /* if all buffer is used, start from 0 */
    if (++proxyMigWriteBufferPtr[rank].voclMigWriteBufferIndex >= VOCL_MIG_BUF_NUM) {
        proxyMigWriteBufferPtr[rank].voclMigWriteBufferIndex = 0;
    }

    /* at most VOCL_MIG_BUF_NUM buffers is in use */
    if (++proxyMigWriteBufferPtr[rank].voclMigWriteBufferRequestNum > VOCL_MIG_BUF_NUM) {
        proxyMigWriteBufferPtr[rank].voclMigWriteBufferRequestNum = VOCL_MIG_BUF_NUM;
    }

    return index;
}

int voclMigFinishDataWrite(int rank)
{
    int i, index, err = CL_SUCCESS;
    int startIndex, endIndex;
    cl_event eventList[VOCL_MIG_BUF_NUM];
    int eventNo = 0;
    MPI_Status status;

    endIndex = proxyMigWriteBufferPtr[rank].voclMigWriteBufferIndex;
    startIndex = endIndex - proxyMigWriteBufferPtr[rank].voclMigWriteBufferRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_MIG_BUF_NUM;
        endIndex += VOCL_MIG_BUF_NUM;
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_MIG_BUF_NUM;
        if (proxyMigWriteBufferPtr[rank].buffers[index].useFlag == MIG_WRT_MPIRECV) {
            MPI_Wait(&proxyMigWriteBufferPtr[rank].buffers[index].request, &status);
            voclMigWriteToGPUMemory(rank, index);
            eventList[eventNo++] = proxyMigWriteBufferPtr[rank].buffers[index].event;
        }
        else if (proxyMigWriteBufferPtr[rank].buffers[index].useFlag == MIG_WRT_WRTGPU) {
            eventList[eventNo++] = proxyMigWriteBufferPtr[rank].buffers[index].event;
        }
    }

    if (eventNo > 0) {
        err = clWaitForEvents(eventNo, eventList);
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_MIG_BUF_NUM;
        voclMigSetWriteBufferFlag(rank, index, MIG_WRT_AVAILABLE);
    }

    return err;
}


/*-------------------Read GPU memory operations --------------- */
static struct strProxyMigReadBufferAll *proxyMigReadBufferPtr = NULL;
static int proxyMigReadBufferPoolNum  = 0;

static void voclMigReadBufferInitialize(int index)
{
    int i;
    for (i = 0; i < VOCL_MIG_BUF_NUM; i++) {
        proxyMigReadBufferPtr[index].buffers[i].size = 0;
        proxyMigReadBufferPtr[index].buffers[i].offset = 0;
        proxyMigReadBufferPtr[index].buffers[i].event = NULL;
        proxyMigReadBufferPtr[index].buffers[i].dest = -1;
        proxyMigReadBufferPtr[index].buffers[i].comm = -1;
        proxyMigReadBufferPtr[index].buffers[i].tag = -1;
        proxyMigReadBufferPtr[index].buffers[i].useFlag = MIG_READ_AVAILABLE;
        proxyMigReadBufferPtr[index].buffers[i].ptr = (char *) malloc(VOCL_MIG_BUF_SIZE 
					* sizeof(char));
    }
    proxyMigReadBufferPtr[index].voclMigReadBufferIndex = 0;
    proxyMigReadBufferPtr[index].voclMigReadBufferRequestNum = 0;
}

void voclMigReadBufferInitializeAll()
{
	int i;
	proxyMigReadBufferPoolNum = VOCL_MIG_BUF_POOL;
	proxyMigReadBufferPtr = 
		(struct strProxyMigReadBufferAll*)malloc(sizeof(struct strProxyMigReadBufferAll) * 
					proxyMigReadBufferPoolNum);
	for (i = 0; i < proxyMigReadBufferPoolNum; i++)
	{
		voclMigReadBufferInitialize(i);
	}
}

static void voclReallocMigReadBuffer(int origBufferNum, int newBufferNum)
{
	int i;
	proxyMigReadBufferPtr = (struct strProxyMigReadBufferAll*)realloc(proxyMigReadBufferPtr,
		sizeof(struct strProxyMigReadBufferAll) * newBufferNum);
	
	for (i = origBufferNum; i < newBufferNum; i++)
	{
		voclMigReadBufferInitialize(i);
	}
	return;
}

void voclMigReadBufferFinalize()
{
    int index, i;
	for (index = 0; index < proxyMigReadBufferPoolNum; index++)
	{
		for (i = 0; i < VOCL_MIG_BUF_NUM; i++) {
			proxyMigReadBufferPtr[index].buffers[i].size = 0;
			proxyMigReadBufferPtr[index].buffers[i].offset = 0;
			proxyMigReadBufferPtr[index].buffers[i].event = NULL;
			proxyMigReadBufferPtr[index].buffers[i].dest = -1;
			proxyMigReadBufferPtr[index].buffers[i].tag = -1;
			proxyMigReadBufferPtr[index].buffers[i].useFlag = MIG_READ_AVAILABLE;
			free(proxyMigReadBufferPtr[index].buffers[i].ptr);
			proxyMigReadBufferPtr[index].buffers[i].ptr = NULL;
		}
		proxyMigReadBufferPtr[index].voclMigReadBufferIndex = 0;
		proxyMigReadBufferPtr[index].voclMigReadBufferRequestNum = 0;
	}

	free(proxyMigReadBufferPtr);

	return;
}

void voclMigSetReadBufferFlag(int rank, int index, int flag)
{
    proxyMigReadBufferPtr[rank].buffers[index].useFlag = flag;
    return;
}

cl_event *voclMigGetReadEventPtr(int rank, int index)
{
    return &proxyMigReadBufferPtr[rank].buffers[index].event;
}

struct strMigReadBufferInfo *voclMigGetReadBufferPtr(int rank, int index)
{
    return &proxyMigReadBufferPtr[rank].buffers[index];
}

static int voclMigSendDataToTarget(int rank, int index)
{
    int err;
    err = MPI_Isend(proxyMigReadBufferPtr[rank].buffers[index].ptr,
                    proxyMigReadBufferPtr[rank].buffers[index].size,
                    MPI_BYTE,
                    proxyMigReadBufferPtr[rank].buffers[index].dest,
                    proxyMigReadBufferPtr[rank].buffers[index].tag,
                    proxyMigReadBufferPtr[rank].buffers[index].commData, 
					&proxyMigReadBufferPtr[rank].buffers[index].request);
    return err;
}

int voclMigGetNextReadBufferIndex(int rank)
{
    int index;
    MPI_Status status;
	if (rank >= proxyMigReadBufferPoolNum)
	{
		voclReallocMigReadBuffer(proxyMigReadBufferPoolNum, 2*proxyMigReadBufferPoolNum);
		proxyMigReadBufferPoolNum *= 2;
	}

    index = proxyMigReadBufferPtr[rank].voclMigReadBufferIndex;
    if (proxyMigReadBufferPtr[rank].buffers[index].useFlag == MIG_READ_RDGPU) {
        clWaitForEvents(1, &proxyMigReadBufferPtr[rank].buffers[index].event);
        voclMigSendDataToTarget(rank, index);
        MPI_Wait(&proxyMigReadBufferPtr[rank].buffers[index].request, &status);
    }
    else if (proxyMigReadBufferPtr[rank].buffers[rank].useFlag == MIG_READ_MPISEND) {
        MPI_Wait(&proxyMigReadBufferPtr[rank].buffers[rank].request, &status);
    }
    voclMigSetReadBufferFlag(rank, index, MIG_READ_AVAILABLE);

    /* if all buffer is used, start from 0 */
    if (++proxyMigReadBufferPtr[rank].voclMigReadBufferIndex >= VOCL_MIG_BUF_NUM) {
        proxyMigReadBufferPtr[rank].voclMigReadBufferIndex = 0;
    }

    /* at most VOCL_MIG_BUF_NUM buffers is in use */
    if (++proxyMigReadBufferPtr[rank].voclMigReadBufferRequestNum > VOCL_MIG_BUF_NUM) {
        proxyMigReadBufferPtr[rank].voclMigReadBufferRequestNum = VOCL_MIG_BUF_NUM;
    }

    return index;
}

int voclMigFinishDataRead(int rank)
{
    int i, index, err = MPI_SUCCESS;
    int startIndex, endIndex;
    MPI_Request requestList[VOCL_MIG_BUF_NUM];
    MPI_Status status[VOCL_MIG_BUF_NUM];
    int requestNo = 0;

    endIndex = proxyMigReadBufferPtr[rank].voclMigReadBufferIndex;
    startIndex = endIndex - proxyMigReadBufferPtr[rank].voclMigReadBufferRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_MIG_BUF_NUM;
        endIndex += VOCL_MIG_BUF_NUM;
    }

    requestNo = 0;
    for (i = startIndex; i < endIndex; i++) {
    	index = i % VOCL_MIG_BUF_NUM;
        if ( proxyMigReadBufferPtr[rank].buffers[index].useFlag == MIG_READ_RDGPU) {
            err = clWaitForEvents(1, & proxyMigReadBufferPtr[rank].buffers[index].event);
            voclMigSendDataToTarget(rank, index);
            requestList[requestNo++] =  proxyMigReadBufferPtr[rank].buffers[index].request;
        }
        else if ( proxyMigReadBufferPtr[rank].buffers[index].useFlag == MIG_READ_MPISEND) {
            requestList[requestNo++] =  proxyMigReadBufferPtr[rank].buffers[index].request;
        }
    }

    if (requestNo > 0) {
        err = MPI_Waitall(requestNo, requestList, status);
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_MIG_BUF_NUM;
        voclMigSetReadBufferFlag(rank, index, MIG_READ_AVAILABLE);
        proxyMigReadBufferPtr[rank].buffers[index].dest = -1;
        proxyMigReadBufferPtr[rank].buffers[index].comm = -1;
    }

    return err;
}

/*-----------------------Read/Write GPU memory operations on the same node---------------*/
static struct strProxyMigRWBufferAll *proxyMigRWBufferPtr = NULL;
static int proxyMigRWBufferPoolNum  = 0;

static void voclMigRWBufferInitialize(int index)
{
    int i;
    for (i = 0; i < VOCL_MIG_BUF_NUM; i++) {
        proxyMigRWBufferPtr[index].buffers[i].wtCmdQueue = NULL;
        proxyMigRWBufferPtr[index].buffers[i].wtMem = NULL;
        proxyMigRWBufferPtr[index].buffers[i].size = 0;
        proxyMigRWBufferPtr[index].buffers[i].offset = 0;
        proxyMigRWBufferPtr[index].buffers[i].useFlag = MIG_RW_SAME_NODE_AVLB;
        proxyMigRWBufferPtr[index].buffers[i].ptr = (char *) malloc(VOCL_MIG_BUF_SIZE * sizeof(char));
        proxyMigRWBufferPtr[index].buffers[i].rdEvent = NULL;
        proxyMigRWBufferPtr[index].buffers[i].wtEvent = NULL;
    }
    proxyMigRWBufferPtr[index].voclMigRWBufferIndex = 0;
    proxyMigRWBufferPtr[index].voclMigRWBufferRequestNum = 0;
}

void voclMigRWBufferInitializeAll()
{
	int i;
	proxyMigRWBufferPoolNum = VOCL_MIG_BUF_POOL;
	proxyMigRWBufferPtr = 
		(struct strProxyMigRWBufferAll*)malloc(sizeof(struct strProxyMigRWBufferAll) * 
					proxyMigRWBufferPoolNum);
	for (i = 0; i < proxyMigRWBufferPoolNum; i++)
	{
		voclMigRWBufferInitialize(i);
	}
}

static void voclReallocMigRWBuffer(int origBufferNum, int newBufferNum)
{
	int i;
	proxyMigRWBufferPtr = (struct strProxyMigRWBufferAll*)realloc(proxyMigRWBufferPtr, 
			sizeof(struct strProxyMigRWBufferAll) * newBufferNum);
	for (i = origBufferNum; i < newBufferNum; i++)
	{
		voclMigRWBufferInitialize(i);
	}

	return;
}

void voclMigRWBufferFinalize()
{
	int index, i;
	for (index = 0; index < proxyMigRWBufferPoolNum; index++)
	{
		for (i = 0; i < VOCL_MIG_BUF_NUM; i++)
		{
			proxyMigRWBufferPtr[index].buffers[i].useFlag = MIG_RW_SAME_NODE_AVLB;
			free(proxyMigRWBufferPtr[index].buffers[i].ptr);
			proxyMigRWBufferPtr[index].buffers[i].ptr = NULL;
		}
	}

	free(proxyMigRWBufferPtr);

	return;
}

static int voclMigRWWriteToGPUMem(int rank, int index)
{
    int err;
	float *tmp = (float *)proxyMigRWBufferPtr[rank].buffers[index].ptr;
    err = clEnqueueWriteBuffer(proxyMigRWBufferPtr[rank].buffers[index].wtCmdQueue,
                               proxyMigRWBufferPtr[rank].buffers[index].wtMem,
                               CL_FALSE,
                               proxyMigRWBufferPtr[rank].buffers[index].offset,
                               proxyMigRWBufferPtr[rank].buffers[index].size,
                               proxyMigRWBufferPtr[rank].buffers[index].ptr,
                               0, NULL, 
							   &proxyMigRWBufferPtr[rank].buffers[index].wtEvent);
    return err;
}

int voclMigRWGetNextBufferIndex(int rank)
{
	if (rank >= proxyMigRWBufferPoolNum)
	{
		voclReallocMigRWBuffer(proxyMigRWBufferPoolNum, 2*proxyMigRWBufferPoolNum);
		proxyMigRWBufferPoolNum *= 2;
	}

    int index = proxyMigRWBufferPtr[rank].voclMigRWBufferIndex;

    if (proxyMigRWBufferPtr[rank].buffers[index].useFlag == MIG_RW_SAME_NODE_RDMEM) {
        clWaitForEvents(1, &proxyMigRWBufferPtr[rank].buffers[index].rdEvent);
        voclMigRWWriteToGPUMem(rank, index);
        clWaitForEvents(1, &proxyMigRWBufferPtr[rank].buffers[index].wtEvent);
    }
    else if (proxyMigRWBufferPtr[rank].buffers[index].useFlag == MIG_RW_SAME_NODE_WTMEM) {
        clWaitForEvents(1, &proxyMigRWBufferPtr[rank].buffers[index].wtEvent);
    }
    proxyMigRWBufferPtr[rank].buffers[index].useFlag = MIG_RW_SAME_NODE_AVLB;

    if (++proxyMigRWBufferPtr[rank].voclMigRWBufferIndex >= VOCL_MIG_BUF_NUM) {
        proxyMigRWBufferPtr[rank].voclMigRWBufferIndex = 0;
    }

    if (++proxyMigRWBufferPtr[rank].voclMigRWBufferRequestNum > VOCL_MIG_BUF_NUM) {
        proxyMigRWBufferPtr[rank].voclMigRWBufferRequestNum = VOCL_MIG_BUF_NUM;
    }

    return index;
}

struct strMigRWBufferSameNode *voclMigRWGetBufferInfoPtr(int rank, int index)
{
	return &proxyMigRWBufferPtr[rank].buffers[index];
}

void voclMigSetRWBufferFlag(int rank, int index, int flag)
{
    proxyMigRWBufferPtr[rank].buffers[index].useFlag = flag;
    return;
}

int voclMigFinishDataRWOnSameNode(int rank)
{
    int i, index, err;
    int startIndex, endIndex;
    cl_event eventList[VOCL_MIG_BUF_NUM];
    int eventNo = 0;

    endIndex = proxyMigRWBufferPtr[rank].voclMigRWBufferIndex;
    startIndex = endIndex - proxyMigRWBufferPtr[rank].voclMigRWBufferRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_MIG_BUF_NUM;
        endIndex += VOCL_MIG_BUF_NUM;
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_MIG_BUF_NUM;
        if (proxyMigRWBufferPtr[rank].buffers[index].useFlag == MIG_RW_SAME_NODE_RDMEM) {
            clWaitForEvents(1, &proxyMigRWBufferPtr[rank].buffers[index].rdEvent);
            voclMigRWWriteToGPUMem(rank, index);
            eventList[eventNo++] = proxyMigRWBufferPtr[rank].buffers[index].wtEvent;
        }
        else if (proxyMigRWBufferPtr[rank].buffers[index].useFlag == MIG_RW_SAME_NODE_WTMEM) {
            eventList[eventNo++] = proxyMigRWBufferPtr[rank].buffers[index].wtEvent;
        }
    }
    if (eventNo > 0) {
        err = clWaitForEvents(eventNo, eventList);
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_MIG_BUF_NUM;
        proxyMigRWBufferPtr[rank].buffers[index].useFlag = MIG_RW_SAME_NODE_AVLB;
        proxyMigRWBufferPtr[rank].buffers[index].rdEvent = NULL;
        proxyMigRWBufferPtr[rank].buffers[index].wtEvent = NULL;
    }

    return err;
}

