#include "vocl_opencl.h"
#include "vocl_structures.h"

/* for receiving data from GPU to local node */
//struct strReadBufferInfo {
//    int isInUse;
//    MPI_Request request;
//};
//
//static int readDataRequestNum;
//static int curReadBufferIndex;
//static struct strReadBufferInfo voclReadBufferInfo[VOCL_READ_BUFFER_NUM];
static struct voclReadBuffer *voclReadBufferPtr = NULL;
static int voclReadBufferNum;

static void initializeReadBuffer(int proxyID)
{
    int i = 0;
    for (i = 0; i < VOCL_READ_BUFFER_NUM; i++) {
        voclReadBufferPtr[proxyID].voclReadBufferInfo[i].isInUse = 0;
    }

    voclReadBufferPtr[proxyID].curReadBufferIndex = 0;
    voclReadBufferPtr[proxyID].readDataRequestNum = 0;

    return;
}

void initializeVoclReadBufferAll()
{
	int i;
	voclReadBufferNum = VOCL_BUFF_NUM;
	voclReadBufferPtr = (struct voclReadBuffer *)malloc(sizeof(struct voclReadBuffer) * voclReadBufferNum);
	for (i = 0; i < voclReadBufferNum; i++)
	{
		initializeReadBuffer(i);
	}
}

void finalizeVoclReadBufferAll()
{
	if (voclReadBufferPtr != NULL)
	{
		free(voclReadBufferPtr);
		voclReadBufferPtr = NULL;
	}

	return;
}

static void reallocateReadBuffer(int origBufferNum, int newBufferNum)
{
	int i;
	voclReadBufferPtr = (struct voclReadBuffer *)malloc(sizeof(struct voclReadBuffer) * newBufferNum);
	for (i = origBufferNum; i < newBufferNum; i++)
	{
		initializeReadBuffer(i);
	}

	return;
}

void setReadBufferInUse(int proxyID, int index)
{
    voclReadBufferPtr[proxyID].voclReadBufferInfo[index].isInUse = 1;
}

MPI_Request *getReadRequestPtr(int proxyID, int index)
{
    return &voclReadBufferPtr[proxyID].voclReadBufferInfo[index].request;
}

int getNextReadBufferIndex(int proxyID)
{
	if (proxyID >= voclReadBufferNum)
	{
		reallocateReadBuffer(voclReadBufferNum, 2*voclReadBufferNum);
		voclReadBufferNum *= 2;
	}

    int index = voclReadBufferPtr[proxyID].curReadBufferIndex;
    MPI_Status status;

    if (voclReadBufferPtr[proxyID].voclReadBufferInfo[index].isInUse == 1) {
        MPI_Wait(getReadRequestPtr(proxyID, index), &status);
        voclReadBufferPtr[proxyID].voclReadBufferInfo[index].isInUse = 0;
    }

    if (++voclReadBufferPtr[proxyID].curReadBufferIndex >= VOCL_READ_BUFFER_NUM) {
        voclReadBufferPtr[proxyID].curReadBufferIndex = 0;
    }

    if (++voclReadBufferPtr[proxyID].readDataRequestNum >= VOCL_READ_BUFFER_NUM) {
        voclReadBufferPtr[proxyID].readDataRequestNum = VOCL_READ_BUFFER_NUM;
    }

    return index;
}

void processReadBuffer(int proxyID, int curIndex, int bufferNum)
{
    int i, index, startIndex, endIndex;
    MPI_Request request[VOCL_READ_BUFFER_NUM];
    MPI_Status status[VOCL_READ_BUFFER_NUM];
    int requestNo;

    /* at most VOCL_READ_BUFFER_NUM buffers to be processed */
    if (bufferNum > VOCL_READ_BUFFER_NUM) {
        bufferNum = VOCL_READ_BUFFER_NUM;
    }

    startIndex = curIndex - bufferNum + 1;
    endIndex = curIndex;
    if (startIndex < 0) {
        startIndex += VOCL_READ_BUFFER_NUM;
        endIndex += VOCL_READ_BUFFER_NUM;
    }

    requestNo = 0;
    for (i = startIndex; i <= endIndex; i++) {
        index = i % VOCL_READ_BUFFER_NUM;
        request[requestNo++] = *getReadRequestPtr(proxyID, index);
    }
    MPI_Waitall(requestNo, request, status);
    for (i = startIndex; i <= endIndex; i++) {
        index = i % VOCL_READ_BUFFER_NUM;
        voclReadBufferPtr[proxyID].voclReadBufferInfo[index].isInUse = 0;
    }

    return;
}

void processAllReads(int proxyID)
{
    int i, index, startIndex, endIndex;
    MPI_Request request[VOCL_READ_BUFFER_NUM];
    MPI_Status status[VOCL_READ_BUFFER_NUM];
    int requestNo;

    endIndex = voclReadBufferPtr[proxyID].curReadBufferIndex;
    startIndex = endIndex - voclReadBufferPtr[proxyID].readDataRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_READ_BUFFER_NUM;
        endIndex += VOCL_READ_BUFFER_NUM;
    }

    requestNo = 0;
    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_READ_BUFFER_NUM;
        if (voclReadBufferPtr[proxyID].voclReadBufferInfo[index].isInUse == 1) {
            request[requestNo++] = *getReadRequestPtr(proxyID, index);
        }
    }

    if (requestNo > 0) {
        MPI_Waitall(requestNo, request, status);
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_READ_BUFFER_NUM;
        voclReadBufferPtr[proxyID].voclReadBufferInfo[index].isInUse = 0;
    }

    voclReadBufferPtr[proxyID].curReadBufferIndex = 0;
    voclReadBufferPtr[proxyID].readDataRequestNum = 0;

    return;
}
