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

static void initializeReadBuffer(int proxyIndex)
{
    int i = 0;
    for (i = 0; i < VOCL_READ_BUFFER_NUM; i++) {
        voclReadBufferPtr[proxyIndex].voclReadBufferInfo[i].isInUse = 0;
    }

    voclReadBufferPtr[proxyIndex].curReadBufferIndex = 0;
    voclReadBufferPtr[proxyIndex].readDataRequestNum = 0;

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
	voclReadBufferPtr = (struct voclReadBuffer *)realloc(voclReadBufferPtr, sizeof(struct voclReadBuffer) * newBufferNum);
	for (i = origBufferNum; i < newBufferNum; i++)
	{
		initializeReadBuffer(i);
	}

	return;
}

void setReadBufferInUse(int proxyIndex, int index)
{
    voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].isInUse = 1;
}

MPI_Request *getReadRequestPtr(int proxyIndex, int index)
{
    return &voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].request;
}

int getNextReadBufferIndex(int proxyIndex)
{
	if (proxyIndex >= voclReadBufferNum)
	{
		reallocateReadBuffer(voclReadBufferNum, 2*voclReadBufferNum);
		voclReadBufferNum *= 2;
	}

    int index = voclReadBufferPtr[proxyIndex].curReadBufferIndex;
    MPI_Status status;

    if (voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].isInUse == 1) {
        MPI_Wait(getReadRequestPtr(proxyIndex, index), &status);
        voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].isInUse = 0;
    }

    if (++voclReadBufferPtr[proxyIndex].curReadBufferIndex >= VOCL_READ_BUFFER_NUM) {
        voclReadBufferPtr[proxyIndex].curReadBufferIndex = 0;
    }

    if (++voclReadBufferPtr[proxyIndex].readDataRequestNum >= VOCL_READ_BUFFER_NUM) {
        voclReadBufferPtr[proxyIndex].readDataRequestNum = VOCL_READ_BUFFER_NUM;
    }

    return index;
}

void processReadBuffer(int proxyIndex, int curIndex, int bufferNum)
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
        request[requestNo++] = *getReadRequestPtr(proxyIndex, index);
    }
    MPI_Waitall(requestNo, request, status);
    for (i = startIndex; i <= endIndex; i++) {
        index = i % VOCL_READ_BUFFER_NUM;
        voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].isInUse = 0;
    }

    return;
}

void processAllReads(int proxyIndex)
{
    int i, index, startIndex, endIndex;
    MPI_Request request[VOCL_READ_BUFFER_NUM];
    MPI_Status status[VOCL_READ_BUFFER_NUM];
    int requestNo;

    endIndex = voclReadBufferPtr[proxyIndex].curReadBufferIndex;
    startIndex = endIndex - voclReadBufferPtr[proxyIndex].readDataRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_READ_BUFFER_NUM;
        endIndex += VOCL_READ_BUFFER_NUM;
    }

    requestNo = 0;
    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_READ_BUFFER_NUM;
        if (voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].isInUse == 1) {
            request[requestNo++] = *getReadRequestPtr(proxyIndex, index);
        }
    }

    if (requestNo > 0) {
        MPI_Waitall(requestNo, request, status);
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_READ_BUFFER_NUM;
        voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].isInUse = 0;
    }

    voclReadBufferPtr[proxyIndex].curReadBufferIndex = 0;
    voclReadBufferPtr[proxyIndex].readDataRequestNum = 0;

    return;
}

