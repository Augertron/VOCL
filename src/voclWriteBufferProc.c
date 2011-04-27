#include "vocl_opencl.h"
#include "vocl_structures.h"

/* for sending data from local node to GPU */
//struct strWriteBufferInfo {
//    int isInUse;
//    MPI_Request request;
//};
//
//static struct strWriteBufferInfo voclWriteBufferInfo[VOCL_WRITE_BUFFER_NUM];
//static int writeDataRequestNum;
//static int curWriteBufferIndex;

static struct voclWriteBuffer *voclWriteBufferPtr = NULL;
static int voclWriteBufferNum;

static void initializeWriteBuffer(int proxyID)
{
    int i = 0;
    for (i = 0; i < VOCL_WRITE_BUFFER_NUM; i++) {
        voclWriteBufferPtr[proxyID].voclWriteBufferInfo[i].isInUse = 0;
    }
    voclWriteBufferPtr[proxyID].curWriteBufferIndex = 0;
    voclWriteBufferPtr[proxyID].writeDataRequestNum = 0;

    return;
}

void initializeVoclWriteBufferAll()
{
    int i;
	voclWriteBufferNum = VOCL_BUFF_NUM;
	voclWriteBufferPtr = (struct voclWriteBuffer *)malloc(sizeof(struct voclWriteBuffer) * voclWriteBufferNum);
	for (i = 0; i < voclWriteBufferNum; i++)
	{
		initializeWriteBuffer(i);
	}

	return;
}

void finalizeVoclWriteBufferAll()
{
	if (voclWriteBufferPtr != NULL)
	{
		free(voclWriteBufferPtr);
		voclWriteBufferPtr = NULL;
	}

	return;
}

static void reallocateWriteBuffer(int origBufferNum, int newBufferNum)
{
	int i;
	voclWriteBufferPtr = (struct voclWriteBuffer *)malloc(sizeof(struct voclWriteBuffer) * newBufferNum);
	for (i = origBufferNum; i < newBufferNum; i++)
	{
		initializeWriteBuffer(i);
	}

	return;
}


void setWriteBufferInUse(int proxyID, int index)
{
    voclWriteBufferPtr[proxyID].voclWriteBufferInfo[index].isInUse = 1;
}

MPI_Request *getWriteRequestPtr(int proxyID, int index)
{
    return &voclWriteBufferPtr[proxyID].voclWriteBufferInfo[index].request;
}

int getNextWriteBufferIndex(int proxyID)
{
	if (proxyID >= voclWriteBufferNum)
	{
		reallocateWriteBuffer(voclWriteBufferNum, 2*voclWriteBufferNum);
		voclWriteBufferNum *= 2;
	}

    int index = voclWriteBufferPtr[proxyID].curWriteBufferIndex;
    MPI_Status status;

    if (voclWriteBufferPtr[proxyID].voclWriteBufferInfo[index].isInUse == 1) {
        MPI_Wait(getWriteRequestPtr(proxyID, index), &status);
        voclWriteBufferPtr[proxyID].voclWriteBufferInfo[index].isInUse = 0;
    }

    if (++voclWriteBufferPtr[proxyID].curWriteBufferIndex >= VOCL_WRITE_BUFFER_NUM) {
        voclWriteBufferPtr[proxyID].curWriteBufferIndex = 0;
    }

    if (++voclWriteBufferPtr[proxyID].writeDataRequestNum >= VOCL_WRITE_BUFFER_NUM) {
        voclWriteBufferPtr[proxyID].writeDataRequestNum = VOCL_WRITE_BUFFER_NUM;
    }

    return index;
}

void processWriteBuffer(int proxyID, int curIndex, int bufferNum)
{
    int i, index, startIndex, endIndex;
    MPI_Request request[VOCL_WRITE_BUFFER_NUM];
    MPI_Status status[VOCL_WRITE_BUFFER_NUM];
    int requestNo;

    if (bufferNum > VOCL_WRITE_BUFFER_NUM) {
        bufferNum = VOCL_WRITE_BUFFER_NUM;
    }

    startIndex = (curIndex - bufferNum + 1);
    endIndex = curIndex;
    if (startIndex < 0) {
        startIndex += VOCL_WRITE_BUFFER_NUM;
        endIndex += VOCL_WRITE_BUFFER_NUM;
    }

    requestNo = 0;
    for (i = startIndex; i <= endIndex; i++) {
        index = i % VOCL_WRITE_BUFFER_NUM;
        request[requestNo++] = *getWriteRequestPtr(proxyID, index);
    }
    MPI_Waitall(requestNo, request, status);

    for (i = startIndex; i <= endIndex; i++) {
        index = i % VOCL_WRITE_BUFFER_NUM;
        voclWriteBufferPtr[proxyID].voclWriteBufferInfo[index].isInUse = 0;
    }

    return;
}

void processAllWrites(int proxyID)
{
    int i, index, startIndex, endIndex;
    MPI_Request request[VOCL_WRITE_BUFFER_NUM];
    MPI_Status status[VOCL_WRITE_BUFFER_NUM];
    int requestNo;

    endIndex = voclWriteBufferPtr[proxyID].curWriteBufferIndex;
    startIndex = endIndex - voclWriteBufferPtr[proxyID].writeDataRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_WRITE_BUFFER_NUM;
        endIndex += VOCL_WRITE_BUFFER_NUM;
    }

    requestNo = 0;
    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_WRITE_BUFFER_NUM;
        if (voclWriteBufferPtr[proxyID].voclWriteBufferInfo[index].isInUse == 1) {
            request[requestNo++] = *getWriteRequestPtr(proxyID, index);
        }
    }

    if (requestNo > 0) {
        MPI_Waitall(requestNo, request, status);
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_WRITE_BUFFER_NUM;
        voclWriteBufferPtr[proxyID].voclWriteBufferInfo[index].isInUse = 0;
    }

    voclWriteBufferPtr[proxyID].curWriteBufferIndex = 0;
    voclWriteBufferPtr[proxyID].writeDataRequestNum = 0;

    return;
}
