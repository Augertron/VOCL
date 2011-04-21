#include "vocl_opencl.h"
#include "voclEventProc.h"

/* for receiving data from GPU to local node */
struct strReadBufferInfo {
    int isInUse;
	int readBufferNum;
    MPI_Request request;
	vocl_event event;
};

static int readDataRequestNum;
static struct strReadBufferInfo voclReadBufferInfo[VOCL_READ_BUFFER_NUM];
/* static MPI_Request readRequest[VOCL_READ_BUFFER_NUM]; */
static int curReadBufferIndex;

void initializeReadBuffer()
{
    int i = 0;
    for (i = 0; i < VOCL_READ_BUFFER_NUM; i++) {
        voclReadBufferInfo[i].isInUse = 0;
		voclReadBufferInfo[i].readBufferNum = 0;
		voclReadBufferInfo[i].event = -1; /*invalid vocl event */
    }
    curReadBufferIndex = 0;
    readDataRequestNum = 0;

    return;
}

void setReadBufferInUse(int index)
{
    voclReadBufferInfo[index].isInUse = 1;
}

void setReadBufferEvent(int index, vocl_event event)
{
	voclReadBufferInfo[index].event = event;
}

void setReadBuffers(int index, int bufferNum)
{
	voclReadBufferInfo[index].readBufferNum = bufferNum;
}

int getReadBuffers(int index)
{
	return voclReadBufferInfo[index].readBufferNum;
}

int getReadBufferIndexFromEvent(vocl_event event)
{
	int index;
	for (index = 0; index < readDataRequestNum; index++)
	{
		if (voclReadBufferInfo[index].event == event)
		{
			return index;
		}
	}

	return -1;
}

MPI_Request *getReadRequestPtr(int index)
{
    return &voclReadBufferInfo[index].request;
}

int getNextReadBufferIndex()
{
    int index = curReadBufferIndex;
    MPI_Status status;

    if (voclReadBufferInfo[curReadBufferIndex].isInUse == 1) {
        MPI_Wait(getReadRequestPtr(curReadBufferIndex), &status);
        voclReadBufferInfo[curReadBufferIndex].isInUse = 0;
    }

    if (++curReadBufferIndex >= VOCL_READ_BUFFER_NUM) {
        curReadBufferIndex = 0;
    }

    if (++readDataRequestNum >= VOCL_READ_BUFFER_NUM) {
        readDataRequestNum = VOCL_READ_BUFFER_NUM;
    }

    return index;
}

void processReadBuffer(int curIndex, int bufferNum)
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
        request[requestNo++] = *getReadRequestPtr(index);
    }
    MPI_Waitall(requestNo, request, status);
    for (i = startIndex; i <= endIndex; i++) {
        index = i % VOCL_READ_BUFFER_NUM;
        voclReadBufferInfo[index].isInUse = 0;
		voclReadBufferInfo[index].readBufferNum = 0;
    }

    return;
}

void processAllReads()
{
    int i, index, startIndex, endIndex;
    MPI_Request request[VOCL_READ_BUFFER_NUM];
    MPI_Status status[VOCL_READ_BUFFER_NUM];
    int requestNo;

    endIndex = curReadBufferIndex;
    startIndex = endIndex - readDataRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_READ_BUFFER_NUM;
        endIndex += VOCL_READ_BUFFER_NUM;
    }

    requestNo = 0;
    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_READ_BUFFER_NUM;
        if (voclReadBufferInfo[index].isInUse == 1) {
            request[requestNo++] = *getReadRequestPtr(index);
        }
    }

    if (requestNo > 0) {
        MPI_Waitall(requestNo, request, status);
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_READ_BUFFER_NUM;
        voclReadBufferInfo[index].isInUse = 0;
        voclReadBufferInfo[index].readBufferNum = 0;
    }

    curReadBufferIndex = 0;

    return;
}
