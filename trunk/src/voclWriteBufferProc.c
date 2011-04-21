#include "vocl_opencl.h"
#include "voclEventProc.h"

/* for sending data from local node to GPU */
struct strWriteBufferInfo {
    int isInUse;
	int writeBufferNum;
    MPI_Request request;
	vocl_event event;
};

static struct strWriteBufferInfo voclWriteBufferInfo[VOCL_WRITE_BUFFER_NUM];
static int writeDataRequestNum;
static int curWriteBufferIndex;

void initializeWriteBuffer()
{
    int i = 0;
    for (i = 0; i < VOCL_WRITE_BUFFER_NUM; i++) {
        voclWriteBufferInfo[i].isInUse = 0;
        voclWriteBufferInfo[i].writeBufferNum = 0;
        voclWriteBufferInfo[i].event = -1;
    }
    curWriteBufferIndex = 0;
    writeDataRequestNum = 0;

    return;
}

void setWriteBufferInUse(int index)
{
    voclWriteBufferInfo[index].isInUse = 1;
}

void setWriteBufferEvent(int index, vocl_event event)
{
	voclWriteBufferInfo[index].event = event;
}

void setWriteBuffers(int index, int bufferNum)
{
	voclWriteBufferInfo[index].writeBufferNum = bufferNum;
}

int getWriteBuffers(int index)
{
	return voclWriteBufferInfo[index].writeBufferNum;
}

int getWriteBufferIndexFromEvent(vocl_event event)
{
	int index;
	for (index = 0; index < writeDataRequestNum; index++)
	{
		if (voclWriteBufferInfo[index].event == event)
		{
			return index;
		}
	}

	return -1;
}

MPI_Request *getWriteRequestPtr(int index)
{
    return &voclWriteBufferInfo[index].request;
}

int getNextWriteBufferIndex()
{
    int index = curWriteBufferIndex;
    MPI_Status status;

    if (voclWriteBufferInfo[curWriteBufferIndex].isInUse == 1) {
        MPI_Wait(getWriteRequestPtr(curWriteBufferIndex), &status);
        voclWriteBufferInfo[curWriteBufferIndex].isInUse = 0;
    }

    if (++curWriteBufferIndex >= VOCL_WRITE_BUFFER_NUM) {
        curWriteBufferIndex = 0;
    }

    if (++writeDataRequestNum >= VOCL_WRITE_BUFFER_NUM) {
        writeDataRequestNum = VOCL_WRITE_BUFFER_NUM;
    }

    return index;
}

void processWriteBuffer(int curIndex, int bufferNum)
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
        request[requestNo++] = *getWriteRequestPtr(index);
    }
    MPI_Waitall(requestNo, request, status);
    for (i = startIndex; i <= endIndex; i++) {
        index = i % VOCL_WRITE_BUFFER_NUM;
        voclWriteBufferInfo[index].isInUse = 0;
        voclWriteBufferInfo[index].writeBufferNum = 0;
    }

    return;
}

void processAllWrites()
{
    int i, index, startIndex, endIndex;
    MPI_Request request[VOCL_WRITE_BUFFER_NUM];
    MPI_Status status[VOCL_WRITE_BUFFER_NUM];
    int requestNo;

    endIndex = curWriteBufferIndex;
    startIndex = curWriteBufferIndex - writeDataRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_WRITE_BUFFER_NUM;
        endIndex += VOCL_WRITE_BUFFER_NUM;
    }

    requestNo = 0;
    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_WRITE_BUFFER_NUM;
        if (voclWriteBufferInfo[index].isInUse == 1) {
            request[requestNo++] = *getWriteRequestPtr(index);
        }
    }

    if (requestNo > 0) {
        MPI_Waitall(requestNo, request, status);
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_WRITE_BUFFER_NUM;
        voclWriteBufferInfo[index].isInUse = 0;
        voclWriteBufferInfo[index].writeBufferNum = 0;
    }

    curWriteBufferIndex = 0;
    writeDataRequestNum = 0;

    return;
}
