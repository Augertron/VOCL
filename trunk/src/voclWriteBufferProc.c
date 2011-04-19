#include "mpi.h"
#include "vocl.h"

/* for sending data from local node to GPU */
struct strWriteBufferInfo {
    int isInUse;
    MPI_Request request;
};

static struct strWriteBufferInfo voclWriteBufferInfo[VOCL_WRITE_BUFFER_NUM];
static int writeDataRequestNum;
static int curWriteBufferIndex;

void initializeWriteBuffer()
{
    int i = 0;
    for (i = 0; i < VOCL_WRITE_BUFFER_NUM; i++) {
        voclWriteBufferInfo[i].isInUse = 0;
    }
    curWriteBufferIndex = 0;
    writeDataRequestNum = 0;

    return;
}

void setWriteBufferInUse(int index)
{
    voclWriteBufferInfo[index].isInUse = 1;
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
    }

    curWriteBufferIndex = 0;
    writeDataRequestNum = 0;

    return;
}
