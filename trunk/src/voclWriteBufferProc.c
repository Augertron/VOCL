#include "voclOpenclMacro.h"
#include "voclStructures.h"

static struct voclWriteBuffer *voclWriteBufferPtr = NULL;
static int voclWriteBufferNum;

static void initializeWriteBuffer(int proxyIndex)
{
    int i = 0;
    for (i = 0; i < VOCL_WRITE_BUFFER_NUM; i++) {
        voclWriteBufferPtr[proxyIndex].voclWriteBufferInfo[i].isInUse = 0;
        voclWriteBufferPtr[proxyIndex].voclWriteBufferInfo[i].event = VOCL_EVENT_NULL;
        voclWriteBufferPtr[proxyIndex].voclWriteBufferInfo[i].bufferNum = 0;
    }
    voclWriteBufferPtr[proxyIndex].curWriteBufferIndex = 0;
    voclWriteBufferPtr[proxyIndex].writeDataRequestNum = 0;

    return;
}

void initializeVoclWriteBufferAll()
{
    int i;
    voclWriteBufferNum = VOCL_BUFF_NUM;
    voclWriteBufferPtr =
        (struct voclWriteBuffer *) malloc(sizeof(struct voclWriteBuffer) * voclWriteBufferNum);
    for (i = 0; i < voclWriteBufferNum; i++) {
        initializeWriteBuffer(i);
    }

    return;
}

void finalizeVoclWriteBufferAll()
{
    if (voclWriteBufferPtr != NULL) {
        free(voclWriteBufferPtr);
        voclWriteBufferPtr = NULL;
    }

    return;
}

static void reallocateWriteBuffer(int origBufferNum, int newBufferNum)
{
    int i;
    voclWriteBufferPtr =
        (struct voclWriteBuffer *) malloc(sizeof(struct voclWriteBuffer) * newBufferNum);
    for (i = origBufferNum; i < newBufferNum; i++) {
        initializeWriteBuffer(i);
    }

    return;
}


void setWriteBufferInUse(int proxyIndex, int index)
{
    voclWriteBufferPtr[proxyIndex].voclWriteBufferInfo[index].isInUse = 1;
}

void setWriteBufferEvent(int proxyIndex, int index, vocl_event event)
{
    voclWriteBufferPtr[proxyIndex].voclWriteBufferInfo[index].event = event;
}

void setWriteBufferNum(int proxyIndex, int index, int bufferNum)
{
    voclWriteBufferPtr[proxyIndex].voclWriteBufferInfo[index].bufferNum = bufferNum;
}

int getWriteBufferNum(int proxyIndex, int index)
{
    return voclWriteBufferPtr[proxyIndex].voclWriteBufferInfo[index].bufferNum;
}

int getWriteBufferIndexFromEvent(int proxyIndex, vocl_event event)
{
    int index;
    for (index = 0; index < voclWriteBufferPtr[proxyIndex].writeDataRequestNum; index++) {
        if (event == voclWriteBufferPtr[proxyIndex].voclWriteBufferInfo[index].event) {
            return index;
        }
    }

    return -1;
}

MPI_Request *getWriteRequestPtr(int proxyIndex, int index)
{
    return &voclWriteBufferPtr[proxyIndex].voclWriteBufferInfo[index].request;
}

int getNextWriteBufferIndex(int proxyIndex)
{
    if (proxyIndex >= voclWriteBufferNum) {
        reallocateWriteBuffer(voclWriteBufferNum, 2 * voclWriteBufferNum);
        voclWriteBufferNum *= 2;
    }

    int index = voclWriteBufferPtr[proxyIndex].curWriteBufferIndex;
    MPI_Status status;

    if (voclWriteBufferPtr[proxyIndex].voclWriteBufferInfo[index].isInUse == 1) {
        MPI_Wait(getWriteRequestPtr(proxyIndex, index), &status);
        voclWriteBufferPtr[proxyIndex].voclWriteBufferInfo[index].isInUse = 0;
    }

    if (++voclWriteBufferPtr[proxyIndex].curWriteBufferIndex >= VOCL_WRITE_BUFFER_NUM) {
        voclWriteBufferPtr[proxyIndex].curWriteBufferIndex = 0;
    }

    if (++voclWriteBufferPtr[proxyIndex].writeDataRequestNum >= VOCL_WRITE_BUFFER_NUM) {
        voclWriteBufferPtr[proxyIndex].writeDataRequestNum = VOCL_WRITE_BUFFER_NUM;
    }

    return index;
}

void processWriteBuffer(int proxyIndex, int curIndex, int bufferNum)
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
        request[requestNo++] = *getWriteRequestPtr(proxyIndex, index);
    }
    MPI_Waitall(requestNo, request, status);

    for (i = startIndex; i <= endIndex; i++) {
        index = i % VOCL_WRITE_BUFFER_NUM;
        voclWriteBufferPtr[proxyIndex].voclWriteBufferInfo[index].isInUse = 0;
        voclWriteBufferPtr[proxyIndex].voclWriteBufferInfo[index].event = VOCL_EVENT_NULL;
        voclWriteBufferPtr[proxyIndex].voclWriteBufferInfo[index].bufferNum = 0;
    }

    return;
}

void processAllWrites(int proxyIndex)
{
    int i, index, startIndex, endIndex;
    MPI_Request request[VOCL_WRITE_BUFFER_NUM];
    MPI_Status status[VOCL_WRITE_BUFFER_NUM];
    int requestNo;

    endIndex = voclWriteBufferPtr[proxyIndex].curWriteBufferIndex;
    startIndex = endIndex - voclWriteBufferPtr[proxyIndex].writeDataRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_WRITE_BUFFER_NUM;
        endIndex += VOCL_WRITE_BUFFER_NUM;
    }

    requestNo = 0;
    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_WRITE_BUFFER_NUM;
        if (voclWriteBufferPtr[proxyIndex].voclWriteBufferInfo[index].isInUse == 1) {
            request[requestNo++] = *getWriteRequestPtr(proxyIndex, index);
        }
    }

    if (requestNo > 0) {
        MPI_Waitall(requestNo, request, status);
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_WRITE_BUFFER_NUM;
        voclWriteBufferPtr[proxyIndex].voclWriteBufferInfo[index].isInUse = 0;
        voclWriteBufferPtr[proxyIndex].voclWriteBufferInfo[index].event = VOCL_EVENT_NULL;
        voclWriteBufferPtr[proxyIndex].voclWriteBufferInfo[index].bufferNum = 0;
    }

    voclWriteBufferPtr[proxyIndex].curWriteBufferIndex = 0;
    voclWriteBufferPtr[proxyIndex].writeDataRequestNum = 0;

    return;
}
