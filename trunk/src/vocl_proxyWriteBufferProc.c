#define _GNU_SOURCE
#include <CL/opencl.h>
#include <mpi.h>
#include <sched.h>
#include "vocl_proxy.h"
#include "vocl_proxyBufferProc.h"

/* declared in the helper thread file for barrier */
/* synchronization between the two threads */
extern pthread_barrier_t barrier;
extern int helperThreadOperFlag;

/*MPI request for control messages, the latter part is shared*/
/*by the Write data request */
extern MPI_Request *conMsgRequest;

/*from read buffer pool */
extern int curReadBufferIndex;
extern int readDataRequestNum;
extern struct strReadBufferInfo *getReadBufferInfoPtr(int index);

/*------------------------------------------------ */
/* functions and variables defined in this file */
/*------------------------------------------------*/

int writeDataRequestNum = 0;
int totalRequestNum;
int allWritesAreEnqueuedFlag = 0;
int allReadBuffersAreCovered = 0;

MPI_Request *writeDataRequest;
static int curWriteBufferIndex;
static struct strWriteBufferInfo writeBufferInfo[VOCL_PROXY_WRITE_BUFFER_NUM];

/* initialize write buffer pool */
void initializeWriteBuffer()
{
    int i;
    for (i = 0; i < VOCL_PROXY_WRITE_BUFFER_NUM; i++) {
        writeBufferInfo[i].isInUse = WRITE_AVAILABLE;
        writeBufferInfo[i].dataPtr = (char *) malloc(VOCL_PROXY_WRITE_BUFFER_SIZE);
        writeBufferInfo[i].eventWaitList = NULL;
        writeBufferInfo[i].numEvents = 0;
        writeBufferInfo[i].sendProcInfo.toBeProcessedNum = 0;
    }

    curWriteBufferIndex = 0;
    totalRequestNum = CMSG_NUM;
    writeDataRequestNum = 0;
    allWritesAreEnqueuedFlag = 1;
    writeDataRequest = conMsgRequest + CMSG_NUM;
    allReadBuffersAreCovered = 1;

    return;
}

void increaseWriteBufferCount()
{
    if (++writeDataRequestNum > VOCL_PROXY_WRITE_BUFFER_NUM) {
        writeDataRequestNum = VOCL_PROXY_WRITE_BUFFER_NUM;
    }
    /* totalRequestNum = CMSG_NUM + writeDataRequestNum; */
    totalRequestNum = CMSG_NUM;

    return;
}

void finalizeWriteBuffer()
{
    int i;
    for (i = 0; i < VOCL_PROXY_WRITE_BUFFER_NUM; i++) {
        if (writeBufferInfo[i].dataPtr)
            free(writeBufferInfo[i].dataPtr);
        if (writeBufferInfo[i].eventWaitList)
            free(writeBufferInfo[i].eventWaitList);
    }

    return;
}

void setWriteBufferFlag(int index, int flag)
{
    writeBufferInfo[index].isInUse = flag;
    return;
}

MPI_Request *getWriteRequestPtr(int index)
{
    return &writeDataRequest[index];
}

struct strWriteBufferInfo *getWriteBufferInfoPtr(int index)
{
    return &writeBufferInfo[index];
}

cl_int writeToGPUMemory(int index)
{
    int err;
    err = clEnqueueWriteBuffer(writeBufferInfo[index].commandQueue,
                               writeBufferInfo[index].mem,
                               CL_FALSE,
                               writeBufferInfo[index].offset,
                               writeBufferInfo[index].size,
                               writeBufferInfo[index].dataPtr,
                               writeBufferInfo[index].numEvents,
                               writeBufferInfo[index].eventWaitList,
                               &writeBufferInfo[index].event);
    setWriteBufferFlag(index, WRITE_GPU_MEM);
    return err;
}

int getNextWriteBufferIndex()
{
    int index = curWriteBufferIndex;
    MPI_Status status;

    /* process buffers in different states */
    if (writeBufferInfo[index].isInUse == WRITE_RECV_DATA) {
        MPI_Wait(getWriteRequestPtr(index), &status);
        writeToGPUMemory(index);
        clWaitForEvents(1, &writeBufferInfo[index].event);
        setWriteBufferFlag(index, WRITE_AVAILABLE);
    }
    else if (writeBufferInfo[index].isInUse == WRITE_RECV_COMPLED) {
        writeToGPUMemory(index);
        clWaitForEvents(1, &writeBufferInfo[index].event);
        setWriteBufferFlag(index, WRITE_AVAILABLE);
    }
    else if (writeBufferInfo[index].isInUse == WRITE_GPU_MEM) {
        clWaitForEvents(1, &writeBufferInfo[index].event);
        setWriteBufferFlag(index, WRITE_AVAILABLE);
    }

    /* mpisend ready data to local node, issue the Isend as soon as possible */
    if (writeBufferInfo[index].sendProcInfo.toBeProcessedNum > 0) {
        pthread_barrier_wait(&barrier);
        helperThreadOperFlag = SEND_LOCAL_PREVIOUS;
        pthread_barrier_wait(&barrier);
        pthread_barrier_wait(&barrier);
    }

    if (++curWriteBufferIndex >= VOCL_PROXY_WRITE_BUFFER_NUM) {
        curWriteBufferIndex = 0;
    }

    return index;
}

cl_int processWriteBuffer(int curIndex, int bufferNum)
{
    int i, index, startIndex, endIndex;
    MPI_Status tmpStatus;
    cl_event event[VOCL_PROXY_WRITE_BUFFER_NUM];
    int eventNo;
    cl_int err = CL_SUCCESS;

    /* at most VOCL_PROXY_WRITE_BUFFER_NUM buffers to be processed */
    if (bufferNum > VOCL_PROXY_WRITE_BUFFER_NUM) {
        bufferNum = VOCL_PROXY_WRITE_BUFFER_NUM;
    }

    startIndex = curIndex - bufferNum + 1;
    endIndex = curIndex;
    if (startIndex < 0) {
        startIndex += VOCL_PROXY_WRITE_BUFFER_NUM;
        endIndex += VOCL_PROXY_WRITE_BUFFER_NUM;
    }

    for (i = startIndex; i <= endIndex; i++) {
        index = i % VOCL_PROXY_WRITE_BUFFER_NUM;
        if (writeBufferInfo[index].isInUse == WRITE_RECV_DATA) {
            MPI_Wait(getWriteRequestPtr(index), &tmpStatus);
            err = writeToGPUMemory(index);
            setWriteBufferFlag(index, WRITE_GPU_MEM);
        }
        else if (writeBufferInfo[index].isInUse == WRITE_GPU_MEM) {
            setWriteBufferFlag(index, WRITE_AVAILABLE);
            writeBufferInfo[index].event = NULL;
        }
    }

    return err;
}

void thrWriteToGPUMemory(void *p)
{
    int i, index, startIndex, endIndex;
    MPI_Status tmpStatus;
    cl_event event[VOCL_PROXY_WRITE_BUFFER_NUM];
    int eventNo;
    cl_int err = CL_SUCCESS;

    endIndex = curWriteBufferIndex;
    startIndex = endIndex - writeDataRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_PROXY_WRITE_BUFFER_NUM;
        endIndex += VOCL_PROXY_WRITE_BUFFER_NUM;
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_PROXY_WRITE_BUFFER_NUM;
        if (writeBufferInfo[index].isInUse == WRITE_RECV_COMPLED) {
            err = writeToGPUMemory(index);
            clWaitForEvents(1, &writeBufferInfo[index].event);
            setWriteBufferFlag(index, WRITE_AVAILABLE);
        }
        else if (writeBufferInfo[index].isInUse == WRITE_GPU_MEM) {
            clWaitForEvents(1, &writeBufferInfo[index].event);
            setWriteBufferFlag(index, WRITE_AVAILABLE);
        }
        pthread_barrier_wait(&barrier);
    }

    return;
}

/* check if any buffer is in use */
static int isWriteBufferInUse()
{
    int i;
    for (i = 0; i < VOCL_PROXY_WRITE_BUFFER_NUM; i++) {
        if (writeBufferInfo[i].isInUse != WRITE_AVAILABLE) {
            /* some buffer is in use */
            return 1;
        }
    }

    /* no buffer is in use */
    return 0;
}

cl_int processAllWrites()
{
    int i, index, startIndex, endIndex;
    MPI_Status tmpStatus;
    cl_event event[VOCL_PROXY_WRITE_BUFFER_NUM];
    int eventNo;
    cl_int err = CL_SUCCESS;

    if (!isWriteBufferInUse()) {
        return err;
    }

    endIndex = curWriteBufferIndex;
    startIndex = endIndex - writeDataRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_PROXY_WRITE_BUFFER_NUM;
        endIndex += VOCL_PROXY_WRITE_BUFFER_NUM;
    }
    pthread_barrier_wait(&barrier);
    helperThreadOperFlag = GPU_MEM_WRITE;

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_PROXY_WRITE_BUFFER_NUM;
        if (writeBufferInfo[index].isInUse == WRITE_RECV_DATA) {
            MPI_Wait(getWriteRequestPtr(index), &tmpStatus);
            setWriteBufferFlag(index, WRITE_RECV_COMPLED);
        }
        pthread_barrier_wait(&barrier);
    }
    allWritesAreEnqueuedFlag = 1;
    pthread_barrier_wait(&barrier);

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_PROXY_WRITE_BUFFER_NUM;
        setWriteBufferFlag(index, WRITE_AVAILABLE);
        writeBufferInfo[index].sendProcInfo.toBeProcessedNum = 0;
		writeBufferInfo[index].numWriteBuffers = 0;
		writeBufferInfo[index].event = NULL;
    }

    curWriteBufferIndex = 0;
    writeDataRequestNum = 0;
    totalRequestNum = CMSG_NUM;
    allReadBuffersAreCovered = 1;

    return err;
}

static void getAllPreviousReadBuffers(int writeIndex)
{
    int i, index, startIndex, endIndex;
    struct strReadBufferInfo *readBufferInfoPtr;

    endIndex = curReadBufferIndex;
    startIndex = endIndex - readDataRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_PROXY_READ_BUFFER_NUM;
        endIndex += VOCL_PROXY_READ_BUFFER_NUM;
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_PROXY_READ_BUFFER_NUM;
        readBufferInfoPtr = getReadBufferInfoPtr(index);
        if (readBufferInfoPtr->isInUse == READ_GPU_MEM) {
            writeBufferInfo[writeIndex].sendProcInfo.
                readBufferIndex[writeBufferInfo[writeIndex].sendProcInfo.toBeProcessedNum++] =
                index;
            setReadBufferFlag(index, READ_GPU_MEM_SUB);
        }
    }

    allReadBuffersAreCovered = 1;
}



/* ensure all previous writes are in the command queue */
/* before the kernel is launched */
cl_int enqueuePreviousWrites()
{
    int i, index, startIndex, endIndex;
    MPI_Status tmpStatus;
    cl_event event[VOCL_PROXY_WRITE_BUFFER_NUM];
    int eventNo;
    int getPreviousFlag = 0;
    cl_int err = CL_SUCCESS;

    if (!isWriteBufferInUse()) {
        pthread_barrier_wait(&barrier);
        return err;
    }

    endIndex = curWriteBufferIndex;
    startIndex = endIndex - writeDataRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_PROXY_WRITE_BUFFER_NUM;
        endIndex += VOCL_PROXY_WRITE_BUFFER_NUM;
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_PROXY_WRITE_BUFFER_NUM;

        if (writeBufferInfo[index].isInUse == WRITE_RECV_DATA) {
            if (getPreviousFlag == 0) {
                /* process all previous read buffer */
                if (allReadBuffersAreCovered == 0) {
                    getAllPreviousReadBuffers(index);
                }
                getPreviousFlag = 1;
            }

            MPI_Wait(getWriteRequestPtr(index), &tmpStatus);
            err = writeToGPUMemory(index);
            setWriteBufferFlag(index, WRITE_GPU_MEM);
        }
    }
    allWritesAreEnqueuedFlag = 1;
    pthread_barrier_wait(&barrier);

    return err;
}

int getWriteBufferIndexFromEvent(cl_event event)
{
    int index;
    for (index = 0; index < writeDataRequestNum; index++) {
        if (event == writeBufferInfo[index].event) {
            return index;
        }
    }

    return -1;
}

void sendReadyReadBufferToLocal()
{
    int i, index;
    struct strReadBufferInfo *readBufferInfoPtr;
    int writeIndex = curWriteBufferIndex;
    MPI_Status status;
    for (i = 0; i < writeBufferInfo[writeIndex].sendProcInfo.toBeProcessedNum; i++) {
        index = writeBufferInfo[writeIndex].sendProcInfo.readBufferIndex[i];
        readBufferInfoPtr = getReadBufferInfoPtr(index);
        if (readBufferInfoPtr->isInUse == READ_GPU_MEM ||
            readBufferInfoPtr->isInUse == READ_GPU_MEM_SUB) {
            clWaitForEvents(1, &readBufferInfoPtr->event);
            readSendToLocal(index);
            setReadBufferFlag(index, READ_SEND_DATA);
        }
        else if (readBufferInfoPtr->isInUse == READ_GPU_MEM_COMP) {
            readSendToLocal(index);
            setReadBufferFlag(index, READ_SEND_DATA);
        }
    }
    writeBufferInfo[writeIndex].sendProcInfo.toBeProcessedNum = 0;
    pthread_barrier_wait(&barrier);
}
