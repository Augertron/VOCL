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

/*variables declared for read buffer pool */
int readDataRequestNum = 0;
int curReadBufferIndex;
static struct strReadBufferInfo readBufferInfo[VOCL_PROXY_READ_BUFFER_NUM];

/* for sending data from GPU to local node */
void initializeReadBuffer()
{
    int i;
    for (i = 0; i < VOCL_PROXY_READ_BUFFER_NUM; i++) {
        readBufferInfo[i].isInUse = READ_AVAILABLE;
        readBufferInfo[i].dataPtr = (char *) malloc(VOCL_PROXY_READ_BUFFER_SIZE);
        readBufferInfo[i].numReadBuffers = 0;
    }
    curReadBufferIndex = 0;
    readDataRequestNum = 0;

    return;
}

void finalizeReadBuffer()
{
    int i;
    for (i = 0; i < VOCL_PROXY_READ_BUFFER_NUM; i++) {
        if (readBufferInfo[i].dataPtr)
            free(readBufferInfo[i].dataPtr);
    }

    return;
}

MPI_Request *getReadRequestPtr(int index)
{
    return &readBufferInfo[index].request;
}

struct strReadBufferInfo *getReadBufferInfoPtr(int index)
{
    return &readBufferInfo[index];
}

int readSendToLocal(int index)
{
    int err;
    err = MPI_Isend(readBufferInfo[index].dataPtr,
                    readBufferInfo[index].size,
                    MPI_BYTE,
                    0,
                    readBufferInfo[index].tag,
                    readBufferInfo[index].comm, getReadRequestPtr(index));

    return err;
}

void setReadBufferFlag(int index, int flag)
{
    readBufferInfo[index].isInUse = flag;
    return;
}

/* check if any buffer is used */
static int isReadBufferInUse()
{
    int i;
    for (i = 0; i < VOCL_PROXY_READ_BUFFER_NUM; i++) {
        if (readBufferInfo[i].isInUse != READ_AVAILABLE) {
            /* some buffer is in use */
            return 1;
        }
    }

    /* no buffer is in use */
    return 0;
}

int getNextReadBufferIndex()
{
    int index = curReadBufferIndex;
    MPI_Status status;

    /* check if any buffer is used */
    if (index == 0 && isReadBufferInUse()) {
        processAllReads();
    }

    if (++curReadBufferIndex >= VOCL_PROXY_READ_BUFFER_NUM) {
        curReadBufferIndex = 0;
    }

    if (++readDataRequestNum >= VOCL_PROXY_READ_BUFFER_NUM) {
        readDataRequestNum = VOCL_PROXY_READ_BUFFER_NUM;
    }

    return index;
}


cl_int processReadBuffer(int curIndex, int bufferNum)
{
    int i, index, startIndex, endIndex;
    MPI_Status status[VOCL_PROXY_READ_BUFFER_NUM];
    MPI_Request request[VOCL_PROXY_READ_BUFFER_NUM];

    int requestNo;
    cl_int err = MPI_SUCCESS;

    /* at most READ_BUFFER_NUM buffers to be processed */
    if (bufferNum > VOCL_PROXY_READ_BUFFER_NUM) {
        bufferNum = VOCL_PROXY_READ_BUFFER_NUM;
    }

    startIndex = curIndex - bufferNum + 1;
    endIndex = curIndex;
    if (startIndex < 0) {
        startIndex += VOCL_PROXY_READ_BUFFER_NUM;
        endIndex += VOCL_PROXY_READ_BUFFER_NUM;
    }

    for (i = startIndex; i <= endIndex; i++) {
        index = i % VOCL_PROXY_READ_BUFFER_NUM;
        if (readBufferInfo[index].isInUse == READ_GPU_MEM ||
            readBufferInfo[index].isInUse == READ_GPU_MEM_SUB) {
            err = readSendToLocal(index);
            MPI_Wait(getReadRequestPtr(index), status);
            setReadBufferFlag(index, READ_AVAILABLE);
            readBufferInfo[index].event = NULL;
        }
    }

    return err;
}

int getReadBufferIndexFromEvent(cl_event event)
{
    int index;
    for (index = 0; index < readDataRequestNum; index++) {
        if (event == readBufferInfo[index].event) {
            return index;
        }
    }

    return -1;
}

void thrSentToLocalNode(void *p)
{
    int i, index, startIndex, endIndex;
    MPI_Status status[VOCL_PROXY_READ_BUFFER_NUM];
    MPI_Request request[VOCL_PROXY_READ_BUFFER_NUM];
    int requestNo;
    int err = MPI_SUCCESS;

    endIndex = curReadBufferIndex;
    startIndex = endIndex - readDataRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_PROXY_READ_BUFFER_NUM;
        endIndex += VOCL_PROXY_READ_BUFFER_NUM;
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_PROXY_READ_BUFFER_NUM;
        if (readBufferInfo[index].isInUse == READ_GPU_MEM_COMP) {
            err = readSendToLocal(index);
            if (err != MPI_SUCCESS) {
                printf("mpi send error, %d\n", err);
            }
            MPI_Wait(getReadRequestPtr(index), status);
        }
        else if (readBufferInfo[index].isInUse == READ_SEND_DATA) {
            MPI_Wait(getReadRequestPtr(index), status);
        }
        pthread_barrier_wait(&barrier);
    }

    return;
}

cl_int processAllReads()
{
    int i, index, startIndex, endIndex;
    MPI_Status status[VOCL_PROXY_READ_BUFFER_NUM];
    MPI_Request request[VOCL_PROXY_READ_BUFFER_NUM];
    int requestNo;
    int err = MPI_SUCCESS;

    /* check if any buffer is in use */
    if (!isReadBufferInUse()) {
        return err;
    }

    endIndex = curReadBufferIndex;
    startIndex = endIndex - readDataRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_PROXY_READ_BUFFER_NUM;
        endIndex += VOCL_PROXY_READ_BUFFER_NUM;
    }

    pthread_barrier_wait(&barrier);
    helperThreadOperFlag = GPU_MEM_READ;

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_PROXY_READ_BUFFER_NUM;
        if (readBufferInfo[index].isInUse == READ_GPU_MEM ||
            readBufferInfo[index].isInUse == READ_GPU_MEM_SUB) {
            clWaitForEvents(1, &readBufferInfo[index].event);
            setReadBufferFlag(index, READ_GPU_MEM_COMP);
        }
        pthread_barrier_wait(&barrier);
    }
    pthread_barrier_wait(&barrier);

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_PROXY_READ_BUFFER_NUM;
        setReadBufferFlag(index, READ_AVAILABLE);
    }
    curReadBufferIndex = 0;
    readDataRequestNum = 0;

    return err;
}
