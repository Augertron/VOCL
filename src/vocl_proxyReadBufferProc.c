#define _GNU_SOURCE
#include <CL/opencl.h>
#include <mpi.h>
#include <sched.h>
#include <pthread.h>
#include "vocl_proxy.h"
#include "vocl_proxy_macro.h"
#include "vocl_proxyBufferProc.h"

/* declared in the helper thread file for barrier */
/* synchronization between the two threads */
extern pthread_barrier_t barrier;
extern int helperThreadOperFlag;
extern int voclProxyAppIndex;

/*variables declared for read buffer pool */
//int readDataRequestNum = 0;
//int curReadBufferIndex;
//static struct strReadBufferInfo readBufferInfoPtr[VOCL_PROXY_READ_BUFFER_NUM];

static struct voclReadBufferInfo *voclProxyReadBufferPtr = NULL;
static int voclProxyReadSupportAppNum;
cl_int processAllReads(int rank);

/* for sending data from GPU to local node */
static void initializeReadBuffer(int rank)
{
    int i;
    for (i = 0; i < VOCL_PROXY_READ_BUFFER_NUM; i++) {
        voclProxyReadBufferPtr[rank].readBufferInfo[i].isInUse = READ_AVAILABLE;
        voclProxyReadBufferPtr[rank].readBufferInfo[i].dataPtr =
            (char *) malloc(VOCL_PROXY_READ_BUFFER_SIZE);
        voclProxyReadBufferPtr[rank].readBufferInfo[i].numReadBuffers = 0;
    }
    voclProxyReadBufferPtr[rank].curReadBufferIndex = 0;
    voclProxyReadBufferPtr[rank].readDataRequestNum = 0;

    return;
}

static void reallocVOCLProxyReadBuffer(int origBufferNum, int newBufferNum)
{
    int i;
    voclProxyReadBufferPtr =
        (struct voclReadBufferInfo *) realloc(voclProxyReadBufferPtr,
                                              sizeof(struct voclReadBufferInfo) *
                                              newBufferNum);
    for (i = origBufferNum; i < newBufferNum; i++) {
        initializeReadBuffer(i);
    }
}


void initializeReadBufferAll()
{
    int i;
    voclProxyReadSupportAppNum = VOCL_PROXY_APP_NUM;
    voclProxyReadBufferPtr =
        (struct voclReadBufferInfo *) malloc(sizeof(struct voclReadBufferInfo) * voclProxyReadSupportAppNum);
    for (i = 0; i < voclProxyReadSupportAppNum; i++) {
        initializeReadBuffer(i);
    }
}

void finalizeReadBufferAll()
{
    int rank, i;
    for (rank = 0; rank < voclProxyReadSupportAppNum; rank++) {
        for (i = 0; i < VOCL_PROXY_READ_BUFFER_NUM; i++) {
            if (voclProxyReadBufferPtr[rank].readBufferInfo[i].dataPtr) {
                free(voclProxyReadBufferPtr[rank].readBufferInfo[i].dataPtr);
                voclProxyReadBufferPtr[rank].readBufferInfo[i].dataPtr = NULL;
            }
        }
    }

    return;
}

MPI_Request *getReadRequestPtr(int rank, int index)
{
    return &voclProxyReadBufferPtr[rank].readBufferInfo[index].request;
}

struct strReadBufferInfo *getReadBufferInfoPtr(int rank, int index)
{
    return &voclProxyReadBufferPtr[rank].readBufferInfo[index];
}

struct voclReadBufferInfo *getVOCLReadBufferInfoPtr(int rank)
{
    return &voclProxyReadBufferPtr[rank];
}

int readSendToLocal(int rank, int index)
{
    int err;
    err = MPI_Isend(voclProxyReadBufferPtr[rank].readBufferInfo[index].dataPtr,
                    voclProxyReadBufferPtr[rank].readBufferInfo[index].size,
                    MPI_BYTE,
                    voclProxyReadBufferPtr[rank].readBufferInfo[index].dest,
                    voclProxyReadBufferPtr[rank].readBufferInfo[index].tag,
                    voclProxyReadBufferPtr[rank].readBufferInfo[index].comm,
                    getReadRequestPtr(rank, index));

    return err;
}

void setReadBufferFlag(int rank, int index, int flag)
{
    voclProxyReadBufferPtr[rank].readBufferInfo[index].isInUse = flag;
    return;
}

/* check if any buffer is used */
static int isReadBufferInUse(int rank)
{
    int i;
    for (i = 0; i < VOCL_PROXY_READ_BUFFER_NUM; i++) {
        if (voclProxyReadBufferPtr[rank].readBufferInfo[i].isInUse != READ_AVAILABLE) {
            /* some buffer is in use */
            return 1;
        }
    }

    /* no buffer is in use */
    return 0;
}

int getNextReadBufferIndex(int rank)
{
    int index;
    MPI_Status status;

    if (rank >= voclProxyReadSupportAppNum) {
        reallocVOCLProxyReadBuffer(voclProxyReadSupportAppNum, 2 * voclProxyReadSupportAppNum);
        voclProxyReadSupportAppNum *= 2;
    }

    index = voclProxyReadBufferPtr[rank].curReadBufferIndex;

    /* check if any buffer is used */
    if (index == 0 && isReadBufferInUse(rank)) {
        processAllReads(rank);
    }

    if (++voclProxyReadBufferPtr[rank].curReadBufferIndex >= VOCL_PROXY_READ_BUFFER_NUM) {
        voclProxyReadBufferPtr[rank].curReadBufferIndex = 0;
    }

    if (++voclProxyReadBufferPtr[rank].readDataRequestNum >= VOCL_PROXY_READ_BUFFER_NUM) {
        voclProxyReadBufferPtr[rank].readDataRequestNum = VOCL_PROXY_READ_BUFFER_NUM;
    }

    return index;
}


cl_int processReadBuffer(int rank, int curIndex, int bufferNum)
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
        if (voclProxyReadBufferPtr[rank].readBufferInfo[index].isInUse == READ_GPU_MEM ||
            voclProxyReadBufferPtr[rank].readBufferInfo[index].isInUse == READ_GPU_MEM_SUB) {
            err = readSendToLocal(rank, index);
            MPI_Wait(getReadRequestPtr(rank, index), status);
            setReadBufferFlag(rank, index, READ_AVAILABLE);
            voclProxyReadBufferPtr[rank].readBufferInfo[index].event = NULL;
        }
    }

    return err;
}

int getReadBufferIndexFromEvent(int rank, cl_event event)
{
    int index;
    for (index = 0; index < voclProxyReadBufferPtr[rank].readDataRequestNum; index++) {
        if (event == voclProxyReadBufferPtr[rank].readBufferInfo[index].event) {
            return index;
        }
    }

    return -1;
}

void thrSentToLocalNode(int rank)
{
    int i, index, startIndex, endIndex;
    MPI_Status status[VOCL_PROXY_READ_BUFFER_NUM];
    MPI_Request request[VOCL_PROXY_READ_BUFFER_NUM];
    int requestNo;
    int err = MPI_SUCCESS;

    endIndex = voclProxyReadBufferPtr[rank].curReadBufferIndex;
    startIndex = endIndex - voclProxyReadBufferPtr[rank].readDataRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_PROXY_READ_BUFFER_NUM;
        endIndex += VOCL_PROXY_READ_BUFFER_NUM;
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_PROXY_READ_BUFFER_NUM;
        if (voclProxyReadBufferPtr[rank].readBufferInfo[index].isInUse == READ_GPU_MEM_COMP) {
            err = readSendToLocal(rank, index);
            if (err != MPI_SUCCESS) {
                printf("mpi send error, %d\n", err);
            }
            MPI_Wait(getReadRequestPtr(rank, index), status);
        }
        else if (voclProxyReadBufferPtr[rank].readBufferInfo[index].isInUse == READ_SEND_DATA) {
            MPI_Wait(getReadRequestPtr(rank, index), status);
        }
        pthread_barrier_wait(&barrier);
    }

    return;
}

cl_int processAllReads(int rank)
{
    int i, index, startIndex, endIndex;
    MPI_Status status[VOCL_PROXY_READ_BUFFER_NUM];
    MPI_Request request[VOCL_PROXY_READ_BUFFER_NUM];
    int requestNo;
    int err = MPI_SUCCESS;

    /* check if any buffer is in use */
    if (!isReadBufferInUse(rank)) {
        return err;
    }

    endIndex = voclProxyReadBufferPtr[rank].curReadBufferIndex;
    startIndex = endIndex - voclProxyReadBufferPtr[rank].readDataRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_PROXY_READ_BUFFER_NUM;
        endIndex += VOCL_PROXY_READ_BUFFER_NUM;
    }

    pthread_barrier_wait(&barrier);
    helperThreadOperFlag = GPU_MEM_READ;
    /* used by the helper thread */
    voclProxyAppIndex = rank;

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_PROXY_READ_BUFFER_NUM;
        if (voclProxyReadBufferPtr[rank].readBufferInfo[index].isInUse == READ_GPU_MEM ||
            voclProxyReadBufferPtr[rank].readBufferInfo[index].isInUse == READ_GPU_MEM_SUB) {
            clWaitForEvents(1, &voclProxyReadBufferPtr[rank].readBufferInfo[index].event);
            setReadBufferFlag(rank, index, READ_GPU_MEM_COMP);
        }
        pthread_barrier_wait(&barrier);
    }
    pthread_barrier_wait(&barrier);

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_PROXY_READ_BUFFER_NUM;
        voclProxyReadBufferPtr[rank].readBufferInfo[index].numReadBuffers = 0;
        setReadBufferFlag(rank, index, READ_AVAILABLE);
    }
    voclProxyReadBufferPtr[rank].curReadBufferIndex = 0;
    voclProxyReadBufferPtr[rank].readDataRequestNum = 0;

    return err;
}
