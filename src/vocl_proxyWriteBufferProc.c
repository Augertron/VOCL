#define _GNU_SOURCE
#include <CL/opencl.h>
#include <mpi.h>
#include <sched.h>
#include "vocl_proxy.h"
#include "vocl_proxy_macro.h"
#include "vocl_proxyBufferProc.h"

/* declared in the helper thread file for barrier */
/* synchronization between the two threads */
extern pthread_barrier_t barrier;
extern int helperThreadOperFlag;
extern int voclProxyAppIndex;

/*MPI request for control messages, the latter part is shared*/
/*by the Write data request */
//extern MPI_Request *conMsgRequest;

/*from read buffer pool */
//extern int curReadBufferIndex;
//extern int readDataRequestNum;
extern struct voclReadBufferInfo *voclReadBufferPtr;
extern struct strReadBufferInfo *getReadBufferInfoPtr(int rank, int index);
extern struct voclReadBufferInfo *getVOCLReadBufferInfoPtr(int rank);

/*------------------------------------------------ */
/* functions and variables defined in this file */
/*------------------------------------------------*/

/*
int writeDataRequestNum = 0;
//int totalRequestNum;
int allWritesAreEnqueuedFlag = 0;
int allReadBuffersAreCovered = 0;

static int curWriteBufferIndex;
static struct strWriteBufferInfo writeBufferInfo[VOCL_PROXY_WRITE_BUFFER_NUM];
*/

static struct voclWriteBufferInfo *voclProxyWriteBufferPtr = NULL;
static int voclProxySupportAppNum;

/* initialize write buffer pool */
static void initializeWriteBuffer(int index)
{
    int i;
    for (i = 0; i < VOCL_PROXY_WRITE_BUFFER_NUM; i++) {
        voclProxyWriteBufferPtr[index].writeBufferInfo[i].isInUse = WRITE_AVAILABLE;
        voclProxyWriteBufferPtr[index].writeBufferInfo[i].dataPtr = (char *)malloc(VOCL_PROXY_WRITE_BUFFER_SIZE);
        voclProxyWriteBufferPtr[index].writeBufferInfo[i].eventWaitList = NULL;
        voclProxyWriteBufferPtr[index].writeBufferInfo[i].numEvents = 0;
        voclProxyWriteBufferPtr[index].writeBufferInfo[i].sendProcInfo.toBeProcessedNum = 0;
    }

    voclProxyWriteBufferPtr[index].curWriteBufferIndex = 0;
    //voclProxyWriteBufferPtr[index].totalRequestNum = CMSG_NUM;
    voclProxyWriteBufferPtr[index].writeDataRequestNum = 0;
    voclProxyWriteBufferPtr[index].allWritesAreEnqueuedFlag = 1;
    voclProxyWriteBufferPtr[index].allReadBuffersAreCovered = 1;

    return;
}

static void reallocVOCLProxyWriteBuffer(int origBufferNum, int newBufferNum)
{
	int i;
	voclProxyWriteBufferPtr = (struct voclWriteBufferInfo *)realloc(voclProxyWriteBufferPtr, sizeof(struct voclWriteBufferInfo) * newBufferNum);
	for (i = origBufferNum; i < newBufferNum; i++)
	{
		initializeWriteBuffer(i);
	}
}

void initializeWriteBufferAll()
{
	int i;
	voclProxySupportAppNum = VOCL_PROXY_APP_NUM;
	voclProxyWriteBufferPtr = (struct voclWriteBufferInfo *)malloc(sizeof(struct voclWriteBufferInfo) * voclProxySupportAppNum);
	for (i = 0; i < voclProxySupportAppNum; i++)
	{
		initializeWriteBuffer(i);
	}
}

void increaseWriteBufferCount(int appRank)
{
    if (++voclProxyWriteBufferPtr[appRank].writeDataRequestNum > VOCL_PROXY_WRITE_BUFFER_NUM) {
        voclProxyWriteBufferPtr[appRank].writeDataRequestNum = VOCL_PROXY_WRITE_BUFFER_NUM;
    }

    return;
}

void finalizeWriteBufferAll()
{
    int rank, i;
	for (rank = 0; rank < voclProxySupportAppNum; rank++)
	{
		for (i = 0; i < VOCL_PROXY_WRITE_BUFFER_NUM; i++) {
			if (voclProxyWriteBufferPtr[rank].writeBufferInfo[i].dataPtr)
				free(voclProxyWriteBufferPtr[rank].writeBufferInfo[i].dataPtr);
			if (voclProxyWriteBufferPtr[rank].writeBufferInfo[i].eventWaitList)
				free(voclProxyWriteBufferPtr[rank].writeBufferInfo[i].eventWaitList);
		}
	}

	if (voclProxyWriteBufferPtr)
	{
		free(voclProxyWriteBufferPtr);
		voclProxyWriteBufferPtr = NULL;
	}

    return;
}

void setWriteBufferFlag(int appRank, int index, int flag)
{
    voclProxyWriteBufferPtr[appRank].writeBufferInfo[index].isInUse = flag;
    return;
}

void voclResetWriteEnqueueFlag(int rank)
{
	voclProxyWriteBufferPtr[rank].allWritesAreEnqueuedFlag = 0;
}

int voclGetWriteEnqueueFlag(int rank)
{
	return voclProxyWriteBufferPtr[rank].allWritesAreEnqueuedFlag;
}

void voclResetReadBufferCoveredFlag(int rank)
{
	voclProxyWriteBufferPtr[rank].allReadBuffersAreCovered = 0;
}

static void voclSetReadBufferCoveredFlag(int rank)
{
	voclProxyWriteBufferPtr[rank].allReadBuffersAreCovered = 1;
}

MPI_Request *getWriteRequestPtr(int appRank, int index)
{
    //return &writeDataRequest[index];
    return &voclProxyWriteBufferPtr[appRank].writeBufferInfo[index].request;
}

struct strWriteBufferInfo *getWriteBufferInfoPtr(int appRank, int index)
{
    return &voclProxyWriteBufferPtr[appRank].writeBufferInfo[index];
}

cl_int writeToGPUMemory(int appRank, int index)
{
    int err;
    err = clEnqueueWriteBuffer(voclProxyWriteBufferPtr[appRank].writeBufferInfo[index].commandQueue,
                               voclProxyWriteBufferPtr[appRank].writeBufferInfo[index].mem,
                               CL_FALSE,
                               voclProxyWriteBufferPtr[appRank].writeBufferInfo[index].offset,
                               voclProxyWriteBufferPtr[appRank].writeBufferInfo[index].size,
                               voclProxyWriteBufferPtr[appRank].writeBufferInfo[index].dataPtr,
                               voclProxyWriteBufferPtr[appRank].writeBufferInfo[index].numEvents,
                               voclProxyWriteBufferPtr[appRank].writeBufferInfo[index].eventWaitList,
                               &voclProxyWriteBufferPtr[appRank].writeBufferInfo[index].event);
    setWriteBufferFlag(appRank, index, WRITE_GPU_MEM);
    return err;
}

int getNextWriteBufferIndex(int rank)
{
	int index;
	MPI_Status status;

	if (rank >= voclProxySupportAppNum)
	{
		reallocVOCLProxyWriteBuffer(voclProxySupportAppNum, 2*voclProxySupportAppNum);
		voclProxySupportAppNum *= 2;
	}

    index = voclProxyWriteBufferPtr[rank].curWriteBufferIndex;

    /* process buffers in different states */
    if (voclProxyWriteBufferPtr[rank].writeBufferInfo[index].isInUse == WRITE_RECV_DATA) {
        MPI_Wait(getWriteRequestPtr(rank, index), &status);
        writeToGPUMemory(rank, index);
        clWaitForEvents(1, &voclProxyWriteBufferPtr[rank].writeBufferInfo[index].event);
        setWriteBufferFlag(rank, index, WRITE_AVAILABLE);
    }
    else if (voclProxyWriteBufferPtr[rank].writeBufferInfo[index].isInUse == WRITE_RECV_COMPLED) {
        writeToGPUMemory(rank, index);
        clWaitForEvents(1, &voclProxyWriteBufferPtr[rank].writeBufferInfo[index].event);
        setWriteBufferFlag(rank, index, WRITE_AVAILABLE);
    }
    else if (voclProxyWriteBufferPtr[rank].writeBufferInfo[index].isInUse == WRITE_GPU_MEM) {
        clWaitForEvents(1, &voclProxyWriteBufferPtr[rank].writeBufferInfo[index].event);
        setWriteBufferFlag(rank, index, WRITE_AVAILABLE);
    }

    /* mpisend ready data to local node, issue the Isend as soon as possible */
    if (voclProxyWriteBufferPtr[rank].writeBufferInfo[index].sendProcInfo.toBeProcessedNum > 0) {
        pthread_barrier_wait(&barrier);
        helperThreadOperFlag = SEND_LOCAL_PREVIOUS;
		/* used by the helper thread */
		voclProxyAppIndex = rank;
        pthread_barrier_wait(&barrier);
        pthread_barrier_wait(&barrier);
    }

    if (++voclProxyWriteBufferPtr[rank].curWriteBufferIndex >= VOCL_PROXY_WRITE_BUFFER_NUM) {
        voclProxyWriteBufferPtr[rank].curWriteBufferIndex = 0;
    }

    return index;
}

cl_int processWriteBuffer(int rank, int curIndex, int bufferNum)
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
        if (voclProxyWriteBufferPtr[rank].writeBufferInfo[index].isInUse == WRITE_RECV_DATA) {
            MPI_Wait(getWriteRequestPtr(rank, index), &tmpStatus);
            err = writeToGPUMemory(rank, index);
            setWriteBufferFlag(rank, index, WRITE_GPU_MEM);
        }
        else if (voclProxyWriteBufferPtr[rank].writeBufferInfo[index].isInUse == WRITE_GPU_MEM) {
            setWriteBufferFlag(rank, index, WRITE_AVAILABLE);
            voclProxyWriteBufferPtr[rank].writeBufferInfo[index].event = NULL;
        }
    }

    return err;
}

void thrWriteToGPUMemory(int rank)
{
    int i, index, startIndex, endIndex;
    MPI_Status tmpStatus;
    cl_event event[VOCL_PROXY_WRITE_BUFFER_NUM];
    int eventNo;
    cl_int err = CL_SUCCESS;

    endIndex = voclProxyWriteBufferPtr[rank].curWriteBufferIndex;
    startIndex = endIndex - voclProxyWriteBufferPtr[rank].writeDataRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_PROXY_WRITE_BUFFER_NUM;
        endIndex += VOCL_PROXY_WRITE_BUFFER_NUM;
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_PROXY_WRITE_BUFFER_NUM;
        if (voclProxyWriteBufferPtr[rank].writeBufferInfo[index].isInUse == WRITE_RECV_COMPLED) {
            err = writeToGPUMemory(rank, index);
            clWaitForEvents(1, &voclProxyWriteBufferPtr[rank].writeBufferInfo[index].event);
            setWriteBufferFlag(rank, index, WRITE_AVAILABLE);
        }
        else if (voclProxyWriteBufferPtr[rank].writeBufferInfo[index].isInUse == WRITE_GPU_MEM) {
            clWaitForEvents(1, &voclProxyWriteBufferPtr[rank].writeBufferInfo[index].event);
            setWriteBufferFlag(rank, index, WRITE_AVAILABLE);
        }
        pthread_barrier_wait(&barrier);
    }

    return;
}

/* check if any buffer is in use */
static int isWriteBufferInUse(int rank)
{
    int i;
    for (i = 0; i < VOCL_PROXY_WRITE_BUFFER_NUM; i++) {
        if (voclProxyWriteBufferPtr[rank].writeBufferInfo[i].isInUse != WRITE_AVAILABLE) {
            /* some buffer is in use */
            return 1;
        }
    }

    /* no buffer is in use */
    return 0;
}

cl_int processAllWrites(int rank)
{
    int i, index, startIndex, endIndex;
    MPI_Status tmpStatus;
    cl_event event[VOCL_PROXY_WRITE_BUFFER_NUM];
    int eventNo;
    cl_int err = CL_SUCCESS;

    if (!isWriteBufferInUse(rank)) {
        return err;
    }

    endIndex = voclProxyWriteBufferPtr[rank].curWriteBufferIndex;
    startIndex = endIndex - voclProxyWriteBufferPtr[rank].writeDataRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_PROXY_WRITE_BUFFER_NUM;
        endIndex += VOCL_PROXY_WRITE_BUFFER_NUM;
    }
    pthread_barrier_wait(&barrier);
    helperThreadOperFlag = GPU_MEM_WRITE;
	/* used in the helper thread */
	voclProxyAppIndex = rank;

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_PROXY_WRITE_BUFFER_NUM;
        if (voclProxyWriteBufferPtr[rank].writeBufferInfo[index].isInUse == WRITE_RECV_DATA) {
            MPI_Wait(getWriteRequestPtr(rank, index), &tmpStatus);
            setWriteBufferFlag(rank, index, WRITE_RECV_COMPLED);
        }
        pthread_barrier_wait(&barrier);
    }
    voclProxyWriteBufferPtr[rank].allWritesAreEnqueuedFlag = 1;
    pthread_barrier_wait(&barrier);

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_PROXY_WRITE_BUFFER_NUM;
        setWriteBufferFlag(rank, index, WRITE_AVAILABLE);
        voclProxyWriteBufferPtr[rank].writeBufferInfo[index].sendProcInfo.toBeProcessedNum = 0;
        voclProxyWriteBufferPtr[rank].writeBufferInfo[index].numWriteBuffers = 0;
    }

    voclProxyWriteBufferPtr[rank].curWriteBufferIndex = 0;
    voclProxyWriteBufferPtr[rank].writeDataRequestNum = 0;
    //voclProxyWriteBufferPtr[rank].totalRequestNum = CMSG_NUM;
    voclProxyWriteBufferPtr[rank].allReadBuffersAreCovered = 1;

    return err;
}

static void getAllPreviousReadBuffers(int rank, int writeIndex)
{
    int i, index, startIndex, endIndex;
    struct strReadBufferInfo *readBufferInfoPtr;
	struct voclReadBufferInfo *bufPtr;

	bufPtr = getVOCLReadBufferInfoPtr(rank);

    endIndex = bufPtr->curReadBufferIndex;
    startIndex = endIndex - bufPtr->readDataRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_PROXY_READ_BUFFER_NUM;
        endIndex += VOCL_PROXY_READ_BUFFER_NUM;
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_PROXY_READ_BUFFER_NUM;
        readBufferInfoPtr = getReadBufferInfoPtr(rank, index);
        if (readBufferInfoPtr->isInUse == READ_GPU_MEM) {
            voclProxyWriteBufferPtr[rank].writeBufferInfo[writeIndex].
                sendProcInfo.readBufferIndex[voclProxyWriteBufferPtr[rank].writeBufferInfo[writeIndex].sendProcInfo.
                                             toBeProcessedNum++] = index;
            setReadBufferFlag(rank, index, READ_GPU_MEM_SUB);
        }
    }

	voclSetReadBufferCoveredFlag(rank);
}



/* ensure all previous writes are in the command queue */
/* before the kernel is launched */
cl_int enqueuePreviousWrites(int rank)
{
    int i, index, startIndex, endIndex;
    MPI_Status tmpStatus;
    cl_event event[VOCL_PROXY_WRITE_BUFFER_NUM];
    int eventNo;
    int getPreviousFlag = 0;
    cl_int err = CL_SUCCESS;

    if (!isWriteBufferInUse(rank)) {
        pthread_barrier_wait(&barrier);
        return err;
    }

    endIndex = voclProxyWriteBufferPtr[rank].curWriteBufferIndex;
    startIndex = endIndex - voclProxyWriteBufferPtr[rank].writeDataRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_PROXY_WRITE_BUFFER_NUM;
        endIndex += VOCL_PROXY_WRITE_BUFFER_NUM;
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_PROXY_WRITE_BUFFER_NUM;

        if (voclProxyWriteBufferPtr[rank].writeBufferInfo[index].isInUse == WRITE_RECV_DATA) {
            if (getPreviousFlag == 0) {
                /* process all previous read buffer */
                if (voclProxyWriteBufferPtr[rank].allReadBuffersAreCovered == 0) {
                    getAllPreviousReadBuffers(rank, index);
                }
                getPreviousFlag = 1;
            }

            MPI_Wait(getWriteRequestPtr(rank, index), &tmpStatus);
            err = writeToGPUMemory(rank, index);
            setWriteBufferFlag(rank, index, WRITE_GPU_MEM);
        }
    }
    voclProxyWriteBufferPtr[rank].allWritesAreEnqueuedFlag = 1;
    pthread_barrier_wait(&barrier);

    return err;
}

int getWriteBufferIndexFromEvent(int rank, cl_event event)
{
    int index;
    for (index = 0; index < voclProxyWriteBufferPtr[rank].writeDataRequestNum; index++) {
        if (event == voclProxyWriteBufferPtr[rank].writeBufferInfo[index].event) {
            return index;
        }
    }

    return -1;
}

void sendReadyReadBufferToLocal(int rank)
{
    int i, index;
    struct strReadBufferInfo *readBufferInfoPtr;
    int writeIndex = voclProxyWriteBufferPtr[rank].curWriteBufferIndex;
    MPI_Status status;
    for (i = 0; i < voclProxyWriteBufferPtr[rank].writeBufferInfo[writeIndex].sendProcInfo.toBeProcessedNum; i++) {
        index = voclProxyWriteBufferPtr[rank].writeBufferInfo[writeIndex].sendProcInfo.readBufferIndex[i];
        readBufferInfoPtr = getReadBufferInfoPtr(rank, index);
        if (readBufferInfoPtr->isInUse == READ_GPU_MEM ||
            readBufferInfoPtr->isInUse == READ_GPU_MEM_SUB) {
            clWaitForEvents(1, &readBufferInfoPtr->event);
            readSendToLocal(rank, index);
            setReadBufferFlag(rank, index, READ_SEND_DATA);
        }
        else if (readBufferInfoPtr->isInUse == READ_GPU_MEM_COMP) {
            readSendToLocal(rank, index);
            setReadBufferFlag(rank, index, READ_SEND_DATA);
        }
    }
    voclProxyWriteBufferPtr[rank].writeBufferInfo[writeIndex].sendProcInfo.toBeProcessedNum = 0;
    pthread_barrier_wait(&barrier);
}
