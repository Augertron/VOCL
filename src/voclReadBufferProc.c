#include <stdio.h>
#include "voclOpenclMacro.h"
#include "voclStructures.h"

static struct voclReadBuffer *voclReadBufferPtr;
static int voclReadBufferNum;
static void initializeReadBuffer(int proxyIndex)
{
    int i = 0;
    for (i = 0; i < VOCL_READ_BUFFER_NUM; i++) {
        voclReadBufferPtr[proxyIndex].voclReadBufferInfo[i].isInUse = 0;
		voclReadBufferPtr[proxyIndex].voclReadBufferInfo[i].event = VOCL_EVENT_NULL;
		voclReadBufferPtr[proxyIndex].voclReadBufferInfo[i].bufferNum = 0;
		voclReadBufferPtr[proxyIndex].voclReadBufferInfo[i].ptr = NULL;
		voclReadBufferPtr[proxyIndex].voclReadBufferInfo[i].size = 0;
		voclReadBufferPtr[proxyIndex].voclReadBufferInfo[i].tag = -1;
    }

    voclReadBufferPtr[proxyIndex].curReadBufferIndex = 0;
    voclReadBufferPtr[proxyIndex].readDataRequestNum = 0;

    return;
}

void initializeVoclReadBufferAll()
{
    int i;
    voclReadBufferNum = VOCL_BUFF_NUM;
    voclReadBufferPtr =
        (struct voclReadBuffer *) malloc(sizeof(struct voclReadBuffer) * voclReadBufferNum);
    for (i = 0; i < voclReadBufferNum; i++) {
        initializeReadBuffer(i);
    }
}

void finalizeVoclReadBufferAll()
{
    if (voclReadBufferPtr != NULL) {
        free(voclReadBufferPtr);
        voclReadBufferPtr = NULL;
    }

    return;
}

static void reallocateReadBuffer(int origBufferNum, int newBufferNum)
{
    int i;
    voclReadBufferPtr =
        (struct voclReadBuffer *) realloc(voclReadBufferPtr,
                                          sizeof(struct voclReadBuffer) * newBufferNum);
    for (i = origBufferNum; i < newBufferNum; i++) {
        initializeReadBuffer(i);
    }

    return;
}

void setReadBufferMPICommInfo(int proxyIndex, int index, 
		void *ptr, size_t size, int tag)
{
	voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].ptr = ptr;
	voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].size = size;
	voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].tag = tag;

	return;
}

void setReadBufferInUse(int proxyIndex, int index)
{
    voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].isInUse = 1;
}

void setReadBufferEvent(int proxyIndex, int index, vocl_event event)
{
    voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].event = event;
}

void setReadBufferNum(int proxyIndex, int index, int bufferNum)
{
    voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].bufferNum = bufferNum;
}

int getReadBufferNum(int proxyIndex, int index)
{
    return voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].bufferNum;
}

int getReadBufferIndexFromEvent(int proxyIndex, vocl_event event)
{
    int index;
    for (index = 0; index < voclReadBufferPtr[proxyIndex].readDataRequestNum; index++) {
        if (event == voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].event) {
            return index;
        }
    }
    return -1;
}

MPI_Request *getReadRequestPtr(int proxyIndex, int index)
{
    return &voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].request;
}

int getNextReadBufferIndex(int proxyIndex)
{
    if (proxyIndex >= voclReadBufferNum) {
        reallocateReadBuffer(voclReadBufferNum, 2 * voclReadBufferNum);
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
        voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].event = VOCL_EVENT_NULL;
        voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].bufferNum = 0;
        voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].ptr = NULL;
        voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].size = 0;
        voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].tag = -1;
    }

    return;
}

void reissueReadBufferRequest(int proxyIndex, int reissueNum, 
			MPI_Comm destCommData, int destProxyRank, int destProxyIndex)
{
	int i, index, startIndex, endIndex;
	int bufferIndex;
	MPI_Status status;
	int completedNum, completedThreshold;
	struct strReadBufferInfo *origReadBufPtr, *destReadBufPtr;

	completedNum = voclReadBufferPtr[proxyIndex].readDataRequestNum - reissueNum;

	if (destProxyIndex != proxyIndex)
	{
		endIndex = voclReadBufferPtr[proxyIndex].curReadBufferIndex;
		startIndex = endIndex - voclReadBufferPtr[proxyIndex].readDataRequestNum;
		if (startIndex < 0) {
			startIndex += VOCL_READ_BUFFER_NUM;
			endIndex += VOCL_READ_BUFFER_NUM;
		}
		completedThreshold = startIndex + completedNum;

		/* reissue read and write request */
		for (i = startIndex; i < endIndex; i++)
		{
			index = i % VOCL_READ_BUFFER_NUM;
			origReadBufPtr = &voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index];
			if (i < completedThreshold)
			{
				if (origReadBufPtr->isInUse == 1)
				{
					MPI_Wait(&origReadBufPtr->request, &status);
				}
			}
			else
			{
				/* cancel previous issued issued requests */
				MPI_Cancel(&origReadBufPtr->request);
				MPI_Request_free(&origReadBufPtr->request);

				/* get available buffer on the dest proxy process */
				bufferIndex = getNextReadBufferIndex(destProxyIndex);
				destReadBufPtr = &voclReadBufferPtr[destProxyIndex].voclReadBufferInfo[bufferIndex];
				destReadBufPtr->ptr = origReadBufPtr->ptr;
				destReadBufPtr->size = origReadBufPtr->size;
				destReadBufPtr->tag = VOCL_READ_TAG + bufferIndex;
				destReadBufPtr->isInUse = 1;
				MPI_Irecv(destReadBufPtr->ptr, destReadBufPtr->size, MPI_BYTE, destProxyRank,
						destReadBufPtr->tag, destCommData, &destReadBufPtr->request);
			}
            origReadBufPtr->isInUse = 0;
            origReadBufPtr->event = VOCL_EVENT_NULL;
            origReadBufPtr->bufferNum = 0;
            origReadBufPtr->ptr = NULL;
            origReadBufPtr->size = 0;
            origReadBufPtr->tag = -1;
		}

//		voclReadBufferPtr[proxyIndex].curReadBufferIndex -= reissueNum;
//		if (voclReadBufferPtr[proxyIndex].curReadBufferIndex < 0) {
//			voclReadBufferPtr[proxyIndex].curReadBufferIndex += VOCL_READ_BUFFER_NUM;
//		}
//		voclReadBufferPtr[proxyIndex].readDataRequestNum -= reissueNum;
		voclReadBufferPtr[proxyIndex].curReadBufferIndex = 0;
		voclReadBufferPtr[proxyIndex].readDataRequestNum = 0;
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
        voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].event = VOCL_EVENT_NULL;
        voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].bufferNum = 0;
        voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].ptr = NULL;
        voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].size = 0;
        voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].tag = -1;
    }

    voclReadBufferPtr[proxyIndex].curReadBufferIndex = 0;
    voclReadBufferPtr[proxyIndex].readDataRequestNum = 0;

    return;
}

void checkReadBufferStatus(int proxyIndex)
{    
	int i, index, startIndex, endIndex;
    MPI_Request request;
    MPI_Status status;
    int flag;

    endIndex = voclReadBufferPtr[proxyIndex].curReadBufferIndex;
    startIndex = endIndex - voclReadBufferPtr[proxyIndex].readDataRequestNum;
    if (startIndex < 0) {
        startIndex += VOCL_READ_BUFFER_NUM;
        endIndex += VOCL_READ_BUFFER_NUM;
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_READ_BUFFER_NUM;
		MPI_Test(&voclReadBufferPtr[proxyIndex].voclReadBufferInfo[index].request, &flag, &status);
    }

    return;
}
