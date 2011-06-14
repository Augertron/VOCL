#include <stdio.h>
#include <CL/opencl.h>
#include "mpi.h"
#include "voclOpenclMacro.h"
#include "voclMigration.h"

extern cl_int
dlCLEnqueueWriteBuffer(cl_command_queue command_queue,
                       cl_mem buffer,
                       cl_bool blocking_write,
                       size_t offset,
                       size_t cb,
                       const void *ptr,
                       cl_uint num_events_in_wait_list,
                       const cl_event * event_wait_list, cl_event * event);
extern cl_int dlCLWaitForEvents(cl_uint num_events, const cl_event * event_list);


/* ----------------------------Write Buffer-------------------------------*/
static struct strMigWTLocalBufferPool *voclMigLocalWTBufferPool = NULL;
static int voclMigLocalWriteBufferNum = 0;

static void voclMigWriteLocalBufferInitialize(int index)
{
    int i;
    for (i = 0; i < VOCL_MIG_BUF_NUM; i++) {
        voclMigLocalWTBufferPool[index].buffers[i].cmdQueue = NULL;
        voclMigLocalWTBufferPool[index].buffers[i].mem = NULL;
        voclMigLocalWTBufferPool[index].buffers[i].size = 0;
        voclMigLocalWTBufferPool[index].buffers[i].offset = 0;
        voclMigLocalWTBufferPool[index].buffers[i].event = NULL;
        voclMigLocalWTBufferPool[index].buffers[i].useFlag = VOCL_MIG_LOCAL_WT_BUF_AVALB;
        voclMigLocalWTBufferPool[index].buffers[i].ptr = (char *) malloc(VOCL_MIG_BUF_SIZE * sizeof(char));
    }

    voclMigLocalWTBufferPool[index].voclMigWriteLocalBufferIndex = 0;
    voclMigLocalWTBufferPool[index].voclMigWriteLocalBufferRstNum = 0;

    return;
}

static void voclMigReallocLocalWTBuffer(int oldBufferNum, int newBufferNum)
{
	int i;
	voclMigLocalWTBufferPool = (struct strMigWTLocalBufferPool*)realloc(voclMigLocalWTBufferPool, 
			sizeof(struct strMigWTLocalBufferPool)* newBufferNum);
	for (i = oldBufferNum; i < newBufferNum; i++)
	{
		voclMigWriteLocalBufferInitialize(i);
	}

	return;
}

void voclMigWriteLocalBufferInitializeAll()
{
	int i;
	voclMigLocalWriteBufferNum = VOCL_MIG_POOL_NUM;
	voclMigLocalWTBufferPool = (struct strMigWTLocalBufferPool*)malloc(sizeof(struct strMigWTLocalBufferPool)*
			voclMigLocalWriteBufferNum);
	for (i = 0; i < voclMigLocalWriteBufferNum; i++)
	{
		voclMigWriteLocalBufferInitialize(i);
	}

	return;
}

void voclMigWriteLocalBufferFinalize()
{
    int i, index;
	for (index = 0; index < voclMigLocalWriteBufferNum; index++)
	{
		for (i = 0; i < VOCL_MIG_BUF_NUM; i++) {
			voclMigLocalWTBufferPool[index].buffers[i].useFlag = VOCL_MIG_LOCAL_WT_BUF_AVALB;
			free(voclMigLocalWTBufferPool[index].buffers[i].ptr);
			voclMigLocalWTBufferPool[index].buffers[i].ptr = NULL;
		}

		voclMigLocalWTBufferPool[index].voclMigWriteLocalBufferIndex = 0;
		voclMigLocalWTBufferPool[index].voclMigWriteLocalBufferRstNum = 0;
	}

    free(voclMigLocalWTBufferPool);

    return;
}

static int voclMigWriteToGPUMem(int rank, int index)
{
    int err;
    err = dlCLEnqueueWriteBuffer(voclMigLocalWTBufferPool[rank].buffers[index].cmdQueue,
                                 voclMigLocalWTBufferPool[rank].buffers[index].mem,
                                 CL_FALSE,
                                 voclMigLocalWTBufferPool[rank].buffers[index].offset,
                                 voclMigLocalWTBufferPool[rank].buffers[index].size,
                                 voclMigLocalWTBufferPool[rank].buffers[index].ptr,
                                 0, NULL, &voclMigLocalWTBufferPool[rank].buffers[index].event);
    return err;
}

int voclMigGetNextLocalWriteBufferIndex(int rank)
{
    int index = voclMigLocalWTBufferPool[rank].voclMigWriteLocalBufferIndex;
    MPI_Status status;

	if (rank >= voclMigLocalWriteBufferNum)
	{
		voclMigReallocLocalWTBuffer(voclMigLocalWriteBufferNum, 2*voclMigLocalWriteBufferNum);
		voclMigLocalWriteBufferNum *= 2;
	}

    if (voclMigLocalWTBufferPool[rank].buffers[index].useFlag == VOCL_MIG_LOCAL_WT_BUF_WAITDATA) {
        MPI_Wait(&voclMigLocalWTBufferPool[rank].buffers[index].request, &status);
        voclMigWriteToGPUMem(rank, index);
        dlCLWaitForEvents(1, &voclMigLocalWTBufferPool[rank].buffers[index].event);
    }
    else if (voclMigLocalWTBufferPool[rank].buffers[index].useFlag == VOCL_MIG_LOCAL_WT_BUF_WTGPUMEM) {
        dlCLWaitForEvents(1, &voclMigLocalWTBufferPool[rank].buffers[index].event);
    }
    voclMigLocalWTBufferPool[rank].buffers[index].useFlag = VOCL_MIG_LOCAL_WT_BUF_AVALB;

    if (++voclMigLocalWTBufferPool[rank].voclMigWriteLocalBufferIndex >= VOCL_MIG_BUF_NUM) {
        voclMigLocalWTBufferPool[rank].voclMigWriteLocalBufferIndex = 0;
    }

    if (++voclMigLocalWTBufferPool[rank].voclMigWriteLocalBufferRstNum > VOCL_MIG_BUF_NUM) {
        voclMigLocalWTBufferPool[rank].voclMigWriteLocalBufferRstNum = VOCL_MIG_BUF_NUM;
    }

    return index;
}

void voclMigSetWriteBufferFlag(int rank, int index, int flag)
{
    voclMigLocalWTBufferPool[rank].buffers[index].useFlag = flag;
    return;
}

MPI_Request *voclMigGetWriteBufferRequestPtr(int rank, int index)
{
    return &voclMigLocalWTBufferPool[rank].buffers[index].request;
}

cl_event *voclMigGetWriteBufferEventPtr(int rank, int index)
{
    return &voclMigLocalWTBufferPool[rank].buffers[index].event;
}

struct strMigWriteLocalBuffer *voclMigGetWriteBufferInfoPtr(int rank, int index)
{
    return &voclMigLocalWTBufferPool[rank].buffers[index];
}

int voclMigFinishLocalDataWrite(int rank, MPI_Comm comm)
{
    int i, index, err;
    int startIndex, endIndex;
    cl_event eventList[VOCL_MIG_BUF_NUM];
//    int activeBufferFlag[VOCL_MIG_BUF_NUM];
    int eventNo = 0;
    MPI_Status status;

    endIndex = voclMigLocalWTBufferPool[rank].voclMigWriteLocalBufferIndex;
    startIndex = endIndex - voclMigLocalWTBufferPool[rank].voclMigWriteLocalBufferRstNum;
    if (startIndex < 0) {
        startIndex += VOCL_MIG_BUF_NUM;
        endIndex += VOCL_MIG_BUF_NUM;
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_MIG_BUF_NUM;
            if (voclMigLocalWTBufferPool[rank].buffers[index].useFlag == VOCL_MIG_LOCAL_WT_BUF_WAITDATA) {
                MPI_Wait(&voclMigLocalWTBufferPool[rank].buffers[index].request, &status);
                voclMigWriteToGPUMem(rank, index);
                eventList[eventNo++] = voclMigLocalWTBufferPool[rank].buffers[index].event;
            }
            else if (voclMigLocalWTBufferPool[rank].buffers[index].useFlag == VOCL_MIG_LOCAL_WT_BUF_WTGPUMEM) {
                eventList[eventNo++] = voclMigLocalWTBufferPool[rank].buffers[index].event;
            }

    }

    if (eventNo > 0) {
        err = dlCLWaitForEvents(eventNo, eventList);
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_MIG_BUF_NUM;
            voclMigLocalWTBufferPool[rank].buffers[index].useFlag = VOCL_MIG_LOCAL_WT_BUF_AVALB;
            voclMigLocalWTBufferPool[rank].buffers[index].comm = -1;
    }

    return err;
}

/* ------------------ read buffer ------------------------------------------------*/
static struct strMigRDLocalBufferPool *voclMigLocalRDBufferPool = NULL;
static int voclMigLocalReadBufferNum = 0;

static void voclMigReadLocalBufferInitialize(int index)
{
    int i;
    for (i = 0; i < VOCL_MIG_BUF_NUM; i++) {
        voclMigLocalRDBufferPool[index].buffers[i].dest = -1;
        voclMigLocalRDBufferPool[index].buffers[i].tag = -1;
        voclMigLocalRDBufferPool[index].buffers[i].size = 0;
        voclMigLocalRDBufferPool[index].buffers[i].offset = 0;
        voclMigLocalRDBufferPool[index].buffers[i].event = NULL;
        voclMigLocalRDBufferPool[index].buffers[i].useFlag = VOCL_MIG_LOCAL_RD_BUF_AVALB;
        voclMigLocalRDBufferPool[index].buffers[i].ptr = (char *) malloc(VOCL_MIG_BUF_SIZE * sizeof(char));
    }

    voclMigLocalRDBufferPool[index].voclMigReadLocalBufferIndex = 0;
    voclMigLocalRDBufferPool[index].voclMigReadLocalBufferRstNum = 0;

    return;
}

static void voclMigReallocLocalRDBuffer(int oldBufferNum, int newBufferNum)
{
	int i;
	voclMigLocalRDBufferPool = (struct strMigRDLocalBufferPool*)realloc(voclMigLocalRDBufferPool, 
			sizeof(struct strMigRDLocalBufferPool)* newBufferNum);
	for (i = oldBufferNum; i < newBufferNum; i++)
	{
		voclMigReadLocalBufferInitialize(i);
	}

	return;
}

void voclMigReadLocalBufferInitializeAll()
{
	int i;
	voclMigLocalReadBufferNum = VOCL_MIG_POOL_NUM;
	voclMigLocalRDBufferPool = (struct strMigRDLocalBufferPool*)malloc(voclMigLocalReadBufferNum * 
						sizeof(struct strMigRDLocalBufferPool));
	for (i = 0; i < voclMigLocalReadBufferNum; i++)
	{
		voclMigReadLocalBufferInitialize(i);
	}
	return;
}

void voclMigReadLocalBufferFinalize()
{
    int i, index;
	for (index = 0; index < voclMigLocalReadBufferNum; index++)
	{
		for (i = 0; i < VOCL_MIG_BUF_NUM; i++) {
			voclMigLocalRDBufferPool[index].buffers[i].useFlag = VOCL_MIG_LOCAL_RD_BUF_AVALB;
			free(voclMigLocalRDBufferPool[index].buffers[i].ptr);
			voclMigLocalRDBufferPool[index].buffers[i].ptr = NULL;
		}

		voclMigLocalRDBufferPool[index].voclMigReadLocalBufferIndex = 0;
		voclMigLocalRDBufferPool[index].voclMigReadLocalBufferRstNum = 0;
	}

    free(voclMigLocalRDBufferPool);

    return;
}

static int voclMigSendToLocalNode(int rank, int index)
{
    int err;
    err = MPI_Isend(voclMigLocalRDBufferPool[rank].buffers[index].ptr,
                    voclMigLocalRDBufferPool[rank].buffers[index].size,
                    MPI_BYTE,
                    voclMigLocalRDBufferPool[rank].buffers[index].dest,
                    voclMigLocalRDBufferPool[rank].buffers[index].tag,
                    voclMigLocalRDBufferPool[rank].buffers[index].commData,
                   &voclMigLocalRDBufferPool[rank].buffers[index].request);
    return err;
}

int voclMigGetNextLocalReadBufferIndex(int rank)
{
    int index = voclMigLocalRDBufferPool[rank].voclMigReadLocalBufferIndex;
    MPI_Status status;

	if (rank >= voclMigLocalReadBufferNum)
	{
		voclMigReallocLocalRDBuffer(voclMigLocalReadBufferNum, 2*voclMigLocalReadBufferNum);
		voclMigLocalReadBufferNum *= 2;
	}

    if (voclMigLocalRDBufferPool[rank].buffers[index].useFlag == VOCL_MIG_LOCAL_RD_BUF_RDGPUMEM) {
        dlCLWaitForEvents(1, &voclMigLocalRDBufferPool[rank].buffers[index].event);
        voclMigSendToLocalNode(rank, index);
        MPI_Wait(&voclMigLocalRDBufferPool[rank].buffers[index].request, &status);
    }
    else if (voclMigLocalRDBufferPool[rank].buffers[index].useFlag == VOCL_MIG_LOCAL_RD_BUF_MPISEND) {
        MPI_Wait(&voclMigLocalRDBufferPool[rank].buffers[index].request, &status);
    }
    voclMigLocalRDBufferPool[rank].buffers[index].useFlag = VOCL_MIG_LOCAL_RD_BUF_AVALB;

    if (++voclMigLocalRDBufferPool[rank].voclMigReadLocalBufferIndex >= VOCL_MIG_BUF_NUM) {
        voclMigLocalRDBufferPool[rank].voclMigReadLocalBufferIndex = 0;
    }

    if (++voclMigLocalRDBufferPool[rank].voclMigReadLocalBufferRstNum > VOCL_MIG_BUF_NUM) {
        voclMigLocalRDBufferPool[rank].voclMigReadLocalBufferRstNum = VOCL_MIG_BUF_NUM;
    }

    return index;
}

void voclMigSetReadBufferFlag(int rank, int index, int flag)
{
    voclMigLocalRDBufferPool[rank].buffers[index].useFlag = flag;
    return;
}

MPI_Request *voclMigGetReadBufferRequestPtr(int rank, int index)
{
    return &voclMigLocalRDBufferPool[rank].buffers[index].request;
}

cl_event *voclMigGetReadBufferEventPtr(int rank, int index)
{
    return &voclMigLocalRDBufferPool[rank].buffers[index].event;
}

struct strMigReadLocalBuffer *voclMigGetReadBufferInfoPtr(int rank, int index)
{
    return &voclMigLocalRDBufferPool[rank].buffers[index];
}

int voclMigFinishLocalDataRead(int rank, MPI_Comm comm)
{
    int i, index, err;
    int startIndex, endIndex;
    MPI_Request request[VOCL_MIG_BUF_NUM];
    MPI_Status status[VOCL_MIG_BUF_NUM];
//    int activeBufferFlag[VOCL_MIG_BUF_NUM];
    int requestNo = 0;

    endIndex = voclMigLocalRDBufferPool[rank].voclMigReadLocalBufferIndex;
    startIndex = endIndex - voclMigLocalRDBufferPool[rank].voclMigReadLocalBufferRstNum;
    if (startIndex < 0) {
        startIndex += VOCL_MIG_BUF_NUM;
        endIndex += VOCL_MIG_BUF_NUM;
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_MIG_BUF_NUM;
        if (voclMigLocalRDBufferPool[rank].buffers[index].useFlag == VOCL_MIG_LOCAL_RD_BUF_RDGPUMEM) {
            dlCLWaitForEvents(1, &voclMigLocalRDBufferPool[rank].buffers[index].event);
            voclMigSendToLocalNode(rank, index);
            request[requestNo++] = voclMigLocalRDBufferPool[rank].buffers[index].request;
        }
        else if (voclMigLocalRDBufferPool[rank].buffers[index].useFlag == VOCL_MIG_LOCAL_RD_BUF_MPISEND) {
            request[requestNo++] = voclMigLocalRDBufferPool[rank].buffers[index].request;
        }
    }

    if (requestNo > 0) {
        err = MPI_Waitall(requestNo, request, status);
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_MIG_BUF_NUM;
            voclMigLocalRDBufferPool[rank].buffers[index].useFlag = VOCL_MIG_LOCAL_RD_BUF_AVALB;
            voclMigLocalRDBufferPool[rank].buffers[index].comm = comm;
    }

    return err;
}

/*----------------------Local to local migration buffer--------------------*/
struct strMigRWLocalBuffer *migRWLocalBufferInfo = NULL;
static int voclMigRWLocalBufferIndex = 0;
static int voclMigRWLocalBufferRstNum = 0;

void voclMigRWLocalBufferInitialize()
{
    int i;
    migRWLocalBufferInfo =
        (struct strMigRWLocalBuffer *) malloc(sizeof(struct strMigRWLocalBuffer) *
                                              VOCL_MIG_BUF_NUM);
    for (i = 0; i < VOCL_MIG_BUF_NUM; i++) {
        migRWLocalBufferInfo[i].wtCmdQueue = NULL;
        migRWLocalBufferInfo[i].wtMem = NULL;
        migRWLocalBufferInfo[i].size = 0;
        migRWLocalBufferInfo[i].offset = 0;
        migRWLocalBufferInfo[i].useFlag = VOCL_MIG_LOCAL_RW_BUF_AVALB;
        migRWLocalBufferInfo[i].ptr = (char *) malloc(VOCL_MIG_BUF_SIZE * sizeof(char));
        migRWLocalBufferInfo[i].rdEvent = NULL;
        migRWLocalBufferInfo[i].wtEvent = NULL;
    }

    voclMigRWLocalBufferIndex = 0;
    voclMigRWLocalBufferRstNum = 0;

    return;
}

void voclMigRWLocalBufferFinalize()
{
    int i;
    for (i = 0; i < VOCL_MIG_BUF_NUM; i++) {
        migRWLocalBufferInfo[i].useFlag = VOCL_MIG_LOCAL_RW_BUF_AVALB;
        free(migRWLocalBufferInfo[i].ptr);
        migRWLocalBufferInfo[i].ptr = NULL;
    }

    voclMigRWLocalBufferIndex = 0;
    voclMigRWLocalBufferRstNum = 0;

    free(migRWLocalBufferInfo);

    return;
}

static int voclMigRWWriteToGPUMem(int index)
{
    int err;
    err = dlCLEnqueueWriteBuffer(migRWLocalBufferInfo[index].wtCmdQueue,
                                 migRWLocalBufferInfo[index].wtMem,
                                 CL_FALSE,
                                 migRWLocalBufferInfo[index].offset,
                                 migRWLocalBufferInfo[index].size,
                                 migRWLocalBufferInfo[index].ptr,
                                 0, NULL, &migRWLocalBufferInfo[index].wtEvent);
    return err;
}

int voclMigRWGetNextLocalBufferIndex()
{
    int index = voclMigRWLocalBufferIndex;

    if (migRWLocalBufferInfo[index].useFlag == VOCL_MIG_LOCAL_RW_BUF_RDGPUMEM) {
        dlCLWaitForEvents(1, &migRWLocalBufferInfo[index].rdEvent);
        voclMigRWWriteToGPUMem(index);
        dlCLWaitForEvents(1, &migRWLocalBufferInfo[index].wtEvent);
    }
    else if (migRWLocalBufferInfo[index].useFlag == VOCL_MIG_LOCAL_RW_BUF_WTGPUMEM) {
        dlCLWaitForEvents(1, &migRWLocalBufferInfo[index].wtEvent);
    }
    migRWLocalBufferInfo[index].useFlag = VOCL_MIG_LOCAL_RW_BUF_AVALB;

    if (++voclMigRWLocalBufferIndex >= VOCL_MIG_BUF_NUM) {
        voclMigRWLocalBufferIndex = 0;
    }

    if (++voclMigRWLocalBufferRstNum > VOCL_MIG_BUF_NUM) {
        voclMigRWLocalBufferRstNum = VOCL_MIG_BUF_NUM;
    }

    return index;
}

struct strMigRWLocalBuffer *voclMigRWGetBufferInfoPtr(int index)
{
    return &migRWLocalBufferInfo[index];
}

void voclMigSetRWBufferFlag(int index, int flag)
{
    migRWLocalBufferInfo[index].useFlag = flag;
    return;
}

int voclMigFinishLocalDataRW(cl_command_queue cmdQueue)
{
    int i, index, err;
    int startIndex, endIndex;
    cl_event eventList[VOCL_MIG_BUF_NUM];
    int activeBufferFlag[VOCL_MIG_BUF_NUM];
    int eventNo = 0;

    endIndex = voclMigRWLocalBufferIndex;
    startIndex = endIndex - voclMigRWLocalBufferRstNum;
    if (startIndex < 0) {
        startIndex += VOCL_MIG_BUF_NUM;
        endIndex += VOCL_MIG_BUF_NUM;
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_MIG_BUF_NUM;
        activeBufferFlag[index] = 0;
        if (migRWLocalBufferInfo[index].wtCmdQueue == cmdQueue) {
            if (migRWLocalBufferInfo[index].useFlag == VOCL_MIG_LOCAL_RW_BUF_RDGPUMEM) {
                dlCLWaitForEvents(1, &migRWLocalBufferInfo[index].rdEvent);
                voclMigRWWriteToGPUMem(index);
                eventList[eventNo++] = migRWLocalBufferInfo[index].wtEvent;
                activeBufferFlag[index] = 1;
            }
            else if (migRWLocalBufferInfo[index].useFlag == VOCL_MIG_LOCAL_RW_BUF_WTGPUMEM) {
                eventList[eventNo++] = migRWLocalBufferInfo[index].wtEvent;
                activeBufferFlag[index] = 1;
            }
        }
    }
    if (eventNo > 0) {
        err = dlCLWaitForEvents(eventNo, eventList);
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_MIG_BUF_NUM;
        if (activeBufferFlag[index] == 1) {
            migRWLocalBufferInfo[index].useFlag = VOCL_MIG_LOCAL_RW_BUF_AVALB;
            migRWLocalBufferInfo[index].rdEvent = NULL;
            migRWLocalBufferInfo[index].wtEvent = NULL;
        }
    }

    return err;
}

