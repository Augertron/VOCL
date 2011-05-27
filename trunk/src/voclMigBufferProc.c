#include <stdio.h>
#include <CL/opencl.h>
#include "mpi.h"
#include "voclOpencl.h"
#include "voclMigration.h"

/* ----------------------------Write Buffer-------------------------------*/
struct strMigWriteLocalBuffer *migWriteLocalBufferInfo = NULL;
int voclMigWriteLocalBufferIndex;
int voclMigWriteLocalBufferRstNum;

void voclMigWriteLocalBufferInitialize()
{
    int i;
    migWriteLocalBufferInfo =
        (struct strMigWriteLocalBuffer *) malloc(sizeof(struct strMigWriteLocalBuffer) *
                                                 VOCL_MIG_BUF_NUM);
    for (i = 0; i < VOCL_MIG_BUF_NUM; i++) {
        migWriteLocalBufferInfo[i].cmdQueue = NULL;
        migWriteLocalBufferInfo[i].mem = NULL;
        migWriteLocalBufferInfo[i].size = 0;
        migWriteLocalBufferInfo[i].offset = 0;
        migWriteLocalBufferInfo[i].event = NULL;
        migWriteLocalBufferInfo[i].useFlag = VOCL_MIG_LOCAL_WT_BUF_AVALB;
        migWriteLocalBufferInfo[i].ptr = (char *) malloc(VOCL_MIG_BUF_SIZE * sizeof(char));
    }

    voclMigWriteLocalBufferIndex = 0;
    voclMigWriteLocalBufferRstNum = 0;

    return;
}

void voclMigWriteLocalBufferFinalize()
{
    int i;
    for (i = 0; i < VOCL_MIG_BUF_NUM; i++) {
        migWriteLocalBufferInfo[i].useFlag = VOCL_MIG_LOCAL_WT_BUF_AVALB;
        free(migWriteLocalBufferInfo[i].ptr);
        migWriteLocalBufferInfo[i].ptr = NULL;
    }

    voclMigWriteLocalBufferIndex = 0;
    voclMigWriteLocalBufferRstNum = 0;

    free(migWriteLocalBufferInfo);

    return;
}

static int voclMigWriteToGPUMem(int index)
{
    int err;
    err = dlCLEnqueueWriteBuffer(migWriteLocalBufferInfo[index].cmdQueue,
                                 migWriteLocalBufferInfo[index].mem,
                                 CL_FALSE,
                                 migWriteLocalBufferInfo[index].offset,
                                 migWriteLocalBufferInfo[index].size,
                                 migWriteLocalBufferInfo[index].ptr,
                                 0, NULL, &migWriteLocalBufferInfo[index].event);
    return err;
}

int voclMigGetNextLocalWriteBufferIndex()
{
    int index = voclMigWriteLocalBufferIndex;
    MPI_Status status;

    if (migWriteLocalBufferInfo[index].useFlag == VOCL_MIG_LOCAL_WT_BUF_WAITDATA) {
        MPI_Wait(&migWriteLocalBufferInfo[index].request, &status);
        voclMigWriteToGPUMem(index);
        dlCLWaitForEvents(1, &migWriteLocalBufferInfo[index].event);
    }
    else if (migWriteLocalBufferInfo[index].useFlag == VOCL_MIG_LOCAL_WT_BUF_WTGPUMEM) {
        dlCLWaitForEvents(1, &migWriteLocalBufferInfo[index].event);
    }
    migWriteLocalBufferInfo[index].useFlag = VOCL_MIG_LOCAL_WT_BUF_AVALB;

    if (++voclMigWriteLocalBufferIndex >= VOCL_MIG_BUF_NUM) {
        voclMigWriteLocalBufferIndex = 0;
    }

    if (++voclMigWriteLocalBufferRstNum > VOCL_MIG_BUF_NUM) {
        voclMigWriteLocalBufferRstNum = VOCL_MIG_BUF_NUM;
    }

    return index;
}

void voclMigSetWriteBufferFlag(int index, int flag)
{
    migWriteLocalBufferInfo[index].useFlag = flag;
    return;
}

MPI_Request *voclMigGetWriteBufferRequestPtr(int index)
{
    return &migWriteLocalBufferInfo[index].request;
}

cl_event *voclMigGetWriteBufferEventPtr(int index)
{
    return &migWriteLocalBufferInfo[index].event;
}

struct strMigWriteLocalBuffer *voclMigGetWriteBufferInfoPtr(int index)
{
    return migWriteLocalBufferInfo + index;
}

int voclMigFinishLocalDataWrite(MPI_Comm comm)
{
    int i, index, err;
    int startIndex, endIndex;
    cl_event eventList[VOCL_MIG_BUF_NUM];
    int activeBufferFlag[VOCL_MIG_BUF_NUM];
    int eventNo = 0;
    MPI_Status status;

    endIndex = voclMigWriteLocalBufferIndex;
    startIndex = endIndex - voclMigWriteLocalBufferRstNum;
    if (startIndex < 0) {
        startIndex += VOCL_MIG_BUF_NUM;
        endIndex += VOCL_MIG_BUF_NUM;
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_MIG_BUF_NUM;
        activeBufferFlag[index] = 0;
        if (migWriteLocalBufferInfo[index].comm == comm) {
            if (migWriteLocalBufferInfo[index].useFlag == VOCL_MIG_LOCAL_WT_BUF_WAITDATA) {
                MPI_Wait(&migWriteLocalBufferInfo[index].request, &status);
                voclMigWriteToGPUMem(index);
                eventList[eventNo++] = migWriteLocalBufferInfo[index].event;
                activeBufferFlag[index] = 1;
            }
            else if (migWriteLocalBufferInfo[index].useFlag == VOCL_MIG_LOCAL_WT_BUF_WTGPUMEM) {
                eventList[eventNo++] = migWriteLocalBufferInfo[index].event;
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
            migWriteLocalBufferInfo[index].useFlag = VOCL_MIG_LOCAL_WT_BUF_AVALB;
            migWriteLocalBufferInfo[index].comm = -1;
        }
    }

    return err;
}

/* ------------------ read buffer ------------------------------------------------*/
struct strMigReadLocalBuffer *migReadLocalBufferInfo = NULL;
int voclMigReadLocalBufferIndex;
int voclMigReadLocalBufferRstNum;

void voclMigReadLocalBufferInitialize()
{
    int i;
    migReadLocalBufferInfo =
        (struct strMigReadLocalBuffer *) malloc(sizeof(struct strMigReadLocalBuffer) *
                                                VOCL_MIG_BUF_NUM);
    for (i = 0; i < VOCL_MIG_BUF_NUM; i++) {
        migReadLocalBufferInfo[i].dest = -1;
        migReadLocalBufferInfo[i].tag = -1;
        migReadLocalBufferInfo[i].size = 0;
        migReadLocalBufferInfo[i].offset = 0;
        migReadLocalBufferInfo[i].event = NULL;
        migReadLocalBufferInfo[i].useFlag = VOCL_MIG_LOCAL_RD_BUF_AVALB;
        migReadLocalBufferInfo[i].ptr = (char *) malloc(VOCL_MIG_BUF_SIZE * sizeof(char));
    }

    voclMigReadLocalBufferIndex = 0;
    voclMigReadLocalBufferRstNum = 0;

    return;
}

void voclMigReadLocalBufferFinalize()
{
    int i;
    for (i = 0; i < VOCL_MIG_BUF_NUM; i++) {
        migReadLocalBufferInfo[i].useFlag = VOCL_MIG_LOCAL_RD_BUF_AVALB;
        free(migReadLocalBufferInfo[i].ptr);
        migReadLocalBufferInfo[i].ptr = NULL;
    }

    voclMigReadLocalBufferIndex = 0;
    voclMigReadLocalBufferRstNum = 0;

    free(migReadLocalBufferInfo);

    return;
}

static int voclMigSendToLocalNode(int index)
{
    int err;
    err = MPI_Isend(migReadLocalBufferInfo[index].ptr,
                    migReadLocalBufferInfo[index].size,
                    MPI_BYTE,
                    migReadLocalBufferInfo[index].dest,
                    migReadLocalBufferInfo[index].tag,
                    migReadLocalBufferInfo[index].commData,
                    &migReadLocalBufferInfo[index].request);
    return err;
}

int voclMigGetNextLocalReadBufferIndex()
{
    int index = voclMigReadLocalBufferIndex;
    MPI_Status status;
    if (migReadLocalBufferInfo[index].useFlag == VOCL_MIG_LOCAL_RD_BUF_RDGPUMEM) {
        dlCLWaitForEvents(1, &migReadLocalBufferInfo[index].event);
        voclMigSendToLocalNode(index);
        MPI_Wait(&migReadLocalBufferInfo[index].request, &status);
    }
    else if (migReadLocalBufferInfo[index].useFlag == VOCL_MIG_LOCAL_RD_BUF_MPISEND) {
        MPI_Wait(&migReadLocalBufferInfo[index].request, &status);
    }
    migReadLocalBufferInfo[index].useFlag = VOCL_MIG_LOCAL_RD_BUF_AVALB;

    if (++voclMigReadLocalBufferIndex >= VOCL_MIG_BUF_NUM) {
        voclMigReadLocalBufferIndex = 0;
    }

    if (++voclMigReadLocalBufferRstNum > VOCL_MIG_BUF_NUM) {
        voclMigReadLocalBufferRstNum = VOCL_MIG_BUF_NUM;
    }

    return index;
}

void voclMigSetReadBufferFlag(int index, int flag)
{
    migReadLocalBufferInfo[index].useFlag = flag;
    return;
}

MPI_Request *voclMigGetReadBufferRequestPtr(int index)
{
    return &migReadLocalBufferInfo[index].request;
}

cl_event *voclMigGetReadBufferEventPtr(int index)
{
    return &migReadLocalBufferInfo[index].event;
}

struct strMigReadLocalBuffer *voclMigGetReadBufferInfoPtr(int index)
{
    return &migReadLocalBufferInfo[index];
}

int voclMigFinishLocalDataRead(MPI_Comm comm)
{
    int i, index, err;
    int startIndex, endIndex;
    MPI_Request request[VOCL_MIG_BUF_NUM];
    MPI_Status status[VOCL_MIG_BUF_NUM];
    int activeBufferFlag[VOCL_MIG_BUF_NUM];
    int requestNo = 0;

    endIndex = voclMigReadLocalBufferIndex;
    startIndex = endIndex - voclMigReadLocalBufferRstNum;
    if (startIndex < 0) {
        startIndex += VOCL_MIG_BUF_NUM;
        endIndex += VOCL_MIG_BUF_NUM;
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_MIG_BUF_NUM;
        activeBufferFlag[index] = 0;
        if (migReadLocalBufferInfo[index].comm == comm) {
            if (migReadLocalBufferInfo[index].useFlag == VOCL_MIG_LOCAL_RD_BUF_RDGPUMEM) {
                dlCLWaitForEvents(1, &migReadLocalBufferInfo[index].event);
                voclMigSendToLocalNode(index);
                request[requestNo++] = migReadLocalBufferInfo[index].request;
                activeBufferFlag[index] = 1;
            }
            else if (migReadLocalBufferInfo[index].useFlag == VOCL_MIG_LOCAL_RD_BUF_MPISEND) {
                request[requestNo++] = migReadLocalBufferInfo[index].request;
                activeBufferFlag[index] = 1;
            }
        }
    }

    if (requestNo > 0) {
        err = MPI_Waitall(requestNo, request, status);
    }

    for (i = startIndex; i < endIndex; i++) {
        index = i % VOCL_MIG_BUF_NUM;
        if (activeBufferFlag[index] == 1) {
            migReadLocalBufferInfo[index].useFlag = VOCL_MIG_LOCAL_RD_BUF_AVALB;
            migReadLocalBufferInfo[index].comm = comm;
        }
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
