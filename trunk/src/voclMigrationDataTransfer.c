#include <CL/opencl.h>
#include "mpi.h"
#include "vocl_opencl.h"
#include "voclMigration.h"

struct strMigGPUMemoryWrite {
    cl_command_queue cmdQueue;
    cl_mem memory;
    size_t size;
    int source;
    int isFromLocal;
    MPI_Comm comm;
};

struct strMigGPUMemoryRead {
    cl_command_queue cmdQueue;
    cl_mem memory;
    size_t size;
    int dest;
    int isToLocal;
    MPI_Comm comm;
};

struct strMigGPUMemoryWriteCmpd {
    int source;
    int isFromLocal;
    MPI_Comm comm;
    int retCode;
};

struct strMigGPUMemoryReadCmpd {
    int dest;
    int isToLocal;
    MPI_Comm comm;
    int retCode;
};

int voclMigIssueGPUMemoryWrite(MPI_Comm oldComm, MPI_Comm oldCommData, int oldRank,
                               MPI_Comm newComm, int newRank, int isFromLocal, int isToLocal,
                               cl_command_queue command_queue, cl_mem mem, size_t size)
{
    struct strMigGPUMemoryWrite migMemWrite;
    struct strMigWriteLocalBuffer *bufferPtr;
    int i, bufferIndex, bufferNum;
    size_t bufferSize, remainingSize;
    MPI_Request request[2];
    MPI_Status status[2];
    int requestNo;
    int migSource = -1;

    /* migrated GPU is local */
    if (isToLocal == 1 && isFromLocal == 0) {
        bufferSize = VOCL_MIG_BUF_SIZE;
        bufferNum = (size - 1) / VOCL_MIG_BUF_SIZE;
        remainingSize = size - bufferNum * VOCL_MIG_BUF_SIZE;
        for (i = 0; i <= bufferNum; i++) {
            if (i == bufferNum)
                bufferSize = remainingSize;
            bufferIndex = voclMigGetNextLocalWriteBufferIndex();
            bufferPtr = voclMigGetWriteBufferInfoPtr(bufferIndex);
            MPI_Irecv(bufferPtr->ptr, bufferSize, MPI_BYTE, oldRank,
                      VOCL_MIG_TAG + bufferIndex, oldCommData,
                      voclMigGetWriteBufferRequestPtr(bufferIndex));
            bufferPtr->cmdQueue = command_queue;
            bufferPtr->mem = mem;
            //bufferPtr->tag = VOCL_MIG_TAG+bufferIndex;
            bufferPtr->size = bufferSize;
            bufferPtr->offset = i * VOCL_MIG_BUF_SIZE;
            bufferPtr->comm = oldComm;
            voclMigSetWriteBufferFlag(bufferIndex, VOCL_MIG_LOCAL_WT_BUF_WAITDATA);
        }
    }
    else if (isToLocal == 0) {
        if (isFromLocal == 0) {
            /* send a message to the migration dest process to get the rank */
            MPI_Isend(NULL, 0, MPI_BYTE, oldRank, MIG_GET_PROXY_RANK, oldComm, &request[0]);
            MPI_Irecv(&migSource, 1, MPI_INT, oldRank, MIG_GET_PROXY_RANK, oldComm,
                      &request[1]);
            MPI_Waitall(2, request, status);
        }

        migMemWrite.cmdQueue = command_queue;
        migMemWrite.memory = mem;
        migMemWrite.size = size;
        migMemWrite.comm = newComm;
        migMemWrite.isFromLocal = isFromLocal;
        migMemWrite.source = migSource;

        MPI_Isend(&migMemWrite, sizeof(struct strMigGPUMemoryWrite), MPI_BYTE, newRank,
                  MIG_MEM_WRITE_REQUEST, newComm, &request[0]);
        MPI_Wait(&request[0], &status[0]);
    }

    /* return the source rank for issuing complete request */
    return migSource;
}

int voclMigIssueGPUMemoryRead(MPI_Comm oldComm, int oldRank, MPI_Comm newComm,
                              MPI_Comm newCommData, int newRank, int isFromLocal,
                              int isToLocal, cl_command_queue command_queue, cl_mem mem,
                              size_t size)
{
    struct strMigGPUMemoryRead migMemRead;
    int i, bufferIndex, bufferNum;
    size_t bufferSize, remainingSize;
    struct strMigReadLocalBuffer *bufferPtr;
    MPI_Request request[2];
    MPI_Status status[2];
    int requestNo;
    int migDest = -1;

    if (isFromLocal == 1 && isToLocal == 0) {
        bufferSize = VOCL_MIG_BUF_SIZE;
        bufferNum = (size - 1) / VOCL_MIG_BUF_SIZE;
        remainingSize = size - bufferNum * VOCL_MIG_BUF_SIZE;
        for (i = 0; i <= bufferNum; i++) {
            if (i == bufferNum)
                bufferSize = remainingSize;
            bufferIndex = voclMigGetNextLocalReadBufferIndex();
            bufferPtr = voclMigGetReadBufferInfoPtr(bufferIndex);
            dlCLEnqueueReadBuffer(command_queue, mem, CL_FALSE, i * VOCL_MIG_BUF_SIZE,
                                  bufferSize, bufferPtr->ptr, 0, NULL, &bufferPtr->event);
            bufferPtr->dest = newRank;
            bufferPtr->tag = VOCL_MIG_TAG + bufferIndex;
            bufferPtr->size = bufferSize;
            bufferPtr->offset = i * VOCL_MIG_BUF_SIZE;
            bufferPtr->comm = newComm;
            bufferPtr->commData = newCommData;
            voclMigSetReadBufferFlag(bufferIndex, VOCL_MIG_LOCAL_RD_BUF_RDGPUMEM);
        }
    }
    else if (isFromLocal == 0) {
        /* send a message to the migration dest process to get the rank */
        if (isToLocal == 0) {
            MPI_Isend(NULL, 0, MPI_BYTE, newRank, MIG_GET_PROXY_RANK, newComm, &request[0]);
            MPI_Irecv(&migDest, 1, MPI_INT, newRank, MIG_GET_PROXY_RANK, newComm, &request[1]);
            MPI_Waitall(2, request, status);
        }

        migMemRead.cmdQueue = command_queue;
        migMemRead.memory = mem;
        migMemRead.size = size;
        migMemRead.comm = oldComm;
        migMemRead.isToLocal = isToLocal;
        migMemRead.dest = migDest;

        MPI_Isend(&migMemRead, sizeof(struct strMigGPUMemoryRead), MPI_BYTE, oldRank,
                  MIG_MEM_READ_REQUEST, oldComm, &request[0]);
        MPI_Wait(&request[0], &status[0]);
    }

    /* return the source rank for issuing complete request */
    return migDest;
}

void voclMigFinishDataTransfer(MPI_Comm oldComm, int oldRank, cl_command_queue oldCmdQueue,
                               MPI_Comm newComm, int newRank, cl_command_queue newCmdQueue,
                               int proxySourceRank, int proxyDestRank, int isFromLocal,
                               int isToLocal)
{
    struct strMigGPUMemoryWriteCmpd writeCmpdRst;
    struct strMigGPUMemoryReadCmpd readCmpdRst;
    MPI_Request request[4];
    MPI_Status status[4];
    int retCodeSource, retCodeDest;
    int requestNo = 0;

    writeCmpdRst.source = proxySourceRank;
    writeCmpdRst.isFromLocal = isFromLocal;
    writeCmpdRst.comm = newComm;

    readCmpdRst.dest = proxyDestRank;
    readCmpdRst.isToLocal = isToLocal;
    readCmpdRst.comm = oldComm;

    if (isFromLocal == 0 && isToLocal == 0) {
        /* if source GPU is on a remote node */
        MPI_Isend(&readCmpdRst, sizeof(struct strMigGPUMemoryReadCmpd), MPI_BYTE,
                  oldRank, MIG_MEM_READ_CMPLD, oldComm, request + (requestNo++));
        MPI_Irecv(&readCmpdRst, sizeof(struct strMigGPUMemoryReadCmpd), MPI_BYTE,
                  oldRank, MIG_MEM_READ_CMPLD, oldComm, request + (requestNo++));
        /* if destination GPU is on a remote node */
        MPI_Isend(&writeCmpdRst, sizeof(struct strMigGPUMemoryWriteCmpd), MPI_BYTE,
                  newRank, MIG_MEM_WRITE_CMPLD, newComm, request + (requestNo++));
        MPI_Irecv(&writeCmpdRst, sizeof(struct strMigGPUMemoryWriteCmpd), MPI_BYTE,
                  newRank, MIG_MEM_WRITE_CMPLD, newComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
    }
    else if (isFromLocal == 0 && isToLocal == 1) {
        /* if source GPU is on a remote node */
        MPI_Isend(&readCmpdRst, sizeof(struct strMigGPUMemoryReadCmpd), MPI_BYTE,
                  oldRank, MIG_MEM_READ_CMPLD, oldComm, request + (requestNo++));
        voclMigFinishLocalDataWrite(oldComm);
        MPI_Irecv(&readCmpdRst, sizeof(struct strMigGPUMemoryReadCmpd), MPI_BYTE,
                  oldRank, MIG_MEM_READ_CMPLD, oldComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);

    }
    else if (isFromLocal == 1 && isToLocal == 0) {
        /* if destination GPU is on a remote node */
        MPI_Isend(&writeCmpdRst, sizeof(struct strMigGPUMemoryWriteCmpd), MPI_BYTE,
                  newRank, MIG_MEM_WRITE_CMPLD, newComm, request + (requestNo++));
        voclMigFinishLocalDataRead(newComm);
        MPI_Irecv(&writeCmpdRst, sizeof(struct strMigGPUMemoryWriteCmpd), MPI_BYTE,
                  newRank, MIG_MEM_WRITE_CMPLD, newComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
    }
    else {
        voclMigFinishLocalDataRW(newCmdQueue);
    }

    return;
}

void voclMigLocalToLocal(cl_command_queue oldCmdQueue, cl_mem oldMem,
                         cl_command_queue newCmdQueue, cl_mem newMem, size_t size)
{
    int bufferIndex, bufferNum, i;
    size_t bufferSize, remainingSize;
    struct strMigRWLocalBuffer *bufferPtr;

    bufferSize = VOCL_MIG_BUF_SIZE;
    bufferNum = (size - 1) / VOCL_MIG_BUF_SIZE;
    remainingSize = size - bufferNum * VOCL_MIG_BUF_SIZE;

    for (i = 0; i <= bufferNum; i++) {
        if (i == bufferNum)
            bufferSize = remainingSize;
        bufferIndex = voclMigRWGetNextLocalBufferIndex();
        bufferPtr = voclMigRWGetBufferInfoPtr(bufferIndex);
        dlCLEnqueueReadBuffer(oldCmdQueue, oldMem, CL_FALSE, 0, bufferSize,
                              bufferPtr->ptr, 0, NULL, &bufferPtr->rdEvent);
        bufferPtr->wtCmdQueue = newCmdQueue;
        bufferPtr->wtMem = newMem;
        bufferPtr->size = bufferSize;
        bufferPtr->offset = i * VOCL_MIG_BUF_SIZE;
        voclMigSetRWBufferFlag(bufferIndex, VOCL_MIG_LOCAL_RW_BUF_RDGPUMEM);
    }

    return;
}
