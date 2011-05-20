#include <CL/opencl.h>
#include "mpi.h"
#include "vocl_proxy_macro.h"

struct strMigGPUMemoryWrite {
	cl_command_queue cmdQueue;
	cl_mem memory;
	size_t size;
	int    source;
};

struct strMigGPUMemoryRead {
	cl_command_queue cmdQueue;
	cl_mem memory;
	size_t size;
	int    dest;
};

struct strMigGPUMemoryWriteCmpd {
	int source;
	int retCode;
};

struct strMigGPUMemoryReadCmpd {
	int dest;
	int retCode;
};

int voclMigIssueGPUMemoryWrite(MPI_Comm oldComm, int oldRank, 
		MPI_Comm newComm, int newRank, cl_command_queue command_queue,
		cl_mem mem, size_t size)
{
	struct strMigGPUMemoryWrite migMemWrite;
	MPI_Request request[2];
	MPI_Status status[2];
	int requestNo;
	int migSource;

	/* send a message to the migration dest process to get the rank */
	MPI_Isend(NULL, 0, MPI_BYTE, oldRank, MIG_GET_PROXY_RANK, oldComm, &request[0]);
	MPI_Irecv(&migSource, 1, MPI_INT, oldRank, MIG_GET_PROXY_RANK, oldComm, &request[1]);
	MPI_Waitall(2, request, status);

	migMemWrite.cmdQueue = command_queue;
	migMemWrite.memory   = mem;
	migMemWrite.size     = size;
	migMemWrite.source   = migSource;

	MPI_Isend(&migMemWrite, sizeof(struct strMigGPUMemoryWrite), MPI_BYTE, newRank, 
			MIG_MEM_WRITE_REQUEST, newComm, &request[0]);
	MPI_Wait(&request[0], &status[0]);

	/* return the source rank for issuing complete request */
	return migSource;
}

int voclMigIssueGPUMemoryRead(MPI_Comm oldComm, int oldRank, 
		MPI_Comm newComm, int newRank, cl_command_queue command_queue,
		cl_mem mem, size_t size)
{
	struct strMigGPUMemoryRead migMemRead;
	MPI_Request request[2];
	MPI_Status status[2];
	int requestNo;
	int migDest;

	/* send a message to the migration dest process to get the rank */
	MPI_Isend(NULL, 0, MPI_BYTE, newRank, MIG_GET_PROXY_RANK, newComm, &request[0]);
	MPI_Irecv(&migDest, 1, MPI_INT, newRank, MIG_GET_PROXY_RANK, newComm, &request[1]);
	MPI_Waitall(2, request, status);

	migMemRead.cmdQueue = command_queue;
	migMemRead.memory   = mem;
	migMemRead.size     = size;
	migMemRead.dest     = migDest;

	MPI_Isend(&migMemRead, sizeof(struct strMigGPUMemoryRead), MPI_BYTE, oldRank, 
			MIG_MEM_READ_REQUEST, oldComm, &request[0]);
	MPI_Wait(&request[0], &status[0]);
	
	/* return the source rank for issuing complete request */
	return migDest;
}

void voclMigFinishDataTransfer(MPI_Comm oldComm, int oldRank, MPI_Comm newComm, 
		int newRank, int proxySourceRank, int proxyDestRank)
{
	struct strMigGPUMemoryWriteCmpd writeCmpdRst;
	struct strMigGPUMemoryReadCmpd readCmpdRst;
	MPI_Request request[4];
	MPI_Status  status[4];
	int retCodeSource, retCodeDest;
	int requestNo = 0;

	writeCmpdRst.source = proxySourceRank;
	readCmpdRst.dest    = proxyDestRank;

	/* send complete request to the source proxy */
	MPI_Isend(&readCmpdRst, sizeof(struct strMigGPUMemoryReadCmpd), MPI_BYTE, 
			oldRank, MIG_MEM_READ_CMPLD, oldComm, request+(requestNo++));
	MPI_Isend(&writeCmpdRst, sizeof(struct strMigGPUMemoryWriteCmpd), MPI_BYTE, 
			newRank, MIG_MEM_WRITE_CMPLD, newComm, request+(requestNo++));
	/* wait data transfer to complete */
	MPI_Irecv(&readCmpdRst, sizeof(struct strMigGPUMemoryReadCmpd), MPI_BYTE, 
			oldRank, MIG_MEM_READ_CMPLD, oldComm, request+(requestNo++));
	MPI_Irecv(&writeCmpdRst, sizeof(struct strMigGPUMemoryWriteCmpd), MPI_BYTE, 
			newRank, MIG_MEM_WRITE_CMPLD, newComm, request+(requestNo++));
	MPI_Waitall(requestNo, request, status);

	return;
}

