#include <stdio.h>
#include <CL/opencl.h>
#include "mpi.h"
#include "voclOpencl.h"
#include "voclOpenclMacro.h"

#define SERVICE_NAME_LEN 64
#define DEFAULT_PROXY_NUM 64

struct strVoclWinInfo {
	char serviceName[SERVICE_NAME_LEN];
	int  proxyRank; /* rank no within the proxy comm_world */
	MPI_Comm commProxy;
	MPI_Comm commWin;  /* MPI communicator for win creation */
};

static struct strVoclWinInfo *voclWinInfoPtr = NULL;
static MPI_Win *voclMappingWin = NULL;
static int totalVoclWinInfoNum;
static int voclWinInfoNo;

void voclWinInfoInitialize()
{
	totalVoclWinInfoNum = DEFAULT_PROXY_NUM;
	voclWinInfoNo = 0;
	voclWinInfoPtr = (struct strVoclWinInfo *)malloc(sizeof(struct strVoclWinInfo) * totalVoclWinInfoNum);
	voclMappingWin = (MPI_Win *)malloc(sizeof(MPI_Win) * totalVoclWinInfoNum);
	
	return;
}

void voclWinInfoFinalize()
{
	int i;

	totalVoclWinInfoNum = 0;
	voclWinInfoNo = 0;
	free(voclWinInfoPtr);
	free(voclMappingWin);

	return;
}

void voclWinInfoFree(int proxyIndex)
{
	MPI_Comm_free(&voclWinInfoPtr[proxyIndex].commWin);
	MPI_Win_free(&voclMappingWin[proxyIndex]);
	return;
}

void voclAddWinInfo(MPI_Comm comm, int proxyRank, char *serviceName)
{
	struct strGetProxyCommInfo msgGetProxyCommInfo;
	MPI_Comm mergedComm;
	int requestNo = 0;
	MPI_Request request[2];
	MPI_Status  status[2];

	/* get the rank of each proxy process */
	MPI_Isend(&msgGetProxyCommInfo, sizeof(struct strGetProxyCommInfo), MPI_BYTE, proxyRank,
			GET_PROXY_COMM_INFO, comm, request+(requestNo++));
	MPI_Irecv(&msgGetProxyCommInfo, sizeof(struct strGetProxyCommInfo), MPI_BYTE, proxyRank,
			GET_PROXY_COMM_INFO, comm, request+(requestNo++));
	MPI_Waitall(requestNo, request, status);

	/* record the rank info of the proxy process */
	if (voclWinInfoNo >= totalVoclWinInfoNum)
	{
		totalVoclWinInfoNum *= 2;
		voclWinInfoPtr = (struct strVoclWinInfo *)realloc(voclWinInfoPtr, sizeof(struct strVoclWinInfo) * totalVoclWinInfoNum);
		voclMappingWin = (MPI_Win *)realloc(voclMappingWin, sizeof(MPI_Win) * totalVoclWinInfoNum);
	}

	sprintf(voclWinInfoPtr[voclWinInfoNo].serviceName, "%s", serviceName);
	voclWinInfoPtr[voclWinInfoNo].proxyRank = msgGetProxyCommInfo.proxyRank;

	/* merge the communicator to create window for later one-sided data communication */
	MPI_Intercomm_merge(comm, 0, &mergedComm);
	voclWinInfoPtr[voclWinInfoNo].commWin = mergedComm;
	voclWinInfoPtr[voclWinInfoNo].commProxy = msgGetProxyCommInfo.comm;

	/* create the window for passive data communication */
	MPI_Win_create(voclWinInfoPtr, sizeof(struct strVoclWinInfo) * voclWinInfoNo, 
				1, MPI_INFO_NULL, mergedComm, &voclMappingWin[voclWinInfoNo]);
	voclWinInfoNo++;

	return;
}




