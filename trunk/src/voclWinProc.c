#include <stdio.h>
#include <string.h>
#include <CL/opencl.h>
#include "mpi.h"
#include "voclOpencl.h"
#include "voclOpenclMacro.h"

#define SERVICE_NAME_LEN 64
#define DEFAULT_PROXY_NUM 64

struct strVoclWinInfo {
	char serviceName[SERVICE_NAME_LEN];
	int  proxyRank; /* rank no within the proxy comm_world */
	int  appIndex;  /* index of the app on the proxy process */
	MPI_Comm commProxy;
	MPI_Comm commWin;  /* MPI communicator for win creation */
};

typedef struct strVoclWinInfoAll {
	int proxyNum;
	int migrationStatus;
	int targetProxyIndex;
	struct strVoclWinInfo wins[DEFAULT_PROXY_NUM];
} vocl_wins;

static vocl_wins *voclWinInfoPtr = NULL;
static MPI_Win *voclMappingWin = NULL;
static int totalVoclWinInfoNum;
static int voclWinInfoNo;

void voclWinInfoInitialize()
{
	totalVoclWinInfoNum = DEFAULT_PROXY_NUM;
	voclWinInfoNo = 0;
	voclWinInfoPtr = (vocl_wins *)malloc(sizeof(vocl_wins));
	memset(voclWinInfoPtr, 0, sizeof(vocl_wins));
	voclMappingWin = (MPI_Win *)malloc(sizeof(MPI_Win) * totalVoclWinInfoNum);
	
	return;
}

void voclWinInfoFinalize()
{
	totalVoclWinInfoNum = 0;
	voclWinInfoNo = 0;
	free(voclWinInfoPtr);
	free(voclMappingWin);

	return;
}

void voclWinInfoFree(int proxyIndex)
{
	MPI_Comm_free(&voclWinInfoPtr->wins[proxyIndex].commWin);
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
		printf("Used winInfo no %d is larger than available no %d, please change the macro DEFAULT_PROXY_NUM to a larger number.\n",
				voclWinInfoNo, totalVoclWinInfoNum);
		exit(1);
	}

	sprintf(voclWinInfoPtr->wins[voclWinInfoNo].serviceName, "%s", serviceName);
	voclWinInfoPtr->wins[voclWinInfoNo].proxyRank = msgGetProxyCommInfo.proxyRank;

	/* merge the communicator to create window for later one-sided data communication */
	MPI_Intercomm_merge(comm, 0, &mergedComm);
	voclWinInfoPtr->wins[voclWinInfoNo].commWin = mergedComm;
	voclWinInfoPtr->wins[voclWinInfoNo].commProxy = msgGetProxyCommInfo.comm;
	voclWinInfoPtr->wins[voclWinInfoNo].appIndex = msgGetProxyCommInfo.appIndex;
	voclWinInfoPtr->proxyNum++;
	voclWinInfoPtr->migrationStatus = 0;
	voclWinInfoPtr->targetProxyIndex = -1;
	printf("voclLib, proxyNum = %d\n", voclWinInfoPtr->proxyNum);

	/* create the window for passive data communication */
	MPI_Win_create(voclWinInfoPtr, sizeof(vocl_wins), 
				1, MPI_INFO_NULL, mergedComm, &voclMappingWin[voclWinInfoNo]);
	voclWinInfoNo++;

	return;
}


void voclPrintWinInfo()
{
	int i;
	for (i = 0; i < voclWinInfoNo; i++)
	{
		printf("proxyIndex = %d, serviceName = %s, proxyRank = %d, commProxy = %p, commWin = %p\n",
				i, voclWinInfoPtr->wins[i].serviceName, voclWinInfoPtr->wins[i].proxyRank, 
				voclWinInfoPtr->wins[i].commProxy, voclWinInfoPtr->wins[i].commWin);
	}

	return;
}

