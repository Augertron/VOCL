#include <stdio.h>
#include <string.h>
#include <CL/opencl.h>
#include "mpi.h"
#include "mpix.h"
#include "voclOpencl.h"
#include "voclOpenclMacro.h"

#define SERVICE_NAME_LEN 64
#define DEFAULT_PROXY_NUM 64

struct strVoclWinInfo {
	char serviceName[SERVICE_NAME_LEN];
	int  proxyRank; /* rank no within the proxy comm_world */
	int  appIndex;  /* index of the app on the proxy process */

	MPI_Comm proxyComm;
	MPI_Comm proxyCommData;
	MPI_Comm commWin;  /* MPI communicator for win creation */

	char migrationStatus;
	char preMigStatus;
	char padding[2];
	int destProxyIndex;
	int destProxyRank;
};

typedef struct strVoclWinInfoAll {
	int proxyNum;
	struct strVoclWinInfo wins[DEFAULT_PROXY_NUM];
} vocl_wins;

static vocl_wins *voclWinInfoPtr = NULL;
static MPI_Win *voclWinPtr = NULL;
static int *voclWinMapPtr = NULL;
static MPIX_Mutex *voclLockers = NULL;
static int totalVoclWinInfoNum;
static int voclWinInfoNo;

extern void processAllWrites(int proxyIndex);
extern void processAllReads(int proxyIndex);

void voclWinInfoInitialize()
{
	totalVoclWinInfoNum = DEFAULT_PROXY_NUM;
	voclWinInfoNo = 0;
	voclWinInfoPtr = (vocl_wins *)malloc(sizeof(vocl_wins));
	memset(voclWinInfoPtr, 0, sizeof(vocl_wins));
	voclWinPtr = (MPI_Win *)malloc(sizeof(MPI_Win) * totalVoclWinInfoNum);
	voclWinMapPtr = (int *)malloc(sizeof(int) * totalVoclWinInfoNum);
	voclLockers = (MPIX_Mutex *)malloc(sizeof(MPIX_Mutex) * totalVoclWinInfoNum);

	return;
}

void voclWinInfoFinalize()
{
	totalVoclWinInfoNum = 0;
	voclWinInfoNo = 0;
	free(voclWinInfoPtr);
	free(voclWinPtr);
	free(voclWinMapPtr);
	free(voclLockers);

	return;
}

void voclWinInfoFree(int proxyIndex)
{
	MPI_Comm_free(&voclWinInfoPtr->wins[proxyIndex].commWin);
	MPI_Win_free(&voclWinPtr[proxyIndex]);
	MPIX_Mutex_destroy(voclLockers[proxyIndex]);
	voclWinInfoPtr->proxyNum--;
	return;
}

void voclAddWinInfo(MPI_Comm comm, int proxyRank, int proxyIndex, char *serviceName)
{
	struct strGetProxyCommInfo msgGetProxyCommInfo;
	MPI_Comm winComm;
	int requestNo = 0;
	MPI_Request request[2];
	MPI_Status  status[2];

	/* get the rank of each proxy process */
	msgGetProxyCommInfo.proxyIndexInApp = proxyIndex;
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

	sprintf(voclWinInfoPtr->wins[proxyIndex].serviceName, "%s", serviceName);
	voclWinInfoPtr->wins[proxyIndex].proxyRank = msgGetProxyCommInfo.proxyRank;

	/* merge the communicator to create window for later one-sided data communication */
	MPI_Intercomm_merge(comm, 0, &winComm);
	voclWinInfoPtr->wins[proxyIndex].commWin = winComm;
	voclWinInfoPtr->wins[proxyIndex].proxyComm = msgGetProxyCommInfo.comm;
	voclWinInfoPtr->wins[proxyIndex].proxyCommData = msgGetProxyCommInfo.commData;
	voclWinInfoPtr->wins[proxyIndex].appIndex = msgGetProxyCommInfo.appIndex;
	voclWinInfoPtr->wins[proxyIndex].migrationStatus = 0;
	voclWinInfoPtr->wins[proxyIndex].preMigStatus = 0;
	voclWinInfoPtr->wins[proxyIndex].destProxyIndex = -1;
	voclWinInfoPtr->wins[proxyIndex].destProxyRank = -1;

	/* create the locker in the lib size */
	voclLockers[proxyIndex] = MPIX_Mutex_create(1, winComm);

	voclWinInfoPtr->proxyNum++;

	/* create the window for passive data communication */
	MPI_Win_create(voclWinInfoPtr, sizeof(vocl_wins), 
				1, MPI_INFO_NULL, winComm, &voclWinPtr[proxyIndex]);
	voclWinMapPtr[voclWinInfoNo] = proxyIndex;

	voclWinInfoNo++;

	return;
}


void voclPrintWinInfo()
{
	int i;
	for (i = 0; i < voclWinInfoNo; i++)
	{
		printf("proxyIndex = %d, serviceName = %s, proxyRank = %d, proxyComm = %p, commWin = %p\n",
				i, voclWinInfoPtr->wins[voclWinMapPtr[i]].serviceName, 
				voclWinInfoPtr->wins[voclWinMapPtr[i]].proxyRank, 
				voclWinInfoPtr->wins[voclWinMapPtr[i]].proxyComm, 
				voclWinInfoPtr->wins[voclWinMapPtr[i]].commWin);
	}

	return;
}

char voclGetMigrationStatus(int proxyIndex)
{
	int migStatus;
	vocl_wins *winPtr;
	int offset;
	struct strVoclWinInfo migWin;

	winPtr = (vocl_wins *)malloc(sizeof(vocl_wins));
	offset = ((char *)&winPtr->wins[proxyIndex]) - ((char*)winPtr);
	MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, voclWinPtr[proxyIndex]);
	MPI_Get(&migWin, sizeof(struct strVoclWinInfo), MPI_BYTE, 0, offset,
			sizeof(struct strVoclWinInfo), MPI_BYTE, voclWinPtr[proxyIndex]);
	MPI_Win_unlock(0, voclWinPtr[proxyIndex]);
	free(winPtr);

	if (migWin.preMigStatus < migWin.migrationStatus)
	{
		migWin.preMigStatus = migWin.migrationStatus;
		if (proxyIndex == 0)
		{
			processAllWrites(proxyIndex);
			processAllReads(proxyIndex);
		}
	}

	return migWin.migrationStatus;
}

int voclGetMigrationDestProxyIndex(int proxyIndex)
{
	int migStatus;
	vocl_wins *winPtr;
	int destProxyRank, destProxyIndex, i;
	struct strVoclWinInfo migWin;

	winPtr = (vocl_wins *)malloc(sizeof(vocl_wins));
	MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, voclWinPtr[proxyIndex]);
	/* get all the window info */
	MPI_Get(winPtr, sizeof(vocl_wins), MPI_BYTE, 0, 0,
			sizeof(vocl_wins), MPI_BYTE, voclWinPtr[proxyIndex]);
	MPI_Win_unlock(0, voclWinPtr[proxyIndex]);

	/* get the dest proxy rank */
	destProxyRank = winPtr->wins[proxyIndex].destProxyRank;
	destProxyIndex = -1;
	for (i = 0; i < winPtr->proxyNum; i++)
	{
		if (destProxyRank == winPtr->wins[i].proxyRank)
		{
			destProxyIndex = i;
			break;
		}
	}

	if (destProxyIndex == -1)
	{
		printf("voclGetMigrationDestProxyIndex, destProxyRank %d does not exist!\n");
		exit (1);
	}

	free(winPtr);

	return destProxyIndex;
}

void voclMigrationMutexLock(int proxyIndex)
{
	MPIX_Mutex_lock(voclLockers[proxyIndex], 0, 0);
	return;
}

void voclMigrationMutexUnlock(int proxyIndex)
{
	MPIX_Mutex_unlock(voclLockers[proxyIndex], 0, 0);
	return;
}
