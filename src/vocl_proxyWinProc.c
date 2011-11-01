#include <stdio.h>
#include <CL/opencl.h>
#include "mpi.h"
#include "mpix.h"
#include "vocl_proxy_macro.h"
#include "vocl_proxyWinProc.h"

extern int voclIsCommUsed(int appIndex);
static MPI_Win *voclProxyWinPtr = NULL;
static MPIX_Mutex *voclProxyLockers = NULL;
static MPI_Comm *voclProxyWinComm = NULL;
static int *voclProxyProxyIndexInApp = NULL;
static int voclProxyWinNum;
static int voclProxyWinNo;

void voclProxyWinInitialize()
{
	voclProxyWinNum = DEFAULT_APP_NUM;
	voclProxyWinPtr = (MPI_Win *)malloc(sizeof(MPI_Win) * voclProxyWinNum);
	voclProxyWinComm = (MPI_Comm *)malloc(sizeof(MPI_Comm) * voclProxyWinNum);
	voclProxyLockers = (MPIX_Mutex *)malloc(sizeof(MPIX_Mutex) * voclProxyWinNum);
	voclProxyProxyIndexInApp = (int *)malloc(sizeof(int) * voclProxyWinNum);

	voclProxyWinNo = 0;

	return;
}

void voclProxyWinFinalize()
{
	voclProxyWinNum = 0;
	voclProxyWinNo = 0;
	free(voclProxyWinPtr);
	free(voclProxyWinComm);
	free(voclProxyLockers);
	free(voclProxyProxyIndexInApp);

	return;
}

void voclProxyCreateWin(MPI_Comm comm, int appIndex, int proxyIndexInApp)
{
	MPI_Comm winComm;
	if (appIndex >= voclProxyWinNum)
	{
		voclProxyWinNum = appIndex + DEFAULT_APP_NUM;
		voclProxyWinPtr = (MPI_Win *)realloc(voclProxyWinPtr, sizeof(MPI_Win) * voclProxyWinNum);
		voclProxyWinComm = (MPI_Comm *)realloc(voclProxyWinComm, sizeof(MPI_Comm) * voclProxyWinNum);
		voclProxyLockers = (MPIX_Mutex *)realloc(voclProxyLockers, sizeof(MPIX_Mutex) * voclProxyWinNum);
		voclProxyProxyIndexInApp = (int *)realloc(voclProxyProxyIndexInApp, sizeof(int) * voclProxyWinNum);
	}

	/* proxy process gets the high rank */
	MPI_Intercomm_merge(comm, 1, &winComm);
	voclProxyWinComm[appIndex] = winComm;

	voclProxyLockers[appIndex] = MPIX_Mutex_create(0, winComm);
	//voclProxyLockers[appIndex] = MPIX_Mutex_create(1, winComm);

	/* create the window, window home rank is 0 */
	MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, winComm, &voclProxyWinPtr[appIndex]);
	voclProxyProxyIndexInApp[appIndex] = proxyIndexInApp;
}

MPI_Win *voclProxyGetWinPtr(int index)
{
	return &voclProxyWinPtr[index];
}

void voclProxyFreeWin(int appIndex)
{
	/* free the merged MPI communicator */
	MPI_Comm_free(&voclProxyWinComm[appIndex]);
	MPI_Win_free(&voclProxyWinPtr[appIndex]);
	MPIX_Mutex_destroy(voclProxyLockers[appIndex]);

	return;
}

void voclProxyPrintWinInfo()
{
	int i, j;
	int proxyNum;
	vocl_proxy_wins *winPtr;
	winPtr = (vocl_proxy_wins *)malloc(sizeof(vocl_proxy_wins));
	/* print the win info on the proxy process */
	for (i = 0; i < voclProxyWinNum; i++)
	{
		if (voclIsCommUsed(i) == 1)
		{
			MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, voclProxyWinPtr[i]);
			MPI_Get(winPtr, sizeof(vocl_proxy_wins), MPI_BYTE, 0, 0,
					sizeof(vocl_proxy_wins), MPI_BYTE, voclProxyWinPtr[i]);
			MPI_Win_unlock(0, voclProxyWinPtr[i]);
			for (j = 0; j < winPtr->proxyNum; j++)
			{
				printf("\tproxyIndex = %d, serviceName = %s, proxyRank = %d, proxyComm = %p, commWin = %p\n", 
						j, winPtr->wins[j].serviceName, winPtr->wins[j].proxyRank, 
						winPtr->wins[j].proxyComm, winPtr->wins[j].commWin);
			}
		}
	}

	free(winPtr);

	return;
}

void voclProxyMigrationMutexLock(int appIndex)
{
	MPIX_Mutex_lock(voclProxyLockers[appIndex], 0, 0);
	//MPIX_Mutex_lock(voclProxyLockers[appIndex], 0, 1);
	return;
}

void voclProxyMigrationMutexUnlock(int appIndex)
{
	MPIX_Mutex_unlock(voclProxyLockers[appIndex], 0, 0);
	//MPIX_Mutex_unlock(voclProxyLockers[appIndex], 0, 1);
	return;
}

void voclProxyUpdateMigStatus(int appIndex, int destProxyRank, int isOnSameNode)
{
	struct strVoclWinInfo win;
	vocl_proxy_wins *tmpWins;
	int proxyIndexInApp;
	int offset;

	proxyIndexInApp = voclProxyProxyIndexInApp[appIndex];
	tmpWins = (vocl_proxy_wins *)malloc(sizeof(vocl_proxy_wins));
	offset = ((char *)&tmpWins->wins[proxyIndexInApp]) - ((char *)tmpWins);
	MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, voclProxyWinPtr[appIndex]);
	MPI_Get(&win, sizeof(struct strVoclWinInfo), MPI_BYTE, 0, offset,
			sizeof(struct strVoclWinInfo), MPI_BYTE, voclProxyWinPtr[appIndex]);
	MPI_Win_unlock(0, voclProxyWinPtr[appIndex]);

	/* update the migration status and target proxy index */
	if (isOnSameNode == 0)
	{
		win.migrationStatus++;
	}
	win.destProxyRank = destProxyRank;

	MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, voclProxyWinPtr[appIndex]);
	MPI_Put(&win, sizeof(struct strVoclWinInfo), MPI_BYTE, 0, offset,
			sizeof(struct strVoclWinInfo), MPI_BYTE, voclProxyWinPtr[appIndex]);
	MPI_Win_unlock(0, voclProxyWinPtr[appIndex]);

	free(tmpWins);
	
	return;
}

void voclProxySetMigStatus(int appIndex, char migStatus)
{
	struct strVoclWinInfo win;
	vocl_proxy_wins *tmpWins;
	int proxyIndexInApp;
	int offset;

	proxyIndexInApp = voclProxyProxyIndexInApp[appIndex];
	tmpWins = (vocl_proxy_wins *)malloc(sizeof(vocl_proxy_wins));
	offset = ((char *)&tmpWins->wins[proxyIndexInApp]) - ((char *)tmpWins);
	MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, voclProxyWinPtr[appIndex]);
	MPI_Get(&win, sizeof(struct strVoclWinInfo), MPI_BYTE, 0, offset,
			sizeof(struct strVoclWinInfo), MPI_BYTE, voclProxyWinPtr[appIndex]);
	MPI_Win_unlock(0, voclProxyWinPtr[appIndex]);

	/* update the migration status and target proxy index */
	win.migrationStatus = migStatus;

	MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, voclProxyWinPtr[appIndex]);
	MPI_Put(&win, sizeof(struct strVoclWinInfo), MPI_BYTE, 0, offset,
			sizeof(struct strVoclWinInfo), MPI_BYTE, voclProxyWinPtr[appIndex]);
	MPI_Win_unlock(0, voclProxyWinPtr[appIndex]);

	free(tmpWins);
	
	return;
}

