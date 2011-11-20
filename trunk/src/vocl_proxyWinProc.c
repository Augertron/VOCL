#include <stdio.h>
#include <CL/opencl.h>
#include <unistd.h>
#include "mpi.h"
#include "mpix.h"
#include "vocl_proxy_macro.h"
#include "vocl_proxyWinProc.h"

extern int voclIsCommUsed(int appIndex);
extern int voclProxyGetCommandNumInInternalQueue(int appIndex);

static MPI_Win *voclProxyWinPtr = NULL;
static MPIX_Mutex *voclProxyMigrationLockers = NULL;
static MPIX_Mutex *voclProxyConMsgLockers = NULL;
static int *voclProxyConMsgLockerAcquired = NULL;
static MPI_Comm *voclProxyWinComm = NULL;
static int *voclProxyProxyIndexInApp = NULL;
static int voclProxyWinNum;
static int voclProxyWinNo;

void voclProxyWinInitialize()
{
	voclProxyWinNum = DEFAULT_APP_NUM;
	voclProxyWinPtr = (MPI_Win *)malloc(sizeof(MPI_Win) * voclProxyWinNum);
	voclProxyWinComm = (MPI_Comm *)malloc(sizeof(MPI_Comm) * voclProxyWinNum);
	voclProxyMigrationLockers = (MPIX_Mutex *)malloc(sizeof(MPIX_Mutex) * voclProxyWinNum);
	voclProxyConMsgLockers = (MPIX_Mutex *)malloc(sizeof(MPIX_Mutex) * voclProxyWinNum);
	voclProxyConMsgLockerAcquired = (int *)malloc(sizeof(int) * voclProxyWinNum);
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
	free(voclProxyMigrationLockers);
	free(voclProxyConMsgLockers);
	free(voclProxyConMsgLockerAcquired);
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
		voclProxyMigrationLockers = (MPIX_Mutex *)realloc(voclProxyMigrationLockers, sizeof(MPIX_Mutex) * voclProxyWinNum);
		voclProxyConMsgLockers = (MPIX_Mutex *)realloc(voclProxyConMsgLockers, sizeof(MPIX_Mutex) * voclProxyWinNum);
		voclProxyConMsgLockerAcquired = (int *)realloc(voclProxyConMsgLockerAcquired, sizeof(int) * voclProxyWinNum);
		voclProxyProxyIndexInApp = (int *)realloc(voclProxyProxyIndexInApp, sizeof(int) * voclProxyWinNum);
	}

	/* proxy process gets the high rank */
	MPI_Intercomm_merge(comm, 1, &winComm);
	voclProxyWinComm[appIndex] = winComm;

	voclProxyMigrationLockers[appIndex] = MPIX_Mutex_create(0, winComm);
	voclProxyConMsgLockers[appIndex] = MPIX_Mutex_create(0, winComm);

	/* MPI mutex is not acquired */
	voclProxyConMsgLockerAcquired[appIndex] = 0;

	/* create the window, window home rank is 0 */
	MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, winComm, &voclProxyWinPtr[appIndex]);
	voclProxyProxyIndexInApp[appIndex] = proxyIndexInApp;
}

MPI_Comm voclProxyGetWinComm(int appIndex)
{
	return voclProxyWinComm[appIndex];
}

MPI_Win *voclProxyGetWinPtr(int index)
{
	return &voclProxyWinPtr[index];
}

void voclProxyMigrationMutexLock(int appIndex)
{
	MPIX_Mutex_lock(voclProxyMigrationLockers[appIndex], 0, 0);
	return;
}

void voclProxyMigrationMutexUnlock(int appIndex)
{
	MPIX_Mutex_unlock(voclProxyMigrationLockers[appIndex], 0, 0);
	return;
}

void voclProxyConMsgMutexLock(int appIndex)
{
	MPIX_Mutex_lock(voclProxyConMsgLockers[appIndex], 0, 0);
	return;
}

void voclProxyConMsgMutexUnlock(int appIndex)
{
	MPIX_Mutex_unlock(voclProxyConMsgLockers[appIndex], 0, 0);
	return;
}

void voclProxyFreeWin(int appIndex)
{
	if (voclProxyConMsgLockerAcquired[appIndex] == 1)
	{
		voclProxyConMsgLockerAcquired[appIndex] = 0;
		voclProxyMigrationMutexUnlock(appIndex);
	}

	/* free the merged MPI communicator */
	MPI_Comm_free(&voclProxyWinComm[appIndex]);
	MPI_Win_free(&voclProxyWinPtr[appIndex]);
	MPIX_Mutex_destroy(voclProxyMigrationLockers[appIndex]);
	MPIX_Mutex_destroy(voclProxyConMsgLockers[appIndex]);

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

int voclProxyGetConMsgLockerAcquiredFlag(int appIndex)
{
	return voclProxyConMsgLockerAcquired[appIndex];
}

void voclProxySetConMsgLockerAcquiredFlag(int appIndex, int flag)
{
	voclProxyConMsgLockerAcquired[appIndex] = flag;
	return;
}

/* if the number of function calls in the queue is larger than */
/* a threshold, restrict the command issue */
void voclProxyConMsgFlowControl(int commSize)
{
	int i;
	/* start from 1 since 0 is for communication across proxy process */
	for (i = 1; i < commSize; i++)
	{
		/* app is available */
		if (voclIsCommUsed(i) == 1)
		{
			/* number of commands in the queue is larger than threshold */
			/* and the MPI mutex locker is not required yet */
			if (voclProxyGetCommandNumInInternalQueue(i) >= VOCL_PROXY_APP_MAX_CMD_NUM)
			{
				if (voclProxyConMsgLockerAcquired[i] == 0)
				{
					/* restrict app from issuing function calls */
					//voclProxyConMsgMutexLock(i);
					voclProxyConMsgLockerAcquired[i] = 1;
printf("i = %d, lockerAcquired = %d\n", i, voclProxyConMsgLockerAcquired[i]);
				}
			}
			else
			{
				if (voclProxyConMsgLockerAcquired[i] == 1)
				{
					/* restrict app from issuing function calls */
					//voclProxyConMsgMutexUnlock(i);
					voclProxyConMsgLockerAcquired[i] = 0;
				}
			}
		}
	}

	return;
}


