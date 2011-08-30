#include <stdio.h>
#include <CL/opencl.h>
#include "mpi.h"
#include "vocl_proxy_macro.h"
#include "vocl_proxyWinProc.h"

extern int voclIsCommUsed(int appIndex);
static MPI_Win *voclProxyWinPtr = NULL;
static MPI_Comm *voclProxyWinComm = NULL;
static int voclProxyWinNum;
static int voclProxyWinNo;

void voclProxyWinInitialize()
{
	voclProxyWinNum = DEFAULT_APP_NUM;
	voclProxyWinPtr = (MPI_Win *)malloc(sizeof(MPI_Win) * voclProxyWinNum);
	voclProxyWinComm = (MPI_Comm *)malloc(sizeof(MPI_Comm) * voclProxyWinNum);
	voclProxyWinNo = 0;

	return;
}

void voclProxyWinFinalize()
{
	voclProxyWinNum = 0;
	voclProxyWinNo = 0;
	free(voclProxyWinPtr);
	free(voclProxyWinComm);

	return;
}

void voclProxyCreateWin(MPI_Comm comm, int appIndex)
{
	MPI_Comm intraComm;
	if (appIndex >= voclProxyWinNum)
	{
		voclProxyWinNum = appIndex + DEFAULT_APP_NUM;
		voclProxyWinPtr = (MPI_Win *)realloc(voclProxyWinPtr, sizeof(MPI_Win) * voclProxyWinNum);
		voclProxyWinComm = (MPI_Comm *)realloc(voclProxyWinComm, sizeof(MPI_Comm) * voclProxyWinNum);
	}

	/* proxy process gets the high rank */
	MPI_Intercomm_merge(comm, 1, &intraComm);
	voclProxyWinComm[appIndex] = intraComm;

	/* create the window, window home rank is 0 */
	MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, intraComm, &voclProxyWinPtr[appIndex]);
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
				printf("\tproxyIndex = %d, serviceName = %s, proxyRank = %d, commProxy = %p, commWin = %p\n", 
						j, winPtr->wins[j].serviceName, winPtr->wins[j].proxyRank, 
						winPtr->wins[j].commProxy, winPtr->wins[j].commWin);
			}
		}
	}

	free(winPtr);

	return;
}

