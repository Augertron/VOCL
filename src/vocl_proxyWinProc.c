#include <stdio.h>
#include <CL/opencl.h>
#include "mpi.h"
#include "vocl_proxy_macro.h"


struct strVoclWinInfo {
    char serviceName[SERVICE_NAME_LEN];
    int  proxyRank; /* rank no within the proxy comm_world */
    MPI_Comm commProxy;
    MPI_Comm commWin;  /* MPI communicator for win creation */
};

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

void voclProxyFreeWin(int appIndex)
{
	MPI_Comm_free(&voclProxyWinComm[appIndex]);
	MPI_Win_free(&voclProxyWinPtr[appIndex]);
}

void voclProxyPrintWinInfo()
{
	int i, j;
	struct strVoclWinInfo *winPtr;
	winPtr = (struct strVoclWinInfo *)malloc(sizeof(struct strVoclWinInfo) * DEFAULT_PROXY_NUM);
	/* print the win info on the proxy process */
	for (i = 0; i < voclProxyWinNum; i++)
	{
		if (voclIsCommUsed(i) == 1)
		{
			MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, voclProxyWinPtr[i]);
			MPI_Get(winPtr, sizeof(struct strVoclWinInfo) * (DEFAULT_PROXY_NUM), MPI_BYTE, 0, 0,
					sizeof(struct strVoclWinInfo) * (DEFAULT_PROXY_NUM), MPI_BYTE, voclProxyWinPtr[i]);
			MPI_Win_unlock(0, voclProxyWinPtr[i]);
			printf("AppIndex = %d:\n", i);
			for (j = 0; j < DEFAULT_PROXY_NUM; j++)
			{
				printf("\tproxyIndex = %j, serviceName = %s, proxyRank = %d, commProxy = %p, commWin = %p\n", 
						j, winPtr[j].serviceName, winPtr[j].proxyRank, winPtr[j].commProxy, winPtr[j].commWin);
			}
		}
	}

	return;
}

