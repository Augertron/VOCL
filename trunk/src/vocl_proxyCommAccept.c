#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <mpi.h>
#include "vocl_proxy_macro.h"

static int voclAppNo;
static int voclAppNum;
static char *voclCommUsedFlag;
static char **conMsgBufferStartPtr;
static int  voclTmpRequestNumForMigration;

extern MPI_Request *conMsgRequest;
extern MPI_Request *conMsgRequestForWait;
extern int *conMsgRequestIndex;
extern MPI_Comm *appComm, *appCommData;
extern int voclTotalRequestNum;
extern int voclCommUsedSize;
extern char voclPortName[MPI_MAX_PORT_NAME];
extern char **conMsgBuffer;

pthread_t thAppComm;
static pthread_mutex_t commLock;

void voclProxyCommInitialize()
{
	int i;
    voclAppNum = DEFAULT_APP_NUM;
    voclAppNo = 0;
    voclTotalRequestNum = 0;
	voclCommUsedSize = 0;
    conMsgRequest = (MPI_Request *) malloc(voclAppNum * CMSG_NUM * sizeof(MPI_Request));
	conMsgRequestForWait = (MPI_Request *)malloc(voclAppNum * sizeof(MPI_Request));
	conMsgRequestIndex = (int *)malloc(voclAppNum * sizeof(int));
    appComm = (MPI_Comm *) malloc(voclAppNum * sizeof(MPI_Comm));
    appCommData = (MPI_Comm *) malloc(voclAppNum * sizeof(MPI_Comm));
	voclCommUsedFlag = (char *)malloc(voclAppNum * sizeof(char));
	/* initialize all flag as being not used */
	memset(voclCommUsedFlag, 0, sizeof(char) * voclAppNum);
	conMsgBuffer = (char **)malloc(voclAppNum * CMSG_NUM * sizeof(char *));
	conMsgBufferStartPtr = conMsgBuffer;
	for (i = 0; i < voclAppNum*CMSG_NUM; i++)
	{
		conMsgBuffer[i] = (char *)malloc(sizeof(CON_MSG_BUFFER));
	}
	/*initialize the lock */
	pthread_mutex_init(&commLock, NULL);
}

void voclProxyCommFinalize()
{
    int i;
	int totalBufferNum;
	totalBufferNum = voclAppNum * CMSG_NUM;

    free(conMsgRequest);
	free(conMsgRequestForWait);
	free(conMsgRequestIndex);
	free(voclCommUsedFlag);
    free(appComm);
    free(appCommData);

	/* free control msg buffer */
	for (i = 0; i < totalBufferNum; i++)
	{
		free(conMsgBuffer[i]);
	}
	
	/* free control msg pointer buffer */
    free(conMsgBufferStartPtr);
	/*destroy the lock */
	pthread_mutex_destroy(&commLock);

    return;
}

static int voclGetAppIndex()
{
	int i, returnIndex;
    /* if memory is not enought */
    if (voclAppNo >= voclAppNum) {
        voclAppNum *= 2;
        conMsgRequest =
            (MPI_Request *) realloc(conMsgRequest,
                                    (voclAppNum * CMSG_NUM) * sizeof(MPI_Request));
		conMsgRequestForWait = (MPI_Request *)realloc(conMsgRequestForWait, 
				voclAppNum * sizeof(MPI_Request));
		conMsgRequestIndex = (int *)realloc(conMsgRequestIndex, voclAppNum * sizeof(int));
        appComm = (MPI_Comm *) realloc(appComm, voclAppNum * sizeof(MPI_Comm));
        appCommData = (MPI_Comm *) realloc(appCommData, voclAppNum * sizeof(MPI_Comm));
		voclCommUsedFlag = (char *) realloc(voclCommUsedFlag, voclAppNum * sizeof(char));
		memset(&voclCommUsedFlag[voclAppNum/2], 0, sizeof(char) * (voclAppNum/2));
        conMsgBufferStartPtr =
            (char **) realloc(conMsgBufferStartPtr,
                                     voclAppNum * CMSG_NUM * sizeof(CON_MSG_BUFFER));
		/* allocate control message buffer */
		for (i = voclAppNum/2*CMSG_NUM; i <= (voclAppNum*CMSG_NUM); i++)
		{
			conMsgBuffer[i] = (char *)malloc(sizeof(CON_MSG_BUFFER));
		}
    }
	
	/* search the communicator with the min index not being used */
	voclAppNo++;
	for (i = 0; i < voclAppNum; i++)
	{
		if (voclCommUsedFlag[i] == 0)
		{
			voclCommUsedFlag[i] = 1;
			return i;
		}
	}
}

/* a new app process is connected */
void voclIssueConMsgIrecv(int index)
{
    int i;
	/* get the locker and issue irecv */
    /* issue MPI_Irecv for control message */
    for (i = 0; i < CMSG_NUM; i++) {
        MPI_Irecv(conMsgBuffer[index * CMSG_NUM + i], MAX_CMSG_SIZE, MPI_BYTE, MPI_ANY_SOURCE,
                  MPI_ANY_TAG, appComm[index], conMsgRequest+(index * CMSG_NUM + i));
    }
	conMsgRequestIndex[index] = index * CMSG_NUM;
	conMsgRequestForWait[index] = conMsgRequest[conMsgRequestIndex[index]];

	if (index >= voclCommUsedSize)
	{
		voclCommUsedSize = index + 1;
		voclTotalRequestNum = voclCommUsedSize * CMSG_NUM;
	}

    return;
}

static MPI_Comm *voclGetAppCommPtr(int index)
{
    return &appComm[index];
}

void voclProxyAcceptOneApp()
{
    int index;
    index = voclGetAppIndex();
    MPI_Comm_accept(voclPortName, MPI_INFO_NULL, 0, MPI_COMM_SELF, voclGetAppCommPtr(index));
    MPI_Comm_dup(appComm[index], &appCommData[index]);
    voclIssueConMsgIrecv(index);

    return;
}

void voclProxyDisconnectOneApp(int commIndex)
{
	int requestOffset, requestNo, i;
	requestOffset = commIndex * CMSG_NUM;

	/*disconnect connections */
	MPI_Comm_disconnect(&appComm[commIndex]);
	MPI_Comm_disconnect(&appCommData[commIndex]);

	/* get the locker to update communicator info */
	pthread_mutex_lock(&commLock);
	/* if not the last communicator, move the last */
	/* communicator to the disconnected one*/
	voclAppNo--;
	voclCommUsedFlag[commIndex] = 0;
	for (requestNo = 0; requestNo < CMSG_NUM; requestNo++)
	{
		conMsgRequest[requestOffset + requestNo] = MPI_REQUEST_NULL;
	}
	conMsgRequestForWait[commIndex] = MPI_REQUEST_NULL;

	if (commIndex == voclCommUsedSize - 1)
	{
		for (i = voclCommUsedSize - 1; i >= 0; i--)
		{
			if (voclCommUsedFlag[i] != 0)
			{
				break;
			}
		}

		voclCommUsedSize = i+1;
		voclTotalRequestNum = voclCommUsedSize * CMSG_NUM;
	}

	/* release the locker */
	pthread_mutex_unlock(&commLock);

	/* continuously checking whether there is a new application */
	/* issuing connection requests */
	while (voclAppNo == 0)
	{
		sleep(1);
	}

	return;
}


void *proxyCommAcceptThread(void *p)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(8, &set);
    sched_setaffinity(0, sizeof(set), &set);

    int index, tmp;
    MPI_Comm comm;

    while (1) {
        MPI_Comm_accept(voclPortName, MPI_INFO_NULL, 0, MPI_COMM_SELF, &comm);
		/* lock the mutex to do update */
		pthread_mutex_lock(&commLock);
        index = voclGetAppIndex();
        appComm[index] = comm;
        MPI_Comm_dup(appComm[index], &appCommData[index]);
        voclIssueConMsgIrecv(index);
		pthread_mutex_unlock(&commLock);
    }

    return NULL;
}

