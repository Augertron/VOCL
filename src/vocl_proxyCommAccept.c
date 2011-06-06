#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <mpi.h>
#include "vocl_proxy_macro.h"

static int voclAppNo;
static int voclAppNum;
static char **conMsgBufferStartPtr;

extern MPI_Request *conMsgRequest;
extern MPI_Comm *appComm, *appCommData;
extern int voclTotalRequestNum;
extern char voclPortName[MPI_MAX_PORT_NAME];
extern char **conMsgBuffer;

pthread_t thAppComm;

void voclProxyCommInitialize()
{
	int i;
    voclAppNum = DEFAULT_APP_NUM;
    voclAppNo = 0;
    voclTotalRequestNum = 0;
    conMsgRequest = (MPI_Request *) malloc(voclAppNum * CMSG_NUM * sizeof(MPI_Request));
    appComm = (MPI_Comm *) malloc(voclAppNum * sizeof(MPI_Comm));
    appCommData = (MPI_Comm *) malloc(voclAppNum * sizeof(MPI_Comm));
	conMsgBuffer = (char **)malloc(voclAppNum * CMSG_NUM * sizeof(char *));
	conMsgBufferStartPtr = conMsgBuffer;
	for (i = 0; i < voclAppNum*CMSG_NUM; i++)
	{
		conMsgBuffer[i] = (char *)malloc(sizeof(CON_MSG_BUFFER));
	}
}

void voclProxyCommFinalize()
{
    int i;
	int totalBufferNum;
	totalBufferNum = voclAppNum * CMSG_NUM;

    free(conMsgRequest);
    free(appComm);
    free(appCommData);

	/* free control msg buffer */
	for (i = 0; i < totalBufferNum; i++)
	{
		free(conMsgBuffer[i]);
	}
	
	/* free control msg pointer buffer */
    free(conMsgBufferStartPtr);

    return;
}

static int voclGetAppIndex()
{
	int i;
    /* if memory is not enought */
    if (voclAppNo >= voclAppNum) {
        voclAppNum *= 2;
        conMsgRequest =
            (MPI_Request *) realloc(conMsgRequest,
                                    (voclAppNum * CMSG_NUM) * sizeof(MPI_Request));
        appComm = (MPI_Comm *) realloc(appComm, voclAppNum * sizeof(MPI_Comm));
        appCommData = (MPI_Comm *) realloc(appCommData, voclAppNum * sizeof(MPI_Comm));
        conMsgBufferStartPtr =
            (char **) realloc(conMsgBufferStartPtr,
                                     voclAppNum * CMSG_NUM * sizeof(CON_MSG_BUFFER));
		/* allocate control message buffer */
		for (i = voclAppNum/2; i <= voclAppNum; i++)
		{
			conMsgBuffer[i] = (char *)malloc(sizeof(CON_MSG_BUFFER));
		}
    }

    return voclAppNo++;
}

/* a new app process is connected */
void voclIssueConMsgIrecv(int index)
{
    int i;
    /* issue MPI_Irecv for control message */
    for (i = 0; i < CMSG_NUM; i++) {
        MPI_Irecv(conMsgBuffer[index * CMSG_NUM + i], MAX_CMSG_SIZE, MPI_BYTE, MPI_ANY_SOURCE,
                  MPI_ANY_TAG, appComm[index], conMsgRequest+(index * CMSG_NUM + i));
    }

    voclTotalRequestNum += CMSG_NUM;
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
	int requestOffset, lastAppOffset, requestNo;
	char *temp;
	requestOffset = commIndex * CMSG_NUM;
	lastAppOffset = (voclAppNo - 1) * CMSG_NUM;

	/*disconnect connections */
	MPI_Comm_disconnect(&appComm[commIndex]);
	MPI_Comm_disconnect(&appCommData[commIndex]);

	/* if not the last communicator, move the last */
	/* communicator to the disconnected one*/
	if (commIndex < voclAppNo - 1)
	{
		/* for communicators */
		appComm[commIndex] = appComm[voclAppNo - 1];
		appCommData[commIndex] = appCommData[voclAppNo - 1];
		for (requestNo = 0; requestNo < CMSG_NUM; requestNo++)
		{
			/* for irecv requests */
			conMsgRequest[requestOffset + requestNo] = conMsgRequest[lastAppOffset + requestNo];
			conMsgRequest[lastAppOffset + requestNo] = MPI_REQUEST_NULL;
			
			/* for control msg buffers */
			temp = conMsgBuffer[requestOffset + requestNo];
			conMsgBuffer[requestOffset + requestNo] = conMsgBuffer[lastAppOffset + requestNo];
			conMsgBuffer[lastAppOffset + requestNo] = temp;
		}
	}
	else
	{
		for (requestNo = 0; requestNo < CMSG_NUM; requestNo++)
		{
			conMsgRequest[lastAppOffset + requestNo] = MPI_REQUEST_NULL;
		}
	}

	voclTotalRequestNum -= CMSG_NUM;
	voclAppNo--;

	/* continuously checking whether a new application issue connection requests */
	while (voclAppNo == 0)
	{
		sleep(1);
	}

	return;
}


void *proxyCommAcceptThread(void *p)
{
    int index, oldType;
    MPI_Comm comm;
    if (pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, &oldType) != 0) {
        printf("set aynchronous interrupt stataus error!\n");
        exit(1);
    }

    while (1) {
        MPI_Comm_accept(voclPortName, MPI_INFO_NULL, 0, MPI_COMM_SELF, &comm);
        index = voclGetAppIndex();
        appComm[index] = comm;
        MPI_Comm_dup(appComm[index], &appCommData[index]);
        voclIssueConMsgIrecv(index);
    }

    return NULL;
}
