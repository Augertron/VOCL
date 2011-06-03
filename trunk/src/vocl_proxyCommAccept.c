#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <mpi.h>
#include "vocl_proxy_macro.h"

static int voclProxyTerminateFlag = 0;
static int voclAppNo;
static int voclAppNum;

extern MPI_Request *conMsgRequest;
extern MPI_Comm *appComm, *appCommData;
extern int voclTotalRequestNum;
extern char voclPortName[MPI_MAX_PORT_NAME];
extern CON_MSG_BUFFER *conMsgBuffer;

pthread_t thAppComm;

void voclProxyCommInitialize()
{
    voclAppNum = DEFAULT_APP_NUM;
    voclAppNo = 0;
    voclTotalRequestNum = 0;
    conMsgRequest = (MPI_Request *) malloc(voclAppNum * CMSG_NUM * sizeof(MPI_Request));
    appComm = (MPI_Comm *) malloc(voclAppNum * sizeof(MPI_Comm));
    appCommData = (MPI_Comm *) malloc(voclAppNum * sizeof(MPI_Comm));
    conMsgBuffer = (CON_MSG_BUFFER *) malloc(voclAppNum * CMSG_NUM * sizeof(CON_MSG_BUFFER));
}

void voclProxyCommFinalize()
{
    int i;
    free(conMsgRequest);
    free(appComm);
    free(appCommData);
    free(conMsgBuffer);

    return;
}

int voclGetAppIndex()
{
    /* if memory is not enought */
    if (voclAppNo >= voclAppNum) {
        voclAppNum *= 2;
        conMsgRequest =
            (MPI_Request *) realloc(conMsgRequest,
                                    voclAppNum * CMSG_NUM * sizeof(MPI_Request));
        appComm = (MPI_Comm *) realloc(appComm, voclAppNum * sizeof(MPI_Comm));
        appCommData = (MPI_Comm *) realloc(appCommData, voclAppNum * sizeof(MPI_Comm));
        conMsgBuffer =
            (CON_MSG_BUFFER *) realloc(conMsgBuffer,
                                       voclAppNum * CMSG_NUM * sizeof(CON_MSG_BUFFER));
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
                  MPI_ANY_TAG, appComm[index], conMsgRequest + index * CMSG_NUM + i);
    }
    /* duplicate the communicator */
    voclTotalRequestNum += CMSG_NUM;
    return;
}

MPI_Comm *voclGetAppCommPtr(int index)
{
    return &appComm[index];
}

void voclProxySetTerminateFlag(int flag)
{
    voclProxyTerminateFlag = flag;
}

int voclProxyGetAppNum()
{
    return voclAppNo;
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


void *proxyCommAcceptThread(void *p)
{
    int index, oldType;
    MPI_Comm comm;
    if (pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, &oldType) != 0) {
        printf("set aynchronous interrupt stataus error!\n");
        exit(1);
    }

    while (voclProxyTerminateFlag == 0) {
        MPI_Comm_accept(voclPortName, MPI_INFO_NULL, 0, MPI_COMM_SELF, &comm);
        index = voclGetAppIndex();
        appComm[index] = comm;
        MPI_Comm_dup(appComm[index], &appCommData[index]);
        voclIssueConMsgIrecv(index);
    }

    return NULL;
}
