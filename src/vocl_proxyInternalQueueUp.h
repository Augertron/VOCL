#ifndef __VOCL_PROXY_CMD_QUEUE_H__
#define __VOCL_PROXY_CMD_QUEUE_H__

#include <stdio.h>
#include <CL/opencl.h>

#define VOCL_PROXY_CMDQUEUE_SIZE 1024
#define VOCL_CMDQUEUE_IN_EXECUTION 20
#define VOCL_PROXY_CMD_AVABL 0
#define VOCL_PROXY_CMD_INUSE 1

struct strVoclCommandQueue {
	pthread_mutex_t  lock;
	int              msgTag;
	MPI_Comm         appComm;
	MPI_Comm         appCommData;
	int              appRank;
	int              appIndex;
	int              internalWaitFlag;
	int              status;
	char             conMsgBuffer[MAX_CMSG_SIZE];
};

#endif

