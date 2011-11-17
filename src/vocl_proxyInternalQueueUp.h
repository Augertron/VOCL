#ifndef __VOCL_PROXY_CMD_QUEUE_H__
#define __VOCL_PROXY_CMD_QUEUE_H__

#include <stdio.h>
#include <CL/opencl.h>
#include "vocl_proxy_macro.h"

#define VOCL_PROXY_CMDQUEUE_SIZE 1024000
#define VOCL_CMDQUEUE_IN_EXECUTION 40
#define VOCL_PROXY_CMD_AVABL 0
#define VOCL_PROXY_CMD_INUSE 1
#define VOCL_PROXY_CMD_MIG   2

typedef struct strVoclCmddQueue {
	pthread_mutex_t  lock;
	int              msgTag;
	MPI_Comm         appComm;
	MPI_Comm         appCommData;
	int              appRank;
	int              appIndex;
	int              internalWaitFlag;
	int              status;
	char			 *paramBuf;
	char             conMsgBuffer[MAX_CMSG_SIZE];
} vocl_internal_command_queue;

#endif

