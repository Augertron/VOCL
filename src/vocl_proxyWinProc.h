#ifndef __VOCL_PROXY_WIN_PROC_H__ 
#define __VOCL_PROXY_WIN_PROC_H__

#include <stdio.h>
#include <CL/opencl.h>
#include "mpi.h"
#include "vocl_proxy_macro.h"

struct strVoclWinInfo {
    char serviceName[SERVICE_NAME_LEN];
    int  proxyRank; /* rank no within the proxy comm_world */
	int  appIndex;
    MPI_Comm commProxy;
    MPI_Comm commWin;  /* MPI communicator for win creation */
};

typedef struct strVoclProxyWinInfoAll {
    int proxyNum;
	int migrationStatus;
	int targetProxyIndex;
    struct strVoclWinInfo wins[DEFAULT_PROXY_NUM];
} vocl_proxy_wins;

#endif
