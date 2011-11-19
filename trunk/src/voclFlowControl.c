#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory.h>
#include "mpi.h"
#include "voclOpenclMacro.h"

static int *voclNumOfKernelPerCheck = NULL;

void voclConMsgFlowInitialize()
{
	voclNumOfKernelPerCheck = (int *) malloc(sizeof(int) * MAX_NPS);
	memset(voclNumOfKernelPerCheck, 0, sizeof(int) * MAX_NPS);

	return;
}

void voclConMsgFlowFinalize()
{
	free(voclNumOfKernelPerCheck);

	return;
}

void voclConMsgFlowControl(int proxyIndex, int proxyRank, MPI_Comm proxyComm)
{
	MPI_Status status;

	if (voclNumOfKernelPerCheck[proxyIndex] > VOCL_NUM_KERNELS_PER_CHECK)
	{
		MPI_Send(NULL, 0, MPI_BYTE, proxyRank, VOCL_CHK_PROYX_INMIG, proxyComm);
		MPI_Recv(NULL, 0, MPI_BYTE, proxyRank, VOCL_CHK_PROYX_INMIG, proxyComm, &status);
		voclNumOfKernelPerCheck[proxyIndex] = 0;
	}
	else
	{
		voclNumOfKernelPerCheck[proxyIndex]++;
	}

	return;
}
