#include <stdio.h>
#include "voclOpencl.h"
#include "voclStructures.h"
#include "voclMigration.h"
#include "voclKernelArgProc.h"

extern size_t voclGetVGPUMsgSize(int proxyIndex, vocl_device_id device);
extern void voclPackVGPUMigMsg(int proxyIndex, vocl_device_id device, char *msgBuf);
extern void voclUpdateVirtualGPU(int proxyIndex, vocl_device_id device,
               int newProxyRank, int newProxyIndex, MPI_Comm newProxyComm, 
			   MPI_Comm newProxyCommData, char *msgBuf);

/* whether migration is needed. no migration by default */
static int voclTaskMigrationCheckCondition = 0;

/* set whether migration is needed according to an environment variable */
void voclSetTaskMigrationCondition()
{
	char *migrationConditionsPtr, *tmpFlagPtr;
	char migrationConditions[][20] = {{"MEMORY_FULL"}};
	char *conditionList;
	size_t len;

	migrationConditionsPtr = getenv("VOCL_MIGRATION_CONDITION");
	if (migrationConditionsPtr == NULL)
	{
		voclTaskMigrationCheckCondition = 0;
	}
	else
	{
		len = strlen(migrationConditionsPtr) + 1;
		conditionList = (char *)malloc(sizeof(char) * len);
		strcpy(conditionList, migrationConditionsPtr);

		tmpFlagPtr = strtok(conditionList, ",");
		while (tmpFlagPtr != NULL)
		{
			if (strcmp(tmpFlagPtr, migrationConditions[0]) == 0)
			{
				voclTaskMigrationCheckCondition = 1;
			}
			else //more conditions are added later
			{
				
			}
			tmpFlagPtr = strtok(NULL, ",");
		}

		free(conditionList);
	}

	return;
}

int voclGetTaskMigrationCondition()
{
	return voclTaskMigrationCheckCondition;
}

void voclMigUpdateVirtualGPU(int origProxyIndex, vocl_device_id origDeviceID, 
			int proxyRank, int proxyIndex, MPI_Comm comm, MPI_Comm commData)
{
	size_t vgpuMigMsgSize;
	char *msgBuf;
	struct strMigUpdateVGPU tmpMigUpdateVGPU;
	MPI_Request request[3];
	MPI_Status status[3];
	int requestNo = 0;

	/* get msg size for vgpu info update */
	vgpuMigMsgSize = voclGetVGPUMsgSize(origProxyIndex, origDeviceID);
	msgBuf = (char *)malloc(vgpuMigMsgSize);

	/* pack up the message */
	voclPackVGPUMigMsg(origProxyIndex, origDeviceID, msgBuf);
	tmpMigUpdateVGPU.msgSize = vgpuMigMsgSize;
	MPI_Isend(&tmpMigUpdateVGPU, sizeof(struct strMigUpdateVGPU), MPI_BYTE, proxyRank,
			  VOCL_UPDATE_VGPU, comm, request+(requestNo++));
	MPI_Isend(msgBuf, vgpuMigMsgSize, MPI_BYTE, proxyRank, VOCL_UPDATE_VGPU,
			  commData, request+(requestNo++));
	MPI_Irecv(msgBuf, vgpuMigMsgSize, MPI_BYTE, proxyRank, VOCL_UPDATE_VGPU,
			  commData, request+(requestNo++));
	MPI_Waitall(requestNo, request, status);

	voclUpdateVirtualGPU(origProxyIndex, origDeviceID, proxyRank, proxyIndex,
						 comm, commData, msgBuf);
	free(msgBuf);

	return;
}

