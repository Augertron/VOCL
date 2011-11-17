#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "voclOpencl.h"
#include "voclStructures.h"
#include "voclMigration.h"
#include "voclKernelArgProc.h"

extern size_t voclGetVGPUMsgSize(int proxyIndex, vocl_device_id device);
extern void voclPackVGPUMigMsg(int proxyIndex, vocl_device_id device, char *msgBuf);
extern void voclUpdateVirtualGPU(int proxyIndex, vocl_device_id device,
               int newProxyRank, int newProxyIndex, MPI_Comm newProxyComm, 
			   MPI_Comm newProxyCommData, char *msgBuf);
extern cl_mem voclVOCLMemory2CLMemory(vocl_mem memory);

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
	//struct timeval t1, t2;
	//float tmpTime;

	//gettimeofday(&t1, NULL);
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
	//gettimeofday(&t2, NULL);
	//tmpTime = 1000.0 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
	//printf("updateVGPU = %.3f\n", tmpTime);
	free(msgBuf);

	return;
}

/* send message to the source proxy process to indicate it is the */
/* last message before migration. Then the proxy process can migrate */
/* all commands that are not issued to the destination proxy process */
void voclMigSendLastMsgToOrigProxy(int origProxyIndex, int origProxyRank,
			MPI_Comm comm, MPI_Comm commData, int *reissueWriteNum, 
			int *reissueReadNum)
{
	MPI_Request request[2];
	MPI_Status status[2];
	int requestNo = 0;
	struct strMigSendLastMessage lastMsg;
	MPI_Isend(&lastMsg, sizeof(struct strMigSendLastMessage), MPI_BYTE, 
			origProxyRank, VOCL_MIG_LAST_MSG, comm, request+(requestNo++));
	MPI_Irecv(&lastMsg, sizeof(struct strMigSendLastMessage), MPI_BYTE, 
			origProxyRank, VOCL_MIG_LAST_MSG, comm, request+(requestNo++));
	MPI_Waitall(requestNo, request, status);

	*reissueWriteNum = lastMsg.reissueWriteNum;
	*reissueReadNum = lastMsg.reissueReadNum;

	return;
}

/* this function will block the execution of the vocl */
/* lib till migration is completed on the proxy process */
/* the target proxy process is in migration */
void voclMigIsProxyInMigration(int proxyIndex, int proxyRank, 
		MPI_Comm comm, MPI_Comm commData)
{
	MPI_Request request[3];
	MPI_Status status[3];
	int requestNo;
	int isInMigration;

	/*send a null message to the proxy process to the proxy 
	to check whether migration is being performed */
	requestNo = 0;
	MPI_Isend(NULL, 0, MPI_BYTE, proxyRank, VOCL_CHK_PROYX_INMIG, comm, request+(requestNo++));
	MPI_Irecv(&isInMigration, sizeof(int), MPI_BYTE, proxyRank, VOCL_CHK_PROYX_INMIG, 
			  comm, request+(requestNo++));
	MPI_Waitall(requestNo, request, status);
	
	/* if proxy is in migration, wait until migration is completed */
	if (isInMigration == 1)
	{
		MPI_Irecv(NULL, 0, MPI_BYTE, proxyRank, VOCL_CHK_PROYX_INMIG, comm, request);
		MPI_Wait(request, status);
	}

	return;
}

void voclMigUpdateKernelArgs(kernel_info *kernelPtr)
{
	cl_uint i, argIndex, argNum;
	kernel_args *argPtr;
	cl_mem mem;

	argNum = kernelPtr->args_num;
	argPtr = kernelPtr->args_ptr;

	/* if migration happens, mem handles need to be updated to migrated values */
	for (i = 0; i < argNum; i++)
	{
		argIndex = argPtr[i].arg_index;
		/* it is a memory */
		if (kernelPtr->args_flag[argIndex] == 1)
		{
			mem = voclVOCLMemory2CLMemory((vocl_mem)argPtr[i].memory);
			memcpy(argPtr[i].arg_value, (void *)&mem, argPtr[i].arg_size);
		}
	}

	return;
}

