#include <string.h>
#include "voclOpencl.h"
#include "voclStructures.h"

extern cl_platform_id voclVOCLPlatformID2CLPlatformIDComm(vocl_platform_id platform, int *proxyRank,
                              int *proxyIndex, MPI_Comm * proxyComm,
                              MPI_Comm * proxyCommData);


struct strDeviceCmdQueueNums {
	int deviceNum;
	cl_device_id deviceIDs[MAX_DEVICE_NUM_PER_NODE];
	int cmdQueueNums[MAX_DEVICE_NUM_PER_NODE];
};

struct strVOCLLBCmdQueueNums {
	cl_device_id deviceID;
	unsigned int cmdQueueNum;
};

static unsigned int voclLBTotalDeviceNum = 0;
static struct strVOCLLBCmdQueueNums *voclLBCmdQueueNums = NULL;
static int voclLBCmdQueueNumBufferAllocated = 0;

void voclLBGetDeviceCmdQueueNums()
{
	int proxyIndex, proxyRank;
	MPI_Comm proxyComm, proxyCommData;
	MPI_Request *request;
	MPI_Status *status;
	struct strDeviceCmdQueueNums *cmdQueueNums;
	int totalDeviceNum, requestNo;
	cl_platform_id *platformIDs, tmpPlatformID;
	cl_uint platformNum, i, j;
	cl_int err;


	err = clGetPlatformIDs(0, NULL, &platformNum);
	platformIDs = (cl_platform_id *)malloc(sizeof(cl_platform_id) * platformNum);
	cmdQueueNums = (struct strDeviceCmdQueueNums *)malloc(sizeof(struct strDeviceCmdQueueNums) * platformNum);
	request = (MPI_Request *)malloc(sizeof(MPI_Request) * 2 * platformNum);
	status = (MPI_Status *)malloc(sizeof(MPI_Status) * 2 * platformNum);
	
	err = clGetPlatformIDs(platformNum, platformIDs, &platformNum);

	requestNo = 0;
	for (i = 0; i < platformNum; i++)
	{
		tmpPlatformID = voclVOCLPlatformID2CLPlatformIDComm((vocl_platform_id)platformIDs[i], 
							&proxyRank, &proxyIndex, &proxyComm, &proxyCommData);
		//MPI_Isend(&cmdQueueNums[i], sizeof(struct strVOCLLBCmdQueueNums), MPI_BYTE,
		MPI_Isend(NULL, 0, MPI_BYTE,
			proxyRank, LB_GET_CMDQUEUE_NUM, proxyComm, request+(requestNo++));
		MPI_Irecv(&cmdQueueNums[i], sizeof(struct strDeviceCmdQueueNums), MPI_BYTE,
			proxyRank, LB_GET_CMDQUEUE_NUM, proxyComm, request+(requestNo++));

	}
	MPI_Waitall(requestNo, request, status);

	totalDeviceNum = 0;
	for (i = 0; i < platformNum; i++)
	{
		totalDeviceNum += cmdQueueNums[i].deviceNum;
	}

	voclLBCmdQueueNums = (struct strVOCLLBCmdQueueNums *)malloc(sizeof(struct strVOCLLBCmdQueueNums) * totalDeviceNum);
	totalDeviceNum = 0;
	for (i = 0; i < platformNum; i++)
	{
		for (j = 0; j < cmdQueueNums[i].deviceNum; j++)
		{
			voclLBCmdQueueNums[totalDeviceNum].deviceID = cmdQueueNums[i].deviceIDs[j];
			voclLBCmdQueueNums[totalDeviceNum].cmdQueueNum = cmdQueueNums[i].cmdQueueNums[j];
			totalDeviceNum++;
		}
	}

	voclLBTotalDeviceNum = totalDeviceNum;

	free(platformIDs);
	free(cmdQueueNums);
	free(request);
	free(status);

	return;
}

