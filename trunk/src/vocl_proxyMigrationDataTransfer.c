#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>
#include "vocl_proxy_macro.h"
#include "vocl_proxyBufferProc.h"

static struct strMigWriteBufferInfo voclMigWriteBuffers[VOCL_MIG_BUF_NUM];
static int voclMigWriteBufferIndex;
static int voclMigWriteBufferRequestNum;

void voclMigWriteBufferInitialize()
{
	int i;
	for (i = 0; i < VOCL_MIG_BUF_NUM; i++)
	{
		voclMigWriteBuffers[i].cmdQueue = NULL;
		voclMigWriteBuffers[i].memory = NULL;
		voclMigWriteBuffers[i].size = 0;
		voclMigWriteBuffers[i].offset = 0;
		voclMigWriteBuffers[i].event = NULL;
		voclMigWriteBuffers[i].source = -1;
		voclMigWriteBuffers[i].tag = -1;
		voclMigWriteBuffers[i].useFlag = MIG_WRT_AVAILABLE;
		voclMigWriteBuffers[i].ptr = (char *)malloc(VOCL_MIG_BUF_SIZE*sizeof(char));
	}
	voclMigWriteBufferIndex = 0;
	voclMigWriteBufferRequestNum = 0;
}

void voclMigWriteBufferFinalize()
{
	int i;
	for (i = 0; i < VOCL_MIG_BUF_NUM; i++)
	{
		voclMigWriteBuffers[i].useFlag = MIG_WRT_AVAILABLE;
		free(voclMigWriteBuffers[i].ptr);
		voclMigWriteBuffers[i].ptr = NULL;
	}
	voclMigWriteBufferIndex = 0;
	voclMigWriteBufferRequestNum = 0;
}

void voclMigSetWriteBufferFlag(int index, int flag)
{
	voclMigWriteBuffers[index].useFlag = flag;
	return;
}

struct strMigWriteBufferInfo *voclMigGetWriteBufferPtr(int index)
{
	return &voclMigWriteBuffers[index];
}

MPI_Request *voclMigGetWriteRequestPtr(int index)
{
	return  &voclMigWriteBuffers[index].request;
}

static int voclMigWriteToGPUMemory(int index)
{
	int err;
	err = clEnqueueWriteBuffer(voclMigWriteBuffers[index].cmdQueue,
							   voclMigWriteBuffers[index].memory,
							   CL_FALSE,
							   voclMigWriteBuffers[index].offset,
							   voclMigWriteBuffers[index].size,
							   voclMigWriteBuffers[index].ptr,
							   0, NULL, &voclMigWriteBuffers[index].event);
	return err;
}

int voclMigGetNextWriteBufferIndex()
{
	int index;
	MPI_Status status;
	index = voclMigWriteBufferIndex;
	if (voclMigWriteBuffers[index].useFlag == MIG_WRT_MPIRECV)
	{
		MPI_Wait(&voclMigWriteBuffers[index].request, &status);
		voclMigWriteToGPUMemory(index);
		clWaitForEvents(1, &voclMigWriteBuffers[index].event);
	}
	else if (voclMigWriteBuffers[index].useFlag == MIG_WRT_WRTGPU)
	{
		 clWaitForEvents(1, &voclMigWriteBuffers[index].event);
	}
	voclMigSetWriteBufferFlag(index, MIG_WRT_AVAILABLE);

	/* if all buffer is used, start from 0 */
	if (++voclMigWriteBufferIndex >= VOCL_MIG_BUF_NUM)
	{
		voclMigWriteBufferIndex = 0;
	}

	/* at most VOCL_MIG_BUF_NUM buffers is in use */
	if (++voclMigWriteBufferRequestNum > VOCL_MIG_BUF_NUM)
	{
		voclMigWriteBufferRequestNum = VOCL_MIG_BUF_NUM;
	}

	return index;
}

int voclMigFinishDataWrite(int source)
{
	int i, index, err = CL_SUCCESS;
	int startIndex, endIndex;
	cl_event eventList[VOCL_MIG_BUF_NUM];
	int activeBufferFlag[VOCL_MIG_BUF_NUM];
	int eventNo =  0;
	MPI_Status status;

	endIndex = voclMigWriteBufferIndex;
	startIndex = endIndex - voclMigWriteBufferRequestNum;
	if (startIndex < 0)
	{
		startIndex += VOCL_MIG_BUF_NUM;
		endIndex += VOCL_MIG_BUF_NUM;
	}

	for (i = startIndex; i < endIndex; i++)
	{
		index = i % VOCL_MIG_BUF_NUM;
		activeBufferFlag[index] = 0;
		if (voclMigWriteBuffers[index].source == source)
		{
			if (voclMigWriteBuffers[index].useFlag == MIG_WRT_MPIRECV)
			{
				MPI_Wait(&voclMigWriteBuffers[index].request, &status);
				voclMigWriteToGPUMemory(index);
				eventList[eventNo++] = voclMigWriteBuffers[index].event;
				activeBufferFlag[index] = 1;
			}
			else if (voclMigWriteBuffers[index].useFlag == MIG_WRT_WRTGPU)
			{
				eventList[eventNo++] = voclMigWriteBuffers[index].event;
				activeBufferFlag[index] = 1;
			}
		}
	}

	if (eventNo > 0)
	{
		err = clWaitForEvents(eventNo, eventList);
	}

	for (i = startIndex; i < endIndex; i++)
	{
		index = i % VOCL_MIG_BUF_NUM;
		if (voclMigWriteBuffers[index].source == source && activeBufferFlag[index] == 1)
		{
			voclMigSetWriteBufferFlag(index, MIG_WRT_AVAILABLE);
		}
	}

	return err;
}


static struct strMigReadBufferInfo voclMigReadBuffers[VOCL_MIG_BUF_NUM];
static int voclMigReadBufferIndex;
static int voclMigReadBufferRequestNum;

void voclMigReadBufferInitialize()
{
	int i;
	for (i = 0; i < VOCL_MIG_BUF_NUM; i++)
	{
		voclMigReadBuffers[i].size = 0;
		voclMigReadBuffers[i].offset = 0;
		voclMigReadBuffers[i].event = NULL;
		voclMigReadBuffers[i].dest = -1;
		voclMigReadBuffers[i].tag = -1;
		voclMigReadBuffers[i].useFlag = MIG_READ_AVAILABLE;
		voclMigReadBuffers[i].ptr = (char *)malloc(VOCL_MIG_BUF_SIZE*sizeof(char));
	}
	voclMigReadBufferIndex = 0;
	voclMigReadBufferRequestNum = 0;
}

void voclMigReadBufferFinalize()
{
	int i;
	for (i = 0; i < VOCL_MIG_BUF_NUM; i++)
	{
		voclMigReadBuffers[i].size = 0;
		voclMigReadBuffers[i].offset = 0;
		voclMigReadBuffers[i].event = NULL;
		voclMigReadBuffers[i].dest = -1;
		voclMigReadBuffers[i].tag = -1;
		voclMigReadBuffers[i].useFlag = MIG_READ_AVAILABLE;
		free(voclMigReadBuffers[i].ptr); 
		voclMigReadBuffers[i].ptr = NULL;
	}
	voclMigReadBufferIndex = 0;
	voclMigReadBufferRequestNum = 0;
}

void voclMigSetReadBufferFlag(int index, int flag)
{
	voclMigReadBuffers[index].useFlag = flag;
	return;
}

cl_event *voclMigGetReadEventPtr(int index)
{
	return &voclMigReadBuffers[index].event;
}

struct strMigReadBufferInfo *voclMigGetReadBufferPtr(int index)
{
	return &voclMigReadBuffers[index];
}

static int voclMigSendDataToTarget(int index)
{
	int err;
	err = MPI_Isend(voclMigReadBuffers[index].ptr, 
					voclMigReadBuffers[index].size, 
					MPI_BYTE,
					voclMigReadBuffers[index].dest,
					voclMigReadBuffers[index].tag,
					MPI_COMM_WORLD,
					&voclMigReadBuffers[index].request);
	return err;
}

int voclMigGetNextReadBufferIndex()
{
	int index;
	MPI_Status status;
	index = voclMigReadBufferIndex;
	if (voclMigReadBuffers[index].useFlag == MIG_READ_RDGPU)
	{
		clWaitForEvents(1, &voclMigReadBuffers[index].event);
		voclMigSendDataToTarget(index);
		MPI_Wait(&voclMigReadBuffers[index].request, &status);
	}
	else if (voclMigWriteBuffers[index].useFlag == MIG_READ_MPISEND)
	{
		MPI_Wait(&voclMigReadBuffers[index].request, &status);
	}
	voclMigSetReadBufferFlag(index, MIG_READ_AVAILABLE);

	/* if all buffer is used, start from 0 */
	if (++voclMigReadBufferIndex >= VOCL_MIG_BUF_NUM)
	{
		voclMigReadBufferIndex = 0;
	}

	/* at most VOCL_MIG_BUF_NUM buffers is in use */
	if (++voclMigReadBufferRequestNum > VOCL_MIG_BUF_NUM)
	{
		voclMigReadBufferRequestNum = VOCL_MIG_BUF_NUM;
	}

	return index;
}

int voclMigFinishDataRead(int dest)
{
	int i, index, err = MPI_SUCCESS;
	int startIndex, endIndex;
	MPI_Request requestList[VOCL_MIG_BUF_NUM];
	MPI_Status  status[VOCL_MIG_BUF_NUM];
	int activeBufferFlag[VOCL_MIG_BUF_NUM];
	int requestNo = 0;

	endIndex = voclMigReadBufferIndex;
	startIndex = endIndex - voclMigReadBufferRequestNum;
	if (startIndex < 0)
	{
		startIndex += VOCL_MIG_BUF_NUM;
		endIndex += VOCL_MIG_BUF_NUM;
	}

	requestNo = 0;
	for (i = startIndex; i < endIndex; i++)
	{
		index = i % VOCL_MIG_BUF_NUM;
		activeBufferFlag[index] = 0;
		if (voclMigReadBuffers[index].dest == dest)
		{
			if (voclMigReadBuffers[index].useFlag == MIG_READ_RDGPU)
			{
				clWaitForEvents(1, &voclMigReadBuffers[index].event);
				voclMigSendDataToTarget(index);
				requestList[requestNo++] = voclMigReadBuffers[index].request;
				activeBufferFlag[index] = 1;
			}
			else if (voclMigReadBuffers[index].useFlag == MIG_READ_MPISEND)
			{
				requestList[requestNo++] = voclMigReadBuffers[index].request;
				activeBufferFlag[index] = 1;
			}
		}
	}
	
	if (requestNo > 0)
	{
		err = MPI_Waitall(requestNo, requestList, status);
	}

	for (i = startIndex; i < endIndex; i++)
	{
		index = i % VOCL_MIG_BUF_NUM;
		if (voclMigReadBuffers[index].dest == dest && activeBufferFlag[index] == 1)
		{
			voclMigSetReadBufferFlag(index, MIG_READ_AVAILABLE);
		}
	}

	return err;
}

