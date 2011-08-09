#include <stdio.h>
#include <CL/opencl.h>
#include "vocl_proxyKernelArgProc.h"

struct strVoclCommandQueue {
	pthread_mutex_t  lock;
	int              msgTag;
	MPI_Comm         appComm;
	MPI_Comm         appCommData;
	int              appRank;
	int              appIndex;
	int              status;
	char             conMsgBuffer[MAX_CMSG_SIZE];
};

#define CMD_NUM_IN_QUEUE 32

#define VOCL_PROXY_CMD_AVABL 0
#define VOCL_PROXY_CMD_INUSE 1

static struct strVoclCommandQueue *voclProxyCmdQueue = NULL;
static int voclProxyCmdNum;
static int voclProxyCmdHead;
static int voclProxyCmdTail;

static struct strEnqueueWriteBuffer tmpEnqueueWriteBuffer;
static struct strEnqueueNDRangeKernel tmpEnqueueNDRangeKernel;
static struct strEnqueueReadBuffer tmpEnqueueReadBuffer;
static struct strFinish tmpFinish;

void voclProxyCmdQueueInit()
{
	int i;
	voclProxyCmdNum = CMD_NUM_IN_QUEUE;
	voclProxyCmdHead = 0;
	voclProxyCmdTail = 0;
	voclProxyCmdQueue = (struct strVoclCommandQueue *)malloc(sizeof(struct strVoclCommandQueue) * voclProxyCmdNum);

	for (i = 0; i < voclProxyCmdNum; i++)
	{
		pthread_mutex_init(&voclProxyCmdQueue[i].lock, NULL);
		voclProxyCmdQueue[i].msgTag = 0;  /* invalid msg tag */
		voclProxyCmdQueue[i].status = VOCL_PROXY_CMD_AVABL;
	}

	return;
}

struct strVoclCommandQueue * voclProxyGetCmdQueueTail()
{
	int index;
	index = voclProxyCmdTail;

	pthread_mutex_lock(&voclProxyCmdQueue[i].lock);
	voclProxyCmdTail++;
	if (voclProxyCmdTail >= voclProxyCmdNum)
	{
		voclProxyCmdTail = 0;
	}

	return &voclProxyCmdQueue[index];
}

struct strVoclCommandQueue * voclProxyGetCmdQueueHead()
{
	int index;
	index = voclProxyCmdHead;

	pthread_mutex_lock(&voclProxyCmdQueue[i].lock);
	voclProxyCmdHead++;
	if (voclProxyCmdHead >= voclProxyCmdNum)
	{
		voclProxyCmdHead = 0;
	}

	return &voclProxyCmdQueue[index];
}

void voclProxyCmdQueueReset()
{
	int i;

	for (i = 0; i < voclProxyCmdNum; i++)
	{
		voclProxyCmdQueue[i].status = VOCL_PROXY_CMD_AVABL;
		voclProxyCmdQueue[i].msgTag = 0;  /* invalid msg tag */
	}

	voclProxyCmdHead = 0;
	voclProxyCmdTail = 0;

	return;
}

void voclProxyCmdQueueFinalize()
{
	int i;

	for (i = 0; i < voclProxyCmdNum; i++)
	{
		voclProxyCmdQueue[i].status = VOCL_PROXY_CMD_AVABL;
	}

	free(voclProxyCmdQueue);

	voclProxyCmdNum = 0;
	voclProxyCmdHead = 0;
	voclProxyCmdTail = 0;

	return;
}

void *voclProxyEnqueueThread(void *p)
{
	struct strVoclCommandQueue *cmdQueuePtr;

	cl_event *event_wait_list;
	cl_uint num_events_in_wait_list;
	int requestNo, bufferNum, bufferIndex, i;
	int appIndex, appRank;
	MPI_Comm appComm, appCommData;
	size_t bufferSize, remainingSize;
    struct strWriteBufferInfo *writeBufferInfoPtr;
	struct strReadBufferInfo *readBufferInfoPtr;

	MPI_Request curRequest[50];
	MPI_Status  curStatus[50];

	while (true)
	{
		cmdQueuePtr = voclProxyGetCmdQueueHead();
		appComm = cmdQueuePtr->appComm;
		appCommData = cmdQueuePtr->appCommData;
		appRank = cmdQueuePtr->appRank;
		appIndex = cmdQueuePtr->appIndex;
		if (cmdQueuePtr->msgTag == ENQUEUE_WRITE_BUFFER)
		{
			memcpy(&tmpEnqueueWriteBuffer, cmdQueuePtr->conMsgBuffer, sizeof(struct strEnqueueWriteBuffer));
			requestNo = 0;
			event_wait_list = NULL;
			num_events_in_wait_list = tmpEnqueueWriteBuffer.num_events_in_wait_list;
			if (num_events_in_wait_list > 0) {
				event_wait_list =
					(cl_event *) malloc(sizeof(cl_event) * num_events_in_wait_list);
				MPI_Irecv(event_wait_list, sizeof(cl_event) * num_events_in_wait_list,
						  MPI_BYTE, appRank, tmpEnqueueWriteBuffer.tag, 
						  appCommData, curRequest + (requestNo++));
			}

			/* issue MPI data receive */
			bufferSize = VOCL_PROXY_WRITE_BUFFER_SIZE;
			bufferNum = (tmpEnqueueWriteBuffer.cb - 1) / bufferSize;
			remainingSize = tmpEnqueueWriteBuffer.cb - bufferSize * bufferNum;
			for (i = 0; i <= bufferNum; i++) {
				if (i == bufferNum)
					bufferSize = remainingSize;

				bufferIndex = getNextWriteBufferIndex(appIndex);
				writeBufferInfoPtr = getWriteBufferInfoPtr(appIndex, bufferIndex);
				MPI_Irecv(writeBufferInfoPtr->dataPtr, bufferSize, MPI_BYTE, appRank,
						  VOCL_PROXY_WRITE_TAG + bufferIndex, appCommData,
						  getWriteRequestPtr(appIndex, bufferIndex));

				/* save information for writing to GPU memory */
				writeBufferInfoPtr->commandQueue = tmpEnqueueWriteBuffer.command_queue;
				writeBufferInfoPtr->size = bufferSize;
				writeBufferInfoPtr->offset = 
					tmpEnqueueWriteBuffer.offset + i * VOCL_PROXY_WRITE_BUFFER_SIZE;
				writeBufferInfoPtr->mem = tmpEnqueueWriteBuffer.buffer;
				writeBufferInfoPtr->blocking_write = tmpEnqueueWriteBuffer.blocking_write;
				writeBufferInfoPtr->numEvents = tmpEnqueueWriteBuffer.num_events_in_wait_list;
				writeBufferInfoPtr->eventWaitList = event_wait_list;

				/* set flag to indicate buffer is being used */
				setWriteBufferFlag(appIndex, bufferIndex, WRITE_RECV_DATA);
				increaseWriteBufferCount(appIndex);
			}
			voclResetWriteEnqueueFlag(appIndex);
			voclProxyUpdateMemoryOnCmdQueue(tmpEnqueueWriteBuffer.command_queue,
											tmpEnqueueWriteBuffer.buffer,
											tmpEnqueueWriteBuffer.cb);

			if (tmpEnqueueWriteBuffer.blocking_write == CL_TRUE) {
				if (requestNo > 0) {
					MPI_Waitall(requestNo, curRequest, curStatus);
					requestNo = 0;
				}

				/* process all previous write and read */
				tmpEnqueueWriteBuffer.res = processAllWrites(appIndex);
				tmpEnqueueWriteBuffer.event = writeBufferInfoPtr->event;

				MPI_Isend(&tmpEnqueueWriteBuffer, sizeof(tmpEnqueueWriteBuffer), MPI_BYTE,
						  appRank, ENQUEUE_WRITE_BUFFER, appComm,
						  curRequest + (requestNo++));
			}
			else {
				if (tmpEnqueueWriteBuffer.event_null_flag == 0) {
					if (requestNo > 0) {
						MPI_Waitall(requestNo, curRequest, curStatus);
						requestNo = 0;
					}
					tmpEnqueueWriteBuffer.res =
						processWriteBuffer(appIndex, bufferIndex, bufferNum + 1);
					tmpEnqueueWriteBuffer.event = writeBufferInfoPtr->event;
					writeBufferInfoPtr->numWriteBuffers = bufferNum + 1;

					MPI_Isend(&tmpEnqueueWriteBuffer, sizeof(tmpEnqueueWriteBuffer), MPI_BYTE,
							  appRank, ENQUEUE_WRITE_BUFFER, appComm, curRequest + (requestNo++));
				}
			}

			if (requestNo > 0) {
				MPI_Wait(curRequest, curStatus);
			}
		}
		else if (cmdQueuePtr->msgTag = ENQUEUE_ND_RANGE_KERNEL)
		{
			memcpy(&tmpEnqueueNDRangeKernel, cmdQueuePtr->conMsgBuffer, sizeof(struct strEnqueueNDRangeKernel));
			requestNo = 0;
			event_wait_list = NULL;
			num_events_in_wait_list = tmpEnqueueNDRangeKernel.num_events_in_wait_list;
			if (num_events_in_wait_list > 0) {
				event_wait_list =
					(cl_event *) malloc(sizeof(cl_event) * num_events_in_wait_list);
				MPI_Irecv(event_wait_list, sizeof(cl_event) * num_events_in_wait_list,
						  MPI_BYTE, appRank, ENQUEUE_ND_RANGE_KERNEL1, appCommData[commIndex],
						  curRequest + (requestNo++));
			}

			work_dim = tmpEnqueueNDRangeKernel.work_dim;
			args_ptr = NULL;
			global_work_offset = NULL;
			global_work_size = NULL;
			local_work_size = NULL;

			if (tmpEnqueueNDRangeKernel.dataSize > 0) {
				if (tmpEnqueueNDRangeKernel.dataSize > kernelMsgSize)
				{
					kernelMsgSize = tmpEnqueueNDRangeKernel.dataSize;
					kernelMsgBuffer = (char *) realloc(kernelMsgBuffer, kernelMsgSize);
				}
				MPI_Irecv(kernelMsgBuffer, tmpEnqueueNDRangeKernel.dataSize, MPI_BYTE, appRank,
						  ENQUEUE_ND_RANGE_KERNEL1, appCommData, curRequest + (requestNo++));
			}

			MPI_Waitall(requestNo, curRequest, curStatus);

			paramOffset = 0;
			if (tmpEnqueueNDRangeKernel.global_work_offset_flag == 1) {
				global_work_offset = (size_t *) (kernelMsgBuffer + paramOffset);
				paramOffset += work_dim * sizeof(size_t);
			}

			if (tmpEnqueueNDRangeKernel.global_work_size_flag == 1) {
				global_work_size = (size_t *) (kernelMsgBuffer + paramOffset);
				paramOffset += work_dim * sizeof(size_t);
			}

			if (tmpEnqueueNDRangeKernel.local_work_size_flag == 1) {
				local_work_size = (size_t *) (kernelMsgBuffer + paramOffset);
				paramOffset += work_dim * sizeof(size_t);
			}

			if (tmpEnqueueNDRangeKernel.args_num > 0) {
				args_ptr = (kernel_args *) (kernelMsgBuffer + paramOffset);
				paramOffset += (sizeof(kernel_args) * tmpEnqueueNDRangeKernel.args_num);
			}

			/* update global memory usage on the device */
			voclProxyUpdateGlobalMemUsage(tmpEnqueueNDRangeKernel.command_queue,
										  args_ptr, tmpEnqueueNDRangeKernel.args_num);

			/* if there are data received, but not write to */
			/* the GPU memory yet, use the helper thread to */
			/* wait MPI receive complete and write to the GPU memory */
			if (voclGetWriteEnqueueFlag(appIndex) == 0) {
				pthread_barrier_wait(&barrier);
				helperThreadOperFlag = GPU_ENQ_WRITE;
				/* used by the helper thread */
				voclProxyAppIndex = appIndex;
				pthread_barrier_wait(&barrier);
				pthread_barrier_wait(&barrier);
			}

			mpiOpenCLEnqueueNDRangeKernel(&tmpEnqueueNDRangeKernel,
										  &kernelLaunchReply,
										  event_wait_list,
										  global_work_offset,
										  global_work_size, local_work_size, args_ptr);

			/* increase the number of kernels in the command queue by 1 */
			voclProxyIncreaseKernelNumInCmdQueue(tmpEnqueueNDRangeKernel.command_queue, 1);

			if (tmpEnqueueNDRangeKernel.event_null_flag == 0)
			{
				MPI_Isend(&kernelLaunchReply, sizeof(struct strEnqueueNDRangeKernelReply),
					  MPI_BYTE, appRank, ENQUEUE_ND_RANGE_KERNEL, appComm,
					  curRequest);
			}

			if (num_events_in_wait_list > 0) {
				free(event_wait_list);
			}

			if (tmpEnqueueNDRangeKernel.event_null_flag == 0)
			{
				MPI_Wait(curRequest, curStatus);
			}
		}
		else if (cmdQueuePtr->msgTag == ENQUEUE_READ_BUFFER)
		{
			memset(&tmpEnqueueReadBuffer, cmdQueuePtr->conMsgBuffer, sizeof(struct strEnqueueReadBuffer));
			num_events_in_wait_list = tmpEnqueueReadBuffer.num_events_in_wait_list;
			event_wait_list = NULL;
			if (num_events_in_wait_list > 0) {
				event_wait_list =
					(cl_event *) malloc(num_events_in_wait_list * sizeof(cl_event));
				MPI_Irecv(event_wait_list, num_events_in_wait_list * sizeof(cl_event),
						  MPI_BYTE, appRank, ENQUEUE_READ_BUFFER1, appCommData,
						  curRequest);
				MPI_Wait(curRequest, curStatus);
			}

			bufferSize = VOCL_PROXY_READ_BUFFER_SIZE;
			bufferNum = (tmpEnqueueReadBuffer.cb - 1) / VOCL_PROXY_READ_BUFFER_SIZE;
			remainingSize = tmpEnqueueReadBuffer.cb - bufferSize * bufferNum;
			for (i = 0; i <= bufferNum; i++) {
				bufferIndex = getNextReadBufferIndex(appIndex);
				if (i == bufferNum)
					bufferSize = remainingSize;
				readBufferInfoPtr = getReadBufferInfoPtr(appIndex, bufferIndex);
				readBufferInfoPtr->comm = appCommData;
				readBufferInfoPtr->tag = VOCL_PROXY_READ_TAG + bufferIndex;
				readBufferInfoPtr->dest = appRank;
				readBufferInfoPtr->size = bufferSize;
				tmpEnqueueReadBuffer.res =
					clEnqueueReadBuffer(tmpEnqueueReadBuffer.command_queue,
										tmpEnqueueReadBuffer.buffer,
										CL_FALSE,
										tmpEnqueueReadBuffer.offset +
										i * VOCL_PROXY_READ_BUFFER_SIZE, bufferSize,
										readBufferInfoPtr->dataPtr,
										tmpEnqueueReadBuffer.num_events_in_wait_list,
										event_wait_list, &readBufferInfoPtr->event);
				setReadBufferFlag(appIndex, bufferIndex, READ_GPU_MEM);
			}
			readBufferInfoPtr->numReadBuffers = bufferNum + 1;

			/* some new read requests are issued */
			voclResetReadBufferCoveredFlag(appIndex);

			if (tmpEnqueueReadBuffer.blocking_read == CL_FALSE) {
				if (tmpEnqueueReadBuffer.event_null_flag == 0) {
					tmpEnqueueReadBuffer.event = readBufferInfoPtr->event;
					MPI_Isend(&tmpEnqueueReadBuffer, sizeof(tmpEnqueueReadBuffer), MPI_BYTE,
							  appRank, ENQUEUE_READ_BUFFER, appComm, curRequest);
				}
			}
			else {      /* blocking, reading is complete, send data to local node */
				tmpEnqueueReadBuffer.res = processAllReads(appIndex);
				if (tmpEnqueueReadBuffer.event_null_flag == 0) {
					tmpEnqueueReadBuffer.event = readBufferInfoPtr->event;
				}
				MPI_Isend(&tmpEnqueueReadBuffer, sizeof(tmpEnqueueReadBuffer), MPI_BYTE,
						  appRank, ENQUEUE_READ_BUFFER, appComm, curRequest);
			}
			MPI_Wait(curRequest, curStatus);
		
		}
		else if (cmdQueuePtr->msgTag == FINISH_FUNC)
		{
			struct strFinish tmpFinish
			memcpy(&tmpFinish, cmdQueuePtr->conMsgBuffer, sizeof(struct strFinish));
			processAllWrites(appIndex);
			processAllReads(appIndex);
			mpiOpenCLFinish(&tmpFinish);

			/* all kernels complete their computation */
			voclProxyResetKernelNumInCmdQueue(tmpFinish.command_queue);

			MPI_Isend(&tmpFinish, sizeof(tmpFinish), MPI_BYTE, appRank,
					  FINISH_FUNC, appComm, curRequest);
			MPI_Wait(curRequest, curStatus);
	
		}
	}
}
