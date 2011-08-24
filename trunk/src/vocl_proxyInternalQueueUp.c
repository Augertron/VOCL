#include <stdio.h>
#include <CL/opencl.h>
#include <pthread.h>
#include <unistd.h>
#include "vocl_proxy.h"
#include "vocl_proxyInternalQueueUp.h"
#include "vocl_proxyKernelArgProc.h"
#include "vocl_proxyBufferProc.h"

static struct strEnqueueWriteBuffer tmpEnqueueWriteBuffer;
static struct strEnqueueNDRangeKernel tmpEnqueueNDRangeKernel;
static struct strEnqueueNDRangeKernelReply kernelLaunchReply;
static struct strEnqueueReadBuffer tmpEnqueueReadBuffer;
static struct strFinish tmpFinish;

static struct strVoclCommandQueue *voclProxyCmdQueue = NULL;
static int voclProxyCmdNum;
static int voclProxyCmdHead;
static int voclProxyCmdTail;
int voclProxyThreadInternalTerminateFlag = 0;
static int voclProxyAppNum = 100;
static int *voclProxyNumOfKernelsLaunched = 0;
static int *voclProxyNumOfKernelsCompleted = 0;

extern int helperThreadOperFlag;
extern int voclProxyAppIndex;
extern pthread_barrier_t barrier;

pthread_t thKernelLaunch;
pthread_barrier_t barrierKernalLaunch;

extern int getNextWriteBufferIndex(int rank);
extern struct strWriteBufferInfo *getWriteBufferInfoPtr(int rank, int index);
extern MPI_Request *getWriteRequestPtr(int rank, int index);
extern void setWriteBufferFlag(int rank, int index, int flag);
extern void increaseWriteBufferCount(int rank);
extern void voclResetWriteEnqueueFlag(int rank);
extern cl_int processAllWrites(int rank);
extern cl_int processWriteBuffer(int rank, int curIndex, int bufferNum);
extern void voclResetReadBufferCoveredFlag(int rank);

extern int getNextReadBufferIndex(int rank);
extern struct strReadBufferInfo *getReadBufferInfoPtr(int rank, int index);
extern void setReadBufferFlag(int rank, int index, int flag);
extern cl_int processAllReads(int rank);


/* device info for migration */
extern void voclProxyUpdateMemoryOnCmdQueue(cl_command_queue cmdQueue, cl_mem mem,
                                            size_t size);
extern void voclProxyUpdateGlobalMemUsage(cl_command_queue comman_queue, kernel_args * argsPtr,
                                          int argsNum);
extern int voclGetWriteEnqueueFlag(int rank);

extern void mpiOpenCLEnqueueNDRangeKernel(struct strEnqueueNDRangeKernel
                                          *tmpEnqueueNDRangeKernel,
                                          struct strEnqueueNDRangeKernelReply
                                          *kernelLaunchReply, cl_event * event_wait_list,
                                          size_t * global_work_offset,
                                          size_t * global_work_size, size_t * local_work_size,
                                          kernel_args * args_ptr);
extern void mpiOpenCLFinish(struct strFinish *tmpFinish);

/* management of kernel numbers on the node */
extern void voclProxyIncreaseKernelNumInCmdQueue(cl_command_queue cmdQueue, int kernelNum);
extern void voclProxyDecreaseKernelNumInCmdQueue(cl_command_queue cmdQueue, int kernelNum);
extern void voclProxyResetKernelNumInCmdQueue(cl_command_queue cmdQueue);


void voclProxyCmdQueueInit()
{
	int i;
	voclProxyCmdNum = VOCL_PROXY_CMDQUEUE_SIZE;
	voclProxyCmdHead = 0;
	voclProxyCmdTail = 0;
	voclProxyNumOfKernelsLaunched = (int *)malloc(sizeof(int) * voclProxyAppNum);
	//voclProxyNumOfKernelsCompleted = (int *)malloc(sizeof(int) * voclProxyAppNum);
	memset(voclProxyNumOfKernelsLaunched, 0, sizeof(int) * voclProxyAppNum);
	//memset(voclProxyNumOfKernelsCompleted, 0, sizeof(int) * voclProxyAppNum);
	voclProxyCmdQueue = (struct strVoclCommandQueue *)malloc(sizeof(struct strVoclCommandQueue) * voclProxyCmdNum);

	for (i = 0; i < voclProxyCmdNum; i++)
	{
		pthread_mutex_init(&voclProxyCmdQueue[i].lock, NULL);
		voclProxyCmdQueue[i].msgTag = 0;  /* invalid msg tag */
		voclProxyCmdQueue[i].status = VOCL_PROXY_CMD_AVABL;
	}

	return;
}

/* increase the number of kernels in the VOCL command queue */
/* and return the flag whether internal wait on the proxy process is needed */
//int voclProxyCmdQueueIncreaseKernelInExecution()
//{
//	voclProxyNumOfKernelsLaunched++;
//	/* internal wait is needed for the current kernel launch */
//	if (voclProxyNumOfKernelsLaunched >= VOCL_CMDQUEUE_IN_EXECUTION)
//	{
//		voclProxyNumOfKernelsLaunched = 0;
//		return 1;
//	}
//
//	return 0;
//}


/* for enqueue operation */
struct strVoclCommandQueue * voclProxyGetCmdQueueTail()
{
	int index;

	index = voclProxyCmdTail % voclProxyCmdNum;
	while (voclProxyCmdHead + voclProxyCmdNum == voclProxyCmdTail)
	{
		usleep(10);
	}
	pthread_mutex_lock(&voclProxyCmdQueue[index].lock);
	voclProxyCmdTail++;

	return &voclProxyCmdQueue[index];
}

/* for dequeue operation */
struct strVoclCommandQueue * voclProxyGetCmdQueueHead()
{
	int index;
	index = voclProxyCmdHead;

	while (voclProxyCmdHead == voclProxyCmdTail)
	{
		if (voclProxyThreadInternalTerminateFlag == 1)
		{
			return NULL;
		}
		usleep(10);
	}
	pthread_mutex_lock(&voclProxyCmdQueue[index].lock);
	voclProxyCmdHead++;
	if (voclProxyCmdHead >= voclProxyCmdNum)
	{
		voclProxyCmdHead = 0;
		voclProxyCmdTail -= voclProxyCmdNum;
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
	voclProxyNumOfKernelsLaunched = 0;

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
	free(voclProxyNumOfKernelsLaunched);
	//free(voclProxyNumOfKernelsCompleted);
	voclProxyCmdTail = 0;

	return;
}

void *proxyEnqueueThread(void *p)
{
	struct strVoclCommandQueue *cmdQueuePtr;

	cl_event *event_wait_list;
	cl_uint num_events_in_wait_list;
	int requestNo, bufferNum, bufferIndex, i;
	int appIndex, appRank;
	size_t bufferSize, remainingSize;
	MPI_Comm appComm, appCommData;
	MPI_Request curRequest[50];
	MPI_Status  curStatus[50];

	char *kernelMsgBuffer;
	size_t kernelMsgSize;

    struct strWriteBufferInfo *writeBufferInfoPtr;
	struct strReadBufferInfo *readBufferInfoPtr;

	int work_dim;
    size_t *global_work_offset, *global_work_size, *local_work_size, paramOffset;
	kernel_args *args_ptr;
	int internalWaitFlag; 

	kernelMsgSize = 2048;
	kernelMsgBuffer = (char *) malloc(sizeof(char) * kernelMsgSize);

	pthread_barrier_wait(&barrierKernalLaunch);

	while (1)
	{
		cmdQueuePtr = voclProxyGetCmdQueueHead();
		if (voclProxyThreadInternalTerminateFlag == 1)
		{
			break;
		}
		appComm = cmdQueuePtr->appComm;
		appCommData = cmdQueuePtr->appCommData;
		appRank = cmdQueuePtr->appRank;
		appIndex = cmdQueuePtr->appIndex;

		/* if the number of app process is larger than voclProxyAppNum */
		/* reallocate memory */
		if (appIndex >= voclProxyAppNum)
		{
			voclProxyNumOfKernelsLaunched = (int *)realloc(voclProxyNumOfKernelsLaunched, sizeof(int) * 2 * appIndex);
			memset(&voclProxyNumOfKernelsLaunched[voclProxyAppNum], 0, sizeof(int) * (2 * appIndex - voclProxyAppNum));
			voclProxyAppNum = appIndex * 2;
		}
			
		if (cmdQueuePtr->msgTag == ENQUEUE_WRITE_BUFFER)
		{
			memcpy(&tmpEnqueueWriteBuffer, cmdQueuePtr->conMsgBuffer, sizeof(struct strEnqueueWriteBuffer));
			pthread_mutex_unlock(&cmdQueuePtr->lock);
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
		else if (cmdQueuePtr->msgTag == ENQUEUE_ND_RANGE_KERNEL)
		{
			memcpy(&tmpEnqueueNDRangeKernel, cmdQueuePtr->conMsgBuffer, sizeof(struct strEnqueueNDRangeKernel));
			internalWaitFlag = cmdQueuePtr->internalWaitFlag;
			pthread_mutex_unlock(&cmdQueuePtr->lock);
			requestNo = 0;
			event_wait_list = NULL;
			num_events_in_wait_list = tmpEnqueueNDRangeKernel.num_events_in_wait_list;
			if (num_events_in_wait_list > 0) {
				event_wait_list =
					(cl_event *) malloc(sizeof(cl_event) * num_events_in_wait_list);
				MPI_Irecv(event_wait_list, sizeof(cl_event) * num_events_in_wait_list,
						  MPI_BYTE, appRank, ENQUEUE_ND_RANGE_KERNEL1, appCommData,
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
			//printf("kernelLaunch, cmdQueue = %p\n", tmpEnqueueNDRangeKernel.command_queue);
			//voclProxyUpdateGlobalMemUsage(tmpEnqueueNDRangeKernel.command_queue,
			//							  args_ptr, tmpEnqueueNDRangeKernel.args_num);

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

			voclProxyNumOfKernelsLaunched[appIndex]++;
			/* if internal wait is needed, call clFinish */
			if (voclProxyNumOfKernelsLaunched[appIndex] >= VOCL_CMDQUEUE_IN_EXECUTION)
			{
				processAllWrites(appIndex);
				processAllReads(appIndex);
				mpiOpenCLFinish(&tmpFinish);

				/* all kernels complete their computation */
				voclProxyDecreaseKernelNumInCmdQueue(tmpEnqueueNDRangeKernel.command_queue, voclProxyNumOfKernelsLaunched[appIndex]);
				voclProxyNumOfKernelsLaunched[appIndex] = 0;
			}
		}
		else if (cmdQueuePtr->msgTag == ENQUEUE_READ_BUFFER)
		{
			//printf("GPUMemoryRead\n");
			memcpy(&tmpEnqueueReadBuffer, cmdQueuePtr->conMsgBuffer, sizeof(struct strEnqueueReadBuffer));
			pthread_mutex_unlock(&cmdQueuePtr->lock);
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
			memcpy(&tmpFinish, cmdQueuePtr->conMsgBuffer, sizeof(struct strFinish));
			pthread_mutex_unlock(&cmdQueuePtr->lock);
			processAllWrites(appIndex);
			processAllReads(appIndex);
			mpiOpenCLFinish(&tmpFinish);

			/* all kernels complete their computation */
			voclProxyResetKernelNumInCmdQueue(tmpFinish.command_queue);
			voclProxyNumOfKernelsLaunched[appIndex] = 0;

			MPI_Isend(&tmpFinish, sizeof(tmpFinish), MPI_BYTE, appRank,
					  FINISH_FUNC, appComm, curRequest);
			MPI_Wait(curRequest, curStatus);
		}
	}

	free(kernelMsgBuffer);

	return NULL;
}
