#include <stdio.h>
#include <CL/opencl.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>
#include "vocl_proxy.h"
#include "vocl_proxyInternalQueueUp.h"
#include "vocl_proxyKernelArgProc.h"
#include "vocl_proxyBufferProc.h"

static struct strEnqueueWriteBuffer tmpEnqueueWriteBuffer;
static struct strEnqueueNDRangeKernel tmpEnqueueNDRangeKernel;
static struct strEnqueueNDRangeKernelReply kernelLaunchReply;
static struct strEnqueueReadBuffer tmpEnqueueReadBuffer;
static struct strFinish tmpFinish;

static vocl_internal_command_queue *voclProxyInternalQueue = NULL;
static int voclProxyCmdNum;
static int voclProxyCmdHead;
static int voclProxyCmdTail;
static int voclProxyAppNum = 100;
static int *voclProxyNumOfKernelsLaunched = NULL;
int voclProxyThreadInternalTerminateFlag = 0;
cl_command_queue voclProxyMigCmdQueue;
int voclProxyMigAppIndex;
static cl_command_queue voclProxyLastCmdQueueStored;
static int voclProxyLastAppIndexStored;

extern int helperThreadOperFlag;
extern int voclProxyAppIndex;
extern pthread_barrier_t barrier;

pthread_t thKernelLaunch;
pthread_mutex_t internalQueueMutex;
pthread_barrier_t barrierMigOperations;

extern int voclMigOrigProxyRank;
extern int voclMigDestProxyRank;
extern cl_device_id voclMigOrigDeviceID;
extern cl_device_id voclMigDestDeviceID;
extern MPI_Comm voclMigDestComm;
extern MPI_Comm voclMigDestCommData;
extern int voclMigAppIndexOnOrigProxy;
extern int voclMigAppIndexOnDestProxy;
extern void voclProxyMigSendOperationsInCmdQueue(int origProxyRank, int destProxyRank,
			cl_device_id origDeviceID, cl_device_id destDeviceID,
        	MPI_Comm destComm, MPI_Comm destCommData, int appIndex, int appIndexOnDestProxy);

extern void voclProxyMigrationMutexLock(int appIndex);
extern void voclProxyMigrationMutexUnlock(int appIndex);

extern void voclProxyStoreKernelArgs(cl_kernel kernel, int argNum, kernel_args *args);
extern int getNextWriteBufferIndex(int rank);
extern struct strWriteBufferInfo *getWriteBufferInfoPtr(int rank, int index);
extern MPI_Request *getWriteRequestPtr(int rank, int index);
extern void setWriteBufferFlag(int rank, int index, int flag);
extern void increaseWriteBufferCount(int rank);
extern void voclResetWriteEnqueueFlag(int rank);
extern cl_int processAllWrites(int rank);
extern cl_int processWriteBuffer(int rank, int curIndex, int bufferNum);
extern void voclResetReadBufferCoveredFlag(int rank);
extern void voclProxySetMemWritten(cl_mem mem, int isWritten);
extern void voclProxySetMemWriteCmdQueue(cl_mem mem, cl_command_queue cmdQueue);

extern int getNextReadBufferIndex(int rank);
extern struct strReadBufferInfo *getReadBufferInfoPtr(int rank, int index);
extern void setReadBufferFlag(int rank, int index, int flag);
extern cl_int processAllReads(int rank);

/* device info for migration */
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

/* get mig condition */
extern int voclProxyGetMigrationCondition();
extern void voclProxySetMigrationCondition(int condition);
extern cl_int voclProxyMigration(int appIndex, cl_device_id deviceID);
extern cl_device_id voclProxyGetCmdQueueDeviceID(cl_command_queue command_queue);
extern void voclProxySetMigrated();
extern int voclProxyIsMigrated();
extern int voclProxyGetKernelNumThreshold();
extern int voclProxyGetForcedMigrationFlag();
extern int rankNo;

void voclProxyInternalQueueInit()
{
	int i;
	voclProxyCmdNum = VOCL_PROXY_CMDQUEUE_SIZE;
	voclProxyCmdHead = 0;
	voclProxyCmdTail = 0;
	voclProxyNumOfKernelsLaunched = (int *)malloc(sizeof(int) * voclProxyAppNum);
	memset(voclProxyNumOfKernelsLaunched, 0, sizeof(int) * voclProxyAppNum);
	voclProxyInternalQueue = (vocl_internal_command_queue *)malloc(sizeof(vocl_internal_command_queue) * voclProxyCmdNum);

	for (i = 0; i < voclProxyCmdNum; i++)
	{
		pthread_mutex_init(&voclProxyInternalQueue[i].lock, NULL);
		voclProxyInternalQueue[i].msgTag = 0;  /* invalid msg tag */
		voclProxyInternalQueue[i].paramBuf = NULL;  
		voclProxyInternalQueue[i].status = VOCL_PROXY_CMD_AVABL;
	}

	return;
}

/* for enqueue operation */
vocl_internal_command_queue * voclProxyGetInternalQueueTail()
{
	int index;
	/* goes back to see if there is available */
	index = (voclProxyCmdTail - 1) % voclProxyCmdNum;
	while (voclProxyCmdTail > voclProxyCmdHead &&
		   (voclProxyInternalQueue[index].status == VOCL_PROXY_CMD_MIG ||
		   voclProxyInternalQueue[index].status == VOCL_PROXY_CMD_AVABL))
	{
		voclProxyCmdTail--;
		index = (voclProxyCmdTail - 1) % voclProxyCmdNum;
	}

	/* the internal queue is full. */
	while (voclProxyCmdHead + voclProxyCmdNum <= voclProxyCmdTail)
	{
		usleep(10);
	}

	index = voclProxyCmdTail % voclProxyCmdNum;
	voclProxyInternalQueue[index].status = VOCL_PROXY_CMD_INUSE;
	pthread_mutex_lock(&voclProxyInternalQueue[index].lock);
	voclProxyCmdTail++;
	return &voclProxyInternalQueue[index];
}

/* for dequeue operation */
vocl_internal_command_queue * voclProxyGetInternalQueueHead()
{
	int index;

	/* get the next valid command, i.e., status is NOT */
	/* VOCL_PROXY_CMD_MIG or VOCL_PROXY_CMD_AVABL */
	while (voclProxyCmdHead < voclProxyCmdTail &&
		   (voclProxyInternalQueue[voclProxyCmdHead].status == VOCL_PROXY_CMD_MIG || 
		    voclProxyInternalQueue[voclProxyCmdHead].status == VOCL_PROXY_CMD_AVABL))
	{
		voclProxyCmdHead++;
		if (voclProxyCmdHead >= voclProxyCmdNum)
		{
			voclProxyCmdHead = 0;
			voclProxyCmdTail -= voclProxyCmdNum;
		}
	}

	index = voclProxyCmdHead;

	while (voclProxyCmdHead >= voclProxyCmdTail)
	{
		if (voclProxyThreadInternalTerminateFlag == 1)
		{
			return NULL;
		}
		usleep(10);
	}

	/* this lock is for the current item to protect it from */
	/* writting by the proxy main thread */
	pthread_mutex_lock(&voclProxyInternalQueue[index].lock);

	voclProxyCmdHead++;
	if (voclProxyCmdHead >= voclProxyCmdNum)
	{
		voclProxyCmdHead = 0;
		voclProxyCmdTail -= voclProxyCmdNum;
	}

	return &voclProxyInternalQueue[index];
}

/* get commands in the internal queue to be migrated */
vocl_internal_command_queue * voclProxyMigGetAppCmds(int appIndex, int cmdIndex)
{
	int index;
	index = voclProxyCmdHead + cmdIndex;

	/* current commands is inuse and equal to the appIndex */
	if (voclProxyInternalQueue[index].appIndex == appIndex &&
		voclProxyInternalQueue[index].status == VOCL_PROXY_CMD_INUSE)
	{
		return &voclProxyInternalQueue[index];
	}
	else
	{
		return NULL;
	}
}

void voclProxyUnlockItem(vocl_internal_command_queue *cmdPtr)
{
	pthread_mutex_unlock(&cmdPtr->lock);
	return;
}

void voclProxyInternalQueueReset()
{
	int i;

	for (i = 0; i < voclProxyCmdNum; i++)
	{
		voclProxyInternalQueue[i].status = VOCL_PROXY_CMD_AVABL;
		voclProxyInternalQueue[i].msgTag = 0;  /* invalid msg tag */
	}

	voclProxyCmdHead = 0;
	voclProxyCmdTail = 0;
	voclProxyNumOfKernelsLaunched = 0;

	return;
}

int voclProxyGetInternalQueueKernelLaunchNum(int appIndex)
{
	return voclProxyNumOfKernelsLaunched[appIndex];
}

/* get the num of operatoins queue up for a specific vgpu */
int voclProxyGetInternalQueueOperationNum(int appIndex)
{
	return voclProxyCmdTail - voclProxyCmdHead;
}

void voclProxyInternalQueueFinalize()
{
	int i;

	for (i = 0; i < voclProxyCmdNum; i++)
	{
		voclProxyInternalQueue[i].status = VOCL_PROXY_CMD_AVABL;
	}

	free(voclProxyInternalQueue);

	voclProxyCmdNum = 0;
	voclProxyCmdHead = 0;
	free(voclProxyNumOfKernelsLaunched);
	voclProxyCmdTail = 0;
	voclProxyThreadInternalTerminateFlag = 1;

	return;
}

void *proxyEnqueueThread(void *p)
{
	vocl_internal_command_queue *cmdQueuePtr;

	cl_event *event_wait_list;
	cl_uint num_events_in_wait_list;
	int requestNo, bufferNum, bufferIndex, i;
	int appIndex, appRank;
	size_t bufferSize, remainingSize;
	MPI_Comm appComm, appCommData;
	MPI_Request curRequest[50];
	MPI_Status  curStatus[50];

	size_t kernelMsgSize;

    struct strWriteBufferInfo *writeBufferInfoPtr;
	struct strReadBufferInfo *readBufferInfoPtr;

	int work_dim;
    size_t *global_work_offset, *global_work_size, *local_work_size, paramOffset;
	kernel_args *args_ptr;
	int internalWaitFlag; 
	struct timeval t1, t2;
	float tmpTime;

	voclProxyMigAppIndex = 0;
	while (1)
	{
		if (voclProxyThreadInternalTerminateFlag == 1)
		{
			break;
		}

//		if (voclProxyGetInternalQueueKernelLaunchNum(appIndex) >= 4 && 
//			voclProxyIsMigrated() == 0 &&
//			rankNo == 0)
		if (voclProxyGetMigrationCondition() == 1 && voclProxyIsMigrated() == 0 && /* &&  rankNo == 0 && */
			voclProxyGetInternalQueueKernelLaunchNum(appIndex) > voclProxyGetKernelNumThreshold())
		{
			voclProxySetMigrated();
			pthread_mutex_lock(&internalQueueMutex);

			if (voclProxyGetForcedMigrationFlag() == 1)
			{
				voclProxyMigCmdQueue = voclProxyLastCmdQueueStored;
				voclProxyMigAppIndex = voclProxyLastAppIndexStored;
			}

			/* acquire the locker to prevent library from issuing more function calls */
			voclProxyMigrationMutexLock(voclProxyMigAppIndex);
				
			gettimeofday(&t1, NULL);
			/* make sure issued commands completed */
			clFinish(voclProxyMigCmdQueue);
			processAllWrites(voclProxyMigAppIndex);
			processAllReads(voclProxyMigAppIndex);
			gettimeofday(&t2, NULL);
			tmpTime = 1000.0 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
			printf("%.3f\n", tmpTime);

			/* migration a vgpu */
			voclProxyMigration(voclProxyMigAppIndex, voclProxyGetCmdQueueDeviceID(voclProxyMigCmdQueue));
			/* release the locker */
			voclProxyMigrationMutexUnlock(voclProxyMigAppIndex);
			/* release the locker */
printf("internalQ1\n");
			pthread_mutex_unlock(&internalQueueMutex);
			/* release the locker */

printf("internalQ2\n");
			/* wait for barrier for transfer commands */
			pthread_barrier_wait(&barrierMigOperations);
			/* release the locker */

printf("internalQ3\n");
			/* send unexecuted commands to destination proxy process */
			gettimeofday(&t1, NULL);
			voclProxyMigSendOperationsInCmdQueue(voclMigOrigProxyRank, voclMigDestProxyRank,
					voclMigOrigDeviceID, voclMigDestDeviceID,
					voclMigDestComm, voclMigDestCommData, voclMigAppIndexOnOrigProxy,
					voclMigAppIndexOnDestProxy);
			gettimeofday(&t2, NULL);
			tmpTime = 1000.0 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
			printf("%.3f\n", tmpTime);
			pthread_barrier_wait(&barrierMigOperations);
		}

		/* get head of internal queue */
		cmdQueuePtr = voclProxyGetInternalQueueHead();
		if (cmdQueuePtr == NULL) /* terminate the helper thread */
		{
			break;
		}

		appComm = cmdQueuePtr->appComm;
		appCommData = cmdQueuePtr->appCommData;
		appRank = cmdQueuePtr->appRank;
		appIndex = cmdQueuePtr->appIndex;
		voclProxyLastAppIndexStored = appIndex;

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
			voclProxyLastCmdQueueStored = tmpEnqueueWriteBuffer.command_queue;
			cmdQueuePtr->status = VOCL_PROXY_CMD_AVABL;
			pthread_mutex_unlock(&cmdQueuePtr->lock);
			event_wait_list = NULL;
			num_events_in_wait_list = tmpEnqueueWriteBuffer.num_events_in_wait_list;
			if (num_events_in_wait_list > 0) {
				event_wait_list =
					(cl_event *) malloc(sizeof(cl_event) * num_events_in_wait_list);
				MPI_Irecv(event_wait_list, sizeof(cl_event) * num_events_in_wait_list,
						  MPI_BYTE, appRank, tmpEnqueueWriteBuffer.tag, 
						  appCommData, curRequest);
				MPI_Wait(curRequest, curStatus);
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
						  //VOCL_PROXY_WRITE_TAG, appCommData,
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

			/* set memory migration state to be written */
			voclProxySetMemWritten(tmpEnqueueWriteBuffer.buffer, 1);
			/* combine command queue to memory */
			voclProxySetMemWriteCmdQueue(tmpEnqueueWriteBuffer.buffer,
										 tmpEnqueueWriteBuffer.command_queue);

			requestNo = 0;
			if (tmpEnqueueWriteBuffer.blocking_write == CL_TRUE) {

				/* process all previous write and read */
				tmpEnqueueWriteBuffer.res = processAllWrites(appIndex);
				tmpEnqueueWriteBuffer.event = writeBufferInfoPtr->event;

				MPI_Isend(&tmpEnqueueWriteBuffer, sizeof(tmpEnqueueWriteBuffer), MPI_BYTE,
						  appRank, ENQUEUE_WRITE_BUFFER, appComm,
						  curRequest + (requestNo++));
			}
			else {
				if (tmpEnqueueWriteBuffer.event_null_flag == 0) {
					tmpEnqueueWriteBuffer.res =
						processWriteBuffer(appIndex, bufferIndex, bufferNum + 1);
					tmpEnqueueWriteBuffer.event = writeBufferInfoPtr->event;
					writeBufferInfoPtr->numWriteBuffers = bufferNum + 1;

					MPI_Isend(&tmpEnqueueWriteBuffer, sizeof(tmpEnqueueWriteBuffer), MPI_BYTE,
							  appRank, ENQUEUE_WRITE_BUFFER, appComm, curRequest + (requestNo++));
				}
			}

			if (requestNo > 0) {
				MPI_Waitall(requestNo, curRequest, curStatus);
			}
		}
		else if (cmdQueuePtr->msgTag == ENQUEUE_ND_RANGE_KERNEL)
		{
			memcpy(&tmpEnqueueNDRangeKernel, cmdQueuePtr->conMsgBuffer, sizeof(struct strEnqueueNDRangeKernel));
			voclProxyLastCmdQueueStored = tmpEnqueueNDRangeKernel.command_queue;
			internalWaitFlag = cmdQueuePtr->internalWaitFlag;
			cmdQueuePtr->status = VOCL_PROXY_CMD_AVABL;
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
				MPI_Waitall(requestNo, curRequest, curStatus);
			}

			work_dim = tmpEnqueueNDRangeKernel.work_dim;
			args_ptr = NULL;
			global_work_offset = NULL;
			global_work_size = NULL;
			local_work_size = NULL;

			paramOffset = 0;
			if (tmpEnqueueNDRangeKernel.global_work_offset_flag == 1) {
				global_work_offset = (size_t *) (cmdQueuePtr->paramBuf + paramOffset);
				paramOffset += work_dim * sizeof(size_t);
			}

			if (tmpEnqueueNDRangeKernel.global_work_size_flag == 1) {
				global_work_size = (size_t *) (cmdQueuePtr->paramBuf + paramOffset);
				paramOffset += work_dim * sizeof(size_t);
			}

			if (tmpEnqueueNDRangeKernel.local_work_size_flag == 1) {
				local_work_size = (size_t *) (cmdQueuePtr->paramBuf + paramOffset);
				paramOffset += work_dim * sizeof(size_t);
			}

			if (tmpEnqueueNDRangeKernel.args_num > 0) {
				args_ptr = (kernel_args *) (cmdQueuePtr->paramBuf + paramOffset);
				paramOffset += (sizeof(kernel_args) * tmpEnqueueNDRangeKernel.args_num);
			}

			/* store the kernel arguments */
			voclProxyStoreKernelArgs(tmpEnqueueNDRangeKernel.kernel, 
								   tmpEnqueueNDRangeKernel.args_num, 
								   args_ptr);

			/* update global memory usage on the device */
			//printf("kernelLaunch, cmdQueue = %p\n", tmpEnqueueNDRangeKernel.command_queue);

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

			if (tmpEnqueueNDRangeKernel.dataSize > 0)
			{
				free(cmdQueuePtr->paramBuf);
			}

			if (tmpEnqueueNDRangeKernel.event_null_flag == 0)
			{
				MPI_Wait(curRequest, curStatus);
			}

			voclProxyNumOfKernelsLaunched[appIndex]++;
			/* if internal wait is needed, call clFinish */
			if (voclProxyNumOfKernelsLaunched[appIndex] >= VOCL_CMDQUEUE_IN_EXECUTION)
			{
				clFinish(tmpEnqueueNDRangeKernel.command_queue);

				/* all kernels complete their computation */
				voclProxyDecreaseKernelNumInCmdQueue(tmpEnqueueNDRangeKernel.command_queue, 
						voclProxyNumOfKernelsLaunched[appIndex]);
				voclProxyNumOfKernelsLaunched[appIndex] = 0;
			}
		}
		else if (cmdQueuePtr->msgTag == ENQUEUE_READ_BUFFER)
		{
			memcpy(&tmpEnqueueReadBuffer, cmdQueuePtr->conMsgBuffer, sizeof(struct strEnqueueReadBuffer));
			voclProxyLastCmdQueueStored = tmpEnqueueReadBuffer.command_queue;
			cmdQueuePtr->status = VOCL_PROXY_CMD_AVABL;
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
					MPI_Wait(curRequest, curStatus);
				}
			}
			else {      /* blocking, reading is complete, send data to local node */
				tmpEnqueueReadBuffer.res = processAllReads(appIndex);
				if (tmpEnqueueReadBuffer.event_null_flag == 0) {
					tmpEnqueueReadBuffer.event = readBufferInfoPtr->event;
				}
				MPI_Isend(&tmpEnqueueReadBuffer, sizeof(tmpEnqueueReadBuffer), MPI_BYTE,
						  appRank, ENQUEUE_READ_BUFFER, appComm, curRequest);
				MPI_Wait(curRequest, curStatus);
			}
		}
		else if (cmdQueuePtr->msgTag == FINISH_FUNC)
		{
			memcpy(&tmpFinish, cmdQueuePtr->conMsgBuffer, sizeof(struct strFinish));
			voclProxyLastCmdQueueStored = tmpFinish.command_queue;
			cmdQueuePtr->status = VOCL_PROXY_CMD_AVABL;
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

	return NULL;
}
