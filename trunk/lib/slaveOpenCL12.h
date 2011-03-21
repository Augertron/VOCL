#ifndef __SLAVE_OPENCL_H__
#define __SLAVE_OPENCL_H__
#include <sys/time.h>
#include <pthread.h>

#define CL_GPUV_READ  0
#define CL_GPUV_WRITE 1

typedef struct strDataTransfer {
	cl_event           event;
	MPI_Comm		   comm;
	void               *host_ptr;
	int                tag;
	size_t             msgSize;
	cl_int             readOrWrite;
	struct strDataTransfer *next;
} DATA_TRANSFER;

typedef struct strCmdQueue {
	cl_command_queue   hCmdQueue;
	DATA_TRANSFER      *dataTransferPtr;
	DATA_TRANSFER      *dataTransferPtrTail;
	struct strCmdQueue *next;
} CMD_QUEUE;

CMD_QUEUE *hCmdQueueHead = NULL;

void createCommandQueue(cl_command_queue command_queue)
{
	CMD_QUEUE *cmdQueuePtr = (CMD_QUEUE *)malloc(sizeof(CMD_QUEUE));
	cmdQueuePtr->hCmdQueue = command_queue;
	cmdQueuePtr->dataTransferPtr = NULL;
	cmdQueuePtr->dataTransferPtrTail = NULL;
	cmdQueuePtr->next = hCmdQueueHead;
	hCmdQueueHead = cmdQueuePtr;

	return;
}

CMD_QUEUE* getCommandQueue(cl_command_queue command_queue)
{
	CMD_QUEUE *hCmdQueues = hCmdQueueHead;
	while (hCmdQueues != NULL)
	{
		if (hCmdQueues->hCmdQueue == command_queue)
		{
			break;
		}
	}

	if (hCmdQueues == NULL)
	{
		printf("Error, command queue does not exist. In getCommandQueue!\n");
		exit(1);
	}

	return hCmdQueues;
}

void releaseCommandQueue(cl_command_queue command_queue)
{
	if (hCmdQueueHead == NULL)
	{
		return;
	}

	CMD_QUEUE *preCmdQueue, *curCmdQueue, *nextCmdQueue;
	DATA_TRANSFER *curDataTransferPtr, *nextDataTransferPtr;
	if (command_queue == hCmdQueueHead->hCmdQueue)
	{
		curCmdQueue = hCmdQueueHead;
		hCmdQueueHead = hCmdQueueHead->next;
		curDataTransferPtr = curCmdQueue->dataTransferPtr;
		while (curDataTransferPtr != NULL)
		{
			nextDataTransferPtr = curDataTransferPtr->next;
			if (curDataTransferPtr->host_ptr != NULL)
			{
				free(curDataTransferPtr->host_ptr);
			}
			free(curDataTransferPtr);
			curDataTransferPtr = nextDataTransferPtr;
		}
		free(curCmdQueue);
		return;
	}

	preCmdQueue = hCmdQueueHead;
	curCmdQueue = preCmdQueue->next;
	while (curCmdQueue != NULL)
	{
		if (command_queue == curCmdQueue->hCmdQueue)
		{
			curDataTransferPtr = curCmdQueue->dataTransferPtr;
			while (curDataTransferPtr != NULL)
			{
				nextDataTransferPtr = curDataTransferPtr->next;
				if (curDataTransferPtr->host_ptr != NULL)
				{
					free(curDataTransferPtr->host_ptr);
				}
				free(curDataTransferPtr);
				curDataTransferPtr = nextDataTransferPtr;
			}
			preCmdQueue->next = curCmdQueue->next;
			free(curCmdQueue);
			break;
		}
		preCmdQueue = curCmdQueue;
		curCmdQueue = curCmdQueue->next;
	}
	
	return;
}

DATA_TRANSFER *createDataTransfer(cl_command_queue command_queue,
								  cl_event         event)
{
	CMD_QUEUE *cmdQueue = getCommandQueue(command_queue);
	DATA_TRANSFER *dataTransferPtr = (DATA_TRANSFER *)malloc(sizeof(DATA_TRANSFER));
	dataTransferPtr->host_ptr = NULL;
	dataTransferPtr->event = event;
	dataTransferPtr->next = NULL;
	if (cmdQueue->dataTransferPtr == NULL &&
		cmdQueue->dataTransferPtrTail == NULL)
	{
		cmdQueue->dataTransferPtr = dataTransferPtr;
		cmdQueue->dataTransferPtrTail = dataTransferPtr;
	}
	else
	{
		cmdQueue->dataTransferPtrTail->next = dataTransferPtr;
		cmdQueue->dataTransferPtrTail = dataTransferPtr;
	}

	return dataTransferPtr;
}

DATA_TRANSFER *getDataTransfer(cl_command_queue command_queue,
                               cl_event         event)
{
	CMD_QUEUE *cmdQueue = getCommandQueue(command_queue);
	DATA_TRANSFER *dataTransferPtr = cmdQueue->dataTransferPtr;
	while (dataTransferPtr != NULL)
	{
		if (dataTransferPtr->event == event)
		{
			break;
		}
		dataTransferPtr = dataTransferPtr->next;
	}

	if (dataTransferPtr == NULL)
	{
		printf("In function getDataTranfer() error, the corresponding event is not there!\n");
		exit (0);
	}

	return dataTransferPtr;
}

DATA_TRANSFER *getDataTransferAll(cl_event event)
{
	CMD_QUEUE *cmdQueue = hCmdQueueHead;
	DATA_TRANSFER *curDataTransferPtr, *dataTransferPtr = NULL;
	while (cmdQueue != NULL)
	{
		curDataTransferPtr = cmdQueue->dataTransferPtr;
		while (curDataTransferPtr != NULL)
		{
			if (curDataTransferPtr->event == event)
			{
				dataTransferPtr = curDataTransferPtr;
				break;
			}
			curDataTransferPtr = curDataTransferPtr->next;
		}

		if (dataTransferPtr != NULL)
		{
			break;
		}

		cmdQueue = cmdQueue->next;
	}

	if (dataTransferPtr == NULL)
	{
		printf("In getDataTransferAll, event does not exist!\n");
		exit (1);
	}

	return dataTransferPtr;
}

void releaseDataTransfer(cl_command_queue command_queue,
                         cl_event         event)
{
	CMD_QUEUE *cmdQueue = getCommandQueue(command_queue);
	DATA_TRANSFER *curDataTransferPtr = cmdQueue->dataTransferPtr;
	DATA_TRANSFER *preDataTransferPtr;
	if (curDataTransferPtr->event == event)
	{
		cmdQueue->dataTransferPtr = curDataTransferPtr->next;
		if (curDataTransferPtr->host_ptr != NULL)
		{
			free(curDataTransferPtr->host_ptr);
		}
		free(curDataTransferPtr);

		//if only one node in the link list
		if (cmdQueue->dataTransferPtr == NULL)
		{
			cmdQueue->dataTransferPtrTail = NULL;
		}

		return;
	}

	preDataTransferPtr = curDataTransferPtr;
	curDataTransferPtr = curDataTransferPtr->next;
	while (curDataTransferPtr != NULL)
	{
		if (curDataTransferPtr->event == event)
		{
			preDataTransferPtr->next = curDataTransferPtr->next;
			if (curDataTransferPtr->host_ptr != NULL)
			{
				free(curDataTransferPtr->host_ptr);
			}
			free(curDataTransferPtr);

			//if it is the last node in the link list
			if (preDataTransferPtr->next == NULL)
			{
				cmdQueue->dataTransferPtrTail = preDataTransferPtr;
			}

			break;
		}
		preDataTransferPtr = curDataTransferPtr;
		curDataTransferPtr = curDataTransferPtr->next;
	}

	return;
}

void releaseDataTransferAll(cl_event event)
{
	CMD_QUEUE *cmdQueue = hCmdQueueHead;
	DATA_TRANSFER *curDataTransferPtr, *preDataTransferPtr;
	while (cmdQueue != NULL)
	{
		curDataTransferPtr = cmdQueue->dataTransferPtr;
		if (curDataTransferPtr->event == event)
		{
			cmdQueue->dataTransferPtr = curDataTransferPtr->next;
			if (curDataTransferPtr->host_ptr != NULL)
			{
				free(curDataTransferPtr->host_ptr);
			}
			if (cmdQueue->dataTransferPtr == NULL)
			{
				cmdQueue->dataTransferPtrTail = NULL;
			}

			free(curDataTransferPtr);
			return;
		}

		preDataTransferPtr = curDataTransferPtr;
		curDataTransferPtr = preDataTransferPtr->next;
		while (curDataTransferPtr != NULL)
		{
			if (curDataTransferPtr->event == event)
			{
				preDataTransferPtr->next = curDataTransferPtr->next;
				if (curDataTransferPtr->host_ptr != NULL)
				{
					free(curDataTransferPtr->host_ptr);
				}
				free(curDataTransferPtr);
				if (preDataTransferPtr->next == NULL)
				{
					cmdQueue->dataTransferPtrTail = preDataTransferPtr;
				}

				return;
			}
			preDataTransferPtr = curDataTransferPtr;
			curDataTransferPtr = preDataTransferPtr->next;
		}

		cmdQueue = cmdQueue->next;
	}

	return;
}

void processEvent(cl_event event)
{
	DATA_TRANSFER *dataTransferPtr;
	dataTransferPtr = getDataTransferAll(event);
	if (dataTransferPtr != NULL)
	{
		if (dataTransferPtr->readOrWrite == CL_GPUV_READ)
		{
			MPI_Send(dataTransferPtr->host_ptr, dataTransferPtr->msgSize, MPI_BYTE,
					 0, dataTransferPtr->tag, dataTransferPtr->comm);
		}
		free(dataTransferPtr->host_ptr);
		dataTransferPtr->host_ptr = NULL;
	}

	//release the event
	releaseDataTransferAll(event);
}

//void processCommandQueue(cl_command_queue command_queue)
//{
//	MPI_Request *request;
//	MPI_Status  *status;
////	cl_event *event_list;
//	int maxReadRequestNum = 100;
////	int maxWriteRequestNum = 100;
//	int readRequestNo = 0;
////	int writeRequestNo = 0;
//	request = (MPI_Request *)malloc(maxReadRequestNum * sizeof(MPI_Request));
//	status  = (MPI_Status *)malloc(maxReadRequestNum * sizeof(MPI_Status));
////	event_list = (cl_event *)malloc(maxWriteRequestNum * sizeof(cl_event));
//
//	CMD_QUEUE *cmdQueue = getCommandQueue(command_queue);
//	DATA_TRANSFER *dataTransferPtr = cmdQueue->dataTransferPtr;
//	DATA_TRANSFER *nextDataTransferPtr;
//	while (dataTransferPtr != NULL)
//	{
//		if (dataTransferPtr->readOrWrite == CL_GPUV_READ)
//		{
//			MPI_Isend(dataTransferPtr->host_ptr, dataTransferPtr->msgSize, MPI_BYTE,
//					 0, dataTransferPtr->tag, dataTransferPtr->comm, request+(readRequestNo++));
//		}
////		else
////		{
////			event_list[writeRequestNo++] = dataTransferPtr->event;
////		}
//
//		if (readRequestNo >= maxReadRequestNum)
//		{
//			maxReadRequestNum *= 2;
//			request = (MPI_Request *)realloc(request, maxReadRequestNum * sizeof(MPI_Request));
//			status  = (MPI_Status *)realloc(status, maxReadRequestNum * sizeof(MPI_Status));
//		}
////		if (writeRequestNo >= maxWriteRequestNum)
////		{
////			maxWriteRequestNum *= 2;
////			event_list = (cl_event *)realloc(event_list, maxWriteRequestNum * sizeof(cl_event));
////		}
//
//		dataTransferPtr = dataTransferPtr->next;
//	}
//	if (readRequestNo > 0)
//	{
//		MPI_Waitall(readRequestNo, request, status);
//	}
////	if (writeRequestNo > 0)
////	{
////		clWaitForEvents(writeRequestNo, event_list);
////	}
//
//	dataTransferPtr = cmdQueue->dataTransferPtr;
//	while (dataTransferPtr != NULL)
//	{
//		free(dataTransferPtr->host_ptr);
//		nextDataTransferPtr = dataTransferPtr->next;
//		free(dataTransferPtr);
//		dataTransferPtr = nextDataTransferPtr;
//	}
//
//	cmdQueue->dataTransferPtr = NULL;
//	cmdQueue->dataTransferPtrTail = NULL;
//
//	free(request);
//	free(status);
////	free(event_list);
//	return;
//}

//process all pending events, each time one event
//void processCommandQueue(cl_command_queue command_queue)
//{
//	MPI_Request *request;
//	MPI_Status  *status;
//	int maxReadRequestNum = 100;
//	int readRequestNo = 0;
//	cl_int err;
//	//debug-------------------
//	//int eventCount = 0;
//	//struct timeval t1, t2, t3, t4, t5;
//	//float sendTime, waitTime;
//	//-------------------------
//	request = (MPI_Request *)malloc(maxReadRequestNum * sizeof(MPI_Request));
//	status  = (MPI_Status *)malloc(maxReadRequestNum * sizeof(MPI_Status));
//
//	CMD_QUEUE *cmdQueue = getCommandQueue(command_queue);
//	DATA_TRANSFER *dataTransferPtr = cmdQueue->dataTransferPtr;
//	DATA_TRANSFER *nextDataTransferPtr;
//	//debug---------------------------
//	//clFinish(command_queue);
//	//gettimeofday(&t1, NULL);
//	//-------------------------------
//	//clFinish(command_queue);
//	while (dataTransferPtr != NULL)
//	{
//		//debug--------------------
//		//gettimeofday(&t4, NULL);
//		//--------------------------
//		err = clWaitForEvents(1, &dataTransferPtr->event);
//		if (err != CL_SUCCESS)
//		{
//			printf("wait for event errror!\n");
//		}
//
//		//debug--------------------
//		//gettimeofday(&t5, NULL);
//		//waitTime = 1000.0 * (t5.tv_sec - t4.tv_sec) + (t5.tv_usec - t4.tv_usec) / 1000.0;
//		//--------------------------
//		//debug-------------------------------------
//		//printf("eventCount = %d\n", eventCount++);
//
//		//gettimeofday(&t4, NULL);
//		//------------------------------------------
//		if (dataTransferPtr->readOrWrite == CL_GPUV_READ)
//		{
//			MPI_Isend(dataTransferPtr->host_ptr, dataTransferPtr->msgSize, MPI_BYTE,
//					 0, dataTransferPtr->tag, dataTransferPtr->comm, request+(readRequestNo++));
//		}
//		//debug-------------------------------------------
//		//gettimeofday(&t5, NULL);
//		//sendTime = 1000.0 * (t5.tv_sec - t4.tv_sec) + (t5.tv_usec - t4.tv_usec) / 1000.0;
//		//printf("oneEvent, sendTime = %.3f, waitTime = %.3f\n", sendTime, waitTime);
//		//------------------------------------------------
//		
//
//		if (readRequestNo >= maxReadRequestNum)
//		{
//			maxReadRequestNum *= 2;
//			request = (MPI_Request *)realloc(request, maxReadRequestNum * sizeof(MPI_Request));
//			status  = (MPI_Status *)realloc(status, maxReadRequestNum * sizeof(MPI_Status));
//		}
//		dataTransferPtr = dataTransferPtr->next;
//	}
//	//debug---------------------------
//	//gettimeofday(&t2, NULL);
//	//-------------------------------
//	if (readRequestNo > 0)
//	{
//		MPI_Waitall(readRequestNo, request, status);
//	}
//	//debug---------------------------
//	//gettimeofday(&t3, NULL);
//	//sendTime = 1000.0 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
//	//waitTime = 1000.0 * (t3.tv_sec - t2.tv_sec) + (t3.tv_usec - t2.tv_usec) / 1000.0;
//	//printf("total Send time = %.3f, waittime = %.3f\n", sendTime, waitTime);
//	//-------------------------------
//
//	dataTransferPtr = cmdQueue->dataTransferPtr;
//	while (dataTransferPtr != NULL)
//	{
//		free(dataTransferPtr->host_ptr);
//		nextDataTransferPtr = dataTransferPtr->next;
//		free(dataTransferPtr);
//		dataTransferPtr = nextDataTransferPtr;
//	}
//
//	cmdQueue->dataTransferPtr = NULL;
//	cmdQueue->dataTransferPtrTail = NULL;
//
//	free(request);
//	free(status);
//	return;
//}

pthread_barrier_t barrier;
pthread_t th;
void *sendMsg(void *ptr)
{
	MPI_Request *request;
	MPI_Status  *status;
	int maxReadRequestNum = 100;
	int readRequestNo = 0;
	cl_int err;
	request = (MPI_Request *)malloc(maxReadRequestNum * sizeof(MPI_Request));
	status  = (MPI_Status *)malloc(maxReadRequestNum * sizeof(MPI_Status));

	CMD_QUEUE *cmdQueue = (CMD_QUEUE *)ptr;
	DATA_TRANSFER *dataTransferPtr = cmdQueue->dataTransferPtr;
	DATA_TRANSFER *nextDataTransferPtr;
	pthread_barrier_wait(&barrier);
	while (dataTransferPtr != NULL)
	{
		if (dataTransferPtr->readOrWrite == CL_GPUV_READ)
		{
			MPI_Isend(dataTransferPtr->host_ptr, dataTransferPtr->msgSize, MPI_BYTE,
					 0, dataTransferPtr->tag, dataTransferPtr->comm, request+(readRequestNo++));
		}
		if (readRequestNo >= maxReadRequestNum)
		{
			maxReadRequestNum *= 2;
			request = (MPI_Request *)realloc(request, maxReadRequestNum * sizeof(MPI_Request));
			status  = (MPI_Status *)realloc(status, maxReadRequestNum * sizeof(MPI_Status));
		}
		dataTransferPtr = dataTransferPtr->next;
		pthread_barrier_wait(&barrier);
	}

	if (readRequestNo > 0)
	{
		MPI_Waitall(readRequestNo, request, status);
	}

	free(request);
	free(status);
	return NULL;
}

void processCommandQueue(cl_command_queue command_queue)
{
	cl_int err;
	//initialize barrier
	pthread_barrier_init(&barrier, NULL, 2);
	CMD_QUEUE *cmdQueue = getCommandQueue(command_queue);
	pthread_create(&th, NULL, sendMsg, (void *)cmdQueue);
	DATA_TRANSFER *dataTransferPtr = cmdQueue->dataTransferPtr;
	DATA_TRANSFER *nextDataTransferPtr;

	if (dataTransferPtr != NULL)
	{
		err = clWaitForEvents(1, &dataTransferPtr->event);
		if (err != CL_SUCCESS)
		{
			printf("wait for event errror!\n");
		}
	}
	pthread_barrier_wait(&barrier);

	dataTransferPtr = dataTransferPtr->next;
	while (dataTransferPtr != NULL)
	{
		err = clWaitForEvents(1, &dataTransferPtr->event);
		if (err != CL_SUCCESS)
		{
			printf("wait for event errror!\n");
		}
		dataTransferPtr = dataTransferPtr->next;
		pthread_barrier_wait(&barrier);
	}

	pthread_barrier_wait(&barrier);
	pthread_join(th, NULL);
	pthread_barrier_destroy(&barrier);

	dataTransferPtr = cmdQueue->dataTransferPtr;
	while (dataTransferPtr != NULL)
	{
		free(dataTransferPtr->host_ptr);
		nextDataTransferPtr = dataTransferPtr->next;
		free(dataTransferPtr);
		dataTransferPtr = nextDataTransferPtr;
	}

	cmdQueue->dataTransferPtr = NULL;
	cmdQueue->dataTransferPtrTail = NULL;

	return;
}

#endif //__SLAVE_OPENCL_H__
