#define CL_GPUV_READ  0
#define CL_GPUV_WRITE 1

typedef struct strDataTransfer {
	cl_event           event;
	MPI_Comm		   comm;
	MPI_Request        request;
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

void processEvents(cl_event *event_list, cl_uint num_events)
{
	DATA_TRANSFER *dataTransferPtr;
	MPI_Status status;
	int i;

	for (i = 0; i < num_events; i++)
	{
		dataTransferPtr = getDataTransferAll(event_list[i]);
		if (dataTransferPtr != NULL)
		{
			if (dataTransferPtr->readOrWrite == CL_GPUV_READ)
			{
				MPI_Isend(dataTransferPtr->host_ptr, dataTransferPtr->msgSize, MPI_BYTE,
						 0, dataTransferPtr->tag, dataTransferPtr->comm, &dataTransferPtr->request);
			}
		}
	}

	for (i = 0; i < num_events; i++)
	{
		dataTransferPtr = getDataTransferAll(event_list[i]);
		if (dataTransferPtr != NULL)
		{
			if (dataTransferPtr->readOrWrite == CL_GPUV_READ)
			{
				MPI_Wait(&dataTransferPtr->request, &status);
			}
			free(dataTransferPtr->host_ptr);
			dataTransferPtr->host_ptr = NULL;
		}

		//release the event
		releaseDataTransferAll(event_list[i]);
	}

}

void processCommandQueue(cl_command_queue command_queue)
{
	CMD_QUEUE *cmdQueue = getCommandQueue(command_queue);
	MPI_Status status;
	DATA_TRANSFER *dataTransferPtr = cmdQueue->dataTransferPtr;
	DATA_TRANSFER *nextDataTransferPtr;
	while (dataTransferPtr != NULL)
	{
		if (dataTransferPtr->readOrWrite == CL_GPUV_READ)
		{
			MPI_Isend(dataTransferPtr->host_ptr, dataTransferPtr->msgSize, MPI_BYTE,
					 0, dataTransferPtr->tag, dataTransferPtr->comm, &dataTransferPtr->request);
		}
		dataTransferPtr = dataTransferPtr->next;
	}

	dataTransferPtr = cmdQueue->dataTransferPtr;
	while (dataTransferPtr != NULL)
	{
		if (dataTransferPtr->readOrWrite == CL_GPUV_READ)
		{
			MPI_Wait(&dataTransferPtr->request, &status);
		}
		free(dataTransferPtr->host_ptr);

		nextDataTransferPtr = dataTransferPtr->next;
		free(dataTransferPtr);
		dataTransferPtr = nextDataTransferPtr;
	}

	cmdQueue->dataTransferPtr = NULL;
	cmdQueue->dataTransferPtrTail = NULL;

	return;
}

