#include <stdio.h>
#include "vocl_structures.h"

static struct strVOCLCommandQueue *voclCommandQueuePtr = NULL;
static vocl_command_queue voclCommandQueue;
static int voclCommandQueueNo;

static vocl_command_queue getVOCLCommandQueueValue()
{
    vocl_command_queue command_queue = voclCommandQueue;
	voclCommandQueue++;

    return command_queue;
}

static struct strVOCLCommandQueue *createVOCLCommandQueue()
{
	struct strVOCLCommandQueue *commandQueuePtr;
	commandQueuePtr = (struct strVOCLCommandQueue *)malloc(sizeof(struct strVOCLCommandQueue));
	commandQueuePtr->next = voclCommandQueuePtr;
	voclCommandQueuePtr = commandQueuePtr;

	return commandQueuePtr;
}

static struct strVOCLCommandQueue *getVOCLCommandQueuePtr(vocl_command_queue command_queue)
{
	struct strVOCLCommandQueue *commandQueuePtr;
	commandQueuePtr = voclCommandQueuePtr;
	while (commandQueuePtr != NULL)
	{
		if (commandQueuePtr->voclCommandQueue == command_queue)
		{
			break;
		}
		commandQueuePtr = commandQueuePtr->next;
	}

	if (commandQueuePtr == NULL)
	{
		printf("Error, command_queue does not exist!\n");
		exit (1);
	}

	return commandQueuePtr;
}

void voclCommandQueueInitialize()
{
    voclCommandQueuePtr = NULL;
    voclCommandQueueNo = 0;
    voclCommandQueue = 0;
}

void voclCommandQueueFinalize()
{
	struct strVOCLCommandQueue *commandQueuePtr, *tmpCmdQueuePtr;
	commandQueuePtr = voclCommandQueuePtr;
	while (commandQueuePtr != NULL)
	{
		tmpCmdQueuePtr = commandQueuePtr->next;
		free(commandQueuePtr);
		commandQueuePtr = tmpCmdQueuePtr;
	}

    voclCommandQueuePtr = NULL;
    voclCommandQueueNo = 0;
    voclCommandQueue = 0;
}

vocl_command_queue voclCLCommandQueue2VOCLCommandQueue(cl_command_queue command_queue, 
                       int proxyID, int proxyIndex, MPI_Comm proxyComm, MPI_Comm proxyCommData)
{
    struct strVOCLCommandQueue *commandQueuePtr = createVOCLCommandQueue();
    commandQueuePtr->clCommandQueue = command_queue;
	commandQueuePtr->proxyID = proxyID;
	commandQueuePtr->proxyIndex = proxyIndex;
	commandQueuePtr->proxyComm = proxyComm;
	commandQueuePtr->proxyCommData = proxyCommData;
    commandQueuePtr->voclCommandQueue = getVOCLCommandQueueValue();

    return commandQueuePtr->voclCommandQueue;
}

cl_command_queue voclVOCLCommandQueue2CLCommandQueueComm(vocl_command_queue command_queue, 
                     int *proxyID, int *proxyIndex, MPI_Comm *proxyComm, MPI_Comm *proxyCommData)
{
	struct strVOCLCommandQueue *commandQueuePtr = getVOCLCommandQueuePtr(command_queue);
	*proxyID = commandQueuePtr->proxyID;
	*proxyIndex = commandQueuePtr->proxyIndex;
	*proxyComm = commandQueuePtr->proxyComm;
	*proxyCommData = commandQueuePtr->proxyCommData;

    return commandQueuePtr->clCommandQueue;
}

int voclReleaseCommandQueue(vocl_command_queue command_queue)
{
	struct strVOCLCommandQueue *commandQueuePtr, *preCommandQueuePtr, *curCommandQueuePtr;
	/* the first node in the link list */
	if (command_queue == voclCommandQueuePtr->voclCommandQueue)
	{
		commandQueuePtr = voclCommandQueuePtr;
		voclCommandQueuePtr = voclCommandQueuePtr->next;
		free(commandQueuePtr);

		return 0;
	}

	commandQueuePtr = NULL;
	preCommandQueuePtr = voclCommandQueuePtr;
	curCommandQueuePtr = voclCommandQueuePtr->next;
	while (curCommandQueuePtr != NULL)
	{
		if (command_queue == curCommandQueuePtr->voclCommandQueue)
		{
			commandQueuePtr = curCommandQueuePtr;
			break;
		}
		preCommandQueuePtr = curCommandQueuePtr;
		curCommandQueuePtr = curCommandQueuePtr->next;
	}

	if (commandQueuePtr == NULL)
	{
		printf("command_queue does not exist!\n");
		exit (1);
	}

	/* remote the current node from link list */
	preCommandQueuePtr->next = curCommandQueuePtr->next;
	free(curCommandQueuePtr);
	
	return 0;
}
