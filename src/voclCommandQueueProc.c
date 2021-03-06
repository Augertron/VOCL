#include <stdio.h>
#include <string.h>
#include "voclOpencl.h"
#include "voclStructures.h"

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
    commandQueuePtr =
        (struct strVOCLCommandQueue *) malloc(sizeof(struct strVOCLCommandQueue));
    commandQueuePtr->migrationStatus = 0; /* not migrated yet */
    commandQueuePtr->next = voclCommandQueuePtr;
    voclCommandQueuePtr = commandQueuePtr;

    return commandQueuePtr;
}

struct strVOCLCommandQueue *voclGetCommandQueuePtr(vocl_command_queue command_queue)
{
    struct strVOCLCommandQueue *commandQueuePtr;
    commandQueuePtr = voclCommandQueuePtr;
    while (commandQueuePtr != NULL) {
        if (commandQueuePtr->voclCommandQueue == command_queue) {
            break;
        }
        commandQueuePtr = commandQueuePtr->next;
    }

    if (commandQueuePtr == NULL) {
        printf("Error, command_queue does not exist!\n");
        exit(1);
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
    while (commandQueuePtr != NULL) {
        tmpCmdQueuePtr = commandQueuePtr->next;
        free(commandQueuePtr);
        commandQueuePtr = tmpCmdQueuePtr;
    }

    voclCommandQueuePtr = NULL;
    voclCommandQueueNo = 0;
    voclCommandQueue = 0;
}

vocl_command_queue voclCLCommandQueue2VOCLCommandQueue(cl_command_queue command_queue,
                                                       int proxyRank, int proxyIndex,
                                                       MPI_Comm proxyComm,
                                                       MPI_Comm proxyCommData)
{
    struct strVOCLCommandQueue *commandQueuePtr = createVOCLCommandQueue();
    commandQueuePtr->clCommandQueue = command_queue;
    commandQueuePtr->proxyRank = proxyRank;
    commandQueuePtr->proxyIndex = proxyIndex;
    commandQueuePtr->proxyComm = proxyComm;
    commandQueuePtr->proxyCommData = proxyCommData;
    commandQueuePtr->voclCommandQueue = getVOCLCommandQueueValue();

    return commandQueuePtr->voclCommandQueue;
}

void voclStoreCmdQueueProperties(vocl_command_queue cmdQueue,
                                 cl_command_queue_properties properties, vocl_context context,
                                 vocl_device_id deviceID)
{
    struct strVOCLCommandQueue *commandQueuePtr = voclGetCommandQueuePtr(cmdQueue);
    commandQueuePtr->properties = properties;
    commandQueuePtr->context = context;
    commandQueuePtr->deviceID = deviceID;

    return;
}

void voclStoreCmdQueueDeviceID(vocl_command_queue cmdQueue, vocl_device_id deviceID)
{
    struct strVOCLCommandQueue *commandQueuePtr = voclGetCommandQueuePtr(cmdQueue);
    commandQueuePtr->deviceID = deviceID;

    return;
}

vocl_device_id voclGetCommandQueueDeviceID(vocl_command_queue cmdQueue)
{
    struct strVOCLCommandQueue *commandQueuePtr = voclGetCommandQueuePtr(cmdQueue);
    return commandQueuePtr->deviceID;
}

vocl_context voclGetCommandQueueContext(vocl_command_queue cmdQueue)
{
    struct strVOCLCommandQueue *commandQueuePtr = voclGetCommandQueuePtr(cmdQueue);
    return commandQueuePtr->context;
}

void voclCommandQueueSetMigrationStatus(vocl_command_queue cmdQueue, char status)
{
	struct strVOCLCommandQueue *commandQueuePtr = voclGetCommandQueuePtr(cmdQueue);
	commandQueuePtr->migrationStatus = status;
	return;
}

char voclCommandQueueGetMigrationStatus(vocl_command_queue cmdQueue)
{
	struct strVOCLCommandQueue *commandQueuePtr = voclGetCommandQueuePtr(cmdQueue);
	return commandQueuePtr->migrationStatus;
}

cl_command_queue voclVOCLCommandQueue2CLCommandQueueComm(vocl_command_queue command_queue,
                                                         int *proxyRank, int *proxyIndex,
                                                         MPI_Comm * proxyComm,
                                                         MPI_Comm * proxyCommData)
{
    struct strVOCLCommandQueue *commandQueuePtr = voclGetCommandQueuePtr(command_queue);
    *proxyRank = commandQueuePtr->proxyRank;
    *proxyIndex = commandQueuePtr->proxyIndex;
    *proxyComm = commandQueuePtr->proxyComm;
    *proxyCommData = commandQueuePtr->proxyCommData;

    return commandQueuePtr->clCommandQueue;
}

cl_command_queue voclVOCLCommandQueue2CLCommandQueue(vocl_command_queue command_queue)
{
    struct strVOCLCommandQueue *commandQueuePtr = voclGetCommandQueuePtr(command_queue);
    return commandQueuePtr->clCommandQueue;
}

void voclUpdateVOCLCommandQueue(vocl_command_queue voclCmdQueue, cl_command_queue newCmdQueue, 
								int proxyRank, int proxyIndex, MPI_Comm comm, MPI_Comm commData)
{
    struct strVOCLCommandQueue *cmdQueuePtr = voclGetCommandQueuePtr(voclCmdQueue);
    int err, migrationStatus;

    cmdQueuePtr->proxyRank = proxyRank;
    cmdQueuePtr->proxyIndex = proxyIndex;
    cmdQueuePtr->proxyComm = comm;
    cmdQueuePtr->proxyCommData = commData;
	cmdQueuePtr->clCommandQueue = newCmdQueue;
								
    return;
}

int voclReleaseCommandQueue(vocl_command_queue command_queue)
{
    struct strVOCLCommandQueue *commandQueuePtr, *preCommandQueuePtr, *curCommandQueuePtr;

    /* the first node in the link list */
    if (command_queue == voclCommandQueuePtr->voclCommandQueue) {
        commandQueuePtr = voclCommandQueuePtr;
        voclCommandQueuePtr = voclCommandQueuePtr->next;
        free(commandQueuePtr);

        return 0;
    }

    commandQueuePtr = NULL;
    preCommandQueuePtr = voclCommandQueuePtr;
    curCommandQueuePtr = voclCommandQueuePtr->next;
    while (curCommandQueuePtr != NULL) {
        if (command_queue == curCommandQueuePtr->voclCommandQueue) {
            commandQueuePtr = curCommandQueuePtr;
            break;
        }
        preCommandQueuePtr = curCommandQueuePtr;
        curCommandQueuePtr = curCommandQueuePtr->next;
    }

    if (commandQueuePtr == NULL) {
        printf("command_queue does not exist!\n");
        exit(1);
    }

    /* remote the current node from link list */
    preCommandQueuePtr->next = curCommandQueuePtr->next;
    free(curCommandQueuePtr);

    return 0;
}


