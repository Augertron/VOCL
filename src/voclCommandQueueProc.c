#include <stdio.h>
#include "vocl_structures.h"

extern void increaseObjCount(int proxyIndex);

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
    commandQueuePtr->isOldValid = 0;
    commandQueuePtr->next = voclCommandQueuePtr;
    voclCommandQueuePtr = commandQueuePtr;

    return commandQueuePtr;
}

static struct strVOCLCommandQueue *getVOCLCommandQueuePtr(vocl_command_queue command_queue)
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
    commandQueuePtr->isOldValid = 0;
    commandQueuePtr->voclCommandQueue = getVOCLCommandQueueValue();

    return commandQueuePtr->voclCommandQueue;
}

void voclStoreCmdQueueProperties(vocl_command_queue cmdQueue,
                                 cl_command_queue_properties properties, vocl_context context,
                                 vocl_device_id deviceID)
{
    struct strVOCLCommandQueue *commandQueuePtr = getVOCLCommandQueuePtr(cmdQueue);
    commandQueuePtr->properties = properties;
    commandQueuePtr->context = context;
    commandQueuePtr->deviceID = deviceID;

    return;
}

vocl_device_id voclGetCommandQueueDeviceID(vocl_command_queue cmdQueue)
{
    struct strVOCLCommandQueue *commandQueuePtr = getVOCLCommandQueuePtr(cmdQueue);
    return commandQueuePtr->deviceID;
}

vocl_context voclGetCommandQueueContext(vocl_command_queue cmdQueue)
{
    struct strVOCLCommandQueue *commandQueuePtr = getVOCLCommandQueuePtr(cmdQueue);
    return commandQueuePtr->context;
}

cl_command_queue voclVOCLCommandQueue2CLCommandQueueComm(vocl_command_queue command_queue,
                                                         int *proxyRank, int *proxyIndex,
                                                         MPI_Comm * proxyComm,
                                                         MPI_Comm * proxyCommData)
{
    struct strVOCLCommandQueue *commandQueuePtr = getVOCLCommandQueuePtr(command_queue);
    *proxyRank = commandQueuePtr->proxyRank;
    *proxyIndex = commandQueuePtr->proxyIndex;
    *proxyComm = commandQueuePtr->proxyComm;
    *proxyCommData = commandQueuePtr->proxyCommData;

    return commandQueuePtr->clCommandQueue;
}

cl_command_queue voclVOCLCommandQueue2OldCLCommandQueueComm(vocl_command_queue command_queue,
                                                            int *proxyRank, int *proxyIndex,
                                                            MPI_Comm * proxyComm,
                                                            MPI_Comm * proxyCommData)
{
    struct strVOCLCommandQueue *commandQueuePtr = getVOCLCommandQueuePtr(command_queue);
    *proxyRank = commandQueuePtr->oldProxyRank;
    *proxyIndex = commandQueuePtr->oldProxyIndex;
    *proxyComm = commandQueuePtr->oldProxyComm;
    *proxyCommData = commandQueuePtr->oldProxyCommData;

    return commandQueuePtr->oldCommandQueue;
}

cl_command_queue voclVOCLCommandQueue2CLCommandQueue(vocl_command_queue command_queue)
{
    struct strVOCLCommandQueue *commandQueuePtr = getVOCLCommandQueuePtr(command_queue);
    return commandQueuePtr->clCommandQueue;
}

void voclUpdateVOCLCommandQueue(vocl_command_queue voclCmdQueue, int proxyRank, int proxyIndex,
                                MPI_Comm comm, MPI_Comm commData, vocl_context context,
                                vocl_device_id device)
{
    struct strVOCLCommandQueue *cmdQueuePtr = getVOCLCommandQueuePtr(voclCmdQueue);
    int err;

    /* store the command queue information before migration */
    cmdQueuePtr->oldCommandQueue = cmdQueuePtr->clCommandQueue;
    cmdQueuePtr->oldProxyRank = cmdQueuePtr->proxyRank;
    cmdQueuePtr->oldProxyIndex = cmdQueuePtr->proxyIndex;
    cmdQueuePtr->oldProxyComm = cmdQueuePtr->proxyComm;
    cmdQueuePtr->oldProxyCommData = cmdQueuePtr->proxyCommData;
    cmdQueuePtr->isOldValid = 1;

    cmdQueuePtr->proxyRank = proxyRank;
    cmdQueuePtr->proxyIndex = proxyIndex;
    cmdQueuePtr->proxyComm = comm;
    cmdQueuePtr->proxyCommData = commData;
    cmdQueuePtr->clCommandQueue = voclMigCreateCommandQueue(context, device,
                                                            cmdQueuePtr->properties, &err);
    /* a new command queue is created */
    //printf("update cmdQueue");
    //increaseObjCount(proxyIndex);

    return;
}

void voclSetOldCommandQueueReleased(vocl_command_queue command_queue)
{
    struct strVOCLCommandQueue *cmdQueuePtr = getVOCLCommandQueuePtr(command_queue);
    cmdQueuePtr->isOldValid = 0;

    return;
}

int voclIsOldCommandQueueValid(vocl_command_queue command_queue)
{
    struct strVOCLCommandQueue *cmdQueuePtr = getVOCLCommandQueuePtr(command_queue);
    return cmdQueuePtr->isOldValid;
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
