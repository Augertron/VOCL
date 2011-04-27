#include "vocl_structures.h"

static struct strVOCLCommandQueue *voclCommandQueuePtr = NULL;
static vocl_command_queue voclCommandQueue;
static int voclCommandQueueNum;
static int voclCommandQueueNo;

static vocl_command_queue getVOCLCommandQueueValue()
{
    vocl_command_queue commandQueue = voclCommandQueue;
	voclCommandQueue++;

    return commandQueue;
}

static struct strVOCLCommandQueue *getVOCLCommandQueuePtr()
{
    if (voclCommandQueueNo >= voclCommandQueueNum) {
        voclCommandQueueNum *= 2;
        voclCommandQueuePtr = (struct strVOCLCommandQueue *) realloc(voclCommandQueuePtr,
                                                   voclCommandQueueNum *
                                                   sizeof(struct strVOCLCommandQueue));
    }
    return &voclCommandQueuePtr[voclCommandQueueNo++];
}


void voclCommandQueueInitialize()
{
    voclCommandQueueNum = VOCL_CMD_QUEUE_NUM;
    voclCommandQueuePtr =
        (struct strVOCLCommandQueue *) malloc(voclCommandQueueNum * sizeof(struct strVOCLCommandQueue));
    voclCommandQueueNo = 0;
    voclCommandQueue = 0;
}

void voclCommandQueueFinalize()
{
    if (voclCommandQueuePtr != NULL) {
        free(voclCommandQueuePtr);
        voclCommandQueuePtr = NULL;
    }
    voclCommandQueueNo = 0;
    voclCommandQueue = 0;
    voclCommandQueueNum = 0;
}

vocl_command_queue voclCLCommandQueue2VOCLCommandQueue(cl_command_queue commandQueue, int proxyID)
{
    struct strVOCLCommandQueue *commandQueuePtr = getVOCLCommandQueuePtr();
    commandQueuePtr->clCommandQueue = commandQueue;
	commandQueuePtr->proxyID = proxyID;
    commandQueuePtr->voclCommandQueue = getVOCLCommandQueueValue();

    return commandQueuePtr->voclCommandQueue;
}

cl_command_queue voclVOCLCommandQueue2CLCommandQueueComm(vocl_command_queue commandQueue, int *proxyID)
{
    /* the vocl commandQueue value indicates its location */
    /* in the event buffer */
    int commandQueueNo = (int) commandQueue;
	*proxyID = voclCommandQueuePtr[commandQueueNo].proxyID;

    return voclCommandQueuePtr[commandQueueNo].clCommandQueue;
}

