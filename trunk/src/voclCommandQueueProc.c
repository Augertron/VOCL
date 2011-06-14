#include <stdio.h>
#include <string.h>
#include "voclOpencl.h"
#include "voclStructures.h"

extern cl_command_queue voclMigCreateCommandQueue(vocl_context context,
                          vocl_device_id device,
                          cl_command_queue_properties properties, cl_int * errcode_ret);
extern int voclContextGetMigrationStatus(vocl_context context);
extern cl_device_id voclVOCLDeviceID2CLDeviceIDComm(vocl_device_id device, int *proxyRank,
                                             int *proxyIndex, MPI_Comm * proxyComm,
                                             MPI_Comm * proxyCommData);
extern void voclUpdateVOCLContext(vocl_context voclContext, int proxyRank, int proxyIndex,
                           MPI_Comm proxyComm, MPI_Comm proxyCommData, vocl_device_id deviceID);
extern size_t voclGetVOCLMemorySize(vocl_mem memory);
extern int voclGetMemWrittenFlag(vocl_mem memory);
extern cl_mem voclVOCLMemory2CLMemory(vocl_mem memory);
extern void voclUpdateVOCLMemory(vocl_mem voclMemory, int proxyRank, int proxyIndex,
                          MPI_Comm proxyComm, MPI_Comm proxyCommData, vocl_context context);
extern cl_int clMigReleaseOldMemObject(vocl_mem memobj);

extern void voclMigrationOnSameRemoteNode(MPI_Comm comm, int rank, cl_command_queue oldCmdQueue,
        cl_mem oldMem, cl_command_queue newCmdQueue, cl_mem newMem, size_t size);
extern int voclMigIssueGPUMemoryRead(MPI_Comm oldComm, int oldRank, MPI_Comm newComm,
                              MPI_Comm newCommData, int newRank, int newIndex, int isFromLocal,
                              int isToLocal, cl_command_queue command_queue, cl_mem mem, size_t size);
extern int voclMigIssueGPUMemoryWrite(MPI_Comm oldComm, MPI_Comm oldCommData, int oldRank, int oldIndex,
                               MPI_Comm newComm, int newRank, int isFromLocal, int isToLocal,
                               cl_command_queue command_queue, cl_mem mem, size_t size);
extern void voclMigLocalToLocal(cl_command_queue oldCmdQueue, cl_mem oldMem,
                         cl_command_queue newCmdQueue, cl_mem newMem, size_t size);
extern void voclMigFinishDataTransfer(MPI_Comm oldComm, int oldRank, int oldIndex, cl_command_queue oldCmdQueue,
                               MPI_Comm newComm, int newRank, int newIndex, cl_command_queue newCmdQueue,
                               int proxySourceRank, int proxyDestRank, int isFromLocal, int isToLocal);


extern int voclIsOnLocalNode(int index);
extern vocl_device_id voclSearchTargetGPU(size_t size);

extern cl_int
clMigEnqueueWriteBuffer(cl_command_queue command_queue,
                     cl_mem buffer,
                     cl_bool blocking_write,
                     size_t offset,
                     size_t cb,
                     const void *ptr,
                     cl_uint num_events_in_wait_list,
                     const cl_event * event_wait_list, cl_event * event);

extern cl_int 
clMigEnqueueReadBuffer(cl_command_queue command_queue /* actual opencl command */ ,
                              cl_mem buffer,
                              cl_bool blocking_read,
                              size_t offset,
                              size_t cb,
                              void *ptr,
                              cl_uint num_events_in_wait_list,
                              const cl_event * event_wait_list, cl_event * event);

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
    commandQueuePtr->migrationStatus = 0; /* not migrated yet */
	commandQueuePtr->deviceMemSize = 0;
	commandQueuePtr->memPtr = NULL;
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
	struct strVOCLMemInfo *memPtr, *tmpMemPtr;
    commandQueuePtr = voclCommandQueuePtr;
    while (commandQueuePtr != NULL) {
        tmpCmdQueuePtr = commandQueuePtr->next;
		memPtr = commandQueuePtr->memPtr;
		while (memPtr != NULL)
		{
			tmpMemPtr = memPtr->next;
			free(memPtr);
			memPtr = tmpMemPtr;
		}
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

void voclCommandQueueSetMigrationStatus(vocl_command_queue cmdQueue, int status)
{
	struct strVOCLCommandQueue *commandQueuePtr = getVOCLCommandQueuePtr(cmdQueue);
	commandQueuePtr->migrationStatus = status;
	return;
}

int voclCommandQueueGetMigrationStatus(vocl_command_queue cmdQueue)
{
	struct strVOCLCommandQueue *commandQueuePtr = getVOCLCommandQueuePtr(cmdQueue);
	return commandQueuePtr->migrationStatus;
}

size_t voclCommandQueueGetDeviceMemorySize(vocl_command_queue cmdQueue)
{
	struct strVOCLCommandQueue *commandQueuePtr = getVOCLCommandQueuePtr(cmdQueue);
	return commandQueuePtr->deviceMemSize;
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
    int err, migrationStatus;

    /* store the command queue information before migration */
    cmdQueuePtr->oldCommandQueue = cmdQueuePtr->clCommandQueue;
    cmdQueuePtr->oldProxyRank = cmdQueuePtr->proxyRank;
    cmdQueuePtr->oldProxyIndex = cmdQueuePtr->proxyIndex;
    cmdQueuePtr->oldProxyComm = cmdQueuePtr->proxyComm;
    cmdQueuePtr->oldProxyCommData = cmdQueuePtr->proxyCommData;
	cmdQueuePtr->oldDeviceID = cmdQueuePtr->deviceID;
    cmdQueuePtr->isOldValid = 1;

    cmdQueuePtr->proxyRank = proxyRank;
    cmdQueuePtr->proxyIndex = proxyIndex;
    cmdQueuePtr->proxyComm = comm;
    cmdQueuePtr->proxyCommData = commData;
	cmdQueuePtr->deviceID = device;
    cmdQueuePtr->clCommandQueue = voclMigCreateCommandQueue(context, device,
                                                            cmdQueuePtr->properties, &err);
	cmdQueuePtr->migrationStatus = voclContextGetMigrationStatus(context);

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
	struct strVOCLMemInfo *memPtr, *tmpMemPtr;

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
	memPtr = curCommandQueuePtr->memPtr;
	while (memPtr != NULL)
	{
		tmpMemPtr = memPtr->next;
		free(memPtr);
		memPtr = tmpMemPtr;
	}
    free(curCommandQueuePtr);

    return 0;
}

void voclUpdateMemoryInCommandQueuePtr(struct strVOCLCommandQueue *commandQueuePtr, 
			vocl_mem mem, size_t size)
{
	struct strVOCLMemInfo *memPtr, *curMemPtr;
	curMemPtr = commandQueuePtr->memPtr;
	while (curMemPtr != NULL)
	{
		if (mem == curMemPtr->memory)
		{
			break;
		}
		curMemPtr = curMemPtr->next;
	}

	/* if the vocl memory is not added yet, add it here */
	if (curMemPtr == NULL)
	{
		memPtr = (struct strVOCLMemInfo *)malloc(sizeof(struct strVOCLMemInfo));
		memPtr->memory = mem;
		memPtr->size = size;
		memPtr->migrationStatus = 0; /* not migrated yet */
		memPtr->next = commandQueuePtr->memPtr;
		commandQueuePtr->memPtr = memPtr;
		commandQueuePtr->deviceMemSize += size;
	}

	return;
}

/* add vocl memory to command queue in case migration is needed */
void voclUpdateMemoryInCommandQueue(vocl_command_queue command_queue, vocl_mem mem, size_t size)
{
	struct strVOCLCommandQueue *commandQueuePtr = getVOCLCommandQueuePtr(command_queue);
	voclUpdateMemoryInCommandQueuePtr(commandQueuePtr, mem, size);
	return;
}

void voclReleaseMemoryFromCommandQueuePtr(struct strVOCLCommandQueue *commandQueuePtr, vocl_mem mem)
{
	struct strVOCLMemInfo *memPtr, *preMemPtr;
	int isMemFound = 0;
	memPtr = commandQueuePtr->memPtr;
	if (memPtr != NULL)
	{
		if (memPtr->memory == mem)
		{
			commandQueuePtr->memPtr = memPtr->next;
			commandQueuePtr->deviceMemSize -= memPtr->size;
			free(memPtr);
			isMemFound = 1;
		}
		
		preMemPtr = memPtr;
		memPtr = memPtr->next;
		while (memPtr != NULL)
		{
			if (memPtr->memory == mem)
			{
				preMemPtr->next = memPtr->next;
				commandQueuePtr->deviceMemSize -= memPtr->size;
				free(memPtr);
				isMemFound = 1;
				break;
			}
			preMemPtr = memPtr;
			memPtr = memPtr->next;
		}
	}

	if (isMemFound == 0)
	{
		printf("voclReleaseMemoryFromCommandQueuePtr, memory not found!\n");
		exit (1);
	}

	return;
}

void voclReleaseMemoryFromCommandQueue(vocl_command_queue command_queue, vocl_mem mem)
{
	struct strVOCLCommandQueue *commandQueuePtr = getVOCLCommandQueuePtr(command_queue);
	voclReleaseMemoryFromCommandQueuePtr(commandQueuePtr, mem);
}

void voclCommandQueueMigration(vocl_command_queue command_queue)
{
	vocl_device_id oldDeviceID, newDeviceID;
	cl_device_id oldClDeviceID, newClDeviceID;
	vocl_context context;
	cl_command_queue oldCmdQueue, newCmdQueue;
	cl_mem oldMem, newMem;
	struct strMigrationCheck tmpMigrationCheck;
	MPI_Comm newComm, newCommData, oldComm, oldCommData;
	MPI_Status status;
	size_t size;
	int newRank, newIndex, oldRank, oldIndex;
	int isFromLocal, isToLocal;
	int proxyDestRank, proxySourceRank;
	int isMemoryWritten, flag;
	char *hostBufPtr;
	struct strVOCLCommandQueue *commandQueuePtr;
	struct strVOCLMemInfo *memPtr;

	clFinish(command_queue);
	/* migrate the related context, memory corresponding to it */
	oldDeviceID = voclGetCommandQueueDeviceID(command_queue);
	size = voclCommandQueueGetDeviceMemorySize(command_queue);
	newDeviceID = voclSearchTargetGPU(size);
	context = voclGetCommandQueueContext(command_queue);
	
	/* get data communication info of the new vocl device */
	oldClDeviceID = voclVOCLDeviceID2CLDeviceIDComm(oldDeviceID, &oldRank, &oldIndex,
					&oldComm, &oldCommData);
	newClDeviceID = voclVOCLDeviceID2CLDeviceIDComm(newDeviceID, &newRank, &newIndex, 
					&newComm, &newCommData);
	printf("newDeviceID = %ld, newIndex = %d\n", newDeviceID, newIndex);

	/* migrate context to the new device */
	voclUpdateVOCLContext(context, newRank, newIndex, newComm, newCommData, newDeviceID);

	/* migrate the command queue */
	oldCmdQueue = voclVOCLCommandQueue2CLCommandQueue(command_queue);
	voclUpdateVOCLCommandQueue(command_queue, newRank, newIndex,
					newComm, newCommData, context, newDeviceID);
	newCmdQueue = voclVOCLCommandQueue2CLCommandQueue(command_queue);
	
	isFromLocal = voclIsOnLocalNode(oldIndex);
	isToLocal = voclIsOnLocalNode(newIndex);

	printf("cmdQueue migration, isFromLocal = %d, isToLocal = %d\n", isFromLocal, isToLocal);

	/* migrate all memory in the command to the new device */
	commandQueuePtr = getVOCLCommandQueuePtr(command_queue);
	memPtr = commandQueuePtr->memPtr;
	isMemoryWritten = 0;
	while (memPtr != NULL)
	{
		size = voclGetVOCLMemorySize(memPtr->memory);
		flag = voclGetMemWrittenFlag(memPtr->memory);

		if (flag == 1)
		{
			/* allocate host buffer */
			hostBufPtr = (char *)malloc(size);
			/* Read data from device memory */
			clMigEnqueueReadBuffer(oldCmdQueue, memPtr->memory,
					CL_TRUE, 0, size, hostBufPtr, 0, NULL, NULL);
			
			//oldMem = voclVOCLMemory2CLMemory(memPtr->memory);
			voclUpdateVOCLMemory(memPtr->memory, newRank, newIndex, newComm,
					newCommData, context);

			clMigEnqueueWriteBuffer(command_queue, memPtr->memory,
					CL_TRUE, 0, size, hostBufPtr, 0, NULL, NULL);
			free(hostBufPtr);
//			newMem = voclVOCLMemory2CLMemory(memPtr->memory);
//			if (isFromLocal == 0 || isToLocal == 0)
//			{
//				if (isFromLocal == 0 && isToLocal == 0 && oldIndex == newIndex)
//				{
//					voclMigrationOnSameRemoteNode(oldComm, oldRank, oldCmdQueue, oldMem,
//						newCmdQueue, newMem, size);
//				}
//				else
//				{
//
//				proxyDestRank = voclMigIssueGPUMemoryRead(oldComm, oldRank, newComm, newCommData,
//						newRank, newIndex, isFromLocal, isToLocal, oldCmdQueue,
//						oldMem, size);
//                proxySourceRank = voclMigIssueGPUMemoryWrite(oldComm, oldCommData, oldRank, 
//						oldIndex, newComm, newRank, isFromLocal, isToLocal,
//						newCmdQueue, newMem, size);
//				}
//			}
//			else
//			{
//				voclMigLocalToLocal(oldCmdQueue, oldMem, newCmdQueue, newMem, size);
//			}
//			isMemoryWritten = 1;
		}
		else
		{
			voclUpdateVOCLMemory(memPtr->memory, newRank, newIndex, 
					newComm, newCommData, context);
		}
		memPtr = memPtr->next;
	}

//	if (isMemoryWritten == 1)
//	{
//		voclMigFinishDataTransfer(oldComm, oldRank, oldIndex, oldCmdQueue, newComm, newRank,
//			newIndex, newCmdQueue, proxySourceRank, proxyDestRank,
//			isFromLocal, isToLocal);
//	}

	/* release old memory */
	memPtr = commandQueuePtr->memPtr;
	while (memPtr != NULL)
	{
		clMigReleaseOldMemObject(memPtr->memory);
		memPtr = memPtr->next;
	}

	/* tell proxy process migration is completed and it can process other messages */
	if (isFromLocal == 0)
	{
		tmpMigrationCheck.releaseMigLock = 1;
		MPI_Send(&tmpMigrationCheck, sizeof(struct strMigrationCheck), MPI_BYTE, 
			oldRank, MIGRATION_CHECK, oldComm);
		MPI_Recv(&tmpMigrationCheck, sizeof(struct strMigrationCheck), MPI_BYTE,
			oldRank, MIGRATION_CHECK, oldComm, &status);
	}

	return;
}

