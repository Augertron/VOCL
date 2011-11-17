#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <memory.h>
#include <CL/opencl.h>
#include <sys/time.h>
#include <sched.h>
#include <pthread.h>
#include "vocl_proxy.h"
#include "vocl_proxy_macro.h"
#include "vocl_proxyStructures.h"
#include "vocl_proxyBufferProc.h"
#include "vocl_proxyInternalQueueUp.h"
#include "vocl_proxyKernelArgProc.h"

#define _PRINT_NODE_NAME

static struct strGetProxyCommInfo tmpGetProxyCommInfo;
static struct strGetPlatformIDs tmpGetPlatformID;
static struct strGetDeviceIDs tmpGetDeviceIDs;
static struct strCreateContext tmpCreateContext;
static struct strCreateCommandQueue tmpCreateCommandQueue;
static struct strCreateProgramWithSource tmpCreateProgramWithSource;
static struct strBuildProgram tmpBuildProgram;
static struct strCreateKernel tmpCreateKernel;
static struct strCreateBuffer tmpCreateBuffer;
static struct strEnqueueWriteBuffer tmpEnqueueWriteBuffer;
static struct strSetKernelArg tmpSetKernelArg;
static struct strVGPUMigration tmpVGPUMigration;
static struct strEnqueueNDRangeKernel tmpEnqueueNDRangeKernel;
static struct strEnqueueNDRangeKernelReply kernelLaunchReply;
static struct strEnqueueReadBuffer tmpEnqueueReadBuffer;
static struct strReleaseMemObject tmpReleaseMemObject;
static struct strReleaseKernel tmpReleaseKernel;
static struct strGetContextInfo tmpGetContextInfo;
static struct strGetProgramBuildInfo tmpGetProgramBuildInfo;
static struct strGetProgramInfo tmpGetProgramInfo;
static struct strReleaseProgram tmpReleaseProgram;
static struct strReleaseCommandQueue tmpReleaseCommandQueue;
static struct strReleaseContext tmpReleaseContext;
static struct strFinish tmpFinish;
static struct strGetDeviceInfo tmpGetDeviceInfo;
static struct strGetPlatformInfo tmpGetPlatformInfo;
static struct strFlush tmpFlush;
static struct strWaitForEvents tmpWaitForEvents;
static struct strCreateSampler tmpCreateSampler;
static struct strGetCommandQueueInfo tmpGetCommandQueueInfo;
static struct strEnqueueMapBuffer tmpEnqueueMapBuffer;
static struct strReleaseEvent tmpReleaseEvent;
static struct strGetEventProfilingInfo tmpGetEventProfilingInfo;
static struct strReleaseSampler tmpReleaseSampler;
static struct strGetKernelWorkGroupInfo tmpGetKernelWorkGroupInfo;
static struct strCreateImage2D tmpCreateImage2D;
static struct strEnqueueCopyBuffer tmpEnqueueCopyBuffer;
static struct strRetainEvent tmpRetainEvent;
static struct strRetainMemObject tmpRetainMemObject;
static struct strRetainKernel tmpRetainKernel;
static struct strRetainCommandQueue tmpRetainCommandQueue;
static struct strEnqueueUnmapMemObject tmpEnqueueUnmapMemObject;
static struct strMigGPUMemoryWrite tmpMigGPUMemoryWrite;
static struct strMigGPUMemoryRead tmpMigGPUMemoryRead;
static struct strMigRemoteGPUMemoryRW tmpMigGPUMemRW;
static struct strMigUpdateVGPU tmpMigUpdateVGPU;
static struct strMigSendLastMessage tmpMigSendLastMessage;
static struct strVoclRebalance tmpVoclRebalance;
static struct strMigQueueOperations tmpMigQueueOperations;
static struct strMigGPUMemoryWriteCmpd tmpMigWriteMemCmpdRst;
static struct strMigGPUMemoryReadCmpd tmpMigReadMemCmpdRst;
static struct strMigRemoteGPURWCmpd tmpMigGPUMemRWCmpd;
static struct strForcedMigration tmpForcedMigration;
static struct strDeviceCmdQueueNums tmpDeviceCmdQueueNums;
static struct strKernelNumOnDevice tmpKernelNumOnDevice;
static struct strVoclProgramEnd tmpVoclProgramEnd;

/* control message requests */
MPI_Request *conMsgRequest;
MPI_Request *conMsgRequestForWait;
int *conMsgRequestIndex;
char voclPortName[MPI_MAX_PORT_NAME];
int voclTotalRequestNum;
int voclCommUsedSize;
extern MPI_Comm *appComm, *appCommData;

/* control message buffer */
//CON_MSG_BUFFER *conMsgBuffer;
char **conMsgBuffer = NULL;

/* variables needed by the helper thread */
extern void *proxyHelperThread(void *);
extern int writeBufferIndexInHelperThread;
extern int helperThreadOperFlag;
extern pthread_barrier_t barrier;
extern pthread_t th, thAppComm;
extern int voclProxyAppIndex;

/* thread for kernel launch on proxy */
extern int voclProxyThreadInternalTerminateFlag;
extern int voclProxyMigReissueWriteNum;
extern int voclProxyMigReissueReadNum;
extern pthread_t thKernelLaunch;
extern pthread_barrier_t barrierMigOperations;
extern pthread_mutex_t internalQueueMutex;
void   *proxyEnqueueThread(void *p);

/* variables from write buffer pool */
//extern int totalRequestNum;
extern int allWritesAreEnqueuedFlag;
extern int allReadBuffersAreCovered;

/* functions from write buffer pool */
extern void initializeWriteBufferAll();
extern void increaseWriteBufferCount(int rank);
extern void finalizeWriteBufferAll();
extern void setWriteBufferFlag(int rank, int index, int flag);
extern void voclResetWriteEnqueueFlag(int rank);
extern int voclGetWriteEnqueueFlag(int rank);
extern void voclResetReadBufferCoveredFlag(int rank);
extern int getNextWriteBufferIndex(int rank);
extern MPI_Request *getWriteRequestPtr(int rank, int index);
extern struct strWriteBufferInfo *getWriteBufferInfoPtr(int rank, int index);
extern cl_int processWriteBuffer(int rank, int curIndex, int bufferNum);
extern cl_int processAllWrites(int rank);
extern int getWriteBufferIndexFromEvent(int rank, cl_event event);

/* functions from read buffer pool */
extern void initializeReadBufferAll();
extern void finalizeReadBufferAll();
extern MPI_Request *getReadRequestPtr(int rank, int index);
extern struct strReadBufferInfo *getReadBufferInfoPtr(int rank, int index);
extern void setReadBufferFlag(int rank, int index, int flag);
extern int getNextReadBufferIndex(int rank);
extern cl_int processReadBuffer(int rank, int curIndex, int bufferNum);
extern int getReadBufferIndexFromEvent(int rank, cl_event event);
extern cl_int processAllReads(int rank);

/*functions for calling actual OpenCL function */
extern void mpiOpenCLGetPlatformIDs(struct strGetPlatformIDs *tmpGetPlatform,
                                    cl_platform_id * platforms);
extern void mpiOpenCLGetDeviceIDs(struct strGetDeviceIDs *tmpGetDeviceIDs,
                                  cl_device_id * devices);
extern void mpiOpenCLCreateContext(struct strCreateContext *tmpCreateContext,
                                   cl_device_id * devices);
extern void mpiOpenCLCreateCommandQueue(struct strCreateCommandQueue *tmpCreateCommandQueue);
extern void mpiOpenCLCreateProgramWithSource(struct strCreateProgramWithSource
                                             *tmpCreateProgramWithSource, char *cSourceCL,
                                             size_t * lengthsArray);
extern void mpiOpenCLBuildProgram(struct strBuildProgram *tmpBuildProgram, char *options,
                                  cl_device_id * devices);
extern void mpiOpenCLCreateKernel(struct strCreateKernel *tmpCreateKernel, char *kernel_name);
extern void mpiOpenCLCreateBuffer(struct strCreateBuffer *tmpCreateBuffer, void *host_ptr);
extern void mpiOpenCLEnqueueWriteBuffer(struct strEnqueueWriteBuffer *tmpEnqueueWriteBuffer,
                                        void *ptr, cl_event * event_wait_list);
extern void mpiOpenCLSetKernelArg(struct strSetKernelArg *tmpSetKernelArg, void *arg_value);
extern void mpiOpenCLEnqueueNDRangeKernel(struct strEnqueueNDRangeKernel
                                          *tmpEnqueueNDRangeKernel,
                                          struct strEnqueueNDRangeKernelReply
                                          *kernelLaunchReply, cl_event * event_wait_list,
                                          size_t * global_work_offset,
                                          size_t * global_work_size, size_t * local_work_size,
                                          kernel_args * args_ptr);
extern void mpiOpenCLEnqueueReadBuffer(struct strEnqueueReadBuffer *tmpEnqueueReadBuffer,
                                       void *ptr, cl_event * event_wait_list);
extern void mpiOpenCLReleaseMemObject(struct strReleaseMemObject *tmpReleaseMemObject);
extern void mpiOpenCLReleaseKernel(struct strReleaseKernel *tmpReleaseKernel);
extern void mpiOpenCLGetContextInfo(struct strGetContextInfo *tmpGetContextInfo,
                                    void *param_value);
extern void mpiOpenCLGetProgramBuildInfo(struct strGetProgramBuildInfo *tmpGetProgramBuildInfo,
                                         void *param_value);
extern void mpiOpenCLGetProgramInfo(struct strGetProgramInfo *tmpGetProgramInfo,
                                    void *param_value);
extern void mpiOpenCLReleaseProgram(struct strReleaseProgram *tmpReleaseProgram);
extern void mpiOpenCLReleaseCommandQueue(struct strReleaseCommandQueue
                                         *tmpReleaseCommandQueue);
extern void mpiOpenCLReleaseContext(struct strReleaseContext *tmpReleaseContext);
extern void mpiOpenCLFinish(struct strFinish *tmpFinish);
extern void mpiOpenCLGetDeviceInfo(struct strGetDeviceInfo *tmpGetDeviceInfo,
                                   void *param_value);
extern void mpiOpenCLGetPlatformInfo(struct strGetPlatformInfo *tmpGetPlatformInfo,
                                     void *param_value);
extern void mpiOpenCLFlush(struct strFlush *tmpFlush);
extern void mpiOpenCLWaitForEvents(struct strWaitForEvents *tmpWaitForEvents,
                                   cl_event * event_list);
extern void mpiOpenCLCreateSampler(struct strCreateSampler *tmpCreateSampler);
extern void mpiOpenCLGetCommandQueueInfo(struct strGetCommandQueueInfo *tmpGetCommandQueueInfo,
                                         void *param_value);
extern void mpiOpenCLEnqueueMapBuffer(struct strEnqueueMapBuffer *tmpEnqueueMapBuffer,
                                      cl_event * event_wait_list);
extern void mpiOpenCLReleaseEvent(struct strReleaseEvent *tmpReleaseEvent);
extern void mpiOpenCLGetEventProfilingInfo(struct strGetEventProfilingInfo
                                           *tmpGetEventProfilingInfo, void *param_value);
extern void mpiOpenCLReleaseSampler(struct strReleaseSampler *tmpReleaseSampler);
extern void mpiOpenCLGetKernelWorkGroupInfo(struct strGetKernelWorkGroupInfo
                                            *tmpGetKernelWorkGroupInfo, void *param_value);
extern void mpiOpenCLCreateImage2D(struct strCreateImage2D *tmpCreateImage2D, void *host_ptr);
extern void mpiOpenCLEnqueueCopyBuffer(struct strEnqueueCopyBuffer *tmpEnqueueCopyBuffer,
                                       cl_event * event_wait_list);
extern void mpiOpenCLRetainEvent(struct strRetainEvent *tmpRetainEvent);
extern void mpiOpenCLRetainMemObject(struct strRetainMemObject *tmpRetainMemObject);
extern void mpiOpenCLRetainKernel(struct strRetainKernel *tmpRetainKernel);
extern void mpiOpenCLRetainCommandQueue(struct strRetainCommandQueue *tmpRetainCommandQueue);
extern void mpiOpenCLEnqueueUnmapMemObject(struct strEnqueueUnmapMemObject
                                           *tmpEnqueueUnmapMemObject,
                                           cl_event * event_wait_list);

extern void voclProxyAddVirtualGPU(int appIndex, int proxyRank, cl_device_id deviceID);
extern void voclProxyAddContextToVGPU(int appIndex, cl_device_id deviceID, vocl_proxy_context *context);
extern void voclProxyRemoveContextFromVGPU(int appIndex, vocl_proxy_context *context);
extern void voclProxyAddCommandQueueToVGPU(int appIndex, cl_device_id deviceID, vocl_proxy_command_queue *command_queue);
extern void voclProxyRemoveCommandQueueFromVGPU(int appIndex, vocl_proxy_command_queue *command_queue);
extern cl_context voclProxyGetNewContextValue(cl_context oldContext);
extern void voclProxyReleaseVirtualGPU(int appIndex, cl_device_id deviceID);
extern void voclProxyReleaseAllVirtualGPU();
extern void voclProxyPrintVirtualGPUs();
extern void voclProxyMigCreateVirtualGPU(int appIndex, int proxyRank, cl_device_id deviceID, char *bufPtr);
extern void vocl_proxyUpdateVirtualGPUInfo(int appIndex, char *msgBuf);
extern void voclProxyGetDeviceIDs();

extern void voclProxyAddContext(cl_context context, cl_uint deviceNum, cl_device_id *deviceIDs);
extern void voclProxyAddCommandQueueToContext(cl_context context, vocl_proxy_command_queue *command_queue);
extern vocl_proxy_context *voclProxyGetContextPtr(cl_context context);
extern char voclProxyGetContextMigStatus(cl_context context);
extern void voclProxySetContextMigStatus(cl_context context, char migStatus);
extern void voclProxyReleaseContext(cl_context context);
extern void voclProxyAddMemToContext(cl_context context, vocl_proxy_mem *mem);
extern void voclProxyRemoveMemFromContext(vocl_proxy_mem *mem);
extern void voclProxyAddProgramToContext(cl_context context, vocl_proxy_program *program);
extern void voclProxyRemoveProgramFromContext(vocl_proxy_program *program);
extern void voclProxyAddCommandQueueToContext(cl_context context, vocl_proxy_command_queue *command_queue);
extern void voclProxyRemoveCommandQueueFromContext(vocl_proxy_command_queue *command_queue);

extern void voclProxyAddCmdQueue(cl_command_queue command_queue, cl_command_queue_properties properties, cl_context context, cl_device_id deviceID);
extern vocl_proxy_command_queue *voclProxyGetCmdQueuePtr(cl_command_queue command_queue);
extern void voclProxySetCommandQueueMigStatus(cl_command_queue command_queue, char migStatus);
extern char voclProxyGetCommandQueueMigStatus(cl_command_queue command_queue);
extern cl_command_queue voclProxyGetNewCommandQueueValue(cl_command_queue oldCommand_queue);
extern void voclProxyReleaseCommandQueue(cl_command_queue command_queue);
extern void voclProxyReleaseAllCommandQueues();
extern void voclProxyAddMemToCmdQueue(cl_command_queue command_queue, vocl_proxy_mem *mem);
extern void voclProxyRemoveMemFromCmdQueue(cl_command_queue command_queue, vocl_proxy_mem *mem);
extern void voclProxyAddKernelToCmdQueue(cl_command_queue command_queue, vocl_proxy_kernel *kernel);
extern void voclProxyRemoveKernelFromCmdQueue(cl_command_queue command_queue, vocl_proxy_kernel *kernel);

/* management of kernel numbers on the node */
extern void voclProxyIncreaseKernelNumInCmdQueue(cl_command_queue cmdQueue, int kernelNum);
extern void voclProxyDecreaseKernelNumInCmdQueue(cl_command_queue cmdQueue, int kernelNum);
extern void voclProxyResetKernelNumInCmdQueue(cl_command_queue cmdQueue);

extern void voclProxyAddMem(cl_mem mem, cl_mem_flags flags, size_t size, cl_context context);
extern vocl_proxy_mem *voclProxyGetMemPtr(cl_mem mem);
extern void voclProxySetMemMigStatus(cl_mem mem, char migStatus);
extern char voclProxyGetMemMigStatus(cl_mem mem);
extern cl_mem voclProxyGetNewMemValue(cl_mem oldMem);
extern void voclProxySetMemWritten(cl_mem mem, int isWritten);
extern void voclProxySetMemWriteCmdQueue(cl_mem mem, cl_command_queue cmdQueue);
extern void voclProxyReleaseMem(cl_mem mem);
extern void voclProxyReleaseAllMems();

extern void voclProxyAddProgram(cl_program program, char *sourceString, size_t sourceSize, int stringNum, size_t *stringSizeArray, cl_context context);
extern vocl_proxy_program *voclProxyGetProgramPtr(cl_program program);
extern void voclProxyAddKernelToProgram(cl_program program, vocl_proxy_kernel *kernel);
extern void voclProxySetProgramMigStatus(cl_program program, char migStatus);
extern char voclProxyGetProgramMigStatus(cl_program program);
extern void voclProxyRemoveKernelFromProgram(vocl_proxy_kernel *kernel);
extern void voclProxySetProgramBuildOptions(cl_program program, cl_uint deviceNum, cl_device_id *device_list, char *buildOptions);
extern char* voclProxyGetProgramBuildOptions(cl_program program);
extern cl_program voclProxyGetNewProgramValue(cl_program oldProgram);
extern cl_device_id* voclProxyGetProgramDevices(cl_program program, cl_uint *deviceNum);
extern void voclProxyReleaseProgram(cl_program program);
extern void voclProxyReleaseAllPrograms();

extern void voclProxyAddKernel(cl_kernel kernel, char *kernelName, cl_program program);
extern vocl_proxy_kernel *voclProxyGetKernelPtr(cl_kernel kernel);
extern void voclProxySetKernelMigStatus(cl_kernel kernel, char migStatus);
extern cl_kernel voclProxyGetNewKernelValue(cl_kernel oldKernel);
extern char voclProxyGetKernelMigStatus(cl_kernel kernel);
extern void voclProxySetKernelArgFlag(cl_kernel kernel, int argNum, char *argFlag);
extern void voclProxyStoreKernelArgs(cl_kernel kernel, int argNum, kernel_args *args);
extern void voclProxyUpdateKernelArgs(cl_kernel kernel, int argNum, kernel_args *args);
extern void voclProxyReleaseKernel(cl_kernel kernel);
extern void voclProxyReleaseAllKernels();

extern void voclProxyCommInitialize();
extern void voclProxyCommFinalize();
extern void voclProxyAcceptOneApp();
extern void voclProxyDisconnectOneApp(int commIndex);
extern void *proxyCommAcceptThread(void *p);
extern void voclProxyAcceptProxyMessages();
extern void voclProxyReleaseProxyMsgReceive();

/* for MPI window management */
extern void voclProxyWinInitialize();
extern void voclProxyWinFinalize();
extern void voclProxyCreateWin(MPI_Comm comm, int appIndex, int proxyIndexInApp);
extern void voclProxyFreeWin(int appIndex);
extern void voclProxyPrintWinInfo();

/* migration functions */
extern void voclMigWriteBufferInitializeAll();
extern void voclMigWriteBufferFinalize();
extern void voclMigSetWriteBufferFlag(int rank, int index, int flag);
extern struct strMigWriteBufferInfo *voclMigGetWriteBufferPtr(int rank, int index);
extern MPI_Request *voclMigGetWriteRequestPtr(int rank, int index);
extern int voclMigGetNextWriteBufferIndex(int rank);
extern int voclMigFinishDataWrite(int rank);

extern void voclMigReadBufferInitializeAll();
extern void voclMigReadBufferFinalize();
extern void voclMigSetReadBufferFlag(int rank, int index, int flag);
extern cl_event *voclMigGetReadEventPtr(int rank, int index);
extern struct strMigReadBufferInfo *voclMigGetReadBufferPtr(int rank, int index);
extern int voclMigGetNextReadBufferIndex(int rank);
extern int voclMigFinishDataRead(int rank);

extern void voclMigRWBufferInitializeAll();
extern void voclMigRWBufferFinalize();
extern int voclMigRWGetNextBufferIndex(int rank);
extern struct strMigRWBufferSameNode *voclMigRWGetBufferInfoPtr(int rank, int index);
extern void voclMigSetRWBufferFlag(int rank, int index, int flag);
extern int voclMigFinishDataRWOnSameNode(int rank);
extern void voclProxyRecvOnlyMigrationMsgs(int index);
extern void voclProxyRecvAllMsgs(int index);

extern void vocl_proxyGetKernelNumsOnGPUs(struct strKernelNumOnDevice *gpuKernelNum);
extern cl_int voclProxyMigration(int appIndex, cl_device_id deviceID);
extern void voclProxyMigRecvDeviceMemoryData(int appIndex, cl_device_id deviceID, int sourceRankNo);
extern void voclProxyMigrationOneVGPUInKernelLaunch(int appIndex, int appRank,
                    MPI_Comm appComm, cl_device_id deviceID,
                    struct strEnqueueNDRangeKernel *conMsgKernelLaunch, char *kernelDimInfo);
extern int voclProxyCheckIsMigrationNeeded(int appIndex, cl_device_id deviceID);
extern size_t voclProxyMigUpdateInternalCommand(vocl_internal_command_queue *cmdPtr, char *msgBuffer);

extern int voclMigOrigProxyRank;
extern int voclMigDestProxyRank;
extern MPI_Comm voclMigDestComm;
extern MPI_Comm voclMigDestCommData;
extern int voclMigAppIndexOnOrigProxy;
extern int voclMigAppIndexOnDestProxy;
extern int voclProxyMigAppIndex;
extern cl_command_queue voclProxyMigCmdQueue;

extern void voclProxyInternalQueueInit();
extern void voclProxyInternalQueueFinalize();
extern void voclProxyUnlockItem(vocl_internal_command_queue *cmdPtr);
extern vocl_internal_command_queue * voclProxyGetInternalQueueTail();

/* functions to manage objects allocated in the proxy process */
extern void voclProxyObjCountInitialize();
extern void voclProxyObjCountFinalize();
extern void voclProxyObjCountIncrease();
extern void voclProxyObjCountDecrease();

//debug----------------------------------------------------
extern int voclProxyGetInternalQueueOperationNum();
extern int voclProxyGetInternalQueueKernelLaunchNum(int appIndex);
extern void voclProxySetMigrationCondition(int condition);
extern int voclProxyGetMigrationCondition();
extern int voclProxyGetIsInMigration();
extern void voclProxySetForcedMigrationFlag(int flag);
extern void voclProxySetKernelNumThreshold(int kernelNum);
int rankNo;
//----------------------------------------------------------

/* proxy process */
int main(int argc, char *argv[])
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(10, &set);
    sched_setaffinity(0, sizeof(set), &set);

    int proxyRank, i, j, appRank, appIndex, commIndex, multiThreadProvided;
    cl_int err;
    MPI_Status status;
    char serviceName[256], voclProxyHostName[256];
    int len;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &multiThreadProvided);

    /* issue non-blocking receive for all control messages */
    MPI_Status *curStatus;
    MPI_Request *curRequest;
    int requestNo, index;
    int requestOffset;
//    int rankNo;
    int bufferNum, bufferIndex;
    size_t bufferSize, remainingSize;

    /* variables used by OpenCL API function */
    cl_platform_id *platforms;
    cl_device_id *devices;
    cl_uint numPlatforms, numDevices, num_entries;
    cl_event *event_wait_list;
    cl_uint num_events_in_wait_list;
    cl_ulong globalSize;

    struct strWriteBufferInfo *writeBufferInfoPtr;
    struct strReadBufferInfo *readBufferInfoPtr;
    struct strMigWriteBufferInfo *migWriteBufferInfoPtr;
    struct strMigReadBufferInfo *migReadBufferInfoPtr;
    struct strMigRWBufferSameNode *migRWBufferInfoPtr;
	int isInMigration;
	vocl_internal_command_queue *voclCmdQueuePtr, *cmdPtr;
	struct strEnqueueNDRangeKernel *enqueueNDRangeKernel;
	struct strEnqueueWriteBuffer *writeBuffer;
	struct strEnqueueReadBuffer *readBuffer;
	struct strFinish *finishPtr;
	struct strFlush *flushPtr;
	
	vocl_virtual_gpu *vgpuPtr;
	vocl_proxy_context *contextPtr;
	vocl_proxy_command_queue *cmdQueuePtr;
	vocl_proxy_program *programPtr;
	vocl_proxy_kernel *kernelPtr;
	vocl_proxy_mem *memPtr;

    size_t *lengthsArray;
    size_t fileSize, paramBufSize;
    char *fileBuffer;
    char *buildOptionBuffer;
    char *kernelName, *paramBuf, *argFlag;
    void *host_ptr;
    void *arg_value;
    int work_dim;

    size_t *global_work_offset, *global_work_size, *local_work_size;
	size_t paramOffset, migOffset, dimMsgSize;
    kernel_args *args_ptr;
    size_t param_value_size;
    void *param_value;
    cl_uint num_events;
    cl_event *event_list;
    size_t host_buff_size, kernelMsgSize;
    char *kernelMsgBuffer;
	char *migMsgBuffer;

	/* flag to control the execution of the kernel launch thread */
//	int kernelLaunchThreadExecuting = 0;

	// timing info for migration-----------
	struct timeval t1, t2;
	float tmpTime;
	//-----------------------------------

	kernelMsgSize = 2048;
    kernelMsgBuffer = (char *) malloc(sizeof(char) * kernelMsgSize);

    /* get the proxy host name */
    MPI_Get_processor_name(voclProxyHostName, &len);
    voclProxyHostName[len] = '\0';
    MPI_Comm_rank(MPI_COMM_WORLD, &rankNo);

#ifdef _PRINT_NODE_NAME
    {
        printf("rank = %d, proxyHostName = %s\n", rankNo, voclProxyHostName);
    }
#endif

    /* service name based on different proxy node name */
    sprintf(serviceName, "voclCloud%s", voclProxyHostName);
    /* open a port on each node */
    MPI_Open_port(MPI_INFO_NULL, voclPortName);
    err = MPI_Publish_name(serviceName, MPI_INFO_NULL, voclPortName);

    curStatus = (MPI_Status *) malloc(sizeof(MPI_Status) * TOTAL_MSG_NUM);
    curRequest = (MPI_Request *) malloc(sizeof(MPI_Request) * TOTAL_MSG_NUM);

    voclProxyCommInitialize();

	voclProxyGetDeviceIDs();

    /* record objects allocated */
    voclProxyObjCountInitialize();

    /* initialize write and read buffer pools */
    initializeWriteBufferAll();
    initializeReadBufferAll();

    voclMigWriteBufferInitializeAll();
    voclMigReadBufferInitializeAll();
    voclMigRWBufferInitializeAll();
	voclProxyWinInitialize();
	/*initialize the internal command queue in the proxy process */
	voclProxyInternalQueueInit();
	/* issue non-blocking receiving for messages from other processes */
	voclProxyAcceptProxyMessages();

	/* set migration condition to not migrate */
	voclProxySetMigrationCondition(0);
	/* set kernel num threshold for migration, a very large num */
	voclProxySetKernelNumThreshold(10);

	pthread_barrier_init(&barrierMigOperations, NULL, 2);
	pthread_mutex_init(&internalQueueMutex, NULL);
	pthread_create(&thKernelLaunch, NULL, proxyEnqueueThread, NULL);

    /* wait for one app to issue connection request */
    voclProxyAcceptOneApp();

    /* create a helper thread */
    pthread_barrier_init(&barrier, NULL, 2);
    pthread_create(&th, NULL, proxyHelperThread, NULL);
    pthread_create(&thAppComm, NULL, proxyCommAcceptThread, NULL);

    while (1) {
        /* wait for any msg from the master process */
		printf("BeforewaitAny\n");
        MPI_Waitany(voclCommUsedSize, conMsgRequestForWait, &commIndex, &status);
        appRank = status.MPI_SOURCE;
        appIndex = commIndex;
        index = conMsgRequestIndex[commIndex];
        conMsgRequest[index] = MPI_REQUEST_NULL;
        if (++conMsgRequestIndex[commIndex] >= (commIndex + 1) * CMSG_NUM) {
            conMsgRequestIndex[commIndex] = commIndex * CMSG_NUM;
        }
        conMsgRequestForWait[commIndex] = conMsgRequest[conMsgRequestIndex[commIndex]];

        //debug-----------------------------
        printf("rank = %d, requestNum = %d, appIndex = %d, index = %d, tag = %d\n",
              rankNo, voclTotalRequestNum, appIndex, index, status.MPI_TAG);
        //-------------------------------------

		if (status.MPI_TAG == GET_PROXY_COMM_INFO) {
			memcpy((void *)&tmpGetProxyCommInfo, (const void *) conMsgBuffer[index],
					sizeof(tmpGetProxyCommInfo));
			tmpGetProxyCommInfo.proxyRank = rankNo;
			/* get communicator across different proxy process */
			tmpGetProxyCommInfo.comm = appComm[0]; 
			tmpGetProxyCommInfo.commData = appCommData[0];
			tmpGetProxyCommInfo.appIndex = appIndex;

			MPI_Send(&tmpGetProxyCommInfo, sizeof(struct strGetProxyCommInfo), MPI_BYTE,
					 appRank, GET_PROXY_COMM_INFO, appComm[commIndex]);

			/* create the win for one-sided data communication */
			voclProxyCreateWin(appComm[commIndex], appIndex, tmpGetProxyCommInfo.proxyIndexInApp);
		}

        else if (status.MPI_TAG == GET_PLATFORM_ID_FUNC) {
            memcpy((void *) &tmpGetPlatformID, (const void *) conMsgBuffer[index],
                   sizeof(tmpGetPlatformID));

            platforms = NULL;
            if (tmpGetPlatformID.platforms != NULL) {
                platforms =
                    (cl_platform_id *) malloc(sizeof(cl_platform_id) *
                                              tmpGetPlatformID.num_entries);
            }

            mpiOpenCLGetPlatformIDs(&tmpGetPlatformID, platforms);
            requestNo = 0;
            MPI_Isend(&tmpGetPlatformID, sizeof(tmpGetPlatformID), MPI_BYTE, appRank,
                      GET_PLATFORM_ID_FUNC, appComm[commIndex], curRequest + (requestNo++));
            if (tmpGetPlatformID.platforms != NULL && tmpGetPlatformID.num_entries > 0) {
                MPI_Isend((void *) platforms,
                          sizeof(cl_platform_id) * tmpGetPlatformID.num_platforms, MPI_BYTE,
                          appRank, GET_PLATFORM_ID_FUNC1, appCommData[commIndex],
                          curRequest + (requestNo++));
            }

            MPI_Waitall(requestNo, curRequest, curStatus);
            if (tmpGetPlatformID.platforms != NULL && tmpGetPlatformID.num_entries > 0) {
                free(platforms);
            }
        }

        else if (status.MPI_TAG == GET_DEVICE_ID_FUNC) {
            memcpy(&tmpGetDeviceIDs, conMsgBuffer[index], sizeof(tmpGetDeviceIDs));
            devices = NULL;
            num_entries = tmpGetDeviceIDs.num_entries;
            if (num_entries > 0 && tmpGetDeviceIDs.devices != NULL) {
                devices = (cl_device_id *) malloc(num_entries * sizeof(cl_device_id));
            }
            mpiOpenCLGetDeviceIDs(&tmpGetDeviceIDs, devices);
            requestNo = 0;
            if (num_entries > 0 && tmpGetDeviceIDs.devices != NULL) {
                MPI_Isend(devices, sizeof(cl_device_id) * num_entries, MPI_BYTE, appRank,
                          GET_DEVICE_ID_FUNC1, appCommData[commIndex],
                          curRequest + (requestNo++));
            }
            MPI_Isend(&tmpGetDeviceIDs, sizeof(tmpGetDeviceIDs), MPI_BYTE, appRank,
                      GET_DEVICE_ID_FUNC, appComm[commIndex], curRequest + (requestNo++));

            MPI_Waitall(requestNo, curRequest, curStatus);
            if (num_entries > 0 && tmpGetDeviceIDs.devices != NULL) {
                free(devices);
            }
        }

        else if (status.MPI_TAG == CREATE_CONTEXT_FUNC) {
            memcpy(&tmpCreateContext, conMsgBuffer[index], sizeof(tmpCreateContext));
            devices = NULL;
            if (tmpCreateContext.devices != NULL) {
                devices =
                    (cl_device_id *) malloc(sizeof(cl_device_id) *
                                            tmpCreateContext.num_devices);
                MPI_Irecv(devices, sizeof(cl_device_id) * tmpCreateContext.num_devices,
                          MPI_BYTE, appRank, CREATE_CONTEXT_FUNC1, appCommData[commIndex],
                          curRequest);
                MPI_Wait(curRequest, curStatus);
            }

            mpiOpenCLCreateContext(&tmpCreateContext, devices);

			/* a new virtual GPU is created for each app on each physical GPU. */
			voclProxyAddContext(tmpCreateContext.hContext, tmpCreateContext.num_devices, devices);
			contextPtr = voclProxyGetContextPtr(tmpCreateContext.hContext);
			tmpCreateContext.migStatus = contextPtr->migStatus;

            MPI_Isend(&tmpCreateContext, sizeof(tmpCreateContext), MPI_BYTE, appRank,
                      CREATE_CONTEXT_FUNC, appComm[commIndex], curRequest);

			for (i = 0; i < tmpCreateContext.num_devices; i++)
			{
				voclProxyAddVirtualGPU(appIndex, rankNo, devices[i]);
				voclProxyAddContextToVGPU(appIndex, devices[i], contextPtr);
			}

            MPI_Wait(curRequest, curStatus);
            if (devices != NULL) {
                free(devices);
            }
        }

        else if (status.MPI_TAG == CREATE_COMMAND_QUEUE_FUNC) {
            memcpy(&tmpCreateCommandQueue, conMsgBuffer[index], sizeof(tmpCreateCommandQueue));
            mpiOpenCLCreateCommandQueue(&tmpCreateCommandQueue);
			tmpCreateCommandQueue.migStatus = voclProxyGetContextMigStatus(tmpCreateCommandQueue.context);
            MPI_Isend(&tmpCreateCommandQueue, sizeof(tmpCreateCommandQueue), MPI_BYTE, appRank,
                      CREATE_COMMAND_QUEUE_FUNC, appComm[commIndex], curRequest);

			/* store the command queue */
			voclProxyAddCmdQueue(tmpCreateCommandQueue.clCommand, 
								 tmpCreateCommandQueue.properties, 
								 tmpCreateCommandQueue.context, 
								 tmpCreateCommandQueue.device);

			/* add the cmdQueue to the proxy */
			cmdQueuePtr = voclProxyGetCmdQueuePtr(tmpCreateCommandQueue.clCommand);
			voclProxySetCommandQueueMigStatus(tmpCreateCommandQueue.clCommand, 
											  tmpCreateCommandQueue.migStatus);

			voclProxyAddCommandQueueToContext(tmpCreateCommandQueue.context, cmdQueuePtr);
			voclProxyAddCommandQueueToVGPU(appIndex, tmpCreateCommandQueue.device, cmdQueuePtr);

            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == CREATE_PROGRMA_WITH_SOURCE) {
            memcpy(&tmpCreateProgramWithSource, conMsgBuffer[index],
                   sizeof(tmpCreateProgramWithSource));
            cl_uint count = tmpCreateProgramWithSource.count;
            lengthsArray = (size_t *) malloc(count * sizeof(size_t));

            fileSize = tmpCreateProgramWithSource.lengths;
            fileBuffer = (char *) malloc(fileSize * sizeof(char));

            requestNo = 0;
            MPI_Irecv(lengthsArray, count * sizeof(size_t), MPI_BYTE, appRank,
                      CREATE_PROGRMA_WITH_SOURCE1, appCommData[commIndex],
                      curRequest + (requestNo++));
            MPI_Irecv(fileBuffer, fileSize, MPI_BYTE, appRank, CREATE_PROGRMA_WITH_SOURCE2,
                      appCommData[commIndex], curRequest + (requestNo++));
            MPI_Waitall(requestNo, curRequest, curStatus);

            mpiOpenCLCreateProgramWithSource(&tmpCreateProgramWithSource, fileBuffer,
                                             lengthsArray);
			tmpCreateProgramWithSource.migStatus = voclProxyGetContextMigStatus(tmpCreateProgramWithSource.context);

            MPI_Isend(&tmpCreateProgramWithSource, sizeof(tmpCreateProgramWithSource),
                      MPI_BYTE, appRank, CREATE_PROGRMA_WITH_SOURCE, appComm[commIndex],
                      curRequest);

			/*store the program */
			voclProxyAddProgram(tmpCreateProgramWithSource.clProgram, 
								fileBuffer, fileSize, 
								tmpCreateProgramWithSource.count, 
								lengthsArray, 
								tmpCreateProgramWithSource.context);

			/* set program migration status */
			voclProxySetProgramMigStatus(tmpCreateProgramWithSource.clProgram, tmpCreateProgramWithSource.migStatus);

			/* add the program to context */
			programPtr = voclProxyGetProgramPtr(tmpCreateProgramWithSource.clProgram);
			voclProxyAddProgramToContext(tmpCreateProgramWithSource.context, programPtr);

            free(fileBuffer);
            free(lengthsArray);
            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == BUILD_PROGRAM) {
            memcpy(&tmpBuildProgram, conMsgBuffer[index], sizeof(tmpBuildProgram));
            buildOptionBuffer = NULL;
            requestNo = 0;
            if (tmpBuildProgram.optionLen > 0) {
                buildOptionBuffer =
                    (char *) malloc((tmpBuildProgram.optionLen + 1) * sizeof(char));
                MPI_Irecv(buildOptionBuffer, tmpBuildProgram.optionLen, MPI_BYTE, appRank,
                          BUILD_PROGRAM1, appCommData[commIndex], curRequest + (requestNo++));
                buildOptionBuffer[tmpBuildProgram.optionLen] = '\0';
            }

            devices = NULL;
            if (tmpBuildProgram.device_list != NULL) {
                devices =
                    (cl_device_id *) malloc(sizeof(cl_device_id) *
                                            tmpBuildProgram.num_devices);
                MPI_Irecv(devices, sizeof(cl_device_id) * tmpBuildProgram.num_devices,
                          MPI_BYTE, appRank, BUILD_PROGRAM, appCommData[commIndex],
                          curRequest + (requestNo++));
            }
            if (requestNo > 0) {
                MPI_Waitall(requestNo, curRequest, curStatus);
            }

            mpiOpenCLBuildProgram(&tmpBuildProgram, buildOptionBuffer, devices);
            MPI_Isend(&tmpBuildProgram, sizeof(tmpBuildProgram), MPI_BYTE, appRank,
                      BUILD_PROGRAM, appComm[commIndex], curRequest);

			/* store the program build info */
			voclProxySetProgramBuildOptions(tmpBuildProgram.program,
											tmpBuildProgram.num_devices,
											devices,
											buildOptionBuffer);

            if (tmpBuildProgram.optionLen > 0) {
                free(buildOptionBuffer);
            }
            if (tmpBuildProgram.device_list != NULL) {
                free(devices);
            }
            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == CREATE_KERNEL) {
            memcpy(&tmpCreateKernel, conMsgBuffer[index], sizeof(tmpCreateKernel));

			paramBufSize = tmpCreateKernel.kernelNameSize + tmpCreateKernel.argNum * sizeof(char);
			paramBuf = (char *) malloc(paramBufSize);
            kernelName = (char *) malloc(tmpCreateKernel.kernelNameSize + 1);

            MPI_Irecv(paramBuf, paramBufSize, MPI_BYTE, appRank,
                      CREATE_KERNEL1, appCommData[commIndex], curRequest);
            MPI_Wait(curRequest, curStatus);

			memcpy(kernelName, paramBuf, tmpCreateKernel.kernelNameSize);
            kernelName[tmpCreateKernel.kernelNameSize] = '\0';
			argFlag = (char *)&paramBuf[tmpCreateKernel.kernelNameSize];

            mpiOpenCLCreateKernel(&tmpCreateKernel, kernelName);
			tmpCreateKernel.migStatus = voclProxyGetProgramMigStatus(tmpCreateKernel.program);

            MPI_Isend(&tmpCreateKernel, sizeof(tmpCreateKernel), MPI_BYTE, appRank,
                      CREATE_KERNEL, appComm[commIndex], curRequest);

			/* store the kernel */
			voclProxyAddKernel(tmpCreateKernel.kernel,
							   kernelName,
							   tmpCreateKernel.program);

			/* set kernel migration status */
			voclProxySetKernelMigStatus(tmpCreateKernel.kernel, tmpCreateKernel.migStatus);

			/* get the kernel pointer and add it to the program */
			kernelPtr = voclProxyGetKernelPtr(tmpCreateKernel.kernel);
			voclProxyAddKernelToProgram(tmpCreateKernel.program, kernelPtr);

			/* set the kernel argu flag to indicate whether a parameter is a global memory */
			voclProxySetKernelArgFlag(tmpCreateKernel.kernel, 
									  tmpCreateKernel.argNum,
									  argFlag);
			
			free(paramBuf);
            free(kernelName);
            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == CREATE_BUFFER_FUNC) {
            memcpy(&tmpCreateBuffer, conMsgBuffer[index], sizeof(tmpCreateBuffer));
            host_ptr = NULL;
            if (tmpCreateBuffer.host_ptr_flag == 1) {
                host_ptr = malloc(tmpCreateBuffer.size);
                MPI_Irecv(host_ptr, tmpCreateBuffer.size, MPI_BYTE, appRank,
                          CREATE_BUFFER_FUNC1, appCommData[commIndex], curRequest);
                MPI_Wait(curRequest, curStatus);
            }
            mpiOpenCLCreateBuffer(&tmpCreateBuffer, host_ptr);
			tmpCreateBuffer.migStatus = voclProxyGetContextMigStatus(tmpCreateBuffer.context);

            MPI_Isend(&tmpCreateBuffer, sizeof(tmpCreateBuffer), MPI_BYTE, appRank,
                      CREATE_BUFFER_FUNC, appComm[commIndex], curRequest);

			/* store the memory */
			voclProxyAddMem(tmpCreateBuffer.deviceMem, 
							tmpCreateBuffer.flags,
							tmpCreateBuffer.size,
							tmpCreateBuffer.context);

			/* set memory migration status */
			voclProxySetMemMigStatus(tmpCreateBuffer.deviceMem, tmpCreateBuffer.migStatus);

			/* store the memory in the context */
			memPtr = voclProxyGetMemPtr(tmpCreateBuffer.deviceMem);
			voclProxyAddMemToContext(tmpCreateBuffer.context, memPtr);

            if (tmpCreateBuffer.host_ptr_flag == 1) {
                free(host_ptr);
            }
            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == ENQUEUE_WRITE_BUFFER) {
			voclCmdQueuePtr = voclProxyGetInternalQueueTail();
			voclCmdQueuePtr->msgTag = ENQUEUE_WRITE_BUFFER;
			memcpy(voclCmdQueuePtr->conMsgBuffer, conMsgBuffer[index], sizeof(tmpEnqueueWriteBuffer));
			voclCmdQueuePtr->appComm = appComm[commIndex];
			voclCmdQueuePtr->appCommData = appCommData[commIndex];
			voclCmdQueuePtr->appRank = appRank;
			voclCmdQueuePtr->appIndex = appIndex;
			voclProxyUnlockItem(voclCmdQueuePtr);
			//if (voclProxyGetMigrationCondition() == 1)
			//{
			//	writeBuffer = (struct strEnqueueWriteBuffer *)voclCmdQueuePtr->conMsgBuffer;
			//	cmdQueuePtr = voclProxyGetCmdQueuePtr(writeBuffer->command_queue);
			//	voclProxyMigration(appIndex, cmdQueuePtr->deviceID);
			//	voclProxySetMigrationCondition(0);
			//}

//			memcpy(&tmpEnqueueWriteBuffer, conMsgBuffer[index], sizeof(tmpEnqueueWriteBuffer));
//			requestNo = 0;
//			event_wait_list = NULL;
//			num_events_in_wait_list = tmpEnqueueWriteBuffer.num_events_in_wait_list;
//			if (num_events_in_wait_list > 0) {
//				event_wait_list =
//					(cl_event *) malloc(sizeof(cl_event) * num_events_in_wait_list);
//				MPI_Irecv(event_wait_list, sizeof(cl_event) * num_events_in_wait_list,
//						  MPI_BYTE, appRank, tmpEnqueueWriteBuffer.tag, appCommData[commIndex],
//						  curRequest + (requestNo++));
//				MPI_Waitall(requestNo, curRequest, curStatus);
//				requestNo = 0;
//			}
//
//			/* issue MPI data receive */
//			bufferSize = VOCL_PROXY_WRITE_BUFFER_SIZE;
//			bufferNum = (tmpEnqueueWriteBuffer.cb - 1) / bufferSize;
//			remainingSize = tmpEnqueueWriteBuffer.cb - bufferSize * bufferNum;
//			for (i = 0; i <= bufferNum; i++) {
//				if (i == bufferNum)
//					bufferSize = remainingSize;
//
//				bufferIndex = getNextWriteBufferIndex(appIndex);
//				writeBufferInfoPtr = getWriteBufferInfoPtr(appIndex, bufferIndex);
//				MPI_Irecv(writeBufferInfoPtr->dataPtr, bufferSize, MPI_BYTE, appRank,
//						  VOCL_PROXY_WRITE_TAG + bufferIndex, appCommData[commIndex],
//						  //VOCL_PROXY_WRITE_TAG, appCommData[commIndex],
//						  getWriteRequestPtr(appIndex, bufferIndex));
//
//				/* save information for writing to GPU memory */
//				writeBufferInfoPtr->commandQueue = tmpEnqueueWriteBuffer.command_queue;
//				writeBufferInfoPtr->size = bufferSize;
//				writeBufferInfoPtr->offset =
//					tmpEnqueueWriteBuffer.offset + i * VOCL_PROXY_WRITE_BUFFER_SIZE;
//				writeBufferInfoPtr->mem = tmpEnqueueWriteBuffer.buffer;
//				writeBufferInfoPtr->blocking_write = tmpEnqueueWriteBuffer.blocking_write;
//				writeBufferInfoPtr->numEvents = tmpEnqueueWriteBuffer.num_events_in_wait_list;
//				writeBufferInfoPtr->eventWaitList = event_wait_list;
//
//				/* set flag to indicate buffer is being used */
//				setWriteBufferFlag(appIndex, bufferIndex, WRITE_RECV_DATA);
//				increaseWriteBufferCount(appIndex);
//			}
//			voclResetWriteEnqueueFlag(appIndex);
//
//			/* set memory migration state to be written */
//			voclProxySetMemWritten(tmpEnqueueWriteBuffer.buffer, 1);
//			voclProxySetMemWriteCmdQueue(tmpEnqueueWriteBuffer.buffer,
//										 tmpEnqueueWriteBuffer.command_queue);
//
////			//debug, for migration test------------------------------------------------
////			if (rankNo == 1 && voclProxyGetMigrationCondition(rankNo) == 0)
////			{
////				processAllWrites(appIndex);
////				voclProxySetMigrationCondition(rankNo, 1);
////				cmdQueuePtr = voclProxyGetCmdQueuePtr(tmpEnqueueWriteBuffer.command_queue);
////				voclProxyMigration(appIndex, cmdQueuePtr->deviceID);
////			}
////			//end of debug----------------------------------------------------------
//
//			if (tmpEnqueueWriteBuffer.blocking_write == CL_TRUE) {
//				if (requestNo > 0) {
//					MPI_Waitall(requestNo, curRequest, curStatus);
//					requestNo = 0;
//				}
//
//				/* process all previous write and read */
//				tmpEnqueueWriteBuffer.res = processAllWrites(appIndex);
//				tmpEnqueueWriteBuffer.event = writeBufferInfoPtr->event;
//
//				MPI_Isend(&tmpEnqueueWriteBuffer, sizeof(tmpEnqueueWriteBuffer), MPI_BYTE,
//						  appRank, ENQUEUE_WRITE_BUFFER, appComm[commIndex],
//						  curRequest + (requestNo++));
//			}
//			else {
//				if (tmpEnqueueWriteBuffer.event_null_flag == 0) {
//					if (requestNo > 0) {
//						MPI_Waitall(requestNo, curRequest, curStatus);
//						requestNo = 0;
//					}
//					tmpEnqueueWriteBuffer.res =
//						processWriteBuffer(appIndex, bufferIndex, bufferNum + 1);
//					tmpEnqueueWriteBuffer.event = writeBufferInfoPtr->event;
//					writeBufferInfoPtr->numWriteBuffers = bufferNum + 1;
//
//					MPI_Isend(&tmpEnqueueWriteBuffer, sizeof(tmpEnqueueWriteBuffer), MPI_BYTE,
//							  appRank, ENQUEUE_WRITE_BUFFER, appComm[commIndex],
//							  curRequest + (requestNo++));
//				}
//				//else
//				//{
//				//	MPI_Isend(NULL, 0, MPI_BYTE, appRank, ENQUEUE_WRITE_BUFFER, appComm[commIndex],
//				//			  curRequest + (requestNo++));
//				//}
//			}
//
//			if (requestNo > 0) {
//				MPI_Wait(curRequest, curStatus);
//			}
        }

        else if (status.MPI_TAG == SET_KERNEL_ARG) {
            memcpy(&tmpSetKernelArg, conMsgBuffer[index], sizeof(tmpSetKernelArg));
            arg_value = NULL;
            if (tmpSetKernelArg.arg_value != NULL) {
                arg_value = (char *) malloc(tmpSetKernelArg.arg_size);
                MPI_Irecv(arg_value, tmpSetKernelArg.arg_size, MPI_BYTE, appRank,
                          SET_KERNEL_ARG1, appCommData[commIndex], curRequest);
            }
            MPI_Wait(curRequest, curStatus);
            mpiOpenCLSetKernelArg(&tmpSetKernelArg, arg_value);
            MPI_Isend(&tmpSetKernelArg, sizeof(tmpSetKernelArg), MPI_BYTE, appRank,
                      SET_KERNEL_ARG, appComm[commIndex], curRequest);
            if (tmpSetKernelArg.arg_value != NULL) {
                free(arg_value);
            }
            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == VOCL_MIGRATION) {
            memcpy(&tmpVGPUMigration, conMsgBuffer[index], sizeof(struct strVGPUMigration));
			migMsgBuffer = (char *)malloc(tmpVGPUMigration.migMsgSize);
			MPI_Irecv(migMsgBuffer, tmpVGPUMigration.migMsgSize, MPI_BYTE, appRank,
					  VOCL_MIGRATION, appCommData[commIndex], curRequest);
			MPI_Wait(curRequest, curStatus);

			voclProxyMigCreateVirtualGPU(tmpVGPUMigration.appIndex, rankNo, 
										 tmpVGPUMigration.deviceID, migMsgBuffer);

			tmpVGPUMigration.retCode = 0; /* correct */
            MPI_Isend(&tmpVGPUMigration, sizeof(struct strVGPUMigration), MPI_BYTE, appRank,
                      VOCL_MIGRATION, appCommData[commIndex], curRequest);
			free(migMsgBuffer);
            MPI_Wait(curRequest, curStatus);
			voclProxyMigRecvDeviceMemoryData(tmpVGPUMigration.appIndex, tmpVGPUMigration.deviceID, appRank);
        }

		else if (status.MPI_TAG == VOCL_MIG_LAST_MSG)
		{
printf("lastMsg\n");
			/* barrier for starting mig unexecuted commands */
			pthread_barrier_wait(&barrierMigOperations);

			/* wait for end of unexecuted commands */
			pthread_barrier_wait(&barrierMigOperations);

			tmpMigSendLastMessage.reissueWriteNum = voclProxyMigReissueWriteNum;
			tmpMigSendLastMessage.reissueReadNum = voclProxyMigReissueReadNum;

			MPI_Isend(&tmpMigSendLastMessage, sizeof(struct strMigSendLastMessage), MPI_BYTE, 
					  appRank, VOCL_MIG_LAST_MSG, appComm[commIndex], curRequest);
			MPI_Wait(curRequest, curStatus);
		}

        else if (status.MPI_TAG == ENQUEUE_ND_RANGE_KERNEL) {
			voclCmdQueuePtr = voclProxyGetInternalQueueTail();
			voclCmdQueuePtr->msgTag = ENQUEUE_ND_RANGE_KERNEL;
			memcpy(voclCmdQueuePtr->conMsgBuffer, conMsgBuffer[index], sizeof(struct strEnqueueNDRangeKernel));
			voclCmdQueuePtr->appComm = appComm[commIndex];
			voclCmdQueuePtr->appCommData = appCommData[commIndex];
			voclCmdQueuePtr->appRank = appRank;
			voclCmdQueuePtr->appIndex = appIndex;
			enqueueNDRangeKernel = (struct strEnqueueNDRangeKernel *)voclCmdQueuePtr->conMsgBuffer;

			/* if input arguments are available */
			if (enqueueNDRangeKernel->dataSize > 0)
			{
				voclCmdQueuePtr->paramBuf = (char *)malloc(enqueueNDRangeKernel->dataSize);
				MPI_Irecv(voclCmdQueuePtr->paramBuf, enqueueNDRangeKernel->dataSize, MPI_BYTE,
						  appRank, ENQUEUE_ND_RANGE_KERNEL1, appCommData[commIndex], curRequest);
				MPI_Wait(curRequest, curStatus);
			}

			voclProxyUnlockItem(voclCmdQueuePtr);

			//if (voclProxyGetMigrationCondition() == 1)
			//{
			//	cmdQueuePtr = voclProxyGetCmdQueuePtr(enqueueNDRangeKernel->command_queue);
			//	voclProxyMigration(appIndex, cmdQueuePtr->deviceID);
			//	voclProxySetMigrationCondition(0);
			//}

//			memcpy(&tmpEnqueueNDRangeKernel, conMsgBuffer[index],
//				   sizeof(tmpEnqueueNDRangeKernel));
//			requestNo = 0;
//			event_wait_list = NULL;
//			num_events_in_wait_list = tmpEnqueueNDRangeKernel.num_events_in_wait_list;
//			if (num_events_in_wait_list > 0) {
//				event_wait_list =
//					(cl_event *) malloc(sizeof(cl_event) * num_events_in_wait_list);
//				MPI_Irecv(event_wait_list, sizeof(cl_event) * num_events_in_wait_list,
//						  MPI_BYTE, appRank, ENQUEUE_ND_RANGE_KERNEL1, appCommData[commIndex],
//						  curRequest + (requestNo++));
//			}
//
//			work_dim = tmpEnqueueNDRangeKernel.work_dim;
//			args_ptr = NULL;
//			global_work_offset = NULL;
//			global_work_size = NULL;
//			local_work_size = NULL;
//			if (tmpEnqueueNDRangeKernel.dataSize > 0) {
//				if (tmpEnqueueNDRangeKernel.dataSize > kernelMsgSize)
//				{
//					kernelMsgSize = tmpEnqueueNDRangeKernel.dataSize;
//					kernelMsgBuffer = (char *) realloc(kernelMsgBuffer, kernelMsgSize);
//				}
//				MPI_Irecv(kernelMsgBuffer, tmpEnqueueNDRangeKernel.dataSize, MPI_BYTE, appRank,
//						  ENQUEUE_ND_RANGE_KERNEL1, appCommData[commIndex],
//						  curRequest + (requestNo++));
//			}
//
//			MPI_Waitall(requestNo, curRequest, curStatus);
//
//			paramOffset = 0;
//			if (tmpEnqueueNDRangeKernel.global_work_offset_flag == 1) {
//				global_work_offset = (size_t *) (kernelMsgBuffer + paramOffset);
//				paramOffset += work_dim * sizeof(size_t);
//			}
//
//			if (tmpEnqueueNDRangeKernel.global_work_size_flag == 1) {
//				global_work_size = (size_t *) (kernelMsgBuffer + paramOffset);
//				paramOffset += work_dim * sizeof(size_t);
//			}
//
//			if (tmpEnqueueNDRangeKernel.local_work_size_flag == 1) {
//				local_work_size = (size_t *) (kernelMsgBuffer + paramOffset);
//				paramOffset += work_dim * sizeof(size_t);
//			}
//
//			if (tmpEnqueueNDRangeKernel.args_num > 0) {
//				args_ptr = (kernel_args *) (kernelMsgBuffer + paramOffset);
//				paramOffset += (sizeof(kernel_args) * tmpEnqueueNDRangeKernel.args_num);
//			}
//
//			/* if migration is performed, check whether parameters */
//			/* should be updated to the new values in the virtual GPU */
//			voclProxyUpdateKernelArgs(tmpEnqueueNDRangeKernel.kernel, 
//									  tmpEnqueueNDRangeKernel.args_num,
//									  args_ptr);
//
//			/* store the kernel arguments */
//			voclProxyStoreKernelArgs(tmpEnqueueNDRangeKernel.kernel, 
//								   tmpEnqueueNDRangeKernel.args_num, 
//								   args_ptr);
//
//			//debug, for migration test------------------------------------------------
//			if (rankNo == 1 && voclProxyGetMigrationCondition(rankNo) == 0)
//			{
//				gettimeofday(&t1, NULL);
//				processAllWrites(appIndex);
//				gettimeofday(&t2, NULL);
//				tmpTime = 1000.0 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
//
//				FILE *pfile = fopen("waitTime.txt", "at");
//				fprintf(pfile, "proxyWaitTime\t%.3f\n", tmpTime);
//				fclose(pfile);
//
//				voclProxySetMigrationCondition(rankNo, 1);
//				cmdQueuePtr = voclProxyGetCmdQueuePtr(tmpEnqueueNDRangeKernel.command_queue);
//				voclProxyMigrationOneVGPUInKernelLaunch(appIndex, appRank, appComm[commIndex], 
//					         cmdQueuePtr->deviceID, &tmpEnqueueNDRangeKernel, kernelMsgBuffer);
//
//				/* if migration is performed, no need to launch the kernel in the current proxy */
//				if (num_events_in_wait_list > 0) {
//					free(event_wait_list);
//				}
//
//				/* issue it for later call of this function */
//				MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG,
//						  appComm[commIndex], conMsgRequest + index);
//
//				continue;
//			}
//			//end of debug----------------------------------------------------------
//
//			/* if there are data received, but not write to */
//			/* the GPU memory yet, use the helper thread to */
//			/* wait MPI receive complete and write to the GPU memory */
//			if (voclGetWriteEnqueueFlag(appIndex) == 0) {
//				pthread_barrier_wait(&barrier);
//				helperThreadOperFlag = GPU_ENQ_WRITE;
//				/* used by the helper thread */
//				voclProxyAppIndex = appIndex;
//				pthread_barrier_wait(&barrier);
//				pthread_barrier_wait(&barrier);
//			}
//
//			mpiOpenCLEnqueueNDRangeKernel(&tmpEnqueueNDRangeKernel,
//										  &kernelLaunchReply,
//										  event_wait_list,
//										  global_work_offset,
//										  global_work_size, local_work_size, args_ptr);
//
//			/* increase the number of kernels in the command queue by 1 */
//			voclProxyIncreaseKernelNumInCmdQueue(tmpEnqueueNDRangeKernel.command_queue, 1);
//
//			requestNo = 0;
//			if (tmpEnqueueNDRangeKernel.event_null_flag == 0)
//			{
//				MPI_Isend(&kernelLaunchReply, sizeof(struct strEnqueueNDRangeKernelReply),
//					  MPI_BYTE, appRank, ENQUEUE_ND_RANGE_KERNEL, appComm[commIndex],
//					  curRequest+(requestNo++));
//			}
//
//			if (num_events_in_wait_list > 0) {
//				free(event_wait_list);
//			}
//
//			if (requestNo > 0)
//			{
//				MPI_Waitall(requestNo, curRequest, curStatus);
//			}
        }

        else if (status.MPI_TAG == ENQUEUE_READ_BUFFER) {
			voclCmdQueuePtr = voclProxyGetInternalQueueTail();
			voclCmdQueuePtr->msgTag = ENQUEUE_READ_BUFFER;
			memcpy(voclCmdQueuePtr->conMsgBuffer, conMsgBuffer[index], sizeof(struct strEnqueueReadBuffer));
			voclCmdQueuePtr->appComm = appComm[commIndex];
			voclCmdQueuePtr->appCommData = appCommData[commIndex];
			voclCmdQueuePtr->appRank = appRank;
			voclCmdQueuePtr->appIndex = appIndex;
			voclProxyUnlockItem(voclCmdQueuePtr);

			//if (voclProxyGetMigrationCondition() == 1)
			//{
			//	readBuffer = (struct strEnqueueReadBuffer *)voclCmdQueuePtr->conMsgBuffer;
			//	cmdQueuePtr = voclProxyGetCmdQueuePtr(readBuffer->command_queue);
			//	voclProxyMigration(appIndex, cmdQueuePtr->deviceID);
			//	voclProxySetMigrationCondition(0);
			//}

//			memcpy(&tmpEnqueueReadBuffer, conMsgBuffer[index], sizeof(tmpEnqueueReadBuffer));
//			num_events_in_wait_list = tmpEnqueueReadBuffer.num_events_in_wait_list;
//			event_wait_list = NULL;
//			if (num_events_in_wait_list > 0) {
//				event_wait_list =
//					(cl_event *) malloc(num_events_in_wait_list * sizeof(cl_event));
//				MPI_Irecv(event_wait_list, num_events_in_wait_list * sizeof(cl_event),
//						  MPI_BYTE, appRank, ENQUEUE_READ_BUFFER1, appCommData[commIndex],
//						  curRequest);
//				MPI_Wait(curRequest, curStatus);
//			}
//
//			bufferSize = VOCL_PROXY_READ_BUFFER_SIZE;
//			bufferNum = (tmpEnqueueReadBuffer.cb - 1) / VOCL_PROXY_READ_BUFFER_SIZE;
//			remainingSize = tmpEnqueueReadBuffer.cb - bufferSize * bufferNum;
//			for (i = 0; i <= bufferNum; i++) {
//				bufferIndex = getNextReadBufferIndex(appIndex);
//				if (i == bufferNum)
//					bufferSize = remainingSize;
//				readBufferInfoPtr = getReadBufferInfoPtr(appIndex, bufferIndex);
//				readBufferInfoPtr->comm = appCommData[commIndex];
//				readBufferInfoPtr->tag = VOCL_PROXY_READ_TAG + bufferIndex;
//				//readBufferInfoPtr->tag = VOCL_PROXY_READ_TAG;
//				readBufferInfoPtr->dest = appRank;
//				readBufferInfoPtr->size = bufferSize;
//				tmpEnqueueReadBuffer.res =
//					clEnqueueReadBuffer(tmpEnqueueReadBuffer.command_queue,
//										tmpEnqueueReadBuffer.buffer,
//										CL_FALSE,
//										tmpEnqueueReadBuffer.offset +
//										i * VOCL_PROXY_READ_BUFFER_SIZE, bufferSize,
//										readBufferInfoPtr->dataPtr,
//										tmpEnqueueReadBuffer.num_events_in_wait_list,
//										event_wait_list, &readBufferInfoPtr->event);
//				setReadBufferFlag(appIndex, bufferIndex, READ_GPU_MEM);
//			}
//			readBufferInfoPtr->numReadBuffers = bufferNum + 1;
//
//			/* some new read requests are issued */
//			voclResetReadBufferCoveredFlag(appIndex);
//
//			if (tmpEnqueueReadBuffer.blocking_read == CL_FALSE) {
//				if (tmpEnqueueReadBuffer.event_null_flag == 0) {
//					tmpEnqueueReadBuffer.event = readBufferInfoPtr->event;
//					MPI_Isend(&tmpEnqueueReadBuffer, sizeof(tmpEnqueueReadBuffer), MPI_BYTE,
//							  appRank, ENQUEUE_READ_BUFFER, appComm[commIndex], curRequest);
//				}
//			}
//			else {      /* blocking, reading is complete, send data to local node */
//				tmpEnqueueReadBuffer.res = processAllReads(appIndex);
//				if (tmpEnqueueReadBuffer.event_null_flag == 0) {
//					tmpEnqueueReadBuffer.event = readBufferInfoPtr->event;
//				}
//				MPI_Isend(&tmpEnqueueReadBuffer, sizeof(tmpEnqueueReadBuffer), MPI_BYTE,
//						  appRank, ENQUEUE_READ_BUFFER, appComm[commIndex], curRequest);
//
//			}
//			MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == RELEASE_MEM_OBJ) {
            memcpy(&tmpReleaseMemObject, conMsgBuffer[index], sizeof(tmpReleaseMemObject));

			/*release the memory */
			memPtr = voclProxyGetMemPtr(tmpReleaseMemObject.memobj);
			voclProxyRemoveMemFromContext(memPtr);
			voclProxyReleaseMem(tmpReleaseMemObject.memobj);

            mpiOpenCLReleaseMemObject(&tmpReleaseMemObject);
            MPI_Isend(&tmpReleaseMemObject, sizeof(tmpReleaseMemObject), MPI_BYTE, appRank,
                      RELEASE_MEM_OBJ, appComm[commIndex], curRequest);

            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == CL_RELEASE_KERNEL_FUNC) {
            memcpy(&tmpReleaseKernel, conMsgBuffer[index], sizeof(tmpReleaseKernel));

			/* release kernel */
			kernelPtr = voclProxyGetKernelPtr(tmpReleaseKernel.kernel);
			voclProxyRemoveKernelFromProgram(kernelPtr);
			voclProxyReleaseKernel(tmpReleaseKernel.kernel);

            mpiOpenCLReleaseKernel(&tmpReleaseKernel);
            MPI_Isend(&tmpReleaseKernel, sizeof(tmpReleaseKernel), MPI_BYTE, appRank,
                      CL_RELEASE_KERNEL_FUNC, appComm[commIndex], curRequest);

            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == FINISH_FUNC) {
			voclCmdQueuePtr = voclProxyGetInternalQueueTail();
			voclCmdQueuePtr->msgTag = FINISH_FUNC;
			memcpy(voclCmdQueuePtr->conMsgBuffer, conMsgBuffer[index], sizeof(struct strFinish));
			voclCmdQueuePtr->appComm = appComm[commIndex];
			voclCmdQueuePtr->appCommData = appCommData[commIndex];
			voclCmdQueuePtr->appRank = appRank;
			voclCmdQueuePtr->appIndex = appIndex;
			voclProxyUnlockItem(voclCmdQueuePtr);

			//if (voclProxyGetMigrationCondition() == 1)
			//{
			//	finishPtr = (struct strFinish *)voclCmdQueuePtr->conMsgBuffer;
			//	cmdQueuePtr = voclProxyGetCmdQueuePtr(finishPtr->command_queue);
			//	voclProxyMigration(appIndex, cmdQueuePtr->deviceID);
			//	voclProxySetMigrationCondition(0);
			//}

//			memcpy(&tmpFinish, conMsgBuffer[index], sizeof(tmpFinish));
//			processAllWrites(appIndex);
//			processAllReads(appIndex);
//			mpiOpenCLFinish(&tmpFinish);

//			/* all kernels complete their computation */
//			voclProxyResetKernelNumInCmdQueue(tmpFinish.command_queue);

//			MPI_Isend(&tmpFinish, sizeof(tmpFinish), MPI_BYTE, appRank,
//					  FINISH_FUNC, appComm[commIndex], curRequest);
//			MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == GET_CONTEXT_INFO_FUNC) {
            memcpy(&tmpGetContextInfo, conMsgBuffer[index], sizeof(tmpGetContextInfo));
            param_value_size = tmpGetContextInfo.param_value_size;
            param_value = NULL;
            if (param_value_size > 0 && tmpGetContextInfo.param_value != NULL) {
                param_value = malloc(param_value_size);
            }
            mpiOpenCLGetContextInfo(&tmpGetContextInfo, param_value);
            requestNo = 0;
            MPI_Isend(&tmpGetContextInfo, sizeof(tmpGetContextInfo), MPI_BYTE, appRank,
                      GET_CONTEXT_INFO_FUNC, appComm[commIndex], curRequest + (requestNo++));

            if (param_value_size > 0 && tmpGetContextInfo.param_value != NULL) {
                MPI_Isend(param_value, param_value_size, MPI_BYTE, appRank,
                          GET_CONTEXT_INFO_FUNC1, appCommData[commIndex],
                          curRequest + (requestNo++));
            }

            MPI_Waitall(requestNo, curRequest, curStatus);
            if (param_value_size > 0 && tmpGetContextInfo.param_value != NULL) {
                free(param_value);
            }
        }

        else if (status.MPI_TAG == GET_BUILD_INFO_FUNC) {
            memcpy(&tmpGetProgramBuildInfo, conMsgBuffer[index],
                   sizeof(tmpGetProgramBuildInfo));
            param_value_size = tmpGetProgramBuildInfo.param_value_size;
            param_value = NULL;
            if (param_value_size > 0 && tmpGetProgramBuildInfo.param_value != NULL) {
                param_value = malloc(param_value_size);
            }
            mpiOpenCLGetProgramBuildInfo(&tmpGetProgramBuildInfo, param_value);
            requestNo = 0;
            MPI_Isend(&tmpGetProgramBuildInfo, sizeof(tmpGetProgramBuildInfo), MPI_BYTE,
                      appRank, GET_BUILD_INFO_FUNC, appComm[commIndex],
                      curRequest + (requestNo++));

            if (param_value_size > 0 && tmpGetProgramBuildInfo.param_value != NULL) {
                MPI_Isend(param_value, param_value_size, MPI_BYTE, appRank,
                          GET_BUILD_INFO_FUNC1, appCommData[commIndex],
                          curRequest + (requestNo++));
            }

            MPI_Waitall(requestNo, curRequest, curStatus);

            if (param_value_size > 0 && tmpGetProgramBuildInfo.param_value != NULL) {
                free(param_value);
            }
        }

        else if (status.MPI_TAG == GET_PROGRAM_INFO_FUNC) {
            memcpy(&tmpGetProgramInfo, conMsgBuffer[index], sizeof(tmpGetProgramInfo));
            param_value_size = tmpGetProgramInfo.param_value_size;
            param_value = NULL;
            if (param_value_size > 0 && tmpGetProgramInfo.param_value != NULL) {
                param_value = malloc(param_value_size);
            }
            mpiOpenCLGetProgramInfo(&tmpGetProgramInfo, param_value);
            requestNo = 0;
            MPI_Isend(&tmpGetProgramInfo, sizeof(tmpGetProgramInfo), MPI_BYTE, appRank,
                      GET_PROGRAM_INFO_FUNC, appComm[commIndex], curRequest + (requestNo++));

            if (param_value_size > 0 && tmpGetProgramInfo.param_value != NULL) {
                MPI_Isend(param_value, param_value_size, MPI_BYTE, appRank,
                          GET_PROGRAM_INFO_FUNC1, appCommData[commIndex],
                          curRequest + (requestNo++));
            }

            MPI_Waitall(requestNo, curRequest, curStatus);
            if (param_value_size > 0 && tmpGetProgramInfo.param_value != NULL) {
                free(param_value);
            }
        }

        else if (status.MPI_TAG == REL_PROGRAM_FUNC) {
            memcpy(&tmpReleaseProgram, conMsgBuffer[index], sizeof(tmpReleaseProgram));

			/* release program */
			programPtr = voclProxyGetProgramPtr(tmpReleaseProgram.program);
			voclProxyRemoveProgramFromContext(programPtr);
			voclProxyReleaseProgram(tmpReleaseProgram.program);

            mpiOpenCLReleaseProgram(&tmpReleaseProgram);
            MPI_Isend(&tmpReleaseProgram, sizeof(tmpReleaseProgram), MPI_BYTE, appRank,
                      REL_PROGRAM_FUNC, appComm[commIndex], curRequest);
            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == REL_COMMAND_QUEUE_FUNC) {
            memcpy(&tmpReleaseCommandQueue, conMsgBuffer[index],
                   sizeof(tmpReleaseCommandQueue));

			/* release command queue */
			cmdQueuePtr = voclProxyGetCmdQueuePtr(tmpReleaseCommandQueue.command_queue);
			voclProxyRemoveCommandQueueFromContext(cmdQueuePtr);
			voclProxyReleaseCommandQueue(tmpReleaseCommandQueue.command_queue);
			voclProxyRemoveCommandQueueFromVGPU(appIndex, cmdQueuePtr);

            mpiOpenCLReleaseCommandQueue(&tmpReleaseCommandQueue);
            MPI_Isend(&tmpReleaseCommandQueue, sizeof(tmpReleaseCommandQueue), MPI_BYTE,
                      appRank, REL_COMMAND_QUEUE_FUNC, appComm[commIndex], curRequest);
            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == REL_CONTEXT_FUNC) {
            memcpy(&tmpReleaseContext, conMsgBuffer[index], sizeof(tmpReleaseContext));

			/* release context */
			contextPtr = voclProxyGetContextPtr(tmpReleaseContext.context);
			voclProxyRemoveContextFromVGPU(appIndex, contextPtr);

			/*release the virtual GPU */
			for (i = 0; i < contextPtr->deviceNum; i++)
			{
				voclProxyReleaseVirtualGPU(appIndex, contextPtr->devices[i]);
			}
			voclProxyReleaseContext(tmpReleaseContext.context);

            mpiOpenCLReleaseContext(&tmpReleaseContext);
            MPI_Isend(&tmpReleaseContext, sizeof(tmpReleaseContext), MPI_BYTE, appRank,
                      REL_CONTEXT_FUNC, appComm[commIndex], curRequest);
            MPI_Wait(curRequest, curStatus);

        }

        else if (status.MPI_TAG == GET_DEVICE_INFO_FUNC) {
            memcpy(&tmpGetDeviceInfo, conMsgBuffer[index], sizeof(tmpGetDeviceInfo));
            param_value_size = tmpGetDeviceInfo.param_value_size;
            param_value = NULL;
            if (param_value_size > 0 && tmpGetDeviceInfo.param_value != NULL) {
                param_value = malloc(param_value_size);
            }
            mpiOpenCLGetDeviceInfo(&tmpGetDeviceInfo, param_value);
            requestNo = 0;
            MPI_Isend(&tmpGetDeviceInfo, sizeof(tmpGetDeviceInfo), MPI_BYTE, appRank,
                      GET_DEVICE_INFO_FUNC, appComm[commIndex], curRequest + (requestNo++));
            if (param_value_size > 0 && tmpGetDeviceInfo.param_value != NULL) {
                MPI_Isend(param_value, param_value_size, MPI_BYTE, appRank,
                          GET_DEVICE_INFO_FUNC1, appCommData[commIndex],
                          curRequest + (requestNo++));
            }

            MPI_Waitall(requestNo, curRequest, curStatus);
            if (param_value_size > 0 && tmpGetDeviceInfo.param_value != NULL) {
                free(param_value);
            }
        }

        else if (status.MPI_TAG == GET_PLATFORM_INFO_FUNC) {
            memcpy(&tmpGetPlatformInfo, conMsgBuffer[index], sizeof(tmpGetPlatformInfo));
            param_value_size = tmpGetPlatformInfo.param_value_size;
            param_value = NULL;
            if (param_value_size > 0 && tmpGetPlatformInfo.param_value != NULL) {
                param_value = malloc(param_value_size);
            }
            mpiOpenCLGetPlatformInfo(&tmpGetPlatformInfo, param_value);
            requestNo = 0;
            MPI_Isend(&tmpGetPlatformInfo, sizeof(tmpGetPlatformInfo), MPI_BYTE, appRank,
                      GET_PLATFORM_INFO_FUNC, appComm[commIndex], curRequest + (requestNo++));

            if (param_value_size > 0 && tmpGetPlatformInfo.param_value != NULL) {
                MPI_Isend(param_value, param_value_size, MPI_BYTE, appRank,
                          GET_PLATFORM_INFO_FUNC1, appCommData[commIndex],
                          curRequest + (requestNo++));
            }

            MPI_Waitall(requestNo, curRequest, curStatus);
            if (param_value_size > 0 && tmpGetPlatformInfo.param_value != NULL) {
                free(param_value);
            }
        }

        else if (status.MPI_TAG == FLUSH_FUNC) {
			voclCmdQueuePtr = voclProxyGetInternalQueueTail();
			voclCmdQueuePtr->msgTag = FLUSH_FUNC;
			memcpy(voclCmdQueuePtr->conMsgBuffer, conMsgBuffer[index], sizeof(struct strFlush));
			voclCmdQueuePtr->appComm = appComm[commIndex];
			voclCmdQueuePtr->appCommData = appCommData[commIndex];
			voclCmdQueuePtr->appRank = appRank;
			voclCmdQueuePtr->appIndex = appIndex;
			voclProxyUnlockItem(voclCmdQueuePtr);

			//if (voclProxyGetMigrationCondition() == 1)
			//{
			//	flushPtr = (struct strFlush *)voclCmdQueuePtr->conMsgBuffer;
			//	cmdQueuePtr = voclProxyGetCmdQueuePtr(flushPtr->command_queue);
			//	voclProxyMigration(appIndex, cmdQueuePtr->deviceID);
			//	voclProxySetMigrationCondition(0);
			//}
			//memcpy(&tmpFlush, conMsgBuffer[index], sizeof(tmpFlush));
			//mpiOpenCLFlush(&tmpFlush);
			//MPI_Isend(&tmpFlush, sizeof(tmpFlush), MPI_BYTE, appRank,
			//		  FLUSH_FUNC, appComm[commIndex], curRequest);
			//MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == WAIT_FOR_EVENT_FUNC) {
            requestNo = 0;
            memcpy(&tmpWaitForEvents, conMsgBuffer[index], sizeof(tmpWaitForEvents));
            num_events = tmpWaitForEvents.num_events;
            event_list = (cl_event *) malloc(sizeof(cl_event) * num_events);
            MPI_Irecv(event_list, sizeof(cl_event) * num_events, MPI_BYTE, appRank,
                      WAIT_FOR_EVENT_FUNC1, appCommData[commIndex],
                      curRequest + (requestNo++));
            MPI_Waitall(requestNo, curRequest, curStatus);
            requestNo = 0;
            mpiOpenCLWaitForEvents(&tmpWaitForEvents, event_list);

            for (i = 0; i < num_events; i++) {
                bufferIndex = getReadBufferIndexFromEvent(appIndex, event_list[i]);
                if (bufferIndex >= 0) {
                    readBufferInfoPtr = getReadBufferInfoPtr(appIndex, bufferIndex);
                    processReadBuffer(appIndex, bufferIndex,
                                      readBufferInfoPtr->numReadBuffers);
                    readBufferInfoPtr->numReadBuffers = 0;
                }

                bufferIndex = getWriteBufferIndexFromEvent(appIndex, event_list[i]);
                if (bufferIndex >= 0) {
                    writeBufferInfoPtr = getWriteBufferInfoPtr(appIndex, bufferIndex);
                    processWriteBuffer(appIndex, bufferIndex,
                                       writeBufferInfoPtr->numWriteBuffers);
                    writeBufferInfoPtr->numWriteBuffers = 0;
                }
            }

            MPI_Isend(&tmpWaitForEvents, sizeof(tmpWaitForEvents), MPI_BYTE, 0,
                      WAIT_FOR_EVENT_FUNC, appComm[commIndex], curRequest + (requestNo++));
            MPI_Waitall(requestNo, curRequest, curStatus);
            free(event_list);
        }

        else if (status.MPI_TAG == CREATE_SAMPLER_FUNC) {
            memcpy(&tmpCreateSampler, conMsgBuffer[index], sizeof(tmpCreateSampler));
            mpiOpenCLCreateSampler(&tmpCreateSampler);
            MPI_Isend(&tmpCreateSampler, sizeof(tmpCreateSampler), MPI_BYTE, appRank,
                      CREATE_SAMPLER_FUNC, appComm[commIndex], curRequest);
            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == GET_CMD_QUEUE_INFO_FUNC) {
            requestNo = 0;
            memcpy(&tmpGetCommandQueueInfo, conMsgBuffer[index],
                   sizeof(tmpGetCommandQueueInfo));
            param_value_size = tmpGetCommandQueueInfo.param_value_size;
            param_value = NULL;
            if (param_value_size > 0 && tmpGetCommandQueueInfo.param_value != NULL) {
                param_value = malloc(param_value_size);
            }
            mpiOpenCLGetCommandQueueInfo(&tmpGetCommandQueueInfo, param_value);
            MPI_Isend(&tmpGetCommandQueueInfo, sizeof(tmpGetCommandQueueInfo), MPI_BYTE,
                      appRank, GET_CMD_QUEUE_INFO_FUNC, appComm[commIndex],
                      curRequest + (requestNo++));

            if (param_value_size > 0 && tmpGetCommandQueueInfo.param_value != NULL) {
                MPI_Isend(param_value, param_value_size, MPI_BYTE, appRank,
                          GET_CMD_QUEUE_INFO_FUNC1, appCommData[commIndex],
                          curRequest + (requestNo++));
            }

            MPI_Waitall(requestNo, curRequest, curStatus);
            if (param_value_size > 0 && tmpGetCommandQueueInfo.param_value != NULL) {
                free(param_value);
            }
        }

        else if (status.MPI_TAG == ENQUEUE_MAP_BUFF_FUNC) {
            requestNo = 0;

            memcpy(&tmpEnqueueMapBuffer, conMsgBuffer[index], sizeof(tmpEnqueueMapBuffer));
            num_events_in_wait_list = tmpEnqueueMapBuffer.num_events_in_wait_list;
            event_wait_list = NULL;
            if (num_events_in_wait_list > 0) {
                event_wait_list =
                    (cl_event *) malloc(sizeof(cl_event) * num_events_in_wait_list);
                MPI_Irecv(event_wait_list, sizeof(cl_event) * num_events_in_wait_list,
                          MPI_BYTE, appRank, ENQUEUE_MAP_BUFF_FUNC1, appCommData[commIndex],
                          curRequest + (requestNo++));
            }
            mpiOpenCLEnqueueMapBuffer(&tmpEnqueueMapBuffer, event_wait_list);
            MPI_Isend(&tmpEnqueueMapBuffer, sizeof(tmpEnqueueMapBuffer), MPI_BYTE, appRank,
                      ENQUEUE_MAP_BUFF_FUNC, appComm[commIndex], curRequest + (requestNo++));

            if (num_events_in_wait_list > 0) {
                free(event_wait_list);
            }
            MPI_Waitall(requestNo, curRequest, curStatus);
        }

        else if (status.MPI_TAG == RELEASE_EVENT_FUNC) {
            memcpy(&tmpReleaseEvent, conMsgBuffer[index], sizeof(tmpReleaseEvent));
            mpiOpenCLReleaseEvent(&tmpReleaseEvent);
            MPI_Isend(&tmpReleaseEvent, sizeof(tmpReleaseEvent), MPI_BYTE, appRank,
                      RELEASE_EVENT_FUNC, appComm[commIndex], curRequest);

            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == GET_EVENT_PROF_INFO_FUNC) {
            requestNo = 0;
            memcpy(&tmpGetEventProfilingInfo, conMsgBuffer[index],
                   sizeof(tmpGetEventProfilingInfo));
            param_value_size = tmpGetEventProfilingInfo.param_value_size;
            param_value = NULL;
            if (param_value_size > 0 && tmpGetEventProfilingInfo.param_value != NULL) {
                param_value = malloc(param_value_size);
            }
            mpiOpenCLGetEventProfilingInfo(&tmpGetEventProfilingInfo, param_value);
            MPI_Isend(&tmpGetEventProfilingInfo, sizeof(tmpGetEventProfilingInfo), MPI_BYTE,
                      appRank, GET_EVENT_PROF_INFO_FUNC, appComm[commIndex],
                      curRequest + (requestNo++));

            if (param_value_size > 0 && tmpGetEventProfilingInfo.param_value != NULL) {
                MPI_Isend(param_value, param_value_size, MPI_BYTE, appRank,
                          GET_EVENT_PROF_INFO_FUNC1, appCommData[commIndex],
                          curRequest + (requestNo++));
            }

            MPI_Waitall(requestNo, curRequest, curStatus);
            if (param_value_size > 0 && tmpGetEventProfilingInfo.param_value != NULL) {
                free(param_value);
            }
        }

        else if (status.MPI_TAG == RELEASE_SAMPLER_FUNC) {
            memcpy(&tmpReleaseSampler, conMsgBuffer[index], sizeof(tmpReleaseSampler));
            mpiOpenCLReleaseSampler(&tmpReleaseSampler);
            MPI_Isend(&tmpReleaseSampler, sizeof(tmpReleaseSampler), MPI_BYTE, appRank,
                      RELEASE_SAMPLER_FUNC, appComm[commIndex], curRequest);

            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == GET_KERNEL_WGP_INFO_FUNC) {
            requestNo = 0;
            memcpy(&tmpGetKernelWorkGroupInfo, conMsgBuffer[index],
                   sizeof(tmpGetKernelWorkGroupInfo));
            param_value_size = tmpGetKernelWorkGroupInfo.param_value_size;
            param_value = NULL;
            if (param_value_size > 0 && tmpGetKernelWorkGroupInfo.param_value != NULL) {
                param_value = malloc(param_value_size);
            }
            mpiOpenCLGetKernelWorkGroupInfo(&tmpGetKernelWorkGroupInfo, param_value);
            MPI_Isend(&tmpGetKernelWorkGroupInfo, sizeof(tmpGetKernelWorkGroupInfo), MPI_BYTE,
                      appRank, GET_KERNEL_WGP_INFO_FUNC, appComm[commIndex],
                      curRequest + (requestNo++));

            if (param_value_size > 0 && tmpGetKernelWorkGroupInfo.param_value != NULL) {
                MPI_Isend(param_value, param_value_size, MPI_BYTE, appRank,
                          GET_KERNEL_WGP_INFO_FUNC1, appCommData[commIndex],
                          curRequest + (requestNo++));
            }

            MPI_Waitall(requestNo, curRequest, curStatus);
            if (param_value_size > 0 && tmpGetKernelWorkGroupInfo.param_value != NULL) {
                free(param_value);
            }
        }

        else if (status.MPI_TAG == CREATE_IMAGE_2D_FUNC) {
            requestNo = 0;
            memcpy(&tmpCreateImage2D, conMsgBuffer[index], sizeof(tmpCreateImage2D));
            host_buff_size = tmpCreateImage2D.host_buff_size;
            host_ptr = NULL;
            if (host_buff_size > 0) {
                host_ptr = malloc(host_buff_size);
                MPI_Irecv(host_ptr, host_buff_size, MPI_BYTE, appRank,
                          CREATE_IMAGE_2D_FUNC1, appCommData[commIndex],
                          curRequest + (requestNo++));
            }
            mpiOpenCLCreateImage2D(&tmpCreateImage2D, host_ptr);
            MPI_Isend(&tmpCreateImage2D, sizeof(tmpCreateImage2D), MPI_BYTE, appRank,
                      CREATE_IMAGE_2D_FUNC, appComm[commIndex], curRequest + (requestNo++));

            MPI_Waitall(requestNo, curRequest, curStatus);
            if (host_buff_size > 0) {
                free(host_ptr);
            }
        }

        else if (status.MPI_TAG == ENQ_COPY_BUFF_FUNC) {
            requestNo = 0;
            memcpy(&tmpEnqueueCopyBuffer, conMsgBuffer[index], sizeof(tmpEnqueueCopyBuffer));
            num_events_in_wait_list = tmpEnqueueCopyBuffer.num_events_in_wait_list;
            event_wait_list = NULL;
            if (num_events_in_wait_list > 0) {
                event_wait_list =
                    (cl_event *) malloc(sizeof(cl_event) * num_events_in_wait_list);
                MPI_Irecv(event_wait_list, sizeof(cl_event) * num_events_in_wait_list,
                          MPI_BYTE, appRank, ENQ_COPY_BUFF_FUNC1, appCommData[commIndex],
                          curRequest + (requestNo++));
            }
            mpiOpenCLEnqueueCopyBuffer(&tmpEnqueueCopyBuffer, event_wait_list);
            MPI_Isend(&tmpEnqueueCopyBuffer, sizeof(tmpEnqueueCopyBuffer), MPI_BYTE, appRank,
                      ENQ_COPY_BUFF_FUNC, appComm[commIndex], curRequest + (requestNo++));

            MPI_Waitall(requestNo, curRequest, curStatus);
            if (num_events_in_wait_list > 0) {
                free(event_wait_list);
            }
        }

        else if (status.MPI_TAG == RETAIN_EVENT_FUNC) {
            memcpy(&tmpRetainEvent, conMsgBuffer[index], sizeof(tmpRetainEvent));
            mpiOpenCLRetainEvent(&tmpRetainEvent);
            MPI_Isend(&tmpRetainEvent, sizeof(tmpRetainEvent), MPI_BYTE, appRank,
                      RETAIN_EVENT_FUNC, appComm[commIndex], curRequest);

            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == RETAIN_MEMOBJ_FUNC) {
            memcpy(&tmpRetainMemObject, conMsgBuffer[index], sizeof(tmpRetainMemObject));
            mpiOpenCLRetainMemObject(&tmpRetainMemObject);
            MPI_Isend(&tmpRetainMemObject, sizeof(tmpRetainMemObject), MPI_BYTE, appRank,
                      RETAIN_MEMOBJ_FUNC, appComm[commIndex], curRequest);

            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == RETAIN_KERNEL_FUNC) {
            requestNo = 0;
            MPI_Irecv(&tmpRetainKernel, sizeof(tmpRetainKernel), MPI_BYTE, appRank,
                      RETAIN_KERNEL_FUNC, appComm[commIndex], curRequest + (requestNo++));
            mpiOpenCLRetainKernel(&tmpRetainKernel);
            MPI_Isend(&tmpRetainKernel, sizeof(tmpRetainKernel), MPI_BYTE, appRank,
                      RETAIN_KERNEL_FUNC, appComm[commIndex], curRequest + (requestNo++));

            MPI_Waitall(requestNo, curRequest, curStatus);
        }

        else if (status.MPI_TAG == RETAIN_CMDQUE_FUNC) {
            memcpy(&tmpRetainCommandQueue, conMsgBuffer[index], sizeof(tmpRetainCommandQueue));
            mpiOpenCLRetainCommandQueue(&tmpRetainCommandQueue);
            MPI_Isend(&tmpRetainCommandQueue, sizeof(tmpRetainCommandQueue), MPI_BYTE, appRank,
                      RETAIN_CMDQUE_FUNC, appComm[commIndex], curRequest);

            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == ENQ_UNMAP_MEMOBJ_FUNC) {
            requestNo = 0;
            memcpy(&tmpEnqueueUnmapMemObject, conMsgBuffer[index],
                   sizeof(tmpEnqueueUnmapMemObject));
            num_events_in_wait_list = tmpEnqueueUnmapMemObject.num_events_in_wait_list;
            event_wait_list = NULL;
            if (num_events_in_wait_list > 0) {
                event_wait_list =
                    (cl_event *) malloc(sizeof(cl_event) * num_events_in_wait_list);
                MPI_Irecv(event_wait_list, sizeof(cl_event) * num_events_in_wait_list,
                          MPI_BYTE, appRank, ENQ_UNMAP_MEMOBJ_FUNC1, appCommData[commIndex],
                          curRequest + (requestNo++));
            }
            mpiOpenCLEnqueueUnmapMemObject(&tmpEnqueueUnmapMemObject, event_wait_list);
            MPI_Isend(&tmpEnqueueUnmapMemObject, sizeof(tmpEnqueueUnmapMemObject), MPI_BYTE,
                      appRank, ENQ_UNMAP_MEMOBJ_FUNC, appComm[commIndex],
                      curRequest + (requestNo++));

            MPI_Waitall(requestNo, curRequest, curStatus);
            if (num_events_in_wait_list > 0) {
                free(event_wait_list);
            }
        }

        else if (status.MPI_TAG == MIG_GET_PROXY_RANK) {
            /* send rank of the proxy process to the app process */
            MPI_Comm_rank(MPI_COMM_WORLD, &proxyRank);
            MPI_Send(&proxyRank, 1, MPI_INT, appRank, MIG_GET_PROXY_RANK, appComm[commIndex]);
        }

        else if (status.MPI_TAG == MIG_MEM_WRITE_REQUEST) {
            memcpy(&tmpMigGPUMemoryWrite, conMsgBuffer[index],
                   sizeof(struct strMigGPUMemoryWrite));
            /* issue irecv to receive data from source process */
            bufferSize = VOCL_MIG_BUF_SIZE;
            bufferNum = (tmpMigGPUMemoryWrite.size - 1) / VOCL_MIG_BUF_SIZE;
            remainingSize = tmpMigGPUMemoryWrite.size - bufferNum * VOCL_MIG_BUF_SIZE;
            for (i = 0; i <= bufferNum; i++) {
                if (i == bufferNum) {
                    bufferSize = remainingSize;
                }
                bufferIndex = voclMigGetNextWriteBufferIndex(appIndex);
                migWriteBufferInfoPtr = voclMigGetWriteBufferPtr(appIndex, bufferIndex);
                /* if the data is from the local node */
                if (tmpMigGPUMemoryWrite.isFromLocal == 1) {
                    err = MPI_Irecv(migWriteBufferInfoPtr->ptr, bufferSize, MPI_BYTE,
                                    appRank, VOCL_PROXY_MIG_TAG,
                                    appCommData[commIndex], voclMigGetWriteRequestPtr(appIndex,
                                                                                      bufferIndex));
                }
                else {
                    err = MPI_Irecv(migWriteBufferInfoPtr->ptr, bufferSize, MPI_BYTE,
                                    tmpMigGPUMemoryWrite.source, VOCL_PROXY_MIG_TAG,
                                    MPI_COMM_WORLD, voclMigGetWriteRequestPtr(appIndex,
                                                                              bufferIndex));
                }
                migWriteBufferInfoPtr->cmdQueue = tmpMigGPUMemoryWrite.cmdQueue;
                migWriteBufferInfoPtr->memory = tmpMigGPUMemoryWrite.memory;
                migWriteBufferInfoPtr->source = tmpMigGPUMemoryWrite.source;
                migWriteBufferInfoPtr->isFromLocal = tmpMigGPUMemoryWrite.isFromLocal;
                migWriteBufferInfoPtr->comm = tmpMigGPUMemoryWrite.comm;
                migWriteBufferInfoPtr->size = bufferSize;
                migWriteBufferInfoPtr->offset = i * VOCL_MIG_BUF_SIZE;
                voclMigSetWriteBufferFlag(appIndex, bufferIndex, MIG_WRT_MPIRECV);
            }

            tmpMigGPUMemoryWrite.res = err;
            MPI_Send(&tmpMigGPUMemoryWrite, sizeof(struct strMigGPUMemoryWrite), MPI_BYTE,
                     appRank, MIG_MEM_WRITE_REQUEST, appComm[commIndex]);
        }

        else if (status.MPI_TAG == MIG_MEM_READ_REQUEST) {
            memcpy(&tmpMigGPUMemoryRead, conMsgBuffer[index],
                   sizeof(struct strMigGPUMemoryRead));
            /* issue GPU memory read to read data from GPU memory */
            bufferSize = VOCL_MIG_BUF_SIZE;
            bufferNum = (tmpMigGPUMemoryRead.size - 1) / VOCL_MIG_BUF_SIZE;
            remainingSize = tmpMigGPUMemoryRead.size - bufferNum * VOCL_MIG_BUF_SIZE;
            for (i = 0; i <= bufferNum; i++) {
                if (i == bufferNum) {
                    bufferSize = remainingSize;
                }
                bufferIndex = voclMigGetNextReadBufferIndex(appIndex);
                migReadBufferInfoPtr = voclMigGetReadBufferPtr(appIndex, bufferIndex);

                err = clEnqueueReadBuffer(tmpMigGPUMemoryRead.cmdQueue,
                                          tmpMigGPUMemoryRead.memory,
                                          CL_FALSE,
                                          i * VOCL_MIG_BUF_SIZE,
                                          bufferSize,
                                          migReadBufferInfoPtr->ptr,
                                          0, NULL, voclMigGetReadEventPtr(appIndex,
                                                                          bufferIndex));
                migReadBufferInfoPtr->size = bufferSize;
                migReadBufferInfoPtr->offset = i * VOCL_MIG_BUF_SIZE;
                migReadBufferInfoPtr->isToLocal = tmpMigGPUMemoryRead.isToLocal;
                migReadBufferInfoPtr->comm = tmpMigGPUMemoryRead.comm;
                if (tmpMigGPUMemoryRead.isToLocal == 1) {
                    migReadBufferInfoPtr->commData = appCommData[commIndex];
                    migReadBufferInfoPtr->dest = appRank;
                }
                else {
                    migReadBufferInfoPtr->commData = MPI_COMM_WORLD;
                    migReadBufferInfoPtr->dest = tmpMigGPUMemoryRead.dest;
                }
                migReadBufferInfoPtr->tag = VOCL_PROXY_MIG_TAG;
                voclMigSetReadBufferFlag(appIndex, bufferIndex, MIG_READ_RDGPU);
            }

            tmpMigGPUMemoryRead.res = err;
            MPI_Send(&tmpMigGPUMemoryRead, sizeof(struct strMigGPUMemoryRead), MPI_BYTE,
                     appRank, MIG_MEM_READ_REQUEST, appComm[commIndex]);
        }

        else if (status.MPI_TAG == MIG_SAME_REMOTE_NODE) {
            memcpy(&tmpMigGPUMemRW, conMsgBuffer[index],
                   sizeof(struct strMigRemoteGPUMemoryRW));
            bufferSize = VOCL_MIG_BUF_SIZE;
            bufferNum = (tmpMigGPUMemRW.size - 1) / VOCL_MIG_BUF_SIZE;
            remainingSize = tmpMigGPUMemRW.size - bufferNum * VOCL_MIG_BUF_SIZE;
            for (i = 0; i <= bufferNum; i++) {
                if (i == bufferNum) {
                    bufferSize = remainingSize;
                }
                bufferIndex = voclMigRWGetNextBufferIndex(appIndex);
                migRWBufferInfoPtr = voclMigRWGetBufferInfoPtr(appIndex, bufferIndex);
                err = clEnqueueReadBuffer(tmpMigGPUMemRW.oldCmdQueue,
                                          tmpMigGPUMemRW.oldMem,
                                          CL_FALSE,
                                          i * VOCL_MIG_BUF_SIZE,
                                          bufferSize,
                                          migRWBufferInfoPtr->ptr,
                                          0, NULL, &migRWBufferInfoPtr->rdEvent);
                migRWBufferInfoPtr->wtCmdQueue = tmpMigGPUMemRW.newCmdQueue;
                migRWBufferInfoPtr->wtMem = tmpMigGPUMemRW.newMem;
                migRWBufferInfoPtr->size = bufferSize;
                migRWBufferInfoPtr->offset = i * VOCL_MIG_BUF_SIZE;
                voclMigSetRWBufferFlag(appIndex, bufferIndex, MIG_RW_SAME_NODE_RDMEM);
            }
            tmpMigGPUMemRW.res = err;
            MPI_Send(&tmpMigGPUMemRW, sizeof(struct strMigRemoteGPUMemoryRW), MPI_BYTE,
                     appRank, MIG_SAME_REMOTE_NODE, appComm[commIndex]);
        }

        else if (status.MPI_TAG == MIG_SAME_REMOTE_NODE_CMPLD) {
            memcpy(&tmpMigGPUMemRWCmpd, conMsgBuffer[index],
                   sizeof(struct strMigRemoteGPURWCmpd));
            tmpMigGPUMemRWCmpd.res = voclMigFinishDataRWOnSameNode(appIndex);
            MPI_Send(&tmpMigGPUMemRWCmpd, sizeof(struct strMigRemoteGPURWCmpd), MPI_BYTE,
                     appRank, MIG_SAME_REMOTE_NODE_CMPLD, appComm[commIndex]);
        }

        else if (status.MPI_TAG == MIG_MEM_WRITE_CMPLD) {
            memcpy(&tmpMigWriteMemCmpdRst, conMsgBuffer[index],
                   sizeof(struct strMigGPUMemoryWriteCmpd));
            tmpMigWriteMemCmpdRst.retCode = voclMigFinishDataWrite(appIndex);
            MPI_Send(&tmpMigWriteMemCmpdRst, sizeof(struct strMigGPUMemoryWriteCmpd), MPI_BYTE,
                     appRank, MIG_MEM_WRITE_CMPLD, appComm[commIndex]);
        }

        else if (status.MPI_TAG == MIG_MEM_READ_CMPLD) {
            memcpy(&tmpMigReadMemCmpdRst, conMsgBuffer[index],
                   sizeof(struct strMigGPUMemoryReadCmpd));
            tmpMigReadMemCmpdRst.retCode = voclMigFinishDataRead(appIndex);
            MPI_Send(&tmpMigReadMemCmpdRst, sizeof(struct strMigGPUMemoryReadCmpd), MPI_BYTE,
                     appRank, MIG_MEM_READ_CMPLD, appComm[commIndex]);
        }

		else if (status.MPI_TAG == VOCL_UPDATE_VGPU)
		{
			memcpy(&tmpMigUpdateVGPU, conMsgBuffer[index], sizeof(struct strMigUpdateVGPU));
			migMsgBuffer = (char *)malloc(tmpMigUpdateVGPU.msgSize);
			MPI_Irecv(migMsgBuffer, tmpMigUpdateVGPU.msgSize, MPI_BYTE, appRank,
					  VOCL_UPDATE_VGPU, appCommData[commIndex], curRequest);
			MPI_Wait(curRequest, curStatus);
			vocl_proxyUpdateVirtualGPUInfo(appIndex, migMsgBuffer);
			MPI_Isend(migMsgBuffer, tmpMigUpdateVGPU.msgSize, MPI_BYTE, appRank,
					  VOCL_UPDATE_VGPU, appCommData[commIndex], curRequest);
			MPI_Wait(curRequest, curStatus);
			free(migMsgBuffer);
		}

		else if (status.MPI_TAG == VOCL_REBALANCE)
		{
			memcpy(&tmpVoclRebalance, conMsgBuffer[index], sizeof(struct strVoclRebalance));
			cmdQueuePtr = voclProxyGetCmdQueuePtr(tmpVoclRebalance.command_queue);

			/* migration has been performed for the current command queue */
			/* no need to migrate once more */
			if (cmdQueuePtr->migStatus > 0)
			{
				tmpVoclRebalance.isMigratedNecessary = 0;
			}
			else
			{
				/* check whether migration is needed */
				tmpVoclRebalance.isMigratedNecessary = voclProxyCheckIsMigrationNeeded(appIndex, cmdQueuePtr->deviceID);
			}

			if (tmpVoclRebalance.isMigratedNecessary == 1)
			{
				voclProxyMigAppIndex = appIndex;
				voclProxyMigCmdQueue = tmpVoclRebalance.command_queue;
				voclProxySetMigrationCondition(1);

				/* wait for migration of commands in internal queues */
				pthread_barrier_wait(&barrierMigOperations);

				pthread_barrier_wait(&barrierMigOperations);

				//pthread_barrier_wait(&barrierMigOperations);
				tmpVoclRebalance.reissueWriteNum = voclProxyMigReissueWriteNum;
				tmpVoclRebalance.reissueReadNum = voclProxyMigReissueReadNum;
			}
			
			/* send back the response */
			MPI_Isend(&tmpVoclRebalance, sizeof(struct strVoclRebalance), MPI_BYTE,
					  appRank, VOCL_REBALANCE, appComm[commIndex], curRequest);
			MPI_Wait(curRequest, curStatus);
		}

		else if (status.MPI_TAG == VOCL_MIG_CMD_OPERATIONS)
		{
			/* migration of the kernel launch operation */
			memcpy(&tmpMigQueueOperations, conMsgBuffer[index],
				   sizeof(struct strMigQueueOperations));
			migMsgBuffer = (char *)malloc(tmpMigQueueOperations.msgSize);
			MPI_Irecv(migMsgBuffer, tmpMigQueueOperations.msgSize, MPI_BYTE, appRank,
					  VOCL_MIG_CMD_OPERATIONS, appCommData[commIndex], curRequest);
			MPI_Wait(curRequest, curStatus);

			/* for each cmd, updated to the migrated value */
			paramOffset = 0;
			for (i = 0; i < tmpMigQueueOperations.operationNum; i++)
			{
				/* decode the cmd message */
				cmdPtr = (vocl_internal_command_queue *)(migMsgBuffer + paramOffset);
				paramOffset += sizeof(vocl_internal_command_queue);

				/* get buffer from internal queue */
				voclCmdQueuePtr = voclProxyGetInternalQueueTail();
				voclCmdQueuePtr->msgTag = cmdPtr->msgTag;
				voclCmdQueuePtr->appComm = appComm[tmpMigQueueOperations.appIndexOnDestProxy];
				voclCmdQueuePtr->appCommData = appCommData[tmpMigQueueOperations.appIndexOnDestProxy];
				voclCmdQueuePtr->appRank = appRank; /* this appRank info is invalid */
				voclCmdQueuePtr->appIndex = tmpMigQueueOperations.appIndexOnDestProxy;
				memcpy(voclCmdQueuePtr->conMsgBuffer, cmdPtr->conMsgBuffer, MAX_CMSG_SIZE);
				/* update the kernel, command queue,and buffer info */
				paramOffset += voclProxyMigUpdateInternalCommand(voclCmdQueuePtr, migMsgBuffer + paramOffset);

				/* release lock for the current entry */
				voclProxyUnlockItem(voclCmdQueuePtr);
			}
			
			MPI_Isend(&tmpMigQueueOperations, sizeof(struct strMigQueueOperations), MPI_BYTE, 
					  appRank, VOCL_MIG_CMD_OPERATIONS, appCommData[commIndex], curRequest);
			/* send a message back to the source proxy process */
			free(migMsgBuffer);
			MPI_Wait(curRequest, curStatus);
		}

		else if (status.MPI_TAG == VOCL_CHK_PROYX_INMIG)
		{
			pthread_mutex_lock(&internalQueueMutex);
			isInMigration = voclProxyGetIsInMigration();
			MPI_Send(&isInMigration, sizeof(int), MPI_BYTE, appRank, 
					 VOCL_CHK_PROYX_INMIG, appComm[commIndex]);
			if (isInMigration == 1)
			{
				do 
				{
					isInMigration = voclProxyGetIsInMigration();
				}
				while (isInMigration == 1);

				MPI_Send(NULL, 0, MPI_BYTE, appRank, 
					 	 VOCL_CHK_PROYX_INMIG, appComm[commIndex]);
			}
			pthread_mutex_unlock(&internalQueueMutex);
		}

        else if (status.MPI_TAG == FORCED_MIGRATION) {
            memcpy(&tmpForcedMigration, conMsgBuffer[index],
                   sizeof(struct strForcedMigration));

            /* record forced migration status */
			voclProxySetMigrationCondition(tmpForcedMigration.status);

			/* it is a forced migration */
			voclProxySetForcedMigrationFlag(1);
			voclProxySetKernelNumThreshold(tmpForcedMigration.kernelNumThreshold);

			//debug----------------------
			printf("migrationStatus = %d\n", tmpForcedMigration.status);
			printf("migKernelNumThreshold = %d\n", tmpForcedMigration.kernelNumThreshold);

            tmpForcedMigration.res = 1;
            MPI_Send(&tmpForcedMigration, sizeof(struct strForcedMigration), MPI_BYTE,
                     appRank, FORCED_MIGRATION, appComm[commIndex]);
        }

        else if (status.MPI_TAG == LB_GET_KERNEL_NUM) {
			vocl_proxyGetKernelNumsOnGPUs(&tmpKernelNumOnDevice);
			tmpKernelNumOnDevice.rankNo = rankNo;
            MPI_Send(&tmpKernelNumOnDevice, sizeof(struct strKernelNumOnDevice), MPI_BYTE,
                     appRank, LB_GET_KERNEL_NUM, appCommData[commIndex]);
        }

        else if (status.MPI_TAG == PROGRAM_END) {
			memcpy(&tmpVoclProgramEnd, conMsgBuffer[index], sizeof(struct strVoclProgramEnd));
				
            /* cancel the corresponding requests */
            requestOffset = commIndex * CMSG_NUM;
            for (requestNo = 0; requestNo < CMSG_NUM; requestNo++) {
                if (conMsgRequest[requestOffset + requestNo] != MPI_REQUEST_NULL) {
                    MPI_Cancel(&conMsgRequest[requestOffset + requestNo]);
                    MPI_Request_free(&conMsgRequest[requestOffset + requestNo]);
                }
            }

            /* remove the correponding requests, communicator, etc */
			if (tmpVoclProgramEnd.isFreeWindow)
			{
				voclProxyFreeWin(commIndex);
			}
            voclProxyDisconnectOneApp(commIndex);

            continue;
        }

        /* issue it for later call of this function */
        MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG,
                  appComm[commIndex], conMsgRequest + index);
    }
    free(curStatus);
    free(curRequest);

    /* terminate the helper thread */
    pthread_barrier_wait(&barrier);
    pthread_barrier_wait(&barrier);
    if (pthread_join(th, NULL) != 0) {
        printf("pthread_join of data pipeline thread error.\n");
        exit(1);
    }
    pthread_barrier_destroy(&barrier);
	pthread_barrier_destroy(&barrierMigOperations);
	pthread_mutex_destroy(&internalQueueMutex);

	if (pthread_join(thKernelLaunch, NULL) != 0) {
        printf("pthread_join of kernel launch thread error.\n");
        exit(1);
    }

    /* unpublish the server name */
    MPI_Unpublish_name(serviceName, MPI_INFO_NULL, voclPortName);

    /* release the write and read buffer pool */
    finalizeReadBufferAll();
    finalizeWriteBufferAll();

    free(kernelMsgBuffer);

    voclMigWriteBufferFinalize();
    voclMigReadBufferFinalize();
    voclMigRWBufferFinalize();

	voclProxyReleaseProxyMsgReceive();
	voclProxyReleaseAllVirtualGPU();
    voclProxyCommFinalize();

    /* record objects allocated */
    voclProxyObjCountFinalize();
	voclProxyInternalQueueFinalize();
	voclProxyWinFinalize();

    MPI_Finalize();

    return 0;
}

