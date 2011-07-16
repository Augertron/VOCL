#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <memory.h>
#include <CL/opencl.h>
#include <sched.h>
#include <pthread.h>
#include "vocl_proxy.h"
#include "vocl_proxy_macro.h"
#include "vocl_proxyBufferProc.h"
#include "vocl_proxyKernelArgProc.h"

#define _PRINT_NODE_NAME

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
static struct strMigrationCheck tmpMigrationCheck;
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
static struct strMigGPUMemoryWriteCmpd tmpMigWriteMemCmpdRst;
static struct strMigGPUMemoryReadCmpd tmpMigReadMemCmpdRst;
static struct strMigRemoteGPURWCmpd tmpMigGPUMemRWCmpd;
static struct strForcedMigration tmpForcedMigration;
static struct strDeviceCmdQueueNums tmpDeviceCmdQueueNums;
/* forced migration status */
static int forcedMigrationStatus = 0;
static int voclRankThreshold = 1000;

/* control message requests */
MPI_Request *conMsgRequest;
MPI_Request *conMsgRequestForWait;
int *conMsgRequestIndex;
MPI_Comm *appComm, *appCommData;
char voclPortName[MPI_MAX_PORT_NAME];
int voclTotalRequestNum;
int voclCommUsedSize;

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
extern int readSendToLocal(int rank, int index);
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

extern void voclProxyCommInitialize();
extern void voclProxyCommFinalize();
extern void voclProxyAcceptOneApp();
extern void voclProxyDisconnectOneApp(int commIndex);
extern void *proxyCommAcceptThread(void *p);

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

/* device info for migration */
extern void voclProxyUpdateMemoryOnCmdQueue(cl_command_queue cmdQueue, cl_mem mem,
                                            size_t size);

/* migration functions related to device info stored in teh proxy */
extern void voclProxyCreateDevice(cl_device_id device, size_t globalSize);
extern void voclProxyReleaseAllDevices();
extern int voclProxyMigrationCheckKernelLaunch(cl_command_queue cmdQueue,
                                               kernel_args * argsPtr, int argsNum);
extern int voclProxyMigrationCheckWriteBuffer(cl_command_queue cmdQueue, size_t size);
extern void voclProxyUpdateCmdQueueOnDeviceID(cl_device_id device, cl_command_queue cmdQueue);
extern void voclProxyReleaseMem(cl_mem mem);
extern void voclProxyUpdateGlobalMemUsage(cl_command_queue comman_queue, kernel_args * argsPtr,
                                          int argsNum);
extern void voclProxyGetDeviceCmdQueueNums(struct strDeviceCmdQueueNums *cmdQueueNums);

/* functions to manage objects allocated in the proxy process */
extern void voclProxyObjCountInitialize();
extern void voclProxyObjCountFinalize();
extern void voclProxyObjCountIncrease();
extern void voclProxyObjCountDecrease();

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
    MPI_Comm comm1, comm2;
    int len;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &multiThreadProvided);

    /* issue non-blocking receive for all control messages */
    MPI_Status *curStatus;
    MPI_Request *curRequest;
    int requestNo, index;
    int requestOffset;
    int bufferNum, bufferIndex;
    size_t bufferSize, remainingSize;

    /* variables used by OpenCL API function */
    cl_platform_id *platforms;
    cl_device_id *devices;
    cl_uint numPlatforms, numDevices;
    cl_uint num_entries;
    cl_event *event_wait_list;
    cl_uint num_events_in_wait_list;
    cl_ulong globalSize;

    struct strWriteBufferInfo *writeBufferInfoPtr;
    struct strReadBufferInfo *readBufferInfoPtr;
    struct strMigWriteBufferInfo *migWriteBufferInfoPtr;
    struct strMigReadBufferInfo *migReadBufferInfoPtr;
    struct strMigRWBufferSameNode *migRWBufferInfoPtr;

    size_t *lengthsArray;
    size_t fileSize;
    char *fileBuffer;
    char *buildOptionBuffer;
    char *kernelName;
    void *host_ptr;
    void *arg_value;
    int work_dim;

    size_t *global_work_offset, *global_work_size, *local_work_size, paramOffset;
    kernel_args *args_ptr;
    size_t param_value_size;
    void *param_value;
    cl_uint num_events;
    cl_event *event_list;
    size_t host_buff_size, kernelMsgSize;
    char *kernelMsgBuffer;

	kernelMsgSize = 2048;
    kernelMsgBuffer = (char *) malloc(sizeof(char) * kernelMsgSize);

    /* get the proxy host name */
    MPI_Get_processor_name(voclProxyHostName, &len);
    voclProxyHostName[len] = '\0';
    //debug---------------------
    int tmp;
    MPI_Comm_rank(MPI_COMM_WORLD, &tmp);
    //-------------------------------------

#ifdef _PRINT_NODE_NAME
    {
        printf("rank = %d, proxyHostName = %s\n", tmp, voclProxyHostName);
    }
#endif

    /* retrieve device information on the node */
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms, NULL);
    for (i = 0; i < numPlatforms; i++) {
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
        devices = (cl_device_id *) malloc(sizeof(cl_device_id) * numDevices);
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
        /*get global memory size of each device */
        for (j = 0; j < numDevices; j++) {
            err =
                clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
                                &globalSize, NULL);
            voclProxyCreateDevice(devices[j], (size_t) globalSize);
        }
        free(devices);
    }
    free(platforms);

    /* service name based on different proxy node name */
    sprintf(serviceName, "voclCloud%s", voclProxyHostName);
    /* open a port on each node */
    MPI_Open_port(MPI_INFO_NULL, voclPortName);
    err = MPI_Publish_name(serviceName, MPI_INFO_NULL, voclPortName);

    curStatus = (MPI_Status *) malloc(sizeof(MPI_Status) * TOTAL_MSG_NUM);
    curRequest = (MPI_Request *) malloc(sizeof(MPI_Request) * TOTAL_MSG_NUM);

    voclProxyCommInitialize();

    /* record objects allocated */
    voclProxyObjCountInitialize();

    /* initialize write and read buffer pools */
    initializeWriteBufferAll();
    initializeReadBufferAll();

    voclMigWriteBufferInitializeAll();
    voclMigReadBufferInitializeAll();
    voclMigRWBufferInitializeAll();

    /* no forced migration needed */
    //debug-------------------
    forcedMigrationStatus = 0;

    /* wait for one app to issue connection request */
    voclProxyAcceptOneApp();

    /* create a helper thread */
    pthread_barrier_init(&barrier, NULL, 2);
    pthread_create(&th, NULL, proxyHelperThread, NULL);
    pthread_create(&thAppComm, NULL, proxyCommAcceptThread, NULL);

    while (1) {
        /* wait for any msg from the master process */
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
              tmp, voclTotalRequestNum, appIndex, index, status.MPI_TAG);
        //-------------------------------------

        if (status.MPI_TAG == GET_PLATFORM_ID_FUNC) {
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

            MPI_Isend(&tmpCreateContext, sizeof(tmpCreateContext), MPI_BYTE, appRank,
                      CREATE_CONTEXT_FUNC, appComm[commIndex], curRequest);

            MPI_Wait(curRequest, curStatus);
            if (devices != NULL) {
                free(devices);
            }
        }

        else if (status.MPI_TAG == CREATE_COMMAND_QUEUE_FUNC) {
            memcpy(&tmpCreateCommandQueue, conMsgBuffer[index], sizeof(tmpCreateCommandQueue));
            mpiOpenCLCreateCommandQueue(&tmpCreateCommandQueue);

            /* store the command queue locally */
            voclProxyUpdateCmdQueueOnDeviceID(tmpCreateCommandQueue.device,
                                              tmpCreateCommandQueue.clCommand);

            MPI_Isend(&tmpCreateCommandQueue, sizeof(tmpCreateCommandQueue), MPI_BYTE, appRank,
                      CREATE_COMMAND_QUEUE_FUNC, appComm[commIndex], curRequest);
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
            MPI_Isend(&tmpCreateProgramWithSource, sizeof(tmpCreateProgramWithSource),
                      MPI_BYTE, appRank, CREATE_PROGRMA_WITH_SOURCE, appComm[commIndex],
                      curRequest);

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
            kernelName = (char *) malloc((tmpCreateKernel.kernelNameSize + 1) * sizeof(char));
            MPI_Irecv(kernelName, tmpCreateKernel.kernelNameSize, MPI_CHAR, appRank,
                      CREATE_KERNEL1, appCommData[commIndex], curRequest);
            kernelName[tmpCreateKernel.kernelNameSize] = '\0';
            MPI_Wait(curRequest, curStatus);
            mpiOpenCLCreateKernel(&tmpCreateKernel, kernelName);
            MPI_Isend(&tmpCreateKernel, sizeof(tmpCreateKernel), MPI_BYTE, appRank,
                      CREATE_KERNEL, appComm[commIndex], curRequest);

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
            MPI_Isend(&tmpCreateBuffer, sizeof(tmpCreateBuffer), MPI_BYTE, appRank,
                      CREATE_BUFFER_FUNC, appComm[commIndex], curRequest);

            if (tmpCreateBuffer.host_ptr_flag == 1) {
                free(host_ptr);
            }
            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == ENQUEUE_WRITE_BUFFER) {
            memcpy(&tmpEnqueueWriteBuffer, conMsgBuffer[index], sizeof(tmpEnqueueWriteBuffer));
            requestNo = 0;
            event_wait_list = NULL;
            num_events_in_wait_list = tmpEnqueueWriteBuffer.num_events_in_wait_list;
            if (num_events_in_wait_list > 0) {
                event_wait_list =
                    (cl_event *) malloc(sizeof(cl_event) * num_events_in_wait_list);
                MPI_Irecv(event_wait_list, sizeof(cl_event) * num_events_in_wait_list,
                          MPI_BYTE, appRank, tmpEnqueueWriteBuffer.tag, appCommData[commIndex],
                          curRequest + (requestNo++));
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
                          VOCL_PROXY_WRITE_TAG + bufferIndex, appCommData[commIndex],
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
                          appRank, ENQUEUE_WRITE_BUFFER, appComm[commIndex],
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
                              appRank, ENQUEUE_WRITE_BUFFER, appComm[commIndex],
                              curRequest + (requestNo++));
                }
            }

            if (requestNo > 0) {
                MPI_Wait(curRequest, curStatus);
            }
        }

        else if (status.MPI_TAG >= VOCL_PROXY_WRITE_TAG &&
                 status.MPI_TAG < VOCL_PROXY_WRITE_TAG + VOCL_PROXY_WRITE_BUFFER_NUM) {
            writeBufferIndexInHelperThread = status.MPI_TAG - VOCL_PROXY_WRITE_TAG;
            pthread_barrier_wait(&barrier);
            helperThreadOperFlag = GPU_WRITE_SINGLE;
            voclProxyAppIndex = appIndex;
            pthread_barrier_wait(&barrier);
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

        else if (status.MPI_TAG == MIGRATION_CHECK) {
            memcpy(&tmpMigrationCheck, conMsgBuffer[index], sizeof(struct strMigrationCheck));
            /* requested from kernel launch */
            tmpMigrationCheck.isMigrationNeeded = 0;
            if (tmpMigrationCheck.checkLocation == 0) {
                args_ptr =
                    (kernel_args *) malloc(tmpMigrationCheck.argsNum * sizeof(kernel_args));
                MPI_Irecv(args_ptr, tmpMigrationCheck.argsNum * sizeof(kernel_args),
                          MPI_BYTE, appRank, MIGRATION_CHECK, appCommData[commIndex],
                          curRequest);
                MPI_Wait(curRequest, curStatus);

                tmpMigrationCheck.isMigrationNeeded =
                    voclProxyMigrationCheckKernelLaunch(tmpMigrationCheck.command_queue,
                                                        args_ptr, tmpMigrationCheck.argsNum);
                free(args_ptr);
            }
            /* requested from enqueue write buffer */
            else if (tmpMigrationCheck.checkLocation == 1) {
                tmpMigrationCheck.isMigrationNeeded =
                    voclProxyMigrationCheckWriteBuffer(tmpMigrationCheck.command_queue,
                                                       tmpMigrationCheck.memSize);
            }

            if (forcedMigrationStatus == 1 && tmpMigrationCheck.rankNo >= voclRankThreshold) {
                tmpMigrationCheck.isMigrationNeeded = 1;
            }

            MPI_Isend(&tmpMigrationCheck, sizeof(struct strMigrationCheck), MPI_BYTE, appRank,
                      MIGRATION_CHECK, appComm[commIndex], curRequest);
            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == ENQUEUE_ND_RANGE_KERNEL) {
            memcpy(&tmpEnqueueNDRangeKernel, conMsgBuffer[index],
                   sizeof(tmpEnqueueNDRangeKernel));
            requestNo = 0;
            event_wait_list = NULL;
            num_events_in_wait_list = tmpEnqueueNDRangeKernel.num_events_in_wait_list;
            if (num_events_in_wait_list > 0) {
                event_wait_list =
                    (cl_event *) malloc(sizeof(cl_event) * num_events_in_wait_list);
                MPI_Irecv(event_wait_list, sizeof(cl_event) * num_events_in_wait_list,
                          MPI_BYTE, appRank, ENQUEUE_ND_RANGE_KERNEL1, appCommData[commIndex],
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
                          ENQUEUE_ND_RANGE_KERNEL1, appCommData[commIndex],
                          curRequest + (requestNo++));
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
            voclProxyUpdateGlobalMemUsage(tmpEnqueueNDRangeKernel.command_queue,
                                          args_ptr, tmpEnqueueNDRangeKernel.args_num);

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

            MPI_Isend(&kernelLaunchReply, sizeof(struct strEnqueueNDRangeKernelReply),
                      MPI_BYTE, appRank, ENQUEUE_ND_RANGE_KERNEL, appComm[commIndex],
                      curRequest);

            if (num_events_in_wait_list > 0) {
                free(event_wait_list);
            }

            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == ENQUEUE_READ_BUFFER) {
            memcpy(&tmpEnqueueReadBuffer, conMsgBuffer[index], sizeof(tmpEnqueueReadBuffer));
            num_events_in_wait_list = tmpEnqueueReadBuffer.num_events_in_wait_list;
            event_wait_list = NULL;
            if (num_events_in_wait_list > 0) {
                event_wait_list =
                    (cl_event *) malloc(num_events_in_wait_list * sizeof(cl_event));
                MPI_Irecv(event_wait_list, num_events_in_wait_list * sizeof(cl_event),
                          MPI_BYTE, appRank, ENQUEUE_READ_BUFFER1, appCommData[commIndex],
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
                readBufferInfoPtr->comm = appCommData[commIndex];
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
                              appRank, ENQUEUE_READ_BUFFER, appComm[commIndex], curRequest);
                }
            }
            else {      /* blocking, reading is complete, send data to local node */
                tmpEnqueueReadBuffer.res = processAllReads(appIndex);
                if (tmpEnqueueReadBuffer.event_null_flag == 0) {
                    tmpEnqueueReadBuffer.event = readBufferInfoPtr->event;
                }
                MPI_Isend(&tmpEnqueueReadBuffer, sizeof(tmpEnqueueReadBuffer), MPI_BYTE,
                          appRank, ENQUEUE_READ_BUFFER, appComm[commIndex], curRequest);

            }
            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == RELEASE_MEM_OBJ) {
            memcpy(&tmpReleaseMemObject, conMsgBuffer[index], sizeof(tmpReleaseMemObject));
            voclProxyReleaseMem(tmpReleaseMemObject.memobj);
            mpiOpenCLReleaseMemObject(&tmpReleaseMemObject);
            MPI_Isend(&tmpReleaseMemObject, sizeof(tmpReleaseMemObject), MPI_BYTE, appRank,
                      RELEASE_MEM_OBJ, appComm[commIndex], curRequest);
            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == CL_RELEASE_KERNEL_FUNC) {
            memcpy(&tmpReleaseKernel, conMsgBuffer[index], sizeof(tmpReleaseKernel));
            mpiOpenCLReleaseKernel(&tmpReleaseKernel);
            MPI_Isend(&tmpReleaseKernel, sizeof(tmpReleaseKernel), MPI_BYTE, appRank,
                      CL_RELEASE_KERNEL_FUNC, appComm[commIndex], curRequest);

            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == FINISH_FUNC) {
            memcpy(&tmpFinish, conMsgBuffer[index], sizeof(tmpFinish));
            processAllWrites(appIndex);
            processAllReads(appIndex);
            mpiOpenCLFinish(&tmpFinish);
            MPI_Isend(&tmpFinish, sizeof(tmpFinish), MPI_BYTE, appRank,
                      FINISH_FUNC, appComm[commIndex], curRequest);

            MPI_Wait(curRequest, curStatus);
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
            mpiOpenCLReleaseProgram(&tmpReleaseProgram);
            MPI_Isend(&tmpReleaseProgram, sizeof(tmpReleaseProgram), MPI_BYTE, appRank,
                      REL_PROGRAM_FUNC, appComm[commIndex], curRequest);
            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == REL_COMMAND_QUEUE_FUNC) {
            memcpy(&tmpReleaseCommandQueue, conMsgBuffer[index],
                   sizeof(tmpReleaseCommandQueue));
            mpiOpenCLReleaseCommandQueue(&tmpReleaseCommandQueue);
            MPI_Isend(&tmpReleaseCommandQueue, sizeof(tmpReleaseCommandQueue), MPI_BYTE,
                      appRank, REL_COMMAND_QUEUE_FUNC, appComm[commIndex], curRequest);
            MPI_Wait(curRequest, curStatus);
        }

        else if (status.MPI_TAG == REL_CONTEXT_FUNC) {
            memcpy(&tmpReleaseContext, conMsgBuffer[index], sizeof(tmpReleaseContext));
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
            memcpy(&tmpFlush, conMsgBuffer[index], sizeof(tmpFlush));
            mpiOpenCLFlush(&tmpFlush);
            MPI_Isend(&tmpFlush, sizeof(tmpFlush), MPI_BYTE, appRank,
                      FLUSH_FUNC, appComm[commIndex], curRequest);

            MPI_Wait(curRequest, curStatus);
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

            voclProxyUpdateMemoryOnCmdQueue(tmpMigGPUMemoryWrite.cmdQueue,
                                            tmpMigGPUMemoryWrite.memory,
                                            tmpMigGPUMemoryWrite.size);
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

        else if (status.MPI_TAG == FORCED_MIGRATION) {
            memcpy(&tmpForcedMigration, conMsgBuffer[index],
                   sizeof(struct strForcedMigration));
            /* record forced migration status */
            forcedMigrationStatus = tmpForcedMigration.status;
            voclRankThreshold = tmpForcedMigration.rankThreshold;
            printf("voclRankThreshold = %d\n", voclRankThreshold);
            tmpForcedMigration.res = 1;
            MPI_Send(&tmpForcedMigration, sizeof(struct strForcedMigration), MPI_BYTE,
                     appRank, FORCED_MIGRATION, appComm[commIndex]);
        }

        else if (status.MPI_TAG == LB_GET_CMDQUEUE_NUM) {
            voclProxyGetDeviceCmdQueueNums(&tmpDeviceCmdQueueNums);
            MPI_Send(&tmpDeviceCmdQueueNums, sizeof(struct strDeviceCmdQueueNums), MPI_BYTE,
                     appRank, LB_GET_CMDQUEUE_NUM, appComm[commIndex]);
        }

        else if (status.MPI_TAG == PROGRAM_END) {
            /* cancel the corresponding requests */
            requestOffset = commIndex * CMSG_NUM;
            for (requestNo = 0; requestNo < CMSG_NUM; requestNo++) {
                //if (requestOffset + requestNo != index)
                if (conMsgRequest[requestOffset + requestNo] != MPI_REQUEST_NULL) {
                    MPI_Cancel(&conMsgRequest[requestOffset + requestNo]);
                    MPI_Request_free(&conMsgRequest[requestOffset + requestNo]);
                }
            }

            /* remove the correponding requests, communicator, etc */
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

    /* unpublish the server name */
    MPI_Unpublish_name(serviceName, MPI_INFO_NULL, voclPortName);

    /* release the write and read buffer pool */
    finalizeReadBufferAll();
    finalizeWriteBufferAll();

    free(kernelMsgBuffer);

    voclMigWriteBufferFinalize();
    voclMigReadBufferFinalize();
    voclMigRWBufferFinalize();

    voclProxyCommFinalize();

    /* record objects allocated */
    voclProxyObjCountFinalize();

    voclProxyReleaseAllDevices();

    MPI_Finalize();

    return 0;
}
