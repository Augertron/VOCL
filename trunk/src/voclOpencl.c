#include <stdio.h>
#include <string.h>
#include "voclOpencl.h"
#include "voclOpenclMacro.h"
#include "voclStructures.h"
#include "voclKernelArgProc.h"
#include <sys/time.h>
#include <unistd.h>

/* used for print node name */
#define _PRINT_NODE_NAME

/* for slave process */
static MPI_Comm *voclProxyComm = NULL;
static MPI_Comm *voclProxyCommData = NULL;
static int *voclProxyRank = NULL;
static int slaveCreated = 0;
static int np;
static int MPIexternalInit = 0;
static int errCodes[MAX_NPS];

/* for proxy host name process */
extern int voclGetProxyHostNum();
extern char *voclGetProxyHostName(int index);
extern void voclProxyHostFinalize();
extern void voclCreateProxyHostNameList();

/* for platform ID processing */
extern void voclPlatformIDInitialize();
extern void voclPlatformIDFinalize();
extern vocl_platform_id voclCLPlatformID2VOCLPlatformID(cl_platform_id platformID,
                                                        int proxyRank, int proxyIndex,
                                                        MPI_Comm proxyComm,
                                                        MPI_Comm proxyCommData);
extern cl_platform_id voclVOCLPlatformID2CLPlatformIDComm(vocl_platform_id platformID,
                                                          int *proxyRank, int *proxyIndex,
                                                          MPI_Comm * proxyComm,
                                                          MPI_Comm * proxyCommData);

/* for device ID processing */
extern void voclDeviceIDInitialize();
extern void voclDeviceIDFinalize();
extern vocl_device_id voclCLDeviceID2VOCLDeviceID(cl_device_id deviceID, int proxyRank,
                                                  int proxyIndex, MPI_Comm proxyComm,
                                                  MPI_Comm proxyCommData);
extern cl_device_id voclVOCLDeviceID2CLDeviceIDComm(vocl_device_id deviceID, int *proxyRank,
                                                    int *proxyIndex, MPI_Comm * proxyComm,
                                                    MPI_Comm * proxyCommData);

/* for context processing */
extern void voclContextInitialize();
extern void voclContextFinalize();
extern vocl_context voclCLContext2VOCLContext(cl_context context, int proxyRank,
                                              int proxyIndex, MPI_Comm proxyComm,
                                              MPI_Comm proxyCommData);
extern cl_context voclVOCLContext2CLContextComm(vocl_context context, int *proxyRank,
                                                int *proxyIndex, MPI_Comm * proxyComm,
                                                MPI_Comm * proxyCommData);
extern int voclReleaseContext(vocl_context context);
extern void voclContextSetMigrationStatus(vocl_context context, int status);
extern int voclContextGetMigrationStatus(vocl_context context);

/* for command queue processing */
extern void voclCommandQueueInitialize();
extern void voclCommandQueueFinalize();
extern vocl_command_queue voclCLCommandQueue2VOCLCommandQueue(cl_command_queue commandQueue,
                                                              int proxyRank, int proxyIndex,
                                                              MPI_Comm proxyComm,
                                                              MPI_Comm proxyCommData);
extern cl_command_queue voclVOCLCommandQueue2CLCommandQueueComm(vocl_command_queue
                                                                commandQueue, int *proxyRank,
                                                                int *proxyIndex,
                                                                MPI_Comm * proxyComm,
                                                                MPI_Comm * proxyCommData);
extern int voclReleaseCommandQueue(vocl_command_queue command_queue);
extern void voclStoreCmdQueueProperties(vocl_command_queue command_queue,
                                        cl_command_queue_properties properties,
                                        vocl_context context, vocl_device_id device);
extern int voclIsOldCommandQueueValid(vocl_command_queue command_queue);
extern cl_command_queue voclVOCLCommandQueue2OldCLCommandQueueComm(vocl_command_queue
                                                                   command_queue,
                                                                   int *proxyRank,
                                                                   int *proxyIndex,
                                                                   MPI_Comm * proxyComm,
                                                                   MPI_Comm * proxyCommData);
extern void voclCommandQueueSetMigrationStatus(vocl_command_queue cmdQueue, int status);
extern void voclCommandQueueSetMigrationStatus(vocl_command_queue cmdQueue, int status);
extern int voclCommandQueueGetMigrationStatus(vocl_command_queue cmdQueue);
extern void voclCommandQueueMigration(vocl_command_queue command_queue);
extern void voclUpdateMemoryInCommandQueue(vocl_command_queue command_queue, vocl_mem mem,
                                           size_t size);

/* for program processing */
extern void voclProgramInitialize();
extern void voclProgramFinalize();
extern vocl_program voclCLProgram2VOCLProgram(cl_program program, int proxyRank,
                                              int proxyIndex, MPI_Comm proxyComm,
                                              MPI_Comm proxyCommData);
extern cl_program voclVOCLProgram2CLProgramComm(vocl_program program, int *proxyRank,
                                                int *proxyIndex, MPI_Comm * proxyComm,
                                                MPI_Comm * proxyCommData);
extern int voclReleaseProgram(vocl_program program);
extern void voclStoreProgramSource(vocl_program program, char *source, size_t sourceSize);
extern void voclStoreProgramContext(vocl_program program, vocl_context context);
extern vocl_context voclGetContextFromProgram(vocl_program program);
extern void voclProgramSetMigrationStatus(vocl_program program, int status);
extern int voclProgramGetMigrationStatus(vocl_program program);

/* for program processing */
extern void voclMemoryInitialize();
extern void voclMemoryFinalize();
extern vocl_mem voclCLMemory2VOCLMemory(cl_mem memory, int proxyRank,
                                        int proxyIndex, MPI_Comm proxyComm,
                                        MPI_Comm proxyCommData);
extern cl_mem voclVOCLMemory2CLMemoryComm(vocl_mem memory, int *proxyRank, int *proxyIndex,
                                          MPI_Comm * proxyComm, MPI_Comm * proxyCommData);
extern int voclReleaseMemory(vocl_mem mem);
extern void voclStoreMemoryParameters(vocl_mem memory, cl_mem_flags flags,
                                      size_t size, vocl_context context);
extern void voclSetMemWrittenFlag(vocl_mem memory, int flag);
extern void voclSetMemHostPtr(vocl_mem memory, void *ptr);
extern size_t voclGetVOCLMemorySize(vocl_mem memory);
extern int voclIsOldMemoryValid(vocl_mem memory);
extern cl_mem voclVOCLMemory2OldCLMemoryComm(vocl_mem memory, int *proxyRank,
                                             int *proxyIndex, MPI_Comm * proxyComm,
                                             MPI_Comm * proxyCommData);
extern cl_mem voclVOCLMemory2CLMemory(vocl_mem memory);
extern void voclMemSetMigrationStatus(vocl_mem mem, int status);
extern int voclMemGetMigrationStatus(vocl_mem mem);
extern void voclUpdateSingleMemory(vocl_mem mem);
extern cl_int clMigReleaseOldMemObject(vocl_mem memobj);

/* for program processing */
extern void voclKernelInitialize();
extern void voclKernelFinalize();
extern vocl_kernel voclCLKernel2VOCLKernel(cl_kernel kernel, int proxyRank,
                                           int proxyIndex, MPI_Comm proxyComm,
                                           MPI_Comm proxyCommData);
extern cl_kernel voclVOCLKernel2CLKernelComm(vocl_kernel kernel, int *proxyRank,
                                             int *proxyIndex, MPI_Comm * proxyComm,
                                             MPI_Comm * proxyCommData);
extern int voclReleaseKernel(vocl_kernel kernel);
extern void voclStoreKernelProgramContext(vocl_kernel kernel, vocl_program program,
                                          vocl_context context);
extern void voclKernelSetMigrationStatus(vocl_kernel kernel, int status);
extern int voclKernelGetMigrationStatus(vocl_kernel kernel);
extern void voclUpdateSingleKernel(vocl_kernel kernel, vocl_command_queue command_queue);

/* kernel argument processing functions */
extern cl_int createKernel(cl_kernel kernel);
extern kernel_info *getKernelPtr(cl_kernel kernel);
extern cl_int releaseKernelPtr(cl_kernel kernel);
extern void createKernelArgInfo(cl_kernel kernel, char *kernel_name, vocl_program program);

/* writeBufferPool API functions */
extern void initializeVoclWriteBufferAll();
extern void finalizeVoclWriteBufferAll();
extern void setWriteBufferInUse(int proxyIndex, int index);
extern MPI_Request *getWriteRequestPtr(int proxyIndex, int index);
extern int getNextWriteBufferIndex(int proxyIndex);
extern void processWriteBuffer(int proxyIndex, int curIndex, int bufferNum);
extern void processAllWrites(int proxyIndex);
extern void setWriteBufferNum(int proxyIndex, int index, int bufferNum);
extern void setWriteBufferEvent(int proxyIndex, int index, vocl_event event);
extern int getWriteBufferIndexFromEvent(int proxyIndex, vocl_event event);
extern int getWriteBufferNum(int proxyIndex, int index);

/* readBufferPool API functions */
extern void initializeVoclReadBufferAll();
extern void finalizeVoclReadBufferAll();
extern void setReadBufferInUse(int proxyIndex, int index);
extern MPI_Request *getReadRequestPtr(int proxyIndex, int index);
extern int getNextReadBufferIndex(int proxyIndex);
extern void processReadBuffer(int proxyIndex, int curIndex, int bufferNum);
extern void processAllReads(int proxyIndex);
extern void setReadBufferNum(int proxyIndex, int index, int bufferNum);
extern void setReadBufferEvent(int proxyIndex, int index, vocl_event event);
extern int getReadBufferIndexFromEvent(int proxyIndex, vocl_event event);
extern int getReadBufferNum(int proxyIndex, int index);


/* vocl event processing API functions */
extern void voclEventInitialize();
extern void voclEventFinalize();
extern vocl_event voclCLEvent2VOCLEvent(cl_event event, int proxy,
                                        int proxyIndex, MPI_Comm proxyComm,
                                        MPI_Comm proxyCommData);
extern cl_event voclVOCLEvent2CLEvent(vocl_event event);
extern cl_event voclVOCLEvent2CLEventComm(vocl_event event, int *proxyRank,
                                          int *proxyIndex, MPI_Comm * proxyComm,
                                          MPI_Comm * proxyCommData);
extern void voclVOCLEvents2CLEvents(vocl_event * voclEventList, cl_event * clEventList,
                                    cl_uint eventNum);
extern void voclVOCLEvents2CLEventsComm(vocl_event * voclEventList, cl_event * clEventList,
                                        cl_uint eventNum, int *proxyRank, int *proxyIndex,
                                        MPI_Comm * proxyComm, MPI_Comm * proxyCommData);
extern int voclReleaseEvent(vocl_event event);
extern void voclVOCLEvents2CLEvents(vocl_event * voclEventList,
                                    cl_event * clEventList, cl_uint eventNum);

/* vocl sampler processing API functions */
extern void voclSamplerInitialize();
extern void voclSamplerFinalize();
extern vocl_sampler voclCLSampler2VOCLSampler(cl_sampler sampler, int proxyRank,
                                              int proxyIndex, MPI_Comm proxyComm,
                                              MPI_Comm proxyCommData);
extern cl_sampler voclVOCLSampler2CLSamplerComm(vocl_sampler sampler, int *proxyRank,
                                                int *proxyIndex, MPI_Comm * proxyComm,
                                                MPI_Comm * proxyCommData);
extern int voclReleaseSampler(vocl_sampler sampler);
extern void voclSamplerSetMigrationStatus(vocl_sampler sampler, int status);

/* handle migration */
extern void voclGetLocalDeviceInfo();
extern void voclLibReleaseAllDevices();
extern void voclLibUpdateCmdQueueOnDeviceID(cl_device_id device, cl_command_queue cmdQueue);
extern void voclLibUpdateGlobalMemOnCommandQueue(cl_command_queue cmdQueue, cl_mem memory,
                                                 size_t size);
extern void voclLibReleaseMem(cl_mem mem);
void voclLibUpdateGlobalMemUsage(cl_command_queue cmdQueue, kernel_args * argsPtr,
                                 int argsNum);

/* proxy process name process */
extern int voclIsOnLocalNode(int index);
extern void voclSetIndex2NodeMapping(int index, int node);
extern void voclStoreKernelName(vocl_kernel kernel, char *kernelName);

/* dynamic Opencl function call */
extern void voclOpenclModuleInitialize();
extern void voclOpenclModuleRelease();

/*migration functions*/
extern void voclMigWriteLocalBufferInitializeAll();
extern void voclMigWriteLocalBufferFinalize();
extern void voclMigReadLocalBufferInitializeAll();
extern void voclMigReadLocalBufferFinalize();
extern void voclMigRWLocalBufferInitialize();
extern void voclMigRWLocalBufferFinalize();
extern void voclTaskMigration(vocl_kernel kernel, vocl_command_queue command_queue);
extern int voclCheckMigrationInKernelLaunch(vocl_command_queue cmdQueue, kernel_args * argsPtr,
                                            int argsNum);
extern void voclSetTaskMigrationCondition();
extern int voclGetTaskMigrationCondition();
extern int voclCheckMigrationInWriteBuffer(cl_command_queue cmdQueue, size_t size);
extern void voidMigrationHack(int *isMigrated);

/*******************************************************************/
/* for Opencl object count processing */
static unsigned int *voclObjCountPtr = NULL;
static unsigned int voclObjCountNum;
static unsigned int voclObjCountNo;
static void voclInitialize();
static void voclFinalize();

static void voclObjCountInitialize()
{
    voclObjCountNum = np;       /* one count for each proxy process */
    voclObjCountNo = 0;
    voclObjCountPtr = (unsigned int *) malloc(sizeof(unsigned int) * voclObjCountNum);
    memset(voclObjCountPtr, 0, voclObjCountNum * sizeof(unsigned int));
}

static void voclObjCountFinalize()
{
    voclObjCountNum = 0;
    voclObjCountNo = 0;

    if (voclObjCountPtr != NULL) {
        free(voclObjCountPtr);
        voclObjCountPtr = NULL;
    }
}

void voclObjCountIncrease(int proxyIndex)
{
    /* all proxy processes share the sam counter */
    proxyIndex = 0;
    if (proxyIndex >= voclObjCountNum) {
        voclObjCountPtr =
            (unsigned int *) realloc(voclObjCountPtr,
                                     sizeof(unsigned int) * 2 * voclObjCountNum);
        memset(&voclObjCountPtr[voclObjCountNum], 0, voclObjCountNum * sizeof(unsigned int));
        voclObjCountNum *= 2;
    }

    voclObjCountPtr[proxyIndex]++;
}

void voclObjCountDecrease(int proxyIndex)
{
    /* all proxy processes share the sam counter */
    proxyIndex = 0;
    voclObjCountPtr[proxyIndex]--;
    if (voclObjCountPtr[proxyIndex] == 0) {
        voclFinalize();
    }
}

/* end of opencl object count processing */
/************************************************************************/

/* send the terminating msg to the proxy */
void voclFinalize()
{
    int i;

    /* send empty msg to proxy to terminate its execution */
    for (i = 0; i < np; i++) {
        /* only for remote node */
        if (voclIsOnLocalNode(i) == VOCL_FALSE) {
            MPI_Send(NULL, 0, MPI_BYTE, voclProxyRank[i], PROGRAM_END, voclProxyComm[i]);
            MPI_Comm_disconnect(&voclProxyComm[i]);
            MPI_Comm_disconnect(&voclProxyCommData[i]);
        }
    }

    if (MPIexternalInit == 0) {
        MPI_Finalize();
    }

    /* free buffer for MPI communicator and proxy ID */
    free(voclProxyComm);
    free(voclProxyCommData);
    free(voclProxyRank);

    /* release memory */
    voclPlatformIDFinalize();
    voclDeviceIDFinalize();
    voclContextFinalize();
    voclCommandQueueFinalize();
    voclProgramFinalize();
    voclMemoryFinalize();
    voclKernelFinalize();
    voclEventFinalize();
    voclSamplerFinalize();
    voclObjCountFinalize();
    voclProxyHostFinalize();

    finalizeVoclWriteBufferAll();
    finalizeVoclReadBufferAll();

    /* close opencl lib */
    voclOpenclModuleRelease();

    voclMigWriteLocalBufferFinalize();
    voclMigReadLocalBufferFinalize();
    voclMigRWLocalBufferFinalize();

    voclLibReleaseAllDevices();
}

void voclInitialize()
{
    /* initialize buffers for vocl event */
    voclPlatformIDInitialize();
    voclDeviceIDInitialize();
    voclContextInitialize();
    voclCommandQueueInitialize();
    voclProgramInitialize();
    voclMemoryInitialize();
    voclKernelInitialize();
    voclEventInitialize();
    voclSamplerInitialize();
    voclObjCountInitialize();

    /* initialize buffer pool for gpu memory reads and writes */
    initializeVoclWriteBufferAll();
    initializeVoclReadBufferAll();

    /* initialization for dynamic opencl function call */
    voclOpenclModuleInitialize();
    voclMigWriteLocalBufferInitializeAll();
    voclMigReadLocalBufferInitializeAll();
    voclMigRWLocalBufferInitialize();

    /*initialize whether migration is requested */
    voclSetTaskMigrationCondition();

    return;
}


/* function for create the proxy process and MPI communicator */
static void checkSlaveProc()
{
    MPI_Info info;
    int i, err, rank;
    int proxyNum;
    int proxyNo = 0;
    FILE *proxyHostNameFile;
    char proxyHostFileName[255];
    char serviceName[256];
    char portName[MPI_MAX_PORT_NAME];

    if (slaveCreated == 0) {
        /* has connnected to some proxy processes */
        slaveCreated = 1;

        /* check whether MPI_Init is already called */
        MPI_Initialized(&MPIexternalInit);
        if (MPIexternalInit == 0) {     /* not called yet */
            MPI_Init(NULL, NULL);
        }

        /* proxy the environment variable for proxy host list */
        voclCreateProxyHostNameList();

        /*retrieve the number of proxy hosts, but currently, each application process */
        /* connects to only one proxy process */
        proxyNum = voclGetProxyHostNum();

        /* currently each app uses only one proxy process */
        np = proxyNum;
        printf("np = %d\n", np);

        /* allocate buffer for MPI communicator and proxy ID */
        voclProxyComm = (MPI_Comm *) malloc(sizeof(MPI_Comm) * np);
        voclProxyCommData = (MPI_Comm *) malloc(sizeof(MPI_Comm) * np);
        voclProxyRank = (int *) malloc(sizeof(int) * np);

        /* initialized resources needed by vocl */
        voclInitialize();

        /* use connect to establish communicator, the proxy is computed from the  */
        //debug-----------------------------------------
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        //----------------------------------------------

        /* rank of the app process */
        for (i = 0; i < np; i++) {
            voclSetIndex2NodeMapping(i, (rank + i) % proxyNum);
            //voclSetIndex2NodeMapping(i, i % proxyNum);
            if (voclIsOnLocalNode(i) == VOCL_FALSE) {
                sprintf(serviceName, "voclCloud%s", voclGetProxyHostName(i));

                err = MPI_Lookup_name(serviceName, MPI_INFO_NULL, portName);
                if (err != MPI_SUCCESS) {
                    printf("Lookup name error, %d\n", err);
                }

                /* establish inter node communicator */
                sleep(1);
                err =
                    MPI_Comm_connect(portName, MPI_INFO_NULL, 0, MPI_COMM_SELF,
                                     &voclProxyComm[i]);
                if (err != MPI_SUCCESS) {
                    printf("MPI_Comm_connect error, %d\n", err);
                    exit(1);
                }

                /* duplicate the communicator */
                err = MPI_Comm_dup(voclProxyComm[i], &voclProxyCommData[i]);
                if (err != MPI_SUCCESS) {
                    printf("MPI_Comm_dup error, %d\n", err);
                    exit(1);
                }

                /* since MPI_COMM_SELF, rankes of all proxy processes are 0 */
                voclProxyRank[i] = 0;
            }
            else { 
                /* lcoal gpu retrieve global memory info */
                voclGetLocalDeviceInfo();
            }
        }

#ifdef _PRINT_NODE_NAME
        {
            char hostName[200];
            int len;
            MPI_Get_processor_name(hostName, &len);
            hostName[len] = '\0';
            printf("libHostName = %s\n", hostName);
        }
#endif
    }
}

/*--------------------VOCL API functions, countparts of OpenCL API functions ----------------*/
cl_int
clGetPlatformIDs(cl_uint num_entries, cl_platform_id * platforms, cl_uint * num_platforms)
{
    MPI_Status *status;
    MPI_Request *request;
    cl_int err;
    int requestNo, i, j, rank, proxyIndex;
    int curPlatformNum, totalPlatformNum;
    struct strGetPlatformIDs *tmpGetPlatform;

    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    status = (MPI_Status *) malloc(sizeof(MPI_Status) * np * 3);
    request = (MPI_Request *) malloc(sizeof(MPI_Request) * np * 3);
    tmpGetPlatform =
        (struct strGetPlatformIDs *) malloc(sizeof(struct strGetPlatformIDs) * np);

    requestNo = 0;
    /* for each proxy process, send the request */
    for (i = 0; i < np; i++) {
        /* MPI is called only on remote node */
        if (voclIsOnLocalNode(i) == VOCL_FALSE) {
            /* initialize structure */
            tmpGetPlatform[i].num_entries = num_entries;
            tmpGetPlatform[i].platforms = platforms;
            tmpGetPlatform[i].num_platforms = 1;

            /* send parameters to remote node */
            MPI_Isend(&tmpGetPlatform[i], sizeof(struct strGetPlatformIDs), MPI_BYTE,
                      voclProxyRank[i], GET_PLATFORM_ID_FUNC, voclProxyComm[i],
                      request + (requestNo++));
        }
    }

    /*receive the number of platforms on each node */
    for (i = 0; i < np; i++) {
        /* MPI is used only on remote node */
        if (voclIsOnLocalNode(i) == VOCL_FALSE) {
            MPI_Irecv(&tmpGetPlatform[i], sizeof(struct strGetPlatformIDs), MPI_BYTE,
                      voclProxyRank[i], GET_PLATFORM_ID_FUNC, voclProxyComm[i],
                      request + (requestNo++));
        }
    }

    /* make sure the number of platforms received */
    MPI_Waitall(requestNo, request, status);
    requestNo = 0;

    totalPlatformNum = 0;
    for (i = 0; i < np; i++) {
        if (voclIsOnLocalNode(i) == VOCL_FALSE) {
            curPlatformNum = tmpGetPlatform[i].num_platforms;

            if (platforms != NULL && num_entries > 0) {
                MPI_Irecv(&platforms[totalPlatformNum],
                          sizeof(cl_platform_id) * curPlatformNum, MPI_BYTE, voclProxyRank[i],
                          GET_PLATFORM_ID_FUNC1, voclProxyCommData[i], request);
                MPI_Wait(request, status);
            }
        }
        /* for local node */
        else {
            if (platforms != NULL) {
                tmpGetPlatform[i].res =
                    dlCLGetPlatformIDs(num_entries - totalPlatformNum,
                                       &platforms[totalPlatformNum], &curPlatformNum);
            }
            else {
                tmpGetPlatform[i].res = dlCLGetPlatformIDs(0, NULL, &curPlatformNum);
            }
        }

        /* convert cl platform id to vocl platform id */
        if (platforms != NULL) {
            for (j = 0; j < curPlatformNum; j++) {
                platforms[totalPlatformNum + j] = (cl_platform_id)
                    voclCLPlatformID2VOCLPlatformID(platforms[totalPlatformNum + j],
                                                    voclProxyRank[i], i, voclProxyComm[i],
                                                    voclProxyCommData[i]);
            }
        }

        totalPlatformNum += curPlatformNum;
    }

    if (num_platforms != NULL) {
        *num_platforms = totalPlatformNum;
    }

    err = tmpGetPlatform[np - 1].res;

    free(status);
    free(request);
    free(tmpGetPlatform);

    return err;
}

cl_int
clGetDeviceIDs(cl_platform_id platform,
               cl_device_type device_type,
               cl_uint num_entries, cl_device_id * devices, cl_uint * num_devices)
{
    struct strGetDeviceIDs tmpGetDeviceIDs;
    int proxyRank, i, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    MPI_Status status[3];
    MPI_Request request[3];
    int requestNo = 0;

    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    /* initialize structure */
    /* convert the vocl platformID to cl platform ID */
    tmpGetDeviceIDs.platform = voclVOCLPlatformID2CLPlatformIDComm((vocl_platform_id) platform,
                                                                   &proxyRank, &proxyIndex,
                                                                   &proxyComm, &proxyCommData);

    /* for GPUs on local node */
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpGetDeviceIDs.res = dlCLGetDeviceIDs(tmpGetDeviceIDs.platform,
                                               device_type, num_entries, devices, num_devices);
    }
    else {
        tmpGetDeviceIDs.device_type = device_type;
        tmpGetDeviceIDs.num_entries = num_entries;
        tmpGetDeviceIDs.devices = devices;

        /* indicate num_device be NOT NULL */
        tmpGetDeviceIDs.num_devices = 1;
        if (num_devices == NULL) {
            tmpGetDeviceIDs.num_devices = 0;
        }
        /* send parameters to remote node */
        MPI_Isend(&tmpGetDeviceIDs, sizeof(tmpGetDeviceIDs), MPI_BYTE, proxyRank,
                  GET_DEVICE_ID_FUNC, proxyComm, request + (requestNo++));

        if (num_entries > 0 && devices != NULL) {
            MPI_Irecv(devices, sizeof(cl_device_id) * num_entries, MPI_BYTE, proxyRank,
                      GET_DEVICE_ID_FUNC1, proxyCommData, request + (requestNo++));
        }
        MPI_Irecv(&tmpGetDeviceIDs, sizeof(tmpGetDeviceIDs), MPI_BYTE, proxyRank,
                  GET_DEVICE_ID_FUNC, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
        if (num_devices != NULL) {
            *num_devices = tmpGetDeviceIDs.num_devices;
        }
    }

    /* convert opencl device id to vocl device id */
    if (num_entries > 0 && devices != NULL) {
        for (i = 0; i < num_entries; i++) {
            devices[i] =
                (cl_device_id) voclCLDeviceID2VOCLDeviceID(devices[i], proxyRank, proxyIndex,
                                                           proxyComm, proxyCommData);
        }
    }

    return tmpGetDeviceIDs.res;
}

cl_context
clCreateContext(const cl_context_properties * properties,
                cl_uint num_devices,
                const cl_device_id * devices,
                void (CL_CALLBACK * pfn_notify) (const char *, const void *, size_t, void *),
                void *user_data, cl_int * errcode_ret)
{
    struct strCreateContext tmpCreateContext;
    MPI_Status status[3];
    MPI_Request request[3];
    vocl_context context;
    cl_device_id *clDevices;
    int proxyRank, i, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    int requestNo = 0;
    int res;

    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    /* initialize structure */
    /* tmpCreateContext.properties = *properties; */
    tmpCreateContext.num_devices = num_devices;
    tmpCreateContext.devices = (cl_device_id *) devices;
    /* convert vocl context to opencl context */
    clDevices = (cl_device_id *) malloc(sizeof(cl_device_id) * num_devices);
    for (i = 0; i < num_devices; i++) {
        clDevices[i] =
            voclVOCLDeviceID2CLDeviceIDComm((vocl_device_id) devices[i], &proxyRank,
                                            &proxyIndex, &proxyComm, &proxyCommData);
    }

    /* local node, call native opencl function directly */
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        dlCLCreateContext(properties, num_devices, clDevices, pfn_notify,
                          user_data, errcode_ret, &tmpCreateContext.hContext);
    }
    else {
        tmpCreateContext.user_data = user_data;
        /* send parameters to remote node */
        res = MPI_Isend(&tmpCreateContext, sizeof(tmpCreateContext), MPI_BYTE, proxyRank,
                        CREATE_CONTEXT_FUNC, proxyComm, request + (requestNo++));

        if (devices != NULL) {
            MPI_Isend((void *) clDevices, sizeof(cl_device_id) * num_devices, MPI_BYTE,
                      proxyRank, CREATE_CONTEXT_FUNC1, proxyCommData, request + (requestNo++));
        }

        MPI_Irecv(&tmpCreateContext, sizeof(tmpCreateContext), MPI_BYTE, proxyRank,
                  CREATE_CONTEXT_FUNC, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
        *errcode_ret = tmpCreateContext.errcode_ret;
    }

    /* increase OpenCL object count */
    voclObjCountIncrease(proxyIndex);

    /*convert opencl context to vocl context */
    context =
        voclCLContext2VOCLContext(tmpCreateContext.hContext, proxyRank, proxyIndex, proxyComm,
                                  proxyCommData);

    /* for the first time a context is created, migration status is 0 */
    voclContextSetMigrationStatus(context, 0);
    free(clDevices);

    return (cl_context) context;
}

/* Command Queue APIs */
cl_command_queue
clCreateCommandQueue(cl_context context,
                     cl_device_id device,
                     cl_command_queue_properties properties, cl_int * errcode_ret)
{
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    struct strCreateCommandQueue tmpCreateCommandQueue;
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
    vocl_command_queue command_queue;
    int proxyRankDevice, proxyRankContext, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;

    tmpCreateCommandQueue.context =
        voclVOCLContext2CLContextComm((vocl_context) context, &proxyRankContext, &proxyIndex,
                                      &proxyComm, &proxyCommData);
    tmpCreateCommandQueue.device =
        voclVOCLDeviceID2CLDeviceIDComm((vocl_device_id) device, &proxyRankDevice, &proxyIndex,
                                        &proxyComm, &proxyCommData);
    if (proxyRankContext != proxyRankDevice) {
        printf("deice and context are on different GPU nodes!\n");
    }


    /* local node, call native opencl function directly */
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        dlCLCreateCommandQueue(tmpCreateCommandQueue.context,
                               tmpCreateCommandQueue.device, properties, errcode_ret,
                               &tmpCreateCommandQueue.clCommand);

        /* store the command queue locally */
        voclLibUpdateCmdQueueOnDeviceID(tmpCreateCommandQueue.device,
                                        tmpCreateCommandQueue.clCommand);

    }
    else {
        tmpCreateCommandQueue.properties = properties;

        /* send parameters to remote node */
        MPI_Isend(&tmpCreateCommandQueue, sizeof(tmpCreateCommandQueue), MPI_BYTE,
                  proxyRankContext, CREATE_COMMAND_QUEUE_FUNC, proxyComm,
                  request + (requestNo++));
        MPI_Irecv(&tmpCreateCommandQueue, sizeof(tmpCreateCommandQueue), MPI_BYTE,
                  proxyRankContext, CREATE_COMMAND_QUEUE_FUNC, proxyComm,
                  request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
        if (errcode_ret != NULL) {
            *errcode_ret = tmpCreateCommandQueue.errcode_ret;
        }
    }

    /* increase OpenCL object count */
    voclObjCountIncrease(proxyIndex);

    /* convert cl command queue to vocl command queue */
    command_queue =
        voclCLCommandQueue2VOCLCommandQueue(tmpCreateCommandQueue.clCommand, proxyRankContext,
                                            proxyIndex, proxyComm, proxyCommData);
    /*set the migration status */
    voclCommandQueueSetMigrationStatus(command_queue,
                                       voclContextGetMigrationStatus((vocl_context) context));
    voclStoreCmdQueueProperties(command_queue, properties, (vocl_context) context,
                                (vocl_device_id) device);

    return (cl_command_queue) command_queue;
}

cl_program
clCreateProgramWithSource(cl_context context,
                          cl_uint count,
                          const char **strings, const size_t * lengths, cl_int * errcode_ret)
{
    struct strCreateProgramWithSource tmpCreateProgramWithSource;
    MPI_Status status[4];
    MPI_Request request[4];
    vocl_program program;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    int requestNo = 0;
    size_t totalLength, *lengthsArray, strStartLoc;
    cl_uint strIndex;
    char *allStrings;

    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    /* initialize structure */
    tmpCreateProgramWithSource.context = voclVOCLContext2CLContextComm((vocl_context) context,
                                                                       &proxyRank, &proxyIndex,
                                                                       &proxyComm,
                                                                       &proxyCommData);

    tmpCreateProgramWithSource.count = count;

    lengthsArray = (size_t *) malloc(count * sizeof(size_t));

    totalLength = 0;
    if (lengths == NULL) {      /* all strings are null-terminated */
        for (strIndex = 0; strIndex < count; strIndex++) {
            lengthsArray[strIndex] = strlen(strings[strIndex]);
            totalLength += lengthsArray[strIndex];
        }
    }
    else {
        for (strIndex = 0; strIndex < count; strIndex++) {
            if (lengths[strIndex] == 0) {
                lengthsArray[strIndex] = strlen(strings[strIndex]);
                totalLength += lengthsArray[strIndex];
            }
            else {
                lengthsArray[strIndex] = lengths[strIndex];
                totalLength += lengthsArray[strIndex];
            }
        }
    }
    allStrings = (char *) malloc(totalLength * sizeof(char));

    strStartLoc = 0;
    for (strIndex = 0; strIndex < count; strIndex++) {
        memcpy(&allStrings[strStartLoc], strings[strIndex],
               sizeof(char) * lengthsArray[strIndex]);
        strStartLoc += lengthsArray[strIndex];
    }

    tmpCreateProgramWithSource.lengths = totalLength;

    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        dlCLCreateProgramWithSource(tmpCreateProgramWithSource.context,
                                    count, strings, lengths, errcode_ret,
                                    &tmpCreateProgramWithSource.clProgram);
    }
    else {
        /* send parameters to remote node */
        MPI_Isend(&tmpCreateProgramWithSource,
                  sizeof(tmpCreateProgramWithSource),
                  MPI_BYTE, proxyRank, CREATE_PROGRMA_WITH_SOURCE, proxyComm,
                  request + (requestNo++));
        MPI_Isend(lengthsArray, sizeof(size_t) * count, MPI_BYTE, proxyRank,
                  CREATE_PROGRMA_WITH_SOURCE1, proxyCommData, request + (requestNo++));
        MPI_Isend((void *) allStrings, totalLength * sizeof(char), MPI_BYTE, proxyRank,
                  CREATE_PROGRMA_WITH_SOURCE2, proxyCommData, request + (requestNo++));
        MPI_Irecv(&tmpCreateProgramWithSource, sizeof(tmpCreateProgramWithSource), MPI_BYTE,
                  proxyRank, CREATE_PROGRMA_WITH_SOURCE, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
        if (errcode_ret != NULL) {
            *errcode_ret = tmpCreateProgramWithSource.errcode_ret;
        }
    }

    /* increase OpenCL object count */
    voclObjCountIncrease(proxyIndex);

    /* convert opencl program to vocl program */
    program = voclCLProgram2VOCLProgram(tmpCreateProgramWithSource.clProgram,
                                        proxyRank, proxyIndex, proxyComm, proxyCommData);
    voclProgramSetMigrationStatus(program,
                                  voclContextGetMigrationStatus((vocl_context) context));

    /*store the source code corresponding to the program */
    voclStoreProgramSource(program, allStrings, totalLength);
    voclStoreProgramContext(program, (vocl_context) context);

    free(allStrings);
    free(lengthsArray);

    return (cl_program) program;
}

cl_int
clBuildProgram(cl_program program,
               cl_uint num_devices,
               const cl_device_id * device_list,
               const char *options,
               void (CL_CALLBACK * pfn_notify) (cl_program program, void *user_data),
               void *user_data)
{
    int optionsLen = 0;
    struct strBuildProgram tmpBuildProgram;
    MPI_Status status[4];
    MPI_Request request[4];
    cl_device_id *clDevices = NULL;
    int proxyRank, i, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    int requestNo = 0;

    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    if (options != NULL) {
        optionsLen = strlen(options);
    }

    /* initialize structure */
    tmpBuildProgram.program = voclVOCLProgram2CLProgramComm((vocl_program) program,
                                                            &proxyRank, &proxyIndex,
                                                            &proxyComm, &proxyCommData);

    if (device_list != NULL) {
        clDevices = (cl_device_id *) malloc(num_devices * sizeof(cl_device_id));
        for (i = 0; i < num_devices; i++) {
            clDevices[i] = voclVOCLDeviceID2CLDeviceIDComm((vocl_device_id) device_list[i],
                                                           &proxyRank, &proxyIndex, &proxyComm,
                                                           &proxyCommData);
        }
    }

    /* local node, call native opencl function directly */
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpBuildProgram.res = dlCLBuildProgram(tmpBuildProgram.program, num_devices, clDevices,
                                               options, pfn_notify, user_data);
    }
    else {
        tmpBuildProgram.num_devices = num_devices;
        tmpBuildProgram.device_list = (cl_device_id *) device_list;
        tmpBuildProgram.optionLen = optionsLen;

        /* send parameters to remote node */
        MPI_Isend(&tmpBuildProgram, sizeof(tmpBuildProgram), MPI_BYTE, proxyRank,
                  BUILD_PROGRAM, proxyComm, request + (requestNo++));
        if (optionsLen > 0) {
            MPI_Isend((void *) options, optionsLen, MPI_BYTE, proxyRank, BUILD_PROGRAM1,
                      proxyCommData, request + (requestNo++));
        }
        if (device_list != NULL) {
            MPI_Isend((void *) clDevices, sizeof(cl_device_id) * num_devices, MPI_BYTE,
                      proxyRank, BUILD_PROGRAM, proxyCommData, request + (requestNo++));
        }

        MPI_Irecv(&tmpBuildProgram, sizeof(tmpBuildProgram), MPI_BYTE, proxyRank,
                  BUILD_PROGRAM, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
    }

    if (device_list != NULL) {
        free(clDevices);
    }

    return tmpBuildProgram.res;
}

cl_kernel clCreateKernel(cl_program program, const char *kernel_name, cl_int * errcode_ret)
{
    MPI_Status status[3];
    MPI_Request request[3];
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    vocl_kernel kernel;
    vocl_context context;
    int requestNo = 0;
    struct strCreateKernel tmpCreateKernel;
    int kernelNameSize = strlen(kernel_name);

    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    tmpCreateKernel.program = voclVOCLProgram2CLProgramComm((vocl_program) program,
                                                            &proxyRank, &proxyIndex,
                                                            &proxyComm, &proxyCommData);

    /* local node, call native opencl function directly */
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        dlCLCreateKernel(tmpCreateKernel.program, kernel_name, errcode_ret,
                         &tmpCreateKernel.kernel);
    }
    else {
        tmpCreateKernel.kernelNameSize = kernelNameSize;

        /* send input parameters to remote node */
        MPI_Isend(&tmpCreateKernel, sizeof(tmpCreateKernel), MPI_BYTE, proxyRank,
                  CREATE_KERNEL, proxyComm, request + (requestNo++));
        MPI_Isend((void *) kernel_name, kernelNameSize, MPI_CHAR, proxyRank, CREATE_KERNEL1,
                  proxyCommData, request + (requestNo++));
        MPI_Irecv(&tmpCreateKernel, sizeof(tmpCreateKernel), MPI_BYTE, proxyRank,
                  CREATE_KERNEL, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
        if (errcode_ret != NULL) {
            *errcode_ret = tmpCreateKernel.errcode_ret;
        }
    }

    /* increase OpenCL object count */
    voclObjCountIncrease(proxyIndex);

    /*convert opencl kernel to vocl kernel */
    kernel =
        voclCLKernel2VOCLKernel(tmpCreateKernel.kernel, proxyRank, proxyIndex, proxyComm,
                                proxyCommData);
    voclStoreKernelName(kernel, (char *) kernel_name);
    voclKernelSetMigrationStatus(kernel,
                                 voclProgramGetMigrationStatus((vocl_program) program));

    /* get context from the vocl program */
    context = voclGetContextFromProgram((vocl_program) program);
    voclStoreKernelProgramContext(kernel, (vocl_program) program, context);

    /* create kernel info on the local node for storing arguments */
    createKernel((cl_kernel) kernel);
    createKernelArgInfo((cl_kernel) kernel, (char *) kernel_name, (vocl_program) program);


    return (cl_kernel) kernel;
}

/* Memory Object APIs */
cl_mem
clCreateBuffer(cl_context context,
               cl_mem_flags flags, size_t size, void *host_ptr, cl_int * errcode_ret)
{
    struct strCreateBuffer tmpCreateBuffer;
    MPI_Status status[3];
    MPI_Request request[3];
    vocl_mem memory;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    int requestNo = 0;

    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    /* initialize structure */
    tmpCreateBuffer.context = voclVOCLContext2CLContextComm((vocl_context) context,
                                                            &proxyRank, &proxyIndex,
                                                            &proxyComm, &proxyCommData);

    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        dlCLCreateBuffer(tmpCreateBuffer.context, flags, size, host_ptr, errcode_ret,
                         &tmpCreateBuffer.deviceMem);
    }
    else {
        tmpCreateBuffer.flags = flags;
        tmpCreateBuffer.size = size;
        tmpCreateBuffer.host_ptr_flag = 0;
        if (host_ptr != NULL) {
            tmpCreateBuffer.host_ptr_flag = 1;
        }

        /* send parameters to remote node */
        MPI_Isend(&tmpCreateBuffer, sizeof(tmpCreateBuffer), MPI_BYTE, proxyRank,
                  CREATE_BUFFER_FUNC, proxyComm, request + (requestNo++));
        if (tmpCreateBuffer.host_ptr_flag == 1) {
            MPI_Isend(host_ptr, size, MPI_BYTE, proxyRank, CREATE_BUFFER_FUNC1, proxyCommData,
                      request + (requestNo++));
        }
        MPI_Irecv(&tmpCreateBuffer, sizeof(tmpCreateBuffer), MPI_BYTE, proxyRank,
                  CREATE_BUFFER_FUNC, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
        if (errcode_ret != NULL) {
            *errcode_ret = tmpCreateBuffer.errcode_ret;
        }
    }

    /* increase OpenCL object count */
    voclObjCountIncrease(proxyIndex);

    memory = voclCLMemory2VOCLMemory(tmpCreateBuffer.deviceMem,
                                     proxyRank, proxyIndex, proxyComm, proxyCommData);
    voclMemSetMigrationStatus(memory, voclContextGetMigrationStatus((vocl_context) context));
    /* store memory parameters for possible migration */
    voclStoreMemoryParameters(memory, flags, size, (vocl_context) context);

    return (cl_mem) memory;
}

cl_int
clEnqueueWriteBuffer(cl_command_queue command_queue,
                     cl_mem buffer,
                     cl_bool blocking_write,
                     size_t offset,
                     size_t cb,
                     const void *ptr,
                     cl_uint num_events_in_wait_list,
                     const cl_event * event_wait_list, cl_event * event)
{
    struct strEnqueueWriteBuffer tmpEnqueueWriteBuffer;
    MPI_Status status[4];
    MPI_Request request[4];
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    int requestNo = 0, errCode;
    int bufferNum, i, bufferIndex;
    int rankNo;
    size_t remainingSize, bufferSize;
    vocl_event voclEvent;
    cl_event *eventList = NULL;
    /* for migration check */
    int isMigrationNeeded = 0;
    struct strMigrationCheck tmpMigrationCheck;

    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

//    tmpEnqueueWriteBuffer.command_queue =
//        voclVOCLCommandQueue2CLCommandQueueComm((vocl_command_queue) command_queue, &proxyRank,
//                                                &proxyIndex, &proxyComm, &proxyCommData);
//    /* initialize structure */
//    tmpEnqueueWriteBuffer.buffer =
//        voclVOCLMemory2CLMemoryComm((vocl_mem) buffer, &proxyRank, &proxyIndex, &proxyComm,
//                                    &proxyCommData);
//      /* check migration */
//      if (voclMemGetMigrationStatus((vocl_mem)buffer) <
//              voclCommandQueueGetMigrationStatus((vocl_command_queue)command_queue) &&
//              voclGetTaskMigrationCondition() != 0)
//      {
//              voclUpdateSingleMemory((vocl_mem)buffer);
//      }
//      else if (voclGetTaskMigrationCondition() != 0)
//      {
//              /* it is a local GPU */
//              if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE)
//              {
//                      isMigrationNeeded = voclCheckMigrationInWriteBuffer(command_queue, cb);
//              }
//              else
//              {
//                      /* migration check is in enqueue write buffer */
//                      MPI_Comm_rank(MPI_COMM_WORLD, &rankNo);
//                      tmpMigrationCheck.command_queue = tmpEnqueueWriteBuffer.command_queue;
//                      tmpMigrationCheck.rankNo = rankNo;
//                      tmpMigrationCheck.checkLocation = 1;
//                      tmpMigrationCheck.memSize = cb;
//            MPI_Isend(&tmpMigrationCheck, sizeof(struct strMigrationCheck), MPI_BYTE,
//                               proxyRank, MIGRATION_CHECK, proxyComm, request + (requestNo++));
//            MPI_Irecv(&tmpMigrationCheck, sizeof(struct strMigrationCheck), MPI_BYTE,
//                               proxyRank, MIGRATION_CHECK, proxyComm, request + (requestNo++));
//                      MPI_Waitall(requestNo, request, status);
//                      for (i = 0; i < requestNo; i++)
//                      {
//                              if (status[i].MPI_ERROR != MPI_SUCCESS)
//                              {
//                                      printf("errcode = %d\n", status[i].MPI_ERROR);
//                              }
//                      }
//
//                      requestNo = 0;
//                      isMigrationNeeded = tmpMigrationCheck.isMigrationNeeded;
//              }
//
//              if (isMigrationNeeded == 1)
//              {
//                      /* migrate previous device memory */
//                      voclCommandQueueMigration((vocl_command_queue)command_queue);
//                      /* migrate the current device memory */
//                      voclUpdateSingleMemory((vocl_mem)buffer);
//              }
//      }

    /* if it is not in the command queue, add it there */
    voclUpdateMemoryInCommandQueue((vocl_command_queue) command_queue, (vocl_mem) buffer, cb);

    /* set memory write state for possible migration */
    voclSetMemWrittenFlag((vocl_mem) buffer, 1);
    /* save the host memory pointer for possible migration */
    voclSetMemHostPtr((vocl_mem) buffer, (void *) ptr);

    if (num_events_in_wait_list > 0) {
        eventList = (cl_event *) malloc(sizeof(cl_event) * num_events_in_wait_list);
        if (eventList == NULL) {
            printf("enqueueWriteBuffer, allocate eventList error!\n");
        }

        /* convert vocl events to opencl events */
        voclVOCLEvents2CLEvents((vocl_event *) event_wait_list, eventList,
                                num_events_in_wait_list);
    }

    tmpEnqueueWriteBuffer.command_queue =
        voclVOCLCommandQueue2CLCommandQueueComm((vocl_command_queue) command_queue, &proxyRank,
                                                &proxyIndex, &proxyComm, &proxyCommData);

    tmpEnqueueWriteBuffer.buffer =
        voclVOCLMemory2CLMemoryComm((vocl_mem) buffer, &proxyRank, &proxyIndex, &proxyComm,
                                    &proxyCommData);

    /* local GPU, call native opencl function */
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        errCode = dlCLEnqueueWriteBuffer(tmpEnqueueWriteBuffer.command_queue,
                                         tmpEnqueueWriteBuffer.buffer, blocking_write, offset,
                                         cb, ptr, num_events_in_wait_list, eventList, event);
        voclLibUpdateGlobalMemOnCommandQueue(tmpEnqueueWriteBuffer.command_queue,
                                             tmpEnqueueWriteBuffer.buffer, cb);
        if (event != NULL) {
            /* convert to vocl event */
            *event = (cl_event) voclCLEvent2VOCLEvent((*event),
                                                      proxyRank, proxyIndex, proxyComm,
                                                      proxyCommData);
        }

        if (num_events_in_wait_list > 0) {
            free(eventList);
        }

        return errCode;
    }

    tmpEnqueueWriteBuffer.blocking_write = blocking_write;
    tmpEnqueueWriteBuffer.offset = offset;
    tmpEnqueueWriteBuffer.cb = cb;
    tmpEnqueueWriteBuffer.tag = ENQUEUE_WRITE_BUFFER1;
    tmpEnqueueWriteBuffer.num_events_in_wait_list = num_events_in_wait_list;
    if (event == NULL) {
        tmpEnqueueWriteBuffer.event_null_flag = 1;
    }
    else {
        tmpEnqueueWriteBuffer.event_null_flag = 0;
    }

    /* send parameters to remote node */
    MPI_Isend(&tmpEnqueueWriteBuffer, sizeof(struct strEnqueueWriteBuffer), MPI_BYTE,
              proxyRank, ENQUEUE_WRITE_BUFFER, proxyComm, request + (requestNo++));

    if (num_events_in_wait_list > 0) {
        MPI_Isend((void *) eventList, sizeof(cl_event) * num_events_in_wait_list,
                  MPI_BYTE, proxyRank, tmpEnqueueWriteBuffer.tag, proxyCommData,
                  request + (requestNo++));
    }

    bufferNum = (cb - 1) / VOCL_WRITE_BUFFER_SIZE;
    bufferSize = VOCL_WRITE_BUFFER_SIZE;
    remainingSize = cb - bufferNum * bufferSize;
    for (i = 0; i <= bufferNum; i++) {
        bufferIndex = getNextWriteBufferIndex(proxyIndex);
        if (i == bufferNum) {
            bufferSize = remainingSize;
        }

        MPI_Isend((void *) ((char *) ptr + i * VOCL_WRITE_BUFFER_SIZE), bufferSize, MPI_BYTE,
                  proxyRank, VOCL_WRITE_TAG + bufferIndex, proxyCommData,
                  getWriteRequestPtr(proxyIndex, bufferIndex));
        /* current buffer is used */
        setWriteBufferInUse(proxyIndex, bufferIndex);
    }
    setWriteBufferNum(proxyIndex, bufferIndex, bufferNum + 1);

    if (blocking_write == CL_TRUE || event != NULL) {
        MPI_Irecv(&tmpEnqueueWriteBuffer, sizeof(struct strEnqueueWriteBuffer), MPI_BYTE,
                  proxyRank, ENQUEUE_WRITE_BUFFER, proxyComm, request + (requestNo++));
        /* for a blocking write, process all previous non-blocking ones */
        if (blocking_write == CL_TRUE) {
            voclSetMemWrittenFlag((vocl_mem) buffer, 2);        /* memory write is completed */
            processAllWrites(proxyIndex);
        }
        else if (event != NULL) {
            processWriteBuffer(proxyIndex, bufferIndex, bufferNum + 1);
        }

        MPI_Waitall(requestNo, request, status);
        if (event != NULL) {
            /* covert opencl event to vocl event */
            voclEvent =
                voclCLEvent2VOCLEvent(tmpEnqueueWriteBuffer.event, proxyRank, proxyIndex,
                                      proxyComm, proxyCommData);
            setWriteBufferEvent(proxyIndex, bufferIndex, voclEvent);
            *event = (cl_event) voclEvent;
        }
        return tmpEnqueueWriteBuffer.res;
    }

    /* delete buffer for vocl events */
    if (num_events_in_wait_list > 0) {
        free(eventList);
    }

    MPI_Waitall(requestNo, request, status);

    return CL_SUCCESS;
}

cl_int
clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value)
{
    cl_mem deviceMem;
    int proxyRank, proxyIndex;
    int kernelMigStatus, memMigStatus;
    MPI_Comm proxyComm, proxyCommData;
    size_t size;
    cl_kernel clKernel =
        voclVOCLKernel2CLKernelComm((vocl_kernel) kernel, &proxyRank, &proxyIndex,
                                    &proxyComm, &proxyCommData);
    kernelMigStatus = voclKernelGetMigrationStatus((vocl_kernel) kernel);
    kernel_info *kernelPtr = getKernelPtr(kernel);

    /* if argument buffer is not enough, extend it */
    if (kernelPtr->args_num >= kernelPtr->maxArgNum) {
        kernelPtr->maxArgNum *= 2;
        kernelPtr->args_ptr =
            (kernel_args *) realloc(kernelPtr->args_ptr,
                                    sizeof(kernel_args) * kernelPtr->maxArgNum);
    }

    kernelPtr->args_ptr[kernelPtr->args_num].arg_index = arg_index;
    kernelPtr->args_ptr[kernelPtr->args_num].arg_size = arg_size;
    kernelPtr->args_ptr[kernelPtr->args_num].arg_null_flag = 1;

    if (arg_index >= kernelPtr->kernel_arg_num) {
        printf("arg_index %d is larger than arg_num %d\n", arg_index,
               kernelPtr->kernel_arg_num);
        exit(1);
    }

    kernelPtr->args_ptr[kernelPtr->args_num].isGlobalMemory = 0;
    if (arg_value != NULL) {
        if (kernelPtr->args_flag[arg_index] == 1) {     /* device memory */
            /*convert from vocl memory to cl memory */
            size = voclGetVOCLMemorySize(*((vocl_mem *) arg_value));
            kernelPtr->globalMemSize += size;
            kernelPtr->args_ptr[kernelPtr->args_num].memory =
                (cl_mem) (*((vocl_mem *) arg_value));
            deviceMem = voclVOCLMemory2CLMemory(*((vocl_mem *) arg_value));
            memMigStatus = voclMemGetMigrationStatus(*((vocl_mem *) arg_value));

            memcpy(kernelPtr->args_ptr[kernelPtr->args_num].arg_value, (void *) &deviceMem,
                   arg_size);
            /* record if it is a global memory and the size */
            kernelPtr->args_ptr[kernelPtr->args_num].isGlobalMemory = 1;
            kernelPtr->args_ptr[kernelPtr->args_num].globalSize = size;
        }
        else {
            memcpy(kernelPtr->args_ptr[kernelPtr->args_num].arg_value, arg_value, arg_size);
        }
        kernelPtr->args_ptr[kernelPtr->args_num].arg_null_flag = 0;
    }
    kernelPtr->args_num++;

    /* local gpu, call native opencl function */
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        if (kernelPtr->args_flag[arg_index] == 1) {     /*device memory */
            /* only if no migration happened, set the argument */
            if (kernelMigStatus == memMigStatus) {
                /* add gpu memory usage */
                return dlCLSetKernelArg(clKernel, arg_index, arg_size, (void *) &deviceMem);
            }
        }
        else {
            return dlCLSetKernelArg(clKernel, arg_index, arg_size, arg_value);
        }
    }

    return 0;
}

cl_int
clEnqueueNDRangeKernel(cl_command_queue command_queue,
                       cl_kernel kernel,
                       cl_uint work_dim,
                       const size_t * global_work_offset,
                       const size_t * global_work_size,
                       const size_t * local_work_size,
                       cl_uint num_events_in_wait_list,
                       const cl_event * event_wait_list, cl_event * event)
{
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    struct strEnqueueNDRangeKernel tmpEnqueueNDRangeKernel;
    struct strEnqueueNDRangeKernelReply kernelLaunchReply;
    struct strMigrationCheck tmpMigrationCheck;
    MPI_Status status[7];
    MPI_Request request[7];
    int requestNo = 0;
    int proxyRank, proxyIndex, i, rankNo;
    int cmdQueueMigStatus, kernelMigStatus;
    MPI_Comm proxyComm, proxyCommData;
    cl_event *eventList = NULL;
    int taskMigrationCheck, isMigrationNeeded = 0;
    char *msgBuffer;
    size_t msgSize, paramOffset = 0;

    msgSize = 2048;
    msgBuffer = (char *) malloc(sizeof(char) * msgSize);

    /* initialize structure */
    kernel_info *kernelPtr = getKernelPtr(kernel);
    /* if no new arguments set for the current kernel, which means */
    /* either no global memory is used or the same set of arguments */
    /* are used as previous kernel launch, so no migration check is needed */
    cmdQueueMigStatus = voclCommandQueueGetMigrationStatus((vocl_command_queue) command_queue);
    if (kernelPtr->args_num > 0 && voclGetTaskMigrationCondition() != 0 &&
        cmdQueueMigStatus == 0) {
        taskMigrationCheck = 1;
    }
    else {
        taskMigrationCheck = 0;
    }

    //debug-------------------------
    //voclLBGetDeviceCmdQueueNums();
    //--------------------------------

    /*check to see whether migration is needed based on GPU memory usage */
    /*GPU memory usage information can be obtained based on kernel arguments */
    /* since the remote GPU can be shared by multiple proxy process, we need */
    /* the help of the proxy process */
//      kernelMigStatus = voclKernelGetMigrationStatus((vocl_kernel)kernel);
//      if (kernelMigStatus < cmdQueueMigStatus && voclGetTaskMigrationCondition() != 0)
//      {
//              voclTaskMigration(kernel, command_queue);
//      }

    if (taskMigrationCheck == 1) {
        tmpMigrationCheck.command_queue =
            voclVOCLCommandQueue2CLCommandQueueComm((vocl_command_queue) command_queue,
                                                    &proxyRank, &proxyIndex, &proxyComm,
                                                    &proxyCommData);
        if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
            isMigrationNeeded =
                voclCheckMigrationInKernelLaunch(tmpMigrationCheck.command_queue,
                                                 kernelPtr->args_ptr, kernelPtr->args_num);
        }
        else {
            /* migration check is requested in kernel launch */
            MPI_Comm_rank(MPI_COMM_WORLD, &rankNo);
            tmpMigrationCheck.checkLocation = 0;
            tmpMigrationCheck.rankNo = rankNo;
            tmpMigrationCheck.argsNum = kernelPtr->args_num;
            requestNo = 0;
            MPI_Isend(&tmpMigrationCheck, sizeof(struct strMigrationCheck), MPI_BYTE,
                      proxyRank, MIGRATION_CHECK, proxyComm, request + (requestNo++));

            MPI_Isend((void *) kernelPtr->args_ptr, sizeof(kernel_args) * kernelPtr->args_num,
                      MPI_BYTE, proxyRank, MIGRATION_CHECK, proxyCommData,
                      request + (requestNo++));

            MPI_Irecv(&tmpMigrationCheck, sizeof(struct strMigrationCheck), MPI_BYTE,
                      proxyRank, MIGRATION_CHECK, proxyComm, request + (requestNo++));
            MPI_Waitall(requestNo, request, status);
            isMigrationNeeded = tmpMigrationCheck.isMigrationNeeded;
        }

        //debug, for manually triggering migration
        if (isMigrationNeeded == 1) {
            voclTaskMigration(kernel, command_queue);
        }
    }

    tmpEnqueueNDRangeKernel.kernel =
        voclVOCLKernel2CLKernelComm((vocl_kernel) kernel, &proxyRank, &proxyIndex, &proxyComm,
                                    &proxyCommData);
    tmpEnqueueNDRangeKernel.command_queue =
        voclVOCLCommandQueue2CLCommandQueueComm((vocl_command_queue) command_queue, &proxyRank,
                                                &proxyIndex, &proxyComm, &proxyCommData);

    if (num_events_in_wait_list > 0) {
        eventList = (cl_event *) malloc(sizeof(cl_event) * num_events_in_wait_list);
        if (eventList == NULL) {
            printf("enqueueNDRangeKernel, allocate eventList error!\n");
        }

        /* convert vocl events to opencl events */
        voclVOCLEvents2CLEvents((vocl_event *) event_wait_list, eventList,
                                num_events_in_wait_list);
    }
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        kernelLaunchReply.res =
            dlCLEnqueueNDRangeKernel(tmpEnqueueNDRangeKernel.command_queue,
                                     tmpEnqueueNDRangeKernel.kernel, work_dim,
                                     global_work_offset, global_work_size, local_work_size,
                                     num_events_in_wait_list, eventList, event);
        if (kernelLaunchReply.res != CL_SUCCESS) {
            printf("Kernel launch error, %d\n", kernelLaunchReply.res);
        }
        /* store the global memory usage info */
        voclLibUpdateGlobalMemUsage(tmpEnqueueNDRangeKernel.command_queue,
                                    kernelPtr->args_ptr, kernelPtr->args_num);
    }
    else {
        tmpEnqueueNDRangeKernel.work_dim = work_dim;
        tmpEnqueueNDRangeKernel.num_events_in_wait_list = num_events_in_wait_list;
        tmpEnqueueNDRangeKernel.global_work_offset_flag = 0;
        tmpEnqueueNDRangeKernel.global_work_size_flag = 0;
        tmpEnqueueNDRangeKernel.local_work_size_flag = 0;
        if (global_work_offset != NULL) {
            tmpEnqueueNDRangeKernel.global_work_offset_flag = 1;
        }
        if (global_work_size != NULL) {
            tmpEnqueueNDRangeKernel.global_work_size_flag = 1;
        }
        if (local_work_size != NULL) {
            tmpEnqueueNDRangeKernel.local_work_size_flag = 1;
        }
        tmpEnqueueNDRangeKernel.args_num = kernelPtr->args_num;
        if (event == NULL) {
            tmpEnqueueNDRangeKernel.event_null_flag = 1;
        }
        else {
            tmpEnqueueNDRangeKernel.event_null_flag = 0;
        }

        paramOffset = 0;
        if (paramOffset + sizeof(size_t) * work_dim * 3 > msgSize) {
            msgSize = paramOffset + sizeof(size_t) * work_dim * 3;
            msgBuffer = (char *) realloc(msgBuffer, msgSize);
        }

        if (tmpEnqueueNDRangeKernel.global_work_offset_flag == 1) {
            memcpy((void *) (msgBuffer + paramOffset), (void *) global_work_offset,
                   sizeof(size_t) * work_dim);
            paramOffset += sizeof(size_t) * work_dim;
        }

        if (tmpEnqueueNDRangeKernel.global_work_size_flag == 1) {
            memcpy((void *) (msgBuffer + paramOffset), (void *) global_work_size,
                   sizeof(size_t) * work_dim);
            paramOffset += sizeof(size_t) * work_dim;
        }

        if (tmpEnqueueNDRangeKernel.local_work_size_flag == 1) {
            memcpy((void *) (msgBuffer + paramOffset), (void *) local_work_size,
                   sizeof(size_t) * work_dim);
            paramOffset += sizeof(size_t) * work_dim;
        }

        if (kernelPtr->args_num > 0) {
            if (paramOffset + sizeof(kernel_args) * kernelPtr->args_num > msgSize) {
                msgSize = paramOffset + sizeof(kernel_args) * kernelPtr->args_num + 1;
                msgBuffer = (char *) realloc(msgBuffer, msgSize);
            }
            memcpy((void *) (msgBuffer + paramOffset), (void *) kernelPtr->args_ptr,
                   sizeof(kernel_args) * kernelPtr->args_num);
            paramOffset += (sizeof(kernel_args) * kernelPtr->args_num);
        }
        /* arguments for current call are processed */
        kernelPtr->args_num = 0;
        tmpEnqueueNDRangeKernel.dataSize = paramOffset;

        /* send parameters to remote node */
        MPI_Isend(&tmpEnqueueNDRangeKernel, sizeof(tmpEnqueueNDRangeKernel), MPI_BYTE,
                  proxyRank, ENQUEUE_ND_RANGE_KERNEL, proxyComm, request + (requestNo++));

        if (num_events_in_wait_list > 0) {
            MPI_Isend((void *) eventList, sizeof(cl_event) * num_events_in_wait_list,
                      MPI_BYTE, proxyRank, ENQUEUE_ND_RANGE_KERNEL1, proxyCommData,
                      request + (requestNo++));
        }

        if (paramOffset > 0) {
            MPI_Isend((void *) msgBuffer, paramOffset, MPI_BYTE, proxyRank,
                      ENQUEUE_ND_RANGE_KERNEL1, proxyCommData, request + (requestNo++));
        }

        kernelLaunchReply.res = CL_SUCCESS;

        MPI_Irecv(&kernelLaunchReply, sizeof(struct strEnqueueNDRangeKernelReply), MPI_BYTE,
                  proxyRank, ENQUEUE_ND_RANGE_KERNEL, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
        for (i = 0; i < requestNo; i++) {
            if (status[i].MPI_ERROR != MPI_SUCCESS) {
                printf("errcode = %d\n", status[i].MPI_ERROR);
            }
        }
    }

    if (event != NULL) {
        /* covert opencl event to vocl event to be stored */
        *event = (cl_event) voclCLEvent2VOCLEvent(kernelLaunchReply.event,
                                                  proxyRank, proxyIndex, proxyComm,
                                                  proxyCommData);
    }

    /* delete buffer for vocl events */
    if (num_events_in_wait_list > 0) {
        free(eventList);
    }

    free(msgBuffer);

    return kernelLaunchReply.res;
}

/* Enqueued Commands for GPU memory read */
cl_int
clEnqueueReadBuffer(cl_command_queue command_queue,
                    cl_mem buffer,
                    cl_bool blocking_read,
                    size_t offset,
                    size_t cb,
                    void *ptr,
                    cl_uint num_events_in_wait_list,
                    const cl_event * event_wait_list, cl_event * event)
{
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    MPI_Request request[4], dataRequest;
    MPI_Status status[4];
    struct strEnqueueReadBuffer tmpEnqueueReadBuffer;
    int tempTag, errCode;
    int requestNo = 0;
    int i, bufferIndex, bufferNum;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    size_t bufferSize = VOCL_READ_BUFFER_SIZE, remainingSize;
    vocl_event voclEvent;
    cl_event *eventList = NULL;

    if (num_events_in_wait_list > 0) {
        eventList = (cl_event *) malloc(sizeof(cl_event) * num_events_in_wait_list);
        if (eventList == NULL) {
            printf("enqueueReadBuffer, allocate eventList error!\n");
        }

        /* convert vocl events to opencl events */
        voclVOCLEvents2CLEvents((vocl_event *) event_wait_list, eventList,
                                num_events_in_wait_list);
    }

    bufferNum = (cb - 1) / bufferSize;
    remainingSize = cb - bufferNum * bufferSize;

    /* initialize structure */
    tmpEnqueueReadBuffer.command_queue =
        voclVOCLCommandQueue2CLCommandQueueComm((vocl_command_queue) command_queue, &proxyRank,
                                                &proxyIndex, &proxyComm, &proxyCommData);
    tmpEnqueueReadBuffer.buffer =
        voclVOCLMemory2CLMemoryComm((vocl_mem) buffer, &proxyRank, &proxyIndex, &proxyComm,
                                    &proxyCommData);
    voclUpdateMemoryInCommandQueue((vocl_command_queue) command_queue, (vocl_mem) buffer, cb);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        errCode =
            dlCLEnqueueReadBuffer(tmpEnqueueReadBuffer.command_queue,
                                  tmpEnqueueReadBuffer.buffer, blocking_read, offset, cb, ptr,
                                  num_events_in_wait_list, eventList, event);
        voclLibUpdateGlobalMemOnCommandQueue(tmpEnqueueReadBuffer.command_queue,
                                             tmpEnqueueReadBuffer.buffer, cb);
        if (event != NULL) {
            *event =
                (cl_event) voclCLEvent2VOCLEvent((cl_event) (*event), proxyRank, proxyIndex,
                                                 proxyComm, proxyCommData);
        }

        if (num_events_in_wait_list > 0) {
            free(eventList);
        }

        return errCode;
    }

    tmpEnqueueReadBuffer.blocking_read = blocking_read;
    tmpEnqueueReadBuffer.readBufferTag = ENQUEUE_READ_BUFFER1;
    tmpEnqueueReadBuffer.offset = offset;
    tmpEnqueueReadBuffer.cb = cb;
    tmpEnqueueReadBuffer.num_events_in_wait_list = num_events_in_wait_list;
    if (event == NULL) {
        tmpEnqueueReadBuffer.event_null_flag = 1;
    }
    else {
        tmpEnqueueReadBuffer.event_null_flag = 0;
    }

    /* send parameters to remote node */
    MPI_Isend(&tmpEnqueueReadBuffer, sizeof(struct strEnqueueReadBuffer), MPI_BYTE, proxyRank,
              ENQUEUE_READ_BUFFER, proxyComm, request + (requestNo++));
    if (num_events_in_wait_list > 0) {
        MPI_Isend((void *) eventList, sizeof(cl_event) * num_events_in_wait_list,
                  MPI_BYTE, proxyRank, ENQUEUE_READ_BUFFER1, proxyCommData,
                  request + (requestNo++));
    }
    MPI_Waitall(requestNo, request, status);

    /* delete buffer for vocl events */
    if (num_events_in_wait_list > 0) {
        free(eventList);
    }

    /* receive all data */
    for (i = 0; i <= bufferNum; i++) {
        if (i == bufferNum) {
            bufferSize = remainingSize;
        }
        bufferIndex = getNextReadBufferIndex(proxyIndex);
        MPI_Irecv((void *) ((char *) ptr + VOCL_READ_BUFFER_SIZE * i), bufferSize, MPI_BYTE,
                  proxyRank, VOCL_READ_TAG + bufferIndex, proxyCommData,
                  getReadRequestPtr(proxyIndex, bufferIndex));
        setReadBufferInUse(proxyIndex, bufferIndex);
    }
    setReadBufferNum(proxyIndex, bufferIndex, bufferNum + 1);

    requestNo = 0;
    if (blocking_read == CL_TRUE || event != NULL) {
        MPI_Irecv(&tmpEnqueueReadBuffer, sizeof(struct strEnqueueReadBuffer), MPI_BYTE,
                  proxyRank, ENQUEUE_READ_BUFFER, proxyComm, request + (requestNo++));
    }

    if (blocking_read == CL_TRUE) {
        processAllReads(proxyIndex);
        MPI_Waitall(requestNo, request, status);
        if (event != NULL) {
            voclEvent = voclCLEvent2VOCLEvent(tmpEnqueueReadBuffer.event,
                                              proxyRank, proxyIndex, proxyComm, proxyCommData);
            setReadBufferEvent(proxyIndex, bufferIndex, voclEvent);
            *event = (cl_event) voclEvent;
        }
        return tmpEnqueueReadBuffer.res;
    }
    else {
        MPI_Waitall(requestNo, request, status);
        if (event != NULL) {
            voclEvent = voclCLEvent2VOCLEvent(tmpEnqueueReadBuffer.event,
                                              proxyRank, proxyIndex, proxyComm, proxyCommData);
            setReadBufferEvent(proxyIndex, bufferIndex, voclEvent);
            *event = (cl_event) voclEvent;
            return tmpEnqueueReadBuffer.res;
        }
        else {
            return CL_SUCCESS;
        }
    }
}

cl_int clReleaseMemObject(cl_mem memobj)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;

    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    struct strReleaseMemObject tmpReleaseMemObject;
    tmpReleaseMemObject.memobj = voclVOCLMemory2CLMemoryComm((vocl_mem) memobj,
                                                             &proxyRank, &proxyIndex,
                                                             &proxyComm, &proxyCommData);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        voclLibReleaseMem(tmpReleaseMemObject.memobj);
        tmpReleaseMemObject.res = dlCLReleaseMemObject(tmpReleaseMemObject.memobj);
    }
    else {
        requestNo = 0;
        MPI_Isend(&tmpReleaseMemObject, sizeof(tmpReleaseMemObject), MPI_BYTE,
                  proxyRank, RELEASE_MEM_OBJ, proxyComm, request + (requestNo++));
        MPI_Irecv(&tmpReleaseMemObject, sizeof(tmpReleaseMemObject), MPI_BYTE,
                  proxyRank, RELEASE_MEM_OBJ, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
    }

    /* decrease the number of OpenCL objects count */
    voclObjCountDecrease(proxyIndex);

    /* the old memory is not released yet after migration, do it here */
    if (voclIsOldMemoryValid((vocl_mem) memobj)) {
        tmpReleaseMemObject.memobj = voclVOCLMemory2OldCLMemoryComm((vocl_mem) memobj,
                                                                    &proxyRank, &proxyIndex,
                                                                    &proxyComm,
                                                                    &proxyCommData);
        if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
            tmpReleaseMemObject.res = dlCLReleaseMemObject(tmpReleaseMemObject.memobj);
        }
        else {
            requestNo = 0;
            MPI_Isend(&tmpReleaseMemObject, sizeof(tmpReleaseMemObject), MPI_BYTE,
                      proxyRank, RELEASE_MEM_OBJ, proxyComm, request + (requestNo++));
            MPI_Irecv(&tmpReleaseMemObject, sizeof(tmpReleaseMemObject), MPI_BYTE,
                      proxyRank, RELEASE_MEM_OBJ, proxyComm, request + (requestNo++));
            MPI_Waitall(requestNo, request, status);
        }

        /* decrease the number of OpenCL objects count */
        voclObjCountDecrease(proxyIndex);
    }

    return tmpReleaseMemObject.res;
}

cl_int clReleaseKernel(cl_kernel kernel)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    struct strReleaseKernel tmpReleaseKernel;

    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    /* release kernel on the remote node */
    tmpReleaseKernel.kernel = voclVOCLKernel2CLKernelComm((vocl_kernel) kernel,
                                                          &proxyRank, &proxyIndex, &proxyComm,
                                                          &proxyCommData);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpReleaseKernel.res = dlCLReleaseKernel(tmpReleaseKernel.kernel);
    }
    else {
        MPI_Isend(&tmpReleaseKernel, sizeof(tmpReleaseKernel), MPI_BYTE,
                  proxyRank, CL_RELEASE_KERNEL_FUNC, proxyComm, request + (requestNo++));
        MPI_Irecv(&tmpReleaseKernel, sizeof(tmpReleaseKernel), MPI_BYTE,
                  proxyRank, CL_RELEASE_KERNEL_FUNC, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
    }

    /* decrease the number of OpenCL objects count */
    voclObjCountDecrease(proxyIndex);

    return tmpReleaseKernel.res;
}

cl_int clFinish(cl_command_queue hInCmdQueue)
{
    MPI_Status status[2];
    MPI_Request request1, request2;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    struct strFinish tmpFinish;
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    tmpFinish.command_queue =
        voclVOCLCommandQueue2CLCommandQueueComm((vocl_command_queue) hInCmdQueue, &proxyRank,
                                                &proxyIndex, &proxyComm, &proxyCommData);

    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpFinish.res = dlCLFinish(tmpFinish.command_queue);
    }
    else {
        MPI_Isend(&tmpFinish, sizeof(tmpFinish), MPI_BYTE, proxyRank, FINISH_FUNC, proxyComm,
                  &request1);

        processAllWrites(proxyIndex);
        processAllReads(proxyIndex);

        MPI_Irecv(&tmpFinish, sizeof(tmpFinish), MPI_BYTE, proxyRank, FINISH_FUNC, proxyComm,
                  &request2);
        MPI_Wait(&request1, status);
        MPI_Wait(&request2, status);
    }

    return tmpFinish.res;
}

cl_int
clGetContextInfo(cl_context context,
                 cl_context_info param_name,
                 size_t param_value_size, void *param_value, size_t * param_value_size_ret)
{
    MPI_Status status[3];
    MPI_Request request[3];
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    struct strGetContextInfo tmpGetContextInfo;

    tmpGetContextInfo.context = voclVOCLContext2CLContextComm((vocl_context) context,
                                                              &proxyRank, &proxyIndex,
                                                              &proxyComm, &proxyCommData);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpGetContextInfo.res = dlCLGetContextInfo(tmpGetContextInfo.context,
                                                   param_name, param_value_size, param_value,
                                                   param_value_size_ret);
    }
    else {
        tmpGetContextInfo.param_name = param_name;
        tmpGetContextInfo.param_value_size = param_value_size;
        tmpGetContextInfo.param_value = param_value;
        tmpGetContextInfo.param_value_size_ret = 1;
        if (param_value_size_ret == NULL) {
            tmpGetContextInfo.param_value_size_ret = 0;
        }

        MPI_Isend(&tmpGetContextInfo, sizeof(tmpGetContextInfo), MPI_BYTE, proxyRank,
                  GET_CONTEXT_INFO_FUNC, proxyComm, request + (requestNo++));
        MPI_Irecv(&tmpGetContextInfo, sizeof(tmpGetContextInfo), MPI_BYTE, proxyRank,
                  GET_CONTEXT_INFO_FUNC, proxyComm, request + (requestNo++));

        if (param_value != NULL) {
            MPI_Irecv(param_value, param_value_size, MPI_BYTE, proxyRank,
                      GET_CONTEXT_INFO_FUNC1, proxyCommData, request + (requestNo++));
        }
        MPI_Waitall(requestNo, request, status);

        if (param_value_size_ret != NULL) {
            *param_value_size_ret = tmpGetContextInfo.param_value_size_ret;
        }
    }

    return tmpGetContextInfo.res;
}

cl_int
clGetProgramBuildInfo(cl_program program,
                      cl_device_id device,
                      cl_program_build_info param_name,
                      size_t param_value_size,
                      void *param_value, size_t * param_value_size_ret)
{
    MPI_Status status[3];
    MPI_Request request[3];
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    struct strGetProgramBuildInfo tmpGetProgramBuildInfo;
    tmpGetProgramBuildInfo.program = voclVOCLProgram2CLProgramComm((vocl_program) program,
                                                                   &proxyRank, &proxyIndex,
                                                                   &proxyComm, &proxyCommData);
    tmpGetProgramBuildInfo.device =
        voclVOCLDeviceID2CLDeviceIDComm((vocl_device_id) device, &proxyRank, &proxyIndex,
                                        &proxyComm, &proxyCommData);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpGetProgramBuildInfo.res = dlCLetProgramBuildInfo(tmpGetProgramBuildInfo.program,
                                                            tmpGetProgramBuildInfo.device,
                                                            param_name, param_value_size,
                                                            param_value, param_value_size_ret);
    }
    else {
        tmpGetProgramBuildInfo.param_name = param_name;
        tmpGetProgramBuildInfo.param_value_size = param_value_size;
        tmpGetProgramBuildInfo.param_value = param_value;
        tmpGetProgramBuildInfo.param_value_size_ret = 1;
        if (param_value_size_ret == NULL) {
            tmpGetProgramBuildInfo.param_value_size_ret = 0;
        }

        MPI_Isend(&tmpGetProgramBuildInfo, sizeof(tmpGetProgramBuildInfo), MPI_BYTE, proxyRank,
                  GET_BUILD_INFO_FUNC, proxyComm, request + (requestNo++));
        MPI_Irecv(&tmpGetProgramBuildInfo, sizeof(tmpGetProgramBuildInfo), MPI_BYTE, proxyRank,
                  GET_BUILD_INFO_FUNC, proxyComm, request + (requestNo++));

        if (param_value != NULL) {
            MPI_Irecv(param_value, param_value_size, MPI_BYTE, proxyRank,
                      GET_BUILD_INFO_FUNC1, proxyCommData, request + (requestNo++));
        }
        MPI_Waitall(requestNo, request, status);

        if (param_value_size_ret != NULL) {
            *param_value_size_ret = tmpGetProgramBuildInfo.param_value_size_ret;
        }
    }

    return tmpGetProgramBuildInfo.res;
}

cl_int
clGetProgramInfo(cl_program program,
                 cl_program_info param_name,
                 size_t param_value_size, void *param_value, size_t * param_value_size_ret)
{
    MPI_Status status[3];
    MPI_Request request[3];
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    struct strGetProgramInfo tmpGetProgramInfo;
    tmpGetProgramInfo.program = voclVOCLProgram2CLProgramComm((vocl_program) program,
                                                              &proxyRank, &proxyIndex,
                                                              &proxyComm, &proxyCommData);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpGetProgramInfo.res = dlCLGetProgramInfo(tmpGetProgramInfo.program,
                                                   param_name, param_value_size, param_value,
                                                   param_value_size_ret);
    }
    else {
        tmpGetProgramInfo.param_name = param_name;
        tmpGetProgramInfo.param_value_size = param_value_size;
        tmpGetProgramInfo.param_value = param_value;
        tmpGetProgramInfo.param_value_size_ret = 1;
        if (param_value_size_ret == NULL) {
            tmpGetProgramInfo.param_value_size_ret = 0;
        }

        MPI_Isend(&tmpGetProgramInfo, sizeof(tmpGetProgramInfo), MPI_BYTE, proxyRank,
                  GET_PROGRAM_INFO_FUNC, proxyComm, request + (requestNo++));
        MPI_Irecv(&tmpGetProgramInfo, sizeof(tmpGetProgramInfo), MPI_BYTE, proxyRank,
                  GET_PROGRAM_INFO_FUNC, proxyComm, request + (requestNo++));

        if (param_value != NULL) {
            MPI_Irecv(param_value, param_value_size, MPI_BYTE, proxyRank,
                      GET_PROGRAM_INFO_FUNC1, proxyCommData, request + (requestNo++));
        }
        MPI_Waitall(requestNo, request, status);

        if (param_value_size_ret != NULL) {
            *param_value_size_ret = tmpGetProgramInfo.param_value_size_ret;
        }
    }

    return tmpGetProgramInfo.res;
}

cl_int clReleaseProgram(cl_program program)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    struct strReleaseProgram tmpReleaseProgram;
    tmpReleaseProgram.program = voclVOCLProgram2CLProgramComm((vocl_program) program,
                                                              &proxyRank, &proxyIndex,
                                                              &proxyComm, &proxyCommData);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpReleaseProgram.res = dlCLReleaseProgram(tmpReleaseProgram.program);
    }
    else {
        MPI_Isend(&tmpReleaseProgram, sizeof(tmpReleaseProgram), MPI_BYTE, proxyRank,
                  REL_PROGRAM_FUNC, proxyComm, request + (requestNo++));
        MPI_Irecv(&tmpReleaseProgram, sizeof(tmpReleaseProgram), MPI_BYTE, proxyRank,
                  REL_PROGRAM_FUNC, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
    }
    /* decrease the number of OpenCL objects count */
    voclObjCountDecrease(proxyIndex);

    return tmpReleaseProgram.res;
}

cl_int clReleaseCommandQueue(cl_command_queue command_queue)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    struct strReleaseCommandQueue tmpReleaseCommandQueue;
    tmpReleaseCommandQueue.command_queue =
        voclVOCLCommandQueue2CLCommandQueueComm((vocl_command_queue) command_queue, &proxyRank,
                                                &proxyIndex, &proxyComm, &proxyCommData);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpReleaseCommandQueue.res =
            dlCLReleaseCommandQueue(tmpReleaseCommandQueue.command_queue);
    }
    else {
        MPI_Isend(&tmpReleaseCommandQueue, sizeof(tmpReleaseCommandQueue), MPI_BYTE, proxyRank,
                  REL_COMMAND_QUEUE_FUNC, proxyComm, request + (requestNo++));
        MPI_Irecv(&tmpReleaseCommandQueue, sizeof(tmpReleaseCommandQueue), MPI_BYTE, proxyRank,
                  REL_COMMAND_QUEUE_FUNC, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
    }

    /* decrease the number of OpenCL objects count */
    voclObjCountDecrease(proxyIndex);

    /* the old command queue is not released yet after migration, release it here */
    if (voclIsOldCommandQueueValid((vocl_command_queue) command_queue)) {
        tmpReleaseCommandQueue.command_queue =
            voclVOCLCommandQueue2OldCLCommandQueueComm((vocl_command_queue) command_queue,
                                                       &proxyRank, &proxyIndex, &proxyComm,
                                                       &proxyCommData);
        if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
            tmpReleaseCommandQueue.res =
                dlCLReleaseCommandQueue(tmpReleaseCommandQueue.command_queue);
        }
        else {
            MPI_Isend(&tmpReleaseCommandQueue, sizeof(tmpReleaseCommandQueue), MPI_BYTE,
                      proxyRank, REL_COMMAND_QUEUE_FUNC, proxyComm, request + (requestNo++));
            MPI_Irecv(&tmpReleaseCommandQueue, sizeof(tmpReleaseCommandQueue), MPI_BYTE,
                      proxyRank, REL_COMMAND_QUEUE_FUNC, proxyComm, request + (requestNo++));
            MPI_Waitall(requestNo, request, status);
        }

        /* decrease the number of OpenCL objects count */
        voclObjCountDecrease(proxyIndex);
    }

    return tmpReleaseCommandQueue.res;
}

cl_int clReleaseContext(cl_context context)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    struct strReleaseContext tmpReleaseContext;
    tmpReleaseContext.context = voclVOCLContext2CLContextComm((vocl_context) context,
                                                              &proxyRank, &proxyIndex,
                                                              &proxyComm, &proxyCommData);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpReleaseContext.res = dlCLReleaseContext(tmpReleaseContext.context);
    }
    else {
        MPI_Isend(&tmpReleaseContext, sizeof(tmpReleaseContext), MPI_BYTE, proxyRank,
                  REL_CONTEXT_FUNC, proxyComm, request + (requestNo++));
        MPI_Irecv(&tmpReleaseContext, sizeof(tmpReleaseContext), MPI_BYTE, proxyRank,
                  REL_CONTEXT_FUNC, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
    }

    /* decrease the number of OpenCL objects count */
    voclObjCountDecrease(proxyIndex);

    return tmpReleaseContext.res;
}

cl_int
clGetDeviceInfo(cl_device_id device,
                cl_device_info param_name,
                size_t param_value_size, void *param_value, size_t * param_value_size_ret)
{
    MPI_Status status[3];
    MPI_Request request[3];
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    struct strGetDeviceInfo tmpGetDeviceInfo;
    tmpGetDeviceInfo.device = voclVOCLDeviceID2CLDeviceIDComm((vocl_device_id) device,
                                                              &proxyRank, &proxyIndex,
                                                              &proxyComm, &proxyCommData);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpGetDeviceInfo.res = dlCLGetDeviceInfo(tmpGetDeviceInfo.device,
                                                 param_name, param_value_size, param_value,
                                                 param_value_size_ret);
    }
    else {
        tmpGetDeviceInfo.param_name = param_name;
        tmpGetDeviceInfo.param_value_size = param_value_size;
        tmpGetDeviceInfo.param_value = param_value;
        tmpGetDeviceInfo.param_value_size_ret = 1;
        if (param_value_size_ret == NULL) {
            tmpGetDeviceInfo.param_value_size_ret = 0;
        }

        MPI_Isend(&tmpGetDeviceInfo, sizeof(tmpGetDeviceInfo), MPI_BYTE, proxyRank,
                  GET_DEVICE_INFO_FUNC, proxyComm, request + (requestNo++));
        MPI_Irecv(&tmpGetDeviceInfo, sizeof(tmpGetDeviceInfo), MPI_BYTE, proxyRank,
                  GET_DEVICE_INFO_FUNC, proxyComm, request + (requestNo++));

        if (param_value != NULL) {
            MPI_Irecv(param_value, param_value_size, MPI_BYTE, proxyRank,
                      GET_DEVICE_INFO_FUNC1, proxyCommData, request + (requestNo++));
        }
        MPI_Waitall(requestNo, request, status);

        if (param_value_size_ret != NULL) {
            *param_value_size_ret = tmpGetDeviceInfo.param_value_size_ret;
        }
    }

    return tmpGetDeviceInfo.res;
}

cl_int
clGetPlatformInfo(cl_platform_id platform,
                  cl_platform_info param_name,
                  size_t param_value_size, void *param_value, size_t * param_value_size_ret)
{
    MPI_Status status[3];
    MPI_Request request[3];
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    struct strGetPlatformInfo tmpGetPlatformInfo;
    tmpGetPlatformInfo.platform =
        voclVOCLPlatformID2CLPlatformIDComm((vocl_platform_id) platform, &proxyRank,
                                            &proxyIndex, &proxyComm, &proxyCommData);
    if (param_name == CL_PLATFORM_IS_LOCAL) {
        if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
            return CL_TRUE;
        }
        else {
            return CL_FALSE;
        }
    }

    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpGetPlatformInfo.res = dlCLGetPlatformInfo(tmpGetPlatformInfo.platform, param_name,
                                                     param_value_size, param_value,
                                                     param_value_size_ret);
    }
    else {
        tmpGetPlatformInfo.param_name = param_name;
        tmpGetPlatformInfo.param_value_size = param_value_size;
        tmpGetPlatformInfo.param_value = param_value;
        tmpGetPlatformInfo.param_value_size_ret = 1;
        if (param_value_size_ret == NULL) {
            tmpGetPlatformInfo.param_value_size_ret = 0;
        }

        MPI_Isend(&tmpGetPlatformInfo, sizeof(tmpGetPlatformInfo), MPI_BYTE, proxyRank,
                  GET_PLATFORM_INFO_FUNC, proxyComm, request + (requestNo++));
        MPI_Irecv(&tmpGetPlatformInfo, sizeof(tmpGetPlatformInfo), MPI_BYTE, proxyRank,
                  GET_PLATFORM_INFO_FUNC, proxyComm, request + (requestNo++));

        if (param_value != NULL) {
            MPI_Irecv(param_value, param_value_size, MPI_BYTE, proxyRank,
                      GET_PLATFORM_INFO_FUNC1, proxyCommData, request + (requestNo++));
        }
        MPI_Waitall(requestNo, request, status);

        if (param_value_size_ret != NULL) {
            *param_value_size_ret = tmpGetPlatformInfo.param_value_size_ret;
        }
    }

    return tmpGetPlatformInfo.res;
}

cl_int clFlush(cl_command_queue hInCmdQueue)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;

    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    struct strFlush tmpFlush;
    tmpFlush.command_queue =
        voclVOCLCommandQueue2CLCommandQueueComm((vocl_command_queue) hInCmdQueue, &proxyRank,
                                                &proxyIndex, &proxyComm, &proxyCommData);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpFlush.res = dlCLFlush(tmpFlush.command_queue);
    }
    else {
        MPI_Isend(&tmpFlush, sizeof(tmpFlush), MPI_BYTE, proxyRank,
                  FLUSH_FUNC, proxyComm, request + (requestNo++));
        MPI_Irecv(&tmpFlush, sizeof(tmpFlush), MPI_BYTE, proxyRank,
                  FLUSH_FUNC, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
    }

    return tmpFlush.res;
}

cl_int clWaitForEvents(cl_uint num_events, const cl_event * event_list)
{
    MPI_Status status[3];
    MPI_Request request[3];
    int i, requestNo = 0;
    int proxyRank, proxyIndex, bufferIndex, bufferNum;
    MPI_Comm proxyComm, proxyCommData;
    cl_event *eventList = NULL;

    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    struct strWaitForEvents tmpWaitForEvents;
    tmpWaitForEvents.num_events = num_events;

    /* allocate opencl event buffer */
    eventList = (cl_event *) malloc(sizeof(cl_event) * num_events);
    if (eventList == NULL) {
        printf("wait for event, allocate eventList error!\n");
    }

    /* convert vocl events to opencl events */
    voclVOCLEvents2CLEventsComm((vocl_event *) event_list, eventList, num_events,
                                &proxyRank, &proxyIndex, &proxyComm, &proxyCommData);

    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpWaitForEvents.res = dlCLWaitForEvents(eventList, num_events);
    }
    else {
        MPI_Isend(&tmpWaitForEvents, sizeof(tmpWaitForEvents), MPI_BYTE, proxyRank,
                  WAIT_FOR_EVENT_FUNC, proxyComm, request + (requestNo++));
        MPI_Isend((void *) eventList, sizeof(cl_event) * num_events, MPI_BYTE, proxyRank,
                  WAIT_FOR_EVENT_FUNC1, proxyCommData, request + (requestNo++));

        /*wait for mpi transmission complete */
        for (i = 0; i < num_events; i++) {
            bufferIndex = getWriteBufferIndexFromEvent(proxyIndex, (vocl_event) event_list[i]);
            if (bufferIndex >= 0) {
                bufferNum = getWriteBufferNum(proxyIndex, bufferIndex);
                processWriteBuffer(proxyIndex, bufferIndex, bufferNum);
            }

            bufferIndex = getReadBufferIndexFromEvent(proxyIndex, (vocl_event) event_list[i]);
            if (bufferIndex >= 0) {
                bufferNum = getReadBufferNum(proxyIndex, bufferIndex);
                processReadBuffer(proxyIndex, bufferIndex, bufferNum);
            }
        }

        MPI_Irecv(&tmpWaitForEvents, sizeof(tmpWaitForEvents), MPI_BYTE, proxyRank,
                  WAIT_FOR_EVENT_FUNC, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
    }

    /* free opencl event buffer */
    free(eventList);
    eventList = NULL;

    return tmpWaitForEvents.res;
}

cl_sampler
clCreateSampler(cl_context context,
                cl_bool normalized_coords,
                cl_addressing_mode addressing_mode,
                cl_filter_mode filter_mode, cl_int * errcode_ret)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
    vocl_sampler sampler;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    struct strCreateSampler tmpCreateSampler;

    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    tmpCreateSampler.context = voclVOCLContext2CLContextComm((vocl_context) context,
                                                             &proxyRank, &proxyIndex,
                                                             &proxyComm, &proxyCommData);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        dlCLCreateSampler(tmpCreateSampler.context,
                          normalized_coords, addressing_mode, filter_mode, errcode_ret,
                          &tmpCreateSampler.sampler);
    }
    else {
        tmpCreateSampler.normalized_coords = normalized_coords;
        tmpCreateSampler.addressing_mode = addressing_mode;
        tmpCreateSampler.filter_mode = filter_mode;
        tmpCreateSampler.errcode_ret = 0;
        if (errcode_ret != NULL) {
            tmpCreateSampler.errcode_ret = 1;
        }

        MPI_Isend(&tmpCreateSampler, sizeof(tmpCreateSampler), MPI_BYTE, proxyRank,
                  CREATE_SAMPLER_FUNC, proxyComm, request + (requestNo++));
        MPI_Irecv(&tmpCreateSampler, sizeof(tmpCreateSampler), MPI_BYTE, proxyRank,
                  CREATE_SAMPLER_FUNC, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);

        if (errcode_ret != NULL) {
            *errcode_ret = tmpCreateSampler.errcode_ret;
        }
    }

    /* increase OpenCL object count */
    voclObjCountIncrease(proxyIndex);

    sampler = voclCLSampler2VOCLSampler(tmpCreateSampler.sampler,
                                        proxyRank, proxyIndex, proxyComm, proxyCommData);
    voclSamplerSetMigrationStatus(sampler,
                                  voclContextGetMigrationStatus((vocl_context) context));

    return (cl_sampler) sampler;
}

cl_int
clGetCommandQueueInfo(cl_command_queue command_queue,
                      cl_command_queue_info param_name,
                      size_t param_value_size,
                      void *param_value, size_t * param_value_size_ret)
{
    MPI_Status status[3];
    MPI_Request request[3];
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();
    struct strGetCommandQueueInfo tmpGetCommandQueueInfo;
    tmpGetCommandQueueInfo.command_queue =
        voclVOCLCommandQueue2CLCommandQueueComm((vocl_command_queue) command_queue, &proxyRank,
                                                &proxyIndex, &proxyComm, &proxyCommData);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpGetCommandQueueInfo.res =
            dlCLGetCommandQueueInfo(tmpGetCommandQueueInfo.command_queue, param_name,
                                    param_value_size, param_value, param_value_size_ret);
    }
    else {
        tmpGetCommandQueueInfo.param_name = param_name;
        tmpGetCommandQueueInfo.param_value_size = param_value_size;
        tmpGetCommandQueueInfo.param_value = param_value;
        tmpGetCommandQueueInfo.param_value_size_ret = 1;
        if (param_value_size_ret == NULL) {
            tmpGetCommandQueueInfo.param_value_size_ret = 0;
        }

        MPI_Isend(&tmpGetCommandQueueInfo, sizeof(tmpGetCommandQueueInfo), MPI_BYTE, proxyRank,
                  GET_CMD_QUEUE_INFO_FUNC, proxyComm, request + (requestNo++));
        MPI_Irecv(&tmpGetCommandQueueInfo, sizeof(tmpGetCommandQueueInfo), MPI_BYTE, proxyRank,
                  GET_CMD_QUEUE_INFO_FUNC, proxyComm, request + (requestNo++));

        if (param_value != NULL) {
            MPI_Irecv(param_value, param_value_size, MPI_BYTE, proxyRank,
                      GET_CMD_QUEUE_INFO_FUNC1, proxyCommData, request + (requestNo++));
        }
        MPI_Waitall(requestNo, request, status);
        if (param_value_size_ret != NULL) {
            *param_value_size_ret = tmpGetCommandQueueInfo.param_value_size_ret;
        }
    }

    return tmpGetCommandQueueInfo.res;
}

void *clEnqueueMapBuffer(cl_command_queue command_queue,
                         cl_mem buffer,
                         cl_bool blocking_map,
                         cl_map_flags map_flags,
                         size_t offset,
                         size_t cb,
                         cl_uint num_events_in_wait_list,
                         const cl_event * event_wait_list,
                         cl_event * event, cl_int * errcode_ret)
{
    MPI_Status status[3];
    MPI_Request request[3];
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    cl_event *eventList = NULL;
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();
    struct strEnqueueMapBuffer tmpEnqueueMapBuffer;
    tmpEnqueueMapBuffer.command_queue =
        voclVOCLCommandQueue2CLCommandQueueComm((vocl_command_queue) command_queue, &proxyRank,
                                                &proxyIndex, &proxyComm, &proxyCommData);
    tmpEnqueueMapBuffer.buffer =
        voclVOCLMemory2CLMemoryComm((vocl_mem) buffer, &proxyRank, &proxyIndex, &proxyComm,
                                    &proxyCommData);
    if (num_events_in_wait_list > 0) {
        eventList = (cl_event *) malloc(sizeof(cl_event) * num_events_in_wait_list);
        if (eventList == NULL) {
            printf("enqueueMapBuffer, allocate eventList error!\n");
        }

        /* convert vocl events to opencl events */
        voclVOCLEvents2CLEvents((vocl_event *) event_wait_list, eventList,
                                num_events_in_wait_list);
    }

    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpEnqueueMapBuffer.ret_ptr = dlCLEnqueueMapBuffer(tmpEnqueueMapBuffer.command_queue,
                                                           tmpEnqueueMapBuffer.buffer,
                                                           blocking_map, map_flags, offset, cb,
                                                           num_events_in_wait_list, eventList,
                                                           event, errcode_ret);
    }
    else {
        tmpEnqueueMapBuffer.blocking_map = blocking_map;
        tmpEnqueueMapBuffer.map_flags = map_flags;
        tmpEnqueueMapBuffer.offset = offset;
        tmpEnqueueMapBuffer.cb = cb;
        tmpEnqueueMapBuffer.num_events_in_wait_list = num_events_in_wait_list;
        if (event == NULL) {
            tmpEnqueueMapBuffer.event_null_flag = 1;
        }
        else {
            tmpEnqueueMapBuffer.event_null_flag = 0;
        }

        /* 0, NOT NULL, 1: NULL */
        tmpEnqueueMapBuffer.errcode_ret = 0;
        if (errcode_ret == NULL) {
            tmpEnqueueMapBuffer.errcode_ret = 1;
        }
        MPI_Isend(&tmpEnqueueMapBuffer, sizeof(tmpEnqueueMapBuffer), MPI_BYTE, proxyRank,
                  ENQUEUE_MAP_BUFF_FUNC, proxyComm, request + (requestNo++));
        if (num_events_in_wait_list > 0) {
            /* convert vocl events to opencl events */
            voclVOCLEvents2CLEvents((vocl_event *) event_wait_list, eventList,
                                    num_events_in_wait_list);

            MPI_Isend((void *) eventList, sizeof(cl_event) * num_events_in_wait_list,
                      MPI_BYTE, proxyRank, ENQUEUE_MAP_BUFF_FUNC1, proxyCommData,
                      request + (requestNo++));
        }
        MPI_Irecv(&tmpEnqueueMapBuffer, sizeof(tmpEnqueueMapBuffer), MPI_BYTE, proxyRank,
                  ENQUEUE_MAP_BUFF_FUNC, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);

        if (errcode_ret != NULL) {
            *errcode_ret = tmpEnqueueMapBuffer.errcode_ret;
        }
    }

    if (event != NULL) {
        /* convert opencl event to vocl event */
        *event = (cl_event) voclCLEvent2VOCLEvent(tmpEnqueueMapBuffer.event,
                                                  proxyRank, proxyIndex, proxyComm,
                                                  proxyCommData);
    }

    /* free opencl event buffer */
    if (num_events_in_wait_list > 0) {
        free(eventList);
        eventList = NULL;
    }

    return tmpEnqueueMapBuffer.ret_ptr;
}

cl_int clReleaseEvent(cl_event event)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;

    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    struct strReleaseEvent tmpReleaseEvent;
    /* convert vocl event to opencl event */
    tmpReleaseEvent.event = voclVOCLEvent2CLEventComm((vocl_event) event,
                                                      &proxyRank, &proxyIndex, &proxyComm,
                                                      &proxyCommData);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpReleaseEvent.res = dlCLReleaseEvent(tmpReleaseEvent.event);
    }
    else {
        MPI_Isend(&tmpReleaseEvent, sizeof(tmpReleaseEvent), MPI_BYTE, proxyRank,
                  RELEASE_EVENT_FUNC, proxyComm, request + (requestNo++));
        MPI_Irecv(&tmpReleaseEvent, sizeof(tmpReleaseEvent), MPI_BYTE, proxyRank,
                  RELEASE_EVENT_FUNC, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
    }
    return tmpReleaseEvent.res;
}

cl_int
clGetEventProfilingInfo(cl_event event,
                        cl_profiling_info param_name,
                        size_t param_value_size,
                        void *param_value, size_t * param_value_size_ret)
{
    MPI_Status status[3];
    MPI_Request request[3];
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;

    checkSlaveProc();

    struct strGetEventProfilingInfo tmpGetEventProfilingInfo;
    /*convert vocl event to opencl event */
    tmpGetEventProfilingInfo.event = voclVOCLEvent2CLEventComm((vocl_event) event,
                                                               &proxyRank, &proxyIndex,
                                                               &proxyComm, &proxyCommData);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpGetEventProfilingInfo.res =
            dlCLGetEventProfilingInfo(tmpGetEventProfilingInfo.event, param_name,
                                      param_value_size, param_value, param_value_size_ret);
    }
    else {
        tmpGetEventProfilingInfo.param_name = param_name;
        tmpGetEventProfilingInfo.param_value_size = param_value_size;
        tmpGetEventProfilingInfo.param_value = param_value;
        tmpGetEventProfilingInfo.param_value_size_ret = 1;
        if (param_value_size_ret == NULL) {
            tmpGetEventProfilingInfo.param_value_size_ret = 0;
        }

        MPI_Isend(&tmpGetEventProfilingInfo, sizeof(tmpGetEventProfilingInfo), MPI_BYTE,
                  proxyRank, GET_EVENT_PROF_INFO_FUNC, proxyComm, request + (requestNo++));
        MPI_Irecv(&tmpGetEventProfilingInfo, sizeof(tmpGetEventProfilingInfo), MPI_BYTE,
                  proxyRank, GET_EVENT_PROF_INFO_FUNC, proxyComm, request + (requestNo++));

        if (param_value != NULL) {
            MPI_Irecv(param_value, param_value_size, MPI_BYTE, proxyRank,
                      GET_EVENT_PROF_INFO_FUNC1, proxyCommData, request + (requestNo++));
        }
        MPI_Waitall(requestNo, request, status);
        if (param_value_size_ret != NULL) {
            *param_value_size_ret = tmpGetEventProfilingInfo.param_value_size_ret;
        }
    }

    return tmpGetEventProfilingInfo.res;
}

cl_int clReleaseSampler(cl_sampler sampler)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    struct strReleaseSampler tmpReleaseSampler;
    checkSlaveProc();
    tmpReleaseSampler.sampler = voclVOCLSampler2CLSamplerComm((vocl_sampler) sampler,
                                                              &proxyRank, &proxyIndex,
                                                              &proxyComm, &proxyCommData);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpReleaseSampler.res = dlCLReleaseSampler(tmpReleaseSampler.sampler);
    }
    else {
        MPI_Isend(&tmpReleaseSampler, sizeof(tmpReleaseSampler), MPI_BYTE, proxyRank,
                  RELEASE_SAMPLER_FUNC, proxyComm, request + (requestNo++));
        MPI_Irecv(&tmpReleaseSampler, sizeof(tmpReleaseSampler), MPI_BYTE, proxyRank,
                  RELEASE_SAMPLER_FUNC, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
    }

    /* decrease the number of OpenCL objects count */
    voclObjCountDecrease(proxyIndex);

    return tmpReleaseSampler.res;
}

cl_int
clGetKernelWorkGroupInfo(cl_kernel kernel,
                         cl_device_id device,
                         cl_kernel_work_group_info param_name,
                         size_t param_value_size,
                         void *param_value, size_t * param_value_size_ret)
{
    MPI_Status status[3];
    MPI_Request request[3];
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    struct strGetKernelWorkGroupInfo tmpGetKernelWorkGroupInfo;

    checkSlaveProc();

    tmpGetKernelWorkGroupInfo.kernel = voclVOCLKernel2CLKernelComm((vocl_kernel) kernel,
                                                                   &proxyRank, &proxyIndex,
                                                                   &proxyComm, &proxyCommData);
    tmpGetKernelWorkGroupInfo.device =
        voclVOCLDeviceID2CLDeviceIDComm((vocl_device_id) device, &proxyRank, &proxyIndex,
                                        &proxyComm, &proxyCommData);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpGetKernelWorkGroupInfo.res =
            dlCLGetKernelWorkGroupInfo(tmpGetKernelWorkGroupInfo.kernel,
                                       tmpGetKernelWorkGroupInfo.device, param_name,
                                       param_value_size, param_value, param_value_size_ret);
    }
    else {
        tmpGetKernelWorkGroupInfo.param_name = param_name;
        tmpGetKernelWorkGroupInfo.param_value_size = param_value_size;
        tmpGetKernelWorkGroupInfo.param_value = param_value;
        tmpGetKernelWorkGroupInfo.param_value_size_ret = 1;
        if (param_value_size_ret == NULL) {
            tmpGetKernelWorkGroupInfo.param_value_size_ret = 0;
        }

        MPI_Isend(&tmpGetKernelWorkGroupInfo, sizeof(tmpGetKernelWorkGroupInfo), MPI_BYTE,
                  proxyRank, GET_KERNEL_WGP_INFO_FUNC, proxyComm, request + (requestNo++));
        MPI_Irecv(&tmpGetKernelWorkGroupInfo, sizeof(tmpGetKernelWorkGroupInfo), MPI_BYTE,
                  proxyRank, GET_KERNEL_WGP_INFO_FUNC, proxyComm, request + (requestNo++));

        if (param_value != NULL) {
            MPI_Irecv(param_value, param_value_size, MPI_BYTE, proxyRank,
                      GET_KERNEL_WGP_INFO_FUNC1, proxyCommData, request + (requestNo++));
        }
        MPI_Waitall(requestNo, request, status);

        if (param_value_size_ret != NULL) {
            *param_value_size_ret = tmpGetKernelWorkGroupInfo.param_value_size_ret;
        }
    }

    return tmpGetKernelWorkGroupInfo.res;
}

cl_mem
clCreateImage2D(cl_context context,
                cl_mem_flags flags,
                const cl_image_format * image_format,
                size_t image_width,
                size_t image_height,
                size_t image_row_pitch, void *host_ptr, cl_int * errcode_ret)
{
    MPI_Status status[3];
    MPI_Request request[3];
    vocl_mem mem_obj;
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    struct strCreateImage2D tmpCreateImage2D;

    checkSlaveProc();
    tmpCreateImage2D.context = voclVOCLContext2CLContextComm((vocl_context) context,
                                                             &proxyRank, &proxyIndex,
                                                             &proxyComm, &proxyCommData);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        dlCLCreateImage2D(tmpCreateImage2D.context,
                          flags, image_format, image_width, image_height, image_row_pitch,
                          host_ptr, errcode_ret, &tmpCreateImage2D.mem_obj);
    }
    else {
        tmpCreateImage2D.flags = flags;
        tmpCreateImage2D.img_format.image_channel_order = image_format->image_channel_order;
        tmpCreateImage2D.img_format.image_channel_data_type =
            image_format->image_channel_data_type;
        tmpCreateImage2D.image_width = image_width;
        tmpCreateImage2D.image_height = image_height;
        tmpCreateImage2D.image_row_pitch = image_row_pitch;
        tmpCreateImage2D.host_buff_size = 0;
        if (host_ptr != NULL) {
            if (image_row_pitch == 0) {
                tmpCreateImage2D.host_buff_size =
                    image_width * sizeof(cl_image_format) * image_height * 2;
            }
            else {
                tmpCreateImage2D.host_buff_size = image_row_pitch * image_height * 2;
            }
        }
        /* default errcode */
        tmpCreateImage2D.errcode_ret = 0;
        if (errcode_ret == NULL) {
            tmpCreateImage2D.errcode_ret = 1;
        }
        MPI_Isend(&tmpCreateImage2D, sizeof(tmpCreateImage2D), MPI_BYTE, proxyRank,
                  CREATE_IMAGE_2D_FUNC, proxyComm, request + (requestNo++));
        if (host_ptr != NULL) {
            MPI_Isend(host_ptr, tmpCreateImage2D.host_buff_size, MPI_BYTE, proxyRank,
                      CREATE_IMAGE_2D_FUNC1, proxyCommData, request + (requestNo++));
        }
        MPI_Irecv(&tmpCreateImage2D, sizeof(tmpCreateImage2D), MPI_BYTE, proxyRank,
                  CREATE_IMAGE_2D_FUNC, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
        if (errcode_ret != NULL) {
            *errcode_ret = tmpCreateImage2D.errcode_ret;
        }
    }

    /* increase OpenCL object count */
    voclObjCountIncrease(proxyIndex);

    mem_obj = voclCLMemory2VOCLMemory(tmpCreateImage2D.mem_obj,
                                      proxyRank, proxyIndex, proxyComm, proxyCommData);

    return (cl_mem) mem_obj;
}

cl_int
clEnqueueCopyBuffer(cl_command_queue command_queue,
                    cl_mem src_buffer,
                    cl_mem dst_buffer,
                    size_t src_offset,
                    size_t dst_offset,
                    size_t cb,
                    cl_uint num_events_in_wait_list,
                    const cl_event * event_wait_list, cl_event * event)
{
    MPI_Status status[3];
    MPI_Request request[3];
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    cl_event *eventList = NULL;
    struct strEnqueueCopyBuffer tmpEnqueueCopyBuffer;

    checkSlaveProc();

    tmpEnqueueCopyBuffer.command_queue = command_queue;
    tmpEnqueueCopyBuffer.src_buffer = voclVOCLMemory2CLMemoryComm((vocl_mem) src_buffer,
                                                                  &proxyRank, &proxyIndex,
                                                                  &proxyComm, &proxyCommData);
    tmpEnqueueCopyBuffer.dst_buffer =
        voclVOCLMemory2CLMemoryComm((vocl_mem) dst_buffer, &proxyRank, &proxyIndex, &proxyComm,
                                    &proxyCommData);
    if (num_events_in_wait_list > 0) {
        eventList = (cl_event *) malloc(sizeof(cl_event) * num_events_in_wait_list);
        if (eventList == NULL) {
            printf("enqueuecopybuffer, allocate eventList error!\n");
        }

        /* convert vocl events to opencl events */
        voclVOCLEvents2CLEvents((vocl_event *) event_wait_list, eventList,
                                num_events_in_wait_list);
    }

    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpEnqueueCopyBuffer.res = dlCLEnqueueCopyBuffer(tmpEnqueueCopyBuffer.command_queue,
                                                         tmpEnqueueCopyBuffer.src_buffer,
                                                         tmpEnqueueCopyBuffer.dst_buffer,
                                                         src_offset, dst_offset, cb,
                                                         num_events_in_wait_list, eventList,
                                                         event);
    }
    else {
        tmpEnqueueCopyBuffer.src_offset = src_offset;
        tmpEnqueueCopyBuffer.dst_offset = dst_offset;
        tmpEnqueueCopyBuffer.cb = cb;
        tmpEnqueueCopyBuffer.num_events_in_wait_list = num_events_in_wait_list;
        tmpEnqueueCopyBuffer.event_null_flag = 0;
        if (event == NULL) {
            tmpEnqueueCopyBuffer.event_null_flag = 1;
        }

        MPI_Isend(&tmpEnqueueCopyBuffer, sizeof(tmpEnqueueCopyBuffer), MPI_BYTE, proxyRank,
                  ENQ_COPY_BUFF_FUNC, proxyComm, request + (requestNo++));
        if (num_events_in_wait_list > 0) {
            MPI_Isend((void *) eventList, sizeof(cl_event) * num_events_in_wait_list,
                      MPI_BYTE, proxyRank, ENQ_COPY_BUFF_FUNC1, proxyCommData,
                      request + (requestNo++));
        }
        MPI_Irecv(&tmpEnqueueCopyBuffer, sizeof(tmpEnqueueCopyBuffer), MPI_BYTE, proxyRank,
                  ENQ_COPY_BUFF_FUNC, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
        if (event != NULL) {
            *event = tmpEnqueueCopyBuffer.event;
        }
    }

    if (event != NULL) {
        /* convert opencl event to vocl event */
        *event = (cl_event) voclCLEvent2VOCLEvent((*event),
                                                  proxyRank, proxyIndex, proxyComm,
                                                  proxyCommData);
    }

    if (num_events_in_wait_list > 0) {
        free(eventList);
        eventList = NULL;
    }

    return tmpEnqueueCopyBuffer.res;
}

cl_int clRetainEvent(cl_event event)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    struct strRetainEvent tmpRetainEvent;

    checkSlaveProc();

    /*convert vocl event to opencl event */
    tmpRetainEvent.event = voclVOCLEvent2CLEventComm((vocl_event) event,
                                                     &proxyRank, &proxyIndex, &proxyComm,
                                                     &proxyCommData);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpRetainEvent.res = clDLRetainEvent(tmpRetainEvent.event);
    }
    else {
        MPI_Isend(&tmpRetainEvent, sizeof(tmpRetainEvent), MPI_BYTE, proxyRank,
                  RETAIN_EVENT_FUNC, proxyComm, request + (requestNo++));
        MPI_Irecv(&tmpRetainEvent, sizeof(tmpRetainEvent), MPI_BYTE, proxyRank,
                  RETAIN_EVENT_FUNC, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
    }

    return tmpRetainEvent.res;
}

cl_int clRetainMemObject(cl_mem memobj)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    struct strRetainMemObject tmpRetainMemObject;

    checkSlaveProc();

    tmpRetainMemObject.memobj = voclVOCLMemory2CLMemoryComm((vocl_mem) memobj,
                                                            &proxyRank, &proxyIndex,
                                                            &proxyComm, &proxyCommData);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpRetainMemObject.res = dlCLRetainMemObject(tmpRetainMemObject.memobj);
    }
    else {
        MPI_Isend(&tmpRetainMemObject, sizeof(tmpRetainMemObject), MPI_BYTE, proxyRank,
                  RETAIN_MEMOBJ_FUNC, proxyComm, request + (requestNo++));
        MPI_Irecv(&tmpRetainMemObject, sizeof(tmpRetainMemObject), MPI_BYTE, proxyRank,
                  RETAIN_MEMOBJ_FUNC, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
    }

    return tmpRetainMemObject.res;
}

cl_int clRetainKernel(cl_kernel kernel)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    struct strRetainKernel tmpRetainKernel;
    int requestNo = 0;
    checkSlaveProc();

    tmpRetainKernel.kernel = voclVOCLKernel2CLKernelComm((vocl_kernel) kernel,
                                                         &proxyRank, &proxyIndex, &proxyComm,
                                                         &proxyCommData);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpRetainKernel.res = dlCLRetainKernel(tmpRetainKernel.kernel);
    }
    else {
        MPI_Isend(&tmpRetainKernel, sizeof(tmpRetainKernel), MPI_BYTE, proxyRank,
                  RETAIN_KERNEL_FUNC, proxyComm, request + (requestNo++));
        MPI_Irecv(&tmpRetainKernel, sizeof(tmpRetainKernel), MPI_BYTE, proxyRank,
                  RETAIN_KERNEL_FUNC, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
    }

    return tmpRetainKernel.res;
}

cl_int clRetainCommandQueue(cl_command_queue command_queue)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    struct strRetainCommandQueue tmpRetainCommandQueue;
    checkSlaveProc();

    tmpRetainCommandQueue.command_queue =
        voclVOCLCommandQueue2CLCommandQueueComm((vocl_command_queue) command_queue, &proxyRank,
                                                &proxyIndex, &proxyComm, &proxyCommData);
    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpRetainCommandQueue.res =
            dlCLRetainCommandQueue(tmpRetainCommandQueue.command_queue);
    }
    else {
        MPI_Isend(&tmpRetainCommandQueue, sizeof(tmpRetainCommandQueue), MPI_BYTE, proxyRank,
                  RETAIN_CMDQUE_FUNC, proxyComm, request + (requestNo++));
        MPI_Irecv(&tmpRetainCommandQueue, sizeof(tmpRetainCommandQueue), MPI_BYTE, proxyRank,
                  RETAIN_CMDQUE_FUNC, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
    }

    return tmpRetainCommandQueue.res;
}

cl_int
clEnqueueUnmapMemObject(cl_command_queue command_queue,
                        cl_mem memobj,
                        void *mapped_ptr,
                        cl_uint num_events_in_wait_list,
                        const cl_event * event_wait_list, cl_event * event)
{
    MPI_Status status[3];
    MPI_Request request[3];
    cl_event *eventList = NULL;
    int requestNo = 0;
    int proxyRank, proxyIndex;
    MPI_Comm proxyComm, proxyCommData;
    struct strEnqueueUnmapMemObject tmpEnqueueUnmapMemObject;
    checkSlaveProc();

    tmpEnqueueUnmapMemObject.command_queue =
        voclVOCLCommandQueue2CLCommandQueueComm((vocl_command_queue) command_queue, &proxyRank,
                                                &proxyIndex, &proxyComm, &proxyCommData);
    tmpEnqueueUnmapMemObject.memobj =
        voclVOCLMemory2CLMemoryComm((vocl_mem) memobj, &proxyRank, &proxyIndex, &proxyComm,
                                    &proxyCommData);
    if (num_events_in_wait_list > 0) {
        eventList = (cl_event *) malloc(sizeof(cl_event) * num_events_in_wait_list);
        if (eventList == NULL) {
            printf("enqueueUnMapMemObject, allocate eventList error!\n");
        }

        /* convert vocl events to opencl events */
        voclVOCLEvents2CLEvents((vocl_event *) event_wait_list, eventList,
                                num_events_in_wait_list);
    }

    if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE) {
        tmpEnqueueUnmapMemObject.res =
            dlCLEnqueueUnmapMemObject(tmpEnqueueUnmapMemObject.command_queue,
                                      tmpEnqueueUnmapMemObject.memobj, mapped_ptr,
                                      num_events_in_wait_list, eventList, event);
    }
    else {
        tmpEnqueueUnmapMemObject.mapped_ptr = mapped_ptr;
        tmpEnqueueUnmapMemObject.num_events_in_wait_list = num_events_in_wait_list;
        tmpEnqueueUnmapMemObject.event_null_flag = 0;
        if (event == NULL) {
            tmpEnqueueUnmapMemObject.event_null_flag = 1;
        }
        MPI_Isend(&tmpEnqueueUnmapMemObject, sizeof(tmpEnqueueUnmapMemObject), MPI_BYTE,
                  proxyRank, ENQ_UNMAP_MEMOBJ_FUNC, proxyComm, request + (requestNo++));
        if (num_events_in_wait_list > 0) {
            MPI_Isend((void *) eventList, sizeof(cl_event) * num_events_in_wait_list,
                      MPI_BYTE, proxyRank, ENQ_UNMAP_MEMOBJ_FUNC1, proxyCommData,
                      request + (requestNo++));
        }
        MPI_Irecv(&tmpEnqueueUnmapMemObject, sizeof(tmpEnqueueUnmapMemObject), MPI_BYTE,
                  proxyRank, ENQ_UNMAP_MEMOBJ_FUNC, proxyComm, request + (requestNo++));
        MPI_Waitall(requestNo, request, status);
        if (event != NULL) {
            *event = tmpEnqueueUnmapMemObject.event;
        }
    }

    if (event != NULL) {
        *event =
            (cl_event) voclCLEvent2VOCLEvent((*event), proxyRank, proxyIndex, proxyComm,
                                             proxyCommData);
    }

    if (num_events_in_wait_list > 0) {
        free(eventList);
        eventList = NULL;
    }

    printf("end of migration!\n");

    return tmpEnqueueUnmapMemObject.res;
}
