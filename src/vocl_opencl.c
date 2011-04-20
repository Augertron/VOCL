#include <stdio.h>
#include <string.h>
#include "vocl_opencl.h"
#include "voclKernelArgProc.h"
#include <sys/time.h>

/* used for print node name */
#define _PRINT_NODE_NAME

/* for slave process */
static int slaveComm;
static int slaveCreated = 0;
static int np = 1;
static int errCodes[MAX_NPS];

/* kernel argument processing functions */
extern cl_int createKernel(cl_kernel kernel);
extern kernel_info *getKernelPtr(cl_kernel kernel);
extern cl_int releaseKernelPtr(cl_kernel kernel);

/* writeBufferPool API functions */
extern void initializeWriteBuffer();
extern void setWriteBufferInUse(int index);
extern MPI_Request *getWriteRequestPtr(int index);
extern int getNextWriteBufferIndex();
extern void processWriteBuffer(int curIndex, int bufferNum);
extern void processAllWrites();

/* readBufferPool API functions */
extern void initializeReadBuffer();
extern void setReadBufferInUse(int index);
extern MPI_Request *getReadRequestPtr(int index);
extern int getNextReadBufferIndex();
extern void processReadBuffer(int curIndex, int bufferNum);
extern void processAllReads();

/* send the terminating msg to the proxy */
static void mpiFinalize()
{
    MPI_Send(NULL, 0, MPI_BYTE, 0, PROGRAM_END, slaveComm);
    MPI_Comm_free(&slaveComm);
    MPI_Finalize();
}

/* function for create the proxy process and MPI communicator */
static void checkSlaveProc()
{
    struct timeval t1, t2;
    float tmpTime;
	char proxyPathName[PROXY_PATH_NAME_LEN];

    if (slaveCreated == 0) {
        MPI_Init(NULL, NULL);

		snprintf(proxyPathName, PROXY_PATH_NAME_LEN, "%s/bin/vocl_proxy", PROXY_PATH_NAME);
        MPI_Comm_spawn(proxyPathName, MPI_ARGV_NULL, np,
                       MPI_INFO_NULL, 0, MPI_COMM_WORLD, &slaveComm, errCodes);
        slaveCreated = 1;

#ifdef _PRINT_NODE_NAME
        char hostName[200];
        int len;
        MPI_Get_processor_name(hostName, &len);
        hostName[len] = '\0';
        printf("libHostName = %s\n", hostName);
#endif

        /* initialize write and read buffer processing */
        initializeWriteBuffer();
        initializeReadBuffer();

        if (atexit(mpiFinalize) != 0) {
            printf("register Finalize error!\n");
            exit(1);
        }
    }
}


cl_int
clGetPlatformIDs(cl_uint num_entries, cl_platform_id * platforms, cl_uint * num_platforms)
{
    MPI_Status status[3];
    MPI_Request request[3];
    int requestNo = 0;
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    struct strGetPlatformIDs tmpGetPlatform;

    /* initialize structure */
    tmpGetPlatform.num_entries = num_entries;
    tmpGetPlatform.platforms = platforms;
    tmpGetPlatform.num_platforms = 0;
    if (num_platforms != NULL) {
        tmpGetPlatform.num_platforms = 1;
    }

    /* send parameters to remote node */
    MPI_Isend(&tmpGetPlatform, sizeof(tmpGetPlatform), MPI_BYTE, 0,
              GET_PLATFORM_ID_FUNC, slaveComm, request + (requestNo++));

    MPI_Irecv(&tmpGetPlatform, sizeof(tmpGetPlatform), MPI_BYTE, 0,
              GET_PLATFORM_ID_FUNC, slaveComm, request + (requestNo++));
    if (platforms != NULL && num_entries > 0) {
        MPI_Irecv(platforms, sizeof(cl_platform_id) * num_entries, MPI_BYTE, 0,
                  GET_PLATFORM_ID_FUNC1, slaveComm, request + (requestNo++));
    }
    MPI_Waitall(requestNo, request, status);
    if (num_platforms != NULL) {
        *num_platforms = tmpGetPlatform.num_platforms;
    }

    return tmpGetPlatform.res;
}

cl_int
clGetDeviceIDs(cl_platform_id platform,
               cl_device_type device_type,
               cl_uint num_entries, cl_device_id * devices, cl_uint * num_devices)
{
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    struct strGetDeviceIDs tmpGetDeviceIDs;
    MPI_Status status[3];
    MPI_Request request[3];
    int requestNo = 0;

    /* initialize structure */
    tmpGetDeviceIDs.platform = platform;
    tmpGetDeviceIDs.device_type = device_type;
    tmpGetDeviceIDs.num_entries = num_entries;
    tmpGetDeviceIDs.devices = devices;

    /* indicate num_device be NOT NULL */
    tmpGetDeviceIDs.num_devices = 1;
    if (num_devices == NULL) {
        tmpGetDeviceIDs.num_devices = 0;
    }
    /* send parameters to remote node */
    MPI_Isend(&tmpGetDeviceIDs, sizeof(tmpGetDeviceIDs), MPI_BYTE, 0,
              GET_DEVICE_ID_FUNC, slaveComm, request + (requestNo++));

    if (num_entries > 0 && devices != NULL) {
        MPI_Irecv(devices, sizeof(cl_device_id) * num_entries, MPI_BYTE, 0,
                  GET_DEVICE_ID_FUNC1, slaveComm, request + (requestNo++));
    }
    MPI_Irecv(&tmpGetDeviceIDs, sizeof(tmpGetDeviceIDs), MPI_BYTE, 0,
              GET_DEVICE_ID_FUNC, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);
    if (num_devices != NULL) {
        *num_devices = tmpGetDeviceIDs.num_devices;
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
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    struct strCreateContext tmpCreateContext;
    MPI_Status status[3];
    MPI_Request request[3];
    int requestNo = 0;
    int res;

    /* initialize structure */
    /* tmpCreateContext.properties = *properties; */
    tmpCreateContext.num_devices = num_devices;
    tmpCreateContext.devices = (cl_device_id *) devices;
    tmpCreateContext.user_data = user_data;

    /* send parameters to remote node */
    res = MPI_Isend(&tmpCreateContext, sizeof(tmpCreateContext), MPI_BYTE, 0,
                    CREATE_CONTEXT_FUNC, slaveComm, request + (requestNo++));

    if (devices != NULL) {
        MPI_Isend((void *) devices, sizeof(cl_device_id) * num_devices, MPI_BYTE, 0,
                  CREATE_CONTEXT_FUNC1, slaveComm, request + (requestNo++));
    }

    MPI_Irecv(&tmpCreateContext, sizeof(tmpCreateContext), MPI_BYTE, 0,
              CREATE_CONTEXT_FUNC, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);
    *errcode_ret = tmpCreateContext.errcode_ret;

    return tmpCreateContext.hContext;
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

    tmpCreateCommandQueue.context = context;
    tmpCreateCommandQueue.device = device;
    tmpCreateCommandQueue.properties = properties;

    /* send parameters to remote node */
    MPI_Isend(&tmpCreateCommandQueue, sizeof(tmpCreateCommandQueue), MPI_BYTE, 0,
              CREATE_COMMAND_QUEUE_FUNC, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpCreateCommandQueue, sizeof(tmpCreateCommandQueue), MPI_BYTE, 0,
              CREATE_COMMAND_QUEUE_FUNC, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);
    if (errcode_ret != NULL) {
        *errcode_ret = tmpCreateCommandQueue.errcode_ret;
    }

    return tmpCreateCommandQueue.clCommand;
}

cl_program
clCreateProgramWithSource(cl_context context,
                          cl_uint count,
                          const char **strings, const size_t * lengths, cl_int * errcode_ret)
{
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    struct strCreateProgramWithSource tmpCreateProgramWithSource;
    MPI_Status status[4];
    MPI_Request request[4];
    int requestNo = 0;

    /* initialize structure */
    tmpCreateProgramWithSource.context = context;
    tmpCreateProgramWithSource.count = count;
    size_t totalLength, *lengthsArray, strStartLoc;
    cl_uint strIndex;
    char *allStrings;

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

    /* send parameters to remote node */
    MPI_Isend(&tmpCreateProgramWithSource,
              sizeof(tmpCreateProgramWithSource),
              MPI_BYTE, 0, CREATE_PROGRMA_WITH_SOURCE, slaveComm, request + (requestNo++));
    MPI_Isend(lengthsArray, sizeof(size_t) * count, MPI_BYTE, 0,
              CREATE_PROGRMA_WITH_SOURCE1, slaveComm, request + (requestNo++));
    MPI_Isend((void *) allStrings, totalLength * sizeof(char), MPI_BYTE, 0,
              CREATE_PROGRMA_WITH_SOURCE2, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpCreateProgramWithSource,
              sizeof(tmpCreateProgramWithSource),
              MPI_BYTE, 0, CREATE_PROGRMA_WITH_SOURCE, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);
    if (errcode_ret != NULL) {
        *errcode_ret = tmpCreateProgramWithSource.errcode_ret;
    }

    free(allStrings);
    free(lengthsArray);

    return tmpCreateProgramWithSource.clProgram;
}

cl_int
clBuildProgram(cl_program program,
               cl_uint num_devices,
               const cl_device_id * device_list,
               const char *options,
               void (CL_CALLBACK * pfn_notify) (cl_program program, void *user_data),
               void *user_data)
{
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();
    int optionsLen = 0;
    if (options != NULL) {
        optionsLen = strlen(options);
    }

    struct strBuildProgram tmpBuildProgram;
    MPI_Status status[4];
    MPI_Request request[4];
    int requestNo = 0;

    /* initialize structure */
    tmpBuildProgram.program = program;
    tmpBuildProgram.num_devices = num_devices;
    tmpBuildProgram.device_list = (cl_device_id *) device_list;
    tmpBuildProgram.optionLen = optionsLen;

    /* send parameters to remote node */
    MPI_Isend(&tmpBuildProgram, sizeof(tmpBuildProgram), MPI_BYTE, 0,
              BUILD_PROGRAM, slaveComm, request + (requestNo++));
    if (optionsLen > 0) {
        MPI_Isend((void *) options, optionsLen, MPI_BYTE, 0, BUILD_PROGRAM1, slaveComm,
                  request + (requestNo++));
    }
    if (device_list != NULL) {
        MPI_Isend((void *) device_list, sizeof(cl_device_id) * num_devices, MPI_BYTE, 0,
                  BUILD_PROGRAM, slaveComm, request + (requestNo++));
    }

    MPI_Irecv(&tmpBuildProgram, sizeof(tmpBuildProgram), MPI_BYTE, 0,
              BUILD_PROGRAM, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);

    return tmpBuildProgram.res;
}

cl_kernel clCreateKernel(cl_program program, const char *kernel_name, cl_int * errcode_ret)
{
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();
    int kernelNameSize = strlen(kernel_name);
    MPI_Status status[3];
    MPI_Request request[3];
    int requestNo = 0;

    struct strCreateKernel tmpCreateKernel;
    tmpCreateKernel.program = program;
    tmpCreateKernel.kernelNameSize = kernelNameSize;

    /* send input parameters to remote node */
    MPI_Isend(&tmpCreateKernel, sizeof(tmpCreateKernel), MPI_BYTE, 0,
              CREATE_KERNEL, slaveComm, request + (requestNo++));
    MPI_Isend((void *) kernel_name, kernelNameSize, MPI_CHAR, 0, CREATE_KERNEL1, slaveComm,
              request + (requestNo++));
    MPI_Irecv(&tmpCreateKernel, sizeof(tmpCreateKernel), MPI_BYTE, 0,
              CREATE_KERNEL, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);
    *errcode_ret = tmpCreateKernel.errcode_ret;

    /* create kernel info on the local node for storing arguments */
    createKernel(tmpCreateKernel.kernel);

    return tmpCreateKernel.kernel;
}

/* Memory Object APIs */
cl_mem
clCreateBuffer(cl_context context,
               cl_mem_flags flags, size_t size, void *host_ptr, cl_int * errcode_ret)
{
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    struct strCreateBuffer tmpCreateBuffer;
    MPI_Status status[3];
    MPI_Request request[3];
    int requestNo = 0;

    /* initialize structure */
    tmpCreateBuffer.context = context;
    tmpCreateBuffer.flags = flags;
    tmpCreateBuffer.size = size;
    tmpCreateBuffer.host_ptr_flag = 0;
    if (host_ptr != NULL) {
        tmpCreateBuffer.host_ptr_flag = 1;
    }

    /* send parameters to remote node */
    MPI_Isend(&tmpCreateBuffer, sizeof(tmpCreateBuffer), MPI_BYTE, 0,
              CREATE_BUFFER_FUNC, slaveComm, request + (requestNo++));
    if (tmpCreateBuffer.host_ptr_flag == 1) {
        MPI_Isend(host_ptr, size, MPI_BYTE, 0, CREATE_BUFFER_FUNC1, slaveComm,
                  request + (requestNo++));
    }
    MPI_Irecv(&tmpCreateBuffer, sizeof(tmpCreateBuffer), MPI_BYTE, 0,
              CREATE_BUFFER_FUNC, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);
    if (errcode_ret != NULL) {
        *errcode_ret = tmpCreateBuffer.errcode_ret;
    }

    return tmpCreateBuffer.deviceMem;
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
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    struct strEnqueueWriteBuffer tmpEnqueueWriteBuffer;
    MPI_Status status[4];
    MPI_Request request[4];
    int requestNo = 0;

    int bufferNum, i, bufferIndex;
    size_t remainingSize, bufferSize;

    /* initialize structure */
    tmpEnqueueWriteBuffer.command_queue = command_queue;
    tmpEnqueueWriteBuffer.buffer = buffer;
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
    MPI_Isend(&tmpEnqueueWriteBuffer, sizeof(struct strEnqueueWriteBuffer), MPI_BYTE, 0,
              ENQUEUE_WRITE_BUFFER, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);

    if (num_events_in_wait_list > 0) {
        MPI_Isend((void *) event_wait_list, sizeof(cl_event) * num_events_in_wait_list,
                  MPI_BYTE, 0, tmpEnqueueWriteBuffer.tag, slaveComm, request + (requestNo++));
    }

    bufferNum = (cb - 1) / VOCL_WRITE_BUFFER_SIZE;
    bufferSize = VOCL_WRITE_BUFFER_SIZE;
    remainingSize = cb - bufferNum * bufferSize;
    for (i = 0; i <= bufferNum; i++) {
        bufferIndex = getNextWriteBufferIndex();
        if (i == bufferNum)
            bufferSize = remainingSize;
        MPI_Isend((void *) ((char *) ptr + i * VOCL_WRITE_BUFFER_SIZE), bufferSize, MPI_BYTE,
                  0, VOCL_WRITE_TAG + bufferIndex, slaveComm, getWriteRequestPtr(bufferIndex));
        /* current buffer is used */
        setWriteBufferInUse(bufferIndex);
    }

    if (blocking_write == CL_TRUE || event != NULL) {
        MPI_Irecv(&tmpEnqueueWriteBuffer, sizeof(struct strEnqueueWriteBuffer), MPI_BYTE, 0,
                  ENQUEUE_WRITE_BUFFER, slaveComm, request + (requestNo++));
        /* for a blocking write, process all previous non-blocking ones */
        if (blocking_write == CL_TRUE) {
            processAllWrites();
        }
        else if (event != NULL) {
            processWriteBuffer(bufferIndex, bufferNum + 1);
        }

        MPI_Waitall(requestNo, request, status);
        if (event != NULL) {
            *event = tmpEnqueueWriteBuffer.event;
        }
        return tmpEnqueueWriteBuffer.res;
    }

    return CL_SUCCESS;
}

cl_int
clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value)
{
    kernel_info *kernelPtr = getKernelPtr(kernel);
    if (kernelPtr->args_allocated == 0) {
        kernelPtr->args_ptr = (kernel_args *) malloc(sizeof(kernel_args) * MAX_ARGS);
        kernelPtr->args_allocated = 1;
    }
    kernelPtr->args_ptr[kernelPtr->args_num].arg_index = arg_index;
    kernelPtr->args_ptr[kernelPtr->args_num].arg_size = arg_size;
    kernelPtr->args_ptr[kernelPtr->args_num].arg_null_flag = 1;
    if (arg_value != NULL) {
        kernelPtr->args_ptr[kernelPtr->args_num].arg_null_flag = 0;
        memcpy(kernelPtr->args_ptr[kernelPtr->args_num].arg_value, arg_value, arg_size);
    }
    kernelPtr->args_num++;

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
    MPI_Status status[7];
    MPI_Request request[7];
    int requestNo = 0;

    struct timeval t1, t2;
    float tmpTime;

    /* initialize structure */
    tmpEnqueueNDRangeKernel.command_queue = command_queue;
    tmpEnqueueNDRangeKernel.kernel = kernel;
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
    kernel_info *kernelPtr = getKernelPtr(kernel);
    tmpEnqueueNDRangeKernel.args_num = kernelPtr->args_num;
    if (event == NULL) {
        tmpEnqueueNDRangeKernel.event_null_flag = 1;
    }
    else {
        tmpEnqueueNDRangeKernel.event_null_flag = 0;
    }

    /* send parameters to remote node */
    MPI_Isend(&tmpEnqueueNDRangeKernel, sizeof(tmpEnqueueNDRangeKernel), MPI_BYTE, 0,
              ENQUEUE_ND_RANGE_KERNEL, slaveComm, request + (requestNo++));

    if (num_events_in_wait_list > 0) {
        MPI_Isend((void *) event_wait_list, sizeof(cl_event) * num_events_in_wait_list,
                  MPI_BYTE, 0, ENQUEUE_ND_RANGE_KERNEL1, slaveComm, request + (requestNo++));
    }

    if (tmpEnqueueNDRangeKernel.global_work_offset_flag == 1) {
        MPI_Isend((void *) global_work_offset, sizeof(size_t) * work_dim, MPI_BYTE, 0,
                  ENQUEUE_ND_RANGE_KERNEL1, slaveComm, request + (requestNo++));
    }

    if (tmpEnqueueNDRangeKernel.global_work_size_flag == 1) {
        MPI_Isend((void *) global_work_size, sizeof(size_t) * work_dim, MPI_BYTE, 0,
                  ENQUEUE_ND_RANGE_KERNEL2, slaveComm, request + (requestNo++));
    }

    if (tmpEnqueueNDRangeKernel.local_work_size_flag == 1) {
        MPI_Isend((void *) local_work_size, sizeof(size_t) * work_dim, MPI_BYTE, 0,
                  ENQUEUE_ND_RANGE_KERNEL3, slaveComm, request + (requestNo++));
    }

    if (kernelPtr->args_num > 0) {
        MPI_Isend((void *) kernelPtr->args_ptr, sizeof(kernel_args) * kernelPtr->args_num,
                  MPI_BYTE, 0, ENQUEUE_ND_RANGE_KERNEL4, slaveComm, request + (requestNo++));
    }
    /* arguments for current call are processed */
    kernelPtr->args_num = 0;

    MPI_Irecv(&tmpEnqueueNDRangeKernel, sizeof(tmpEnqueueNDRangeKernel), MPI_BYTE, 0,
              ENQUEUE_ND_RANGE_KERNEL, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);
    if (event != NULL) {
        *event = tmpEnqueueNDRangeKernel.event;
    }

    return tmpEnqueueNDRangeKernel.res;
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
    int tempTag;
    int requestNo = 0;
    int i, bufferIndex, bufferNum;
    size_t bufferSize = VOCL_READ_BUFFER_SIZE, remainingSize;
    bufferNum = (cb - 1) / bufferSize;
    remainingSize = cb - bufferNum * bufferSize;

    /* initialize structure */
    tmpEnqueueReadBuffer.command_queue = command_queue;
    tmpEnqueueReadBuffer.buffer = buffer;
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
    MPI_Isend(&tmpEnqueueReadBuffer, sizeof(struct strEnqueueReadBuffer), MPI_BYTE, 0,
              ENQUEUE_READ_BUFFER, slaveComm, request + (requestNo++));
    if (num_events_in_wait_list > 0) {
        MPI_Isend((void *) event_wait_list, sizeof(cl_event) * num_events_in_wait_list,
                  MPI_BYTE, 0, ENQUEUE_READ_BUFFER1, slaveComm, request + (requestNo++));
    }
    MPI_Waitall(requestNo, request, status);

    /* receive all data */
    for (i = 0; i <= bufferNum; i++) {
        if (i == bufferNum) {
            bufferSize = remainingSize;
        }
        bufferIndex = getNextReadBufferIndex();
        MPI_Irecv((void *) ((char *) ptr + VOCL_READ_BUFFER_SIZE * i), bufferSize, MPI_BYTE, 0,
                  VOCL_READ_TAG + bufferIndex, slaveComm, getReadRequestPtr(bufferIndex));
    }

    requestNo = 0;
    if (blocking_read == CL_TRUE || event != NULL) {
        MPI_Irecv(&tmpEnqueueReadBuffer, sizeof(struct strEnqueueReadBuffer), MPI_BYTE, 0,
                  ENQUEUE_READ_BUFFER, slaveComm, request + (requestNo++));
    }

    if (blocking_read == CL_TRUE) {
        processAllReads();
        MPI_Waitall(requestNo, request, status);
        if (event != NULL) {
            *event = tmpEnqueueReadBuffer.event;
        }
        return tmpEnqueueReadBuffer.res;
    }
    else {
        if (event != NULL) {
            MPI_Waitall(requestNo, request, status);
            *event = tmpEnqueueReadBuffer.event;
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
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    struct strReleaseMemObject tmpReleaseMemObject;
    tmpReleaseMemObject.memobj = memobj;

    requestNo = 0;
    MPI_Isend(&tmpReleaseMemObject, sizeof(tmpReleaseMemObject), MPI_BYTE,
              0, RELEASE_MEM_OBJ, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpReleaseMemObject, sizeof(tmpReleaseMemObject), MPI_BYTE,
              0, RELEASE_MEM_OBJ, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);

    return tmpReleaseMemObject.res;
}

cl_int clReleaseKernel(cl_kernel kernel)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;

    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    /* release kernel and parameter buffers related */
    /* to the kernel */
    releaseKernelPtr(kernel);

    /* release kernel on the remote node */
    struct strReleaseKernel tmpReleaseKernel;
    tmpReleaseKernel.kernel = kernel;
    MPI_Isend(&tmpReleaseKernel, sizeof(tmpReleaseKernel), MPI_BYTE,
              0, CL_RELEASE_KERNEL_FUNC, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpReleaseKernel, sizeof(tmpReleaseKernel), MPI_BYTE,
              0, CL_RELEASE_KERNEL_FUNC, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);
    return tmpReleaseKernel.res;
}

cl_int clFinish(cl_command_queue hInCmdQueue)
{
    MPI_Status status[2];
    MPI_Request request1, request2;
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    struct strFinish tmpFinish;
    tmpFinish.command_queue = hInCmdQueue;
    processAllWrites();
    processAllReads();


    MPI_Isend(&tmpFinish, sizeof(tmpFinish), MPI_BYTE, 0, FINISH_FUNC, slaveComm, &request1);
    MPI_Wait(&request1, status);
    MPI_Irecv(&tmpFinish, sizeof(tmpFinish), MPI_BYTE, 0, FINISH_FUNC, slaveComm, &request2);
    MPI_Wait(&request2, status);
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
    struct strGetContextInfo tmpGetContextInfo;
    tmpGetContextInfo.context = context;
    tmpGetContextInfo.param_name = param_name;
    tmpGetContextInfo.param_value_size = param_value_size;
    tmpGetContextInfo.param_value = param_value;
    tmpGetContextInfo.param_value_size_ret = 1;
    if (param_value_size_ret == NULL) {
        tmpGetContextInfo.param_value_size_ret = 0;
    }

    MPI_Isend(&tmpGetContextInfo, sizeof(tmpGetContextInfo), MPI_BYTE, 0,
              GET_CONTEXT_INFO_FUNC, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpGetContextInfo, sizeof(tmpGetContextInfo), MPI_BYTE, 0,
              GET_CONTEXT_INFO_FUNC, slaveComm, request + (requestNo++));

    if (param_value != NULL) {
        MPI_Irecv(param_value, param_value_size, MPI_BYTE, 0,
                  GET_CONTEXT_INFO_FUNC1, slaveComm, request + (requestNo++));
    }
    MPI_Waitall(requestNo, request, status);

    if (param_value_size_ret != NULL) {
        *param_value_size_ret = tmpGetContextInfo.param_value_size_ret;
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
    struct strGetProgramBuildInfo tmpGetProgramBuildInfo;
    tmpGetProgramBuildInfo.program = program;
    tmpGetProgramBuildInfo.device = device;
    tmpGetProgramBuildInfo.param_name = param_name;
    tmpGetProgramBuildInfo.param_value_size = param_value_size;
    tmpGetProgramBuildInfo.param_value = param_value;
    tmpGetProgramBuildInfo.param_value_size_ret = 1;
    if (param_value_size_ret == NULL) {
        tmpGetProgramBuildInfo.param_value_size_ret = 0;
    }

    MPI_Isend(&tmpGetProgramBuildInfo, sizeof(tmpGetProgramBuildInfo), MPI_BYTE, 0,
              GET_BUILD_INFO_FUNC, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpGetProgramBuildInfo, sizeof(tmpGetProgramBuildInfo), MPI_BYTE, 0,
              GET_BUILD_INFO_FUNC, slaveComm, request + (requestNo++));

    if (param_value != NULL) {
        MPI_Irecv(param_value, param_value_size, MPI_BYTE, 0,
                  GET_BUILD_INFO_FUNC1, slaveComm, request + (requestNo++));
    }
    MPI_Waitall(requestNo, request, status);

    if (param_value_size_ret != NULL) {
        *param_value_size_ret = tmpGetProgramBuildInfo.param_value_size_ret;
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
    struct strGetProgramInfo tmpGetProgramInfo;
    tmpGetProgramInfo.program = program;
    tmpGetProgramInfo.param_name = param_name;
    tmpGetProgramInfo.param_value_size = param_value_size;
    tmpGetProgramInfo.param_value = param_value;
    tmpGetProgramInfo.param_value_size_ret = 1;
    if (param_value_size_ret == NULL) {
        tmpGetProgramInfo.param_value_size_ret = 0;
    }

    MPI_Isend(&tmpGetProgramInfo, sizeof(tmpGetProgramInfo), MPI_BYTE, 0,
              GET_PROGRAM_INFO_FUNC, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpGetProgramInfo, sizeof(tmpGetProgramInfo), MPI_BYTE, 0,
              GET_PROGRAM_INFO_FUNC, slaveComm, request + (requestNo++));

    if (param_value != NULL) {
        MPI_Irecv(param_value, param_value_size, MPI_BYTE, 0,
                  GET_PROGRAM_INFO_FUNC1, slaveComm, request + (requestNo++));
    }
    MPI_Waitall(requestNo, request, status);

    if (param_value_size_ret != NULL) {
        *param_value_size_ret = tmpGetProgramInfo.param_value_size_ret;
    }
    return tmpGetProgramInfo.res;
}

cl_int clReleaseProgram(cl_program program)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
    struct strReleaseProgram tmpReleaseProgram;
    tmpReleaseProgram.program = program;
    MPI_Isend(&tmpReleaseProgram, sizeof(tmpReleaseProgram), MPI_BYTE, 0,
              REL_PROGRAM_FUNC, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpReleaseProgram, sizeof(tmpReleaseProgram), MPI_BYTE, 0,
              REL_PROGRAM_FUNC, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);
    return tmpReleaseProgram.res;
}

cl_int clReleaseCommandQueue(cl_command_queue command_queue)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
    struct strReleaseCommandQueue tmpReleaseCommandQueue;
    tmpReleaseCommandQueue.command_queue = command_queue;

    MPI_Isend(&tmpReleaseCommandQueue, sizeof(tmpReleaseCommandQueue), MPI_BYTE, 0,
              REL_COMMAND_QUEUE_FUNC, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpReleaseCommandQueue, sizeof(tmpReleaseCommandQueue), MPI_BYTE, 0,
              REL_COMMAND_QUEUE_FUNC, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);
    return tmpReleaseCommandQueue.res;
}

cl_int clReleaseContext(cl_context context)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
    struct strReleaseContext tmpReleaseContext;
    tmpReleaseContext.context = context;
    MPI_Isend(&tmpReleaseContext, sizeof(tmpReleaseContext), MPI_BYTE, 0,
              REL_CONTEXT_FUNC, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpReleaseContext, sizeof(tmpReleaseContext), MPI_BYTE, 0,
              REL_CONTEXT_FUNC, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);
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
    struct strGetDeviceInfo tmpGetDeviceInfo;
    tmpGetDeviceInfo.device = device;
    tmpGetDeviceInfo.param_name = param_name;
    tmpGetDeviceInfo.param_value_size = param_value_size;
    tmpGetDeviceInfo.param_value = param_value;
    tmpGetDeviceInfo.param_value_size_ret = 1;
    if (param_value_size_ret == NULL) {
        tmpGetDeviceInfo.param_value_size_ret = 0;
    }

    MPI_Isend(&tmpGetDeviceInfo, sizeof(tmpGetDeviceInfo), MPI_BYTE, 0,
              GET_DEVICE_INFO_FUNC, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpGetDeviceInfo, sizeof(tmpGetDeviceInfo), MPI_BYTE, 0,
              GET_DEVICE_INFO_FUNC, slaveComm, request + (requestNo++));

    if (param_value != NULL) {
        MPI_Irecv(param_value, param_value_size, MPI_BYTE, 0,
                  GET_DEVICE_INFO_FUNC1, slaveComm, request + (requestNo++));
    }
    MPI_Waitall(requestNo, request, status);

    if (param_value_size_ret != NULL) {
        *param_value_size_ret = tmpGetDeviceInfo.param_value_size_ret;
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

    struct strGetPlatformInfo tmpGetPlatformInfo;
    tmpGetPlatformInfo.platform = platform;
    tmpGetPlatformInfo.param_name = param_name;
    tmpGetPlatformInfo.param_value_size = param_value_size;
    tmpGetPlatformInfo.param_value = param_value;
    tmpGetPlatformInfo.param_value_size_ret = 1;
    if (param_value_size_ret == NULL) {
        tmpGetPlatformInfo.param_value_size_ret = 0;
    }

    MPI_Isend(&tmpGetPlatformInfo, sizeof(tmpGetPlatformInfo), MPI_BYTE, 0,
              GET_PLATFORM_INFO_FUNC, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpGetPlatformInfo, sizeof(tmpGetPlatformInfo), MPI_BYTE, 0,
              GET_PLATFORM_INFO_FUNC, slaveComm, request + (requestNo++));

    if (param_value != NULL) {
        MPI_Irecv(param_value, param_value_size, MPI_BYTE, 0,
                  GET_PLATFORM_INFO_FUNC1, slaveComm, request + (requestNo++));
    }
    MPI_Waitall(requestNo, request, status);

    if (param_value_size_ret != NULL) {
        *param_value_size_ret = tmpGetPlatformInfo.param_value_size_ret;
    }

    return tmpGetPlatformInfo.res;
}

cl_int clFlush(cl_command_queue hInCmdQueue)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    struct strFlush tmpFlush;
    tmpFlush.command_queue = hInCmdQueue;
    MPI_Isend(&tmpFlush, sizeof(tmpFlush), MPI_BYTE, 0,
              FLUSH_FUNC, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpFlush, sizeof(tmpFlush), MPI_BYTE, 0,
              FLUSH_FUNC, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);
    return tmpFlush.res;
}

cl_int clWaitForEvents(cl_uint num_events, const cl_event * event_list)
{
    MPI_Status status[3];
    MPI_Request request[3];
    int i, requestNo = 0;
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    struct strWaitForEvents tmpWaitForEvents;
    tmpWaitForEvents.num_events = num_events;
    MPI_Isend(&tmpWaitForEvents, sizeof(tmpWaitForEvents), MPI_BYTE, 0,
              WAIT_FOR_EVENT_FUNC, slaveComm, request + (requestNo++));
    MPI_Isend((void *) event_list, sizeof(cl_event) * num_events, MPI_BYTE, 0,
              WAIT_FOR_EVENT_FUNC1, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpWaitForEvents, sizeof(tmpWaitForEvents), MPI_BYTE, 0,
              WAIT_FOR_EVENT_FUNC, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);

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
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    struct strCreateSampler tmpCreateSampler;
    tmpCreateSampler.context = context;
    tmpCreateSampler.normalized_coords = normalized_coords;
    tmpCreateSampler.addressing_mode = addressing_mode;
    tmpCreateSampler.filter_mode = filter_mode;
    tmpCreateSampler.errcode_ret = 0;
    if (errcode_ret != NULL) {
        tmpCreateSampler.errcode_ret = 1;
    }

    MPI_Isend(&tmpCreateSampler, sizeof(tmpCreateSampler), MPI_BYTE, 0,
              CREATE_SAMPLER_FUNC, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpCreateSampler, sizeof(tmpCreateSampler), MPI_BYTE, 0,
              CREATE_SAMPLER_FUNC, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);

    if (errcode_ret != NULL) {
        *errcode_ret = tmpCreateSampler.errcode_ret;
    }

    return tmpCreateSampler.sampler;
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
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();
    struct strGetCommandQueueInfo tmpGetCommandQueueInfo;
    tmpGetCommandQueueInfo.command_queue = command_queue;
    tmpGetCommandQueueInfo.param_name = param_name;
    tmpGetCommandQueueInfo.param_value_size = param_value_size;
    tmpGetCommandQueueInfo.param_value = param_value;
    tmpGetCommandQueueInfo.param_value_size_ret = 1;
    if (param_value_size_ret == NULL) {
        tmpGetCommandQueueInfo.param_value_size_ret = 0;
    }

    MPI_Isend(&tmpGetCommandQueueInfo, sizeof(tmpGetCommandQueueInfo), MPI_BYTE, 0,
              GET_CMD_QUEUE_INFO_FUNC, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpGetCommandQueueInfo, sizeof(tmpGetCommandQueueInfo), MPI_BYTE, 0,
              GET_CMD_QUEUE_INFO_FUNC, slaveComm, request + (requestNo++));

    if (param_value != NULL) {
        MPI_Irecv(param_value, param_value_size, MPI_BYTE, 0,
                  GET_CMD_QUEUE_INFO_FUNC1, slaveComm, request + (requestNo++));
    }
    MPI_Waitall(requestNo, request, status);
    if (param_value_size_ret != NULL) {
        *param_value_size_ret = tmpGetCommandQueueInfo.param_value_size_ret;
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
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();
    struct strEnqueueMapBuffer tmpEnqueueMapBuffer;
    tmpEnqueueMapBuffer.command_queue = command_queue;
    tmpEnqueueMapBuffer.buffer = buffer;
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
    MPI_Isend(&tmpEnqueueMapBuffer, sizeof(tmpEnqueueMapBuffer), MPI_BYTE, 0,
              ENQUEUE_MAP_BUFF_FUNC, slaveComm, request + (requestNo++));
    if (num_events_in_wait_list > 0) {
        MPI_Isend((void *) event_wait_list, sizeof(cl_event) * num_events_in_wait_list,
                  MPI_BYTE, 0, ENQUEUE_MAP_BUFF_FUNC1, slaveComm, request + (requestNo++));
    }
    MPI_Irecv(&tmpEnqueueMapBuffer, sizeof(tmpEnqueueMapBuffer), MPI_BYTE, 0,
              ENQUEUE_MAP_BUFF_FUNC, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);
    if (event != NULL) {
        *event = tmpEnqueueMapBuffer.event;
    }

    if (errcode_ret != NULL) {
        *errcode_ret = tmpEnqueueMapBuffer.errcode_ret;
    }

    return tmpEnqueueMapBuffer.ret_ptr;
}

cl_int clReleaseEvent(cl_event event)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
    /* check whether the slave process is created. If not, create one. */
    checkSlaveProc();

    struct strReleaseEvent tmpReleaseEvent;
    tmpReleaseEvent.event = event;
    MPI_Isend(&tmpReleaseEvent, sizeof(tmpReleaseEvent), MPI_BYTE, 0,
              RELEASE_EVENT_FUNC, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpReleaseEvent, sizeof(tmpReleaseEvent), MPI_BYTE, 0,
              RELEASE_EVENT_FUNC, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);
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

    checkSlaveProc();

    struct strGetEventProfilingInfo tmpGetEventProfilingInfo;
    tmpGetEventProfilingInfo.event = event;
    tmpGetEventProfilingInfo.param_name = param_name;
    tmpGetEventProfilingInfo.param_value_size = param_value_size;
    tmpGetEventProfilingInfo.param_value = param_value;
    tmpGetEventProfilingInfo.param_value_size_ret = 1;
    if (param_value_size_ret == NULL) {
        tmpGetEventProfilingInfo.param_value_size_ret = 0;
    }

    MPI_Isend(&tmpGetEventProfilingInfo, sizeof(tmpGetEventProfilingInfo), MPI_BYTE, 0,
              GET_EVENT_PROF_INFO_FUNC, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpGetEventProfilingInfo, sizeof(tmpGetEventProfilingInfo), MPI_BYTE, 0,
              GET_EVENT_PROF_INFO_FUNC, slaveComm, request + (requestNo++));

    if (param_value != NULL) {
        MPI_Irecv(param_value, param_value_size, MPI_BYTE, 0,
                  GET_EVENT_PROF_INFO_FUNC1, slaveComm, request + (requestNo++));
    }
    MPI_Waitall(requestNo, request, status);
    if (param_value_size_ret != NULL) {
        *param_value_size_ret = tmpGetEventProfilingInfo.param_value_size_ret;
    }

    return tmpGetEventProfilingInfo.res;
}

cl_int clReleaseSampler(cl_sampler sampler)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;

    checkSlaveProc();
    struct strReleaseSampler tmpReleaseSampler;
    tmpReleaseSampler.sampler = sampler;
    MPI_Isend(&tmpReleaseSampler, sizeof(tmpReleaseSampler), MPI_BYTE, 0,
              RELEASE_SAMPLER_FUNC, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpReleaseSampler, sizeof(tmpReleaseSampler), MPI_BYTE, 0,
              RELEASE_SAMPLER_FUNC, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);

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

    checkSlaveProc();

    struct strGetKernelWorkGroupInfo tmpGetKernelWorkGroupInfo;
    tmpGetKernelWorkGroupInfo.kernel = kernel;
    tmpGetKernelWorkGroupInfo.device = device;
    tmpGetKernelWorkGroupInfo.param_name = param_name;
    tmpGetKernelWorkGroupInfo.param_value_size = param_value_size;
    tmpGetKernelWorkGroupInfo.param_value = param_value;
    tmpGetKernelWorkGroupInfo.param_value_size_ret = 1;
    if (param_value_size_ret == NULL) {
        tmpGetKernelWorkGroupInfo.param_value_size_ret = 0;
    }

    MPI_Isend(&tmpGetKernelWorkGroupInfo, sizeof(tmpGetKernelWorkGroupInfo), MPI_BYTE, 0,
              GET_KERNEL_WGP_INFO_FUNC, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpGetKernelWorkGroupInfo, sizeof(tmpGetKernelWorkGroupInfo), MPI_BYTE, 0,
              GET_KERNEL_WGP_INFO_FUNC, slaveComm, request + (requestNo++));

    if (param_value != NULL) {
        MPI_Irecv(param_value, param_value_size, MPI_BYTE, 0,
                  GET_KERNEL_WGP_INFO_FUNC1, slaveComm, request + (requestNo++));
    }
    MPI_Waitall(requestNo, request, status);

    if (param_value_size_ret != NULL) {
        *param_value_size_ret = tmpGetKernelWorkGroupInfo.param_value_size_ret;
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
    int requestNo = 0;

    checkSlaveProc();
    struct strCreateImage2D tmpCreateImage2D;
    tmpCreateImage2D.context = context;
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
    MPI_Isend(&tmpCreateImage2D, sizeof(tmpCreateImage2D), MPI_BYTE, 0,
              CREATE_IMAGE_2D_FUNC, slaveComm, request + (requestNo++));
    if (host_ptr != NULL) {
        MPI_Isend(host_ptr, tmpCreateImage2D.host_buff_size, MPI_BYTE, 0,
                  CREATE_IMAGE_2D_FUNC1, slaveComm, request + (requestNo++));
    }
    MPI_Irecv(&tmpCreateImage2D, sizeof(tmpCreateImage2D), MPI_BYTE, 0,
              CREATE_IMAGE_2D_FUNC, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);
    if (errcode_ret != NULL) {
        *errcode_ret = tmpCreateImage2D.errcode_ret;
    }

    return tmpCreateImage2D.mem_obj;
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

    checkSlaveProc();

    struct strEnqueueCopyBuffer tmpEnqueueCopyBuffer;
    tmpEnqueueCopyBuffer.command_queue = command_queue;
    tmpEnqueueCopyBuffer.src_buffer = src_buffer;
    tmpEnqueueCopyBuffer.dst_buffer = dst_buffer;
    tmpEnqueueCopyBuffer.src_offset = src_offset;
    tmpEnqueueCopyBuffer.dst_offset = dst_offset;
    tmpEnqueueCopyBuffer.cb = cb;
    tmpEnqueueCopyBuffer.num_events_in_wait_list = num_events_in_wait_list;
    tmpEnqueueCopyBuffer.event_null_flag = 0;
    if (event == NULL) {
        tmpEnqueueCopyBuffer.event_null_flag = 1;
    }

    MPI_Isend(&tmpEnqueueCopyBuffer, sizeof(tmpEnqueueCopyBuffer), MPI_BYTE, 0,
              ENQ_COPY_BUFF_FUNC, slaveComm, request + (requestNo++));
    if (num_events_in_wait_list > 0) {
        MPI_Isend((void *) event_wait_list, sizeof(cl_event) * num_events_in_wait_list,
                  MPI_BYTE, 0, ENQ_COPY_BUFF_FUNC1, slaveComm, request + (requestNo++));
    }
    MPI_Irecv(&tmpEnqueueCopyBuffer, sizeof(tmpEnqueueCopyBuffer), MPI_BYTE, 0,
              ENQ_COPY_BUFF_FUNC, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);
    if (event != NULL) {
        *event = tmpEnqueueCopyBuffer.event;
    }

    return tmpEnqueueCopyBuffer.res;
}

cl_int clRetainEvent(cl_event event)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
    checkSlaveProc();

    struct strRetainEvent tmpRetainEvent;
    tmpRetainEvent.event = event;
    MPI_Isend(&tmpRetainEvent, sizeof(tmpRetainEvent), MPI_BYTE, 0,
              RETAIN_EVENT_FUNC, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpRetainEvent, sizeof(tmpRetainEvent), MPI_BYTE, 0,
              RETAIN_EVENT_FUNC, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);
    return tmpRetainEvent.res;
}

cl_int clRetainMemObject(cl_mem memobj)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
    checkSlaveProc();

    struct strRetainMemObject tmpRetainMemObject;
    tmpRetainMemObject.memobj = memobj;
    MPI_Isend(&tmpRetainMemObject, sizeof(tmpRetainMemObject), MPI_BYTE, 0,
              RETAIN_MEMOBJ_FUNC, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpRetainMemObject, sizeof(tmpRetainMemObject), MPI_BYTE, 0,
              RETAIN_MEMOBJ_FUNC, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);
    return tmpRetainMemObject.res;
}

cl_int clRetainKernel(cl_kernel kernel)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
    checkSlaveProc();

    struct strRetainKernel tmpRetainKernel;
    tmpRetainKernel.kernel = kernel;
    MPI_Isend(&tmpRetainKernel, sizeof(tmpRetainKernel), MPI_BYTE, 0,
              RETAIN_KERNEL_FUNC, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpRetainKernel, sizeof(tmpRetainKernel), MPI_BYTE, 0,
              RETAIN_KERNEL_FUNC, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);
    return tmpRetainKernel.res;
}

cl_int clRetainCommandQueue(cl_command_queue command_queue)
{
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
    checkSlaveProc();

    struct strRetainCommandQueue tmpRetainCommandQueue;
    tmpRetainCommandQueue.command_queue = command_queue;
    MPI_Isend(&tmpRetainCommandQueue, sizeof(tmpRetainCommandQueue), MPI_BYTE, 0,
              RETAIN_CMDQUE_FUNC, slaveComm, request + (requestNo++));
    MPI_Irecv(&tmpRetainCommandQueue, sizeof(tmpRetainCommandQueue), MPI_BYTE, 0,
              RETAIN_CMDQUE_FUNC, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);
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
    int requestNo = 0;
    checkSlaveProc();

    struct strEnqueueUnmapMemObject tmpEnqueueUnmapMemObject;
    tmpEnqueueUnmapMemObject.command_queue = command_queue;
    tmpEnqueueUnmapMemObject.memobj = memobj;
    tmpEnqueueUnmapMemObject.mapped_ptr = mapped_ptr;
    tmpEnqueueUnmapMemObject.num_events_in_wait_list = num_events_in_wait_list;
    tmpEnqueueUnmapMemObject.event_null_flag = 0;
    if (event == NULL) {
        tmpEnqueueUnmapMemObject.event_null_flag = 1;
    }
    MPI_Isend(&tmpEnqueueUnmapMemObject, sizeof(tmpEnqueueUnmapMemObject), MPI_BYTE, 0,
              ENQ_UNMAP_MEMOBJ_FUNC, slaveComm, request + (requestNo++));
    if (num_events_in_wait_list > 0) {
        MPI_Isend((void *) event_wait_list, sizeof(cl_event) * num_events_in_wait_list,
                  MPI_BYTE, 0, ENQ_UNMAP_MEMOBJ_FUNC1, slaveComm, request + (requestNo++));
    }
    MPI_Irecv(&tmpEnqueueUnmapMemObject, sizeof(tmpEnqueueUnmapMemObject), MPI_BYTE, 0,
              ENQ_UNMAP_MEMOBJ_FUNC, slaveComm, request + (requestNo++));
    MPI_Waitall(requestNo, request, status);
    if (event != NULL) {
        *event = tmpEnqueueUnmapMemObject.event;
    }

    return tmpEnqueueUnmapMemObject.res;
}
