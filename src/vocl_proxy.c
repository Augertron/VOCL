#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <memory.h>
#include <CL/opencl.h>
#include <sched.h>
#include "vocl_proxy.h"
#include "vocl_proxyBufferProc.h"
#include "vocl_proxyKernelArgProc.h"

#define _PRINT_NODE_NAME

struct strGetPlatformIDs           getPlatformIDStr;
struct strGetDeviceIDs             tmpGetDeviceIDs;
struct strCreateContext            tmpCreateContext;
struct strCreateCommandQueue       tmpCreateCommandQueue;
struct strCreateProgramWithSource  tmpCreateProgramWithSource;
struct strBuildProgram             tmpBuildProgram;
struct strCreateKernel             tmpCreateKernel;
struct strCreateBuffer             tmpCreateBuffer;
struct strEnqueueWriteBuffer       tmpEnqueueWriteBuffer;
struct strSetKernelArg             tmpSetKernelArg;
struct strEnqueueNDRangeKernel     tmpEnqueueNDRangeKernel;
struct strEnqueueReadBuffer        tmpEnqueueReadBuffer;
struct strReleaseMemObject         tmpReleaseMemObject;
struct strReleaseKernel            tmpReleaseKernel;
struct strGetContextInfo           tmpGetContextInfo;
struct strGetProgramBuildInfo      tmpGetProgramBuildInfo;
struct strGetProgramInfo           tmpGetProgramInfo;
struct strReleaseProgram           tmpReleaseProgram;
struct strReleaseCommandQueue      tmpReleaseCommandQueue;
struct strReleaseContext           tmpReleaseContext;
struct strFinish                   tmpFinish;
struct strGetDeviceInfo            tmpGetDeviceInfo;
struct strGetPlatformInfo          tmpGetPlatformInfo;
struct strFlush                    tmpFlush;
struct strWaitForEvents            tmpWaitForEvents;
struct strCreateSampler            tmpCreateSampler;
struct strGetCommandQueueInfo      tmpGetCommandQueueInfo;
struct strEnqueueMapBuffer         tmpEnqueueMapBuffer;
struct strReleaseEvent             tmpReleaseEvent;
struct strGetEventProfilingInfo    tmpGetEventProfilingInfo;
struct strReleaseSampler           tmpReleaseSampler;
struct strGetKernelWorkGroupInfo   tmpGetKernelWorkGroupInfo;
struct strCreateImage2D            tmpCreateImage2D;
struct strEnqueueCopyBuffer        tmpEnqueueCopyBuffer;
struct strRetainEvent              tmpRetainEvent;
struct strRetainMemObject          tmpRetainMemObject;
struct strRetainKernel             tmpRetainKernel;
struct strRetainCommandQueue       tmpRetainCommandQueue;
struct strEnqueueUnmapMemObject    tmpEnqueueUnmapMemObject;

/* control message pointer */
MPI_Request *conMsgRequest;

/* variables needed by the helper thread */
extern void              *proxyHelperThread(void *);
extern int               writeBufferIndexInHelperThread;
extern int               helperThreadOperFlag;
extern pthread_barrier_t barrier;
extern pthread_t         th;

/* variables from write buffer pool */
extern int totalRequestNum;
extern int allWritesAreEnqueuedFlag;
extern int allReadBuffersAreCovered;

/* functions from write buffer pool */
extern void        initializeWriteBuffer();
extern void        increaseWriteBufferCount();
extern void        finalizeWriteBuffer();
extern void        setWriteBufferFlag(int index, int flag);
extern int         getNextWriteBufferIndex();
extern MPI_Request *getWriteRequestPtr(int index);
extern struct strWriteBufferInfo *getWriteBufferInfoPtr(int index);
extern cl_int      processWriteBuffer(int curIndex, int bufferNum);
extern cl_int      processAllWrites();
extern int         getWriteBufferIndexFromEvent(cl_event event);

/* functions from read buffer pool */
extern void        initializeReadBuffer();
extern void        finalizeReadBuffer();
extern MPI_Request *getReadRequestPtr(int index);
extern struct strReadBufferInfo *getReadBufferInfoPtr(int index);
extern int         readSendToLocal(int index);
extern void        setReadBufferFlag(int index, int flag);
extern int         getNextReadBufferIndex();
extern cl_int      processReadBuffer(int curIndex, int bufferNum);
extern int         getReadBufferIndexFromEvent(cl_event event);
extern cl_int      processAllReads();

/*functions for calling actual OpenCL function */
extern void mpiOpenCLGetPlatformIDs(struct strGetPlatformIDs *tmpGetPlatform, cl_platform_id *platforms);
extern void mpiOpenCLGetDeviceIDs(struct strGetDeviceIDs *tmpGetDeviceIDs, 
						   cl_device_id *devices);
extern void mpiOpenCLCreateContext(struct strCreateContext *tmpCreateContext, cl_device_id *devices);
extern void mpiOpenCLCreateCommandQueue(struct strCreateCommandQueue *tmpCreateCommandQueue);
extern void mpiOpenCLCreateProgramWithSource(struct strCreateProgramWithSource *tmpCreateProgramWithSource, 
									  char *cSourceCL, size_t *lengthsArray);
extern void mpiOpenCLBuildProgram(struct strBuildProgram *tmpBuildProgram, 
						   char *options, cl_device_id *devices);
extern void mpiOpenCLCreateKernel(struct strCreateKernel *tmpCreateKernel, char *kernel_name);
extern void mpiOpenCLCreateBuffer(struct strCreateBuffer *tmpCreateBuffer, void *host_ptr);
extern void mpiOpenCLEnqueueWriteBuffer(struct strEnqueueWriteBuffer *tmpEnqueueWriteBuffer,
								 void *ptr, cl_event *event_wait_list);
extern void mpiOpenCLSetKernelArg(struct strSetKernelArg *tmpSetKernelArg, void *arg_value);
extern void mpiOpenCLEnqueueNDRangeKernel(struct strEnqueueNDRangeKernel *tmpEnqueueNDRangeKernel,
							  cl_event   *event_wait_list,
							  size_t     *global_work_offset,
							  size_t     *global_work_size,
							  size_t     *local_work_size,
							  kernel_args *args_ptr);
extern void mpiOpenCLEnqueueReadBuffer(struct strEnqueueReadBuffer *tmpEnqueueReadBuffer, 
								void *ptr, cl_event *event_wait_list);
extern void mpiOpenCLReleaseMemObject(struct strReleaseMemObject *tmpReleaseMemObject);
extern void mpiOpenCLReleaseKernel(struct strReleaseKernel *tmpReleaseKernel);
extern void mpiOpenCLGetContextInfo(struct strGetContextInfo *tmpGetContextInfo, void *param_value);
extern void mpiOpenCLGetProgramBuildInfo(struct strGetProgramBuildInfo *tmpGetProgramBuildInfo, void *param_value);
extern void mpiOpenCLGetProgramInfo(struct strGetProgramInfo *tmpGetProgramInfo, void *param_value);
extern void mpiOpenCLReleaseProgram(struct strReleaseProgram *tmpReleaseProgram);
extern void mpiOpenCLReleaseCommandQueue(struct strReleaseCommandQueue *tmpReleaseCommandQueue);
extern void mpiOpenCLReleaseContext(struct strReleaseContext *tmpReleaseContext);
extern void mpiOpenCLFinish(struct strFinish *tmpFinish);
extern void mpiOpenCLGetDeviceInfo(struct strGetDeviceInfo *tmpGetDeviceInfo, void *param_value);
extern void mpiOpenCLGetPlatformInfo(struct strGetPlatformInfo *tmpGetPlatformInfo, void *param_value);
extern void mpiOpenCLFlush(struct strFlush *tmpFlush);
extern void mpiOpenCLWaitForEvents(struct strWaitForEvents *tmpWaitForEvents, cl_event *event_list);
extern void mpiOpenCLCreateSampler(struct strCreateSampler *tmpCreateSampler);
extern void mpiOpenCLGetCommandQueueInfo(struct strGetCommandQueueInfo *tmpGetCommandQueueInfo,
								  void *param_value);
extern void mpiOpenCLEnqueueMapBuffer(struct strEnqueueMapBuffer *tmpEnqueueMapBuffer,
							   cl_event * event_wait_list);
extern void mpiOpenCLReleaseEvent(struct strReleaseEvent *tmpReleaseEvent);
extern void mpiOpenCLGetEventProfilingInfo(struct strGetEventProfilingInfo *tmpGetEventProfilingInfo,
								  void *param_value);
extern void mpiOpenCLReleaseSampler(struct strReleaseSampler *tmpReleaseSampler);
extern void mpiOpenCLGetKernelWorkGroupInfo(struct strGetKernelWorkGroupInfo *tmpGetKernelWorkGroupInfo,
								  void *param_value);
extern void mpiOpenCLCreateImage2D(struct strCreateImage2D *tmpCreateImage2D, void *host_ptr);
extern void mpiOpenCLEnqueueCopyBuffer(struct strEnqueueCopyBuffer *tmpEnqueueCopyBuffer, 
								cl_event * event_wait_list);
extern void mpiOpenCLRetainEvent(struct strRetainEvent *tmpRetainEvent);
extern void mpiOpenCLRetainMemObject(struct strRetainMemObject *tmpRetainMemObject);
extern void mpiOpenCLRetainKernel(struct strRetainKernel *tmpRetainKernel);
extern void mpiOpenCLRetainCommandQueue(struct strRetainCommandQueue *tmpRetainCommandQueue);
extern void mpiOpenCLEnqueueUnmapMemObject(struct strEnqueueUnmapMemObject *tmpEnqueueUnmapMemObject, 
									cl_event *event_wait_list);

/* proxy process */
int main(int argc, char *argv[])
{
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET(8, &set);
	sched_setaffinity(0, sizeof(set), &set);

	int rank, i;
	cl_int err;
	MPI_Status status;
	MPI_Request request;
	MPI_Comm parentComm;
	MPI_Init(&argc, &argv);
	MPI_Comm_get_parent(&parentComm);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef _PRINT_NODE_NAME
	char hostName[200];
	int  len;
	MPI_Get_processor_name(hostName, &len);
	hostName[len] = '\0';
	printf("slaveHostName = %s\n", hostName);
#endif

	/* issue non-blocking receive for all control messages */
	MPI_Status  *curStatus;
	MPI_Request *curRequest;
	int requestNo, dataIndex = 0, index;
	char *conMsgBuffer[CMSG_NUM];
	int bufferNum, bufferIndex;
	size_t bufferSize, remainingSize;

	/* variables used by OpenCL API function */
	cl_platform_id *platforms;
	cl_device_id *devices;
	cl_uint num_entries;
	cl_event *event_wait_list;
	cl_uint num_events_in_wait_list;

	struct strWriteBufferInfo *writeBufferInfoPtr;
	struct strReadBufferInfo *readBufferInfoPtr;

	size_t *lengthsArray;
	size_t fileSize;
	char *fileBuffer;
	char *buildOptionBuffer;
	char *kernelName;
	void *host_ptr;
	void *arg_value;
	int work_dim;

	size_t *global_work_offset, *global_work_size, *local_work_size;
	kernel_args *args_ptr;
	size_t param_value_size;
	void *param_value;
	cl_uint num_events;
	cl_event *event_list;
	size_t host_buff_size;


	curStatus = (MPI_Status *)malloc(sizeof(MPI_Status) * TOTAL_MSG_NUM);
	conMsgRequest = (MPI_Request *)malloc(sizeof(MPI_Request) * TOTAL_MSG_NUM);
	curRequest = (MPI_Request *)malloc(sizeof(MPI_Request) * TOTAL_MSG_NUM);

	/* initialize write and read buffer pools */
	initializeWriteBuffer();
	initializeReadBuffer();

	/* create a helper thread */
    pthread_barrier_init(&barrier, NULL, 2);
	pthread_create(&th, NULL, proxyHelperThread, NULL);

	for (i = 0; i < CMSG_NUM; i++)
	{
		/* allocate buffer for each contral message */
		conMsgBuffer[i] = (char *)malloc(MAX_CMSG_SIZE);
		MPI_Irecv(conMsgBuffer[i], MAX_CMSG_SIZE, MPI_BYTE, 0, i+OFFSET,
				  parentComm, conMsgRequest+i);
	}

	while (1)
	{
		/* wait for any msg from the master process */
		MPI_Waitany(totalRequestNum, conMsgRequest, &index, &status);

		if (status.MPI_TAG == GET_PLATFORM_ID_FUNC)
		{
			memcpy((void *)&getPlatformIDStr, (const void *)conMsgBuffer[index], sizeof(getPlatformIDStr));
			
			platforms = NULL;
			if (getPlatformIDStr.platforms != NULL)
			{
				platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * getPlatformIDStr.num_entries);
			}

			mpiOpenCLGetPlatformIDs(&getPlatformIDStr, platforms);
			requestNo = 0;
			MPI_Isend(&getPlatformIDStr, sizeof(getPlatformIDStr), MPI_BYTE, 0,
					 GET_PLATFORM_ID_FUNC, parentComm, curRequest + (requestNo++));
			if (getPlatformIDStr.platforms != NULL && getPlatformIDStr.num_entries > 0)
			{
				MPI_Isend((void *)platforms, sizeof(cl_platform_id) * getPlatformIDStr.num_entries, MPI_BYTE, 0,
						 GET_PLATFORM_ID_FUNC1, parentComm, curRequest + (requestNo++));
			}

			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, GET_PLATFORM_ID_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Waitall(requestNo, curRequest, curStatus);
			if (getPlatformIDStr.platforms != NULL && getPlatformIDStr.num_entries > 0)
			{
				free(platforms);
			}
		}

		if (status.MPI_TAG == GET_DEVICE_ID_FUNC)
		{
			memcpy(&tmpGetDeviceIDs, conMsgBuffer[index], sizeof(tmpGetDeviceIDs));
			devices = NULL;
			num_entries = tmpGetDeviceIDs.num_entries;
			if (num_entries > 0 && tmpGetDeviceIDs.devices != NULL)
			{
				devices = (cl_device_id *)malloc(num_entries * sizeof(cl_device_id));
			}
			mpiOpenCLGetDeviceIDs(&tmpGetDeviceIDs, devices);
			requestNo = 0;
			if (num_entries > 0 && tmpGetDeviceIDs.devices != NULL)
			{
				MPI_Isend(devices, sizeof(cl_device_id) * num_entries, MPI_BYTE, 0,
					GET_DEVICE_ID_FUNC1, parentComm, curRequest + (requestNo++));
			}
			MPI_Isend(&tmpGetDeviceIDs, sizeof(tmpGetDeviceIDs), MPI_BYTE, 0,
					GET_DEVICE_ID_FUNC, parentComm, curRequest + (requestNo++));

			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, GET_DEVICE_ID_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Waitall(requestNo, curRequest, curStatus);
			if (num_entries > 0 && tmpGetDeviceIDs.devices != NULL)
			{
				free(devices);
			}

		}

		if (status.MPI_TAG == CREATE_CONTEXT_FUNC)
		{
			memcpy(&tmpCreateContext, conMsgBuffer[index], sizeof(tmpCreateContext));
			devices = NULL;
			if (tmpCreateContext.devices != NULL)
			{
				devices = (cl_device_id *)malloc(sizeof(cl_device_id) * tmpCreateContext.num_devices);
				MPI_Irecv(devices, sizeof(cl_device_id) * tmpCreateContext.num_devices, MPI_BYTE, 0,
						 CREATE_CONTEXT_FUNC1, parentComm, curRequest);
				MPI_Wait(curRequest, curStatus);
			}

			mpiOpenCLCreateContext(&tmpCreateContext, devices);

			MPI_Isend(&tmpCreateContext, sizeof(tmpCreateContext), MPI_BYTE, 0,
					CREATE_CONTEXT_FUNC, parentComm, curRequest);

			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, CREATE_CONTEXT_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Wait(curRequest, curStatus);
			if (devices != NULL)
			{
				free(devices);
			}
		}

		if (status.MPI_TAG == CREATE_COMMAND_QUEUE_FUNC)
		{
			memcpy(&tmpCreateCommandQueue, conMsgBuffer[index], sizeof(tmpCreateCommandQueue));
			mpiOpenCLCreateCommandQueue(&tmpCreateCommandQueue);

			MPI_Isend(&tmpCreateCommandQueue, sizeof(tmpCreateCommandQueue), MPI_BYTE, 0,
					 CREATE_COMMAND_QUEUE_FUNC, parentComm, curRequest);

			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, CREATE_COMMAND_QUEUE_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Wait(curRequest, curStatus);
		}

		if (status.MPI_TAG == CREATE_PROGRMA_WITH_SOURCE)
		{
			memcpy(&tmpCreateProgramWithSource, conMsgBuffer[index], sizeof(tmpCreateProgramWithSource));
			cl_uint count = tmpCreateProgramWithSource.count;
			lengthsArray = (size_t *)malloc(count * sizeof(size_t));
			
			fileSize = tmpCreateProgramWithSource.lengths;
			fileBuffer = (char *)malloc(fileSize * sizeof(char));

			requestNo = 0;
			MPI_Irecv(lengthsArray, count * sizeof(size_t), MPI_BYTE, 0,
					 CREATE_PROGRMA_WITH_SOURCE1, parentComm, curRequest+(requestNo++));
			MPI_Irecv(fileBuffer, fileSize, MPI_BYTE, 0,
					 CREATE_PROGRMA_WITH_SOURCE2, parentComm, curRequest+(requestNo++));
			MPI_Waitall(requestNo, curRequest, curStatus);

			mpiOpenCLCreateProgramWithSource(&tmpCreateProgramWithSource, fileBuffer, lengthsArray);
			MPI_Isend(&tmpCreateProgramWithSource, sizeof(tmpCreateProgramWithSource), MPI_BYTE, 0,
					 CREATE_PROGRMA_WITH_SOURCE, parentComm, curRequest);

			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, CREATE_PROGRMA_WITH_SOURCE,
					  parentComm, conMsgRequest + index);
			free(fileBuffer);
			free(lengthsArray);
			MPI_Wait(curRequest, curStatus);
		}

		if (status.MPI_TAG == BUILD_PROGRAM)
		{
			memcpy(&tmpBuildProgram, conMsgBuffer[index], sizeof(tmpBuildProgram));
			buildOptionBuffer = NULL;
			requestNo = 0;
			if (tmpBuildProgram.optionLen > 0)
			{
				buildOptionBuffer = (char *)malloc((tmpBuildProgram.optionLen + 1) * sizeof(char));
				MPI_Irecv(buildOptionBuffer, tmpBuildProgram.optionLen, MPI_BYTE, 0,
					     BUILD_PROGRAM1, parentComm, curRequest + (requestNo++));
				buildOptionBuffer[tmpBuildProgram.optionLen] = '\0';
			}

			devices = NULL;
			if (tmpBuildProgram.device_list != NULL)
			{
				devices = (cl_device_id *)malloc(sizeof(cl_device_id) * tmpBuildProgram.num_devices);
				MPI_Irecv(devices, sizeof(cl_device_id) * tmpBuildProgram.num_devices, MPI_BYTE, 0,
						 BUILD_PROGRAM, parentComm, curRequest + (requestNo++));
			}
			if (requestNo > 0)
			{
				MPI_Waitall(requestNo, curRequest, curStatus);
			}

			mpiOpenCLBuildProgram(&tmpBuildProgram, buildOptionBuffer, devices);
			MPI_Isend(&tmpBuildProgram, sizeof(tmpBuildProgram), MPI_BYTE, 0,
					 BUILD_PROGRAM, parentComm, curRequest);
			
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, BUILD_PROGRAM,
					  parentComm, conMsgRequest + index);
			if (tmpBuildProgram.optionLen > 0)
			{
				free(buildOptionBuffer);
			}
			if (tmpBuildProgram.device_list != NULL)
			{
				free(devices);
			}
			MPI_Wait(curRequest, curStatus);
		}

		if (status.MPI_TAG == CREATE_KERNEL)
		{
			memcpy(&tmpCreateKernel, conMsgBuffer[index], sizeof(tmpCreateKernel));
			kernelName = (char *)malloc((tmpCreateKernel.kernelNameSize + 1)* sizeof(char));
			MPI_Irecv(kernelName, tmpCreateKernel.kernelNameSize, MPI_CHAR, 0,
					 CREATE_KERNEL1, parentComm, curRequest);
			kernelName[tmpCreateKernel.kernelNameSize] = '\0';
			MPI_Wait(curRequest, curStatus);
			mpiOpenCLCreateKernel(&tmpCreateKernel, kernelName);
			MPI_Isend(&tmpCreateKernel, sizeof(tmpCreateKernel), MPI_BYTE, 0,
					 CREATE_KERNEL, parentComm, curRequest);
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, CREATE_KERNEL,
					  parentComm, conMsgRequest + index);
			free(kernelName);
			MPI_Wait(curRequest, curStatus);
		}

		if (status.MPI_TAG == CREATE_BUFFER_FUNC)
		{
			memcpy(&tmpCreateBuffer, conMsgBuffer[index], sizeof(tmpCreateBuffer));
			host_ptr = NULL;
			if (tmpCreateBuffer.host_ptr_flag == 1)
			{
				host_ptr = malloc(tmpCreateBuffer.size);
				MPI_Irecv(host_ptr, tmpCreateBuffer.size, MPI_BYTE, 0, 
						 CREATE_BUFFER_FUNC1, parentComm, curRequest);
				MPI_Wait(curRequest, curStatus);
			}
			mpiOpenCLCreateBuffer(&tmpCreateBuffer, host_ptr);
			MPI_Isend(&tmpCreateBuffer, sizeof(tmpCreateBuffer), MPI_BYTE, 0,
					 CREATE_BUFFER_FUNC, parentComm, curRequest);
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, CREATE_BUFFER_FUNC,
					  parentComm, conMsgRequest + index);
			if (tmpCreateBuffer.host_ptr_flag == 1)
			{
				free(host_ptr);
			}
			MPI_Wait(curRequest, curStatus);
		}

		if (status.MPI_TAG == ENQUEUE_WRITE_BUFFER)
		{
			memcpy(&tmpEnqueueWriteBuffer, conMsgBuffer[index], sizeof(tmpEnqueueWriteBuffer));
			requestNo = 0;
			event_wait_list = NULL;
			num_events_in_wait_list = tmpEnqueueWriteBuffer.num_events_in_wait_list;
			if (num_events_in_wait_list > 0)
			{
				event_wait_list = (cl_event *)malloc(sizeof(cl_event) * num_events_in_wait_list);
				MPI_Irecv(event_wait_list, sizeof(cl_event) * num_events_in_wait_list, MPI_BYTE, 0,
						 tmpEnqueueWriteBuffer.tag, parentComm, curRequest+(requestNo++));
			}

			/* issue MPI data receive */
			bufferSize = VOCL_PROXY_WRITE_BUFFER_SIZE;
			bufferNum = (tmpEnqueueWriteBuffer.cb - 1) / bufferSize;
			remainingSize = tmpEnqueueWriteBuffer.cb - bufferSize * bufferNum;
			for (i = 0; i <= bufferNum; i++)
			{
				if (i == bufferNum) bufferSize = remainingSize;
				bufferIndex = getNextWriteBufferIndex();
				writeBufferInfoPtr = getWriteBufferInfoPtr(bufferIndex);
				MPI_Irecv(writeBufferInfoPtr->dataPtr, bufferSize, MPI_BYTE, 0,
						  VOCL_PROXY_WRITE_TAG + bufferIndex, parentComm, getWriteRequestPtr(bufferIndex));

				/* save information for writing to GPU memory */
				writeBufferInfoPtr->commandQueue = tmpEnqueueWriteBuffer.command_queue;
				writeBufferInfoPtr->size = bufferSize;
				writeBufferInfoPtr->offset = tmpEnqueueWriteBuffer.offset + i * VOCL_PROXY_WRITE_BUFFER_SIZE;
				writeBufferInfoPtr->mem = tmpEnqueueWriteBuffer.buffer;
				writeBufferInfoPtr->blocking_write = tmpEnqueueWriteBuffer.blocking_write;
				writeBufferInfoPtr->numEvents = tmpEnqueueWriteBuffer.num_events_in_wait_list;
				writeBufferInfoPtr->eventWaitList = event_wait_list;

				/* set flag to indicate buffer is being used */
				setWriteBufferFlag(bufferIndex, WRITE_RECV_DATA);
				increaseWriteBufferCount();
			}
			allWritesAreEnqueuedFlag = 0;

			if (tmpEnqueueWriteBuffer.blocking_write == CL_TRUE)
			{
				if (requestNo > 0)
				{
					MPI_Waitall(requestNo, curRequest, curStatus);
					requestNo = 0;
				}

				/* process all previous write and read */
				tmpEnqueueWriteBuffer.res = processAllWrites();
				tmpEnqueueWriteBuffer.event = writeBufferInfoPtr->event;

				if (num_events_in_wait_list > 0)
				{
					free(event_wait_list);
				}

				MPI_Isend(&tmpEnqueueWriteBuffer, sizeof(tmpEnqueueWriteBuffer), MPI_BYTE, 0,
						  ENQUEUE_WRITE_BUFFER, parentComm, curRequest+(requestNo++));
			}
			else if (tmpEnqueueWriteBuffer.event_null_flag == 0)
			{
				if (requestNo > 0) 
				{
					MPI_Waitall(requestNo, curRequest, curStatus);
					requestNo = 0;
				}
				tmpEnqueueWriteBuffer.res = processWriteBuffer(bufferIndex, bufferNum + 1);
				tmpEnqueueWriteBuffer.event = writeBufferInfoPtr->event;
				writeBufferInfoPtr->numWriteBuffers = bufferNum + 1;

				MPI_Isend(&tmpEnqueueWriteBuffer, sizeof(tmpEnqueueWriteBuffer), MPI_BYTE, 0,
						  ENQUEUE_WRITE_BUFFER, parentComm, curRequest+(requestNo++));
			}

			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, ENQUEUE_WRITE_BUFFER,
					  parentComm, conMsgRequest + index);
			if (requestNo > 0)
			{
				MPI_Wait(curRequest, curStatus);
			}
		}

		if (status.MPI_TAG >= VOCL_PROXY_WRITE_TAG && 
			status.MPI_TAG < VOCL_PROXY_WRITE_TAG + VOCL_PROXY_WRITE_BUFFER_NUM)
		{
			writeBufferIndexInHelperThread = status.MPI_TAG - VOCL_PROXY_WRITE_TAG;
			pthread_barrier_wait(&barrier);
			helperThreadOperFlag = GPU_WRITE_SINGLE;
			pthread_barrier_wait(&barrier);
		}
		
		if (status.MPI_TAG == SET_KERNEL_ARG)
		{
			memcpy(&tmpSetKernelArg, conMsgBuffer[index], sizeof(tmpSetKernelArg));
			arg_value = NULL;
			if (tmpSetKernelArg.arg_value != NULL)
			{
				arg_value = (char *)malloc(tmpSetKernelArg.arg_size);
				MPI_Irecv(arg_value, tmpSetKernelArg.arg_size, MPI_BYTE, 0,
						 SET_KERNEL_ARG1, parentComm, curRequest);
			}
			MPI_Wait(curRequest, curStatus);
			mpiOpenCLSetKernelArg(&tmpSetKernelArg, arg_value);
			MPI_Isend(&tmpSetKernelArg, sizeof(tmpSetKernelArg), MPI_BYTE, 0,
					 SET_KERNEL_ARG, parentComm, curRequest);
			if (tmpSetKernelArg.arg_value != NULL)
			{
				free(arg_value);
			}
			MPI_Wait(curRequest, curStatus);
		}

		if (status.MPI_TAG == ENQUEUE_ND_RANGE_KERNEL)
		{
			memcpy(&tmpEnqueueNDRangeKernel, conMsgBuffer[index], sizeof(tmpEnqueueNDRangeKernel));
			requestNo = 0;

			event_wait_list = NULL;
			num_events_in_wait_list = tmpEnqueueNDRangeKernel.num_events_in_wait_list;
			if (num_events_in_wait_list > 0)
			{
				event_wait_list = (cl_event *)malloc(sizeof(cl_event) * num_events_in_wait_list);
				MPI_Irecv(event_wait_list, sizeof(cl_event) * num_events_in_wait_list, MPI_BYTE, 0,
						 ENQUEUE_ND_RANGE_KERNEL1, parentComm, curRequest+(requestNo++));
			}

			work_dim = tmpEnqueueNDRangeKernel.work_dim;
			args_ptr = NULL;
			global_work_offset = NULL;
			global_work_size   = NULL;
			local_work_size    = NULL;

			if (tmpEnqueueNDRangeKernel.global_work_offset_flag == 1)
			{
				global_work_offset = (size_t *)malloc(work_dim * sizeof(size_t));
				MPI_Irecv(global_work_offset, work_dim * sizeof(size_t), MPI_BYTE, 0,
						 ENQUEUE_ND_RANGE_KERNEL1, parentComm, curRequest+(requestNo++));
			}

			if (tmpEnqueueNDRangeKernel.global_work_size_flag == 1)
			{
				global_work_size   = (size_t *)malloc(work_dim * sizeof(size_t));
				MPI_Irecv(global_work_size, work_dim * sizeof(size_t), MPI_BYTE, 0,
						 ENQUEUE_ND_RANGE_KERNEL2, parentComm, curRequest+(requestNo++));
			}

			if (tmpEnqueueNDRangeKernel.local_work_size_flag == 1)
			{
				local_work_size    = (size_t *)malloc(work_dim * sizeof(size_t));
				MPI_Irecv(local_work_size, work_dim * sizeof(size_t), MPI_BYTE, 0,
						 ENQUEUE_ND_RANGE_KERNEL3, parentComm, curRequest+(requestNo++));
			}

			if (tmpEnqueueNDRangeKernel.args_num > 0)
			{
				args_ptr = (kernel_args *)malloc(tmpEnqueueNDRangeKernel.args_num * sizeof(kernel_args));
				MPI_Irecv(args_ptr, tmpEnqueueNDRangeKernel.args_num * sizeof(kernel_args), MPI_BYTE, 0,
						 ENQUEUE_ND_RANGE_KERNEL4, parentComm, curRequest+(requestNo++));
			}
			MPI_Waitall(requestNo, curRequest, curStatus);

			/* if there are data received, but not write to */
			/* the GPU memory yet, use the helper thread to */
			/* wait MPI receive complete and write to the GPU memory */
			if (allWritesAreEnqueuedFlag == 0)
			{
				pthread_barrier_wait(&barrier);
				helperThreadOperFlag = GPU_ENQ_WRITE;
				pthread_barrier_wait(&barrier);
				pthread_barrier_wait(&barrier);
			}

			mpiOpenCLEnqueueNDRangeKernel(&tmpEnqueueNDRangeKernel,
										  event_wait_list,
										  global_work_offset,
										  global_work_size,
										  local_work_size,
										  args_ptr);

			MPI_Isend(&tmpEnqueueNDRangeKernel, sizeof(tmpEnqueueNDRangeKernel), MPI_BYTE, 0,
					 ENQUEUE_ND_RANGE_KERNEL, parentComm, curRequest);
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, ENQUEUE_ND_RANGE_KERNEL,
					  parentComm, conMsgRequest + index);

			if (tmpEnqueueNDRangeKernel.global_work_offset_flag == 1)
			{
				free(global_work_offset);
			}

			if (tmpEnqueueNDRangeKernel.global_work_size_flag == 1)
			{
				free(global_work_size);
			}

			if (tmpEnqueueNDRangeKernel.local_work_size_flag == 1)
			{
				free(local_work_size);
			}

			if (tmpEnqueueNDRangeKernel.args_num > 0)
			{
				free(args_ptr);
			}

			if (num_events_in_wait_list > 0)
			{
				free(event_wait_list);
			}

			MPI_Wait(curRequest, curStatus);
		}

		if (status.MPI_TAG == ENQUEUE_READ_BUFFER)
		{
			memcpy(&tmpEnqueueReadBuffer, conMsgBuffer[index], sizeof(tmpEnqueueReadBuffer));
			num_events_in_wait_list = tmpEnqueueReadBuffer.num_events_in_wait_list;
			event_wait_list = NULL;
			if (num_events_in_wait_list > 0)
			{
				event_wait_list = (cl_event *)malloc(num_events_in_wait_list * sizeof(cl_event));
				MPI_Irecv(event_wait_list, num_events_in_wait_list * sizeof(cl_event), MPI_BYTE, 0,
						 ENQUEUE_READ_BUFFER1, parentComm, curRequest);
				MPI_Wait(curRequest, curStatus);
			}

			bufferSize = VOCL_PROXY_READ_BUFFER_SIZE;
			bufferNum = (tmpEnqueueReadBuffer.cb - 1) / VOCL_PROXY_READ_BUFFER_SIZE;
			remainingSize = tmpEnqueueReadBuffer.cb - bufferSize * bufferNum;
			for (i = 0; i <= bufferNum; i++)
			{
				bufferIndex = getNextReadBufferIndex();
				if (i == bufferNum) bufferSize = remainingSize;
				readBufferInfoPtr = getReadBufferInfoPtr(bufferIndex);
				readBufferInfoPtr->comm = parentComm;
				readBufferInfoPtr->tag = VOCL_PROXY_READ_TAG + bufferIndex;
				readBufferInfoPtr->size = bufferSize;
				tmpEnqueueReadBuffer.res = 
					clEnqueueReadBuffer(tmpEnqueueReadBuffer.command_queue,
										tmpEnqueueReadBuffer.buffer,
										CL_FALSE,
										tmpEnqueueReadBuffer.offset + i * VOCL_PROXY_READ_BUFFER_SIZE,
										bufferSize,
										readBufferInfoPtr->dataPtr,
										tmpEnqueueReadBuffer.num_events_in_wait_list,
										event_wait_list,
										&readBufferInfoPtr->event);
				setReadBufferFlag(bufferIndex, READ_GPU_MEM);
			}
			readBufferInfoPtr->numReadBuffers = bufferNum + 1;

			/* some new read requests are issued */
			allReadBuffersAreCovered = 0;

			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, ENQUEUE_READ_BUFFER,
					  parentComm, conMsgRequest + index);

			if (tmpEnqueueReadBuffer.blocking_read == CL_FALSE)
			{
				if (tmpEnqueueReadBuffer.event_null_flag == 0)
				{
					tmpEnqueueReadBuffer.event = readBufferInfoPtr->event;
					MPI_Isend(&tmpEnqueueReadBuffer, sizeof(tmpEnqueueReadBuffer), MPI_BYTE, 0,
							 ENQUEUE_READ_BUFFER, parentComm, curRequest);
					MPI_Wait(curRequest, curStatus);
				}
			}
			else /* blocking, reading is complete, send data to local node */
			{
				tmpEnqueueReadBuffer.res = processAllReads();
				if (tmpEnqueueReadBuffer.event_null_flag == 0)
				{
					tmpEnqueueReadBuffer.event = readBufferInfoPtr->event;
				}
				MPI_Isend(&tmpEnqueueReadBuffer, sizeof(tmpEnqueueReadBuffer), MPI_BYTE, 0,
						 ENQUEUE_READ_BUFFER, parentComm, curRequest);

				MPI_Wait(curRequest, curStatus);
			}
		}

		if (status.MPI_TAG == RELEASE_MEM_OBJ)
		{
			memcpy(&tmpReleaseMemObject, conMsgBuffer[index], sizeof(tmpReleaseMemObject));
			mpiOpenCLReleaseMemObject(&tmpReleaseMemObject);
			MPI_Isend(&tmpReleaseMemObject, sizeof(tmpReleaseMemObject), MPI_BYTE, 0,
					 RELEASE_MEM_OBJ, parentComm, curRequest);
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, RELEASE_MEM_OBJ,
					  parentComm, conMsgRequest + index);
			MPI_Wait(curRequest, curStatus);
		}

		if (status.MPI_TAG == CL_RELEASE_KERNEL_FUNC)
		{
			memcpy(&tmpReleaseKernel, conMsgBuffer[index], sizeof(tmpReleaseKernel));
			mpiOpenCLReleaseKernel(&tmpReleaseKernel);
			MPI_Isend(&tmpReleaseKernel, sizeof(tmpReleaseKernel), MPI_BYTE, 0,
					 CL_RELEASE_KERNEL_FUNC, parentComm, curRequest);
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, CL_RELEASE_KERNEL_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Wait(curRequest, curStatus);
		}

		if (status.MPI_TAG == FINISH_FUNC)
		{
			memcpy(&tmpFinish, conMsgBuffer[index], sizeof(tmpFinish));
			processAllWrites();
			processAllReads();
			mpiOpenCLFinish(&tmpFinish);
			MPI_Isend(&tmpFinish, sizeof(tmpFinish), MPI_BYTE, 0,
					 FINISH_FUNC, parentComm, curRequest);
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, FINISH_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Wait(curRequest, curStatus);
		}

		if (status.MPI_TAG == GET_CONTEXT_INFO_FUNC)
		{
			memcpy(&tmpGetContextInfo, conMsgBuffer[index], sizeof(tmpGetContextInfo));
			param_value_size = tmpGetContextInfo.param_value_size;
			param_value = NULL;
			if (param_value_size > 0 && tmpGetContextInfo.param_value != NULL)
			{
				param_value = malloc(param_value_size);
			}
			mpiOpenCLGetContextInfo(&tmpGetContextInfo, param_value);
			requestNo = 0;
			MPI_Isend(&tmpGetContextInfo, sizeof(tmpGetContextInfo), MPI_BYTE, 0,
					 GET_CONTEXT_INFO_FUNC, parentComm, curRequest+(requestNo++));

			if (param_value_size > 0 && tmpGetContextInfo.param_value != NULL)
			{
				MPI_Isend(param_value, param_value_size, MPI_BYTE, 0,
						 GET_CONTEXT_INFO_FUNC1, parentComm, curRequest+(requestNo++));
			}
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, GET_CONTEXT_INFO_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Waitall(requestNo, curRequest, curStatus);
			if (param_value_size > 0 && tmpGetContextInfo.param_value != NULL)
			{
				free(param_value);
			}
		}

		if (status.MPI_TAG == GET_BUILD_INFO_FUNC)
		{
			memcpy(&tmpGetProgramBuildInfo, conMsgBuffer[index], sizeof(tmpGetProgramBuildInfo));
			param_value_size = tmpGetProgramBuildInfo.param_value_size;
			param_value = NULL;
			if (param_value_size > 0 && tmpGetProgramBuildInfo.param_value != NULL)
			{
				param_value = malloc(param_value_size);
			}
			mpiOpenCLGetProgramBuildInfo(&tmpGetProgramBuildInfo, param_value);
			requestNo = 0;
			MPI_Isend(&tmpGetProgramBuildInfo, sizeof(tmpGetProgramBuildInfo), MPI_BYTE, 0,
					 GET_BUILD_INFO_FUNC, parentComm, curRequest+(requestNo++));

			if (param_value_size > 0 && tmpGetProgramBuildInfo.param_value != NULL)
			{
				MPI_Isend(param_value, param_value_size, MPI_BYTE, 0,
						 GET_BUILD_INFO_FUNC1, parentComm, curRequest+(requestNo++));
			}
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, GET_BUILD_INFO_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Waitall(requestNo, curRequest, curStatus);
			
			if (param_value_size > 0 && tmpGetProgramBuildInfo.param_value != NULL)
			{
				free(param_value);
			}
		}

		if (status.MPI_TAG == GET_PROGRAM_INFO_FUNC)
		{
			memcpy(&tmpGetProgramInfo, conMsgBuffer[index], sizeof(tmpGetProgramInfo));
			param_value_size = tmpGetProgramInfo.param_value_size;
			param_value = NULL;
			if (param_value_size > 0 && tmpGetProgramInfo.param_value != NULL)
			{
				param_value = malloc(param_value_size);
			}
			mpiOpenCLGetProgramInfo(&tmpGetProgramInfo, param_value);
			requestNo = 0;
			MPI_Isend(&tmpGetProgramInfo, sizeof(tmpGetProgramInfo), MPI_BYTE, 0,
					 GET_PROGRAM_INFO_FUNC, parentComm, curRequest+(requestNo++));

			if (param_value_size > 0 && tmpGetProgramInfo.param_value != NULL)
			{
				MPI_Isend(param_value, param_value_size, MPI_BYTE, 0,
						 GET_PROGRAM_INFO_FUNC1, parentComm, curRequest+(requestNo++));
			}
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, GET_PROGRAM_INFO_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Waitall(requestNo, curRequest, curStatus);
			if (param_value_size > 0 && tmpGetProgramInfo.param_value != NULL)
			{
				free(param_value);
			}

		}

		if (status.MPI_TAG == REL_PROGRAM_FUNC)
		{	
			memcpy(&tmpReleaseProgram, conMsgBuffer[index], sizeof(tmpReleaseProgram));
			mpiOpenCLReleaseProgram(&tmpReleaseProgram);
			MPI_Isend(&tmpReleaseProgram, sizeof(tmpReleaseProgram), MPI_BYTE, 0,
					 REL_PROGRAM_FUNC, parentComm, curRequest);
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, REL_PROGRAM_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Wait(curRequest, curStatus);
		}

		if (status.MPI_TAG == REL_COMMAND_QUEUE_FUNC)
		{
			memcpy(&tmpReleaseCommandQueue, conMsgBuffer[index], sizeof(tmpReleaseCommandQueue));
			mpiOpenCLReleaseCommandQueue(&tmpReleaseCommandQueue);
			MPI_Isend(&tmpReleaseCommandQueue, sizeof(tmpReleaseCommandQueue), MPI_BYTE, 0,
					 REL_COMMAND_QUEUE_FUNC, parentComm, curRequest);
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, REL_COMMAND_QUEUE_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Wait(curRequest, curStatus);
		}

		if (status.MPI_TAG == REL_CONTEXT_FUNC)
		{
			memcpy(&tmpReleaseContext, conMsgBuffer[index], sizeof(tmpReleaseContext));
			mpiOpenCLReleaseContext(&tmpReleaseContext);
			MPI_Isend(&tmpReleaseContext, sizeof(tmpReleaseContext), MPI_BYTE, 0,
					REL_CONTEXT_FUNC, parentComm, curRequest);
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, REL_CONTEXT_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Wait(curRequest, curStatus);
		}

		if (status.MPI_TAG == GET_DEVICE_INFO_FUNC)
		{
			memcpy(&tmpGetDeviceInfo, conMsgBuffer[index], sizeof(tmpGetDeviceInfo));
			param_value_size = tmpGetDeviceInfo.param_value_size;
			param_value = NULL;
			if (param_value_size > 0 && tmpGetDeviceInfo.param_value != NULL)
			{
				param_value = malloc(param_value_size);
			}
			mpiOpenCLGetDeviceInfo(&tmpGetDeviceInfo, param_value);
			requestNo = 0;
			MPI_Isend(&tmpGetDeviceInfo, sizeof(tmpGetDeviceInfo), MPI_BYTE, 0,
					 GET_DEVICE_INFO_FUNC, parentComm, curRequest+(requestNo++));
			if (param_value_size > 0 && tmpGetDeviceInfo.param_value != NULL)
			{
				MPI_Isend(param_value, param_value_size, MPI_BYTE, 0,
						 GET_DEVICE_INFO_FUNC1, parentComm, curRequest+(requestNo++));
			}

			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, GET_DEVICE_INFO_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Waitall(requestNo, curRequest, curStatus);
			if (param_value_size > 0 && tmpGetDeviceInfo.param_value != NULL)
			{
				free(param_value);
			}
		}

		if (status.MPI_TAG == GET_PLATFORM_INFO_FUNC)
		{
			memcpy(&tmpGetPlatformInfo, conMsgBuffer[index], sizeof(tmpGetPlatformInfo));
			param_value_size = tmpGetPlatformInfo.param_value_size;
			param_value = NULL;
			if (param_value_size > 0 && tmpGetPlatformInfo.param_value != NULL)
			{
				param_value = malloc(param_value_size);
			}
			mpiOpenCLGetPlatformInfo(&tmpGetPlatformInfo, param_value);
			requestNo = 0;
			MPI_Isend(&tmpGetPlatformInfo, sizeof(tmpGetPlatformInfo), MPI_BYTE, 0,
					 GET_PLATFORM_INFO_FUNC, parentComm, curRequest+(requestNo++));

			if (param_value_size > 0 && tmpGetPlatformInfo.param_value != NULL)
			{
				MPI_Isend(param_value, param_value_size, MPI_BYTE, 0,
						 GET_PLATFORM_INFO_FUNC1, parentComm, curRequest+(requestNo++));
			}
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, GET_PLATFORM_INFO_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Waitall(requestNo, curRequest, curStatus);
			if (param_value_size > 0 && tmpGetPlatformInfo.param_value != NULL)
			{
				free(param_value);
			}
		}

		if (status.MPI_TAG == FLUSH_FUNC)
		{
			memcpy(&tmpFlush, conMsgBuffer[index], sizeof(tmpFlush));
			mpiOpenCLFlush(&tmpFlush);
			MPI_Isend(&tmpFlush, sizeof(tmpFlush), MPI_BYTE, 0,
					 FLUSH_FUNC, parentComm, curRequest);
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, FLUSH_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Wait(curRequest, curStatus);
		}

		if (status.MPI_TAG == WAIT_FOR_EVENT_FUNC)
		{
			requestNo = 0;
			memcpy(&tmpWaitForEvents, conMsgBuffer[index], sizeof(tmpWaitForEvents));
			num_events = tmpWaitForEvents.num_events;
			event_list = (cl_event *)malloc(sizeof(cl_event) * num_events);
			MPI_Irecv(event_list, sizeof(cl_event) * num_events, MPI_BYTE, 0,
					 WAIT_FOR_EVENT_FUNC1, parentComm, curRequest+(requestNo++));
			MPI_Waitall(requestNo, curRequest, curStatus);
			requestNo = 0;
			mpiOpenCLWaitForEvents(&tmpWaitForEvents, event_list);

			for (i = 0; i < num_events; i++)
			{
				bufferIndex = getReadBufferIndexFromEvent(event_list[i]);
				if (bufferIndex >= 0)
				{
					readBufferInfoPtr = getReadBufferInfoPtr(bufferIndex);
					processReadBuffer(bufferIndex, readBufferInfoPtr->numReadBuffers);
					readBufferInfoPtr->numReadBuffers = 0;
				}

				bufferIndex = getWriteBufferIndexFromEvent(event_list[i]);
				if (bufferIndex >= 0)
				{
					writeBufferInfoPtr = getWriteBufferInfoPtr(bufferIndex);
					processWriteBuffer(bufferIndex, writeBufferInfoPtr->numWriteBuffers);
					writeBufferInfoPtr->numWriteBuffers = 0;
				}
			}

			MPI_Isend(&tmpWaitForEvents, sizeof(tmpWaitForEvents), MPI_BYTE, 0,
					 WAIT_FOR_EVENT_FUNC, parentComm, curRequest+(requestNo++));
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, WAIT_FOR_EVENT_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Waitall(requestNo, curRequest, curStatus);
			free(event_list);
		}

		if (status.MPI_TAG == CREATE_SAMPLER_FUNC)
		{
			memcpy(&tmpCreateSampler, conMsgBuffer[index], sizeof(tmpCreateSampler));
			mpiOpenCLCreateSampler(&tmpCreateSampler);
			MPI_Isend(&tmpCreateSampler, sizeof(tmpCreateSampler), MPI_BYTE, 0,
					 CREATE_SAMPLER_FUNC, parentComm, curRequest);
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, CREATE_SAMPLER_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Wait(curRequest, curStatus);
		}

		if (status.MPI_TAG == GET_CMD_QUEUE_INFO_FUNC)
		{
			requestNo = 0;
			memcpy(&tmpGetCommandQueueInfo, conMsgBuffer[index], sizeof(tmpGetCommandQueueInfo));
			param_value_size = tmpGetCommandQueueInfo.param_value_size;
			param_value = NULL;
			if (param_value_size > 0 && tmpGetCommandQueueInfo.param_value != NULL)
			{
				param_value = malloc(param_value_size);
			}
			mpiOpenCLGetCommandQueueInfo(&tmpGetCommandQueueInfo, param_value);
			MPI_Isend(&tmpGetCommandQueueInfo, sizeof(tmpGetCommandQueueInfo), MPI_BYTE, 0,
					 GET_CMD_QUEUE_INFO_FUNC, parentComm, curRequest+(requestNo++));

			if (param_value_size > 0 && tmpGetCommandQueueInfo.param_value != NULL)
			{
				MPI_Isend(param_value, param_value_size, MPI_BYTE, 0,
						 GET_CMD_QUEUE_INFO_FUNC1, parentComm, curRequest+(requestNo++));
			}

			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, GET_CMD_QUEUE_INFO_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Waitall(requestNo, curRequest, curStatus);
			if (param_value_size > 0 && tmpGetCommandQueueInfo.param_value != NULL)
			{
				free(param_value);
			}
		}

		if (status.MPI_TAG == ENQUEUE_MAP_BUFF_FUNC)
		{
			requestNo = 0;

			memcpy(&tmpEnqueueMapBuffer, conMsgBuffer[index], sizeof(tmpEnqueueMapBuffer));
			num_events_in_wait_list = tmpEnqueueMapBuffer.num_events_in_wait_list;
			event_wait_list = NULL;
			if (num_events_in_wait_list > 0)
			{
				event_wait_list = (cl_event *)malloc(sizeof(cl_event) * num_events_in_wait_list);
				MPI_Irecv(event_wait_list, sizeof(cl_event) * num_events_in_wait_list, MPI_BYTE, 0,
						 ENQUEUE_MAP_BUFF_FUNC1, parentComm, curRequest+(requestNo++));
			}
			mpiOpenCLEnqueueMapBuffer(&tmpEnqueueMapBuffer, event_wait_list);
			MPI_Isend(&tmpEnqueueMapBuffer, sizeof(tmpEnqueueMapBuffer), MPI_BYTE, 0,
					 ENQUEUE_MAP_BUFF_FUNC, parentComm, curRequest+(requestNo++));
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, ENQUEUE_MAP_BUFF_FUNC,
					  parentComm, conMsgRequest + index);
			if (num_events_in_wait_list > 0)
			{
				free(event_wait_list);
			}
			MPI_Waitall(requestNo, curRequest, curStatus);
		}

		if (status.MPI_TAG == RELEASE_EVENT_FUNC)
		{
			memcpy(&tmpReleaseEvent, conMsgBuffer[index], sizeof(tmpReleaseEvent));
			mpiOpenCLReleaseEvent(&tmpReleaseEvent);
			MPI_Isend(&tmpReleaseEvent, sizeof(tmpReleaseEvent), MPI_BYTE, 0,
					 RELEASE_EVENT_FUNC, parentComm, curRequest);
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, RELEASE_EVENT_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Wait(curRequest, curStatus);
		}

		if (status.MPI_TAG == GET_EVENT_PROF_INFO_FUNC)
		{
			requestNo = 0;
			memcpy(&tmpGetEventProfilingInfo, conMsgBuffer[index], sizeof(tmpGetEventProfilingInfo));
			param_value_size = tmpGetEventProfilingInfo.param_value_size;
			param_value = NULL;
			if (param_value_size > 0 && tmpGetEventProfilingInfo.param_value != NULL)
			{
				param_value = malloc(param_value_size);
			}
			mpiOpenCLGetEventProfilingInfo(&tmpGetEventProfilingInfo, param_value);
			MPI_Isend(&tmpGetEventProfilingInfo, sizeof(tmpGetEventProfilingInfo), MPI_BYTE, 0,
					 GET_EVENT_PROF_INFO_FUNC, parentComm, curRequest+(requestNo++));

			if (param_value_size > 0 && tmpGetEventProfilingInfo.param_value != NULL)
			{
				MPI_Isend(param_value, param_value_size, MPI_BYTE, 0,
						 GET_EVENT_PROF_INFO_FUNC1, parentComm, curRequest+(requestNo++));
			}
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, GET_EVENT_PROF_INFO_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Waitall(requestNo, curRequest, curStatus);
			if (param_value_size > 0 && tmpGetEventProfilingInfo.param_value != NULL)
			{
				free(param_value);
			}
		}

		if (status.MPI_TAG == RELEASE_SAMPLER_FUNC)
		{
			memcpy(&tmpReleaseSampler, conMsgBuffer[index], sizeof(tmpReleaseSampler));
			mpiOpenCLReleaseSampler(&tmpReleaseSampler);
			MPI_Isend(&tmpReleaseSampler, sizeof(tmpReleaseSampler), MPI_BYTE, 0,
					 RELEASE_SAMPLER_FUNC, parentComm, curRequest);
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, RELEASE_SAMPLER_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Wait(curRequest, curStatus);
		}

		if (status.MPI_TAG == GET_KERNEL_WGP_INFO_FUNC)
		{
			requestNo = 0;
			memcpy(&tmpGetKernelWorkGroupInfo, conMsgBuffer[index], sizeof(tmpGetKernelWorkGroupInfo));
			param_value_size = tmpGetKernelWorkGroupInfo.param_value_size;
			param_value = NULL;
			if (param_value_size > 0 && tmpGetKernelWorkGroupInfo.param_value != NULL)
			{
				param_value = malloc(param_value_size);
			}
			mpiOpenCLGetKernelWorkGroupInfo(&tmpGetKernelWorkGroupInfo, param_value);
			MPI_Isend(&tmpGetKernelWorkGroupInfo, sizeof(tmpGetKernelWorkGroupInfo), MPI_BYTE, 0,
					 GET_KERNEL_WGP_INFO_FUNC, parentComm, curRequest+(requestNo++));

			if (param_value_size > 0 && tmpGetKernelWorkGroupInfo.param_value != NULL)
			{
				MPI_Isend(param_value, param_value_size, MPI_BYTE, 0,
						 GET_KERNEL_WGP_INFO_FUNC1, parentComm, curRequest+(requestNo++));
				free(param_value);
			}
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, GET_KERNEL_WGP_INFO_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Waitall(requestNo, curRequest, curStatus);
		}

		if (status.MPI_TAG == CREATE_IMAGE_2D_FUNC)
		{
			requestNo = 0;
			memcpy(&tmpCreateImage2D, conMsgBuffer[index], sizeof(tmpCreateImage2D));
			host_buff_size = tmpCreateImage2D.host_buff_size;
			host_ptr = NULL;
			if (host_buff_size > 0)
			{
				host_ptr = malloc(host_buff_size);
				MPI_Irecv(host_ptr, host_buff_size, MPI_BYTE, 0,
						 CREATE_IMAGE_2D_FUNC1, parentComm, curRequest+(requestNo++));
			}
			mpiOpenCLCreateImage2D(&tmpCreateImage2D, host_ptr);
			MPI_Isend(&tmpCreateImage2D, sizeof(tmpCreateImage2D), MPI_BYTE, 0,
					 CREATE_IMAGE_2D_FUNC, parentComm, curRequest+(requestNo++));
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, CREATE_IMAGE_2D_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Waitall(requestNo, curRequest, curStatus);
			if (host_buff_size > 0)
			{
				free(host_ptr);
			}
		}

		if (status.MPI_TAG == ENQ_COPY_BUFF_FUNC)
		{
			requestNo = 0;
			memcpy(&tmpEnqueueCopyBuffer, conMsgBuffer[index], sizeof(tmpEnqueueCopyBuffer));
			num_events_in_wait_list = tmpEnqueueCopyBuffer.num_events_in_wait_list;
			event_wait_list = NULL;
			if (num_events_in_wait_list > 0)
			{
				event_wait_list = (cl_event *)malloc(sizeof(cl_event) * num_events_in_wait_list);
				MPI_Irecv(event_wait_list, sizeof(cl_event) * num_events_in_wait_list, MPI_BYTE, 0,
						 ENQ_COPY_BUFF_FUNC1, parentComm, curRequest+(requestNo++));
			}
			mpiOpenCLEnqueueCopyBuffer(&tmpEnqueueCopyBuffer, event_wait_list);
			MPI_Isend(&tmpEnqueueCopyBuffer, sizeof(tmpEnqueueCopyBuffer), MPI_BYTE, 0,
					 ENQ_COPY_BUFF_FUNC, parentComm, curRequest+(requestNo++));
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, ENQ_COPY_BUFF_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Waitall(requestNo, curRequest, curStatus);
			if (num_events_in_wait_list > 0)
			{
				free(event_wait_list);
			}
		}

		if (status.MPI_TAG == RETAIN_EVENT_FUNC)
		{
			memcpy(&tmpRetainEvent, conMsgBuffer[index], sizeof(tmpRetainEvent));
			mpiOpenCLRetainEvent(&tmpRetainEvent);
			MPI_Isend(&tmpRetainEvent, sizeof(tmpRetainEvent), MPI_BYTE, 0,
					 RETAIN_EVENT_FUNC, parentComm, curRequest);
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, RETAIN_EVENT_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Wait(curRequest, curStatus);
		}

		if (status.MPI_TAG == RETAIN_MEMOBJ_FUNC)
		{
			memcpy(&tmpRetainMemObject, conMsgBuffer[index], sizeof(tmpRetainMemObject));
			mpiOpenCLRetainMemObject(&tmpRetainMemObject);
			MPI_Isend(&tmpRetainMemObject, sizeof(tmpRetainMemObject), MPI_BYTE, 0,
					 RETAIN_MEMOBJ_FUNC, parentComm, curRequest);
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, RETAIN_MEMOBJ_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Wait(curRequest, curStatus);
		}

		if (status.MPI_TAG == RETAIN_KERNEL_FUNC)
		{
			requestNo = 0;
			MPI_Irecv(&tmpRetainKernel, sizeof(tmpRetainKernel), MPI_BYTE, 0,
					 RETAIN_KERNEL_FUNC, parentComm, curRequest+(requestNo++));
			mpiOpenCLRetainKernel(&tmpRetainKernel);
			MPI_Isend(&tmpRetainKernel, sizeof(tmpRetainKernel), MPI_BYTE, 0,
					 RETAIN_KERNEL_FUNC, parentComm, curRequest+(requestNo++));
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, RETAIN_KERNEL_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Waitall(requestNo, curRequest, curStatus);
		}

		if (status.MPI_TAG == RETAIN_CMDQUE_FUNC)
		{
			memcpy(&tmpRetainCommandQueue, conMsgBuffer[index], sizeof(tmpRetainCommandQueue));
			mpiOpenCLRetainCommandQueue(&tmpRetainCommandQueue);
			MPI_Isend(&tmpRetainCommandQueue, sizeof(tmpRetainCommandQueue), MPI_BYTE, 0,
					 RETAIN_CMDQUE_FUNC, parentComm, curRequest);
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, RETAIN_CMDQUE_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Wait(curRequest, curStatus);
		}

		if (status.MPI_TAG == ENQ_UNMAP_MEMOBJ_FUNC)
		{
			requestNo = 0;
			memcpy(&tmpEnqueueUnmapMemObject, conMsgBuffer[index], sizeof(tmpEnqueueUnmapMemObject));
			num_events_in_wait_list = tmpEnqueueUnmapMemObject.num_events_in_wait_list;
			event_wait_list = NULL;
			if (num_events_in_wait_list > 0)
			{
				event_wait_list = (cl_event *)malloc(sizeof(cl_event) * num_events_in_wait_list);
				MPI_Irecv(event_wait_list, sizeof(cl_event) * num_events_in_wait_list, MPI_BYTE, 0,
						 ENQ_UNMAP_MEMOBJ_FUNC1, parentComm, curRequest+(requestNo++));
			}
			mpiOpenCLEnqueueUnmapMemObject(&tmpEnqueueUnmapMemObject, event_wait_list);
			MPI_Isend(&tmpEnqueueUnmapMemObject, sizeof(tmpEnqueueUnmapMemObject), MPI_BYTE, 0,
					 ENQ_UNMAP_MEMOBJ_FUNC, parentComm, curRequest+(requestNo++));
			/* issue it for later call of this function */
			MPI_Irecv(conMsgBuffer[index], MAX_CMSG_SIZE, MPI_BYTE, 0, ENQ_UNMAP_MEMOBJ_FUNC,
					  parentComm, conMsgRequest + index);
			MPI_Waitall(requestNo, curRequest, curStatus);
			if (num_events_in_wait_list > 0)
			{
				free(event_wait_list);
			}
		}
		
		if (status.MPI_TAG == PROGRAM_END)
		{
			break;
		}
	}


	for (i = 0; i < CMSG_NUM; i++)
	{
		free(conMsgBuffer[i]);
	}
	free(conMsgRequest);
	free(curStatus);
	free(curRequest);

	/* terminate the helper thread */
    pthread_barrier_wait(&barrier);
    pthread_barrier_wait(&barrier);
	pthread_join(th, NULL);
	pthread_barrier_destroy(&barrier);

	/* release the write and read buffer pool */
	finalizeWriteBuffer();
	finalizeReadBuffer();

	MPI_Comm_free(&parentComm);
	MPI_Finalize();

	return 0;
}

