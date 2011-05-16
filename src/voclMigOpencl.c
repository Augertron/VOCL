#include <stdio.h>
#include <string.h>
#include "vocl_opencl.h"
#include "vocl_structures.h"
#include "voclKernelArgProc.h"
#include <sys/time.h>

/* for context */
extern cl_context voclVOCLContext2CLContextComm(vocl_context context, int *proxyRank,
                      int *proxyIndex, MPI_Comm *proxyComm, MPI_Comm *proxyCommData);

/* for command queue processing */
extern cl_command_queue voclVOCLCommandQueue2CLCommandQueueComm(vocl_command_queue commandQueue, int *proxyRank,
                            int *proxyIndex, MPI_Comm *proxyComm, MPI_Comm *proxyCommData);

/* for program processing */
extern cl_program voclVOCLProgram2CLProgramComm(vocl_program program, int *proxyRank,
                      int *proxyIndex, MPI_Comm *proxyComm, MPI_Comm *proxyCommData);

/* for program processing */
extern cl_mem voclVOCLMemory2CLMemoryComm(vocl_mem memory, int *proxyRank,
                  int *proxyIndex, MPI_Comm *proxyComm, MPI_Comm *proxyCommData);

/* for program processing */
extern cl_kernel voclVOCLKernel2CLKernelComm(vocl_kernel kernel, int *proxyRank,
                     int *proxyIndex, MPI_Comm *proxyComm, MPI_Comm *proxyCommData);

/* vocl sampler processing API functions */
extern cl_sampler voclVOCLSampler2CLSamplerComm(vocl_sampler sampler, int *proxyRank,
                      int *proxyIndex, MPI_Comm *proxyComm, MPI_Comm *proxyCommData);

/*--------------------VOCL API functions, countparts of OpenCL API functions ----------------*/
cl_context
voclMigCreateContext(const cl_context_properties * properties,
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

    /* initialize structure */
    /* tmpCreateContext.properties = *properties; */
    tmpCreateContext.num_devices = num_devices;
    tmpCreateContext.devices = (cl_device_id *) devices;
    /* convert vocl context to opencl context */
	clDevices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
    for (i = 0; i < num_devices; i++)
	{
		clDevices[i] = voclVOCLDeviceID2CLDeviceIDComm((vocl_device_id)devices[i], &proxyRank, &proxyIndex, &proxyComm, &proxyCommData);
	}

	/* local node, call native opencl function directly */
	if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE)
	{
		dlCLCreateContext(properties, num_devices, clDevices, pfn_notify,
                   user_data, errcode_ret, &tmpCreateContext.hContext);
	}
	else
	{
    	tmpCreateContext.user_data = user_data;
		/* send parameters to remote node */
		res = MPI_Isend(&tmpCreateContext, sizeof(tmpCreateContext), MPI_BYTE, proxyRank,
						CREATE_CONTEXT_FUNC, proxyComm, request + (requestNo++));

		if (devices != NULL) {
			MPI_Isend((void *) clDevices, sizeof(cl_device_id) * num_devices, MPI_BYTE, proxyRank,
					  CREATE_CONTEXT_FUNC1, proxyCommData, request + (requestNo++));
		}

		MPI_Irecv(&tmpCreateContext, sizeof(tmpCreateContext), MPI_BYTE, proxyRank,
				  CREATE_CONTEXT_FUNC, proxyComm, request + (requestNo++));
		MPI_Waitall(requestNo, request, status);
		*errcode_ret = tmpCreateContext.errcode_ret;

		/* increase OpenCL object count */
		//increaseObjCount(proxyIndex);
	}
	
	/*convert opencl context to vocl context */
	context = tmpCreateContext.hContext;
	//context = voclCLContext2VOCLContext(tmpCreateContext.hContext, proxyRank, proxyIndex, proxyComm, proxyCommData);
	free(clDevices);

    return context;
}

/* Command Queue APIs */
cl_command_queue
voclMigCreateCommandQueue(cl_context context,
                     cl_device_id device,
                     cl_command_queue_properties properties, cl_int * errcode_ret)
{
    struct strCreateCommandQueue tmpCreateCommandQueue;
    MPI_Status status[2];
    MPI_Request request[2];
    int requestNo = 0;
	vocl_command_queue command_queue;
	int proxyRankDevice, proxyRankContext, proxyIndex;
	MPI_Comm proxyComm, proxyCommData;

	printf("migCreateCmd\n");
    tmpCreateCommandQueue.context = voclVOCLContext2CLContextComm((vocl_context)context, &proxyRankContext, &proxyIndex, &proxyComm, &proxyCommData);
    tmpCreateCommandQueue.device = voclVOCLDeviceID2CLDeviceIDComm((vocl_device_id)device, &proxyRankDevice, &proxyIndex, &proxyComm, &proxyCommData);
	if (proxyRankContext != proxyRankDevice)
	{
		printf("deice and context are on different GPU nodes!\n");
	}

	/* local node, call native opencl function directly */
	if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE)
	{
		dlCLCreateCommandQueue(tmpCreateCommandQueue.context, 
	    	tmpCreateCommandQueue.device, properties, errcode_ret, &tmpCreateCommandQueue.clCommand);
	}
	else
	{
		tmpCreateCommandQueue.properties = properties;

		/* send parameters to remote node */
		MPI_Isend(&tmpCreateCommandQueue, sizeof(tmpCreateCommandQueue), MPI_BYTE, proxyRankContext,
				  CREATE_COMMAND_QUEUE_FUNC, proxyComm, request + (requestNo++));
		MPI_Irecv(&tmpCreateCommandQueue, sizeof(tmpCreateCommandQueue), MPI_BYTE, proxyRankContext,
				  CREATE_COMMAND_QUEUE_FUNC, proxyComm, request + (requestNo++));
		MPI_Waitall(requestNo, request, status);
		if (errcode_ret != NULL) {
			*errcode_ret = tmpCreateCommandQueue.errcode_ret;
		}

		/* increase OpenCL object count */
		//increaseObjCount(proxyIndex);
	}

    return tmpCreateCommandQueue.clCommand;
}

cl_program
voclMigCreateProgramWithSource(cl_context context,
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

    /* initialize structure */
    tmpCreateProgramWithSource.context = voclVOCLContext2CLContextComm((vocl_context)context, 
        &proxyRank, &proxyIndex, &proxyComm, &proxyCommData);

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

	if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE)
	{
		dlCLCreateProgramWithSource(tmpCreateProgramWithSource.context, 
			count, strings, lengths, errcode_ret, &tmpCreateProgramWithSource.clProgram);
	}
	else
	{
		/* send parameters to remote node */
		MPI_Isend(&tmpCreateProgramWithSource,
				  sizeof(tmpCreateProgramWithSource),
				  MPI_BYTE, proxyRank, CREATE_PROGRMA_WITH_SOURCE, proxyComm, request + (requestNo++));
		MPI_Isend(lengthsArray, sizeof(size_t) * count, MPI_BYTE, proxyRank,
				  CREATE_PROGRMA_WITH_SOURCE1, proxyCommData, request + (requestNo++));
		MPI_Isend((void *) allStrings, totalLength * sizeof(char), MPI_BYTE, proxyRank,
				  CREATE_PROGRMA_WITH_SOURCE2, proxyCommData, request + (requestNo++));
		MPI_Irecv(&tmpCreateProgramWithSource,
				  sizeof(tmpCreateProgramWithSource),
				  MPI_BYTE, proxyRank, CREATE_PROGRMA_WITH_SOURCE, proxyComm, request + (requestNo++));
		MPI_Waitall(requestNo, request, status);
		if (errcode_ret != NULL) {
			*errcode_ret = tmpCreateProgramWithSource.errcode_ret;
		}

		/* increase OpenCL object count */
		//increaseObjCount(proxyIndex);
	}

	free(allStrings);
	free(lengthsArray);

    return tmpCreateProgramWithSource.clProgram;
}

cl_kernel voclMigCreateKernel(cl_program program, const char *kernel_name, cl_int * errcode_ret)
{
    MPI_Status status[3];
    MPI_Request request[3];
	int proxyRank, proxyIndex;
	MPI_Comm proxyComm, proxyCommData;
	vocl_kernel kernel;
    int requestNo = 0;
    struct strCreateKernel tmpCreateKernel;
    int kernelNameSize = strlen(kernel_name);

    tmpCreateKernel.program = voclVOCLProgram2CLProgramComm((vocl_program)program, 
        &proxyRank, &proxyIndex, &proxyComm, &proxyCommData);

	/* local node, call native opencl function directly */
	if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE)
	{
		dlCLCreateKernel(tmpCreateKernel.program, kernel_name, errcode_ret, &tmpCreateKernel.kernel);
	}
	else
	{
		tmpCreateKernel.kernelNameSize = kernelNameSize;

		/* send input parameters to remote node */
		MPI_Isend(&tmpCreateKernel, sizeof(tmpCreateKernel), MPI_BYTE, proxyRank,
				  CREATE_KERNEL, proxyComm, request + (requestNo++));
		MPI_Isend((void *) kernel_name, kernelNameSize, MPI_CHAR, proxyRank, CREATE_KERNEL1, proxyCommData,
				  request + (requestNo++));
		MPI_Irecv(&tmpCreateKernel, sizeof(tmpCreateKernel), MPI_BYTE, proxyRank,
				  CREATE_KERNEL, proxyComm, request + (requestNo++));
		MPI_Waitall(requestNo, request, status);
		if (errcode_ret != NULL)
		{
			*errcode_ret = tmpCreateKernel.errcode_ret;
		}

		/* increase OpenCL object count */
		//increaseObjCount(proxyIndex);
	}

    return tmpCreateKernel.kernel;
}

/* Memory Object APIs */
cl_mem
voclMigCreateBuffer(cl_context context,
               cl_mem_flags flags, size_t size, void *host_ptr, cl_int * errcode_ret)
{
    struct strCreateBuffer tmpCreateBuffer;
    MPI_Status status[3];
    MPI_Request request[3];
	vocl_mem memory;
	int proxyRank, proxyIndex;
	MPI_Comm proxyComm, proxyCommData;
    int requestNo = 0;

    /* initialize structure */
    tmpCreateBuffer.context = voclVOCLContext2CLContextComm((vocl_context)context, 
        &proxyRank, &proxyIndex, &proxyComm, &proxyCommData);

	if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE)
	{
		dlCLCreateBuffer(tmpCreateBuffer.context, flags, size, host_ptr, errcode_ret, &tmpCreateBuffer.deviceMem);
	}
	else
	{
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

		/* increase OpenCL object count */
		//increaseObjCount(proxyIndex);
	}
	
    return tmpCreateBuffer.deviceMem;
}

cl_sampler
voclMigCreateSampler(cl_context context,
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

    tmpCreateSampler.context = voclVOCLContext2CLContextComm((vocl_context)context, 
                                   &proxyRank, &proxyIndex, &proxyComm, &proxyCommData);
	if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE)
	{
		dlCLCreateSampler(tmpCreateSampler.context, 
            normalized_coords, addressing_mode, filter_mode, errcode_ret, &tmpCreateSampler.sampler);
	}
	else
	{
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

		/* increase OpenCL object count */
		//increaseObjCount(proxyIndex);
	}

    return tmpCreateSampler.sampler;
}

/* Enqueued Commands for GPU memory read */
cl_int
clMigEnqueueReadBuffer(cl_command_queue command_queue /* actual opencl command */,
                    cl_mem buffer,
                    cl_bool blocking_read,
                    size_t offset,
                    size_t cb,
                    void *ptr,
                    cl_uint num_events_in_wait_list,
                    const cl_event * event_wait_list, cl_event * event)
{
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
    tmpEnqueueReadBuffer.command_queue = command_queue;
    tmpEnqueueReadBuffer.buffer = voclVOCLMemory2CLMemoryComm((vocl_mem)buffer, &proxyRank, &proxyIndex,
                                      &proxyComm, &proxyCommData);
	if (voclIsOnLocalNode(proxyIndex) == VOCL_TRUE)
	{
		errCode = dlCLEnqueueReadBuffer(tmpEnqueueReadBuffer.command_queue, tmpEnqueueReadBuffer.buffer,
					  blocking_read, offset, cb, ptr, num_events_in_wait_list, eventList, event);
		if (event != NULL)
		{
			*event = (cl_event)voclCLEvent2VOCLEvent((cl_event)(*event), proxyRank, proxyIndex, proxyComm, proxyCommData);
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
                  MPI_BYTE, proxyRank, ENQUEUE_READ_BUFFER1, proxyCommData, request + (requestNo++));
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
        MPI_Irecv((void *) ((char *) ptr + VOCL_READ_BUFFER_SIZE * i), bufferSize, MPI_BYTE, proxyRank,
                  VOCL_READ_TAG + bufferIndex, proxyCommData, getReadRequestPtr(proxyIndex, bufferIndex));
    }

    requestNo = 0;
    if (blocking_read == CL_TRUE || event != NULL) {
        MPI_Irecv(&tmpEnqueueReadBuffer, sizeof(struct strEnqueueReadBuffer), MPI_BYTE, proxyRank,
                  ENQUEUE_READ_BUFFER, proxyComm, request + (requestNo++));
    }

    if (blocking_read == CL_TRUE) {
        processAllReads(proxyIndex);
        MPI_Waitall(requestNo, request, status);
        if (event != NULL) {
            voclEvent = voclCLEvent2VOCLEvent(tmpEnqueueReadBuffer.event, 
                            proxyRank, proxyIndex, proxyComm, proxyCommData);
            *event = (cl_event) voclEvent;
        }
        return tmpEnqueueReadBuffer.res;
    }
    else {
        if (event != NULL) {
            MPI_Waitall(requestNo, request, status);
			voclEvent = voclCLEvent2VOCLEvent(tmpEnqueueReadBuffer.event, 
                            proxyRank, proxyIndex, proxyComm, proxyCommData);
            *event = (cl_event) voclEvent;
            return tmpEnqueueReadBuffer.res;
        }
        else {
            return CL_SUCCESS;
        }
    }
}


