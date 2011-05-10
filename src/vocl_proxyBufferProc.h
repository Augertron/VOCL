#ifndef __VOCL_PROXY_BUFFER_PROC_H__
#define __VOCL_PROXY_BUFFER_PROC_H__

/* used to record the GPU memory reads that are completed */
/* before the current GPU memory write. Then the MPI_Isend */
/* can be called to send data back to the local node. */
struct strReadMPISendProcInfo {
    int toBeProcessedNum;
    int readBufferIndex[VOCL_PROXY_READ_BUFFER_NUM];
};

struct strWriteBufferInfo {
    /*write buffer state, defined in voclProxy.h */
    int isInUse;
	int appRank;
    MPI_Request request;
    cl_command_queue commandQueue;
    size_t size;
    char *dataPtr;
	//int buffIndex;
    size_t offset;
    cl_mem mem;
    cl_int blocking_write;
    cl_int numEvents;
    cl_event *eventWaitList;
    cl_event event;
    int numWriteBuffers;
    struct strReadMPISendProcInfo sendProcInfo;
};

struct voclWriteBufferInfo {
	int writeDataRequestNum;
	int curWriteBufferIndex;
	int allWritesAreEnqueuedFlag;
	int allReadBuffersAreCovered;
	struct strWriteBufferInfo writeBufferInfo[VOCL_PROXY_WRITE_BUFFER_NUM];
};

struct strReadBufferInfo {
    /* read buffer state, defined in voclProxy.h */
    int isInUse;
    MPI_Request request;
    MPI_Comm comm;
    int tag;
	int dest;
	int appRank; /* indicate which app proc to send */
    size_t size;
    char *dataPtr;
	//int buffIndex;
    cl_event event;

    int numReadBuffers;
};

struct voclReadBufferInfo {
	int readDataRequestNum;
	int curReadBufferIndex;
	struct strReadBufferInfo readBufferInfo[VOCL_PROXY_READ_BUFFER_NUM];
};

#endif
