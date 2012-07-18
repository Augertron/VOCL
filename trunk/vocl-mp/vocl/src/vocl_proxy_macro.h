#ifndef __VOCL_PROXY_MACRO_H__
#define __VOCL_PROXY_MACRO_H__


#define MAX_CMSG_SIZE 112        /* max control message size */
typedef char CON_MSG_BUFFER[MAX_CMSG_SIZE];

#define GET_PROXY_COMM_INFO         9
#define GET_PLATFORM_ID_FUNC        10
#define GET_DEVICE_ID_FUNC          11
#define CREATE_CONTEXT_FUNC         12
#define LOAD_SOURCE_FUNC            13
#define CREATE_PROGRMA_WITH_SOURCE  14
#define CREATE_COMMAND_QUEUE_FUNC   15
#define BUILD_PROGRAM               16
#define CREATE_KERNEL               17
#define CREATE_BUFFER_FUNC          18
#define ENQUEUE_WRITE_BUFFER        19
#define SET_KERNEL_ARG              20
#define ENQUEUE_ND_RANGE_KERNEL     21
#define ENQUEUE_READ_BUFFER         22
#define RELEASE_MEM_OBJ             23
#define FINISH_FUNC                 24
#define GET_CONTEXT_INFO_FUNC	    25
#define CL_RELEASE_KERNEL_FUNC      26
#define GET_BUILD_INFO_FUNC         27
#define GET_PROGRAM_INFO_FUNC       28
#define REL_PROGRAM_FUNC            29
#define REL_COMMAND_QUEUE_FUNC      30
#define REL_CONTEXT_FUNC            31
#define GET_DEVICE_INFO_FUNC        32
#define GET_PLATFORM_INFO_FUNC      33
#define FLUSH_FUNC				    34
#define WAIT_FOR_EVENT_FUNC         35
#define GET_CMD_QUEUE_INFO_FUNC     36
#define CREATE_SAMPLER_FUNC         37
#define ENQUEUE_MAP_BUFF_FUNC       38
#define RELEASE_EVENT_FUNC          39
#define RELEASE_SAMPLER_FUNC        40
#define GET_EVENT_PROF_INFO_FUNC    41
#define GET_KERNEL_WGP_INFO_FUNC    42
#define CREATE_IMAGE_2D_FUNC        43
#define ENQ_COPY_BUFF_FUNC          44
#define RETAIN_EVENT_FUNC           45
#define RETAIN_MEMOBJ_FUNC          46
#define RETAIN_KERNEL_FUNC          47
#define RETAIN_CMDQUE_FUNC          48
#define ENQ_UNMAP_MEMOBJ_FUNC       49
#define MIG_MEM_WRITE_REQUEST       50
#define MIG_MEM_READ_REQUEST        51
#define MIG_MEM_WRITE_CMPLD         52
#define MIG_MEM_READ_CMPLD          53
#define MIG_GET_PROXY_RANK          54
#define MIG_SAME_REMOTE_NODE        55
#define MIG_SAME_REMOTE_NODE_CMPLD  56
#define VOCL_MIGRATION              57
#define VOCL_UPDATE_VGPU            58
#define VOCL_MIG_LAST_MSG			59
#define VOCL_REBALANCE				60
#define VOCL_MIG_CMD_OPERATIONS     61
#define FORCED_MIGRATION            62
#define LB_GET_KERNEL_NUM           63
#define PROGRAM_END                 64
#define VOCL_CONMSG_FLOW_CONTROL	65

#define CMSG_NUM                    50 /* number of buffers for control messages */
#define DMSG_NUM      				100     /* number of buffers for data messages */
#define TOTAL_MSG_NUM				(CMSG_NUM + DMSG_NUM)


#define SERVICE_NAME_LEN            64
#define DEFAULT_PROXY_NUM           64
#define DEFAULT_APP_NUM             20
#define VOCL_PROXY_APP_MAX_CMD_NUM  512

#define GET_PLATFORM_ID_FUNC1	    10000
#define GET_DEVICE_ID_FUNC1         10001
#define CREATE_CONTEXT_FUNC1        10002
#define LOAD_SOURCE_FUNC1           10003
#define CREATE_PROGRMA_WITH_SOURCE1 10004
#define CREATE_PROGRMA_WITH_SOURCE2 10005
#define BUILD_PROGRAM1              10006
#define CREATE_KERNEL1              10007
#define CREATE_BUFFER_FUNC1         10008
#define ENQUEUE_WRITE_BUFFER1       10009
#define ENQUEUE_WRITE_BUFFER2       10010
#define SET_KERNEL_ARG1             10011
#define ENQUEUE_ND_RANGE_KERNEL1    10012
#define ENQUEUE_ND_RANGE_KERNEL2    10013
#define ENQUEUE_ND_RANGE_KERNEL3    10014
#define ENQUEUE_ND_RANGE_KERNEL4    10015
#define ENQUEUE_READ_BUFFER1        10016
#define GET_CONTEXT_INFO_FUNC1	    10017
#define GET_BUILD_INFO_FUNC1        10018
#define GET_PROGRAM_INFO_FUNC1      10019
#define GET_DEVICE_INFO_FUNC1       10020
#define GET_PLATFORM_INFO_FUNC1     10021
#define WAIT_FOR_EVENT_FUNC1        10022
#define GET_CMD_QUEUE_INFO_FUNC1 	10023
#define ENQUEUE_MAP_BUFF_FUNC1      10024
#define GET_EVENT_PROF_INFO_FUNC1   10025
#define GET_KERNEL_WGP_INFO_FUNC1   10026
#define CREATE_IMAGE_2D_FUNC1       10027
#define ENQ_COPY_BUFF_FUNC1         10028
#define ENQ_UNMAP_MEMOBJ_FUNC1      10029


/* default number of applications supported by each proxy */
#define VOCL_PROXY_APP_NUM 4
#define MAX_DEVICE_NUM_PER_NODE 8

/* flags for helper thread */
#define GPU_MEM_READ        0
#define GPU_MEM_WRITE       1
#define GPU_WRITE_SINGLE    2
#define GPU_ENQ_WRITE       3
#define GPU_MEM_NULL        4
#define SEND_LOCAL_PREVIOUS 5

#define VOCL_PROXY_WRITE_BUFFER_NUM 16
//#define VOCL_PROXY_WRITE_BUFFER_SIZE 67108864   /*64MB 1024 X 1024 X 64 */
#define VOCL_PROXY_WRITE_BUFFER_SIZE 33554432   /*64MB 1024 X 1024 X 64 */

#define VOCL_PROXY_READ_BUFFER_NUM 16
//#define VOCL_PROXY_READ_BUFFER_SIZE 67108864    /*64MB 1024 X 1024 X 64 */
#define VOCL_PROXY_READ_BUFFER_SIZE 33554432    /*64MB 1024 X 1024 X 64 */

#define VOCL_PROXY_READ_TAG  4000
#define VOCL_PROXY_WRITE_TAG 5000
#define VOCL_PROXY_MIG_TAG   6000

/* states of write buffer pool */
#define WRITE_AVAILABLE 	0
#define WRITE_RECV_DATA    	1
#define WRITE_RECV_COMPLED	2
#define WRITE_GPU_MEM	   	3

/* states of read buffer pool */
#define READ_AVAILABLE 		0
#define READ_GPU_MEM    	1
#define READ_GPU_MEM_SUB	2
#define READ_GPU_MEM_COMP 	3
#define READ_SEND_DATA  	4

#define VOCL_MIG_BUF_POOL 4
/* macros for migration */
#define VOCL_MIG_BUF_NUM 8
//#define VOCL_MIG_BUF_SIZE 67108864
#define VOCL_MIG_BUF_SIZE 33554432

/* migration buffer states */
#define MIG_WRT_AVAILABLE 0
#define MIG_WRT_MPIRECV  1
#define MIG_WRT_WRTGPU   2

#define MIG_READ_AVAILABLE 3
#define MIG_READ_RDGPU     4
#define MIG_READ_MPISEND   5

#define MIG_RW_SAME_NODE_AVLB   6
#define MIG_RW_SAME_NODE_RDMEM  7
#define MIG_RW_SAME_NODE_WTMEM  8

#define VOCL_MIG_UPDATE -100
#define VOCL_MIG_NOUPDATE -101

#endif
