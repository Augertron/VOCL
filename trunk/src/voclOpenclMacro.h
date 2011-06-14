#ifndef __VOCL_LIB_MACRO_H__
#define __VOCL_LIB_MACRO_H__

#ifdef __cplusplus
extern "C" {
#endif

/* number of slave process */
#define MAX_NPS 100

/* for kernel arguments */
#define MAX_TAG 65535

/* vocl true and false */
#define VOCL_TRUE  1
#define VOCL_FALSE 0

/* parameters of buffer pool */
/* write buffer */
#define VOCL_WRITE_BUFFER_NUM 8
//#define VOCL_WRITE_BUFFER_SIZE 64108864 /* 64MB 1024 X 1024 X 64 */
#define VOCL_WRITE_BUFFER_SIZE 33554432 /* 64MB 1024 X 1024 X 64 */
#define VOCL_WRITE_TAG 5000

/* read buffer */
#define VOCL_READ_BUFFER_NUM 16
//#define VOCL_READ_BUFFER_SIZE 64108864  /* 64MB 1024 X 1024 X 64 */
#define VOCL_READ_BUFFER_SIZE 33554432  /* 64MB 1024 X 1024 X 64 */
#define VOCL_READ_TAG 4000

/* macros for migration */
#define VOCL_MIG_BUF_NUM 8
#define VOCL_MIG_POOL_NUM 4
//#define VOCL_MIG_BUF_SIZE 64108864
#define VOCL_MIG_BUF_SIZE 33554432
#define VOCL_MIG_TAG 6000

#define VOCL_MIG_LOCAL_WT_BUF_AVALB     0
#define VOCL_MIG_LOCAL_WT_BUF_WAITDATA  1
#define VOCL_MIG_LOCAL_WT_BUF_WTGPUMEM  2

#define VOCL_MIG_LOCAL_RD_BUF_AVALB     3
#define VOCL_MIG_LOCAL_RD_BUF_RDGPUMEM  4
#define VOCL_MIG_LOCAL_RD_BUF_MPISEND   5

/* flag use for local to local migration */
#define VOCL_MIG_LOCAL_RW_BUF_AVALB     6
#define VOCL_MIG_LOCAL_RW_BUF_RDGPUMEM  7
#define VOCL_MIG_LOCAL_RW_BUF_WTGPUMEM  8

/* for api function identification */
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
#define GET_CONTEXT_INFO_FUNC       25
#define CL_RELEASE_KERNEL_FUNC      26
#define GET_BUILD_INFO_FUNC         27
#define GET_PROGRAM_INFO_FUNC       28
#define REL_PROGRAM_FUNC            29
#define REL_COMMAND_QUEUE_FUNC      30
#define REL_CONTEXT_FUNC            31
#define GET_DEVICE_INFO_FUNC        32
#define GET_PLATFORM_INFO_FUNC      33
#define FLUSH_FUNC                  34
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
#define MIGRATION_CHECK             57
#define FORCED_MIGRATION            58
#define PROGRAM_END                 59

#define CMSG_NUM                    60
#define DATAMSG_NUM                 100
#define TOTAL_MSG_NUM               (CMSG_NUM + DATAMSG_NUM)

#define GET_PLATFORM_ID_FUNC1       10000
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
#define GET_CONTEXT_INFO_FUNC1      10017
#define GET_BUILD_INFO_FUNC1        10018
#define GET_PROGRAM_INFO_FUNC1      10019
#define GET_DEVICE_INFO_FUNC1       10020
#define GET_PLATFORM_INFO_FUNC1     10021
#define WAIT_FOR_EVENT_FUNC1        10022
#define GET_CMD_QUEUE_INFO_FUNC1    10023
#define ENQUEUE_MAP_BUFF_FUNC1      10024
#define GET_EVENT_PROF_INFO_FUNC1   10025
#define GET_KERNEL_WGP_INFO_FUNC1   10026
#define CREATE_IMAGE_2D_FUNC1       10027
#define ENQ_COPY_BUFF_FUNC1         10028
#define ENQ_UNMAP_MEMOBJ_FUNC1      10029

#ifdef __cplusplus
}
#endif
#endif
