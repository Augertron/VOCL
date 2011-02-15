#ifndef __OPENCL_REMOTE_ACCESS_H__
#define __OPENCL_REMOTE_ACCESS_H__

int slaveComm;
int slaveCreated = 0;
int np = 1;
int argumentNo = 0;
int errCodes[MAX_NPS];

#define GET_PLATFORM_ID_FUNC      10000
#define GET_DEVICE_ID_FUNC        10001
#define CREATE_CONTEXT_FUNC       10002
#define CREATE_COMMAND_QUEUE_FUNC 10003
#define LOAD_SOURCE_FUNC          10004
#define CREATE_PROGRMA_WITH_SOURCE 10005
#define BUILD_PROGRAM			  10006
#define CREATE_KERNEL			  10007
#define ENQUEUE_WRITE_BUFFER	  10008
#define SET_KERNEL_ARG			  10009
#define ENQUEUE_ND_RANGE_KERNEL   10010
#define ENQUEUE_READ_BUFFER       10011
#define RELEASE_MEM_OBJ			  10012



extern CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformIDs(cl_uint          /* num_entries */,
				 cl_platform_id * /* platforms */,
				 cl_uint *        /* num_platforms */) CL_API_SUFFIX__VERSION_1_0;

/* Device APIs */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceIDs(cl_platform_id   /* platform */,
               cl_device_type   /* device_type */,
               cl_uint          /* num_entries */,
               cl_device_id *   /* devices */,
               cl_uint *        /* num_devices */) CL_API_SUFFIX__VERSION_1_0;

/* Context APIs  */
extern CL_API_ENTRY cl_context CL_API_CALL
clCreateContext(const cl_context_properties * /* properties */,
                cl_uint                       /* num_devices */,
                const cl_device_id *          /* devices */,
                void (CL_CALLBACK * /* pfn_notify */)(const char *, const void *, size_t, void *),
                void *                        /* user_data */,
                cl_int *                      /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

/* Command Queue APIs */
extern CL_API_ENTRY cl_command_queue CL_API_CALL
clCreateCommandQueue(cl_context                     /* context */,
                     cl_device_id                   /* device */,
                     cl_command_queue_properties    /* properties */,
                     cl_int *                       /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

char * loadSource(char *filePathName, size_t *fileSize);

extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithSource(cl_context        /* context */,
                          cl_uint           /* count */,
                          const char **     /* strings */,
                          const size_t *    /* lengths */,
                          cl_int *          /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
clBuildProgram(cl_program           /* program */,
               cl_uint              /* num_devices */,
               const cl_device_id * /* device_list */,
               const char *         /* options */, 
               void (CL_CALLBACK *  /* pfn_notify */)(cl_program /* program */, void * /* user_data */),
               void *               /* user_data */) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_kernel CL_API_CALL
clCreateKernel(cl_program      /* program */,
               const char *    /* kernel_name */,
               cl_int *        /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

/* Memory Object APIs */
extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateBuffer(cl_context   /* context */,
               cl_mem_flags /* flags */,
               size_t       /* size */,
               void *       /* host_ptr */,
               cl_int *     /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteBuffer(cl_command_queue   /* command_queue */, 
                     cl_mem             /* buffer */, 
                     cl_bool            /* blocking_write */, 
                     size_t             /* offset */, 
                     size_t             /* cb */, 
                     const void *       /* ptr */, 
                     cl_uint            /* num_events_in_wait_list */, 
                     const cl_event *   /* event_wait_list */, 
                     cl_event *         /* event */) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
clSetKernelArg(cl_kernel    /* kernel */,
               cl_uint      /* arg_index */,
               size_t       /* arg_size */,
               const void * /* arg_value */) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueNDRangeKernel(cl_command_queue /* command_queue */,
                       cl_kernel        /* kernel */,
                       cl_uint          /* work_dim */,
                       const size_t *   /* global_work_offset */,
                       const size_t *   /* global_work_size */,
                       const size_t *   /* local_work_size */,
                       cl_uint          /* num_events_in_wait_list */,
                       const cl_event * /* event_wait_list */,
                       cl_event *       /* event */) CL_API_SUFFIX__VERSION_1_0;

/* Enqueued Commands APIs */
extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadBuffer(cl_command_queue    /* command_queue */,
                    cl_mem              /* buffer */,
                    cl_bool             /* blocking_read */,
                    size_t              /* offset */,
                    size_t              /* cb */,
                    void *              /* ptr */,
                    cl_uint             /* num_events_in_wait_list */,
                    const cl_event *    /* event_wait_list */,
                    cl_event *          /* event */) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseMemObject(cl_mem /* memobj */) CL_API_SUFFIX__VERSION_1_0;

#endif

