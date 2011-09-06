#ifndef __VOCL_PROXY_KERNEL_ARG_H__
#define __VOCL_PROXY_KERNEL_ARG_H__

#include <CL/opencl.h>

cl_int err;
typedef struct strKernelArgs {
    cl_uint arg_index;
    size_t arg_size;
    char arg_value[32];
    cl_char arg_null_flag;
	cl_int  isGlobalMemory;
	char migStatus;
    cl_mem memory;
	size_t globalSize;
} kernel_args;

#endif
