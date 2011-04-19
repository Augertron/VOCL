#ifndef __VOCL_PROXY_KERNEL_ARG_H__
#define __VOCL_PROXY_KERNEL_ARG_H__

#include <CL/opencl.h>

cl_int err;
typedef struct strKernelArgs {
	cl_uint arg_index;
	size_t  arg_size;
	char    arg_value[64];
	cl_char arg_null_flag;
} kernel_args;

#endif

