#ifndef __VOCL_KERNEL_ARG_PROC_H__
#define __VOCL_KERNEL_ARG_PROC_H__

#include <stdio.h>
#include <string.h>
#include <CL/opencl.h>
#include <sys/time.h>

typedef struct strKernelArgs {
    cl_uint arg_index;
    size_t arg_size;
    char arg_value[64];
    cl_char arg_null_flag;
} kernel_args;

typedef struct strKernelInfo {
    cl_kernel kernel;
    cl_uint args_num;
    cl_bool args_allocated;
    kernel_args *args_ptr;
    struct strKernelInfo *next;
} kernel_info;

#endif