#include <stdio.h>
#include <string.h>
#include <CL/opencl.h>
#include <sys/time.h>
#include "voclKernelArgProc.h"

/* for storing kernel arguments */
static kernel_info *kernelInfo = NULL;

#ifdef __cplusplus
extern "C" {
#endif

    cl_int createKernel(cl_kernel kernel);
    kernel_info *getKernelPtr(cl_kernel kernel);
    cl_int releaseKernelPtr(cl_kernel kernel);

#ifdef __cplusplus
}
#endif
/* for each kernel, a different pointer *//* is used for storing kernel arguments */
    cl_int createKernel(cl_kernel kernel)
{
    kernel_info *kernelPtr;
    kernelPtr = (kernel_info *) malloc(sizeof(kernel_info));
    kernelPtr->kernel = kernel;
    kernelPtr->args_num = 0;
    kernelPtr->args_allocated = 0;
    kernelPtr->args_ptr = NULL;
    kernelPtr->next = kernelInfo;

    kernelInfo = kernelPtr;
}

kernel_info *getKernelPtr(cl_kernel kernel)
{
    kernel_info *kernelPtr = NULL;

    kernel_info *nextKernel = kernelInfo;
    while (nextKernel != NULL) {
        if (kernel == nextKernel->kernel) {
            kernelPtr = nextKernel;
            break;
        }
        nextKernel = nextKernel->next;
    }

    if (kernelPtr == NULL) {
        printf("Error, kernel does not exist. In getKernelPtr!\n");
        exit(1);
    }

    return kernelPtr;
}

/* release the structure corresponing to the current kernel */
cl_int releaseKernelPtr(cl_kernel kernel)
{
    kernel_info *kernelPtr, *curKernel, *preKernel;
    if (kernel == kernelInfo->kernel) {
        kernelPtr = kernelInfo;
        kernelInfo = kernelInfo->next;
        if (kernelPtr->args_allocated == 1) {
            free(kernelPtr->args_ptr);
        }
        free(kernelPtr);
        return 0;
    }

    kernelPtr = NULL;
    curKernel = kernelInfo->next;
    preKernel = kernelInfo;
    while (curKernel != NULL) {
        if (kernel == curKernel->kernel) {
            kernelPtr = curKernel;
            break;
        }
        preKernel = curKernel;
        curKernel = curKernel->next;
    }

    if (kernelPtr == NULL) {
        printf("Kernel does not exist!\n");
        exit(1);
    }

    preKernel->next = curKernel->next;
    if (kernelPtr->args_allocated == 1) {
        free(kernelPtr->args_ptr);
    }
    free(kernelPtr);

    return 0;
}
