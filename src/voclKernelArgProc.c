#include <stdio.h>
#include <string.h>
#include <CL/opencl.h>
#include <sys/time.h>
#include "vocl_structures.h"
#include "voclKernelArgProc.h"

/* for storing kernel arguments */
static kernel_info *kernelInfo = NULL;
extern char *voclGetProgramSource(vocl_program program, size_t *sourceSize);
extern void voclRemoveComments(char *sourceIn, size_t inLen, char *sourceOut, size_t *outLen);
extern char *voclKernelPrototye(char *sourceIn, char *kernelName, unsigned int *kernelArgNum);

#ifdef __cplusplus
extern "C" {
#endif

cl_int createKernel(cl_kernel kernel);
kernel_info *getKernelPtr(cl_kernel kernel);
cl_int releaseKernelPtr(cl_kernel kernel);
void createKernelArgInfo(cl_kernel kernel, char *kernel_name, vocl_program program);

#ifdef __cplusplus
}
#endif
/* for each kernel, a different pointer *//* is used for storing kernel arguments */
cl_int createKernel(cl_kernel kernel)
{
    kernel_info *kernelPtr;
    kernelPtr = (kernel_info *) malloc(sizeof(kernel_info));
    kernelPtr->kernel = kernel;
	kernelPtr->globalMemSize = 0;
    kernelPtr->args_num = 0;
    kernelPtr->args_allocated = 0;
    kernelPtr->args_ptr = NULL;
	kernelPtr->args_flag = NULL;
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
		if (kernelPtr->args_flag != NULL)
		{
			free(kernelPtr->args_flag);
			kernelPtr->args_flag = NULL;
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
	if (kernelPtr->args_flag != NULL)
	{
		free(kernelPtr->args_flag);
		kernelPtr->args_flag = NULL;
	}
    free(kernelPtr);

    return 0;
}

void createKernelArgInfo(cl_kernel kernel, char *kernel_name, vocl_program program)
{
	kernel_info *kernelPtr;
	size_t sourceSize, codeSize;
	char *programSource;
	char *codeSource;

	kernelPtr = getKernelPtr(kernel);
	programSource = voclGetProgramSource(program, &sourceSize);

	codeSource = (char *)malloc(sourceSize * sizeof(char));

	/* remove all comments in the program source */
	voclRemoveComments(programSource, sourceSize, codeSource, &codeSize);

	/* get argument info of the kernel */
	kernelPtr->args_flag = voclKernelPrototye(codeSource, kernel_name, &kernelPtr->kernel_arg_num);
	
	free(codeSource);

	return;
}

