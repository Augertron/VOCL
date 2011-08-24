#include <stdio.h>
#include <string.h>
#include "vocl_proxyStructures.h"

static str_vocl_proxy_kernel *voclProxyKernelPtr = NULL;

void voclProxyAddKernel(cl_kernel kernel, char *kernelName, cl_program program)
{
    str_vocl_proxy_kernel *kernelPtr;
	int kernelNameLen;

    kernelPtr = (str_vocl_proxy_kernel *)malloc(sizeof(str_vocl_proxy_kernel));
	kernelPtr->kernel = kernel;
	kernelPtr->program = program;
	kernelNameLen = strlen(kernelName);
	kernelPtr->kernelName = (char *)malloc(kernelNameLen);
	memcpy(kernelPtr->kernelName, kernelName, kernelNameLen);

    kernelPtr->next = voclProxyKernelPtr;
    voclProxyKernelPtr = kernelPtr;

    return;
}

str_vocl_proxy_kernel *voclProxyGetKernelPtr(cl_kernel kernel)
{
    str_vocl_proxy_kernel *kernelPtr;
    kernelPtr = voclProxyKernelPtr;
    while (kernelPtr != NULL)
    {
        if (kernelPtr->kernel == kernel)
        {
            break;
        }
    }

    if (kernelPtr == NULL)
    {
        printf("voclProxyGetKernelPtr, kernel %p does not exist!\n", kernel);
        exit (1);
    }

    return kernelPtr;
}

void voclProxyReleaseKernel(cl_kernel kernel)
{
    str_vocl_proxy_kernel *kernelPtr, *preKernelPtr;

    /* if the cmdQueue is in the first node */
	kernelPtr = voclProxyKernelPtr;

	if (kernelPtr != NULL)
	{
		if (kernelPtr->kernel == kernel)
		{
			kernelPtr = voclProxyKernelPtr;
			voclProxyKernelPtr = kernelPtr->next;
			free(kernelPtr->kernelName);
			free(kernelPtr);
			return;
		}

		preKernelPtr = voclProxyKernelPtr;
		kernelPtr = preKernelPtr->next;
		while (kernelPtr != NULL)
		{
			if (kernelPtr->kernel == kernel)
			{
				break;
			}

			preKernelPtr = kernelPtr;
			kernelPtr = kernelPtr->next;
		}
	}

    if (kernelPtr == NULL)
    {
        printf("voclProxyReleaseKernel, Kernel %p does not exist!\n", kernel);
        exit (1);
    }

    preKernelPtr->next = kernelPtr->next;
	free(kernelPtr->kernelName);
    free(kernelPtr);

    return;
}

void voclProxyReleaseAllKernels()
{
    str_vocl_proxy_kernel *kernelPtr, *nextKernelPtr;

    kernelPtr = voclProxyKernelPtr;
    while (kernelPtr != NULL)
    {
        nextKernelPtr = kernelPtr->next;
		free(kernelPtr->kernelName);
        free(kernelPtr);
        kernelPtr = nextKernelPtr;
    }

    voclProxyKernelPtr = NULL;

    return;
}

