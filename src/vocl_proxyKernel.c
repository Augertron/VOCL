#include <stdio.h>
#include <string.h>
#include "vocl_proxyStructures.h"
#include "vocl_proxyKernelArgProc.h"

static vocl_proxy_kernel *voclProxyKernelPtr = NULL;

void voclProxyAddKernel(cl_kernel kernel, char *kernelName, cl_program program)
{
    vocl_proxy_kernel *kernelPtr;

    kernelPtr = (vocl_proxy_kernel *)malloc(sizeof(vocl_proxy_kernel));
	kernelPtr->kernel = kernel;
	kernelPtr->program = program;
	kernelPtr->nameLen = strlen(kernelName)+1;
	kernelPtr->kernelName = (char *)malloc(kernelPtr->nameLen);
	memcpy(kernelPtr->kernelName, kernelName, kernelPtr->nameLen);
	kernelPtr->kernelName[kernelPtr->nameLen-1] = '\0';
	kernelPtr->argNum = 0;
	kernelPtr->argFlag = NULL;
	kernelPtr->args = NULL;
	kernelPtr->migStatus = 0;

    kernelPtr->next = voclProxyKernelPtr;
    voclProxyKernelPtr = kernelPtr;

    return;
}

vocl_proxy_kernel *voclProxyGetKernelPtr(cl_kernel kernel)
{
    vocl_proxy_kernel *kernelPtr;
    kernelPtr = voclProxyKernelPtr;
    while (kernelPtr != NULL)
    {
        if (kernelPtr->kernel == kernel)
        {
            break;
        }
		kernelPtr = kernelPtr->next;
    }

    if (kernelPtr == NULL)
    {
        printf("voclProxyGetKernelPtr, kernel %p does not exist!\n", kernel);
        exit (1);
    }

    return kernelPtr;
}

void voclProxySetKernelArgFlag(cl_kernel kernel, int argNum, char *argFlag)
{
	vocl_proxy_kernel *kernelPtr;
	int i;
	kernelPtr = voclProxyGetKernelPtr(kernel);
	kernelPtr->argNum = argNum;
	if (argNum > 0)
	{
		kernelPtr->argFlag = (char *)malloc(argNum * sizeof(char));
		memcpy(kernelPtr->argFlag, argFlag, argNum * sizeof(char));
		kernelPtr->args = (kernel_args *)malloc(sizeof(kernel_args) * argNum);
		memset(kernelPtr->args, 0, sizeof(kernel_args) * argNum);
		for (i = 0; i < argNum; i++)
		{
			kernelPtr->args[i].arg_index = -1;
		}
	}

	return;
}

char *voclProxyGetKernelArgFlagAll(cl_kernel kernel, int *argNum)
{
	vocl_proxy_kernel *kernelPtr;
	char *argFlag = NULL;

	kernelPtr = voclProxyGetKernelPtr(kernel);
	*argNum = kernelPtr->argNum;
	if (argNum > 0)
	{
		argFlag = (char *)malloc(kernelPtr->argNum * sizeof(char));
		memcpy(argFlag, kernelPtr->argFlag, kernelPtr->argNum * sizeof(char));
	}

	return argFlag;
}

char voclProxyKernelArgIsDeviceMem(cl_kernel kernel, int index)
{
	vocl_proxy_kernel *kernelPtr;
	kernelPtr = voclProxyGetKernelPtr(kernel);

	return kernelPtr->argFlag[index];
}

void voclProxyStoreKernelArgs(cl_kernel kernel, int argNum, kernel_args *args)
{
	cl_uint i, argIndex;
	vocl_proxy_kernel *kernelPtr;
	kernelPtr = voclProxyGetKernelPtr(kernel);

	for (i = 0; i < argNum; i++)
	{
		argIndex = args[i].arg_index;
		memcpy(&kernelPtr->args[argIndex], &args[i], sizeof(kernel_args));
	}

	return;
}

kernel_args *voclProxyGetKernelArg(cl_kernel kernel, cl_uint argIndex)
{
	vocl_proxy_kernel *kernelPtr;
	kernelPtr = voclProxyGetKernelPtr(kernel);

	return &kernelPtr->args[argIndex];
}

void voclProxySetKernelMigStatus(cl_kernel kernel, char migStatus)
{
	vocl_proxy_kernel *kernelPtr;
	kernelPtr = voclProxyGetKernelPtr(kernel);
	kernelPtr->migStatus = migStatus;
	return;
}

char voclProxyGetKernelMigStatus(cl_kernel kernel)
{
	vocl_proxy_kernel *kernelPtr;
	kernelPtr = voclProxyGetKernelPtr(kernel);
	return kernelPtr->migStatus;
}

void voclProxyReleaseKernel(cl_kernel kernel)
{
    vocl_proxy_kernel *kernelPtr, *preKernelPtr;

    /* if the cmdQueue is in the first node */
	kernelPtr = voclProxyKernelPtr;

	if (kernelPtr != NULL)
	{
		if (kernelPtr->kernel == kernel)
		{
			kernelPtr = voclProxyKernelPtr;
			voclProxyKernelPtr = kernelPtr->next;
			free(kernelPtr->kernelName);
			if (kernelPtr->argNum > 0)
			{
				free(kernelPtr->argFlag);
				free(kernelPtr->args);
			}
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
	if(kernelPtr->argNum > 0)
	{
		free(kernelPtr->argFlag);
		free(kernelPtr->args);
	}
    free(kernelPtr);

    return;
}

void voclProxyReleaseAllKernels()
{
    vocl_proxy_kernel *kernelPtr, *nextKernelPtr;

    kernelPtr = voclProxyKernelPtr;
    while (kernelPtr != NULL)
    {
        nextKernelPtr = kernelPtr->next;
		free(kernelPtr->kernelName);
		if(kernelPtr->argNum > 0)
		{
			free(kernelPtr->argFlag);
			free(kernelPtr->args);
		}
        free(kernelPtr);
        kernelPtr = nextKernelPtr;
    }

    voclProxyKernelPtr = NULL;

    return;
}

