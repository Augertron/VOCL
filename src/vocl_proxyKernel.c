#include <stdio.h>
#include <string.h>
#include "vocl_proxyStructures.h"
#include "vocl_proxyKernelArgProc.h"

extern cl_mem voclProxyGetNewMemValue(cl_mem oldMem);

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
	printf("kernelSetFlag1\n");
	kernelPtr = voclProxyGetKernelPtr(kernel);
	printf("kernelSetFlag2\n");
	kernelPtr->argNum = argNum;
	if (argNum > 0)
	{
	printf("kernelSetFlag3\n");
		kernelPtr->argFlag = (char *)malloc(argNum * sizeof(char));
	printf("kernelSetFlag4\n");
		memcpy(kernelPtr->argFlag, argFlag, argNum * sizeof(char));
	printf("kernelSetFlag5\n");
		kernelPtr->args = (kernel_args *)malloc(sizeof(kernel_args) * argNum);
	printf("kernelSetFlag6\n");
		memset(kernelPtr->args, 0, sizeof(kernel_args) * argNum);
	printf("kernelSetFlag7\n");
		for (i = 0; i < argNum; i++)
		{
			kernelPtr->args[i].arg_index = -1;
		}
	printf("kernelSetFlag8\n");
	}

	printf("kernelSetFlag9\n");
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
		if (argIndex != -1)
		{
			memcpy(&kernelPtr->args[argIndex], &args[i], sizeof(kernel_args));
		}
	}

	return;
}

/* if migration is performed, check whether kernel arguments */
/* should be updated to the new value */
void voclProxyUpdateKernelArgs(cl_kernel kernel, int argNum, kernel_args *args)
{
	cl_uint i, argIndex;
	char kernelMigStatus, memMigStatus;
	cl_mem oldMem, newMem;
	vocl_proxy_kernel *kernelPtr;
	
	kernelPtr = voclProxyGetKernelPtr(kernel);
	kernelMigStatus = kernelPtr->migStatus;

	/* no migration is performed, return directly */
	if (kernelMigStatus == 0)
	{
		return;
	}

	for (i = 0; i < argNum; i++)
	{
		argIndex = args[i].arg_index;
		/* device memory */
		if (kernelPtr->argFlag[argIndex] == 1)
		{
			/* migration is performed, but memory value is not update. */
			if (args[i].migStatus < kernelMigStatus)
			{
				oldMem = *((cl_mem *)args[i].arg_value);
				newMem = voclProxyGetNewMemValue(oldMem);
				memcpy(args[i].arg_value, &newMem, args[i].arg_size);
				args[i].migStatus = kernelMigStatus;
			}
		}
	}

	return;
}

/* if migration is performed, set the kernel args on the new proxy process */
void voclProxySetKernelArgs(cl_kernel kernel)
{
	cl_uint i;
	cl_int retCode;
	char kernelMigStatus;
	cl_mem oldMem, newMem;
	vocl_proxy_kernel *kernelPtr;
	printf("proxySetKernelArg0\n");
	kernelPtr = voclProxyGetKernelPtr(kernel);
	kernelMigStatus = kernelPtr->migStatus;

	printf("proxySetKernelArg1\n");
	for (i = 0; i < kernelPtr->argNum; i++)
	{
		/* if this argument is also set */
	printf("proxySetKernelArg2\n");
		if (kernelPtr->args[i].arg_index != -1)
		{
			/* the current argument is a device memory */
	printf("proxySetKernelArg3\n");
			if (kernelPtr->argFlag[i] == 1)
			{
	printf("proxySetKernelArg4\n");
				/* migration is performed */
				if (kernelPtr->args[i].migStatus < kernelMigStatus)
				{
	printf("proxySetKernelArg5\n");
					oldMem = *((cl_mem *)kernelPtr->args[i].arg_value);
					newMem = voclProxyGetNewMemValue(oldMem);
					retCode = clSetKernelArg(kernel, 
											 kernelPtr->args[i].arg_index, 
											 kernelPtr->args[i].arg_size,
											 (const void *)&newMem);

					/* update the kernel argument and the migration status */
					kernelPtr->args[i].migStatus = kernelMigStatus;
					memcpy((void *)kernelPtr->args[i].arg_value, &newMem, kernelPtr->args[i].arg_size);
	printf("proxySetKernelArg6\n");
				}
				else
				{
	printf("proxySetKernelArg7\n");
					retCode = clSetKernelArg(kernel, 
											 kernelPtr->args[i].arg_index, 
											 kernelPtr->args[i].arg_size,
											 kernelPtr->args[i].arg_value);
	printf("proxySetKernelArg8\n");
				}
			}
			else /* other types of kernel arguments */
			{
	printf("proxySetKernelArg9\n");
				/* argument value is NULL */
				if (kernelPtr->args[i].arg_null_flag == 1)
				{
					retCode = clSetKernelArg(kernel, 
											 kernelPtr->args[i].arg_index, 
											 kernelPtr->args[i].arg_size,
											 NULL);
				}
				else
				{
					retCode = clSetKernelArg(kernel, 
											 kernelPtr->args[i].arg_index, 
											 kernelPtr->args[i].arg_size,
											 kernelPtr->args[i].arg_value);
				}
			}
		}

		if (retCode != CL_SUCCESS)
		{
			printf("voclProxySetKernelArgs, set kernel argument %d error, %d\n",
					i, retCode);
			exit (1);
		}
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

void voclProxyStoreOldKernelValue(cl_kernel kernel, cl_kernel oldKernel)
{
    vocl_proxy_kernel *kernelPtr;
    kernelPtr = voclProxyGetKernelPtr(kernel);
    kernelPtr->oldKernel = oldKernel;

    return;
}

cl_kernel voclProxyGetOldKernelValue(cl_kernel kernel)
{
    vocl_proxy_kernel *kernelPtr;
    kernelPtr = voclProxyGetKernelPtr(kernel);
    return kernelPtr->oldKernel;
}

cl_kernel voclProxyGetNewKernelValue(cl_kernel oldKernel)
{
    vocl_proxy_kernel *kernelPtr;
	kernelPtr = voclProxyKernelPtr;
	while (kernelPtr != NULL)
	{
		if (kernelPtr->oldKernel == oldKernel)
		{
			break;
		}
	}

	if (kernelPtr == NULL)
	{
		printf("voclProxyGetNewKernelValue, old kernel %p does not exist!\n", oldKernel);
		exit (1);
	}

    return kernelPtr->kernel;
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

