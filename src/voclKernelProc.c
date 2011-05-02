#include <stdio.h>
#include "vocl_structures.h"

static struct strVOCLKernel *voclKernelPtr = NULL;
static vocl_kernel voclKernel;
static int voclKernelNo;

static vocl_kernel getVOCLKernelValue()
{
    vocl_kernel kernel = voclKernel;
	voclKernel++;

    return kernel;
}

static struct strVOCLKernel *createVOCLKernel()
{
	struct strVOCLKernel *kernelPtr;
	kernelPtr = (struct strVOCLKernel *)malloc(sizeof(struct strVOCLKernel));
	kernelPtr->next = voclKernelPtr;
	voclKernelPtr = kernelPtr;

	return kernelPtr;
}

static struct strVOCLKernel *getVOCLKernelPtr(vocl_kernel kernel)
{
	struct strVOCLKernel *kernelPtr;
	kernelPtr = voclKernelPtr;
	while (kernelPtr != NULL)
	{
		if (kernelPtr->voclKernel == kernel)
		{
			break;
		}
		kernelPtr = kernelPtr->next;
	}

	if (kernelPtr == NULL)
	{
		printf("Error, kernel does not exist!\n");
		exit (1);
	}

	return kernelPtr;
}

void voclKernelInitialize()
{
    voclKernelPtr = NULL;
    voclKernelNo = 0;
    voclKernel = 0;
}

void voclKernelFinalize()
{
	struct strVOCLKernel *kernelPtr, *tmpkernelPtr;
	kernelPtr = voclKernelPtr;
	while (kernelPtr != NULL)
	{
		tmpkernelPtr = kernelPtr->next;
		free(kernelPtr);
		kernelPtr = tmpkernelPtr;
	}

    voclKernelPtr = NULL;
    voclKernelNo = 0;
    voclKernel = 0;
}

vocl_kernel voclCLKernel2VOCLKernel(cl_kernel kernel, int proxyID,
                int proxyIndex, MPI_Comm proxyComm, MPI_Comm proxyCommData)
{
    struct strVOCLKernel *kernelPtr = createVOCLKernel();
    kernelPtr->clKernel = kernel;
	kernelPtr->proxyID = proxyID;
	kernelPtr->proxyIndex = proxyIndex;
	kernelPtr->proxyComm = proxyComm;
	kernelPtr->proxyCommData = proxyCommData;
    kernelPtr->voclKernel = getVOCLKernelValue();

    return kernelPtr->voclKernel;
}

cl_kernel voclVOCLKernel2CLKernelComm(vocl_kernel kernel, int *proxyID,
              int *proxyIndex, MPI_Comm *proxyComm, MPI_Comm *proxyCommData)
{
	struct strVOCLKernel *kernelPtr = getVOCLKernelPtr(kernel);
	*proxyID = kernelPtr->proxyID;
	*proxyIndex = kernelPtr->proxyIndex;
	*proxyComm = kernelPtr->proxyComm;
	*proxyCommData = kernelPtr->proxyCommData;

    return kernelPtr->clKernel;
}

int voclReleaseKernel(vocl_kernel kernel)
{
	struct strVOCLKernel *kernelPtr, *preKernelPtr, *curKernelPtr;
	/* the first node in the link list */
	if (kernel == voclKernelPtr->voclKernel)
	{
		kernelPtr = voclKernelPtr;
		voclKernelPtr = voclKernelPtr->next;
		free(kernelPtr);

		return 0;
	}

	kernelPtr = NULL;
	preKernelPtr = voclKernelPtr;
	curKernelPtr = voclKernelPtr->next;
	while (curKernelPtr != NULL)
	{
		if (kernel == curKernelPtr->voclKernel)
		{
			kernelPtr = curKernelPtr;
			break;
		}
		preKernelPtr = curKernelPtr;
		curKernelPtr = curKernelPtr->next;
	}

	if (kernelPtr == NULL)
	{
		printf("kernel does not exist!\n");
		exit (1);
	}

	/* remote the current node from link list */
	preKernelPtr->next = curKernelPtr->next;
	free(curKernelPtr);
	
	return 0;
}
