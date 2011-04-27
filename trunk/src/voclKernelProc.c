#include "vocl_structures.h"

static struct strVOCLKernel *voclKernelPtr = NULL;
static vocl_kernel voclKernel;
static int voclKernelNum;
static int voclKernelNo;

static vocl_kernel getVOCLKernelValue()
{
    vocl_kernel kernel = voclKernel;
	voclKernel++;

    return kernel;
}

static struct strVOCLKernel *getVOCLKernelPtr()
{
    if (voclKernelNo >= voclKernelNum) {
        voclKernelNum *= 2;
        voclKernelPtr = (struct strVOCLKernel *) realloc(voclKernelPtr,
                                                   voclKernelNum *
                                                   sizeof(struct strVOCLKernel));
    }
    return &voclKernelPtr[voclKernelNo++];
}


void voclKernelInitialize()
{
    voclKernelNum = VOCL_CONTEXT_NUM;
    voclKernelPtr =
        (struct strVOCLKernel *) malloc(voclKernelNum * sizeof(struct strVOCLKernel));
    voclKernelNo = 0;
    voclKernel = 0;
}

void voclKernelFinalize()
{
    if (voclKernelPtr != NULL) {
        free(voclKernelPtr);
        voclKernelPtr = NULL;
    }
    voclKernelNo = 0;
    voclKernel = 0;
    voclKernelNum = 0;
}

vocl_kernel voclCLKernel2VOCLKernel(cl_kernel kernel, int proxyID)
{
    struct strVOCLKernel *kernelPtr = getVOCLKernelPtr();
    kernelPtr->clKernel = kernel;
	kernelPtr->proxyID = proxyID;
    kernelPtr->voclKernel = getVOCLKernelValue();

    return kernelPtr->voclKernel;
}

cl_kernel voclVOCLKernel2CLKernelComm(vocl_kernel kernel, int *proxyID)
{
    /* the vocl kernel value indicates its location */
    /* in the event buffer */
    int kernelNo = (int) kernel;
	*proxyID = voclKernelPtr[kernelNo].proxyID;

    return voclKernelPtr[kernelNo].clKernel;
}

