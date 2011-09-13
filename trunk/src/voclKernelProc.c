#include <stdio.h>
#include <string.h>
#include "voclStructures.h"

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
    kernelPtr = (struct strVOCLKernel *) malloc(sizeof(struct strVOCLKernel));
    kernelPtr->kernelName = NULL;
    kernelPtr->next = voclKernelPtr;
    voclKernelPtr = kernelPtr;

    return kernelPtr;
}

struct strVOCLKernel *voclGetKernelPtr(vocl_kernel kernel)
{
    struct strVOCLKernel *kernelPtr;
    kernelPtr = voclKernelPtr;
    while (kernelPtr != NULL) {
        if (kernelPtr->voclKernel == kernel) {
            break;
        }
        kernelPtr = kernelPtr->next;
    }

    if (kernelPtr == NULL) {
        printf("Error, kernel does not exist!\n");
        exit(1);
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
    while (kernelPtr != NULL) {
        tmpkernelPtr = kernelPtr->next;
        if (kernelPtr->kernelName != NULL) {
            free(kernelPtr->kernelName);
        }
        free(kernelPtr);
        kernelPtr = tmpkernelPtr;
    }

    voclKernelPtr = NULL;
    voclKernelNo = 0;
    voclKernel = 0;
}

vocl_kernel voclCLKernel2VOCLKernel(cl_kernel kernel, int proxyRank,
                                    int proxyIndex, MPI_Comm proxyComm, MPI_Comm proxyCommData)
{
    struct strVOCLKernel *kernelPtr = createVOCLKernel();
    kernelPtr->clKernel = kernel;
    kernelPtr->proxyRank = proxyRank;
    kernelPtr->proxyIndex = proxyIndex;
    kernelPtr->proxyComm = proxyComm;
    kernelPtr->proxyCommData = proxyCommData;
    kernelPtr->voclKernel = getVOCLKernelValue();

    return kernelPtr->voclKernel;
}

void voclStoreKernelName(vocl_kernel kernel, char *kernelName)
{
    struct strVOCLKernel *kernelPtr = voclGetKernelPtr(kernel);
    int len = strlen(kernelName) + 1;
    kernelPtr->kernelName = (char *) malloc(len);
    strcpy(kernelPtr->kernelName, kernelName);

    return;
}

void voclStoreKernelProgramContext(vocl_kernel kernel, vocl_program program,
                                   vocl_context context)
{
    struct strVOCLKernel *kernelPtr = voclGetKernelPtr(kernel);
    kernelPtr->program = program;
    kernelPtr->context = context;
}

vocl_program voclGetProgramFromKernel(vocl_kernel kernel)
{
    struct strVOCLKernel *kernelPtr = voclGetKernelPtr(kernel);
    return kernelPtr->program;
}

vocl_context voclGetContextFromKernel(vocl_kernel kernel)
{
    struct strVOCLKernel *kernelPtr = voclGetKernelPtr(kernel);
    return kernelPtr->context;
}

void voclKernelSetMigrationStatus(vocl_kernel kernel, char status)
{
	struct strVOCLKernel *kernelPtr = voclGetKernelPtr(kernel);
	kernelPtr->migrationStatus = status;
	return;
}

char voclKernelGetMigrationStatus(vocl_kernel kernel)
{
	struct strVOCLKernel *kernelPtr = voclGetKernelPtr(kernel);
	return kernelPtr->migrationStatus;
}

cl_kernel voclVOCLKernel2CLKernelComm(vocl_kernel kernel, int *proxyRank,
                                      int *proxyIndex, MPI_Comm * proxyComm,
                                      MPI_Comm * proxyCommData)
{
    struct strVOCLKernel *kernelPtr = voclGetKernelPtr(kernel);
    *proxyRank = kernelPtr->proxyRank;
    *proxyIndex = kernelPtr->proxyIndex;
    *proxyComm = kernelPtr->proxyComm;
    *proxyCommData = kernelPtr->proxyCommData;

    return kernelPtr->clKernel;
}

cl_kernel voclVOCLKernel2CLKernel(vocl_kernel kernel)
{
    struct strVOCLKernel *kernelPtr = voclGetKernelPtr(kernel);
    return kernelPtr->clKernel;
}

void voclUpdateVOCLKernel(vocl_kernel voclKernel, cl_kernel newKernel, int proxyRank, 
						  int proxyIndex, MPI_Comm proxyComm, MPI_Comm proxyCommData)
{
    struct strVOCLKernel *kernelPtr = voclGetKernelPtr(voclKernel);
    int err;

    kernelPtr->proxyRank = proxyRank;
    kernelPtr->proxyIndex = proxyIndex;
    kernelPtr->proxyComm = proxyComm;
    kernelPtr->proxyCommData = proxyCommData;
    kernelPtr->clKernel = newKernel;

    return;
}

int voclReleaseKernel(vocl_kernel kernel)
{
    struct strVOCLKernel *kernelPtr, *preKernelPtr, *curKernelPtr;
    /* the first node in the link list */
    if (kernel == voclKernelPtr->voclKernel) {
        kernelPtr = voclKernelPtr;
        voclKernelPtr = voclKernelPtr->next;
        free(kernelPtr);

        return 0;
    }

    kernelPtr = NULL;
    preKernelPtr = voclKernelPtr;
    curKernelPtr = voclKernelPtr->next;
    while (curKernelPtr != NULL) {
        if (kernel == curKernelPtr->voclKernel) {
            kernelPtr = curKernelPtr;
            break;
        }
        preKernelPtr = curKernelPtr;
        curKernelPtr = curKernelPtr->next;
    }

    if (kernelPtr == NULL) {
        printf("kernel does not exist!\n");
        exit(1);
    }

    /* remote the current node from link list */
    preKernelPtr->next = curKernelPtr->next;
    free(curKernelPtr);

    return 0;
}
