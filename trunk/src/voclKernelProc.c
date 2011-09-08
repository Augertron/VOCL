#include <stdio.h>
#include <string.h>
#include "voclStructures.h"

extern cl_kernel voclMigCreateKernel(vocl_program program, const char *kernel_name,
                              cl_int * errcode_ret);
extern char voclProgramGetMigrationStatus(vocl_program program);
extern char voclContextGetMigrationStatus(vocl_context context);
extern cl_context voclVOCLContext2CLContextComm(vocl_context context, int *proxyRank,
					int *proxyIndex, MPI_Comm * proxyComm,
					MPI_Comm * proxyCommData);
extern void voclUpdateVOCLProgram(vocl_program voclProgram, int proxyRank, int proxyIndex,
                           MPI_Comm proxyComm, MPI_Comm proxyCommData, vocl_context context);
extern vocl_device_id voclGetCommandQueueDeviceID(vocl_command_queue cmdQueue);
extern char *voclGetProgramBuildOptions(vocl_program program);

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

static struct strVOCLKernel *getVOCLKernelPtr(vocl_kernel kernel)
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
    struct strVOCLKernel *kernelPtr = getVOCLKernelPtr(kernel);
    int len = strlen(kernelName) + 1;
    kernelPtr->kernelName = (char *) malloc(len);
    strcpy(kernelPtr->kernelName, kernelName);

    return;
}

void voclStoreKernelProgramContext(vocl_kernel kernel, vocl_program program,
                                   vocl_context context)
{
    struct strVOCLKernel *kernelPtr = getVOCLKernelPtr(kernel);
    kernelPtr->program = program;
    kernelPtr->context = context;
}

vocl_program voclGetProgramFromKernel(vocl_kernel kernel)
{
    struct strVOCLKernel *kernelPtr = getVOCLKernelPtr(kernel);
    return kernelPtr->program;
}

vocl_context voclGetContextFromKernel(vocl_kernel kernel)
{
    struct strVOCLKernel *kernelPtr = getVOCLKernelPtr(kernel);
    return kernelPtr->context;
}

void voclKernelSetMigrationStatus(vocl_kernel kernel, char status)
{
	struct strVOCLKernel *kernelPtr = getVOCLKernelPtr(kernel);
	kernelPtr->migrationStatus = status;
	return;
}

char voclKernelGetMigrationStatus(vocl_kernel kernel)
{
	struct strVOCLKernel *kernelPtr = getVOCLKernelPtr(kernel);
	return kernelPtr->migrationStatus;
}

cl_kernel voclVOCLKernel2CLKernelComm(vocl_kernel kernel, int *proxyRank,
                                      int *proxyIndex, MPI_Comm * proxyComm,
                                      MPI_Comm * proxyCommData)
{
    struct strVOCLKernel *kernelPtr = getVOCLKernelPtr(kernel);
    *proxyRank = kernelPtr->proxyRank;
    *proxyIndex = kernelPtr->proxyIndex;
    *proxyComm = kernelPtr->proxyComm;
    *proxyCommData = kernelPtr->proxyCommData;

    return kernelPtr->clKernel;
}

cl_kernel voclVOCLKernel2CLKernel(vocl_kernel kernel)
{
    struct strVOCLKernel *kernelPtr = getVOCLKernelPtr(kernel);
    return kernelPtr->clKernel;
}

void voclUpdateVOCLKernel(vocl_kernel voclKernel, int proxyRank, int proxyIndex,
                          MPI_Comm proxyComm, MPI_Comm proxyCommData, vocl_program program)
{
    struct strVOCLKernel *kernelPtr = getVOCLKernelPtr(voclKernel);
    int err;

    /* release previous kernel */
    clReleaseKernel((cl_kernel)voclKernel);

    kernelPtr->proxyRank = proxyRank;
    kernelPtr->proxyIndex = proxyIndex;
    kernelPtr->proxyComm = proxyComm;
    kernelPtr->proxyCommData = proxyCommData;
    kernelPtr->clKernel = voclMigCreateKernel(program, kernelPtr->kernelName, &err);
    if (err != CL_SUCCESS) {
        printf("create kernel error, %d!\n", err);
    }
	kernelPtr->migrationStatus = voclProgramGetMigrationStatus(program);

    return;
}

void voclUpdateSingleKernel(vocl_kernel kernel, vocl_command_queue command_queue)
{
	vocl_context context; 
	vocl_program program;
	vocl_device_id deviceID;
	cl_context clContext;
	int proxyRank, proxyIndex;
	MPI_Comm proxyComm, proxyCommData;
	int contextMigStatus, programMigStatus, kernelMigStatus;

	context = voclGetContextFromKernel(kernel);
	program = voclGetProgramFromKernel(kernel);

	clContext = voclVOCLContext2CLContextComm(context, &proxyRank, &proxyIndex,
					&proxyComm, &proxyCommData);

	contextMigStatus = voclContextGetMigrationStatus(context);
	programMigStatus = voclProgramGetMigrationStatus(program);
	kernelMigStatus = voclKernelGetMigrationStatus(kernel);

	if (programMigStatus < contextMigStatus)
	{
		voclUpdateVOCLProgram(program, proxyRank, proxyIndex,
				proxyComm, proxyCommData, context);
		deviceID = voclGetCommandQueueDeviceID(command_queue);
		clBuildProgram(program, 1, &deviceID, voclGetProgramBuildOptions(program), 0, 0);
	}

	programMigStatus = voclProgramGetMigrationStatus(program);
	kernelMigStatus = voclKernelGetMigrationStatus(kernel);
	if (kernelMigStatus < programMigStatus)
	{
		voclUpdateVOCLKernel(kernel, proxyRank, proxyIndex,
				proxyComm, proxyCommData, program);
	}

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
