#include <stdio.h>
#include <string.h>
#include "vocl_proxyStructures.h"

static vocl_proxy_program *voclProxyProgramPtr = NULL;

void voclProxyAddProgram(cl_program program, char *sourceString, size_t sourceSize, int stringNum, size_t *stringSizeArray, cl_context context)
{
    vocl_proxy_program *programPtr;
    programPtr = (vocl_proxy_program *)malloc(sizeof(vocl_proxy_program));
	programPtr->program = program;
	programPtr->oldProgram = NULL;
	programPtr->sourceSize = sourceSize;
	programPtr->context = context;
	programPtr->sourceString = (char *)malloc(sourceSize);
	memcpy(programPtr->sourceString, sourceString, sourceSize);

	programPtr->buildOptionLen = 0;
	programPtr->buildOptions = NULL;
	programPtr->deviceNum = 0;
	programPtr->device_list = NULL;

	programPtr->stringNum = stringNum;
	programPtr->stringSizeArray = (size_t *)malloc(stringNum * sizeof(size_t));
	memcpy(programPtr->stringSizeArray, stringSizeArray, stringNum * sizeof(size_t));

	programPtr->kernelNum = 20;
	programPtr->kernelNo = 0;
	programPtr->kernelPtr = (vocl_proxy_kernel **)malloc(sizeof(vocl_proxy_kernel *) * programPtr->kernelNum);

	programPtr->migStatus = 0;

    programPtr->next = voclProxyProgramPtr;
    voclProxyProgramPtr = programPtr;

    return;
}

vocl_proxy_program *voclProxyGetProgramPtr(cl_program program)
{
    vocl_proxy_program *programPtr;
    programPtr = voclProxyProgramPtr;
    while (programPtr != NULL)
    {
        if (programPtr->program == program)
        {
            break;
        }

		programPtr = programPtr->next;
    }

    if (programPtr == NULL)
    {
        printf("voclProxyGetProgramPtr, program %p does not exist!\n", program);
        exit (1);
    }

    return programPtr;
}

void voclProxySetProgramMigStatus(cl_program program, char migStatus)
{
	vocl_proxy_program *programPtr;
	programPtr = voclProxyGetProgramPtr(program);
	programPtr->migStatus = migStatus;

	return;
}

char voclProxyGetProgramMigStatus(cl_program program)
{
	vocl_proxy_program *programPtr;
	programPtr = voclProxyGetProgramPtr(program);
	return programPtr->migStatus;
}

void voclProxyStoreOldProgramValue(cl_program program, cl_program oldProgram)
{
    vocl_proxy_program *programPtr;
    programPtr = voclProxyGetProgramPtr(program);
    programPtr->oldProgram = oldProgram;

    return;
}

cl_program voclProxyGetOldProgramValue(cl_program program)
{
    vocl_proxy_program *programPtr;
    programPtr = voclProxyGetProgramPtr(program);
    return programPtr->oldProgram;
}

cl_program voclProxyGetNewProgramValue(cl_program oldProgram)
{
    vocl_proxy_program *programPtr;

	programPtr = voclProxyProgramPtr;
	while (programPtr != NULL)
	{
		if (programPtr->oldProgram == oldProgram)
		{
			break;
		}
		programPtr = programPtr->next;
	}

	if (programPtr == NULL)
	{
		printf("voclProxyGetNewProgramValue, old program %p does not exist!\n",
				oldProgram);
		exit (1);
	}

    return programPtr->program;
}

void voclProxyAddKernelToProgram(cl_program program, vocl_proxy_kernel *kernel)
{
	int i;
	vocl_proxy_program *programPtr;
	programPtr = voclProxyGetProgramPtr(program);
	for (i = 0; i < programPtr->kernelNo; i++)
	{
		if (programPtr->kernelPtr[i] == kernel)
		{
			break;
		}
	}

	if (i == programPtr->kernelNo)
	{	
		programPtr->kernelPtr[programPtr->kernelNo] = kernel;
		programPtr->kernelNo++;
		if (programPtr->kernelNo >= programPtr->kernelNum)
		{
			programPtr->kernelPtr = (vocl_proxy_kernel **)realloc(programPtr->kernelPtr, sizeof(vocl_proxy_kernel *) * programPtr->kernelNum * 2);
			memset(&programPtr->kernelPtr[programPtr->kernelNum], 0, sizeof(vocl_proxy_kernel *) * programPtr->kernelNum);
			programPtr->kernelNum *= 2;
		}
	}

	return;
}

void voclProxyRemoveKernelFromProgram(vocl_proxy_kernel *kernel)
{
	int i, j;
	int kernelFound = 0;
	vocl_proxy_program *programPtr;
	programPtr = voclProxyProgramPtr;
	
	while (programPtr != NULL)
	{
		for (i = 0; i < programPtr->kernelNo; i++)
		{
			if (programPtr->kernelPtr[i] == kernel)
			{
				kernelFound = 1;
				break;
			}
		}

		if (i < programPtr->kernelNo)
		{
			for (j = i; j < programPtr->kernelNo - 1; j++)
			{
				programPtr->kernelPtr[j] = programPtr->kernelPtr[j+1];
			}
			programPtr->kernelNo--;
		}

		programPtr = programPtr->next;
	}

	if (kernelFound == 0)
	{
        printf("voclProxyRemoveKernelFromProgram, cl_kernel %p does not exist!\n", kernel->kernel);
        exit(1);
	}

	return;
}

void voclProxyRemoveKernelFromProgramSimple(cl_program program, vocl_proxy_kernel *kernel)
{
	int i, j;
	vocl_proxy_program *programPtr;
	programPtr = voclProxyGetProgramPtr(program);
	for (i = 0; i < programPtr->kernelNo; i++)
	{
		if (programPtr->kernelPtr[i] == kernel)
		{
			break;
		}
	}

	if (i == programPtr->kernelNo)
	{
        printf("voclProxyRemoveKernelFromProgram, cl_kernel %p does not exist!\n", kernel->kernel);
        exit(1);
	}
	else
	{
		for (j = i; j < programPtr->kernelNo - 1; j++)
		{
			programPtr->kernelPtr[j] = programPtr->kernelPtr[j+1];
		}
		programPtr->kernelNo--;
	}

	return;
}

void voclProxySetProgramBuildOptions(cl_program program, cl_uint deviceNum, cl_device_id *device_list, char *buildOptions)
{
	vocl_proxy_program *programPtr;
	programPtr = voclProxyGetProgramPtr(program);

	if (buildOptions != NULL)
	{
		programPtr->buildOptionLen = strlen(buildOptions)+1;
		programPtr->buildOptions = (char *)malloc(programPtr->buildOptionLen);
		memcpy(programPtr->buildOptions, buildOptions, programPtr->buildOptionLen);
		programPtr->buildOptions[programPtr->buildOptionLen-1] = '\0';
	}

	if (deviceNum > 0)
	{
		programPtr->deviceNum = deviceNum;
		programPtr->device_list = (cl_device_id *)malloc(sizeof(cl_device_id) * deviceNum);
		memcpy(programPtr->device_list, device_list, sizeof(cl_device_id) * deviceNum);
	}
	
	return;
}

char* voclProxyGetProgramBuildOptions(cl_program program)
{
	int buildOptionLen;
	vocl_proxy_program *programPtr;
	programPtr = voclProxyGetProgramPtr(program);

	return programPtr->buildOptions;
}

cl_device_id* voclProxyGetProgramDevices(cl_program program, cl_uint *deviceNum)
{
	int buildOptionLen;
	vocl_proxy_program *programPtr;
	programPtr = voclProxyGetProgramPtr(program);

	*deviceNum = programPtr->deviceNum;

	return programPtr->device_list;
}

void voclProxyReleaseProgram(cl_program program)
{
    vocl_proxy_program *programPtr, *preProgramPtr;

    /* if the cmdQueue is in the first node */
	programPtr = voclProxyProgramPtr;

	if (programPtr != NULL)
	{
		if (programPtr->program == program)
		{
			programPtr = voclProxyProgramPtr;
			voclProxyProgramPtr = programPtr->next;
			free(programPtr->sourceString);
			free(programPtr->stringSizeArray);
			free(programPtr->buildOptions);
			free(programPtr->device_list);
			free(programPtr->kernelPtr);
			free(programPtr);
			return;
		}

		preProgramPtr = voclProxyProgramPtr;
		programPtr = preProgramPtr->next;
		while (programPtr != NULL)
		{
			if (programPtr->program == program)
			{
				break;
			}

			preProgramPtr = programPtr;
			programPtr = programPtr->next;
		}
	}

    if (programPtr == NULL)
    {
        printf("voclProxyReleaseProgram, Program %p does not exist!\n", program);
        exit (1);
    }

    preProgramPtr->next = programPtr->next;
	free(programPtr->sourceString);
	free(programPtr->stringSizeArray);
	free(programPtr->buildOptions);
	free(programPtr->device_list);
	free(programPtr->kernelPtr);
    free(programPtr);

    return;
}

void voclProxyReleaseAllPrograms()
{
    vocl_proxy_program *programPtr, *nextProgramPtr;

    programPtr = voclProxyProgramPtr;
    while (programPtr != NULL)
    {
        nextProgramPtr = programPtr->next;
		free(programPtr->sourceString);
		free(programPtr->stringSizeArray);
		free(programPtr->buildOptions);
		free(programPtr->device_list);
		free(programPtr->kernelPtr);
        free(programPtr);
        programPtr = nextProgramPtr;
    }

    voclProxyProgramPtr = NULL;

    return;
}


