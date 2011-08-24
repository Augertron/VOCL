#include <stdio.h>
#include <string.h>
#include "vocl_proxyStructures.h"

static str_vocl_proxy_program *voclProxyProgramPtr = NULL;

void voclProxyAddProgram(cl_program program, char *sourceString, size_t sourceSize, int stringNum, size_t *stringSizeArray, cl_context context)
{
    str_vocl_proxy_program *programPtr;
    programPtr = (str_vocl_proxy_program *)malloc(sizeof(str_vocl_proxy_program));
	programPtr->program = program;
	programPtr->sourceSize = sourceSize;
	programPtr->context = context;
	programPtr->sourceString = (char *)malloc(sourceSize);
	memcpy(programPtr->sourceString, sourceString, sourceSize);

	programPtr->buildOptions = NULL;
	programPtr->deviceNum = 0;
	programPtr->device_list = NULL;

	programPtr->stringNum = stringNum;
	programPtr->stringSizeArray = (size_t *)malloc(stringNum * sizeof(size_t));
	memcpy(programPtr->stringSizeArray, stringSizeArray, stringNum);

    programPtr->next = voclProxyProgramPtr;
    voclProxyProgramPtr = programPtr;

    return;
}

str_vocl_proxy_program *voclProxyGetProgramPtr(cl_program program)
{
    str_vocl_proxy_program *programPtr;
    programPtr = voclProxyProgramPtr;
    while (programPtr != NULL)
    {
        if (programPtr->program == program)
        {
            break;
        }
    }

    if (programPtr == NULL)
    {
        printf("voclProxyGetProgramPtr, program %p does not exist!\n", program);
        exit (1);
    }

    return programPtr;
}

void voclProxySetProgramBuildOptions(cl_program program, cl_uint deviceNum, cl_device_id *device_list, char *buildOptions)
{
	int buildOptionLen;
	str_vocl_proxy_program *programPtr;
	programPtr = voclProxyProgramPtr;

	if (buildOptions != NULL)
	{
		buildOptionLen = strlen(buildOptions);
		programPtr->buildOptions = (char *)malloc(buildOptionLen+1);
		memcpy(programPtr->buildOptions, buildOptions, buildOptionLen);
	}

	if (deviceNum > 0)
	{
		programPtr->deviceNum = deviceNum;
		programPtr->device_list = (cl_device_id *)malloc(sizeof(cl_device_id) * deviceNum);
		memcpy(programPtr->device_list, device_list, sizeof(cl_device_id) * deviceNum);
	}
	
	return;
}

void voclProxyReleaseProgram(cl_program program)
{
    str_vocl_proxy_program *programPtr, *preProgramPtr;

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
    free(programPtr);

    return;
}

void voclProxyReleaseAllPrograms()
{
    str_vocl_proxy_program *programPtr, *nextProgramPtr;

    programPtr = voclProxyProgramPtr;
    while (programPtr != NULL)
    {
        nextProgramPtr = programPtr->next;
		free(programPtr->sourceString);
		free(programPtr->stringSizeArray);
		free(programPtr->buildOptions);
		free(programPtr->device_list);
        free(programPtr);
        programPtr = nextProgramPtr;
    }

    voclProxyProgramPtr = NULL;

    return;
}

