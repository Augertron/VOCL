#include <stdio.h>
#include <string.h>
#include "voclStructures.h"

static struct strVOCLProgram *voclProgramPtr = NULL;
static vocl_program voclProgram;
static int voclProgramNo;

static vocl_program getVOCLProgramValue()
{
    vocl_program program = voclProgram;
    voclProgram++;

    return program;
}

static struct strVOCLProgram *createVOCLProgram()
{
    struct strVOCLProgram *programPtr;
    programPtr = (struct strVOCLProgram *) malloc(sizeof(struct strVOCLProgram));
    programPtr->next = voclProgramPtr;
    programPtr->voclSourceString = NULL;
    programPtr->sourceSize = 0;
    programPtr->buildOptions = NULL;

	programPtr->kernelNum = 20;
	programPtr->kernelNo = 0;
	programPtr->kernelPtr = (vocl_kernel_str **)malloc(sizeof(vocl_kernel_str *) * programPtr->kernelNum);
	memset(programPtr->kernelPtr, 0, sizeof(vocl_kernel_str *) * programPtr->kernelNum);

	programPtr->migrationStatus = 0;

    voclProgramPtr = programPtr;

    return programPtr;
}

struct strVOCLProgram *voclGetProgramPtr(vocl_program program)
{
    struct strVOCLProgram *programPtr;
    programPtr = voclProgramPtr;
    while (programPtr != NULL) {
        if (programPtr->voclProgram == program) {
            break;
        }
        programPtr = programPtr->next;
    }

    if (programPtr == NULL) {
        printf("Error, program does not exist!\n");
        exit(1);
    }

    return programPtr;
}

void voclAddKernelToProgram(vocl_program program, vocl_kernel_str *kernelPtr)
{
	int i;
	vocl_program_str *programPtr;
	programPtr = voclGetProgramPtr(program);
	
	for (i = 0; i < programPtr->kernelNo; i++)
	{
		if (programPtr->kernelPtr[i] == kernelPtr)
		{
			break;
		}
	}

	if (i == programPtr->kernelNo)
	{
		programPtr->kernelPtr[i] = kernelPtr;
		programPtr->kernelNo++;

		if (programPtr->kernelNo >= programPtr->kernelNum)
		{
			programPtr->kernelPtr = (vocl_kernel_str **)realloc(programPtr->kernelPtr, sizeof(vocl_kernel_str*) * programPtr->kernelNum * 2);
			memset(&programPtr->kernelPtr[programPtr->kernelNum], 0, sizeof(vocl_kernel_str *) * programPtr->kernelNum);
			programPtr->kernelNum *= 2;
		}
	}

	return;
}

void voclRemoveKernelFromProgram(vocl_kernel_str *kernelPtr)
{
    int i, j;
    int kernelFound = 0;
    vocl_program_str *programPtr;
    programPtr = voclProgramPtr;

    while (programPtr != NULL)
    {
        for (i = 0; i < programPtr->kernelNo; i++)
        {
            if (programPtr->kernelPtr[i] == kernelPtr)
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
        printf("voclRemoveKernelFromProgram, vocl_kernel %d does not exist!\n", kernelPtr->voclKernel);
        exit(1);
    }

    return;
}

void voclProgramInitialize()
{
    voclProgramPtr = NULL;
    voclProgramNo = 0;
    voclProgram = 0;
}

void voclProgramFinalize()
{
    struct strVOCLProgram *programPtr, *tmpprogramPtr;
    programPtr = voclProgramPtr;
    while (programPtr != NULL) {
        tmpprogramPtr = programPtr->next;
        if (programPtr->voclSourceString != NULL) {
            free(programPtr->voclSourceString);
        }
        if (programPtr->buildOptions != NULL) {
            free(programPtr->buildOptions);
        }

		if (programPtr->kernelPtr != NULL)
		{
			free(programPtr->kernelPtr);
		}

        free(programPtr);
        programPtr = tmpprogramPtr;
    }

    voclProgramPtr = NULL;
    voclProgramNo = 0;
    voclProgram = 0;
}

void voclStoreProgramSource(vocl_program program, char *source, size_t sourceSize)
{
    struct strVOCLProgram *programPtr;
    programPtr = voclGetProgramPtr(program);
    if (programPtr->voclSourceString != NULL) {
        free(programPtr->voclSourceString);
    }
    programPtr->voclSourceString = (char *) malloc(sizeof(char) * (sourceSize + 1));
    memcpy(programPtr->voclSourceString, source, sizeof(char) * sourceSize);
    /* null terminated string */
    programPtr->voclSourceString[sourceSize] = '\0';
    programPtr->sourceSize = sourceSize;
    return;
}

void voclStoreProgramBuildOptions(vocl_program program, char *buildOptions)
{
    struct strVOCLProgram *programPtr;
    int optionLen;
    programPtr = voclGetProgramPtr(program);
    optionLen = strlen(buildOptions);
    if (optionLen > 0) {
        if (programPtr->buildOptions != NULL) {
            free(programPtr->buildOptions);
        }
        programPtr->buildOptions = (char *) malloc(sizeof(char) * (optionLen + 1));
        memcpy(programPtr->buildOptions, buildOptions, sizeof(char) * optionLen);
        /* null terminated string */
        programPtr->buildOptions[optionLen] = '\0';
    }

    return;
}

char *voclGetProgramBuildOptions(vocl_program program)
{
    struct strVOCLProgram *programPtr;
    programPtr = voclGetProgramPtr(program);
    return programPtr->buildOptions;
}


void voclStoreProgramContext(vocl_program program, vocl_context context)
{
    struct strVOCLProgram *programPtr;
    programPtr = voclGetProgramPtr(program);
    programPtr->context = context;
    return;
}

vocl_context voclGetContextFromProgram(vocl_program program)
{
    struct strVOCLProgram *programPtr;
    programPtr = voclGetProgramPtr(program);
    return programPtr->context;
}

void voclProgramSetMigrationStatus(vocl_program program, char status)
{
	struct strVOCLProgram *programPtr;
	programPtr = voclGetProgramPtr(program);
	programPtr->migrationStatus = status;
	return;
}

char voclProgramGetMigrationStatus(vocl_program program)
{
	struct strVOCLProgram *programPtr;
	programPtr = voclGetProgramPtr(program);
	return programPtr->migrationStatus;
}

char *voclGetProgramSource(vocl_program program, size_t * sourceSize)
{
    struct strVOCLProgram *programPtr;
    programPtr = voclGetProgramPtr(program);

    *sourceSize = programPtr->sourceSize;
    return programPtr->voclSourceString;
}

vocl_program voclCLProgram2VOCLProgram(cl_program program, int proxyRank,
                                       int proxyIndex, MPI_Comm proxyComm,
                                       MPI_Comm proxyCommData)
{
    struct strVOCLProgram *programPtr = createVOCLProgram();
    programPtr->clProgram = program;
    programPtr->proxyRank = proxyRank;
    programPtr->proxyIndex = proxyIndex;
    programPtr->proxyComm = proxyComm;
    programPtr->proxyCommData = proxyCommData;
    programPtr->voclProgram = getVOCLProgramValue();

    return programPtr->voclProgram;
}

cl_program voclVOCLProgram2CLProgramComm(vocl_program program, int *proxyRank,
                                         int *proxyIndex, MPI_Comm * proxyComm,
                                         MPI_Comm * proxyCommData)
{
    /* the vocl program value indicates its location */
    /* in the event buffer */
    struct strVOCLProgram *programPtr = voclGetProgramPtr(program);

    *proxyRank = programPtr->proxyRank;
    *proxyIndex = programPtr->proxyIndex;
    *proxyComm = programPtr->proxyComm;
    *proxyCommData = programPtr->proxyCommData;

    return programPtr->clProgram;
}

void voclUpdateVOCLProgram(vocl_program voclProgram, cl_program newProgram, int proxyRank, int proxyIndex,
                           MPI_Comm proxyComm, MPI_Comm proxyCommData)
{
    struct strVOCLProgram *programPtr = voclGetProgramPtr(voclProgram);

    programPtr->proxyRank = proxyRank;
    programPtr->proxyIndex = proxyIndex;
    programPtr->proxyComm = proxyComm;
    programPtr->proxyCommData = proxyCommData;

    programPtr->clProgram = newProgram;

    return;
}

int voclReleaseProgram(vocl_program program)
{
    struct strVOCLProgram *programPtr, *preProgramPtr, *curProgramPtr;
    /* the first node in the link list */
    if (program == voclProgramPtr->voclProgram) {
        programPtr = voclProgramPtr;
        voclProgramPtr = voclProgramPtr->next;

		if (programPtr->voclSourceString)
		{
			free(programPtr->voclSourceString);
		}

		if (programPtr->buildOptions)
		{
			free(programPtr->buildOptions);
		}

		if (programPtr->kernelPtr)
		{
			free(programPtr->kernelPtr);
		}

        free(programPtr);

        return 0;
    }

    programPtr = NULL;
    preProgramPtr = voclProgramPtr;
    curProgramPtr = voclProgramPtr->next;
    while (curProgramPtr != NULL) {
        if (program == curProgramPtr->voclProgram) {
            programPtr = curProgramPtr;
            break;
        }
        preProgramPtr = curProgramPtr;
        curProgramPtr = curProgramPtr->next;
    }

    if (programPtr == NULL) {
        printf("program does not exist!\n");
        exit(1);
    }

    /* remote the current node from link list */
    preProgramPtr->next = curProgramPtr->next;
	if (curProgramPtr->voclSourceString)
	{
		free(curProgramPtr->voclSourceString);
	}

	if (curProgramPtr->buildOptions)
	{
		free(curProgramPtr->buildOptions);
	}

	if (curProgramPtr->kernelPtr)
	{
		free(curProgramPtr->kernelPtr);
	}

    free(curProgramPtr);

    return 0;
}
