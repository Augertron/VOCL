#include <stdio.h>
#include <string.h>
#include "voclStructures.h"

extern cl_program
voclMigCreateProgramWithSource(vocl_context context,
                               cl_uint count,
                               const char **strings, const size_t * lengths,
                               cl_int * errcode_ret);
extern char voclContextGetMigrationStatus(vocl_context context);

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
    voclProgramPtr = programPtr;

    return programPtr;
}

static struct strVOCLProgram *getVOCLProgramPtr(vocl_program program)
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
    programPtr = getVOCLProgramPtr(program);
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
    programPtr = getVOCLProgramPtr(program);
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
    programPtr = getVOCLProgramPtr(program);
    return programPtr->buildOptions;
}


void voclStoreProgramContext(vocl_program program, vocl_context context)
{
    struct strVOCLProgram *programPtr;
    programPtr = getVOCLProgramPtr(program);
    programPtr->context = context;
    return;
}

vocl_context voclGetContextFromProgram(vocl_program program)
{
    struct strVOCLProgram *programPtr;
    programPtr = getVOCLProgramPtr(program);
    return programPtr->context;
}

void voclProgramSetMigrationStatus(vocl_program program, char status)
{
	struct strVOCLProgram *programPtr;
	programPtr = getVOCLProgramPtr(program);
	programPtr->migrationStatus = status;
	return;
}

char voclProgramGetMigrationStatus(vocl_program program)
{
	struct strVOCLProgram *programPtr;
	programPtr = getVOCLProgramPtr(program);
	return programPtr->migrationStatus;
}

char *voclGetProgramSource(vocl_program program, size_t * sourceSize)
{
    struct strVOCLProgram *programPtr;
    programPtr = getVOCLProgramPtr(program);

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
    struct strVOCLProgram *programPtr = getVOCLProgramPtr(program);

    *proxyRank = programPtr->proxyRank;
    *proxyIndex = programPtr->proxyIndex;
    *proxyComm = programPtr->proxyComm;
    *proxyCommData = programPtr->proxyCommData;

    return programPtr->clProgram;
}

void voclUpdateVOCLProgram(vocl_program voclProgram, int proxyRank, int proxyIndex,
                           MPI_Comm proxyComm, MPI_Comm proxyCommData, vocl_context context)
{
    struct strVOCLProgram *programPtr = getVOCLProgramPtr(voclProgram);
    char *cSource;
    size_t sourceSize;
    int err;

    /*release previous program */
    clReleaseProgram((cl_program)voclProgram);

    programPtr->proxyRank = proxyRank;
    programPtr->proxyIndex = proxyIndex;
    programPtr->proxyComm = proxyComm;
    programPtr->proxyCommData = proxyCommData;

    cSource = voclGetProgramSource(voclProgram, &sourceSize);

    programPtr->clProgram = voclMigCreateProgramWithSource(context, 1,
                                                           &cSource, &sourceSize, &err);
    if (err != CL_SUCCESS) {
        printf("create program error, %d!\n", err);
    }

	programPtr->migrationStatus = voclContextGetMigrationStatus(context);
    return;
}

int voclReleaseProgram(vocl_program program)
{
    struct strVOCLProgram *programPtr, *preProgramPtr, *curProgramPtr;
    /* the first node in the link list */
    if (program == voclProgramPtr->voclProgram) {
        programPtr = voclProgramPtr;
        voclProgramPtr = voclProgramPtr->next;
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
    free(curProgramPtr);

    return 0;
}
