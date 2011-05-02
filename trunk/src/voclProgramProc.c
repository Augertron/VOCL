#include <stdio.h>
#include <string.h>
#include "vocl_structures.h"


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
	programPtr = (struct strVOCLProgram *)malloc(sizeof(struct strVOCLProgram));
	programPtr->next = voclProgramPtr;
	programPtr->voclSourceString = NULL;
	programPtr->sourceSize = 0;
	voclProgramPtr = programPtr;

	return programPtr;
}

static struct strVOCLProgram *getVOCLProgramPtr(vocl_program program)
{
	struct strVOCLProgram *programPtr;
	programPtr = voclProgramPtr;
	while (programPtr != NULL)
	{
		if (programPtr->voclProgram == program)
		{
			break;
		}
		programPtr = programPtr->next;
	}

	if (programPtr == NULL)
	{
		printf("Error, program does not exist!\n");
		exit (1);
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
	while (programPtr != NULL)
	{
		tmpprogramPtr = programPtr->next;
		if (programPtr->voclSourceString != NULL)
		{
			free(programPtr->voclSourceString);
		}
		free(programPtr);
		programPtr = tmpprogramPtr;
	}

    voclProgramPtr = NULL;
    voclProgramNo = 0;
    voclProgram = 0;
}

void voclStoreProgramSource(char *source, size_t sourceSize)
{
	struct strVOCLProgram *programPtr;
	programPtr = voclProgramPtr;
	if (programPtr->voclSourceString != NULL)
	{
		free(programPtr->voclSourceString);
	}
	programPtr->voclSourceString = (char *)malloc(sizeof(char) * (sourceSize+1));
	memcpy(programPtr->voclSourceString, source, sizeof(char) * sourceSize);
	/* null terminated string */
	programPtr->voclSourceString[sourceSize] = '\0';
	programPtr->sourceSize = sourceSize;
	return;
}

char *voclGetProgramSource(vocl_program program, size_t *sourceSize)
{
	struct strVOCLProgram *programPtr;
	programPtr = getVOCLProgramPtr(program);

	*sourceSize = programPtr->sourceSize;
	return programPtr->voclSourceString;
}

vocl_program voclCLProgram2VOCLProgram(cl_program program, int proxyID,
                 int proxyIndex, MPI_Comm proxyComm, MPI_Comm proxyCommData)
{
    struct strVOCLProgram *programPtr = createVOCLProgram();
    programPtr->clProgram = program;
	programPtr->proxyID = proxyID;
	programPtr->proxyIndex = proxyIndex;
	programPtr->proxyComm = proxyComm;
	programPtr->proxyCommData = proxyCommData;
    programPtr->voclProgram = getVOCLProgramValue();

    return programPtr->voclProgram;
}

cl_program voclVOCLProgram2CLProgramComm(vocl_program program, int *proxyID,
               int *proxyIndex, MPI_Comm *proxyComm, MPI_Comm *proxyCommData)
{
    /* the vocl program value indicates its location */
    /* in the event buffer */
	struct strVOCLProgram *programPtr = getVOCLProgramPtr(program);

	*proxyID = programPtr->proxyID;
	*proxyIndex = programPtr->proxyIndex;
	*proxyComm = programPtr->proxyComm;
	*proxyCommData = programPtr->proxyCommData;

    return programPtr->clProgram;
}

int voclReleaseProgram(vocl_program program)
{
	struct strVOCLProgram *programPtr, *preProgramPtr, *curProgramPtr;
	/* the first node in the link list */
	if (program == voclProgramPtr->voclProgram)
	{
		programPtr = voclProgramPtr;
		voclProgramPtr = voclProgramPtr->next;
		free(programPtr);

		return 0;
	}

	programPtr = NULL;
	preProgramPtr = voclProgramPtr;
	curProgramPtr = voclProgramPtr->next;
	while (curProgramPtr != NULL)
	{
		if (program == curProgramPtr->voclProgram)
		{
			programPtr = curProgramPtr;
			break;
		}
		preProgramPtr = curProgramPtr;
		curProgramPtr = curProgramPtr->next;
	}

	if (programPtr == NULL)
	{
		printf("program does not exist!\n");
		exit (1);
	}

	/* remote the current node from link list */
	preProgramPtr->next = curProgramPtr->next;
	free(curProgramPtr);
	
	return 0;
}
