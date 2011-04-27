#include <string.h>
#include "vocl_structures.h"


static struct strVOCLProgram *voclProgramPtr = NULL;
static vocl_program voclProgram;
static int voclProgramNum;
static int voclProgramNo;

static vocl_program getVOCLProgramValue()
{
    vocl_program program = voclProgram;
	voclProgram++;

    return program;
}

static struct strVOCLProgram *getVOCLProgramPtr()
{
    if (voclProgramNo >= voclProgramNum) {
        voclProgramNum *= 2;
        voclProgramPtr = (struct strVOCLProgram *) realloc(voclProgramPtr,
                                                   voclProgramNum * sizeof(struct strVOCLProgram));
    }
    return &voclProgramPtr[voclProgramNo++];
}


void voclProgramInitialize()
{
	int i;
    voclProgramNum = VOCL_PROGRAM_NUM;
    voclProgramPtr =
        (struct strVOCLProgram *) malloc(voclProgramNum * sizeof(struct strVOCLProgram));
    voclProgramNo = 0;
    voclProgram = 0;
	for (i = 0; i < voclProgramNum; i++)
	{
		voclProgramPtr[i].voclSourceString = NULL;
	}
}

void voclProgramFinalize()
{
	int i;
	for (i = 0; i < voclProgramNum; i++)
	{
		if (voclProgramPtr[i].voclSourceString != NULL)
		{
			free(voclProgramPtr[i].voclSourceString);
			voclProgramPtr[i].voclSourceString = NULL;
			voclProgramPtr[i].sourceSize = 0;
		}
	}

    if (voclProgramPtr != NULL) {
        free(voclProgramPtr);
        voclProgramPtr = NULL;
    }
    voclProgramNo = 0;
    voclProgram = 0;
    voclProgramNum = 0;
}

void voclStoreProgramSource(char *source, size_t sourceSize)
{
	int programIndex = voclProgramNo - 1;
	if (voclProgramPtr[programIndex].voclSourceString != NULL)
	{
		free(voclProgramPtr[programIndex].voclSourceString);
	}
	voclProgramPtr[programIndex].voclSourceString = (char *)malloc(sizeof(char) * (sourceSize+1));
	memcpy(voclProgramPtr[programIndex].voclSourceString, source, sizeof(char) * sourceSize);
	/* null terminated string */
	voclProgramPtr[programIndex].voclSourceString[sourceSize] = '\0';
	voclProgramPtr[programIndex].sourceSize = sourceSize;
	return;
}

char *voclGetProgramSource(vocl_program program, size_t *sourceSize)
{
	int programNo = (int)program;
	*sourceSize = voclProgramPtr[programNo].sourceSize;
	return voclProgramPtr[programNo].voclSourceString;
}

vocl_program voclCLProgram2VOCLProgram(cl_program program, int proxyID)
{
    struct strVOCLProgram *programPtr = getVOCLProgramPtr();
    programPtr->clProgram = program;
	programPtr->proxyID = proxyID;
    programPtr->voclProgram = getVOCLProgramValue();

    return programPtr->voclProgram;
}

cl_program voclVOCLProgram2CLProgramComm(vocl_program program, int *proxyID)
{
    /* the vocl program value indicates its location */
    /* in the event buffer */
    int programNo = (int) program;

	*proxyID = voclProgramPtr[programNo].proxyID;

    return voclProgramPtr[programNo].clProgram;
}

