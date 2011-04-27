#include "vocl_structures.h"

static struct strVOCLContext *voclContextPtr = NULL;
static vocl_context voclContext;
static int voclContextNum;
static int voclContextNo;

static vocl_context getVOCLContextValue()
{
    vocl_context context = voclContext;
	voclContext++;

    return context;
}

static struct strVOCLContext *getVOCLContextPtr()
{
    if (voclContextNo >= voclContextNum) {
        voclContextNum *= 2;
        voclContextPtr = (struct strVOCLContext *) realloc(voclContextPtr,
                                                   voclContextNum *
                                                   sizeof(struct strVOCLContext));
    }
    return &voclContextPtr[voclContextNo++];
}


void voclContextInitialize()
{
    voclContextNum = VOCL_CONTEXT_NUM;
    voclContextPtr =
        (struct strVOCLContext *) malloc(voclContextNum * sizeof(struct strVOCLContext));
    voclContextNo = 0;
    voclContext = 0;
}

void voclContextFinalize()
{
    if (voclContextPtr != NULL) {
        free(voclContextPtr);
        voclContextPtr = NULL;
    }
    voclContextNo = 0;
    voclContext = 0;
    voclContextNum = 0;
}

vocl_context voclCLContext2VOCLContext(cl_context context, int proxyID)
{
    struct strVOCLContext *contextPtr = getVOCLContextPtr();
    contextPtr->clContext = context;
	contextPtr->proxyID = proxyID;
    contextPtr->voclContext = getVOCLContextValue();

    return contextPtr->voclContext;
}

cl_context voclVOCLContext2CLContextComm(vocl_context context, int *proxyID)
{
    /* the vocl event value indicates its location */
    /* in the event buffer */
    int contextNo = (int) context;
	*proxyID = voclContextPtr[contextNo].proxyID;

    return voclContextPtr[contextNo].clContext;
}

