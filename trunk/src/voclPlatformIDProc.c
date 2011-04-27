#include "vocl_structures.h"

static struct strVOCLPlatformID *voclPlatformIDPtr = NULL;
static vocl_platform_id voclPlatformID;
static int voclPlatformIDNum;
static int voclPlatformIDNo;

static vocl_platform_id getVOCLPlatformIDValue()
{
    vocl_platform_id platform_id = voclPlatformID;
	voclPlatformID++;

    return platform_id;
}

static struct strVOCLPlatformID *getVOCLPlatformIDPtr()
{
    if (voclPlatformIDNo >= voclPlatformIDNum) {
        voclPlatformIDNum *= 2;
        voclPlatformIDPtr = (struct strVOCLPlatformID *) realloc(voclPlatformIDPtr,
                                                   voclPlatformIDNum *
                                                   sizeof(struct strVOCLPlatformID));
    }
    return &voclPlatformIDPtr[voclPlatformIDNo++];
}


void voclPlatformIDInitialize()
{
    voclPlatformIDNum = VOCL_PLATFORM_ID_NUM;
    voclPlatformIDPtr =
        (struct strVOCLPlatformID *) malloc(voclPlatformIDNum * sizeof(struct strVOCLPlatformID));
    voclPlatformIDNo = 0;
    voclPlatformID = 0;
}

void voclPlatformIDFinalize()
{
    if (voclPlatformIDPtr != NULL) {
        free(voclPlatformIDPtr);
        voclPlatformIDPtr = NULL;
    }
    voclPlatformIDNo = 0;
    voclPlatformID = 0;
    voclPlatformIDNum = 0;
}

vocl_platform_id voclCLPlatformID2VOCLPlatformID(cl_platform_id platformID, int proxyID)
{
    struct strVOCLPlatformID *platformIDPtr = getVOCLPlatformIDPtr();
    platformIDPtr->clPlatformID = platformID;
	platformIDPtr->proxyID = proxyID;
    platformIDPtr->voclPlatformID = getVOCLPlatformIDValue();

    return platformIDPtr->voclPlatformID;
}

cl_platform_id voclVOCLPlatformID2CLPlatformIDComm(vocl_platform_id platformID, int *proxyID)
/*proxyID indicates the proxy process */
/*that the platformID corresponds to. It is the output of this function */
{
    /* the vocl platformID value indicates its location */
    /* in the platformID buffer */
    int platformIDNo = (int) platformID;

	*proxyID = voclPlatformIDPtr[platformIDNo].proxyID;

    return voclPlatformIDPtr[platformIDNo].clPlatformID;
}

