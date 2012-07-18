#include <stdio.h>
#include "voclStructures.h"

static struct strVOCLPlatformID *voclPlatformIDPtr = NULL;
static vocl_platform_id voclPlatformID;
static int voclPlatformIDNo;

static vocl_platform_id getVOCLPlatformIDValue()
{
    vocl_platform_id platform = voclPlatformID;
    voclPlatformID++;

    return platform;
}

static struct strVOCLPlatformID *createVOCLPlatformID()
{
    struct strVOCLPlatformID *platformPtr;
    platformPtr = (struct strVOCLPlatformID *) malloc(sizeof(struct strVOCLPlatformID));
    platformPtr->next = voclPlatformIDPtr;
    voclPlatformIDPtr = platformPtr;

    return platformPtr;
}

static struct strVOCLPlatformID *getVOCLPlatformIDPtr(vocl_platform_id platform)
{
    struct strVOCLPlatformID *platformPtr;
    platformPtr = voclPlatformIDPtr;
    while (platformPtr != NULL) {
        if (platformPtr->voclPlatformID == platform) {
            break;
        }
        platformPtr = platformPtr->next;
    }

    if (platformPtr == NULL) {
        printf("Error, platform does not exist!\n");
        exit(1);
    }

    return platformPtr;
}

void voclPlatformIDInitialize()
{
    voclPlatformIDPtr = NULL;
    voclPlatformIDNo = 0;
    voclPlatformID = 0;
}

void voclPlatformIDFinalize()
{
    struct strVOCLPlatformID *platformPtr, *tmpplatformPtr;
    platformPtr = voclPlatformIDPtr;
    while (platformPtr != NULL) {
        tmpplatformPtr = platformPtr->next;
        free(platformPtr);
        platformPtr = tmpplatformPtr;
    }

    voclPlatformIDPtr = NULL;
    voclPlatformIDNo = 0;
    voclPlatformID = 0;
}

vocl_platform_id voclCLPlatformID2VOCLPlatformID(cl_platform_id platform, int proxyRank,
                                                 int proxyIndex, MPI_Comm proxyComm,
                                                 MPI_Comm proxyCommData)
{
    struct strVOCLPlatformID *platformPtr = createVOCLPlatformID();
    platformPtr->clPlatformID = platform;
    platformPtr->proxyRank = proxyRank;
    platformPtr->proxyIndex = proxyIndex;
    platformPtr->proxyComm = proxyComm;
    platformPtr->proxyCommData = proxyCommData;
    platformPtr->voclPlatformID = getVOCLPlatformIDValue();

    return platformPtr->voclPlatformID;
}

cl_platform_id voclVOCLPlatformID2CLPlatformIDComm(vocl_platform_id platform, int *proxyRank,
                                                   int *proxyIndex, MPI_Comm * proxyComm,
                                                   MPI_Comm * proxyCommData)
{
    struct strVOCLPlatformID *platformPtr = getVOCLPlatformIDPtr(platform);
    *proxyRank = platformPtr->proxyRank;
    *proxyIndex = platformPtr->proxyIndex;
    *proxyComm = platformPtr->proxyComm;
    *proxyCommData = platformPtr->proxyCommData;

    return platformPtr->clPlatformID;
}

void voclUpdateVOCLPlatformID(vocl_platform_id voclPlatform, int proxyRank,
                              int proxyIndex, MPI_Comm proxyComm, MPI_Comm proxyCommData,
                              cl_platform_id clPlatform)
{
    struct strVOCLPlatformID *platformPtr = getVOCLPlatformIDPtr(voclPlatform);
    platformPtr->clPlatformID = clPlatform;
    platformPtr->proxyRank = proxyRank;
    platformPtr->proxyIndex = proxyIndex;
    platformPtr->proxyComm = proxyComm;
    platformPtr->proxyCommData = proxyCommData;

    return;
}
