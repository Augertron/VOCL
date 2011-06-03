#include <stdio.h>
#include "voclStructures.h"

extern cl_sampler
voclMigCreateSampler(cl_context context,
                     cl_bool normalized_coords,
                     cl_addressing_mode addressing_mode,
                     cl_filter_mode filter_mode, cl_int * errcode_ret);


static struct strVOCLSampler *voclSamplerPtr = NULL;
static vocl_sampler voclSampler;
static int voclSamplerNo;

static vocl_sampler getVOCLSamplerValue()
{
    vocl_sampler sampler = voclSampler;
    voclSampler++;

    return sampler;
}

static struct strVOCLSampler *createVOCLSampler()
{
    struct strVOCLSampler *samplerPtr;
    samplerPtr = (struct strVOCLSampler *) malloc(sizeof(struct strVOCLSampler));
    samplerPtr->next = voclSamplerPtr;
    voclSamplerPtr = samplerPtr;

    return samplerPtr;
}

static struct strVOCLSampler *getVOCLSamplerPtr(vocl_sampler sampler)
{
    struct strVOCLSampler *samplerPtr;
    samplerPtr = voclSamplerPtr;
    while (samplerPtr != NULL) {
        if (samplerPtr->voclSampler == sampler) {
            break;
        }
        samplerPtr = samplerPtr->next;
    }

    if (samplerPtr == NULL) {
        printf("Error, sampler does not exist!\n");
        exit(1);
    }

    return samplerPtr;
}

void voclSamplerInitialize()
{
    voclSamplerPtr = NULL;
    voclSamplerNo = 0;
    voclSampler = 0;
}

void voclSamplerFinalize()
{
    struct strVOCLSampler *samplerPtr, *tmpsamplerPtr;
    samplerPtr = voclSamplerPtr;
    while (samplerPtr != NULL) {
        tmpsamplerPtr = samplerPtr->next;
        free(samplerPtr);
        samplerPtr = tmpsamplerPtr;
    }

    voclSamplerPtr = NULL;
    voclSamplerNo = 0;
    voclSampler = 0;
}

vocl_sampler voclCLSampler2VOCLSampler(cl_sampler sampler, int proxyRank,
                                       int proxyIndex, MPI_Comm proxyComm,
                                       MPI_Comm proxyCommData)
{
    struct strVOCLSampler *samplerPtr = createVOCLSampler();
    samplerPtr->clSampler = sampler;
    samplerPtr->proxyRank = proxyRank;
    samplerPtr->proxyIndex = proxyIndex;
    samplerPtr->proxyComm = proxyComm;
    samplerPtr->proxyCommData = proxyCommData;
    samplerPtr->voclSampler = getVOCLSamplerValue();

    return samplerPtr->voclSampler;
}

cl_sampler voclVOCLSampler2CLSamplerComm(vocl_sampler sampler, int *proxyRank,
                                         int *proxyIndex, MPI_Comm * proxyComm,
                                         MPI_Comm * proxyCommData)
{
    struct strVOCLSampler *samplerPtr = getVOCLSamplerPtr(sampler);
    *proxyRank = samplerPtr->proxyRank;
    *proxyIndex = samplerPtr->proxyIndex;
    *proxyComm = samplerPtr->proxyComm;
    *proxyCommData = samplerPtr->proxyCommData;

    return samplerPtr->clSampler;
}

int voclReleaseSampler(vocl_sampler sampler)
{
    struct strVOCLSampler *samplerPtr, *preSamplerPtr, *curSamplerPtr;
    /* the first node in the link list */
    if (sampler == voclSamplerPtr->voclSampler) {
        samplerPtr = voclSamplerPtr;
        voclSamplerPtr = voclSamplerPtr->next;
        free(samplerPtr);

        return 0;
    }

    samplerPtr = NULL;
    preSamplerPtr = voclSamplerPtr;
    curSamplerPtr = voclSamplerPtr->next;
    while (curSamplerPtr != NULL) {
        if (sampler == curSamplerPtr->voclSampler) {
            samplerPtr = curSamplerPtr;
            break;
        }
        preSamplerPtr = curSamplerPtr;
        curSamplerPtr = curSamplerPtr->next;
    }

    if (samplerPtr == NULL) {
        printf("sampler does not exist!\n");
        exit(1);
    }

    /* remote the current node from link list */
    preSamplerPtr->next = curSamplerPtr->next;
    free(curSamplerPtr);

    return 0;
}
