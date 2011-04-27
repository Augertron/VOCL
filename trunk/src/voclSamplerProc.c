#include "vocl_structures.h"

static struct strVOCLSampler *voclSamplerPtr = NULL;
static vocl_sampler voclSampler;
static int voclSamplerNum;
static int voclSamplerNo;

static vocl_sampler getVOCLSamplerValue()
{
    vocl_sampler sampler = voclSampler;
	voclSampler++;

    return sampler;
}

static struct strVOCLSampler *getVOCLSamplerPtr()
{
    if (voclSamplerNo >= voclSamplerNum) {
        voclSamplerNum *= 2;
        voclSamplerPtr = (struct strVOCLSampler *) realloc(voclSamplerPtr,
                                                   voclSamplerNum *
                                                   sizeof(struct strVOCLSampler));
    }
    return &voclSamplerPtr[voclSamplerNo++];
}


void voclSamplerInitialize()
{
    voclSamplerNum = VOCL_CONTEXT_NUM;
    voclSamplerPtr =
        (struct strVOCLSampler *) malloc(voclSamplerNum * sizeof(struct strVOCLSampler));
    voclSamplerNo = 0;
    voclSampler = 0;
}

void voclSamplerFinalize()
{
    if (voclSamplerPtr != NULL) {
        free(voclSamplerPtr);
        voclSamplerPtr = NULL;
    }
    voclSamplerNo = 0;
    voclSampler = 0;
    voclSamplerNum = 0;
}

vocl_sampler voclCLSampler2VOCLSampler(cl_sampler sampler, int proxyID)
{
    struct strVOCLSampler *samplerPtr = getVOCLSamplerPtr();
    samplerPtr->clSampler = sampler;
	samplerPtr->proxyID = proxyID;
    samplerPtr->voclSampler = getVOCLSamplerValue();

    return samplerPtr->voclSampler;
}

cl_sampler voclVOCLSampler2CLSamplerComm(vocl_sampler sampler, int *proxyID)
{
    /* the vocl event value indicates its location */
    /* in the event buffer */
    int samplerNo = (int) sampler;

	*proxyID = voclSamplerPtr[samplerNo].proxyID;

    return voclSamplerPtr[samplerNo].clSampler;
}

