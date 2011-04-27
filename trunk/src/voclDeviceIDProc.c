#include "vocl_structures.h"

static struct strVOCLDeviceID *voclDeviceIDPtr = NULL;
static vocl_device_id voclDeviceID;
static int voclDeviceIDNum;
static int voclDeviceIDNo;

static vocl_device_id getVOCLDeviceIDValue()
{
    vocl_device_id device_id = voclDeviceID;
	voclDeviceID++;

    return device_id;
}

static struct strVOCLDeviceID *getVOCLDeviceIDPtr()
{
    if (voclDeviceIDNo >= voclDeviceIDNum) {
        voclDeviceIDNum *= 2;
        voclDeviceIDPtr = (struct strVOCLDeviceID *) realloc(voclDeviceIDPtr,
                                                   voclDeviceIDNum *
                                                   sizeof(struct strVOCLDeviceID));
    }
    return &voclDeviceIDPtr[voclDeviceIDNo++];
}


void voclDeviceIDInitialize()
{
    voclDeviceIDNum = VOCL_DEVICE_ID_NUM;
    voclDeviceIDPtr =
        (struct strVOCLDeviceID *) malloc(voclDeviceIDNum * sizeof(struct strVOCLDeviceID));
    voclDeviceIDNo = 0;
    voclDeviceID = 0;
}

void voclDeviceIDFinalize()
{
    if (voclDeviceIDPtr != NULL) {
        free(voclDeviceIDPtr);
        voclDeviceIDPtr = NULL;
    }
    voclDeviceIDNo = 0;
    voclDeviceID = 0;
    voclDeviceIDNum = 0;
}

vocl_device_id voclCLDeviceID2VOCLDeviceID(cl_device_id deviceID, int proxyID)
{
    struct strVOCLDeviceID *deviceIDPtr = getVOCLDeviceIDPtr();
    deviceIDPtr->clDeviceID = deviceID;
	deviceIDPtr->proxyID = proxyID;
    deviceIDPtr->voclDeviceID = getVOCLDeviceIDValue();

    return deviceIDPtr->voclDeviceID;
}

cl_device_id voclVOCLDeviceID2CLDeviceIDComm(vocl_device_id deviceID, int *proxyID)
/*comm and commData indicate the proxy process */
/*that the event corresponds to. They are the output of this function */
{
    /* the vocl event value indicates its location */
    /* in the event buffer */
    int deviceIDNo = (int) deviceID;

	*proxyID = voclDeviceIDPtr[deviceIDNo].proxyID;

    return voclDeviceIDPtr[deviceIDNo].clDeviceID;
}

