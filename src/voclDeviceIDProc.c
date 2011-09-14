#include <stdio.h>
#include "voclStructures.h"

static struct strVOCLDeviceID *voclDeviceIDPtr = NULL;
static vocl_device_id voclDeviceID;
static int voclDeviceIDNo;

static vocl_device_id getVOCLDeviceIDValue()
{
    vocl_device_id device = voclDeviceID;
    voclDeviceID++;

    return device;
}

static struct strVOCLDeviceID *createVOCLDeviceID()
{
    struct strVOCLDeviceID *devicePtr;
    devicePtr = (struct strVOCLDeviceID *) malloc(sizeof(struct strVOCLDeviceID));
    devicePtr->next = voclDeviceIDPtr;
    voclDeviceIDPtr = devicePtr;

    return devicePtr;
}

static struct strVOCLDeviceID *voclGetDeviceIDPtr(vocl_device_id device)
{
    struct strVOCLDeviceID *devicePtr;
    devicePtr = voclDeviceIDPtr;
    while (devicePtr != NULL) {
        if (devicePtr->voclDeviceID == device) {
            break;
        }
        devicePtr = devicePtr->next;
    }

    if (devicePtr == NULL) {
        printf("Error, device does not exist!\n");
        exit(1);
    }

    return devicePtr;
}

void voclDeviceIDInitialize()
{
    voclDeviceIDPtr = NULL;
    voclDeviceIDNo = 0;
    voclDeviceID = 0;
}

void voclDeviceIDFinalize()
{
    struct strVOCLDeviceID *devicePtr, *tmpdevicePtr;
    devicePtr = voclDeviceIDPtr;
    while (devicePtr != NULL) {
        tmpdevicePtr = devicePtr->next;
        free(devicePtr);
        devicePtr = tmpdevicePtr;
    }

    voclDeviceIDPtr = NULL;
    voclDeviceIDNo = 0;
    voclDeviceID = 0;
}

vocl_device_id voclCLDeviceID2VOCLDeviceID(cl_device_id device, int proxyRank,
                                           int proxyIndex, MPI_Comm proxyComm,
                                           MPI_Comm proxyCommData)
{
    struct strVOCLDeviceID *devicePtr = createVOCLDeviceID();
    devicePtr->clDeviceID = device;
    devicePtr->proxyRank = proxyRank;
    devicePtr->proxyIndex = proxyIndex;
    devicePtr->proxyComm = proxyComm;
    devicePtr->proxyCommData = proxyCommData;
    devicePtr->voclDeviceID = getVOCLDeviceIDValue();

    return devicePtr->voclDeviceID;
}

cl_device_id voclVOCLDeviceID2CLDeviceIDComm(vocl_device_id device, int *proxyRank,
                                             int *proxyIndex, MPI_Comm * proxyComm,
                                             MPI_Comm * proxyCommData)
{
    struct strVOCLDeviceID *devicePtr = voclGetDeviceIDPtr(device);
    *proxyRank = devicePtr->proxyRank;
    *proxyIndex = devicePtr->proxyIndex;
    *proxyComm = devicePtr->proxyComm;
    *proxyCommData = devicePtr->proxyCommData;

    return devicePtr->clDeviceID;
}

void voclUpdateVOCLDeviceID(vocl_device_id voclDevice, int proxyRank,
                            int proxyIndex, MPI_Comm proxyComm, MPI_Comm proxyCommData,
                            cl_device_id clDevice)
{
    struct strVOCLDeviceID *devicePtr = voclGetDeviceIDPtr(voclDevice);
    devicePtr->clDeviceID = clDevice;
    devicePtr->proxyRank = proxyRank;
    devicePtr->proxyIndex = proxyIndex;
    devicePtr->proxyComm = proxyComm;
    devicePtr->proxyCommData = proxyCommData;
}
