#include "vocl_structures.h"

static struct strVOCLEvent *voclEventPtr = NULL;
static vocl_event voclEvent; /* event value */
static int voclEventNum;     /* max number of events */
static int voclEventNo;      /* actual number of events */

static vocl_event getVOCLEventValue()
{
    vocl_event event = voclEvent;
    voclEvent++;

    return event;
}

static struct strVOCLEvent *getVOCLEventPtr()
{
    if (voclEventNo >= voclEventNum) {
        voclEventNum *= 2;
        voclEventPtr = (struct strVOCLEvent *) realloc(voclEventPtr,
                                                   voclEventNum *
                                                   sizeof(struct strVOCLEvent));
    }
    return &voclEventPtr[voclEventNo++];
}


void voclEventInitialize()
{
    voclEventNum = VOCL_EVENT_NUM;
    voclEventPtr =
        (struct strVOCLEvent *) malloc(voclEventNum * sizeof(struct strVOCLEvent));
    voclEventNo = 0;
    voclEvent = 0;
}

void voclEventFinalize()
{
    if (voclEventPtr != NULL) {
        free(voclEventPtr);
        voclEventPtr = NULL;
    }
    voclEventNo = 0;
    voclEvent = 0;
    voclEventNum = 0;
}

vocl_event voclCLEvent2VOCLEvent(cl_event event, int proxyID)
{
    struct strVOCLEvent *eventPtr = getVOCLEventPtr();
    eventPtr->clEvent = event;
	eventPtr->proxyID = proxyID;
    eventPtr->voclEvent = getVOCLEventValue();

    return eventPtr->voclEvent;
}


cl_event voclVOCLEvent2CLEventComm(vocl_event event, int *proxyID)
/*comm and commData indicate the proxy process */
/*that the event corresponds to. They are the output of this function */
{
    /* the vocl event value indicates its location */
    /* in the event buffer */
    int eventNo = (int) event;
	*proxyID = voclEventPtr[eventNo].proxyID;

    return voclEventPtr[eventNo].clEvent;
}

static cl_event voclVOCLEvent2CLEvent(vocl_event event)
/*comm and commData indicate the proxy process */
/*that the event corresponds to. They are the output of this function */
{
    /* the vocl event value indicates its location */
    /* in the event buffer */
    int eventNo = (int) event;

    return voclEventPtr[eventNo].clEvent;
}



/* diferent events correspond to different proxy process */
void voclVOCLEvents2CLEventsComm(vocl_event * voclEventList,
       cl_event * clEventList, cl_uint eventNum, int *proxyID)
{
    cl_uint i;
    for (i = 0; i < eventNum; i++) {
        clEventList[i] = voclVOCLEvent2CLEventComm(voclEventList[i], proxyID);
    }

    return;
}

/* all events correspond to the same proxy process */
void voclVOCLEvents2CLEvents(vocl_event * voclEventList,
                             cl_event * clEventList, cl_uint eventNum)
{
    cl_uint i;
    for (i = 0; i < eventNum; i++) {
        clEventList[i] = voclVOCLEvent2CLEvent(voclEventList[i]);
    }

    return;
}


