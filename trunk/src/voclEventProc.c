#include "voclEventProc.h"

static struct strVOCLEventBuffer *voclEventBufferPtr = NULL;
static vocl_event voclEvent;
static int voclEventNum;
static int voclEventNo;

static vocl_event getVOCLEvent()
{
    vocl_event event = voclEvent;
    voclEvent++;

    return event;
}

static struct strVOCLEventBuffer *getVOCLEventBufferPtr()
{
    if (voclEventNo >= voclEventNum) {
        voclEventNum *= 2;
        voclEventBufferPtr = (struct strVOCLEventBuffer *) realloc(voclEventBufferPtr,
                                                                   voclEventNum *
                                                                   sizeof(struct
                                                                          strVOCLEventBuffer));
    }
    return &voclEventBufferPtr[voclEventNo++];
}


void voclEventInitialize()
{
    voclEventNum = VOCL_EVENT_NUM;
    voclEventBufferPtr =
        (struct strVOCLEventBuffer *) malloc(voclEventNum * sizeof(struct strVOCLEventBuffer));
    voclEventNo = 0;
    voclEvent = 0;
}

void voclEventFinalize()
{
    if (voclEventBufferPtr != NULL) {
        free(voclEventBufferPtr);
        voclEventBufferPtr = NULL;
    }
    voclEventNo = 0;
    voclEvent = 0;
    voclEventNum = 0;
}

vocl_event voclCLEvent2VOCLEvent(cl_event event)
{
    struct strVOCLEventBuffer *eventBufferPtr = getVOCLEventBufferPtr();
    eventBufferPtr->clEvent = event;
    eventBufferPtr->voclEvent = getVOCLEvent();

    return eventBufferPtr->voclEvent;
}

cl_event voclVOCLEvent2CLEvent(vocl_event event)
{
    /* the vocl event value indicates its location */
    /* in the event buffer */
    int eventNo = (int) event;
    return voclEventBufferPtr[eventNo].clEvent;
}

void voclVOCLEvents2CLEvents(vocl_event * voclEventList,
                             cl_event * clEventList, cl_uint eventNum)
{
    cl_uint i;
    for (i = 0; i < eventNum; i++) {
        clEventList[i] = voclVOCLEvent2CLEvent(voclEventList[i]);
    }

    return;
}
