#include <stdio.h>
#include "voclStructures.h"

static struct strVOCLEvent *voclEventPtr = NULL;
static vocl_event voclEvent;    /* event value */
static int voclEventNo;         /* actual number of events */

static vocl_event getVOCLEventValue()
{
    vocl_event event = voclEvent;
    voclEvent++;

    return event;
}

static struct strVOCLEvent *createVOCLEvent()
{
    struct strVOCLEvent *eventPtr;
    eventPtr = (struct strVOCLEvent *) malloc(sizeof(struct strVOCLEvent));
    eventPtr->next = voclEventPtr;
    voclEventPtr = eventPtr;
    return eventPtr;
}

static struct strVOCLEvent *getVOCLEventPtr(vocl_event event)
{
    struct strVOCLEvent *eventPtr;
    eventPtr = voclEventPtr;
    while (eventPtr != NULL) {
        if (eventPtr->voclEvent == event) {
            break;
        }
        eventPtr = eventPtr->next;
    }

    if (eventPtr == NULL) {
        printf("Error, event does not exist!\n");
        exit(1);
    }

    return eventPtr;
}

void voclEventInitialize()
{
    voclEventPtr = NULL;
    voclEventNo = 0;
    voclEvent = 0;
}

void voclEventFinalize()
{
    struct strVOCLEvent *eventPtr, *tmpEventPtr;
    eventPtr = voclEventPtr;
    while (eventPtr != NULL) {
        tmpEventPtr = eventPtr->next;
        free(eventPtr);
        eventPtr = tmpEventPtr;
    }

    voclEventPtr = NULL;
    voclEventNo = 0;
    voclEvent = 0;
}

vocl_event voclCLEvent2VOCLEvent(cl_event event, int proxyRank,
                                 int proxyIndex, MPI_Comm proxyComm, MPI_Comm proxyCommData)
{
    struct strVOCLEvent *eventPtr = createVOCLEvent();
    eventPtr->clEvent = event;
    eventPtr->proxyRank = proxyRank;
    eventPtr->proxyIndex = proxyIndex;
    eventPtr->proxyComm = proxyComm;
    eventPtr->proxyCommData = proxyCommData;
    eventPtr->voclEvent = getVOCLEventValue();

    return eventPtr->voclEvent;
}


cl_event voclVOCLEvent2CLEventComm(vocl_event event, int *proxyRank,
                                   int *proxyIndex, MPI_Comm * proxyComm,
                                   MPI_Comm * proxyCommData)
/*comm and commData indicate the proxy process */
/*that the event corresponds to. They are the output of this function */
{
    /* the vocl event value indicates its location */
    /* in the event buffer */
    struct strVOCLEvent *eventPtr = getVOCLEventPtr(event);
    *proxyRank = eventPtr->proxyRank;
    *proxyIndex = eventPtr->proxyIndex;
    *proxyComm = eventPtr->proxyComm;
    *proxyCommData = eventPtr->proxyCommData;

    return eventPtr->clEvent;
}

void voclUpdateVOCLEvent(vocl_event voclEvent, int proxyRank, int proxyIndex,
                         MPI_Comm proxyComm, MPI_Comm proxyCommData, cl_event clEvent)
{
    struct strVOCLEvent *eventPtr = getVOCLEventPtr(voclEvent);
    eventPtr->clEvent = clEvent;
    eventPtr->proxyRank = proxyRank;
    eventPtr->proxyIndex = proxyIndex;
    eventPtr->proxyComm = proxyComm;
    eventPtr->proxyCommData = proxyCommData;

    return;
}

static cl_event voclVOCLEvent2CLEvent(vocl_event event)
/*comm and commData indicate the proxy process */
/*that the event corresponds to. They are the output of this function */
{
    /* the vocl event value indicates its location */
    /* in the event buffer */
    struct strVOCLEvent *eventPtr = getVOCLEventPtr(event);

    return eventPtr->clEvent;
}



/* diferent events correspond to different proxy process */
void voclVOCLEvents2CLEventsComm(vocl_event * voclEventList,
                                 cl_event * clEventList, cl_uint eventNum, int *proxyRank,
                                 int *proxyIndex, MPI_Comm * proxyComm,
                                 MPI_Comm * proxyCommData)
{
    cl_uint i;
    for (i = 0; i < eventNum; i++) {
        clEventList[i] = voclVOCLEvent2CLEventComm(voclEventList[i],
                                                   proxyRank, proxyIndex, proxyComm,
                                                   proxyCommData);
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

int voclReleaseEvent(vocl_event event)
{
    struct strVOCLEvent *eventPtr, *preEventPtr, *curEventPtr;
    /* the first node in the link list */
    if (event == voclEventPtr->voclEvent) {
        eventPtr = voclEventPtr;
        voclEventPtr = voclEventPtr->next;
        free(eventPtr);

        return 0;
    }

    eventPtr = NULL;
    preEventPtr = voclEventPtr;
    curEventPtr = voclEventPtr->next;
    while (curEventPtr != NULL) {
        if (event == curEventPtr->voclEvent) {
            eventPtr = curEventPtr;
            break;
        }
        preEventPtr = curEventPtr;
        curEventPtr = curEventPtr->next;
    }

    if (eventPtr == NULL) {
        printf("event does not exist!\n");
        exit(1);
    }

    /* remote the current node from link list */
    preEventPtr->next = curEventPtr->next;
    free(curEventPtr);

    return 0;
}
