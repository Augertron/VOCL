#include <stdio.h>
#include "voclStructures.h"

extern cl_context voclMigCreateContext(const cl_context_properties * properties,
                     cl_uint num_devices,
                     vocl_device_id * devices,
                     void (CL_CALLBACK * pfn_notify) (const char *, const void *, size_t,
                                                      void *), void *user_data,
                     cl_int * errcode_ret);

static struct strVOCLContext *voclContextPtr = NULL;
static vocl_context voclContext;
static int voclContextNo;

static vocl_context getVOCLContextValue()
{
    vocl_context context = voclContext;
    voclContext++;

    return context;
}

static struct strVOCLContext *createVOCLContext()
{
    struct strVOCLContext *contextPtr;
    contextPtr = (struct strVOCLContext *) malloc(sizeof(struct strVOCLContext));
    contextPtr->next = voclContextPtr;
    voclContextPtr = contextPtr;

    return contextPtr;
}

static struct strVOCLContext *getVOCLContextPtr(vocl_context context)
{
    struct strVOCLContext *contextPtr;
    contextPtr = voclContextPtr;
    while (contextPtr != NULL) {
        if (contextPtr->voclContext == context) {
            break;
        }
        contextPtr = contextPtr->next;
    }

    if (contextPtr == NULL) {
        printf("Error, context does not exist!\n");
        exit(1);
    }

    return contextPtr;
}

void voclContextInitialize()
{
    voclContextPtr = NULL;
    voclContextNo = 0;
    voclContext = 0;
}

void voclContextFinalize()
{
    struct strVOCLContext *contextPtr, *tmpcontextPtr;
    contextPtr = voclContextPtr;
    while (contextPtr != NULL) {
        tmpcontextPtr = contextPtr->next;
        free(contextPtr);
        contextPtr = tmpcontextPtr;
    }

    voclContextPtr = NULL;
    voclContextNo = 0;
    voclContext = 0;
}

void voclContextSetMigrationStatus(vocl_context context, char status)
{
	struct strVOCLContext *contextPtr = getVOCLContextPtr(context);
	contextPtr->migrationStatus = status;
	return;
}

char voclContextGetMigrationStatus(vocl_context context)
{
	struct strVOCLContext *contextPtr = getVOCLContextPtr(context);
	return contextPtr->migrationStatus;
}

vocl_context voclCLContext2VOCLContext(cl_context context, int proxyRank,
                                       int proxyIndex, MPI_Comm proxyComm,
                                       MPI_Comm proxyCommData)
{
    struct strVOCLContext *contextPtr = createVOCLContext();
    contextPtr->clContext = context;
    contextPtr->proxyRank = proxyRank;
    contextPtr->proxyIndex = proxyIndex;
    contextPtr->proxyComm = proxyComm;
    contextPtr->proxyCommData = proxyCommData;
	contextPtr->migrationStatus = 0;
    contextPtr->voclContext = getVOCLContextValue();

    return contextPtr->voclContext;
}

cl_context voclVOCLContext2CLContextComm(vocl_context context, int *proxyRank,
                                         int *proxyIndex, MPI_Comm * proxyComm,
                                         MPI_Comm * proxyCommData)
{
    struct strVOCLContext *contextPtr = getVOCLContextPtr(context);
    *proxyRank = contextPtr->proxyRank;
    *proxyIndex = contextPtr->proxyIndex;
    *proxyComm = contextPtr->proxyComm;
    *proxyCommData = contextPtr->proxyCommData;

    return contextPtr->clContext;
}

cl_context voclVOCLContext2CLContext(vocl_context context)
{
    struct strVOCLContext *contextPtr = getVOCLContextPtr(context);
    return contextPtr->clContext;
}

void voclUpdateVOCLContext(vocl_context voclContext, int proxyRank, int proxyIndex,
                           MPI_Comm proxyComm, MPI_Comm proxyCommData, vocl_device_id deviceID)
{
    struct strVOCLContext *contextPtr = getVOCLContextPtr(voclContext);
    int err;

    /* release previous context */
    clReleaseContext((cl_context)voclContext);

    contextPtr->proxyRank = proxyRank;
    contextPtr->proxyIndex = proxyIndex;
    contextPtr->proxyComm = proxyComm;
    contextPtr->proxyCommData = proxyCommData;

    contextPtr->clContext = voclMigCreateContext(0, 1, &deviceID, NULL, NULL, &err);

	/* context is the higheest-level object to be migrated, each time */
	/* a migration is performed, migration status is increased by one */
	contextPtr->migrationStatus++;

    return;
}

int voclReleaseContext(vocl_context context)
{
    struct strVOCLContext *contextPtr, *preContextPtr, *curContextPtr;
    /* the first node in the link list */
    if (context == voclContextPtr->voclContext) {
        contextPtr = voclContextPtr;
        voclContextPtr = voclContextPtr->next;
        free(contextPtr);

        return 0;
    }

    contextPtr = NULL;
    preContextPtr = voclContextPtr;
    curContextPtr = voclContextPtr->next;
    while (curContextPtr != NULL) {
        if (context == curContextPtr->voclContext) {
            contextPtr = curContextPtr;
            break;
        }
        preContextPtr = curContextPtr;
        curContextPtr = curContextPtr->next;
    }

    if (contextPtr == NULL) {
        printf("context does not exist!\n");
        exit(1);
    }

    /* remote the current node from link list */
    preContextPtr->next = curContextPtr->next;
    free(curContextPtr);

    return 0;
}
