#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "voclOpencl.h"

#define MAX_PROXY_NUM 100
typedef char VOCL_HOST_NAME[63];

static VOCL_HOST_NAME *voclProxyNamePtr = NULL;
static int *voclIsOnLocalNodePtr = NULL;
static int *voclIndexToNodeMapping = NULL;
static int voclTotalProxyNum = 0;
static int voclProxyNo = 0;

static char *voclCreateProxyHostNameBuffer()
{
    /* first time called */
    if (voclTotalProxyNum == 0) {
        /* just setting this num to 100, it not enough, relcate */
        voclTotalProxyNum = MAX_PROXY_NUM;
        voclProxyNo = 0;
        /* allocate buffer for proxy host name */
        voclProxyNamePtr =
            (VOCL_HOST_NAME *) malloc(sizeof(VOCL_HOST_NAME) * voclTotalProxyNum);
        if (voclProxyNamePtr == NULL) {
            printf("allocate proxyHostName buffer error!\n");
            exit(1);
        }

        voclIsOnLocalNodePtr = (int *) malloc(sizeof(int) * voclTotalProxyNum);
        if (voclIsOnLocalNodePtr == NULL) {
            printf("allocate voclIsOnLocalNodePtr buffer error!\n");
            exit(1);
        }

        voclIndexToNodeMapping = (int *) malloc(sizeof(int) * voclTotalProxyNum);
        if (voclIndexToNodeMapping == NULL) {
            printf("allocate voclIndexToNodeMapping buffer error!\n");
            exit(1);
        }

    }

    /* if the allocate buffer is not enough, re-allocate */
    if (voclProxyNo >= voclTotalProxyNum) {
        voclTotalProxyNum *= 2;
        voclProxyNamePtr =
            (VOCL_HOST_NAME *) realloc(voclProxyNamePtr,
                                       sizeof(VOCL_HOST_NAME) * voclTotalProxyNum);
        if (voclProxyNamePtr == NULL) {
            printf("re-allocate proxyHostName buffer error!\n");
            exit(1);
        }

        voclIsOnLocalNodePtr =
            (int *) realloc(voclIsOnLocalNodePtr, sizeof(int) * voclTotalProxyNum);
        if (voclIsOnLocalNodePtr == NULL) {
            printf("re-allocate voclIsOnLocalNodePtr buffer error!\n");
            exit(1);
        }

        voclIndexToNodeMapping =
            (int *) realloc(voclIndexToNodeMapping, sizeof(int) * voclTotalProxyNum);
        if (voclIndexToNodeMapping == NULL) {
            printf("re-allocate voclIndexToNodeMapping buffer error!\n");
            exit(1);
        }
    }

    return voclProxyNamePtr[voclProxyNo++];
}

int voclGetProxyHostNum()
{
    return voclProxyNo;
}

void voclSetIndex2NodeMapping(int index, int node)
{
    voclIndexToNodeMapping[index] = node;
    return;
}

int voclGetIndex2NodeMapping(int index)
{
    return voclIndexToNodeMapping[index];
}

char *voclGetProxyHostName(int index)
{
    return (char *) voclProxyNamePtr[voclIndexToNodeMapping[index]];
}

int voclIsOnLocalNode(int index)
{
    return voclIsOnLocalNodePtr[voclIndexToNodeMapping[index]];
}

void voclProxyHostFinalize()
{
    voclTotalProxyNum = 0;
    voclProxyNo = 0;
    if (voclProxyNamePtr != NULL) {
        free(voclProxyNamePtr);
        voclProxyNamePtr = NULL;
    }

    if (voclIsOnLocalNodePtr) {
        free(voclIsOnLocalNodePtr);
        voclIsOnLocalNodePtr = NULL;
    }

    if (voclIndexToNodeMapping) {
        free(voclIndexToNodeMapping);
        voclIndexToNodeMapping = NULL;
    }

    return;
}

/* check whether the proxy name is already existed */
static int voclIsProxyNameExisted(char *name)
{
    int i, proxyNum = voclProxyNo;
    char *namePtr;

    for (i = 0; i < proxyNum; i++) {
        namePtr = (char *) voclProxyNamePtr[i];
        if (strcmp(name, namePtr) == 0) {
            return 1;
        }
    }

    return 0;
}

void voclCreateProxyHostNameList()
{
    char *hostNamePtr;
    char *fileNamePtr, *envPtr;
    char *tmpNamePtr, *nameBufferPtr;
    VOCL_HOST_NAME voclName, localNodeName;
    size_t size;
    int len;
    FILE *pfile;

    envPtr = getenv("PROXY_HOST_LIST");
    fileNamePtr = getenv("PROXY_HOST_FILE");

    MPI_Get_processor_name(localNodeName, &len);
    localNodeName[len] = '\0';
    /* host name is indicated directory */
    if (envPtr != NULL) {
        size = strlen(envPtr);
        hostNamePtr = (char *) malloc(size * sizeof(char));
        strcpy(hostNamePtr, envPtr);

        tmpNamePtr = strtok(hostNamePtr, ",");
        while (tmpNamePtr != NULL) {
            if (voclIsProxyNameExisted(tmpNamePtr) == 0) {
                nameBufferPtr = (char *) voclCreateProxyHostNameBuffer();
                strcpy(nameBufferPtr, tmpNamePtr);
                /* check it is a local node */
                if (strcmp(localNodeName, nameBufferPtr) == 0) {
                    voclIsOnLocalNodePtr[voclProxyNo - 1] = VOCL_TRUE;
                }
                else {
                    voclIsOnLocalNodePtr[voclProxyNo - 1] = VOCL_FALSE;
                }
            }
            tmpNamePtr = strtok(NULL, ",");
        }

        free(hostNamePtr);
    }
    /* host name is indicated in a file */
    else if (fileNamePtr != NULL) {
        pfile = fopen(fileNamePtr, "rt");
        if (pfile == NULL) {
            printf("File %s open error!\n", fileNamePtr);
            exit(1);
        }

        fscanf(pfile, "%s", voclName);
        while (!feof(pfile)) {
            if (strlen(voclName) > 0) {
                if (voclIsProxyNameExisted(voclName) == 0) {
                    nameBufferPtr = (char *) voclCreateProxyHostNameBuffer();
                    strcpy(nameBufferPtr, voclName);
                    /* check it is a local node */
                    if (strcmp(localNodeName, nameBufferPtr) == 0) {
                        voclIsOnLocalNodePtr[voclProxyNo - 1] = VOCL_TRUE;
                    }
                    else {
                        voclIsOnLocalNodePtr[voclProxyNo - 1] = VOCL_FALSE;
                    }
                }
                fscanf(pfile, "%s", voclName);
            }
        }
        fclose(pfile);
    }
    else {      /* create the slave on the local node */

        nameBufferPtr = (char *) voclCreateProxyHostNameBuffer();
        strcpy(nameBufferPtr, localNodeName);
        /*it is a local node */
        voclIsOnLocalNodePtr[voclProxyNo - 1] = VOCL_TRUE;
    }

    return;
}
