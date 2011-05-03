#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

#define MAX_PROXY_NUM 100
typedef char VOCL_HOST_NAME[63];

static VOCL_HOST_NAME *voclProxyNamePtr = NULL;
static int voclTotalProxyNum = 0;
static int voclProxyNo = 0;

static char *voclCreateProxyHostNameBuffer()
{
	/* first time called */
	if (voclTotalProxyNum == 0)
	{
		/* just setting this num to 100, it not enough, relcate */
		voclTotalProxyNum = MAX_PROXY_NUM; 
		voclProxyNo = 0;
		/* allocate buffer for proxy host name */
		voclProxyNamePtr = (VOCL_HOST_NAME *)malloc(sizeof(VOCL_HOST_NAME) * voclTotalProxyNum);
		if (voclProxyNamePtr == NULL)
		{
			printf("allocate proxyHostName buffer error!\n");
			exit(1);
		}
	}

	/* if the allocate buffer is not enough, re-allocate */
	if (voclProxyNo >= voclTotalProxyNum)
	{
		voclTotalProxyNum *= 2;
		voclProxyNamePtr = (VOCL_HOST_NAME *)realloc(voclProxyNamePtr, sizeof(VOCL_HOST_NAME) * voclTotalProxyNum);
		if (voclProxyNamePtr == NULL)
		{
			printf("allocate proxyHostName buffer error!\n");
			exit(1);
		}
	}

	return voclProxyNamePtr[voclProxyNo++];

}

int voclGetProxyHostNum()
{
	return voclProxyNo;
}

char *voclGetProxyHostName(int index)
{
	return (char *)voclProxyNamePtr[index];
}

void voclProxyHostFinalize()
{
	voclTotalProxyNum = 0;
	voclProxyNo = 0;
	if (voclProxyNamePtr != NULL)
	{
		free(voclProxyNamePtr);
		voclProxyNamePtr = NULL;
	}

	return;
}

/* check whether the proxy name is already existed */
static int voclIsProxyNameExisted(char *name)
{
	int i, proxyNum = voclProxyNo;
	char *namePtr;

	for (i = 0; i < proxyNum; i++)
	{
		namePtr = voclGetProxyHostName(i);
		if (strcmp(name, namePtr) == 0)
		{
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
	VOCL_HOST_NAME voclName;
	size_t size;
	int len;
	FILE *pfile;

	envPtr = getenv("PROXY_HOST_LIST");
	fileNamePtr = getenv("PROXY_HOST_FILE");

	/* host name is indicated directory */
	if (envPtr != NULL)
	{
		size = strlen(envPtr);
		hostNamePtr = (char *)malloc(size * sizeof(char));
		strcpy(hostNamePtr, envPtr);

		tmpNamePtr = strtok(hostNamePtr, ",");
		while (tmpNamePtr != NULL)
		{
			if (voclIsProxyNameExisted(tmpNamePtr) == 0)
			{
				nameBufferPtr = (char *)voclCreateProxyHostNameBuffer();
				strcpy(nameBufferPtr, tmpNamePtr);
			}
			tmpNamePtr = strtok(NULL, ",");
		}

		free(hostNamePtr);
	}

	/* host name is indicated in a file */
	else if (fileNamePtr != NULL)
	{
		pfile = fopen(fileNamePtr, "rt");
		if (pfile == NULL)
		{
			printf("File %s open error!\n", fileNamePtr);
			exit(1);
		}

		fscanf(pfile, "%s", voclName);
		while (!feof(pfile))
		{
			if (strlen(voclName) > 0)
			{
				if (voclIsProxyNameExisted(voclName) == 0)
				{
					nameBufferPtr = (char *)voclCreateProxyHostNameBuffer();
					strcpy(nameBufferPtr, voclName);
				}
				fscanf(pfile, "%s", voclName);
			}
		}
		fclose(pfile);
	}
	else /* create the slave on the local node */
	{
		MPI_Get_processor_name(voclName, &len);
		voclName[len] = '\0';
		nameBufferPtr = (char *)voclCreateProxyHostNameBuffer();
		strcpy(nameBufferPtr, voclName);
	}

	return;
}


