#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

#define FORCED_MIGRATION            62
#define PROGRAM_END                 64

extern void voclCreateProxyHostNameList();
extern int voclGetProxyHostNum();
extern char *voclGetProxyHostName(int index);
extern void voclProxyHostFinalize();

struct strForcedMigration {
	int status;
	int rankThreshold;
	int res;
};

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		printf("Usage: %s rankThreshold\n", argv[0]);
		return 1;
	}

	int proxyNum, proxyNo, i;
	int rankThreshold = 0;
	int err;
	char serviceName[256];
	char portName[MPI_MAX_PORT_NAME];
	struct strForcedMigration tmpForcedMigration;
	MPI_Comm *proxyComm, *proxyCommData;
	MPI_Request request[2];
	MPI_Status status[2];

	MPI_Init(&argc, &argv);

	voclCreateProxyHostNameList();
	proxyNum = voclGetProxyHostNum();

	rankThreshold = atoi(argv[1]);

	/* allocate buffer for communicators */
	proxyComm = (MPI_Comm *)malloc(sizeof(MPI_Comm) * proxyNum);
	proxyCommData = (MPI_Comm *)malloc(sizeof(MPI_Comm) * proxyNum);

	/* establish connection */
	for (i = 0; i < proxyNum; i++)
	{
		sprintf(serviceName, "voclCloud%s", voclGetProxyHostName(i));
		err = MPI_Lookup_name(serviceName, MPI_INFO_NULL, portName);
		if (err != MPI_SUCCESS)
		{
			printf("Lookup service name %s error, %d!\n", serviceName, err);
			MPI_Finalize();
			exit(1);
		}

		err = MPI_Comm_connect(portName,MPI_INFO_NULL, 0, MPI_COMM_SELF, 
				&proxyComm[i]);
		if (err != MPI_SUCCESS)
		{
			printf("MPI_Comm_connect error, %d\n", err);
			MPI_Finalize();
			exit(1);
		}

		err = MPI_Comm_dup(proxyComm[i], &proxyCommData[i]);
		if (err != MPI_SUCCESS)
		{
			printf("MPI_Comm_dup error, %d\n", err);
			MPI_Finalize();
			exit(1);
		}	
	}

	/* send forced migration message */
	for (i = 0; i < proxyNum; i++)
	{
		tmpForcedMigration.status = 1;
		tmpForcedMigration.rankThreshold = rankThreshold;
		MPI_Isend(&tmpForcedMigration, sizeof(struct strForcedMigration), MPI_BYTE,
			0, FORCED_MIGRATION, proxyComm[i], &request[0]);
		MPI_Irecv(&tmpForcedMigration, sizeof(struct strForcedMigration), MPI_BYTE,
			0, FORCED_MIGRATION, proxyComm[i], &request[1]);
		MPI_Waitall(2, request, status);
	}

	/* disconnect from the proxy */
	for (i = 0; i < proxyNum; i++)
	{
		MPI_Send(NULL, 0, MPI_BYTE, 0, PROGRAM_END, proxyComm[i]);
		MPI_Comm_disconnect(&proxyComm[i]);
		MPI_Comm_disconnect(&proxyCommData[i]);
	}

	free(proxyComm);
	free(proxyCommData);

	voclProxyHostFinalize();
	MPI_Finalize();

	return 0;
}
