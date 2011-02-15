#include <stdio.h>
#include <stdlib.h>
#include "param.h"

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		printf("Usage: %s inDB\n", argv[0]);
		return 1;
	}

	char inDataFilePathName[255];
	char inLenFilePathName[255];
	FILE *pDataFile, *pLenFile;
	
	int nSeqLen, nSeqNum;
	char *strTemp;

	sprintf(inDataFilePathName, "%s.data", argv[1]);
	sprintf(inLenFilePathName, "%s.loc", argv[1]);

	strTemp = new char[100000000];

	//open the data and length files
	pDataFile = fopen(inDataFilePathName, "rb");
	if (pDataFile == NULL)
	{
		printf("DB data file %s open error!\n", inDataFilePathName);
		return 1;
	}

	pLenFile = fopen(inLenFilePathName, "rb");
	if (pLenFile == NULL)
	{
		printf("DB len file %s open error!\n", inLenFilePathName);
		return 1;
	}

	fread(&nSeqNum, sizeof(int), 1, pLenFile);
	//while (!feof(pLenFile))
	for (int nSeqNo = 0; nSeqNo < nSeqNum; nSeqNo++)
	{
		//fscanf(pLenFile, "%d", &nSeqLen);
		fread(&nSeqLen, sizeof(int), 1, pLenFile);
		if (nSeqLen <= 0)
		{
			printf("Error, length of sequence %d is %d\n", nSeqNum, nSeqLen);
			delete strTemp;
			fclose(pLenFile);
			fclose(pDataFile);

			return 1;
		}

		fread(strTemp, sizeof(char), nSeqLen, pDataFile);

		for (int i = 0; i < nSeqLen; i++)
		{
			printf("%c", amino_acids[strTemp[i]]);
		}
		printf("\n");
	}

	fclose(pLenFile);
	fclose(pDataFile);
	delete strTemp;
}
