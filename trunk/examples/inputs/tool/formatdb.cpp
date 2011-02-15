#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct seqInfo {
	unsigned long disStartPos;
	unsigned long seqStartPos;
	int seqSize;
} SEQINFO;

void quicksort(SEQINFO *seqInfo, int nBegin, int nEnd);
void swap(SEQINFO *a, SEQINFO *b);
void BubbleSort(SEQINFO *seqInfo, int nSeqNum);
void encoding(char *seq, int &nLen);
short char2index(char inch);

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		printf("Usage: %s inDB\n", argv[0]);
		return 1;
	}

	char inDBFilePathName[255];
	char outDataFilePathName[255];
	char outLenFilePathName[255];
	char *strTemp;
	int nSeqNum, nSeqNo;
	int nSeqSize, nDiscriptionSize;
	unsigned long nFilePos;

	FILE *pInFile, *pOutDataFile, *pOutLenFile;

	sprintf(inDBFilePathName, "%s", argv[1]);
	sprintf(outDataFilePathName, "%s.data", argv[1]);
	sprintf(outLenFilePathName, "%s.loc", argv[1]);

	struct seqInfo *seqInfoPtr;
	seqInfoPtr = new SEQINFO[50000000];
	if (seqInfoPtr == NULL)
	{
		printf("Allocate info buffer error!\n");
		return 1;
	}
	
	strTemp = new char[500000];
	if (strTemp == NULL)
	{
		printf("Temp buffer allocation error!\n");
		return 1;
	}

	pInFile = fopen(inDBFilePathName, "rt");
	if (pInFile == NULL)
	{
		printf("Open input db %s error!\n", inDBFilePathName);
		return 1;
	}

	nSeqNum = -1;
	nFilePos = 0;
	while (!feof(pInFile))
	{
		strTemp[0] = '\0';
		fgets(strTemp, 500000, pInFile);
		if (*strTemp == '>')
		{
			nSeqNum++;
			printf("nSeqNum = %d\n", nSeqNum);
			if (nSeqNum > 0)
			{
				seqInfoPtr[nSeqNum - 1].seqSize = nSeqSize;
				nFilePos += nSeqSize;
			}

			nDiscriptionSize = strlen(strTemp);
			seqInfoPtr[nSeqNum].disStartPos = nFilePos;
			nFilePos += nDiscriptionSize;
			seqInfoPtr[nSeqNum].seqStartPos = nFilePos;
			nSeqSize = 0;
		}
		else
		{
			nSeqSize += strlen(strTemp);
		}
	}
	//For the last sequence
	seqInfoPtr[nSeqNum++].seqSize = nSeqSize;
	fclose(pInFile);

	pInFile = fopen(inDBFilePathName, "rb");
	if (pInFile == NULL)
	{
		printf("Open input db %s error!\n", inDBFilePathName);
		return 1;
	}

	pOutDataFile = fopen(outDataFilePathName, "wb");
	if (pOutDataFile == NULL)
	{
		printf("Open output data file %s error!\n", outDataFilePathName);
		return 1;
	}

	pOutLenFile = fopen(outLenFilePathName, "wb");
	if (pOutLenFile == NULL)
	{
		printf("Open output len file %s error!\n", outLenFilePathName);
		return 1;
	}

	//sort the structure via sequence length
	//from shortest to longest
	//quicksort(seqInfoPtr, 0, nSeqNum);
	//BubbleSort(seqInfoPtr, nSeqNum);

	//store the sorted database to output file
	//fprintf(pOutLenFile, "%d\n", nSeqNum);
	fwrite(&nSeqNum, sizeof(int), 1, pOutLenFile);
	for (nSeqNo = 0; nSeqNo < nSeqNum; nSeqNo++)
	{
//		printf("seq = %d, disStart = %d, seqStart = %d, seqSize = %d\n",
//				nSeqNo,
//				seqInfoPtr[nSeqNo].disStartPos,
//				seqInfoPtr[nSeqNo].seqStartPos,
//				seqInfoPtr[nSeqNo].seqSize);
	
		//read the discription
		fseek(pInFile, seqInfoPtr[nSeqNo].disStartPos, SEEK_SET);
		nDiscriptionSize = seqInfoPtr[nSeqNo].seqStartPos - seqInfoPtr[nSeqNo].disStartPos;
		fread(strTemp, sizeof(char), nDiscriptionSize, pInFile);
		//fwrite(strTemp, sizeof(char), nDiscriptionSize, pOutFile);

		//read the sequence
		fread(strTemp, sizeof(char), seqInfoPtr[nSeqNo].seqSize, pInFile);
		encoding(strTemp, seqInfoPtr[nSeqNo].seqSize);
		//fprintf(pOutLenFile, "%d\n", seqInfoPtr[nSeqNo].seqSize);
		fwrite(&seqInfoPtr[nSeqNo].seqSize, sizeof(int), 1, pOutLenFile);
		fwrite(strTemp, sizeof(char), seqInfoPtr[nSeqNo].seqSize, pOutDataFile);
	}
	fclose(pInFile);
	fclose(pOutDataFile);
	fclose(pOutLenFile);

	delete seqInfoPtr;
	delete strTemp;
	return 0;
}

void swap(SEQINFO *a, SEQINFO *b)
{
	SEQINFO TEMP;
	TEMP = *a;
	*a = *b;
	*b = TEMP;
}

void quicksort(SEQINFO *seqInfo, int nBegin, int nEnd)
{
	SEQINFO piv;
	int r, l;
	if (nEnd > nBegin)
	{
		piv = seqInfo[nBegin];
		l = nBegin + 1;
		r = nEnd;
		while (l < r)
		{
			if (seqInfo[l].seqSize > piv.seqSize)
			{
				l++;
			}
			else
			{
				swap(&seqInfo[l], &seqInfo[--r]);
			}
		}
		swap(&seqInfo[--l], &seqInfo[nBegin]);
		quicksort(seqInfo, nBegin, l);
		quicksort(seqInfo, r, nEnd);
	}
}

void BubbleSort(SEQINFO *seqInfo, int nSeqNum)
{
	int i, j;
	SEQINFO temp;

	for (i = 0; i < nSeqNum - 1; i++)
	{
		for (j = nSeqNum - 1; j > i; j--)
		{
			if (seqInfo[j].seqSize > seqInfo[j - 1].seqSize)
			{
				temp = seqInfo[j];
				seqInfo[j] = seqInfo[j - 1];
				seqInfo[j - 1] = temp;
			}
		}
	}
}

void encoding(char *seq, int &nLen)
{
	int i;
	int nOutLoc = 0;
	char code;
	for (i = 0; i < nLen; i++)
	{
		code = char2index(seq[i]);
		if (code >= 0)
		{
			seq[nOutLoc] = code;
			nOutLoc++;
		}
	}

	nLen = nOutLoc;

	return;
}

short char2index(char inch)
{
	int result;
	if(inch >= 65 && inch <= 73) //'A' --> 'I'
	{
		result = inch - 65;
	}
	else if (inch >= 75 && inch <= 78) //'K' --> 'N'
	{
		result = inch - 66;
	}
	else if (inch >= 80 && inch <= 84) //'P' --> 'T'
	{
		result = inch - 67;
	}
	else if (inch >= 86 && inch <= 90) //'V' --> 'Z'
	{
		result = inch - 68;
	}
	else if (inch >= 97 && inch <= 105)
	{
		result = inch - 97;
	}
	else if (inch >= 107 && inch <= 110)
	{
		result = inch - 98;
	}
	else if (inch >= 112 && inch <= 116)
	{
		result = inch - 99;
	}
	else if (inch >= 118 && inch <= 122)
	{
		result = inch - 100;
	}
	else
	{
		return -1;
	}

	return result;
}
