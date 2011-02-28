#include "global.h"
#include "functions.h"
#include "timeRec.h"
#include <CL/opencl.h>

/*************************************************************
 **************** Version 1 **********************************
32. This version is based on version 27.
	1) Constant memory will be used for scoring matrix
	2) Input string are stored in constant memory, it is only used for trace back, 
	   not be used for matrix filling.
db1.This version is based on version 32, modified for database search
db2.Add time record in the program.
V2. opencl implementation of smith-waterman corresponding to version 32
V3. For set argument, unchanged arguments for kernel is set only once
**************************************************************/

#define CHECK_ERR(err, str) \
	if (err != CL_SUCCESS)  \
	{ \
		fprintf(stderr, "CL Error %d: %s\n", err, str); \
		exit(1); \
	} \

char * loadSource(char *filePathName, size_t *fileSize)
{
	FILE *pfile;
	size_t tmpFileSize;
	char *fileBuffer;
	pfile = fopen(filePathName, "rb");

	if (pfile == NULL)
	{
		printf("Open file %s open error!\n", filePathName);
		return NULL;
	}

	fseek(pfile, 0, SEEK_END);
	tmpFileSize = ftell(pfile);

	fileBuffer = (char *)malloc(tmpFileSize);

	fseek(pfile, 0, SEEK_SET);
	fread(fileBuffer, sizeof(char), tmpFileSize, pfile);

	fclose(pfile);

	//debug================================
	//for (int i = 0; i < tmpFileSize; i++)
	//{
	//	printf("%c", fileBuffer[i]);
	//}
	//=====================================

	*fileSize = tmpFileSize;
	return fileBuffer;
}

int main(int argc, char ** argv)
{
	if (argc < 3)
	{
		printf("Calculate similarities between two strings.\n");
		printf("Maximum length of each string is: %d\n", MAX_LEN);
		printf("Usage: %s query database [openPenalty extensionPenalty]\n", argv[0]);
		printf("openPenalty (5.0), extensionPenalty (0.5)\n");
		return 1;
	}

	char queryFilePathName[255], dbDataFilePathName[255], dbLenFilePathName[255];
	int querySize, subSequenceNum, subSequenceSize;
	float openPenalty, extensionPenalty;
	int coalescedOffset = COALESCED_OFFSET;
	int nblosumWidth = 23;
	int blockSize = 260;
	int blockNum;
	
	//record time
	memset(&strTime, 0, sizeof(STRUCT_TIME));
	//timerStart();

	openPenalty = 5.0f;
	extensionPenalty = 0.5;

	if (argc == 5)
	{
		openPenalty = atof(argv[3]);
		extensionPenalty = atof(argv[4]);
	}

	//for opencl initialization
	cl_int err;
	cl_platform_id platformID;
	cl_device_id deviceID;
	cl_context hContext;
	cl_command_queue hCmdQueue;
	cl_program hProgram;
	cl_kernel hMatchStringKernel, hTraceBackKernel;
	size_t sourceFileSize;
	char *cSourceCL = NULL;

	timerStart();
	err = clGetPlatformIDs(1, &platformID, NULL);
	CHECK_ERR(err, "Get platform ID error!");
	timerEnd();
	strTime.getPlatform = elapsedTime();
	strTime.numGetPlatform++;

	timerStart();
	err = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, NULL);
	CHECK_ERR(err, "Get device ID error!");
	timerEnd();
	strTime.getDeviceID = elapsedTime();
	strTime.numGetDeviceID++;
	
	timerStart();
	hContext = clCreateContext(0, 1, &deviceID, 0, 0, &err);
	CHECK_ERR(err, "Create context from type error");
	timerEnd();
	strTime.createContext = elapsedTime();
	strTime.numCreateContext++;
	
	timerStart();
	hCmdQueue = clCreateCommandQueue(hContext, deviceID, 0, &err);
	CHECK_ERR(err, "Create command queue error");
	timerEnd();
	strTime.createCommandQueue = elapsedTime();
	strTime.numCreateCommandQueue++;

	//load the source file
	cSourceCL = loadSource("kernels.cl", &sourceFileSize);

	timerStart();
	hProgram = clCreateProgramWithSource(hContext, 1, (const char **)&cSourceCL, 
				&sourceFileSize, &err);
	CHECK_ERR(err, "Create program with source error");
	timerEnd();
	strTime.createProgramWithSource = elapsedTime();
	strTime.numCreateProgramWithSource++;
	
	timerStart();
	err = clBuildProgram(hProgram, 0, 0, 0, 0, 0);
	CHECK_ERR(err, "Build program error");
	timerEnd();
	strTime.buildProgram = elapsedTime();
	strTime.numBuildProgram++;
//	//debug================================
//	int logSize = 3000, i;
//	size_t retSize;
//	char logTxt[3000];
//	err = clGetProgramBuildInfo(hProgram, deviceID, CL_PROGRAM_BUILD_LOG, logSize, logTxt, &retSize);
//	for (i = 0; i < retSize; i++)
//	{
//		printf("%c", logTxt[i]);
//	}
//	//===================================

	timerStart();
	hMatchStringKernel = clCreateKernel(hProgram, "MatchString", &err);
	CHECK_ERR(err, "Create MatchString kernel error");
	hTraceBackKernel = clCreateKernel(hProgram, "trace_back2", &err);
	CHECK_ERR(err, "Create trace_back2 kernel error");
	timerEnd();
	strTime.createKernel = elapsedTime();
	strTime.numCreateKernel += 2;

	sprintf(queryFilePathName, "%s", argv[1]);
	sprintf(dbDataFilePathName, "%s.data", argv[2]);
	sprintf(dbLenFilePathName, "%s.loc", argv[2]);

	char *allSequences, *querySequence, *subSequence;
	char *seq1, *seq2;
	cl_mem seq1D, seq2D;

	allSequences = new char[2 * (MAX_LEN)];
	if (allSequences == NULL)
	{
		printf("Allocate sequence buffer error!\n");
		return 1;
	}
	querySequence = allSequences;

	timerStart();
	seq1D = clCreateBuffer(hContext, CL_MEM_READ_ONLY, sizeof(cl_char) * MAX_LEN * 5, 0, &err);
	CHECK_ERR(err, "Create seq1D memory");
	seq2D = clCreateBuffer(hContext, CL_MEM_READ_ONLY, sizeof(cl_char) * MAX_LEN * 5, 0, &err);
	CHECK_ERR(err, "Create seq2D memory");
	timerEnd();
	strTime.createBuffer += elapsedTime();
	strTime.numCreateBuffer += 2;

	//read query sequence
	querySize = readQuerySequence(queryFilePathName, querySequence);
	if (querySize <= 0 || querySize > MAX_LEN)
	{
		printf("Query size %d is out of range (0, %d)\n",
				MAX_LEN,
				querySize);
		return 1;
	}
	encoding(querySequence, querySize);
	subSequence = allSequences + querySize;

	//allocate output sequence buffer
	char *outSeq1, *outSeq2;
	outSeq1 = new char[2 * MAX_LEN];
	outSeq2 = new char[2 * MAX_LEN];
	if (outSeq1 == NULL ||
		outSeq2 == NULL)
	{
		printf("Allocate output sequence buffer on host error!\n");
		return 1;
	}

	timerStart();
	cl_mem outSeq1D, outSeq2D;
	outSeq1D = clCreateBuffer(hContext, CL_MEM_READ_WRITE, sizeof(cl_char) * MAX_LEN * 2, 0, &err);
	CHECK_ERR(err, "Create outSeq1D memory");
	outSeq2D = clCreateBuffer(hContext, CL_MEM_READ_WRITE, sizeof(cl_char) * MAX_LEN * 2, 0, &err);
	CHECK_ERR(err, "Create outSeq2D memory");
	timerEnd();
	strTime.createBuffer += elapsedTime();
	strTime.numCreateBuffer += 2;

	//allocate thread number per launch and 
	//location difference information
	int *threadNum, *diffPos;
	threadNum = new int[2 * MAX_LEN];
	diffPos = new int[2 * MAX_LEN];
	if (threadNum == NULL ||
		diffPos == NULL)
	{
		printf("Allocate location buffer on host error!\n");
		return 1;
	}

	timerStart();
	cl_mem threadNumD, diffPosD;
	threadNumD = clCreateBuffer(hContext, CL_MEM_READ_ONLY, sizeof(cl_int) * (2 * MAX_LEN), 0, &err);
	CHECK_ERR(err, "Create threadNumD memory");
	diffPosD = clCreateBuffer(hContext, CL_MEM_READ_ONLY, sizeof(cl_int) * (2 * MAX_LEN), 0, &err);
	CHECK_ERR(err, "Create diffPosD memory");
	timerEnd();
	strTime.createBuffer += elapsedTime();
	strTime.numCreateBuffer += 2;

	//allocate matrix buffer
	char *pathFlag, *extFlag; 
	float *nGapDist, *hGapDist, *vGapDist;
	int maxElemNum = (MAX_LEN + 1) * (MAX_LEN + 1);
	pathFlag  = new char[maxElemNum];
	extFlag   = new char[maxElemNum];
	nGapDist = new float[maxElemNum];
	hGapDist = new float[maxElemNum];
	vGapDist = new float[maxElemNum];
	if (pathFlag  == NULL ||
		extFlag   == NULL ||
		nGapDist == NULL ||
		hGapDist == NULL ||
		vGapDist == NULL)
	{
		printf("Allocate DP matrices on host error!\n");
		return 1;
	}

	cl_mem pathFlagD, extFlagD,	nGapDistD, hGapDistD, vGapDistD;
	timerStart();
	pathFlagD = clCreateBuffer(hContext, CL_MEM_READ_WRITE, sizeof(cl_char) * maxElemNum, 0, &err);
	CHECK_ERR(err, "Create pathFlagD memory");
	extFlagD = clCreateBuffer(hContext,  CL_MEM_READ_WRITE, sizeof(cl_char) * maxElemNum, 0, &err);
	CHECK_ERR(err, "Create extFlagD memory");
	nGapDistD = clCreateBuffer(hContext, CL_MEM_READ_WRITE, sizeof(cl_float) * maxElemNum, 0, &err);
	CHECK_ERR(err, "Create nGapDistD memory");
	hGapDistD = clCreateBuffer(hContext, CL_MEM_READ_WRITE, sizeof(cl_float) * maxElemNum, 0, &err);
	CHECK_ERR(err, "Create hGapDistD memory");
	vGapDistD = clCreateBuffer(hContext, CL_MEM_READ_WRITE, sizeof(cl_float) * maxElemNum, 0, &err);
	CHECK_ERR(err, "Create vGapDistD memory");
	timerEnd();
	strTime.createBuffer += elapsedTime();
	strTime.numCreateBuffer += 5;

	//Allocate the MAX INFO structure
	MAX_INFO *maxInfo;
	maxInfo = new MAX_INFO[1];
	if (maxInfo == NULL)
	{
		printf("Alloate maxInfo on host error!\n");
		return 1;
	}
	
	cl_mem maxInfoD;
	timerStart();
	maxInfoD = clCreateBuffer(hContext, CL_MEM_READ_WRITE, sizeof(MAX_INFO), 0, &err);
	CHECK_ERR(err, "Create maxInfoD memory");
	timerEnd();
	strTime.createBuffer += elapsedTime();
	strTime.numCreateBuffer += 1;

	//allocate the distance table
	cl_mem blosum62D;
	int nblosumHeight = 23;
	timerStart();
	blosum62D = clCreateBuffer(hContext, CL_MEM_READ_ONLY, sizeof(cl_float) * nblosumWidth * nblosumHeight, 0, &err);
	timerEnd();
	strTime.createBuffer += elapsedTime();
	strTime.numCreateBuffer += 1;

	timerStart();
	err = clEnqueueWriteBuffer(hCmdQueue, blosum62D, CL_TRUE, 0,
							   nblosumWidth * nblosumHeight * sizeof(cl_float), blosum62[0], 0, NULL, NULL);
	CHECK_ERR(err, "copy blosum62 to device");
	timerEnd();
	strTime.enqueueWriteBuffer += elapsedTime();
	strTime.numEnqueueWriteBuffer++;


	//copy the scoring matrix to the constant memory
	//copyScoringMatrixToConstant();

	//open the database
	pDBDataFile = fopen(dbDataFilePathName, "rb");
	if (pDBDataFile == NULL)
	{
		printf("DB data file %s open error!\n", dbDataFilePathName);
		return 1;
	}

	pDBLenFile = fopen(dbLenFilePathName, "rb");
	if (pDBLenFile == NULL)
	{
		printf("DB length file %s open error!\n", dbLenFilePathName);
		return 1;
	}

	//record time
	//timerEnd();
	//strTime.iniTime = elapsedTime();

	//read the total number of sequences
	fread(&subSequenceNum, sizeof(int), 1, pDBLenFile);

	//get the larger and smaller of the row and colum number
	int subSequenceNo, launchNum, launchNo;
	int rowNum, columnNum, matrixIniNum;
	int DPMatrixSize;
	int seq1Pos, seq2Pos, nOffset, startPos;

	for (subSequenceNo = 0; subSequenceNo < subSequenceNum; subSequenceNo++)
	{
		//record time
		//timerStart();

		//read subject sequence
		fread(&subSequenceSize, sizeof(int), 1, pDBLenFile);
		if (subSequenceSize <= 0 || subSequenceSize > MAX_LEN)
		{
			printf("Size %d of bubject sequence %d is out of range!\n",
					subSequenceSize,
					subSequenceNo);
			break;
		}
		fread(subSequence, sizeof(char), subSequenceSize, pDBDataFile);

		if (subSequenceSize > querySize)
		{
			seq1 = subSequence;
			seq2 = querySequence;
			rowNum = subSequenceSize + 1;
			columnNum = querySize + 1;
		}
		else
		{
			seq1 = querySequence;
			seq2 = subSequence;
			rowNum = querySize + 1;
			columnNum = subSequenceSize + 1;
		}

		launchNum = rowNum + columnNum - 1;

		//preprocessing for sequences
		DPMatrixSize = preProcessing(rowNum,
					  columnNum,
					  threadNum,
					  diffPos,
					  matrixIniNum);

		//record time
		//timerEnd();
		//strTime.preprocessingTime += elapsedTime();

		//record time
		//timerStart();

		//Initialize DP matrices
		memset(pathFlag, 0, DPMatrixSize * sizeof(char));
		memset(extFlag,  0, DPMatrixSize * sizeof(char));
		memset(nGapDist, 0, matrixIniNum * sizeof(float));
		memset(hGapDist, 0, matrixIniNum * sizeof(float));
		memset(vGapDist, 0, matrixIniNum * sizeof(float));
		memset(maxInfo,  0, sizeof(MAX_INFO));
		timerStart();
		err  = clEnqueueWriteBuffer(hCmdQueue, pathFlagD, CL_TRUE, 0, DPMatrixSize * sizeof(cl_char), pathFlag, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(hCmdQueue, extFlagD,  CL_TRUE, 0, DPMatrixSize * sizeof(cl_char), extFlag,  0, NULL, NULL);
		err |= clEnqueueWriteBuffer(hCmdQueue, nGapDistD, CL_TRUE, 0, matrixIniNum * sizeof(cl_float), nGapDist, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(hCmdQueue, hGapDistD, CL_TRUE, 0, matrixIniNum * sizeof(cl_float), hGapDist, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(hCmdQueue, vGapDistD, CL_TRUE, 0, matrixIniNum * sizeof(cl_float), vGapDist, 0, NULL, NULL);
		err != clEnqueueWriteBuffer(hCmdQueue, maxInfoD, CL_TRUE, 0, sizeof(MAX_INFO), maxInfo, 0, NULL, NULL);
		CHECK_ERR(err, "copy DP matrix");

		//copy input sequences to device
		err  = clEnqueueWriteBuffer(hCmdQueue, seq1D, CL_TRUE, 0, (rowNum - 1) * sizeof(cl_char), seq1, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(hCmdQueue, seq2D, CL_TRUE, 0, (columnNum - 1) * sizeof(cl_char), seq2, 0, NULL, NULL);
		CHECK_ERR(err, "copy input sequence");

		err  = clEnqueueWriteBuffer(hCmdQueue, diffPosD, CL_TRUE, 0, launchNum * sizeof(cl_int), diffPos, 0, NULL, NULL);
		CHECK_ERR(err, "copy diffpos info");
		clFinish(hCmdQueue);
		timerEnd();
		strTime.enqueueWriteBuffer += elapsedTime();
		strTime.numEnqueueWriteBuffer += 9;
		//record time
		//timerEnd();
		//strTime.copyTimeHostToDevice += elapsedTime();
		timerStart();
		err  = clSetKernelArg(hMatchStringKernel, 0, sizeof(cl_mem), (void *)&pathFlagD);
		err |= clSetKernelArg(hMatchStringKernel, 1, sizeof(cl_mem), (void *)&extFlagD);
		err |= clSetKernelArg(hMatchStringKernel, 2, sizeof(cl_mem), (void *)&nGapDistD);
		err |= clSetKernelArg(hMatchStringKernel, 3, sizeof(cl_mem), (void *)&hGapDistD);
		err |= clSetKernelArg(hMatchStringKernel, 4, sizeof(cl_mem), (void *)&vGapDistD);
		err |= clSetKernelArg(hMatchStringKernel, 11, sizeof(cl_mem), (void *)&seq1D);
		err |= clSetKernelArg(hMatchStringKernel, 12, sizeof(cl_mem), (void *)&seq2D);	
		err != clSetKernelArg(hMatchStringKernel, 14, sizeof(cl_float), (void *)&openPenalty);
		err != clSetKernelArg(hMatchStringKernel, 15, sizeof(cl_float), (void *)&extensionPenalty);
		err != clSetKernelArg(hMatchStringKernel, 16, sizeof(cl_mem), (void *)&maxInfoD);
		err != clSetKernelArg(hMatchStringKernel, 17, sizeof(cl_mem), (void *)&blosum62D);
		CHECK_ERR(err, "Set match string argument error!");
		timerEnd();
		strTime.setKernelArg += elapsedTime();
		strTime.numSetKernelArg += 11;
		
		seq1Pos = -1;
		seq2Pos = 0;
		nOffset = 0;
		startPos = 2 * coalescedOffset; 
		for (launchNo = 2; launchNo < launchNum; launchNo++)
		{
			if (launchNo < rowNum)
			{
				seq1Pos++;
			}
			else if (launchNo == rowNum)
			{
				seq1Pos++;
				nOffset = 1;
			}
			else
			{
				seq2Pos++;
			}

			size_t dimBlock[1], dimGrid[1];
			dimBlock[0] = blockSize;

			blockNum = (threadNum[launchNo] - 1)/blockSize + 1;
			dimGrid[0] = blockNum * dimBlock[0];

			//set arguments
			timerStart();
			err  = clSetKernelArg(hMatchStringKernel, 5, sizeof(cl_int), (void *)&startPos);
			err != clSetKernelArg(hMatchStringKernel, 6, sizeof(cl_int), (void *)&seq1Pos);
			err != clSetKernelArg(hMatchStringKernel, 7, sizeof(cl_int), (void *)&seq2Pos);
			err != clSetKernelArg(hMatchStringKernel, 8, sizeof(cl_int), (void *)&diffPos[launchNo - 1]);
			err != clSetKernelArg(hMatchStringKernel, 9, sizeof(cl_int), (void *)&diffPos[launchNo]);
			err != clSetKernelArg(hMatchStringKernel, 10, sizeof(cl_int), (void *)&threadNum[launchNo]);
			err != clSetKernelArg(hMatchStringKernel, 13, sizeof(cl_int), (void *)&nblosumWidth);
			CHECK_ERR(err, "Set match string argument error!");
			timerEnd();
			strTime.setKernelArg += elapsedTime();
			strTime.numSetKernelArg += 7;

			timerStart();
			err = clEnqueueNDRangeKernel(hCmdQueue, hMatchStringKernel, 1, NULL, dimGrid,
										 dimBlock, 0, NULL, NULL);
			CHECK_ERR(err, "Launch kernel match string error");
			clFinish(hCmdQueue);
			timerEnd();
			strTime.kernelExecution += elapsedTime();
			strTime.numKernelExecution++;

			//start position for next kernel launch
			startPos += diffPos[launchNo + 1] + nOffset;
		}
		
		//clFinish(hCmdQueue);
		//record time

		//record time
		timerStart();
		err  = clSetKernelArg(hTraceBackKernel, 0, sizeof(cl_mem), (void *)&pathFlagD);
		err |= clSetKernelArg(hTraceBackKernel, 1, sizeof(cl_mem), (void *)&extFlagD);
		err != clSetKernelArg(hTraceBackKernel, 2, sizeof(cl_mem), (void *)&diffPosD);
		err |= clSetKernelArg(hTraceBackKernel, 3, sizeof(cl_mem), (void *)&seq1D);
		err |= clSetKernelArg(hTraceBackKernel, 4, sizeof(cl_mem), (void *)&seq2D);	
		err |= clSetKernelArg(hTraceBackKernel, 5, sizeof(cl_mem), (void *)&outSeq1D);
		err |= clSetKernelArg(hTraceBackKernel, 6, sizeof(cl_mem), (void *)&outSeq2D);	
		err != clSetKernelArg(hTraceBackKernel, 7, sizeof(cl_mem), (void *)&maxInfoD);
		timerEnd();
		strTime.setKernelArg += elapsedTime();
		strTime.numSetKernelArg += 8;
		
		size_t tbGlobalSize[1] = {1};
		size_t tbLocalSize[1]  = {1};
		timerStart();
		err = clEnqueueNDRangeKernel(hCmdQueue, hTraceBackKernel, 1, NULL, tbGlobalSize,
									 tbLocalSize, 0, NULL, NULL);
		CHECK_ERR(err, "Launch kernel trace back error");
		clFinish(hCmdQueue);
		//record time
		timerEnd();
		strTime.kernelExecution += elapsedTime();
		strTime.numKernelExecution++;

		//record time
		timerStart();
		//copy matrix score structure back
		err = clEnqueueReadBuffer(hCmdQueue, maxInfoD, CL_TRUE, 0, sizeof(MAX_INFO),
								  maxInfo, 0, 0, 0);
		CHECK_ERR(err, "Read maxInfo buffer error!");

		int nlength = maxInfo->noutputlen;
		err  = clEnqueueReadBuffer(hCmdQueue, outSeq1D, CL_TRUE, 0, nlength * sizeof(cl_char),
								   outSeq1, 0, 0, 0);
		err != clEnqueueReadBuffer(hCmdQueue, outSeq2D, CL_TRUE, 0, nlength * sizeof(cl_char),
								   outSeq2, 0, 0, 0);
		CHECK_ERR(err, "Read output sequence error!");
		//record time
		timerEnd();
		strTime.enqueueReadBuffer = elapsedTime();
		strTime.numEnqueueReadBuffer += 3;

		//call the print function to print the match result
		printf("============================================================\n");
		printf("Sequence pair %d:\n", subSequenceNo);
		PrintAlignment(outSeq1, outSeq2, nlength, CHAR_PER_LINE, openPenalty, extensionPenalty);
		printf("Max alignment score (on device) is %.1f\n", maxInfo->fmaxscore);
		//obtain max alignment score on host
		err = clEnqueueReadBuffer(hCmdQueue, nGapDistD, CL_TRUE, 0, sizeof(cl_float) * DPMatrixSize,
								  nGapDist, 0, 0, 0);
		//cudaMemcpy(nGapDist, nGapDistD, sizeof(float) * DPMatrixSize, cudaMemcpyDeviceToHost);
		//printf("Max alignment score (on host) is %.1f\n", maxScore(nGapDist, DPMatrixSize));

		printf("openPenalty = %.1f, extensionPenalty = %.1f\n", openPenalty, extensionPenalty);
		printf("Input sequence size, querySize: %d, subSequenceSize: %d\n", 
				querySize, subSequenceSize);

		printf("Max position, seq1 = %d, seq2 = %d\n", maxInfo->nposi, maxInfo->nposj);
	}
	//print time
	fclose(pDBLenFile);
	fclose(pDBDataFile);


	timerStart();
	clReleaseProgram(hProgram);
	timerEnd();
	strTime.releaseProgram += elapsedTime();
	strTime.numReleaseProgram++;

	timerStart();
	clReleaseCommandQueue(hCmdQueue);
	timerEnd();
	strTime.releaseCmdQueue += elapsedTime();
	strTime.numReleaseCmdQueue++;

	timerStart();
	clReleaseContext(hContext);
	timerEnd();
	strTime.releaseContext += elapsedTime();
	strTime.numReleaseContext++;

	delete allSequences;

	delete outSeq1;
	delete outSeq2;

	delete threadNum;
	delete diffPos;

	delete pathFlag;
	delete extFlag;
	delete nGapDist;
	delete hGapDist;
	delete vGapDist;
	delete maxInfo;
	free(cSourceCL);

	timerStart();
	clReleaseMemObject(seq1D);
	clReleaseMemObject(seq2D);
	clReleaseMemObject(outSeq1D);
	clReleaseMemObject(outSeq2D);
	clReleaseMemObject(threadNumD);
	clReleaseMemObject(diffPosD);
	clReleaseMemObject(pathFlagD);
	clReleaseMemObject(extFlagD);
	clReleaseMemObject(nGapDistD);
	clReleaseMemObject(hGapDistD);
	clReleaseMemObject(vGapDistD);
	clReleaseMemObject(maxInfoD);
	clReleaseMemObject(blosum62D);
	timerEnd();
	strTime.releaseMemObj += elapsedTime();
	strTime.numReleaseMemObj++; 
	
	timerStart();
	clReleaseKernel(hMatchStringKernel);
	clReleaseKernel(hTraceBackKernel);
	timerEnd();
	strTime.releaseKernel += elapsedTime();
	strTime.numReleaseKernel += 2;
	
	printTime_toStandardOutput();
	printTime_toFile();


	return 0;
}

