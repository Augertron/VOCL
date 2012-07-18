#include "global.h"
#include "functions.h"
#include "swat_timer.h"
#include <CL/opencl.h>
#include <sched.h>

/* if enable vocl reablance, include these files */
#if VOCL_BALANCE 
#include <dlfcn.h>
#include "vocl.h"
#include "mpi.h"
#endif

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
		printf("Usage: %s query database [openPenalty extensionPenalty #ofIterations deviceNo]\n", argv[0]);
		printf("       deviceNo = -1 indicate using all virtual GPUs\n");
		printf("openPenalty (5.0), extensionPenalty (0.5)\n");
		return 1;
	}

#if VOCL_BALANCE
	MPI_Init(&argc, &argv);
#endif

	char queryFilePathName[255], dbDataFilePathName[255], dbLenFilePathName[255];
	int querySize, subSequenceNum, subSequenceSize, iterNo, i, index;
	float openPenalty, extensionPenalty;
	int numIterations = 20;
	int coalescedOffset = COALESCED_OFFSET;
	int nblosumWidth = 23;
	int blockSize = 260;
	cl_uint deviceNo = 0;
  	char kernel_source[KERNEL_SOURCE_FILE_LEN];
	int blockNum;

#if VOCL_BALANCE
	int rankNo;
	void *voclModulePtr;
	const char *error;
	dlVOCLRebalance dlvbPtr;
	MPI_Comm_rank(MPI_COMM_WORLD, &rankNo);
#endif

	cpu_set_t set;
	CPU_ZERO(&set);

	memset(&strTime, 0, sizeof(STRUCT_TIME));

	openPenalty = 5.0f;
	extensionPenalty = 0.5;

	if (argc == 7)
	{
		openPenalty = atof(argv[3]);
		extensionPenalty = atof(argv[4]);
		numIterations = atoi(argv[5]);
		deviceNo = atoi(argv[6]);
	}

	cl_int err;
	cl_uint platformNum, totalDeviceNum, usedDeviceNum, *deviceNums;
	cl_platform_id *platformIDs;
	cl_device_id *deviceIDs;
	cl_context *hContexts;
	cl_command_queue *hCmdQueues;
	cl_program *hPrograms;
	cl_kernel *hMatchStringKernels, *hTraceBackKernels;
	size_t sourceFileSize;
	char *cSourceCL = NULL;

	timerStart();
	err  = clGetPlatformIDs(0, NULL, &platformNum);
	platformIDs = (cl_platform_id *)malloc(sizeof(cl_platform_id) * platformNum);
	deviceNums = (cl_uint *)malloc(sizeof(cl_uint) * platformNum);
	err |= clGetPlatformIDs(platformNum, platformIDs, NULL);
	CHECK_ERR(err, "Get platform ID error!");
	timerEnd();
	strTime.getPlatform = elapsedTime();

	sched_getaffinity(0, sizeof(cpu_set_t), &set);
	printf("cpuid = %d\n", set.__bits[0]);

	timerStart();
	totalDeviceNum = 0;
	for (i = 0; i < platformNum; i++)
	{
		err  = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceNums[i]);
		CHECK_ERR(err, "Get device ID error!");
		totalDeviceNum += deviceNums[i];
	}

	deviceIDs = (cl_device_id *)malloc(sizeof(cl_device_id) * totalDeviceNum);
	hContexts = (cl_context *)malloc(sizeof(cl_context) * totalDeviceNum);
	hCmdQueues = (cl_command_queue *)malloc(sizeof(cl_command_queue) * totalDeviceNum);
	hPrograms = (cl_program *)malloc(sizeof(cl_program) * totalDeviceNum);
	hMatchStringKernels = (cl_kernel *)malloc(sizeof(cl_kernel) * totalDeviceNum);
	hTraceBackKernels = (cl_kernel *)malloc(sizeof(cl_kernel) * totalDeviceNum);

	totalDeviceNum = 0;
	for (i = 0; i < platformNum; i++)
	{
		err = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_GPU, deviceNums[i], &deviceIDs[totalDeviceNum], NULL);
		CHECK_ERR(err, "Get device ID error!");
		totalDeviceNum += deviceNums[i];
	}
	timerEnd();
	strTime.getDeviceID = elapsedTime();
	
	if (deviceNo == -1)
	{
		printf("All virtual GPUs are used..., deviceCount = %d\n\n", totalDeviceNum);
		usedDeviceNum = totalDeviceNum;
		deviceNo = 0;
	}
	else
	{
		printf("Virtual GPU %d is used...\n\n", deviceNo);
		usedDeviceNum = 1;
	}


	timerStart();
	for (i = 0; i < usedDeviceNum; i++)
	{
		index = i + deviceNo;
		hContexts[i] = clCreateContext(0, 1, &deviceIDs[index], 0, 0, &err);
		CHECK_ERR(err, "Create context from type error");
	}
	timerEnd();
	strTime.createContext = elapsedTime();
	
	timerStart();
	for (i = 0; i < usedDeviceNum; i++)
	{
		index = i + deviceNo;
		hCmdQueues[i] = clCreateCommandQueue(hContexts[i], deviceIDs[index], 0, &err);
		CHECK_ERR(err, "Create command queue error");
	}
	timerEnd();
	strTime.createCommandQueue = elapsedTime();

	//load the source file
	snprintf(kernel_source, KERNEL_SOURCE_FILE_LEN, "%s/examples/swat/kernels.cl",
		 ABS_SRCDIR);
	cSourceCL = loadSource(kernel_source, &sourceFileSize);

	timerStart();
	for (i = 0; i < usedDeviceNum; i++)
	{
		hPrograms[i] = clCreateProgramWithSource(hContexts[i], 1, (const char **)&cSourceCL, 
							&sourceFileSize, &err);
		CHECK_ERR(err, "Create program with source error");
	}
	timerEnd();
	strTime.createProgramWithSource = elapsedTime();
	
	timerStart();
	for (i = 0; i < usedDeviceNum; i++)
	{
		err = clBuildProgram(hPrograms[i], 0, 0, 0, 0, 0);
		CHECK_ERR(err, "Build program error");
	}
	timerEnd();
	strTime.buildProgram = elapsedTime();

	sprintf(queryFilePathName, "%s", argv[1]);
	sprintf(dbDataFilePathName, "%s.data", argv[2]);
	sprintf(dbLenFilePathName, "%s.loc", argv[2]);

	char *allSequences, *querySequence, *subSequence;
	char *seq1, *seq2;

	allSequences = new char[2 * (MAX_LEN)];
	if (allSequences == NULL)
	{
		printf("Allocate sequence buffer error!\n");
		return 1;
	}
	querySequence = allSequences;

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

	//Allocate the MAX INFO structure
	MAX_INFO *maxInfo;
	maxInfo = new MAX_INFO[1];
	if (maxInfo == NULL)
	{
		printf("Alloate maxInfo on host error!\n");
		return 1;
	}
	
	int nblosumHeight = 23;

	//create kernels for filling matrix and trace back
	timerStart();
	for (i = 0; i < usedDeviceNum; i++)
	{
		hMatchStringKernels[i] = clCreateKernel(hPrograms[i], "MatchString", &err);
		CHECK_ERR(err, "Create MatchString kernel error");
		hTraceBackKernels[i] = clCreateKernel(hPrograms[i], "trace_back2", &err);
		CHECK_ERR(err, "Create trace_back2 kernel error");
	}
	timerEnd();
	strTime.createKernel += elapsedTime();

	cl_mem *diffPosDs, *maxInfoDs, *blosum62Ds, *seq1Ds, *seq2Ds, *outSeq1Ds, *outSeq2Ds;
	cl_mem *pathFlagDs, *extFlagDs, *nGapDistDs, *hGapDistDs, *vGapDistDs;

	diffPosDs = (cl_mem *)malloc(sizeof(cl_mem) * usedDeviceNum);
	maxInfoDs = (cl_mem *)malloc(sizeof(cl_mem) * usedDeviceNum);
	blosum62Ds = (cl_mem *)malloc(sizeof(cl_mem) * usedDeviceNum);
	seq1Ds = (cl_mem *)malloc(sizeof(cl_mem) * usedDeviceNum);
	seq2Ds = (cl_mem *)malloc(sizeof(cl_mem) * usedDeviceNum);
	outSeq1Ds = (cl_mem *)malloc(sizeof(cl_mem) * usedDeviceNum);
	outSeq2Ds = (cl_mem *)malloc(sizeof(cl_mem) * usedDeviceNum);
	pathFlagDs = (cl_mem *)malloc(sizeof(cl_mem) * usedDeviceNum);
	extFlagDs = (cl_mem *)malloc(sizeof(cl_mem) * usedDeviceNum);
	nGapDistDs = (cl_mem *)malloc(sizeof(cl_mem) * usedDeviceNum);
	hGapDistDs = (cl_mem *)malloc(sizeof(cl_mem) * usedDeviceNum);
	vGapDistDs = (cl_mem *)malloc(sizeof(cl_mem) * usedDeviceNum);

	int DPMatrixSize;
	timerStart();
	for (i = 0; i < usedDeviceNum; i++)
	{
		seq1Ds[i] = clCreateBuffer(hContexts[i], CL_MEM_READ_ONLY, sizeof(cl_char) * MAX_LEN, 0, &err);
		CHECK_ERR(err, "Create seq1D memory");
		seq2Ds[i] = clCreateBuffer(hContexts[i], CL_MEM_READ_ONLY, sizeof(cl_char) * MAX_LEN, 0, &err);
		CHECK_ERR(err, "Create seq2D memory");

		outSeq1Ds[i] = clCreateBuffer(hContexts[i], CL_MEM_READ_WRITE, sizeof(cl_char) * MAX_LEN * 2, 0, &err);
		CHECK_ERR(err, "Create outSeq1D memory");
		outSeq2Ds[i] = clCreateBuffer(hContexts[i], CL_MEM_READ_WRITE, sizeof(cl_char) * MAX_LEN * 2, 0, &err);
		CHECK_ERR(err, "Create outSeq2D memory");

		diffPosDs[i] = clCreateBuffer(hContexts[i], CL_MEM_READ_ONLY, sizeof(cl_int) * MAX_LEN * 2, 0, &err);
		CHECK_ERR(err, "Create diffPosD memory");

		DPMatrixSize = MAX_LEN * MAX_LEN;
		pathFlagDs[i] = clCreateBuffer(hContexts[i], CL_MEM_READ_WRITE, sizeof(cl_char) * DPMatrixSize, 0, &err);
		CHECK_ERR(err, "Create pathFlagD memory");
		extFlagDs[i] = clCreateBuffer(hContexts[i],  CL_MEM_READ_WRITE, sizeof(cl_char) * DPMatrixSize, 0, &err);
		CHECK_ERR(err, "Create extFlagD memory");
		nGapDistDs[i] = clCreateBuffer(hContexts[i], CL_MEM_READ_WRITE, sizeof(cl_float) * DPMatrixSize, 0, &err);
		CHECK_ERR(err, "Create nGapDistD memory");
		hGapDistDs[i] = clCreateBuffer(hContexts[i], CL_MEM_READ_WRITE, sizeof(cl_float) * DPMatrixSize, 0, &err);
		CHECK_ERR(err, "Create hGapDistD memory");
		vGapDistDs[i] = clCreateBuffer(hContexts[i], CL_MEM_READ_WRITE, sizeof(cl_float) * DPMatrixSize, 0, &err);
		CHECK_ERR(err, "Create vGapDistD memory");

		maxInfoDs[i] = clCreateBuffer(hContexts[i], CL_MEM_READ_WRITE, sizeof(MAX_INFO), 0, &err);
		CHECK_ERR(err, "Create maxInfoD memory");

		blosum62Ds[i] = clCreateBuffer(hContexts[i], CL_MEM_READ_ONLY, sizeof(cl_float) * nblosumWidth * nblosumHeight, 0, &err);
		CHECK_ERR(err, "Create scoring matrix memory");
	}
	timerEnd();
	strTime.createBuffer += elapsedTime();

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

	//read the total number of sequences
	fread(&subSequenceNum, sizeof(int), 1, pDBLenFile);

#if VOCL_BALANCE
	voclModulePtr = dlopen("libvocl.so", RTLD_NOW);
	if (voclModulePtr == NULL)
	{
		printf("open libvocl.so error, %s\n", dlerror());
		exit (1); 
	}

	dlvbPtr = (dlVOCLRebalance)dlsym(voclModulePtr, "voclRebalance");
	if (error = dlerror()) {
		printf("Could find voclRebalance: %s\n", error);
		exit(1);
	}
#endif

	//get the larger and smaller of the row and colum number
	int subSequenceNo, launchNum, launchNo;
	int rowNum, columnNum, matrixIniNum;
	int seq1Pos, seq2Pos, nOffset, startPos;

	for (subSequenceNo = 0; subSequenceNo < subSequenceNum; subSequenceNo++)
	{
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
					  
		timerStart();
		for (iterNo = 0; iterNo < numIterations; iterNo++)
		{
			//Initialize DP matrices
			memset(pathFlag, 0, DPMatrixSize * sizeof(char));
			memset(extFlag,  0, DPMatrixSize * sizeof(char));
			memset(nGapDist, 0, matrixIniNum * sizeof(float));
			memset(hGapDist, 0, matrixIniNum * sizeof(float));
			memset(vGapDist, 0, matrixIniNum * sizeof(float));
			memset(maxInfo,  0, sizeof(MAX_INFO));

			for (i = 0; i < usedDeviceNum; i++)
			{
				err  = clEnqueueWriteBuffer(hCmdQueues[i], pathFlagDs[i], CL_FALSE, 0, DPMatrixSize * sizeof(cl_char), pathFlag, 0, NULL, NULL);
				err |= clEnqueueWriteBuffer(hCmdQueues[i], extFlagDs[i],  CL_FALSE, 0, DPMatrixSize * sizeof(cl_char), extFlag,  0, NULL, NULL);
				err |= clEnqueueWriteBuffer(hCmdQueues[i], nGapDistDs[i], CL_FALSE, 0, matrixIniNum * sizeof(cl_float), nGapDist, 0, NULL, NULL);
				err |= clEnqueueWriteBuffer(hCmdQueues[i], hGapDistDs[i], CL_FALSE, 0, matrixIniNum * sizeof(cl_float), hGapDist, 0, NULL, NULL);
				err |= clEnqueueWriteBuffer(hCmdQueues[i], vGapDistDs[i], CL_FALSE, 0, matrixIniNum * sizeof(cl_float), vGapDist, 0, NULL, NULL);
				err != clEnqueueWriteBuffer(hCmdQueues[i], maxInfoDs[i], CL_FALSE, 0, sizeof(MAX_INFO), maxInfo, 0, NULL, NULL);
				CHECK_ERR(err, "copy DP matrix");

				//copy input sequences to device
				err  = clEnqueueWriteBuffer(hCmdQueues[i], seq1Ds[i], CL_FALSE, 0, (rowNum - 1) * sizeof(cl_char), seq1, 0, NULL, NULL);
				err |= clEnqueueWriteBuffer(hCmdQueues[i], seq2Ds[i], CL_FALSE, 0, (columnNum - 1) * sizeof(cl_char), seq2, 0, NULL, NULL);
				CHECK_ERR(err, "copy input sequence");

				err  = clEnqueueWriteBuffer(hCmdQueues[i], diffPosDs[i], CL_FALSE, 0, launchNum * sizeof(cl_int), diffPos, 0, NULL, NULL);
				CHECK_ERR(err, "copy diffpos info");
				err = clEnqueueWriteBuffer(hCmdQueues[i], blosum62Ds[i], CL_FALSE, 0,
										   nblosumWidth * nblosumHeight * sizeof(cl_float), blosum62[0], 0, NULL, NULL);
				CHECK_ERR(err, "copy blosum62 to device");
			}

			for (i = 0; i < usedDeviceNum; i++)
			{
				err  = clSetKernelArg(hMatchStringKernels[i], 0, sizeof(cl_mem), (void *)&pathFlagDs[i]);
				err |= clSetKernelArg(hMatchStringKernels[i], 1, sizeof(cl_mem), (void *)&extFlagDs[i]);
				err |= clSetKernelArg(hMatchStringKernels[i], 2, sizeof(cl_mem), (void *)&nGapDistDs[i]);
				err |= clSetKernelArg(hMatchStringKernels[i], 3, sizeof(cl_mem), (void *)&hGapDistDs[i]);
				err |= clSetKernelArg(hMatchStringKernels[i], 4, sizeof(cl_mem), (void *)&vGapDistDs[i]);
				err |= clSetKernelArg(hMatchStringKernels[i], 11, sizeof(cl_mem), (void *)&seq1Ds[i]);
				err |= clSetKernelArg(hMatchStringKernels[i], 12, sizeof(cl_mem), (void *)&seq2Ds[i]);	
				err != clSetKernelArg(hMatchStringKernels[i], 14, sizeof(cl_float), (void *)&openPenalty);
				err != clSetKernelArg(hMatchStringKernels[i], 15, sizeof(cl_float), (void *)&extensionPenalty);
				err != clSetKernelArg(hMatchStringKernels[i], 16, sizeof(cl_mem), (void *)&maxInfoDs[i]);
				err != clSetKernelArg(hMatchStringKernels[i], 17, sizeof(cl_mem), (void *)&blosum62Ds[i]);
				CHECK_ERR(err, "Set match string argument error!");
				
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
					err  = clSetKernelArg(hMatchStringKernels[i], 5, sizeof(cl_int), (void *)&startPos);
					err != clSetKernelArg(hMatchStringKernels[i], 6, sizeof(cl_int), (void *)&seq1Pos);
					err != clSetKernelArg(hMatchStringKernels[i], 7, sizeof(cl_int), (void *)&seq2Pos);
					err != clSetKernelArg(hMatchStringKernels[i], 8, sizeof(cl_int), (void *)&diffPos[launchNo - 1]);
					err != clSetKernelArg(hMatchStringKernels[i], 9, sizeof(cl_int), (void *)&diffPos[launchNo]);
					err != clSetKernelArg(hMatchStringKernels[i], 10, sizeof(cl_int), (void *)&threadNum[launchNo]);
					err != clSetKernelArg(hMatchStringKernels[i], 13, sizeof(cl_int), (void *)&nblosumWidth);
					CHECK_ERR(err, "Set match string argument error!");

					err = clEnqueueNDRangeKernel(hCmdQueues[i], hMatchStringKernels[i], 1, NULL, dimGrid,
												 dimBlock, 0, NULL, NULL);
					CHECK_ERR(err, "Launch kernel match string error");

					//start position for next kernel launch
					startPos += diffPos[launchNo + 1] + nOffset;
				}
			}

			for (i = 0; i < usedDeviceNum; i++)
			{
				//record time
				err  = clSetKernelArg(hTraceBackKernels[i], 0, sizeof(cl_mem), (void *)&pathFlagDs[i]);
				err |= clSetKernelArg(hTraceBackKernels[i], 1, sizeof(cl_mem), (void *)&extFlagDs[i]);
				err != clSetKernelArg(hTraceBackKernels[i], 2, sizeof(cl_mem), (void *)&diffPosDs[i]);
				err |= clSetKernelArg(hTraceBackKernels[i], 3, sizeof(cl_mem), (void *)&seq1Ds[i]);
				err |= clSetKernelArg(hTraceBackKernels[i], 4, sizeof(cl_mem), (void *)&seq2Ds[i]);	
				err |= clSetKernelArg(hTraceBackKernels[i], 5, sizeof(cl_mem), (void *)&outSeq1Ds[i]);
				err |= clSetKernelArg(hTraceBackKernels[i], 6, sizeof(cl_mem), (void *)&outSeq2Ds[i]);	
				err != clSetKernelArg(hTraceBackKernels[i], 7, sizeof(cl_mem), (void *)&maxInfoDs[i]);
				
				size_t tbGlobalSize[1] = {1};
				size_t tbLocalSize[1]  = {1};
				err = clEnqueueNDRangeKernel(hCmdQueues[i], hTraceBackKernels[i], 1, NULL, tbGlobalSize,
											 tbLocalSize, 0, NULL, NULL);
				CHECK_ERR(err, "Launch kernel trace back error");
			}

			for (i = 0; i < usedDeviceNum; i++)
			{
				//copy matrix score structure back
				err = clEnqueueReadBuffer(hCmdQueues[i], maxInfoDs[i], CL_FALSE, 0, sizeof(MAX_INFO),
										  maxInfo, 0, 0, 0);
				CHECK_ERR(err, "Read maxInfo buffer error!");

				err  = clEnqueueReadBuffer(hCmdQueues[i], outSeq1Ds[i], CL_FALSE, 0, (rowNum + columnNum) * sizeof(cl_char),
										   outSeq1, 0, 0, 0);
				err != clEnqueueReadBuffer(hCmdQueues[i], outSeq2Ds[i], CL_FALSE, 0, (rowNum + columnNum) * sizeof(cl_char),
										   outSeq2, 0, 0, 0);
				CHECK_ERR(err, "Read output sequence error!");
			}

#if VOCL_BALANCE
			if (rankNo == 0 && iterNo == 1)
			{
				for (i = 0; i  < usedDeviceNum; i++)
				{
					(*dlvbPtr)(hCmdQueues[i]);
				}
			}
#endif
		}

		for (i = 0; i < usedDeviceNum; i++)
		{
			clFinish(hCmdQueues[i]);
		}
		//record time
		timerEnd();
		strTime.kernelExecution += elapsedTime();
		
		int nlength = maxInfo->noutputlen;
		//call the print function to print the match result
		printf("============================================================\n");
		printf("Sequence pair %d:\n", subSequenceNo);
		//PrintAlignment(outSeq1, outSeq2, nlength, CHAR_PER_LINE, openPenalty, extensionPenalty);
		printf("Max alignment score (on device) is %.1f\n", maxInfo->fmaxscore);

		printf("openPenalty = %.1f, extensionPenalty = %.1f\n", openPenalty, extensionPenalty);
		printf("Input sequence size, querySize: %d, subSequenceSize: %d\n", 
				querySize, subSequenceSize);

		printf("Max position, seq1 = %d, seq2 = %d\n", maxInfo->nposi, maxInfo->nposj);
	}
	//print time
	fclose(pDBLenFile);
	fclose(pDBDataFile);

	timerStart();
	for (i = 0; i < usedDeviceNum; i++)
	{
		clReleaseMemObject(seq1Ds[i]);
		clReleaseMemObject(seq2Ds[i]);
		clReleaseMemObject(outSeq1Ds[i]);
		clReleaseMemObject(outSeq2Ds[i]);
		clReleaseMemObject(diffPosDs[i]);
		clReleaseMemObject(pathFlagDs[i]);
		clReleaseMemObject(extFlagDs[i]);
		clReleaseMemObject(nGapDistDs[i]);
		clReleaseMemObject(hGapDistDs[i]);
		clReleaseMemObject(vGapDistDs[i]);
		clReleaseMemObject(maxInfoDs[i]);
		clReleaseMemObject(blosum62Ds[i]);
	}
	timerEnd();
	strTime.releaseMemObj += elapsedTime();
	
	timerStart();
	for (i = 0; i < usedDeviceNum; i++)
	{
		clReleaseKernel(hMatchStringKernels[i]);
		clReleaseKernel(hTraceBackKernels[i]);
	}
	timerEnd();
	strTime.releaseKernel += elapsedTime();

	timerStart();
	for (i = 0; i < usedDeviceNum; i++)
	{
		clReleaseProgram(hPrograms[i]);
	}
	timerEnd();
	strTime.releaseProgram += elapsedTime();

	timerStart();
	for (i = 0; i < usedDeviceNum; i++)
	{
		clReleaseCommandQueue(hCmdQueues[i]);
	}
	timerEnd();
	strTime.releaseCmdQueue += elapsedTime();

	timerStart();
	for (i = 0; i < usedDeviceNum; i++)
	{
		clReleaseContext(hContexts[i]);
	}
	timerEnd();
	strTime.releaseContext += elapsedTime();

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

	free(deviceIDs);
	free(hContexts);
	free(hCmdQueues);
	free(hPrograms);
	free(hMatchStringKernels);
	free(hTraceBackKernels);
	free(platformIDs);
	free(deviceNums);
	free(diffPosDs);
	free(maxInfoDs);
	free(blosum62Ds);
	free(seq1Ds);
	free(seq2Ds);
	free(outSeq1Ds);
	free(outSeq2Ds);
	free(pathFlagDs);
	free(extFlagDs);
	free(nGapDistDs);
	free(hGapDistDs);
	free(vGapDistDs);

	printTime_toStandardOutput();
	printTime_toFile();

#if VOCL_BALANCE
	dlclose(voclModulePtr);
	MPI_Finalize();
#endif

	return 0;
}



