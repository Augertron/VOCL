#include "timeRec.h"
#include <stdio.h>
#include <stdlib.h>

struct timeval timeStart;
struct timeval timeEnd;
struct timeval time1Start;
struct timeval time1End;

STRUCT_TIME strTime;

void timerStart()
{
	gettimeofday(&timeStart, NULL);
}

void timerEnd()
{
	gettimeofday(&timeEnd, NULL);
}

void timer1Start()
{
	gettimeofday(&time1Start, NULL);
}

void timer1End()
{
	gettimeofday(&time1End, NULL);
}

//return value is ms
double elapsedTime()
{
	double deltaTime;
	deltaTime = (timeEnd.tv_sec - timeStart.tv_sec) * 1000.0 + 
				(timeEnd.tv_usec - timeStart.tv_usec) / 1000.0;
	
	return deltaTime;
}

double elapsedTime1()
{
	double deltaTime;
	deltaTime = (time1End.tv_sec - time1Start.tv_sec) * 1000.0 + 
				(time1End.tv_usec - time1Start.tv_usec) / 1000.0;
	
	return deltaTime;
}

void printTime_toStandardOutput()
{
	strTime.totalTime = strTime.getPlatform
						+ strTime.getDeviceID
						+ strTime.createContext
						+ strTime.createCommandQueue
						+ strTime.createProgramWithSource
						+ strTime.buildProgram
						+ strTime.createKernel
						+ strTime.readMatrix
						+ strTime.createBuffer
						+ strTime.enqueueWriteBuffer
						+ strTime.setKernelArg
						+ strTime.kernelExecution
						+ strTime.enqueueReadBuffer
						+ strTime.releaseKernel
						+ strTime.releaseMemObj
						+ strTime.releaseProgram
						+ strTime.releaseCmdQueue
						+ strTime.releaseContext
						+ strTime.printMatrix;

	printf("TimeInfo:\n");
	printf("getPlatform = %.3f\ngetDeviceID = %.3f\ncreateContext = %.3f\ncreateCommandQueue = %.3f\ncreateProgramWithSource = %.3f\nbuildProgram = %.3f\ncreateKernel = %.3f\nreadMatrix = %.3f\ncreateBuffer = %.3f\nenqueueWriteBuffer = %.3f\nsetKernelArg = %.3f\nkernelExecution = %.3f\nenqueueReadBuffer = %.3f\nreleaseKernel=%.3f\nreleaseMemObj=%.3f\nreleaseProgram=%.3f\nreleaseCmdQueue=%.3f\nreleaseContext=%.3f\nprintMatrix = %.3f\ntotalTime = %.3f\n",
			strTime.getPlatform,
			strTime.getDeviceID,
			strTime.createContext,
			strTime.createCommandQueue,
			strTime.createProgramWithSource,
			strTime.buildProgram,
			strTime.createKernel,
			strTime.readMatrix,
			strTime.createBuffer,
			strTime.enqueueWriteBuffer,
			strTime.setKernelArg,
			strTime.kernelExecution,
			strTime.enqueueReadBuffer,
			strTime.releaseKernel,
			strTime.releaseMemObj,
			strTime.releaseProgram,
			strTime.releaseCmdQueue,
			strTime.releaseContext,
			strTime.printMatrix,
			strTime.totalTime);

//	printf("matrixMul:\t %d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
//			strTime.numGetPlatform,
//			strTime.numGetDeviceID,
//			strTime.numCreateContext,
//			strTime.numCreateCommandQueue,
//			strTime.numCreateProgramWithSource,
//			strTime.numBuildProgram,
//			strTime.numCreateKernel,
//			strTime.numCreateBuffer,
//			strTime.numEnqueueWriteBuffer,
//			strTime.numSetKernelArg,
//			strTime.numKernelExecution,
//			strTime.numEnqueueReadBuffer,
//			strTime.numReleaseKernel,
//			strTime.numReleaseMemObj,
//			strTime.numReleaseProgram,
//			strTime.numReleaseCmdQueue,
//			strTime.numReleaseContext);
}

void printTime_toFile()
{
	FILE *pTimeFile, *pCountFile;
	pTimeFile = fopen("../runtime.txt", "at");
	if (pTimeFile == NULL)
	{
		printf("File runtime.txt open error!\n");
		return;
	}

	strTime.totalTime = strTime.getPlatform
						+ strTime.getDeviceID
						+ strTime.createContext
						+ strTime.createCommandQueue
						+ strTime.createProgramWithSource
						+ strTime.buildProgram
						+ strTime.createKernel
						//+ strTime.readMatrix
						+ strTime.createBuffer
						+ strTime.enqueueWriteBuffer
						+ strTime.setKernelArg
						+ strTime.kernelExecution
						+ strTime.enqueueReadBuffer
						+ strTime.releaseKernel
						+ strTime.releaseMemObj
						+ strTime.releaseProgram
						+ strTime.releaseCmdQueue
						+ strTime.releaseContext;
						//+ strTime.printMatrix;
	//fprintf(pTimeFile, "getPlatform = %.3f\ngetDeviceID = %.3f\ncreateContext = %.3f\ncreateCommandQueue = %.3f\ncreateProgramWithSource = %.3f\nbuildProgram = %.3f\ncreateKernel = %.3f\nreadMatrix = %.3f\ncreateBuffer = %.3f\nenqueueWriteBuffer = %.3f\nkernelExecution = %.3f\nenqueueReadBuffer = %.3f\nprintMatrix = %.3f\ntotalTime = %.3f\n",
	//fprintf(pTimeFile, "%.3f\n%.3f\n%.3f\n%.3f\n%.3f\n%.3f\n%.3f\n%.3f\n%.3f\n%.3f\n%.3f\n%.3f\n%.3f\n%.3f\n%.3f\n\n",
	fprintf(pTimeFile, "SW\t %.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n",
			strTime.getPlatform,
			strTime.getDeviceID,
			strTime.createContext,
			strTime.createCommandQueue,
			strTime.createProgramWithSource,
			strTime.buildProgram,
			strTime.createKernel,
			//strTime.readMatrix,
			strTime.createBuffer,
			strTime.enqueueWriteBuffer,
			strTime.setKernelArg,
			strTime.kernelExecution,
			strTime.enqueueReadBuffer,
			strTime.releaseKernel,
			strTime.releaseMemObj,
			strTime.releaseProgram,
			strTime.releaseCmdQueue,
			strTime.releaseContext,
			//strTime.printMatrix,
			strTime.totalTime);
	fclose(pTimeFile);
	
	pCountFile = fopen("../counts.txt", "at");
	if (pCountFile == NULL)
	{
		printf("File counts.txt open error!\n");
		return;
	}

	fprintf(pCountFile, "SW\t %d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
			strTime.numGetPlatform,
			strTime.numGetDeviceID,
			strTime.numCreateContext,
			strTime.numCreateCommandQueue,
			strTime.numCreateProgramWithSource,
			strTime.numBuildProgram,
			strTime.numCreateKernel,
			strTime.numCreateBuffer,
			strTime.numEnqueueWriteBuffer,
			strTime.numSetKernelArg,
			strTime.numKernelExecution,
			strTime.numEnqueueReadBuffer,
			strTime.numReleaseKernel,
			strTime.numReleaseMemObj,
			strTime.numReleaseProgram,
			strTime.numReleaseCmdQueue,
			strTime.numReleaseContext);
	fclose(pCountFile);

}

//void printTime_toFile()
//{
//	FILE *pTimeFile;
//	pTimeFile = fopen("../runtime.txt", "at");
//	if (pTimeFile == NULL)
//	{
//		printf("File runtime.txt open error!\n");
//		return;
//	}
//
//	//calculate the total time
//	strTime.totalTime = strTime.preprocessingTime +
//						strTime.copyTimeHostToDevice +
//						strTime.matrixFillingTime +
//						strTime.copyTimeDeviceToHost +
//						strTime.traceBackTime;
//	fprintf(pTimeFile, "Version 6:\t %.3lf\t%.3lf\t%.3lf\t%.3lf\t%.3lf\t%.3lf\t%.3lf\n",
//			strTime.iniTime,
//			strTime.preprocessingTime,
//			strTime.copyTimeHostToDevice,
//			strTime.matrixFillingTime,
//			strTime.traceBackTime,
//			strTime.copyTimeDeviceToHost,
//			strTime.totalTime);
//	fclose(pTimeFile);
//
//	return;
//}
//
//void print_throughputTime()
//{
//	printf("Throughput time = %.3lf\n", strTime.throughtputTime);
//}
