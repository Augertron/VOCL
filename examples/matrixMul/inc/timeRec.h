#ifndef __TIME_RECORD_H__
#define __TIME_RECORD_H__
#include <sys/time.h>

//timer
void timerStart();
void timerEnd();
double elapsedTime();

//timer1
void timer1Start();
void timer1End();
double elapsedTime1();

void printTime_toStandardOutput();
void printTime_toFile();
void print_throughputTime();

typedef struct timeStruct {
	double getPlatform;
	double getDeviceID;
	double createContext;
	double createCommandQueue;
	double createProgramWithSource;
	double buildProgram;
	double createKernel;
	double readMatrix;
	double createBuffer;
	double enqueueWriteBuffer;
	double kernelExecution;
	double enqueueReadBuffer;
	double printMatrix;
	double totalTime;
} STRUCT_TIME;

extern struct timeval timeStart;
extern struct timeval timeEnd;
extern struct timeval time1Start;
extern struct timeval time1End;

extern STRUCT_TIME strTime;

#endif //__TIME_RECORD_H__
