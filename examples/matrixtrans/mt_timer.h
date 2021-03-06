#ifndef __TIME_RECORD_H__
#define __TIME_RECORD_H__
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif
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
        double setKernelArg;
        double kernelExecution;
        double enqueueReadBuffer;
        double releaseKernel;
        double releaseMemObj;
        double releaseProgram;
        double releaseCmdQueue;
        double releaseContext;
        double printMatrix;
        double totalTime;

        //count of each operation
        int numGetPlatform;
        int numGetDeviceID;
        int numCreateContext;
        int numCreateCommandQueue;
        int numCreateProgramWithSource;
        int numBuildProgram;
        int numCreateKernel;
        int numReadMatrix;
        int numCreateBuffer;
        int numEnqueueWriteBuffer;
        int numSetKernelArg;
        int numKernelExecution;
        int numEnqueueReadBuffer;
        int numReleaseKernel;
        int numReleaseMemObj;
        int numReleaseProgram;
        int numReleaseCmdQueue;
        int numReleaseContext;
        int numPrintMatrix;
        int numTotalTime;

    } STRUCT_TIME;

    extern struct timeval timeStart;
    extern struct timeval timeEnd;
    extern struct timeval time1Start;
    extern struct timeval time1End;

    extern STRUCT_TIME strTime;
#ifdef __cplusplus
}
#endif
#endif                          //__TIME_RECORD_H__
