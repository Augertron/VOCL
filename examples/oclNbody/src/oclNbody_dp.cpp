/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Utility and system includes
#include "oclUtils.h"
#include <algorithm>
#include "nbody_timer.h"
// Project includes
#include "oclBodySystemOpencl_dp.h"
#include "oclBodySystemCpu_dp.h"
#include <sched.h>


// view, GLUT and display params
static int ox = 0, oy = 0;
static int buttonState = 0;
static double camera_trans[] = { 0, -2, -100 };
static double camera_rot[] = { 0, 0, 0 };
static double camera_trans_lag[] = { 0, -2, -100 };
static double camera_rot_lag[] = { 0, 0, 0 };
static const double inertia = 0.1;
static bool displayEnabled = true;
static bool bPause = false;
static bool bUsePBO = false;
static int disableCPU = 1;
static bool bFullScreen = false;
static bool bShowSliders = true;
static int iGLUTWindowHandle;   // handle to the GLUT window
static int iGraphicsWinPosX = 0;        // GLUT Window X location
static int iGraphicsWinPosY = 0;        // GLUT Window Y location
static int iGraphicsWinWidth = 1024;    // GLUT Window width
static int iGraphicsWinHeight = 768;    // GL Window height
//static GLint iVsyncState;       // state var to cache startup Vsync setting
static int flopsPerInteraction = 20;

// Struct defintion for Nbody demo physical parameters
struct NBodyParams {
    double m_timestep;
    double m_clusterScale;
    double m_velocityScale;
    double m_softening;
    double m_damping;
    double m_pointSize;
    double m_x, m_y, m_z;

    void print() {
        shrLog("{ %f, %f, %f, %f, %f, %f, %f, %f, %f },\n",
               m_timestep, m_clusterScale, m_velocityScale,
               m_softening, m_damping, m_pointSize, m_x, m_y, m_z);
}};

// Array of structs of physical parameters to flip among
NBodyParams demoParams[] = {
    {0.016f, 1.54f, 8.0f, 0.1f, 1.0f, 1.0f, 0, -2, -100},
    {0.016f, 0.68f, 20.0f, 0.1f, 1.0f, 0.8f, 0, -2, -30},
    {0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    {0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    {0.0019f, 0.32f, 276.0f, 1.0f, 1.0f, 0.07f, 0, 0, -5.0f},
    {0.0016f, 0.32f, 272.0f, 0.145f, 1.0f, 0.08f, 0, 0, -5.0f},
    {0.016f, 6.04f, 0.0f, 1.0f, 1.0f, 0.76f, 0, 0, -50.0f},
};

shrBOOL compareDoublee(double *ref, double *data, unsigned int size, double errorThreshold)
{
    shrBOOL res = shrTRUE;
    double delta = 0.0, tmp;
    unsigned int i;
    for (i = 0; i < size; i++) {
        tmp = ref[i] - data[i];
        if (tmp < 0.0)
            tmp = tmp * (-1.0);
        delta += tmp;
    }

    if (delta > errorThreshold) {
        res = shrFALSE;
    }

    return res;
}

// Basic simulation parameters
static int numBodies = 7680;    // default # of bodies in sim (can be overridden by command line switch --n=<N>)
static int numIterations = 20, iterNo;
static bool bDouble = true;     //false: sp double, true: dp
static int numDemos = sizeof(demoParams) / sizeof(NBodyParams);
static int activeDemo = 0;
static NBodyParams activeParams = demoParams[activeDemo];
static BodySystem **nbody = 0;
static BodySystemOpenCL **nbodyGPU = 0;
static double **hPos = 0;
static double **hVel = 0;
static double **hColor = 0;

// OpenCL vars
static cl_platform_id *cpPlatforms;     // OpenCL Platform
static cl_uint platformNum;
static cl_context *cxContexts;  // OpenCL Context
static cl_command_queue *cqCommandQueues;       // OpenCL Command Queue
static cl_device_id *cdDevices = NULL;  // OpenCL device list
static cl_uint uiNumDevices = 0, *deviceNums;   // Number of OpenCL devices available
static cl_uint deviceNumUsed = 1;       // Number of OpenCL devices used in this sample
static const char *cExecutablePath;

// Timers
#define DEMOTIME 0
#define FUNCTIME 1
#define FPSTIME 2

// fps, quick test and qatest vars
static int iFrameCount = 0;     // FPS count for averaging
static int iFrameTrigger = 90;  // FPS trigger for sampling
static int iFramesPerSec = 60;  // frames per second
static double dElapsedTime = 0.0;       // timing var to hold elapsed time in each phase of tour mode
static double demoTime = 5.0;   // length of each demo phase in sec
static shrBOOL bTour = shrTRUE; // true = cycles between modes, false = stays on selected 1 mode (manually switchable)
static shrBOOL bNoPrompt = shrFALSE;    // false = normal GL loop, true = Finite period of GL loop (a few seconds)
static shrBOOL bQATest = shrFALSE;      // false = normal GL loop, true = run No-GL test sequence (checks against host and also does a perf test)
static int iTestSets = 3;

// Simulation
void ResetSim(BodySystem * system, int numBodies, NBodyConfig config, bool useGL, int index);
void copyDataH2D(BodySystem * system, int index);
void copyDataD2H(BodySystem * system);
void InitNbody(cl_device_id dev, cl_context ctx, cl_command_queue cmdq,
               int numBodies, int p, int q, bool bUsePBO, bool bDouble, int index);
void CompareResults(int numBodies, int index);
void RunProfiling(int iterations, unsigned int uiWorkgroup, int index);
void ComputePerfStats(double &dGigaInteractionsPerSecond, double &dGigaFlops,
                      double dSeconds, int iterations);

// helpers
void Cleanup(int iExitCode);
void (*pCleanup) (int) = &Cleanup;
void TriggerFPSUpdate();

// Main program
//*****************************************************************************
int main(int argc, char **argv)
{
    int i, index, deviceNo = 0;
    bool bUseAllDevices = false;
    // Locals used with command line args
    int p = 256;                // workgroup X dimension
    int q = 1;                  // workgroup Y dimension
    struct timeval t1, t2;

    // latch the executable path for other funcs to use
    cExecutablePath = argv[0];

    // start logs and show command line help
    shrSetLogFileName("oclNbody.txt");
    shrLog("%s Starting...\n\n", cExecutablePath);
    shrLog("Command line switches:\n");
    shrLog("  --noprompt\t\tQuit simulation automatically after a brief period\n");
    shrLog("  --n=<numbodies>\tSpecify # of bodies to simulate (default = %d)\n", numBodies);
    shrLog("  --p=<workgroup X dim>\tSpecify X dimension of workgroup (default = %d)\n", p);
    shrLog("  --q=<workgroup Y dim>\tSpecify Y dimension of workgroup (default = %d)\n\n", q);
    shrLog("  --iter=<numIterations>\tSpecify the number of iterations (default = %d)\n\n",
           numIterations);
    shrLog("  --device=<deviceNo>\t Specify the device no to be used (default = %d)\n\n",
           deviceNo);
    shrLog("  --deviceall \t\t Use all virtual GPUs.\n\n");

    // Get command line arguments if there are any and set vars accordingly
    if (argc > 0) {
        shrGetCmdLineArgumenti(argc, (const char **) argv, "p", &p);
        shrGetCmdLineArgumenti(argc, (const char **) argv, "q", &q);
        shrGetCmdLineArgumenti(argc, (const char **) argv, "n", &numBodies);
        shrGetCmdLineArgumenti(argc, (const char **) argv, "iter", &numIterations);
        shrGetCmdLineArgumenti(argc, (const char **) argv, "disablecpu", &disableCPU);
        shrGetCmdLineArgumenti(argc, (const char **) argv, "device", &deviceNo);
        bNoPrompt = shrCheckCmdLineFlag(argc, (const char **) argv, "noprompt");
        bUseAllDevices =
            (shrTRUE == shrCheckCmdLineFlag(argc, (const char **) argv, "deviceall"));
    }
    bQATest = shrTRUE;

    shrLog("Initialize timer...\n\n");
    memset(&strTime, sizeof(STRUCT_TIME), 0);

    shrLog("Iteration num = %d\n", numIterations);

    //Get the NVIDIA platform
    timerStart();
    cl_int ciErrNum = clGetPlatformIDs(0, NULL, &platformNum);
    cpPlatforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * platformNum);
    deviceNums = (cl_uint *) malloc(sizeof(cl_uint) * platformNum);
    ciErrNum != clGetPlatformIDs(platformNum, cpPlatforms, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    timerEnd();
    strTime.getPlatform += elapsedTime();
    shrLog("clGetPlatformID...\n\n");

    shrLog("Single precision execution...\n\n");

    flopsPerInteraction = bDouble ? 30 : 20;

    //Get all the devices
    shrLog("Get the Device info and select Device...\n");
    timerStart();
    uiNumDevices = 0;
    for (i = 0; i < platformNum; i++) {
        ciErrNum = clGetDeviceIDs(cpPlatforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceNums[i]);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        uiNumDevices += deviceNums[i];
    }
    cdDevices = (cl_device_id *) malloc(uiNumDevices * sizeof(cl_device_id));
    cxContexts = (cl_context *) malloc(uiNumDevices * sizeof(cl_context));
    cqCommandQueues = (cl_command_queue *) malloc(uiNumDevices * sizeof(cl_command_queue));
    nbody = (BodySystem **) malloc(uiNumDevices * sizeof(BodySystem *));
    nbodyGPU = (BodySystemOpenCL **) malloc(uiNumDevices * sizeof(BodySystemOpenCL *));
    hPos = (double **) malloc(uiNumDevices * sizeof(double *));
    hVel = (double **) malloc(uiNumDevices * sizeof(double *));
    hColor = (double **) malloc(uiNumDevices * sizeof(double *));

    uiNumDevices = 0;
    for (i = 0; i < platformNum; i++) {
        ciErrNum = clGetDeviceIDs(cpPlatforms[i], CL_DEVICE_TYPE_GPU, deviceNums[i],
                                  &cdDevices[uiNumDevices], NULL);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        uiNumDevices += deviceNums[i];
    }
    timerEnd();
    strTime.getDeviceID += elapsedTime();

    if (bUseAllDevices == true) {
        shrLog("All virtual GPUs are used..., deviceCount = %d\n\n", uiNumDevices);
        deviceNo = 0;
        deviceNumUsed = uiNumDevices;
    }
    else {
        shrLog("Device %d is used...\n\n", deviceNo);
        deviceNumUsed = 1;
    }

    //Create the context
    shrLog("clCreateContext...\n");
    timerStart();
    for (i = 0; i < deviceNumUsed; i++) {
        index = i + deviceNo;
        cxContexts[i] = clCreateContext(0, 1, &cdDevices[index], NULL, NULL, &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    timerEnd();
    strTime.createContext += elapsedTime();

    // Create a command-queue
    shrLog("clCreateCommandQueue...\n\n");
    timerStart();
    for (i = 0; i < deviceNumUsed; i++) {
        index = i + deviceNo;
        cqCommandQueues[i] =
            clCreateCommandQueue(cxContexts[i], cdDevices[index], CL_QUEUE_PROFILING_ENABLE,
                                 &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    timerEnd();
    strTime.createCommandQueue += elapsedTime();

    // Log and config for number of bodies
    shrLog("Number of Bodies = %d\n", numBodies);
    switch (numBodies) {
    case 1024:
        activeParams.m_clusterScale = 1.52f;
        activeParams.m_velocityScale = 2.f;
        break;
    case 2048:
        activeParams.m_clusterScale = 1.56f;
        activeParams.m_velocityScale = 2.64f;
        break;
    case 4096:
        activeParams.m_clusterScale = 1.68f;
        activeParams.m_velocityScale = 2.98f;
        break;
    case 7680:
    case 8192:
        activeParams.m_clusterScale = 1.98f;
        activeParams.m_velocityScale = 2.9f;
        break;
    default:
    case 15360:
    case 16384:
        activeParams.m_clusterScale = 1.54f;
        activeParams.m_velocityScale = 8.f;
        break;
    case 30720:
    case 32768:
        activeParams.m_clusterScale = 1.44f;
        activeParams.m_velocityScale = 11.f;
        break;
    }

    if ((q * p) > 256) {
        p = 256 / q;
        shrLog("Setting p=%d to maintain %d threads per block\n", p, 256);
    }

    if ((q == 1) && (numBodies < p)) {
        p = numBodies;
        shrLog("Setting p=%d because # of bodies < p\n", p);
    }
    shrLog("Workgroup Dims = (%d x %d)\n\n", p, q);

    // CL/GL interop disabled
    for (i = 0; i < deviceNumUsed; i++) {
        index = i + deviceNo;
        InitNbody(cdDevices[index], cxContexts[i], cqCommandQueues[i], numBodies, p, q,
                  bUsePBO, bDouble, i);
        ResetSim(nbody[i], numBodies, NBODY_CONFIG_SHELL, bUsePBO, i);
        copyDataH2D(nbody[i], i);
        nbody[i]->synchronizeThreads();
        // Compare to host, profile and write out file for regression analysis
        if (disableCPU == 0) {
            shrLog("Running oclNbody Results Comparison...\n\n");
            CompareResults(numBodies, i);
        }
    }

    //data transmission
    timerStart();
    gettimeofday(&t1, NULL);
    for (iterNo = 0; iterNo < numIterations; iterNo++) {
        for (i = 0; i < deviceNumUsed; i++) {
            copyDataH2D(nbody[i], i);
        }

        for (i = 0; i < deviceNumUsed; i++) {
            RunProfiling(100, (unsigned int) (p * q), i);       // 100 iterations
        }

        for (i = 0; i < deviceNumUsed; i++) {
            copyDataD2H(nbody[i]);
        }
    }

    for (i = 0; i < deviceNumUsed; i++) {
        nbody[i]->synchronizeThreads();
    }
    gettimeofday(&t2, NULL);
    timerEnd();
    strTime.kernelExecution =
        1000.0 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;

    // Cleanup/exit
    Cleanup(EXIT_SUCCESS);

    free(cpPlatforms);
    free(deviceNums);
    free(cdDevices);
    free(cxContexts);
    free(cqCommandQueues);
    free(nbody);
    free(nbodyGPU);
    free(hPos);
    free(hVel);
    free(hColor);

    printTime_toStandardOutput();
    printTime_toFile();

    exit(EXIT_SUCCESS);

}

//*****************************************************************************
void RunProfiling(int iterations, unsigned int uiWorkgroup, int index)
{
    // once without timing to prime the GPU
    nbody[index]->update(activeParams.m_timestep);

    //Start timer 0 and process n loops on the GPU
    for (int i = 0; i < iterations; ++i) {
        nbody[index]->update(activeParams.m_timestep);
    }
}

// Helper to trigger reset of fps vars at transition
//*****************************************************************************
void TriggerFPSUpdate()
{
    iFrameCount = 0;
    iFramesPerSec = 1;
    iFrameTrigger = 2;
}

//*****************************************************************************
void ResetSim(BodySystem * system, int numBodies, NBodyConfig config, bool useGL, int index)
{
    shrLog("\nReset Nbody System...\n\n");

    // initalize the memory
    randomizeBodies(config, hPos[index], hVel[index], hColor[index],
                    activeParams.m_clusterScale, activeParams.m_velocityScale, numBodies);
}

void copyDataH2D(BodySystem * system, int index)
{
    system->setArray(BodySystem::BODYSYSTEM_POSITION, hPos[index]);
    system->setArray(BodySystem::BODYSYSTEM_VELOCITY, hVel[index]);
}

void copyDataD2H(BodySystem * system)
{
    system->getArray(BodySystem::BODYSYSTEM_POSITION);
}

//*****************************************************************************
void InitNbody(cl_device_id dev, cl_context ctx, cl_command_queue cmdq,
               int numBodies, int p, int q, bool bUsePBO, bool bDouble, int index)
{
    // New nbody system for Device/GPU computations
    nbodyGPU[index] = new BodySystemOpenCL(numBodies, dev, ctx, cmdq, p, q, bUsePBO, bDouble);
    nbody[index] = nbodyGPU[index];

    // allocate host memory
    hPos[index] = new double[numBodies * 4];
    hVel[index] = new double[numBodies * 4];
    hColor[index] = new double[numBodies * 4];

    // Set sim parameters
    nbody[index]->setSoftening(activeParams.m_softening);
    nbody[index]->setDamping(activeParams.m_damping);
}

//*****************************************************************************
void CompareResults(int numBodies, int index)
{
    // Run computation on the device/GPU
    shrLog("  Computing on the Device / GPU...\n");
    nbodyGPU[index]->update(0.001f);
    nbodyGPU[index]->synchronizeThreads();

    // Write out device/GPU data file for regression analysis
    shrLog("  Writing out Device/GPU data file for analysis...\n");
    double *fGPUData = nbodyGPU[index]->getArray(BodySystem::BODYSYSTEM_POSITION);
    //shrWriteFilef("oclNbody_Regression.dat", fGPUData, numBodies, 0.0, false);

    // Run computation on the host CPU
    shrLog("  Computing on the Host / CPU...\n\n");
    BodySystemCPU *nbodyCPU = new BodySystemCPU(numBodies);
    nbodyCPU->setArray(BodySystem::BODYSYSTEM_POSITION, hPos[index]);
    nbodyCPU->setArray(BodySystem::BODYSYSTEM_VELOCITY, hVel[index]);
    nbodyCPU->update(0.001f);

    // Check if result matches
    shrBOOL bMatch = compareDoublee(fGPUData,
                                    nbodyGPU[index]->getArray(BodySystem::BODYSYSTEM_POSITION),
                                    numBodies, .001f);
    shrLog("%s\n\n", (shrTRUE == bMatch) ? "PASSED" : "FAILED");

    // Cleanup local allocation
    if (nbodyCPU)
        delete nbodyCPU;
}


//*****************************************************************************
void ComputePerfStats(double &dGigaInteractionsPerSecond, double &dGigaFlops, double dSeconds,
                      int iterations)
{
    //int flopsPerInteraction = bDouble ? 30 : 20;
    dGigaInteractionsPerSecond =
        1.0e-9 * (double) numBodies *(double) numBodies *(double) iterations / dSeconds;
    dGigaFlops = dGigaInteractionsPerSecond * (double) flopsPerInteraction;
}

// Helper to clean up
//*****************************************************************************
void Cleanup(int iExitCode)
{
    int i;
    shrLog("\nStarting Cleanup...\n\n");

    // Restore startup Vsync state, if supported
#ifdef _WIN32
    if (wglewIsSupported("WGL_EXT_swap_control")) {
        wglSwapIntervalEXT(iVsyncState);
    }
#else
#if defined (__APPLE__) || defined(MACOSX)
    CGLSetParameter(CGLGetCurrentContext(), kCGLCPSwapInterval, &iVsyncState);
#endif
#endif

    // Cleanup allocated objects
    for (i = 0; i < deviceNumUsed; i++) {
        if (nbodyGPU[i])
            delete nbodyGPU[i];
        timerStart();
        clReleaseCommandQueue(cqCommandQueues[i]);
        timerEnd();
        strTime.releaseCmdQueue += elapsedTime();

        timerStart();
        clReleaseContext(cxContexts[i]);
        timerEnd();
        strTime.releaseContext += elapsedTime();

        if (hPos[i])
            delete[]hPos[i];
        if (hVel[i])
            delete[]hVel[i];
        if (hColor[i])
            delete[]hColor[i];
    }

    return;
}
