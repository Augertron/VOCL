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
//#include <paramgl.h>
#include <algorithm>
#include "nbody_timer.h"
// Project includes
#include "oclBodySystemOpencl_dp.h"
#include "oclBodySystemCpu_dp.h"
#include <sched.h>


// view, GLUT and display params
int ox = 0, oy = 0;
int buttonState          = 0;
double camera_trans[]     = {0, -2, -100};
double camera_rot[]       = {0, 0, 0};
double camera_trans_lag[] = {0, -2, -100};
double camera_rot_lag[]   = {0, 0, 0};
const double inertia      = 0.1;
bool displayEnabled = true;
bool bPause = false;
bool bUsePBO = false;
bool bFullScreen = false;
bool bShowSliders = true;
int iGLUTWindowHandle;              // handle to the GLUT window
int iGraphicsWinPosX = 0;           // GLUT Window X location
int iGraphicsWinPosY = 0;           // GLUT Window Y location
int iGraphicsWinWidth = 1024;       // GLUT Window width
int iGraphicsWinHeight = 768;       // GL Window height
GLint iVsyncState;                  // state var to cache startup Vsync setting
int flopsPerInteraction = 20;

// Struct defintion for Nbody demo physical parameters
struct NBodyParams
{       
    double m_timestep;
    double m_clusterScale;
    double m_velocityScale;
    double m_softening;
    double m_damping;
    double m_pointSize;
    double m_x, m_y, m_z;

    void print()
    { 
        shrLog("{ %f, %f, %f, %f, %f, %f, %f, %f, %f },\n", 
                   m_timestep, m_clusterScale, m_velocityScale, 
                   m_softening, m_damping, m_pointSize, m_x, m_y, m_z); 
    }
};

// Array of structs of physical parameters to flip among
NBodyParams demoParams[] = 
{
    { 0.016f, 1.54f, 8.0f, 0.1f, 1.0f, 1.0f, 0, -2, -100},
    { 0.016f, 0.68f, 20.0f, 0.1f, 1.0f, 0.8f, 0, -2, -30},
    { 0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    { 0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    { 0.0019f, 0.32f, 276.0f, 1.0f, 1.0f, 0.07f, 0, 0, -5.0f},
    { 0.0016f, 0.32f, 272.0f, 0.145f, 1.0f, 0.08f, 0, 0, -5.0f},
    { 0.016f, 6.04f, 0.0f, 1.0f, 1.0f, 0.76f, 0, 0, -50.0f},
};

shrBOOL compareDoublee(double *ref, double *data, unsigned int size, double errorThreshold)
{
	shrBOOL res = shrTRUE;
	double delta = 0.0, tmp;
	unsigned int i;
	for (i = 0; i < size; i++)
	{
		tmp = ref[i] - data[i];
		if (tmp < 0.0) tmp = tmp * (-1.0);
		delta += tmp;
	}

	if (delta > errorThreshold)
	{
		res = shrFALSE;
	}

	return res;
}

// Basic simulation parameters
int numBodies = 7680;               // default # of bodies in sim (can be overridden by command line switch --n=<N>)
int numIterations = 20, iterNo;
bool bDouble = true;               //false: sp double, true: dp 
int numDemos = sizeof(demoParams) / sizeof(NBodyParams);
int activeDemo = 0;
NBodyParams activeParams = demoParams[activeDemo];
BodySystem *nbody         = 0;
BodySystemOpenCL *nbodyGPU = 0;
double* hPos = 0;
double* hVel = 0;
double* hColor = 0;

// OpenCL vars
cl_platform_id cpPlatform;          // OpenCL Platform
cl_context cxContext;               // OpenCL Context
cl_command_queue cqCommandQueue;    // OpenCL Command Queue
cl_device_id *cdDevices = NULL;     // OpenCL device list
cl_uint uiNumDevices = 0;           // Number of OpenCL devices available
cl_uint uiNumDevsUsed = 1;          // Number of OpenCL devices used in this sample 
cl_uint uiTargetDevice = 0;	        // OpenCL Device to compute on
const char* cExecutablePath;

// Timers
#define DEMOTIME 0
#define FUNCTIME 1
#define FPSTIME 2

// fps, quick test and qatest vars
int iFrameCount = 0;                // FPS count for averaging
int iFrameTrigger = 90;             // FPS trigger for sampling
int iFramesPerSec = 60;             // frames per second
double dElapsedTime = 0.0;          // timing var to hold elapsed time in each phase of tour mode
double demoTime = 5.0;              // length of each demo phase in sec
shrBOOL bTour = shrTRUE;            // true = cycles between modes, false = stays on selected 1 mode (manually switchable)
shrBOOL bNoPrompt = shrFALSE;       // false = normal GL loop, true = Finite period of GL loop (a few seconds)
shrBOOL bQATest = shrFALSE;         // false = normal GL loop, true = run No-GL test sequence (checks against host and also does a perf test)
int iTestSets = 3;

// Forward Function declarations
//*****************************************************************************
// OpenGL (GLUT) functionality
//void InitGL(int* argc, char **argv);
//void DisplayGL();
//void ReshapeGL(int w, int h);
//void IdleGL(void);
//void KeyboardGL(unsigned char key, int x, int y);
//void MouseGL(int button, int state, int x, int y);
//void MotionGL(int x, int y);
//void SpecialGL (int key, int x, int y);

// Simulation
void ResetSim(BodySystem *system, int numBodies, NBodyConfig config, bool useGL);
void copyDataH2D(BodySystem *system);
void copyDataD2H(BodySystem *system);
void InitNbody(cl_device_id dev, cl_context ctx, cl_command_queue cmdq,
               int numBodies, int p, int q, bool bUsePBO, bool bDouble);
void SelectDemo(int index);
void CompareResults(int numBodies);
void RunProfiling(int iterations, unsigned int uiWorkgroup);
void ComputePerfStats(double &dGigaInteractionsPerSecond, double &dGigaFlops, 
                      double dSeconds, int iterations);

// helpers
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;
void TriggerFPSUpdate();

// Main program
//*****************************************************************************
int main(int argc, char** argv) 
{
    // Locals used with command line args
    int p = 256;            // workgroup X dimension
    int q = 1;              // workgroup Y dimension
	cpu_set_t set;
	CPU_ZERO(&set);

    // latch the executable path for other funcs to use
    cExecutablePath = argv[0];

    // start logs and show command line help
	shrSetLogFileName ("oclNbody.txt");
    shrLog("%s Starting...\n\n", cExecutablePath);
    shrLog("Command line switches:\n");
	//shrLog("  --qatest\t\tCheck correctness of GPU execution and measure performance)\n");
	shrLog("  --noprompt\t\tQuit simulation automatically after a brief period\n");
    shrLog("  --n=<numbodies>\tSpecify # of bodies to simulate (default = %d)\n", numBodies);
	shrLog("  --p=<workgroup X dim>\tSpecify X dimension of workgroup (default = %d)\n", p);
	shrLog("  --q=<workgroup Y dim>\tSpecify Y dimension of workgroup (default = %d)\n\n", q);
	shrLog("  --iter=<numIterations>\tSpecify the number of iterations (default = %d)\n\n", numIterations);

	// Get command line arguments if there are any and set vars accordingly
    if (argc > 0)
    {
        shrGetCmdLineArgumenti(argc, (const char**)argv, "p", &p);
        shrGetCmdLineArgumenti(argc, (const char**)argv, "q", &q);
        shrGetCmdLineArgumenti(argc, (const char**)argv, "n", &numBodies);
        shrGetCmdLineArgumenti(argc, (const char**)argv, "iter", &numIterations);
        bNoPrompt = shrCheckCmdLineFlag(argc, (const char**)argv, "noprompt");
        bQATest = shrCheckCmdLineFlag(argc, (const char**)argv, "qatest");
    }
	bQATest = shrTRUE;

	shrLog("Initialize timer...\n\n");
	memset(&strTime, sizeof(STRUCT_TIME), 0);

	shrLog("Iteration num = %d\n", numIterations);

    //Get the NVIDIA platform
	timerStart();
    //cl_int ciErrNum = oclGetPlatformID(&cpPlatform);
	cl_int ciErrNum = clGetPlatformIDs(1, &cpPlatform, NULL);
	timerEnd();
	strTime.getPlatform += elapsedTime();
	strTime.numGetPlatform++;
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("clGetPlatformID...\n\n"); 
	
	sched_getaffinity(0, sizeof(set), &set);
	printf("cpuid = %d\n", set.__bits[0]);

	if (bDouble)
	{
		shrLog("Double precision execution...\n\n");
	}
	else
	{
		shrLog("Single precision execution...\n\n");
	}

	flopsPerInteraction = bDouble ? 30 : 20; 
    
	//Get all the devices
    shrLog("Get the Device info and select Device...\n");
	timerStart();
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiNumDevices);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    cdDevices = (cl_device_id *)malloc(uiNumDevices * sizeof(cl_device_id) );
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, uiNumDevices, cdDevices, NULL);
	timerEnd();
	strTime.getDeviceID += elapsedTime();
	strTime.numGetDeviceID += 2;
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Set target device and Query number of compute units on uiTargetDevice
    shrLog("  # of Devices Available = %u\n", uiNumDevices); 
    if(shrGetCmdLineArgumentu(argc, (const char**)argv, "device", &uiTargetDevice)== shrTRUE) 
    {
        uiTargetDevice = CLAMP(uiTargetDevice, 0, (uiNumDevices - 1));
    }
    shrLog("  Using Device %u, ", uiTargetDevice); 
    oclPrintDevName(LOGBOTH, cdDevices[uiTargetDevice]);  
    cl_uint uiNumComputeUnits;        
    clGetDeviceInfo(cdDevices[uiTargetDevice], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uiNumComputeUnits), &uiNumComputeUnits, NULL);
    shrLog("  # of Compute Units = %u\n", uiNumComputeUnits); 

    //Create the context
    shrLog("clCreateContext...\n"); 
	timerStart();
    cxContext = clCreateContext(0, uiNumDevsUsed, &cdDevices[uiTargetDevice], NULL, NULL, &ciErrNum);
	timerEnd();
	strTime.createContext += elapsedTime();
	strTime.numCreateContext++;
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Create a command-queue 
    shrLog("clCreateCommandQueue...device = %d, \n\n", uiTargetDevice); 
	timerStart();
    cqCommandQueue = clCreateCommandQueue(cxContext, cdDevices[uiTargetDevice], CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
	timerEnd();
	strTime.createCommandQueue += elapsedTime();
	strTime.numCreateCommandQueue++;
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Log and config for number of bodies
    shrLog("Number of Bodies = %d\n", numBodies); 
    switch (numBodies)
    {
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

    if ((q * p) > 256)
    {
        p = 256 / q;
        shrLog("Setting p=%d to maintain %d threads per block\n", p, 256);
    }

    if ((q == 1) && (numBodies < p))
    {
        p = numBodies;
        shrLog("Setting p=%d because # of bodies < p\n", p);
    }
    shrLog("Workgroup Dims = (%d x %d)\n\n", p, q); 

    // CL/GL interop disabled
    InitNbody(cdDevices[uiTargetDevice], cxContext, cqCommandQueue, numBodies, p, q, bUsePBO, bDouble);
    ResetSim(nbody, numBodies, NBODY_CONFIG_SHELL, bUsePBO);
	copyDataH2D(nbody);
	nbody->synchronizeThreads();

	// Compare to host, profile and write out file for regression analysis
	shrLog("Running oclNbody Results Comparison...\n\n"); 
	CompareResults(numBodies);

	//data transmission
	timerStart();
	for (iterNo = 0; iterNo < numIterations; iterNo++)
	{
		copyDataH2D(nbody);
		RunProfiling(100, (unsigned int)(p * q));  // 100 iterations
		copyDataD2H(nbody);
	}
	nbody->synchronizeThreads();
	timerEnd();
	strTime.kernelExecution += elapsedTime();
	strTime.numEnqueueReadBuffer += numIterations;

    // init timers
    shrDeltaT(DEMOTIME); // timer 0 is for timing demo periods
    shrDeltaT(FUNCTIME); // timer 1 is for logging function delta t's
    shrDeltaT(FPSTIME);  // timer 2 is for fps measurement   


    // Cleanup/exit 
    Cleanup(EXIT_SUCCESS);

	printTime_toStandardOutput();
	printTime_toFile();

	exit(EXIT_SUCCESS);

}

//*****************************************************************************
void RunProfiling(int iterations, unsigned int uiWorkgroup)
{
    // once without timing to prime the GPU
	nbody->update(activeParams.m_timestep);
	//nbody->synchronizeThreads();

	//Start timer 0 and process n loops on the GPU
	shrDeltaT(FUNCTIME);
    for (int i = 0; i < iterations; ++i)
    {
        nbody->update(activeParams.m_timestep);
    }
    //nbody->synchronizeThreads();

    //Get elapsed time and throughput, then log to sample and master logs
//	double dSeconds = shrDeltaT(FUNCTIME);
//	double dGigaInteractionsPerSecond = 0.0;
//	double dGigaFlops = 0.0;
//	ComputePerfStats(dGigaInteractionsPerSecond, dGigaFlops, dSeconds, iterations);
//	shrLogEx(LOGBOTH | MASTER, 0, "oclNBody-%s, Throughput = %.4f GFLOP/s, Time = %.5f s, Size = %u bodies, NumDevsUsed = %u, Workgroup = %u\n", 
//		(bDouble ? "DP" : "SP"), dGigaFlops, dSeconds/(double)iterations, numBodies, uiNumDevsUsed, uiWorkgroup); 
}

// Helper to trigger reset of fps vars at transition 
//*****************************************************************************
void TriggerFPSUpdate()
{
    iFrameCount = 0; 
    shrDeltaT(FPSTIME);
    iFramesPerSec = 1;
    iFrameTrigger = 2;
}

//*****************************************************************************
void ResetSim(BodySystem *system, int numBodies, NBodyConfig config, bool useGL)
{
    shrLog("\nReset Nbody System...\n\n");

    // initalize the memory
    randomizeBodies(config, hPos, hVel, hColor, activeParams.m_clusterScale, 
		            activeParams.m_velocityScale, numBodies);

//    system->setArray(BodySystem::BODYSYSTEM_POSITION, hPos);
//    system->setArray(BodySystem::BODYSYSTEM_VELOCITY, hVel);
}

void copyDataH2D(BodySystem *system)
{
    system->setArray(BodySystem::BODYSYSTEM_POSITION, hPos);
    system->setArray(BodySystem::BODYSYSTEM_VELOCITY, hVel);
}

void copyDataD2H(BodySystem *system)
{
	system->getArray(BodySystem::BODYSYSTEM_POSITION);
}

//*****************************************************************************
void InitNbody(cl_device_id dev, cl_context ctx, cl_command_queue cmdq,
               int numBodies, int p, int q, bool bUsePBO, bool bDouble)
{
    // New nbody system for Device/GPU computations
    nbodyGPU = new BodySystemOpenCL(numBodies, dev, ctx, cmdq, p, q, bUsePBO, bDouble);
    nbody = nbodyGPU;

    // allocate host memory
    hPos = new double[numBodies*4];
    hVel = new double[numBodies*4];
    hColor = new double[numBodies*4];

    // Set sim parameters
    nbody->setSoftening(activeParams.m_softening);
    nbody->setDamping(activeParams.m_damping);
}

//*****************************************************************************
void SelectDemo(int index)
{
    oclCheckErrorEX((index < numDemos), shrTRUE, pCleanup);

    activeParams = demoParams[index];
    camera_trans[0] = camera_trans_lag[0] = activeParams.m_x;
    camera_trans[1] = camera_trans_lag[1] = activeParams.m_y;
    camera_trans[2] = camera_trans_lag[2] = activeParams.m_z;
    ResetSim(nbody, numBodies, NBODY_CONFIG_SHELL, true);

    //Rest the demo timer
    shrDeltaT(DEMOTIME);
}

//*****************************************************************************
void CompareResults(int numBodies)
{
    // Run computation on the device/GPU
    shrLog("  Computing on the Device / GPU...\n");
    nbodyGPU->update(0.001f);
    nbodyGPU->synchronizeThreads();

    // Write out device/GPU data file for regression analysis
    shrLog("  Writing out Device/GPU data file for analysis...\n");
    double* fGPUData = nbodyGPU->getArray(BodySystem::BODYSYSTEM_POSITION);
    //shrWriteFilef( "oclNbody_Regression.dat", fGPUData, numBodies, 0.0, false);

    // Run computation on the host CPU
    shrLog("  Computing on the Host / CPU...\n\n");
    BodySystemCPU* nbodyCPU = new BodySystemCPU(numBodies);
    nbodyCPU->setArray(BodySystem::BODYSYSTEM_POSITION, hPos);
    nbodyCPU->setArray(BodySystem::BODYSYSTEM_VELOCITY, hVel);
    nbodyCPU->update(0.001f);

    // Check if result matches 
    shrBOOL bMatch = compareDoublee(fGPUData, 
                        nbodyGPU->getArray(BodySystem::BODYSYSTEM_POSITION), 
						numBodies, .001f);
    shrLog("%s\n\n", (shrTRUE == bMatch) ? "PASSED" : "FAILED");

    // Cleanup local allocation
    if(nbodyCPU)delete nbodyCPU; 
}


//*****************************************************************************
void ComputePerfStats(double &dGigaInteractionsPerSecond, double &dGigaFlops, double dSeconds, int iterations)
{
	//int flopsPerInteraction = bDouble ? 30 : 20; 
    dGigaInteractionsPerSecond = 1.0e-9 * (double)numBodies * (double)numBodies * (double)iterations / dSeconds;
    dGigaFlops = dGigaInteractionsPerSecond * (double)flopsPerInteraction;	
}

// Helper to clean up
//*****************************************************************************
void Cleanup(int iExitCode)
{
    shrLog("\nStarting Cleanup...\n\n");

    // Restore startup Vsync state, if supported
    #ifdef _WIN32
        if (wglewIsSupported("WGL_EXT_swap_control")) 
        {
            wglSwapIntervalEXT(iVsyncState);
        }
    #else
        #if defined (__APPLE__) || defined(MACOSX)
            CGLSetParameter(CGLGetCurrentContext(), kCGLCPSwapInterval, &iVsyncState); 
        #endif
    #endif

    // Cleanup allocated objects
    if(nbodyGPU)delete nbodyGPU;
	timerStart();
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
	timerEnd();
	strTime.releaseCmdQueue += elapsedTime();
	strTime.numReleaseCmdQueue++;

	timerStart();
    if(cxContext)clReleaseContext(cxContext);
	timerEnd();
	strTime.releaseContext += elapsedTime();
	strTime.numReleaseContext++;
    if(hPos)delete [] hPos;
    if(hVel)delete [] hVel;
    if(hColor)delete [] hColor;

    // finalize logs and leave
    if (bNoPrompt || bQATest)
    {
        shrLogEx(LOGBOTH | CLOSELOG, 0, "oclNbody.exe Exiting...\n");
    }
    else 
    {
        shrLogEx(LOGBOTH | CLOSELOG, 0, "oclNbody.exe Exiting...\nPress <Enter> to Quit\n");
        #ifdef WIN32
            getchar();
        #endif
    }
    //exit (iExitCode);
}
