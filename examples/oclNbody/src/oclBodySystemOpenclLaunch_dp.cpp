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

#include "oclBodySystemOpenclLaunch_dp.h"
#include <oclUtils.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <sstream>
#include "nbody_timer.h"
// var to hold path to executable
extern const char* cExecutablePath;

extern "C"
{
    char clSourcefile[KERNEL_SOURCE_FILE_LEN];

    void AllocateNBodyArrays(cl_context cxGPUContext, cl_mem* vel, int numBodies, int dFlag)
    {
        // 4 doubles each for alignment reasons
        unsigned int memSize;
		memSize = sizeof( double) * 4 * numBodies;

		timerStart();
		vel[0] = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, memSize, NULL, NULL);
		vel[1] = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, memSize, NULL, NULL);
		timerEnd();
		strTime.createBuffer += elapsedTime();
		strTime.numCreateBuffer += 2;
    }

    void DeleteNBodyArrays(cl_mem vel[2])
    {
		timerStart();
        clReleaseMemObject(vel[0]);
        clReleaseMemObject(vel[1]);
		timerEnd();
		strTime.releaseMemObj += elapsedTime();
		strTime.numReleaseMemObj += 2;
    }

    void CopyArrayFromDevice(cl_command_queue cqCommandQueue, double *host, cl_mem device, cl_mem pboCL, int numBodies, bool bDouble)
    {   
        cl_int ciErrNum;
        unsigned int size;
		size = numBodies * 4 * sizeof(double);
		ciErrNum = clEnqueueReadBuffer(cqCommandQueue, device, CL_FALSE, 0, size, host, 0, NULL, NULL);
		oclCheckError(ciErrNum, CL_SUCCESS);
		
    }

    void CopyArrayToDevice(cl_command_queue cqCommandQueue, cl_mem device, const double* host, int numBodies, bool bDouble)
    {
        cl_int ciErrNum;
        unsigned int size;
		size = numBodies*4*sizeof(double);
		ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, device, CL_FALSE, 0, size, host, 0, NULL, NULL);
		oclCheckError(ciErrNum, CL_SUCCESS);
    }

    void ThreadSync(cl_command_queue cqCommandQueue) 
    { 
        clFinish(cqCommandQueue); 
    }

    void IntegrateNbodySystem(cl_command_queue cqCommandQueue,
                              cl_kernel MT_kernel, cl_kernel noMT_kernel,
                              cl_mem newPos, cl_mem newVel,
                              cl_mem oldPos, cl_mem oldVel,
                              cl_mem pboCLOldPos, cl_mem pboCLNewPos,
                              double deltaTime, double damping, double softSq,
                              int numBodies, int p, int q,
                              int bUsePBO, bool bDouble)
    {
        int sharedMemSize;

		//for double precision
		sharedMemSize = p * q * sizeof(cl_double4); // 4 doubles for pos
        size_t global_work_size[2];
        size_t local_work_size[2];
        cl_int ciErrNum = CL_SUCCESS;
        cl_kernel kernel;

        // When the numBodies / thread block size is < # multiprocessors 
        // (16 on G80), the GPU is underutilized. For example, with 256 threads per
        // block and 1024 bodies, there will only be 4 thread blocks, so the 
        // GPU will only be 25% utilized.  To improve this, we use multiple threads
        // per body.  We still can use blocks of 256 threads, but they are arranged
        // in q rows of p threads each.  Each thread processes 1/q of the forces 
        // that affect each body, and then 1/q of the threads (those with 
        // threadIdx.y==0) add up the partial sums from the other threads for that 
        // body.  To enable this, use the "--p=" and "--q=" command line options to
        // this example.  e.g.: "nbody.exe --n=1024 --p=64 --q=4" will use 4 
        // threads per body and 256 threads per block. There will be n/p = 16 
        // blocks, so a G80 GPU will be 100% utilized.

        if (q == 1)
        {
            kernel = MT_kernel;
        }
        else
        {
            kernel = noMT_kernel;
        }

		timerStart();
	    ciErrNum |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&newPos);
        ciErrNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&newVel);
        ciErrNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&oldPos);
        ciErrNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&oldVel);
		ciErrNum |= clSetKernelArg(kernel, 4, sizeof(cl_double), (void *)&deltaTime);
		ciErrNum |= clSetKernelArg(kernel, 5, sizeof(cl_double), (void *)&damping);
		ciErrNum |= clSetKernelArg(kernel, 6, sizeof(cl_double), (void *)&softSq);
        ciErrNum |= clSetKernelArg(kernel, 7, sizeof(cl_int), (void *)&numBodies);
        ciErrNum |= clSetKernelArg(kernel, 8, sharedMemSize, NULL);
		timerEnd();
		strTime.setKernelArg += elapsedTime();
		strTime.numSetKernelArg += 9;

        oclCheckError(ciErrNum, CL_SUCCESS);

        // set work-item dimensions
        local_work_size[0] = p;
        local_work_size[1] = q;
        global_work_size[0]= numBodies;
        global_work_size[1]= q;

        // execute the kernel:
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);

    }

    // Function to read in kernel from uncompiled source, create the OCL program and build the OCL program 
    // **************************************************************************************************
    int CreateProgramAndKernel(cl_context cxGPUContext, cl_device_id* cdDevices, const char *kernel_name, cl_program *program, cl_kernel *kernel, bool bDouble)
    {
        size_t szSourceLen;
        cl_int ciErrNum = CL_SUCCESS; 

        // Read the kernel in from file
		snprintf(clSourcefile, KERNEL_SOURCE_FILE_LEN, "%s/examples/oclNbody/oclNbodyKernel_sp.cl",
		 ABS_SRCDIR);
        shrLog("\nLoading Uncompiled kernel from .cl file, using %s\n", clSourcefile);
        char* pcSource = oclLoadProgSource(clSourcefile, "", &szSourceLen);
        oclCheckError(pcSource != NULL, shrTRUE);

		// Check OpenCL version -> vec3 types are supported only from version 1.1 and above
		char cOCLVersion[32];
		clGetDeviceInfo(cdDevices[0], CL_DEVICE_VERSION, sizeof(cOCLVersion), &cOCLVersion, 0);

		int iVec3Length = 3;
		if( strncmp("OpenCL 1.0", cOCLVersion, 10) == 0 ) {
			iVec3Length = 4;
		}


		//for double precision
		char *pcSourceForDouble;
		std::stringstream header;
		header << "#define REAL double";
		header << std::endl;
		header << "#define REAL4 double4";
		header << std::endl;
		header << "#define REAL3 double" << iVec3Length;
		header << std::endl;
		header << "#define ZERO3 {0.0, 0.0, 0.0" << ((iVec3Length == 4) ? ", 0.0}" : "}");
		header << std::endl;
		
		header << pcSource;
		pcSourceForDouble = (char *)malloc(header.str().size() + 1);
		szSourceLen = header.str().size();
#ifdef WIN32
        strcpy_s(pcSourceForDouble, szSourceLen + 1, header.str().c_str());
#else
        strcpy(pcSourceForDouble, header.str().c_str());
#endif

        // create the program 
		timerStart();
        *program = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&pcSourceForDouble, &szSourceLen, &ciErrNum);
		timerEnd();
		strTime.createProgramWithSource += elapsedTime();
        oclCheckError(ciErrNum, CL_SUCCESS);
        shrLog("clCreateProgramWithSource\n"); 

        // Build the program with 'mad' Optimization option
#ifdef MAC
	char *flags = "-cl-fast-relaxed-math -DMAC";
#else
	char *flags = "-cl-fast-relaxed-math";
#endif
		timerStart();
        ciErrNum = clBuildProgram(*program, 0, NULL, flags, NULL, NULL);
		timerEnd();
		strTime.buildProgram += elapsedTime();
        if (ciErrNum != CL_SUCCESS)
        {
            // write out standard error, Build Log and PTX, then cleanup and exit
            shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
            oclLogBuildInfo(*program, oclGetFirstDev(cxGPUContext));
            oclLogPtx(*program, oclGetFirstDev(cxGPUContext), "oclNbody.ptx");
            oclCheckError(ciErrNum, CL_SUCCESS); 
        }
        shrLog("clBuildProgram\n"); 

        // create the kernel
		timerStart();
        *kernel = clCreateKernel(*program, kernel_name, &ciErrNum);
		timerEnd();
		strTime.createKernel += elapsedTime();
        oclCheckError(ciErrNum, CL_SUCCESS); 
        shrLog("clCreateKernel\n"); 

		free(pcSourceForDouble);

        return 0;
    }
}
