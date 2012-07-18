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

#include "oclUtils.h"
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "oclBodySystemOpenclLaunch_dp.h"
#include "nbody_timer.h"

BodySystemOpenCL::BodySystemOpenCL(int numBodies, cl_device_id dev, cl_context ctx, cl_command_queue cmdq, 
                                   unsigned int p, unsigned int q, bool usePBO, bool bDouble)
: BodySystem(numBodies),
  device(dev),
  cxContext(ctx),
  cqCommandQueue(cmdq),
  m_hPos(0),
  m_hVel(0),
  m_bUsePBO(usePBO),
  m_currentRead(0),
  m_currentWrite(1),
  m_p(p),
  m_q(q),
  m_bDouble(bDouble)
{
    m_dPos[0] = m_dPos[1] = 0;
    m_dVel[0] = m_dVel[1] = 0;

    _initialize(numBodies);

    // create non multithreaded program and kernel
    shrLog("\nCreateProgramAndKernel _noMT... ");  
    if (CreateProgramAndKernel(ctx, &dev, "integrateBodies_noMT", &noMT_program, &noMT_kernel, m_bDouble)) 
    {
        exit(shrLogEx(LOGBOTH | CLOSELOG, -1.0, "CreateProgramAndKernel _noMT ", STDERROR)); 
    }

    // create multithreaded program and kernel
    shrLog("\nCreateProgramAndKernel _MT... ");
    if (CreateProgramAndKernel(ctx, &dev, "integrateBodies_MT", &MT_program, &MT_kernel, m_bDouble)) 
    {
        exit(shrLogEx(LOGBOTH | CLOSELOG, -1.0, "CreateProgramAndKernel _MT ", STDERROR)); 
    }

    setSoftening(0.00125f);
    setDamping(0.995f);   
}

BodySystemOpenCL::~BodySystemOpenCL()
{
    _finalize();
    m_numBodies = 0;
}

void BodySystemOpenCL::_initialize(int numBodies)
{
    oclCheckError(m_bInitialized, shrFALSE);

    m_numBodies = numBodies;

    m_hPos = new double[m_numBodies*4];
    m_hVel = new double[m_numBodies*4];

    memset(m_hPos, 0, m_numBodies*4*sizeof(double));
    memset(m_hVel, 0, m_numBodies*4*sizeof(double));

    AllocateNBodyArrays(cxContext, m_dPos, m_numBodies, m_bDouble);
    shrLog("\nAllocateNBodyArrays m_dPos\n"); 
    
    AllocateNBodyArrays(cxContext, m_dVel, m_numBodies, m_bDouble);
    shrLog("\nAllocateNBodyArrays m_dVel\n"); 

    m_bInitialized = true;
}

void BodySystemOpenCL::_finalize()
{
    oclCheckError(m_bInitialized, shrTRUE);

    delete [] m_hPos;
    delete [] m_hVel;

	timerStart();
	clReleaseKernel(MT_kernel);
	clReleaseKernel(noMT_kernel);
	timerEnd();
	strTime.releaseKernel += elapsedTime();
	strTime.numReleaseKernel += 2;

	timerStart();
	clReleaseProgram(MT_program);
	clReleaseProgram(noMT_program);
	timerEnd();
	strTime.releaseProgram += elapsedTime();
	strTime.numReleaseProgram++;

    DeleteNBodyArrays(m_dVel);
    DeleteNBodyArrays(m_dPos);
}

void BodySystemOpenCL::setSoftening(double softening)
{
    m_softeningSq = softening * softening;
}

void BodySystemOpenCL::setDamping(double damping)
{
    m_damping = damping;
}

void BodySystemOpenCL::update(double deltaTime)
{
    oclCheckError(m_bInitialized, shrTRUE);
    
    IntegrateNbodySystem(cqCommandQueue,
                         MT_kernel, noMT_kernel,
                         m_dPos[m_currentWrite], m_dVel[m_currentWrite], 
                         m_dPos[m_currentRead], m_dVel[m_currentRead],
                         m_pboCL[m_currentRead], m_pboCL[m_currentWrite],
                         deltaTime, m_damping, m_softeningSq,
                         m_numBodies, m_p, m_q,
                         (m_bUsePBO ? 1 : 0),
						 m_bDouble);

    std::swap(m_currentRead, m_currentWrite);
}

double* BodySystemOpenCL::getArray(BodyArray array)
{
    oclCheckError(m_bInitialized, shrTRUE);
 
    double *hdata = 0;
    cl_mem ddata = 0;
    cl_mem pbo = 0;

    switch (array)
    {
        default:
        case BODYSYSTEM_POSITION:
            hdata = m_hPos;
            ddata = m_dPos[m_currentRead];
            break;

        case BODYSYSTEM_VELOCITY:
            hdata = m_hVel;
            ddata = m_dVel[m_currentRead];
            break;
    }

    CopyArrayFromDevice(cqCommandQueue, hdata, ddata, pbo, m_numBodies, m_bDouble);

    return hdata;
}

void BodySystemOpenCL::setArray(BodyArray array, const double* data)
{
    oclCheckError(m_bInitialized, shrTRUE);
 
    switch (array)
    {
        default:
        case BODYSYSTEM_POSITION:
            CopyArrayToDevice(cqCommandQueue, m_dPos[m_currentRead], data, m_numBodies, m_bDouble);
        	break;

        case BODYSYSTEM_VELOCITY:
            CopyArrayToDevice(cqCommandQueue, m_dVel[m_currentRead], data, m_numBodies, m_bDouble);
            break;
    }       
}

void BodySystemOpenCL::synchronizeThreads() const
{
    ThreadSync(cqCommandQueue);
}


