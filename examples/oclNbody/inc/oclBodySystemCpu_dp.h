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

#ifndef __CL_BODYSYSTEMCPU_H__
#define __CL_BODYSYSTEMCPU_H__

#include "oclBodySystem_dp.h"

    // CPU Body System
class BodySystemCPU:public BodySystem {
  public:
    BodySystemCPU(int numBodies);
     virtual ~ BodySystemCPU();

    virtual void update(double deltaTime);

    virtual void setSoftening(double softening) {
        m_softeningSquared = softening * softening;
    } virtual void setDamping(double damping) {
        m_damping = damping;
    }

    virtual double *getArray(BodyArray array);
    virtual void setArray(BodyArray array, const double *data);

    virtual size_t getCurrentReadBuffer() const {
        return m_currentRead;
  } protected:                 // methods
     BodySystemCPU() {
    }   // default constructor

    virtual void _initialize(int numBodies);
    virtual void _finalize();

    void _computeNBodyGravitation();
    void _integrateNBodySystem(double deltaTime);

  protected:   // data
    double *m_pos[2];
    double *m_vel[2];
    double *m_force;

    double m_softeningSquared;
    double m_damping;

    unsigned int m_currentRead;
    unsigned int m_currentWrite;
};

#endif // __CL_BODYSYSTEMCPU_H__
