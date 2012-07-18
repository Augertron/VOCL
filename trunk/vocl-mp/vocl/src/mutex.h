/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef VCLX_H_INCLUDED
#define VCLX_H_INCLUDED

#include "mpi.h"

/* Prototypes in this file must be interpreted as C routines when this header
 * file is compiled by a C++ compiler.  "mpi.h" already does this for the main
 * MPI_ routines. */
#if defined(__cplusplus)
extern "C" {
#endif

/* RMA Mutexes extension declarations: */

struct mpixi_mutex_s;
typedef struct mpixi_mutex_s * VCLX_Mutex;

int VCLX_Mutex_create(int count, MPI_Comm comm, VCLX_Mutex *hdl);
int VCLX_Mutex_free  (VCLX_Mutex *hdl);
int VCLX_Mutex_lock  (VCLX_Mutex hdl, int mutex, int proc);
int VCLX_Mutex_unlock(VCLX_Mutex hdl, int mutex, int proc);

#if defined(__cplusplus)
}
#endif

#endif /* VCLX_H_INCLUDED */
