dnl
dnl Copyright (C) 2010. See COPYRIGHT in top-level directory.
dnl

AC_PREREQ(2.62)

AC_INIT([vgpu],[0])
AC_CONFIG_AUX_DIR(m4)
AC_CONFIG_MACRO_DIR(m4)
AM_INIT_AUTOMAKE([-Wall -Werror foreign 1.11])

LT_INIT
LT_PREREQ([2.2.6])

AC_PROG_CC([mpicc])
AC_HEADER_STDC

PAC_SET_HEADER_LIB_PATH(opencl)
PAC_CHECK_HEADER_LIB_FATAL(opencl,CL/opencl.h,OpenCL,clGetPlatformIDs)

AC_SUBST(top_srcdir)

## Final output
AC_OUTPUT(Makefile)
