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

AC_PROG_CC
AC_PROG_CXX
AC_HEADER_STDC

PAC_SET_HEADER_LIB_PATH(opencl)
PAC_CHECK_HEADER_LIB_FATAL(opencl,CL/opencl.h,OpenCL,clGetPlatformIDs)

AC_ARG_WITH(cuda-sdk,
	AC_HELP_STRING([--with-cuda-sdk],[Path to the CUDA SDK]),
	cuda_sdk_path=$withval,)
if test "$cuda_sdk_path" != "" ; then
   LDFLAGS="$LDFLAGS -L${cuda_sdk_path}/shared/lib"
   LIBS="$LIBS -lshrutil_x86_64"
   LDFLAGS="$LDFLAGS -L${cuda_sdk_path}/OpenCL/common/lib"
   LIBS="$LIBS -loclUtil_x86_64"
   CPPFLAGS="$CPPFLAGS -I${cuda_sdk_path}/shared/inc"
   CPPFLAGS="$CPPFLAGS -I${cuda_sdk_path}/OpenCL/common/inc"
else
   AC_ERROR([No CUDA SDK path provided])
fi

abs_srcdir=`(cd $srcdir && pwd)`
AC_SUBST(abs_srcdir)
AC_SUBST(top_srcdir)

## Final output
AC_OUTPUT(Makefile)
