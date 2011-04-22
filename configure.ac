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

## Non-verbose make
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

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

AC_PROG_CC_C99

AC_CHECK_SIZEOF(cl_event, ,[#include <CL/opencl.h>])
AC_CHECK_SIZEOF(uint8_t)
AC_CHECK_SIZEOF(uint16_t)
AC_CHECK_SIZEOF(uint32_t)
AC_CHECK_SIZEOF(uint64_t)

if test "$ac_cv_sizeof_uint8_t" = "$ac_cv_sizeof_cl_event" ; then
   vocl_event=uint8_t
elif test "$ac_cv_sizeof_uint16_t" = "$ac_cv_sizeof_cl_event" ; then
   vocl_event=uint16_t
elif test "$ac_cv_sizeof_uint32_t" = "$ac_cv_sizeof_cl_event" ; then
   vocl_event=uint32_t
elif test "$ac_cv_sizeof_uint64_t" = "$ac_cv_sizeof_cl_event" ; then
   vocl_event=uint64_t
else
   AC_ERROR([Cannot find cl_event size])
fi
AC_SUBST(vocl_event)

AC_CHECK_SIZEOF(CONTROL_MSG_UNION, ,[#include "src/vocl_proxy.h"])
control_msg_size=$ac_cv_sizeof_CONTROL_MSG_UNION
AC_SUBST(control_msg_size)


abs_srcdir=`(cd $srcdir && pwd)`
AC_SUBST(abs_srcdir)
AC_SUBST(top_srcdir)

## Final output
AC_OUTPUT(Makefile
	src/voclEventProc.h
	src/vocl_proxy_macro.h
)
