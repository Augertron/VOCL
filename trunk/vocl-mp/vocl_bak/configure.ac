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

AC_CHECK_SIZEOF(cl_platform_id, ,[#include <CL/opencl.h>])
AC_CHECK_SIZEOF(cl_device_id, ,[#include <CL/opencl.h>])
AC_CHECK_SIZEOF(cl_context, ,[#include <CL/opencl.h>])
AC_CHECK_SIZEOF(cl_command_queue, ,[#include <CL/opencl.h>])
AC_CHECK_SIZEOF(cl_mem, ,[#include <CL/opencl.h>])
AC_CHECK_SIZEOF(cl_program, ,[#include <CL/opencl.h>])
AC_CHECK_SIZEOF(cl_kernel, ,[#include <CL/opencl.h>])
AC_CHECK_SIZEOF(cl_event, ,[#include <CL/opencl.h>])
AC_CHECK_SIZEOF(cl_sampler, ,[#include <CL/opencl.h>])
AC_CHECK_SIZEOF(uint8_t)
AC_CHECK_SIZEOF(uint16_t)
AC_CHECK_SIZEOF(uint32_t)
AC_CHECK_SIZEOF(uint64_t)

if test "$ac_cv_sizeof_uint8_t" = "$ac_cv_sizeof_cl_platform_id" ; then
   vocl_platform_id=uint8_t
elif test "$ac_cv_sizeof_uint16_t" = "$ac_cv_sizeof_cl_platform_id" ; then
   vocl_platform_id=uint16_t
elif test "$ac_cv_sizeof_uint32_t" = "$ac_cv_sizeof_cl_platform_id" ; then
   vocl_platform_id=uint32_t
elif test "$ac_cv_sizeof_uint64_t" = "$ac_cv_sizeof_cl_platform_id" ; then
   vocl_platform_id=uint64_t
else
   AC_ERROR([Cannot find cl_platform_id size])
fi


if    test "$ac_cv_sizeof_uint8_t" = "$ac_cv_sizeof_cl_device_id" ; then
   vocl_device_id=uint8_t
elif test "$ac_cv_sizeof_uint16_t" = "$ac_cv_sizeof_cl_device_id" ; then
   vocl_device_id=uint16_t
elif test "$ac_cv_sizeof_uint32_t" = "$ac_cv_sizeof_cl_device_id" ; then
   vocl_device_id=uint32_t
elif test "$ac_cv_sizeof_uint64_t" = "$ac_cv_sizeof_cl_device_id" ; then
   vocl_device_id=uint64_t
else
   AC_ERROR([Cannot find cl_device_id size])
fi

if    test "$ac_cv_sizeof_uint8_t" = "$ac_cv_sizeof_cl_context" ; then
   vocl_context=uint8_t
elif test "$ac_cv_sizeof_uint16_t" = "$ac_cv_sizeof_cl_context" ; then
   vocl_context=uint16_t
elif test "$ac_cv_sizeof_uint32_t" = "$ac_cv_sizeof_cl_context" ; then
   vocl_context=uint32_t
elif test "$ac_cv_sizeof_uint64_t" = "$ac_cv_sizeof_cl_context" ; then
   vocl_context=uint64_t
else
   AC_ERROR([Cannot find cl_context size])
fi

if    test "$ac_cv_sizeof_uint8_t" = "$ac_cv_sizeof_cl_command_queue" ; then
   vocl_command_queue=uint8_t
elif test "$ac_cv_sizeof_uint16_t" = "$ac_cv_sizeof_cl_command_queue" ; then
   vocl_command_queue=uint16_t
elif test "$ac_cv_sizeof_uint32_t" = "$ac_cv_sizeof_cl_command_queue" ; then
   vocl_command_queue=uint32_t
elif test "$ac_cv_sizeof_uint64_t" = "$ac_cv_sizeof_cl_command_queue" ; then
   vocl_command_queue=uint64_t
else
   AC_ERROR([Cannot find cl_command_queue size])
fi

if    test "$ac_cv_sizeof_uint8_t" = "$ac_cv_sizeof_cl_mem" ; then
   vocl_mem=uint8_t
elif test "$ac_cv_sizeof_uint16_t" = "$ac_cv_sizeof_cl_mem" ; then
   vocl_mem=uint16_t
elif test "$ac_cv_sizeof_uint32_t" = "$ac_cv_sizeof_cl_mem" ; then
   vocl_mem=uint32_t
elif test "$ac_cv_sizeof_uint64_t" = "$ac_cv_sizeof_cl_mem" ; then
   vocl_mem=uint64_t
else
   AC_ERROR([Cannot find cl_mem size])
fi

if    test "$ac_cv_sizeof_uint8_t" = "$ac_cv_sizeof_cl_program" ; then
   vocl_program=uint8_t
elif test "$ac_cv_sizeof_uint16_t" = "$ac_cv_sizeof_cl_program" ; then
   vocl_program=uint16_t
elif test "$ac_cv_sizeof_uint32_t" = "$ac_cv_sizeof_cl_program" ; then
   vocl_program=uint32_t
elif test "$ac_cv_sizeof_uint64_t" = "$ac_cv_sizeof_cl_program" ; then
   vocl_program=uint64_t
else
   AC_ERROR([Cannot find cl_program size])
fi

if    test "$ac_cv_sizeof_uint8_t" = "$ac_cv_sizeof_cl_kernel" ; then
   vocl_kernel=uint8_t
elif test "$ac_cv_sizeof_uint16_t" = "$ac_cv_sizeof_cl_kernel" ; then
   vocl_kernel=uint16_t
elif test "$ac_cv_sizeof_uint32_t" = "$ac_cv_sizeof_cl_kernel" ; then
   vocl_kernel=uint32_t
elif test "$ac_cv_sizeof_uint64_t" = "$ac_cv_sizeof_cl_kernel" ; then
   vocl_kernel=uint64_t
else
   AC_ERROR([Cannot find cl_kernel size])
fi

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

if    test "$ac_cv_sizeof_uint8_t" = "$ac_cv_sizeof_cl_sampler" ; then
   vocl_sampler=uint8_t
elif test "$ac_cv_sizeof_uint16_t" = "$ac_cv_sizeof_cl_sampler" ; then
   vocl_sampler=uint16_t
elif test "$ac_cv_sizeof_uint32_t" = "$ac_cv_sizeof_cl_sampler" ; then
   vocl_sampler=uint32_t
elif test "$ac_cv_sizeof_uint64_t" = "$ac_cv_sizeof_cl_sampler" ; then
   vocl_sampler=uint64_t
else
   AC_ERROR([Cannot find cl_sampler size])
fi
AC_SUBST(vocl_platform_id)
AC_SUBST(vocl_device_id)
AC_SUBST(vocl_context)
AC_SUBST(vocl_command_queue)
AC_SUBST(vocl_mem)
AC_SUBST(vocl_program)
AC_SUBST(vocl_kernel)
AC_SUBST(vocl_event)
AC_SUBST(vocl_sampler)

AC_CHECK_SIZEOF(CONTROL_MSG_UNION, ,[#include "src/vocl_proxy.h"])
control_msg_size=$ac_cv_sizeof_CONTROL_MSG_UNION
AC_SUBST(control_msg_size)

abs_srcdir=`(cd $srcdir && pwd)`
AC_SUBST(abs_srcdir)
AC_SUBST(top_srcdir)

AC_ARG_ENABLE([rebalance], [--enable-rebalance  enable whether function voclRebalance() is called in application.],
		voclrebalance="$enableval", voclrebalance="no")
if test "$voclrebalance" = "no"; then
	vocl_rebalance=0
else
	vocl_rebalance=1
	CPPFLAGS="$CPPFLAGS -I${abs_srcdir}/src"
fi
AC_SUBST(vocl_rebalance)

## Final output
AC_OUTPUT(Makefile
	src/voclStructures.h
	src/vocl_proxy_macro.h
)
