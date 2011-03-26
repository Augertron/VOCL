#
# Copyright (C) 2010. See COPYRIGHT in top-level directory.
#

ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -DABS_SRCDIR=\"$(abs_srcdir)\" -DKERNEL_SOURCE_FILE_LEN=1024

lib_LTLIBRARIES = libvocl.la
libvocl_la_SOURCES = $(top_srcdir)/src/vocl.c

bin_PROGRAMS = vocl_proxy bandwidth matrixmult nbody

vocl_proxy_SOURCES = $(top_srcdir)/src/vocl_proxy.c

bandwidth_SOURCES = $(top_srcdir)/examples/bandwidth/bandwidth.c \
	$(top_srcdir)/examples/bandwidth/bw_timer.c

matrixmult_SOURCES = $(top_srcdir)/examples/matrixmult/matrixmult.c \
	$(top_srcdir)/examples/matrixmult/mm_timer.c

nbody_CPPFLAGS = -I$(top_srcdir)/examples/oclNbody/inc
nbody_SOURCES = $(top_srcdir)/examples/oclNbody/src/nbody_timer.cpp \
	$(top_srcdir)/examples/oclNbody/src/oclBodySystemOpenclLaunch.cpp \
	$(top_srcdir)/examples/oclNbody/src/oclRenderParticles.cpp \
	$(top_srcdir)/examples/oclNbody/src/oclBodySystemCpu.cpp \
	$(top_srcdir)/examples/oclNbody/src/oclNbody.cpp \
	$(top_srcdir)/examples/oclNbody/src/param.cpp \
	$(top_srcdir)/examples/oclNbody/src/oclBodySystemOpencl.cpp \
	$(top_srcdir)/examples/oclNbody/src/oclNbodyGold.cpp \
	$(top_srcdir)/examples/oclNbody/src/paramgl.cpp
