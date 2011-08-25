#
# Copyright (C) 2010. See COPYRIGHT in top-level directory.
#

ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -DABS_SRCDIR=\"$(abs_srcdir)\" -DKERNEL_SOURCE_FILE_LEN=1024 -DPROXY_PATH_NAME=\"$(exec_prefix)\" -DPROXY_PATH_NAME_LEN=256
AM_CXXFLAGS = -DABS_SRCDIR=\"$(abs_srcdir)\" -DKERNEL_SOURCE_FILE_LEN=1024 -DPROXY_PATH_NAME=\"$(exec_prefix)\" -DPROXY_PATH_NAME_LEN=256

lib_LTLIBRARIES = libvocl.la
libvocl_la_SOURCES = $(top_srcdir)/src/voclOpencl.c \
					 $(top_srcdir)/src/voclReadBufferProc.c \
					 $(top_srcdir)/src/voclWriteBufferProc.c \
					 $(top_srcdir)/src/voclPlatformIDProc.c \
					 $(top_srcdir)/src/voclDeviceIDProc.c \
					 $(top_srcdir)/src/voclContextProc.c \
					 $(top_srcdir)/src/voclCommandQueueProc.c \
					 $(top_srcdir)/src/voclKernelPrototype.c \
					 $(top_srcdir)/src/voclKernelProc.c \
					 $(top_srcdir)/src/voclKernelArgProc.c \
					 $(top_srcdir)/src/voclEventProc.c \
					 $(top_srcdir)/src/voclMemoryProc.c \
					 $(top_srcdir)/src/voclProxyNameProc.c \
					 $(top_srcdir)/src/voclProgramProc.c \
					 $(top_srcdir)/src/voclSamplerProc.c \
					 $(top_srcdir)/src/voclDynamicOpenCLLoad.c \
					 $(top_srcdir)/src/voclMigOpencl.c \
					 $(top_srcdir)/src/voclMigration.c \
					 $(top_srcdir)/src/voclMigrationDataTransfer.c \
					 $(top_srcdir)/src/voclMigBufferProc.c \
					 $(top_srcdir)/src/voclLoadBalance.c \
					 $(top_srcdir)/src/voclWinProc.c

bin_PROGRAMS = vocl_proxy bandwidth matrixmult_sp matrixmult_dp nbody_sp nbody_dp swat matrixtrans_sp matrixtrans_dp

vocl_proxy_SOURCES = $(top_srcdir)/src/vocl_proxy.c \
					 $(top_srcdir)/src/vocl_proxyFunctions.c \
					 $(top_srcdir)/src/vocl_proxyHelperThread.c \
					 $(top_srcdir)/src/vocl_proxyReadBufferProc.c \
					 $(top_srcdir)/src/vocl_proxyWriteBufferProc.c \
					 $(top_srcdir)/src/vocl_proxyCommAccept.c \
					 $(top_srcdir)/src/vocl_proxyMigrationDataTransfer.c \
					 $(top_srcdir)/src/vocl_proxyWinProc.c

bandwidth_SOURCES = $(top_srcdir)/examples/bandwidth/bandwidth.c \
	$(top_srcdir)/examples/bandwidth/bw_timer.c

matrixmult_sp_SOURCES = $(top_srcdir)/examples/matrixmult/matrixmult_sp.c \
	$(top_srcdir)/examples/matrixmult/mm_timer.c

matrixmult_dp_SOURCES = $(top_srcdir)/examples/matrixmult/matrixmult_dp.c \
	$(top_srcdir)/examples/matrixmult/mm_timer.c

nbody_sp_CPPFLAGS = -I$(top_srcdir)/examples/oclNbody/inc
nbody_sp_SOURCES = $(top_srcdir)/examples/oclNbody/src/nbody_timer.cpp \
	$(top_srcdir)/examples/oclNbody/src/oclBodySystemOpenclLaunch_sp.cpp \
	$(top_srcdir)/examples/oclNbody/src/oclBodySystemCpu_sp.cpp \
	$(top_srcdir)/examples/oclNbody/src/oclNbody_sp.cpp \
	$(top_srcdir)/examples/oclNbody/src/param_sp.cpp \
	$(top_srcdir)/examples/oclNbody/src/oclBodySystemOpencl_sp.cpp \
	$(top_srcdir)/examples/oclNbody/src/oclNbodyGold_sp.cpp

nbody_dp_CPPFLAGS = -I$(top_srcdir)/examples/oclNbody/inc
nbody_dp_SOURCES = $(top_srcdir)/examples/oclNbody/src/nbody_timer.cpp \
	$(top_srcdir)/examples/oclNbody/src/oclBodySystemOpenclLaunch_dp.cpp \
	$(top_srcdir)/examples/oclNbody/src/oclBodySystemCpu_dp.cpp \
	$(top_srcdir)/examples/oclNbody/src/oclNbody_dp.cpp \
	$(top_srcdir)/examples/oclNbody/src/param_dp.cpp \
	$(top_srcdir)/examples/oclNbody/src/oclBodySystemOpencl_dp.cpp \
	$(top_srcdir)/examples/oclNbody/src/oclNbodyGold_dp.cpp

swat_SOURCES = $(top_srcdir)/examples/swat/alignments.cpp \
	$(top_srcdir)/examples/swat/prints.cpp \
	$(top_srcdir)/examples/swat/sequences.cpp \
	$(top_srcdir)/examples/swat/swat.cpp \
	$(top_srcdir)/examples/swat/param.cpp \
	$(top_srcdir)/examples/swat/swat_timer.cpp

matrixtrans_sp_SOURCES = $(top_srcdir)/examples/matrixtrans/oclTranspose_sp.cpp \
	$(top_srcdir)/examples/matrixtrans/transpose_gold_sp.cpp \
	$(top_srcdir)/examples/matrixtrans/mt_timer.c

matrixtrans_dp_SOURCES = $(top_srcdir)/examples/matrixtrans/oclTranspose_dp.cpp \
	$(top_srcdir)/examples/matrixtrans/transpose_gold_dp.cpp \
	$(top_srcdir)/examples/matrixtrans/mt_timer.c
