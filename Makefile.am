#
# Copyright (C) 2010. See COPYRIGHT in top-level directory.
#

ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(top_srcdir)/src

lib_LTLIBRARIES = libvocl.la
libvocl_la_SOURCES = $(top_srcdir)/src/vocl.cpp

bin_PROGRAMS = vocl_proxy bandwidth
vocl_proxy_SOURCES = $(top_srcdir)/src/vocl_proxy.cpp