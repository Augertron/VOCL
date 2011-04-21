#ifndef __VOCL_EVENT_PROC_H__
#define __VOCL_EVENT_PROC_H__

#include <CL/opencl.h>

#define vocl_event long long

/* allocate elements in the buffer */
/* if not enough, re-allocate */
#define VOCL_EVENT_NUM 1000

/* map between real opencl event and vocl event */
struct strVOCLEventBuffer {
	cl_event    clEvent;
	/* long long int is used to indicate vocl event */
	vocl_event  voclEvent;
};



#endif
