#ifndef __VOCL_FUNC_H__
#define __VOCL_FUNC_H__

#include <CL/opencl.h>

struct strVoclRebalance {
	cl_command_queue command_queue;
	int isMigrated;
	int reissueWriteNum;
	int reissueReadNum;
};

void voclRebalance(cl_command_queue);

typedef void 
	(*dlVOCLRebalance) (cl_command_queue queue);

#endif
