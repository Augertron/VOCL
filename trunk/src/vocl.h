#ifndef __VOCL_FUNC_H__
#define __VOCL_FUNC_H__

#include <CL/opencl.h>

struct strVoclRebalance {
	cl_command_queue command_queue;
	int isMigrated;
	int reissueWriteNum;
	int reissueReadNum;
};

void vocl_rebalance(cl_command_queue);

#endif
