#include "vocl_structures.h"

//static unsigned int *voclObjCountPtr = NULL;
//static unsigned int voclObjCountNum;
//static unsigned int voclObjCountNo;
//
//void voclObjCountInitialize()
//{
//	voclObjCountNum = VOCL_OBJ_COUNT_NUM;
//	voclObjCountNo = 0;
//	voclObjCountPtr = (unsigned int *)malloc(sizeof(unsigned int) * voclObjCountNum);
//	memset(voclObjCountPtr, 0, sizeof(unsigned int) * voclObjCountNum);
//}
//
//void voclObjCountFinalize()
//{
//	voclObjCountNum = 0;
//	voclObjCountNo = 0;
//
//	if (voclObjCountPtr != NULL) {
//		free(voclObjCountPtr);
//		voclObjCountPtr = NULL:
//	}
//}
//
//void increaseObjCount(int proxyID)
//{
//	if (proxyID >= voclObjCountNum)
//	{
//		voclObjCountPtr = (unsigned int *)realloc(voclObjCountPtr, sizeof(unsigned int) * 2 * voclObjCountNum);
//		memset(&voclObjCountNum[voclObjCountNum], 0, voclObjCountNum * sizeof(unsigned int));
//		voclObjCountNum *= 2;
//	}
//
//	voclObjCountNum[proxyID]++;
//}
//
//void decreaseObjCount(int proxyID)
//{
//	voclObjCountNum[proxyID]--;
//}
