#ifndef SCAN_H_
#define SCAN_H_

// Block size
static const int BLOCK_SIZE = 256;

template <class T, class vecT>
void
scanArrayRecursive(T* outArray, T* inArray, int numElements,
                   int level, T** blockSums);

template <class T>
bool scanCPU(T *data, T* reference, T* dev_result, const size_t size);

template <class T, class vecT>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op);

#endif // SCAN_H_
