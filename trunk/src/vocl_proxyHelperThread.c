#define _GNU_SOURCE
#include <CL/opencl.h>
#include "pthread.h"
#include "vocl_proxy.h"
#include "vocl_proxy_macro.h"
#include <sched.h>

pthread_barrier_t barrier;
pthread_t th;
int helperThreadOperFlag = GPU_MEM_NULL;
int writeBufferIndexInHelperThread = 0;

extern void thrSentToLocalNode(void *p);
extern void thrWriteToGPUMemory(void *p);
extern cl_int writeToGPUMemory(int index);
extern cl_int enqueuePreviousWrites();
extern void sendReadyReadBufferToLocal();

void *proxyHelperThread(void *p)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(10, &set);
    sched_setaffinity(0, sizeof(set), &set);

    helperThreadOperFlag = GPU_MEM_NULL;
    pthread_barrier_wait(&barrier);
    pthread_barrier_wait(&barrier);
    while (helperThreadOperFlag != GPU_MEM_NULL) {
        if (helperThreadOperFlag == GPU_MEM_READ) {
            thrSentToLocalNode(NULL);
        }
        else if (helperThreadOperFlag == GPU_MEM_WRITE) {
            thrWriteToGPUMemory(NULL);
        }
        else if (helperThreadOperFlag == GPU_WRITE_SINGLE) {
            writeToGPUMemory(writeBufferIndexInHelperThread);
        }
        else if (helperThreadOperFlag == GPU_ENQ_WRITE) {
            enqueuePreviousWrites();
        }
        else if (helperThreadOperFlag == SEND_LOCAL_PREVIOUS) {
            sendReadyReadBufferToLocal();
        }

        helperThreadOperFlag = GPU_MEM_NULL;
        pthread_barrier_wait(&barrier);
        pthread_barrier_wait(&barrier);
    }

    return;
}