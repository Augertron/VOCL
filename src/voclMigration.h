#ifndef __VOCL_MIGRATION_H__
#define __VOCL_MIGRATION_H__
#include <stdio.h>
#include <stdlib.h>
#include "vocl_structures.h"

struct strMigWriteLocalBuffer {
    cl_command_queue cmdQueue;
    cl_mem mem;
    //int source;
    int tag;
    size_t size;
    size_t offset;
    int useFlag;
    cl_event event;
    char *ptr;
    MPI_Comm comm;
    MPI_Request request;
};

/* read buffer */
struct strMigReadLocalBuffer {
    int dest;
    int tag;
    size_t size;
    size_t offset;
    int useFlag;
    cl_event event;
    char *ptr;
    MPI_Comm comm;
    MPI_Comm commData;
    MPI_Request request;
};

struct strMigRWLocalBuffer {
    cl_command_queue wtCmdQueue;
    cl_mem wtMem;
    size_t size;
    size_t offset;
    int useFlag;
    char *ptr;
    cl_event rdEvent;
    cl_event wtEvent;
};

#endif
