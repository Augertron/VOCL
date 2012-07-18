#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* for removing comments */
#define VOCL_COMPILER_PROGRAM 0
#define PRE_COMM              1
#define COMMENT1              2
#define COMMENT2              3
#define END_COMM              4

/* function to remove comments in C code */
void voclRemoveComments(char *sourceIn, size_t inLen, char *sourceOut, size_t * outLen)
{
    int state = VOCL_COMPILER_PROGRAM;
    int i;
    size_t len = 0;
    for (i = 0; i < inLen; i++) {
        switch (state) {
        case VOCL_COMPILER_PROGRAM:
            if (sourceIn[i] == '/') {
                state = PRE_COMM;
            }
            else {
                /* record the character */
                sourceOut[len++] = sourceIn[i];
            }
            break;
        case PRE_COMM:
            if (sourceIn[i] == '*') {
                state = COMMENT1;
            }
            else if (sourceIn[i] == '/') {
                state = COMMENT2;
            }
            else {
                state = VOCL_COMPILER_PROGRAM;
                sourceOut[len++] = sourceIn[i - 1];
                sourceOut[len++] = sourceIn[i];
            }
            break;
        case COMMENT1:
            if (sourceIn[i] == '*') {
                state = END_COMM;
            }
            break;
        case COMMENT2:
            if (sourceIn[i] == '\n') {
                state = VOCL_COMPILER_PROGRAM;
            }
            break;
        case END_COMM:
            if (sourceIn[i] == '/') {
                state = VOCL_COMPILER_PROGRAM;
            }
            else if (sourceIn[i] == '*') {
            }
            else {
                state = COMMENT1;
            }
            break;
        }
    }

    *outLen = len;

    return;
}

/* get indexes of global memeory arguments */
char *voclKernelPrototye(char *sourceIn, const char *kernelName, unsigned int *kernelArgNum)
{
    char *tokenStart, *tokenEnd, *args, *arg, *sourcePtr, *kernelNamePtr;
    char *argFlag;
    size_t nameSize, argSize;
    unsigned int argIndex, tmpArgNum = 100;
    int kernelFoundFlag = 0;    /* not found yet */

    argFlag = (char *) malloc(sizeof(char) * tmpArgNum);

    /* before the kernel name is the "__kernel" and "void" */
    sourcePtr = sourceIn;
    //printf("%s\n", sourceIn);
    while (kernelFoundFlag == 0) {
        tokenStart = strstr(sourcePtr, "__kernel");
        if (tokenStart == NULL)
            break;
        tokenEnd = strtok(tokenStart, "(");
        args = strtok(NULL, ")");

        /* search kernel name */
        kernelNamePtr = strstr(tokenEnd, kernelName);
        if (kernelNamePtr != NULL && strlen(kernelNamePtr) == strlen(kernelName)) {
            argIndex = 0;
            if (args != NULL) {
                arg = strtok(args, ",\0");
                if (strstr(arg, "__global") || strstr(arg, "__constant")) {
                    argFlag[argIndex] = 1;
                }
                else {
                    argFlag[argIndex] = 0;
                }
                argIndex++;

                while (1) {
                    arg = strtok(NULL, ",\0");
                    if (arg == NULL) {
                        break;
                    }
                    if (strstr(arg, "__global") || strstr(arg, "__constant")) {
                        argFlag[argIndex] = 1;
                    }
                    else {
                        argFlag[argIndex] = 0;
                    }
                    argIndex++;
                    if (argIndex >= tmpArgNum) {
                        argFlag = (char *) realloc(argFlag, sizeof(char) * tmpArgNum * 2);
                        tmpArgNum *= 2;
                    }
                }
            }
            kernelFoundFlag = 1;
        }
        /* not the kernel we search, search the next one */
        sourcePtr = args + strlen(args) + 1;
    }

    *kernelArgNum = argIndex;

    return argFlag;
}
