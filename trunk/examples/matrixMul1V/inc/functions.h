/* vim: set sts=4 sw=4 expandtab:*/
/*
 * =====================================================================================
 *
 *       Filename:  game-of-life.c
 *
 *    Description:  Conway's Game of Life
 *
 *        Version:  1.0
 *        Created:  11/04/10 00:22:53
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Tom Scogland (), 
 *        Company:  
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

char * allocate_matrix(int width, int height);
void update_matrix(char ** in_matrix, int width, int height, int steps);
char * allocate_matrix(int width, int height);
void read_matrix(const char * filename, char ** matrix, int *width, int *height);
void print_matrix(const char * filename, char * matrix, int width, int height);

