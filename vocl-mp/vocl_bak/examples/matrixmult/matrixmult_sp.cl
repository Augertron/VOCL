#ifdef cl_amd_fp64
	#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define BLOCK_SIZE 16
#define REAL float
#define As(i, j) As[i * (BLOCK_SIZE + 1) + j]
#define Bs(i, j) Bs[i * (BLOCK_SIZE + 1) + j]


__kernel void matrixMul(__global REAL *A, 
				   __global REAL *B, 
				   __global REAL *C,
				   __local  REAL *As,
				   __local  REAL *Bs,
				   int            hA, 
				   int            wA, 
				   int            wB)
{
   //Block index
   int bx = get_group_id(0);
   int by = get_group_id(1);

   //Thread index
   int tx = get_local_id(0);
   int ty = get_local_id(1);

   //Index of the first sbu-matrix of A processed by teh block
   int aBegin = wA * BLOCK_SIZE * by;

   //Index of the last sbu-matrix of A processed by the block
   int aEnd = aBegin + wA - 1;

   //Step size used to iterate throught the sub-matrxi of A
   int aStep = BLOCK_SIZE;

   //Index of the first sub-matrix of B processed by the block
   int bBegin = BLOCK_SIZE * bx;

   //Step size used to iterate through the sub-matrices of B
   int bStep = BLOCK_SIZE * wB;
   
   //The element of the block sub-matrix that is computed by the thread
   REAL Csub = 0;

   //Loop over all the sub-matrix of A and B required to 
   //compute the block sub-matrix
   for (int a = aBegin, b = bBegin;
   		         a < aEnd;
   		         a += aStep, b += bStep)
   {
       //Shared memeory of the sub-matrix of A
       //__local REAL As[BLOCK_SIZE][BLOCK_SIZE + 1];

       //Shared memory for the sub-matrix of B
       //__local REAL Bs[BLOCK_SIZE][BLOCK_SIZE + 1];
   
       //Load the matrices from global memory to shared memory
       //each thread loads one element of each matrix

       //As[ty][tx] = A[a + wA * ty + tx];
       //Bs[ty][tx] = B[b + wB * ty + tx];
       As(ty, tx) = A[a + wA * ty + tx];
       Bs(ty, tx) = B[b + wB * ty + tx];

       //Synchronize to make sure the matrices are loaded
       barrier(CLK_LOCAL_MEM_FENCE);

       //Multiply the two matrices together;
       //each thread  computes one element
       //of the blocksub-matrix
       for (int k = 0; k < BLOCK_SIZE; ++k)
           //Csub += As[ty][k] * Bs[k][tx];
           Csub += As(ty, k) * Bs(k, tx);

       //Synchronize to make sure the preceding computation is done before loading two 
       //new sub-matrices of A and B in the next iteration
       barrier(CLK_LOCAL_MEM_FENCE);
   }

   //Write the block sub-matrix to global memory
   //each thread writes one element
   int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
   C[c+ wB * ty + tx] = Csub;
}

