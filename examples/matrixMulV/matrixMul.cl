#ifdef cl_amd_fp64
	#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

__kernel void
matrixMul(__global double *a,
		  __global double *b,
		  __global double *c,
		  int  			 hA,
		  int			 wA,
		  int			 wB)
{
	int i, j, k;

	j = get_global_id(0);
	i = get_global_id(1);
	double cSub;
	
	if (j >= wB)
	{
		return;
	}

	cSub = 0.0;
	for (k = 0; k < wA; k++)
	{
		cSub += a[i * wA + k] * b[k * wB + j];
	}

	c[i * wB + j] += cSub;
}

//__kernel void
//matrixMul(__global double *in_matrix,
//			 __global double *out_matrix,
//			 int  			width,
//			 int			height)
//{
//	int i, j;
//	int matrixWidth = width + 2;
//
//	i = get_global_id(0) + 1;
//	j = get_global_id(1) + 1;
//	
//	if (i >= width + 1)
//	{
//		return;
//	}
//
//	int sum = in_matrix[(i-1) * matrixWidth + j-1] + 
//			  in_matrix[(i-1) * matrixWidth + j  ] + 
//			  in_matrix[(i-1) * matrixWidth + j+1] +
//			  in_matrix[ i    * matrixWidth + j-1] +
//			  in_matrix[ i    * matrixWidth + j+1] +
//			  in_matrix[(i+1) * matrixWidth + j-1] + 
//			  in_matrix[(i+1) * matrixWidth + j  ] + 
//			  in_matrix[(i+1) * matrixWidth + j+1];
//
//	if (in_matrix[i * matrixWidth + j] == 0 && sum == 3) 
//	{
//		out_matrix[i * matrixWidth + j] = 1;
//	}else{
//		if (sum < 2 || sum > 3)
//			out_matrix[i * matrixWidth + j] = 0;
//		else
//			out_matrix[i * matrixWidth + j] = in_matrix[i * matrixWidth + j];
//	}
//}


