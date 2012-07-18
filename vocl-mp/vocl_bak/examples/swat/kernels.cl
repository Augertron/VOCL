//#include "global.h"

typedef struct {
    int nposi, nposj;
    int nmaxpos;
    float fmaxscore;
    int noutputlen;
}   MAX_INFO;

#define PATH_END 0

//constant memory for score matrix
//__device__ __constant__ float blosum62C[529];
//__device__ __constant__ char instr1C[MAX_LEN];
//__device__ __constant__ char instr2C[MAX_LEN];

//void copyScoringMatrixToConstant()
//{
//	cudaMemcpyToSymbol(blosum62C, blosum62[0], sizeof(float) * 529,
//					   0, cudaMemcpyHostToDevice);
//	return;
//}

__kernel void MatchString(__global char *str_npathflagp,
						  __global char *str_nExtFlagp,
						  __global float *str_fngapdistp,
						  __global float *str_fhgapdistp,
						  __global float *str_fvgapdistp,
						  int nstartpos,
						  int nposi,
						  int nposj,
						  int npreprewidth,
						  int nprewidth,
						  int nthreadnum,
						  __constant char * const instr1D,
						  __constant char * const instr2D,
						  int ntableWidth,
						  float open_penalty, 
						  float ext_penalty,
						  __global MAX_INFO *strMaxInfop,
						  __constant float *const blosum62D)
{
	int npos, ntablepos;
	int npreposngap, npreposhgap, npreposvgap;
	int indexi, indexj;
	int tid = get_global_id(0);
	
	float fdist;
	float fdistngap, fdisthgap, fdistvgap;
	float ext_dist;
	float fmaxdist;

	if (tid >= nthreadnum)
	{
		return;
	}

	npos = nstartpos + tid;

	indexj = nposj + tid;
	indexi = nposi - tid;

	npreposhgap = npos - nprewidth;
	npreposvgap = npreposhgap - 1;
	npreposngap = npreposvgap - npreprewidth;

	ntablepos = instr1D[indexi] * ntableWidth + instr2D[indexj];
	//ntablepos = instr1C[indexi] * ntableWidth + instr2C[indexj];
	fdist = blosum62D[ntablepos];
	//fdist = blosum62C[ntablepos];

	fmaxdist = str_fngapdistp[npreposngap];
    fdistngap = fmaxdist + fdist;

	ext_dist  = str_fhgapdistp[npreposhgap] - ext_penalty;
	fdisthgap = str_fngapdistp[npreposhgap] - open_penalty;

	if (fdisthgap <= ext_dist)
	{
		fdisthgap = ext_dist;
		//str_npathflagp[npreposhgap] += 4;
		str_nExtFlagp[npreposhgap] = 1;
	}

	ext_dist  = str_fvgapdistp[npreposvgap] - ext_penalty;
	fdistvgap = str_fngapdistp[npreposvgap] - open_penalty;

	if (fdistvgap <= ext_dist)
	{
		fdistvgap = ext_dist;
		str_npathflagp[npreposvgap] += 8;
	}
	
	if (fdistngap < 0.0f)
	{
		fdistngap = 0.0f;
	}
	if (fdisthgap < 0.0f)
	{
		fdisthgap = 0.0f;
	}
	if (fdistvgap <	0.0f)
	{
		fdistvgap =	0.0f;
	}
	
	str_fhgapdistp[npos] = fdisthgap;
	str_fvgapdistp[npos] = fdistvgap;

	//priority 00, 01, 10
	if (fdistngap >= fdisthgap && fdistngap >= fdistvgap)
	{
		fmaxdist = fdistngap;
		str_npathflagp[npos] = 2;
	}
	else if (fdisthgap >= fdistngap && fdisthgap >= fdistvgap)
	{
		fmaxdist = fdisthgap;
		str_npathflagp[npos] = 1;
	}
	else //fdistvgap >= fdistngap && fdistvgap >= fdisthgap
	{
		fmaxdist = fdistvgap;
		str_npathflagp[npos] = 3;
	}

	str_fngapdistp[npos] = fmaxdist;

	//Here, the maximum match distance is 0, which means
	//previous alignment is useless
	if (fmaxdist <= 0.00000001f)
	{
		str_npathflagp[npos] = PATH_END;
	}

	if (strMaxInfop->fmaxscore < fmaxdist)
	{
		strMaxInfop->nposi = indexi + 1;
		strMaxInfop->nposj = indexj + 1;
		strMaxInfop->nmaxpos = npos;
		strMaxInfop->fmaxscore = fmaxdist;
	}
}

__kernel void trace_back2(__global char *str_npathflagp,
						  __global char *str_nExtFlagp,
						  __global int  *ndiffpos,
						  __global char *instr1D,
						  __global char *instr2D,
						  __global char *outstr1,
						  __global char *outstr2,
						  __global MAX_INFO * strMaxInfop)
{
	int i, j;
	int npos, nlen;
	int npathflag;
	int nlaunchno;
	
	npos = strMaxInfop->nmaxpos;
	npathflag = str_npathflagp[npos] & 0x3;
	nlen = 0;

	i = strMaxInfop->nposi;
	j = strMaxInfop->nposj;
	nlaunchno = i + j;

	while (1)
	{
		if (npathflag == 3)
		{
			outstr1[nlen] = 23;
			outstr2[nlen] = instr2D[j - 1];
			nlen++;
			j--;

			//position in the transformed matrix
			npos = npos - ndiffpos[nlaunchno] - 1;
			nlaunchno--;
		}
		else if (npathflag == 1)
		{
			outstr1[nlen] = instr1D[i - 1];
			outstr2[nlen] = 23;
			nlen++;
			i--;

			//position in the transformed matrix
			npos = npos - ndiffpos[nlaunchno];
			nlaunchno--;
		}
		else if (npathflag == 2)
		{
			outstr1[nlen] = instr1D[i - 1];
			outstr2[nlen] = instr2D[j - 1];
			nlen++;
			i--;
			j--;

			//position in the transformed matrix
			npos = npos - ndiffpos[nlaunchno] - ndiffpos[nlaunchno - 1] - 1;
			nlaunchno = nlaunchno - 2;
		}
		else
		{
			//printf("npathflag = %d, npos = %d\n", npathflag, npos);
			//printf("find path error!\n");
			return;
		}

		//only if it is not an extension gap, will the path direction change
		//otherwise, back on the same direction
		int nExtFlag = str_npathflagp[npos] / 4;
		if (npathflag == 3 && (nExtFlag == 2 || nExtFlag == 3))
		{
			npathflag = 3;
		}
		//else if (npathflag == 1 && (nExtFlag == 1 || nExtFlag == 3))
		else if (npathflag == 1 && str_nExtFlagp[npos] == 1)
		{
			npathflag = 1;
		}
		else
		{
			npathflag = str_npathflagp[npos] & 0x3;
		}

		if (i == 0 || j == 0)
		{
			break;
		}

		if (npathflag == PATH_END)
		{
			break;
		}
	}

	i--;
	j--;

	while(i >= 0)
	{
		outstr1[nlen] = instr1D[i];
		outstr2[nlen] = 23;
		nlen++;
		i--;
	}

	while(j >= 0)
	{
		outstr1[nlen] = 23;
		outstr2[nlen] = instr2D[j];
		nlen++; 
		j--;
	}

	strMaxInfop->noutputlen = nlen;

	return;
}

__kernel void setZeroSwat(__global char *a,
                      int arraySize)
{
    int index = get_global_id(0);
    if (index < arraySize)
    {
        a[index] = 0;
    }

	return;
}

