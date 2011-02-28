const char *cl_source_sgemmN =
"// Code derived from work done by the authors quoted in the original header below:\n"
"\n"
"//\n"
"// (c) January 24, 2008 Vasily Volkov @ UC Berkeley\n"
"//\n"
"// Other credits:\n"
"// - Paul Leventis @ Altera Corp. for prefetching and -maxrregcount techniques\n"
"// - many thanks to Wladimir J. van der Laan @ the University of Groningen\n"
"// for his cubin disassembler (http://www.cs.rug.nl/~wladimir/decuda/)\n"
"//\n"
"//\n"
"\n"
"#ifdef SINGLE_PRECISION\n"
"#define FPTYPE float\n"
"#elif K_DOUBLE_PRECISION\n"
"#pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
"#define FPTYPE double\n"
"#elif AMD_DOUBLE_PRECISION\n"
"#pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
"#define FPTYPE double\n"
"#endif\n"
"\n"
"#define SAXPY( _A_, _BS_ , _C_) do{ \\\n"
"	_C_[0] += _A_ * _BS_[0]; \\\n"
"	_C_[1] += _A_ * _BS_[1]; \\\n"
"	_C_[2] += _A_ * _BS_[2]; \\\n"
"	_C_[3] += _A_ * _BS_[3]; \\\n"
"	_C_[4] += _A_ * _BS_[4]; \\\n"
"	_C_[5] += _A_ * _BS_[5]; \\\n"
"	_C_[6] += _A_ * _BS_[6]; \\\n"
"	_C_[7] += _A_ * _BS_[7]; \\\n"
"	_C_[8] += _A_ * _BS_[8]; \\\n"
"	_C_[9] += _A_ * _BS_[9]; \\\n"
"	_C_[10] += _A_ * _BS_[10]; \\\n"
"	_C_[11] += _A_ * _BS_[11]; \\\n"
"	_C_[12] += _A_ * _BS_[12]; \\\n"
"	_C_[13] += _A_ * _BS_[13]; \\\n"
"	_C_[14] += _A_ * _BS_[14]; \\\n"
"	_C_[15] += _A_ * _BS_[15]; \\\n"
"    }while(0)\n"
"\n"
"__kernel void sgemmNT( __global const FPTYPE *A, int lda,\n"
"                       __global const FPTYPE *B, int ldb,\n"
"                       __global FPTYPE *C, int ldc, int k,\n"
"                       FPTYPE alpha, FPTYPE beta )\n"
"{\n"
"	const int inx = get_local_id(0);\n"
"	const int iny = get_local_id(1);\n"
"	const int ibx = get_group_id(0) * 64;\n"
"	const int iby = get_group_id(1) * 16;\n"
"	const int id  = inx + iny*16;\n"
"\n"
"        int i, counter = 0;\n"
"\n"
"	A += ibx + id;\n"
"	B += iby + inx + (iny*ldb);\n"
"	C += ibx + id  + (iby*ldc );\n"
"	\n"
"	FPTYPE a[4];\n"
"	for(i=0; i<4; ++i){ a[i] = A[i*lda]; }\n"
"	__private FPTYPE b;\n"
"	b = B[0];\n"
"\n"
"	A += 4*lda;\n"
"	B += 4*ldb;\n"
"        counter+= 4*ldb;\n"
"    \n"
"	__local FPTYPE bs[4][16];\n"
"	FPTYPE c[16];\n"
"        for(i=0; i<16; ++i){\n"
"            c[i] = 0.0;\n"
"        }\n"
"    \n"
"	do\n"
"	{\n"
"	        __private FPTYPE as[4];\n"
"		for(i=0; i<4; ++i){ as[i] = a[i]; }\n"
"		\n"
"		bs[iny][inx] = b;\n"
"  		barrier(CLK_LOCAL_MEM_FENCE);\n"
"		\n"
"		a[0] = A[0*lda];\n"
"		a[1] = A[1*lda];\n"
"		a[2] = A[2*lda];\n"
"		a[3] = A[3*lda];\n"
"		b    = B[0];\n"
"		\n"
"		SAXPY( as[0], bs[0], c );\n"
"		SAXPY( as[1], bs[1], c );\n"
"		SAXPY( as[2], bs[2], c );\n"
"		SAXPY( as[3], bs[3], c );\n"
"\n"
"		A += 4*lda;\n"
"		B += 4*ldb;\n"
"                counter += 4*ldb;\n"
"  		barrier(CLK_LOCAL_MEM_FENCE);\n"
"		\n"
"	} while( counter < k*ldb );\n"
"	\n"
"	bs[iny][inx] = b;\n"
"	barrier(CLK_LOCAL_MEM_FENCE);\n"
"	\n"
"	SAXPY( a[0], bs[0], c );\n"
"	SAXPY( a[1], bs[1], c );\n"
"	SAXPY( a[2], bs[2], c );\n"
"	SAXPY( a[3], bs[3], c );\n"
"\n"
"	for( int i = 0; i < 16; i++, C += ldc ){\n"
"		C[0] = alpha*c[i] + beta*C[0];\n"
"        }\n"
"}\n"
"\n"
"\n"
"__kernel void sgemmNN( __global const FPTYPE *A, int lda,\n"
"                       __global const FPTYPE *B, int ldb,\n"
"                       __global FPTYPE *C, int ldc, int k,\n"
"                       FPTYPE alpha, FPTYPE beta )\n"
"{\n"
"	const int inx = get_local_id(0);\n"
"	const int iny = get_local_id(1);\n"
"	const int ibx = get_group_id(0) * 64;\n"
"	const int iby = get_group_id(1) * 16;\n"
"	const int id = inx + iny*16;\n"
"	\n"
"        int i, j, ii, counter=0;\n"
"\n"
"	A += ibx + id;\n"
"\n"
"	B += inx + (iby+iny) * ldb;\n"
"\n"
"	C += ibx + id  + (iby*ldc);\n"
"	\n"
"	FPTYPE c[16];\n"
"        for(i=0; i<16; ++i){\n"
"            c[i] = 0.0;\n"
"	}\n"
"\n"
"       	__local FPTYPE bs[16][17];\n"
"\n"
"	do\n"
"	{\n"
"		__private FPTYPE a[4];\n"
"		for(ii=0; ii<4; ++ii) { a[ii] = A[ii*lda]; }\n"
"\n"
"		bs[inx][iny]    = B[0*ldb];\n"
"		bs[inx][iny+4]  = B[4*ldb];\n"
"		bs[inx][iny+8]  = B[8*ldb];\n"
"		bs[inx][iny+12] = B[12*ldb];\n"
"		barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"		A += 4*lda;\n"
"\n"
"		SAXPY( a[0], bs[0], c );	a[0] = A[0*lda];\n"
"		SAXPY( a[1], bs[1], c );	a[1] = A[1*lda];\n"
"		SAXPY( a[2], bs[2], c );	a[2] = A[2*lda];\n"
"		SAXPY( a[3], bs[3], c );	a[3] = A[3*lda];	\n"
" \n"
"		A += 4*lda;\n"
"		SAXPY( a[0], bs[4], c );	a[0] = A[0*lda];\n"
"		SAXPY( a[1], bs[5], c );	a[1] = A[1*lda];\n"
"		SAXPY( a[2], bs[6], c );	a[2] = A[2*lda];\n"
"		SAXPY( a[3], bs[7], c );	a[3] = A[3*lda];\n"
"		\n"
"		A += 4*lda;\n"
"		SAXPY( a[0], bs[8], c );	a[0] = A[0*lda];\n"
"		SAXPY( a[1], bs[9], c );	a[1] = A[1*lda];\n"
"		SAXPY( a[2], bs[10], c );	a[2] = A[2*lda];\n"
"		SAXPY( a[3], bs[11], c );	a[3] = A[3*lda];\n"
"		\n"
"		A += 4*lda;\n"
"		SAXPY( a[0], bs[12], c );\n"
"		SAXPY( a[1], bs[13], c );\n"
"		SAXPY( a[2], bs[14], c );\n"
"		SAXPY( a[3], bs[15], c );\n"
"\n"
"		B += 16;\n"
"	        counter += 16;\n"
"		barrier(CLK_LOCAL_MEM_FENCE);\n"
"	} while( counter < k );\n"
"	\n"
"	for( int i = 0; i < 16; i++, C += ldc ){\n"
"		C[0] = alpha*c[i] + beta*C[0]; \n"
"	}\n"
"\n"
"}	\n"
"\n"
;
