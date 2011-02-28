const char *cl_source_ratt6 =
"#ifdef K_DOUBLE_PRECISION\n"
"#define DOUBLE_PRECISION\n"
"#pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
"#elif AMD_DOUBLE_PRECISION\n"
"#define DOUBLE_PRECISION\n"
"#pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
"#endif\n"
"\n"
"//replace divisions by multiplication with the reciprocal\n"
"#define REPLACE_DIV_WITH_RCP 1\n"
"\n"
"//Call the appropriate math function based on precision\n"
"#ifdef DOUBLE_PRECISION\n"
"#define real double\n"
"#if REPLACE_DIV_WITH_RCP\n"
"#define DIV(x,y) ((x)*(1.0/(y)))\n"
"#else\n"
"#define DIV(x,y) ((x)/(y))\n"
"#endif\n"
"#define POW pow\n"
"#define EXP exp\n"
"#define EXP10 exp10\n"
"#define EXP2 exp2\n"
"#define MAX fmax\n"
"#define MIN fmin\n"
"#define LOG log\n"
"#define LOG10 log10\n"
"#else\n"
"#define real float\n"
"#if REPLACE_DIV_WITH_RCP\n"
"#define DIV(x,y) ((x)*(1.0f/(y)))\n"
"#else\n"
"#define DIV(x,y) ((x)/(y))\n"
"#endif\n"
"#define POW pow\n"
"#define EXP exp\n"
"#define EXP10 exp10\n"
"#define EXP2 exp2\n"
"#define MAX fmax\n"
"#define MIN fmin\n"
"#define LOG log\n"
"#define LOG10 log10\n"
"#endif\n"
"\n"
"//Kernel indexing macros\n"
"#define thread_num (get_global_id(0))\n"
"#define idx2(p,z) (p[(((z)-1)*(N_GP)) + thread_num])\n"
"#define idx(x, y) ((x)[(y)-1])\n"
"#define C(q)     idx2(C, q)\n"
"#define Y(q)     idx2(Y, q)\n"
"#define RF(q)    idx2(RF, q)\n"
"#define EG(q)    idx2(EG, q)\n"
"#define RB(q)    idx2(RB, q)\n"
"#define RKLOW(q) idx2(RKLOW, q)\n"
"#define ROP(q)   idx(ROP, q)\n"
"#define WDOT(q)  idx2(WDOT, q)\n"
"#define RKF(q)   idx2(RKF, q)\n"
"#define RKR(q)   idx2(RKR, q)\n"
"#define A_DIM    (11)\n"
"#define A(b, c)  idx2(A, (((b)*A_DIM)+c) )\n"
"\n"
"\n"
"__kernel void\n"
"ratt6_kernel(__global const real* T, __global const real* RF,\n"
"		__global real* RB, __global const real* EG, const real TCONV)\n"
"{\n"
"\n"
"    const real TEMP = T[get_global_id(0)]*TCONV;\n"
"    const real ALOGT = LOG(TEMP);\n"
"#ifdef DOUBLE_PRECISION\n"
"    const real SMALL_INV = 1e+300;\n"
"#else \n"
"    const real SMALL_INV = 1e+20f;\n"
"#endif\n"
"\n"
"    const real RU=8.31451e7;\n"
"    const real PATM = 1.01325e6;\n"
"    const real PFAC = DIV (PATM, (RU*(TEMP)));\n"
"\n"
"    real rtemp_inv;\n"
"\n"
"    rtemp_inv = DIV ((EG(4)*EG(18)), (EG(7)*EG(17)));\n"
"    RB(101) = RF(101) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV ((EG(2)*EG(13)), (EG(1)*EG(12)));\n"
"    RB(102) = RF(102) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV ((EG(3)*EG(13)), (EG(5)*EG(12)));\n"
"    RB(103) = RF(103) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV ((EG(5)*EG(13)), (EG(6)*EG(12)));\n"
"    RB(104) = RF(104) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV ((EG(9)*EG(13)), (EG(2)*EG(22)));\n"
"    RB(105) = RF(105) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV ((EG(10)*EG(13)), (EG(12)*EG(12)));\n"
"    RB(106) = RF(106) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV ((EG(11)*EG(13)), (EG(12)*EG(12)));\n"
"    RB(107) = RF(107) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV ((EG(2)*EG(25)), (EG(11)*EG(14)));\n"
"    RB(108) = RF(108) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV ((EG(3)*EG(25)), (EG(2)*EG(14)*EG(14)*PFAC));\n"
"    RB(109) = RF(109) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV ((EG(4)*EG(25)), (EG(5)*EG(14)*EG(14)*PFAC));\n"
"    RB(110) = RF(110) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV ((EG(9)*EG(25)), (EG(14)*EG(19)));\n"
"    RB(111) = RF(111) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV ((EG(10)*EG(25)), (EG(14)*EG(21)));\n"
"    RB(112) = RF(112) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV ((EG(25)*EG(25)), (EG(14)*EG(14)*EG(19)*PFAC));\n"
"    RB(113) = RF(113) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV (EG(19), EG(20));\n"
"    RB(114) = RF(114) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV (EG(21), (EG(2)*EG(19)*PFAC));\n"
"    RB(115) = RF(115) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV ((EG(3)*EG(19)), (EG(2)*EG(25)));\n"
"    RB(116) = RF(116) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV ((EG(3)*EG(19)), (EG(10)*EG(14)));\n"
"    RB(117) = RF(117) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV ((EG(5)*EG(19)), (EG(2)*EG(26)));\n"
"    RB(118) = RF(118) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV ((EG(5)*EG(19)), (EG(12)*EG(14)));\n"
"    RB(119) = RF(119) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV ((EG(16)*EG(19)), (EG(14)*EG(21)));\n"
"    RB(120) = RF(120) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV ((EG(12)*EG(19)*PFAC), EG(29));\n"
"    RB(121) = RF(121) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV (EG(19), EG(20));\n"
"    RB(122) = RF(122) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV ((EG(3)*EG(20)), (EG(10)*EG(14)));\n"
"    RB(123) = RF(123) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV ((EG(5)*EG(20)), (EG(2)*EG(26)));\n"
"    RB(124) = RF(124) * MIN(rtemp_inv, SMALL_INV);\n"
"\n"
"    rtemp_inv = DIV ((EG(4)*EG(20)), (EG(10)*EG(15)));\n"
"    RB(125) = RF(125) * MIN(rtemp_inv, SMALL_INV);\n"
"}\n"
;
