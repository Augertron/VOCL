const char *cl_source_rdwdot10 =
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
"#define ROP2(a)  (RKF(a) - RKR (a))\n"
"\n"
"\n"
"__kernel void\n"
"rdwdot10_kernel (__global const real* RKF, __global const real* RKR,\n"
"		__global real* WDOT, const real rateconv, __global const real* molwt)\n"
"{\n"
" \n"
"    real ROP12 = ROP2(12) + ROP2(13) + ROP2(14)+ ROP2(15);\n"
"    real ROP22 = ROP2(22) + ROP2(23);\n"
"    real ROP27 = ROP2(27) + ROP2(28);\n"
"    real ROP5 = ROP2(5) + ROP2(6) + ROP2(7) + ROP2(8);\n"
"\n"
"    WDOT(1) = (-ROP2(2) -ROP2(3) +ROP5 +ROP2(18)\n"
"            +ROP2(24) -ROP2(31) -ROP2(36) +ROP2(42)\n"
"            -ROP2(49) +ROP2(58) +ROP2(60) +ROP2(61)\n"
"            -ROP2(64) +ROP2(72) +ROP2(96) +ROP2(102)\n"
"            +ROP2(127) +ROP2(133) +ROP2(134) +ROP2(150)\n"
"            +ROP2(155) +ROP2(157) +ROP2(171) +ROP2(180)\n"
"            +ROP2(192) +ROP2(200))*rateconv*molwt[0] ;\n"
"\n"
"    WDOT(3) = (+ROP2(1) -ROP2(2) +ROP2(4) -ROP2(10)\n"
"            -ROP2(11) -ROP2(11) +ROP2(17) -ROP2(20)\n"
"            -ROP2(26) -ROP2(29) +ROP2(32) -ROP2(34)\n"
"            +ROP2(38) -ROP2(43) -ROP2(44) -ROP2(50)\n"
"            -ROP2(61) -ROP2(62) -ROP2(73) -ROP2(79)\n"
"            +ROP2(82) -ROP2(99) -ROP2(103) -ROP2(109)\n"
"            -ROP2(116) -ROP2(117) -ROP2(123) -ROP2(129)\n"
"            -ROP2(130) -ROP2(135) -ROP2(136) +ROP2(139)\n"
"            -ROP2(151) -ROP2(158) -ROP2(159) -ROP2(160)\n"
"            -ROP2(172) -ROP2(173) -ROP2(181) -ROP2(193)\n"
"            -ROP2(194) -ROP2(195) -ROP2(201))*rateconv *molwt[2];\n"
"\n"
"    WDOT(4) = (-ROP2(1) +ROP2(11) -ROP12 +ROP2(18)\n"
"            +ROP2(20) +ROP2(21) +ROP22 -ROP2(32)\n"
"            -ROP2(38) -ROP2(47) -ROP2(51) -ROP2(52)\n"
"            -ROP2(65) -ROP2(66) -ROP2(75) -ROP2(82)\n"
"            -ROP2(83) +ROP2(84) -ROP2(101) -ROP2(110)\n"
"            -ROP2(125) -ROP2(138) -ROP2(139) -ROP2(140)\n"
"            -ROP2(153) -ROP2(154) -ROP2(162) -ROP2(174)\n"
"            +ROP2(175) +ROP2(187) -ROP2(203))*rateconv *molwt[3];\n"
"    WDOT(6) = (+ROP2(3) +ROP2(4) +ROP2(9) +ROP2(17)\n"
"            +ROP2(21) +ROP2(25) +ROP27 -ROP2(37)\n"
"            +ROP2(45) +ROP2(54) +ROP2(66) +ROP2(74)\n"
"            +ROP2(80) +ROP2(81) +ROP2(98) +ROP2(100)\n"
"            +ROP2(104) +ROP2(131) +ROP2(137) +ROP2(152)\n"
"            +ROP2(161) +ROP2(182) +ROP2(196) +ROP2(202))*rateconv *molwt[5];\n"
"}\n"
;
