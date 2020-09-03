#ifndef FD_H
#define FD_H

#include "fpga.h"

#define PERF_COUNTERS

void op_L(int nz, int nx, float **p, float **laplace, int order);
void fd_init(int order, int nx, int nz, float dx, float dz, float dt);
void fd_reinit(int order, int nx, int nz);
void fd_destroy();
float *calc_coefs(int order);

#ifdef LCORE_FPGA
void fd_step(ssize_t fid, int order, float **p, float **pp, float **v2, int nz, int nx,
		int log);
#else
void fd_step(int order, float **p, float **pp, float **v2, int nz, int nx);
#endif

void fd_print_report(int nx, int nz);

#endif
