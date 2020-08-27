#ifndef FD_H
#define FD_H
#include <cuda.h>
#define PERF_COUNTERS
#define CUDA

void fd_init(int order, int nx, int nz, int nxb, int nzb, int nt, 
		int ns, float fac, float dx, float dz, float dt);
void fd_reinit(int order, int nx, int nz);

void fd_forward(int order, float **p, float **pp, float **v2, 
    float ***upb, int nz, int nx, int nt, int is, int sz, int *sx, float *srce);

void fd_back(int order, float **p, float **pp, float **pr, float **ppr, float **v2, float ***upb,
    int nz, int nx, int nt, int is, int sz, int gz, float ***snaps, float **imloc, float **d_obs); 

void write_buffers(float **p, float **pp, float **v2, float ***upb, 
    float *taperx, float *taperz, float **d_obs, float **imloc, int is, int flag);

void fd_destroy();
float *calc_coefs(int order);

void fd_print_report(int nx, int nz);

#endif
