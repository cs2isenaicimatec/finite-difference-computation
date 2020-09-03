#ifndef FD_H
#define FD_H

#include <CL/cl.h> 
#include "clmisc.h" 

#ifdef UPDATE_PP
#define KERNEL_FUNCTION_NAME 	"fdstep"
#else
#define KERNEL_FUNCTION_NAME 	"laplacian"
#endif

void fd_init_opencl (int order, int nxe, int nze, int nxb, int nzb, 
                    int nt, int ns, float fac, char * kernelPath);
void fd_init 		(int order, int nx, int nz, int nxb, int nzb, int nt, int ns,
    				float fac, float dx, float dz, float dt, char * kernelPath);
void write_buffers(float **p, float **pp, float **v2, float *srce, float ***upb, 
    float *taperx, float *taperz, float **d_obs, float **imloc, int *sx, int is, int flag); 

void set_args(int order, int nx, int nz, int nt, float dt2, 
    int *sx, int is, int sz, float *srce, int it, int gz, int flag); 
void fd_step_forward(int order, float **p, float **pp, float **v2, float ***upb, 
                    int nz, int nx, int nt, int is, int sz, int *sx, float *srce);
void fd_step_back   (int order, float **p, float **pp, float **pr, float **ppr, float **v2, float ***upb, int *sx,
    int nz, int nx, int nt, int is, int it, int sz, int gz, float ***snaps, float **imloc, float **d_obs, float *srce); 
void fd_step 		(float **p, float **pp, float **v2);
void fd_laplacian 	(int nz, int nx, float **p, float **laplace, int order);
void fd_reinit		(int order, int nx, int nz);
void fd_destroy 	(void);
void fd_print_report(int nx, int nz);


float *calc_coefs 	(int order);
void   makeo2 		(float *coef,int order);

#endif // FD_H
