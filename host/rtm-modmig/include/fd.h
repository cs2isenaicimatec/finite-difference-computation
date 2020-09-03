#ifndef FD_H
#define FD_H

#include "rtm.h"

#define FDCONTROL_INITED 	0xAB



//////////////////////////////////////////////////////

void fd_init(RTMExecParam * execParam);
void fd_reinit(int order, float dx, float dz, float dt, float * coefsx, float * coefsz);
//void fd_step(int order, float **p, float **pp, int32_t **P, int32_t **PP, float **v2, int nz, int nx);
void fd_step_float(RTMExecParam * exec, float ** P, float ** PP, float ** shotlaplace,
	int start_x, int end_x, int start_z, int end_z );

void laplacian_float(int xstart, int zstart, int xend, int zend, int order, 
	float **P, float * coefsx, float * coefsz, float ** laplace, int it);
void fd_calc_norm(int order, float norm_vel2dt2_main);
void fd_destroy();
void coef_matrix(int order, int nxe, int nze);
float *calc_coefs(int order);

#endif
