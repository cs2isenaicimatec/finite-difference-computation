#ifndef TAPER_H
#define TAPER_H
#include "rtm.h"

void extendvel(int nx,int nz,int nxb,int nzb,float *vel);
void extendvel_hyb(int nx,int nz,int nxb,int nzb,float **vel);
void taper_filldata(float * taperx, float *taperz, int nxb, int nzb, float F);
void taper_init(RTMExecParam * execParam);
void taper_apply(float **pp,int nx, int nz, int nxb, int nzb, float * tpx, float *tpz);
void taper_apply_hyb_float(float **PP, int nx, int nz, int nxb, int nzb, float * tpx, float *tpz);
void taper_apply_hyb(float **PP, int nx, int nz, int nxb, int nzb, 
	float * tpx, float *tpz);
void taper_destroy();
#endif

