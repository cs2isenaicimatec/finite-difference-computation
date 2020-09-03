#include "cwp.h"
#include "ptsrc.h"
#include "rtm.h"

/* prototype of subroutine used internally */
float ricker (float t, float fpeak);

//void ptsrc (int xs, int zs, int nx, int nz, float ts, float **s, int32_t **S)
//{
//	int ix,iz,ixs,izs;
//	float xn,zn,xsn,zsn,m;
//
//	ixs = xs;
//	izs = zs;
//	xsn = xs;
//	zsn = zs;
//
//	for (ix=MAX(0,ixs-3); ix<=MIN(nx-1,ixs+3); ++ix) {
//		for (iz=MAX(0,izs-3); iz<=MIN(nz-1,izs+3); ++iz) {
//			xn = ix-xsn;
//			zn = iz-zsn;
//			m = exp(-xn*xn-zn*zn);
//
//			// Floating point calc.
//			s[ix][iz] += ts*m;
//
//			// Fixed-point calc.			
//			S[ix][iz] = QMN_ADD(S[ix][iz],QMN_FROM_REAL(ts*m));
//
//		}
//	}
//}

// Usando fonte pontual
int it=0;
void ptsrc (int xs, int zs, float ts, float **s)
{
	// Floating point calc.
	// printf("[%d,%d] SRC=%.20f PP=%.20f PP'=%.20f \n",
	// 	xs, zs, ts, s[xs][zs], s[xs][zs]+ts);
	it++;
	s[xs][zs] += ts;
}

float ricker (float t, float fpeak)
/*****************************************************************************
ricker - Compute Ricker wavelet as a function of time
******************************************************************************
Input:
t		time at which to evaluate Ricker wavelet
fpeak		peak (dominant) frequency of wavelet
******************************************************************************
Notes:
The amplitude of the Ricker wavelet at a frequency of 2.5*fpeak is 
approximately 4 percent of that at the dominant frequency fpeak.
The Ricker wavelet effectively begins at time t = -1.0/fpeak.  Therefore,
for practical purposes, a causal wavelet may be obtained by a time delay
of 1.0/fpeak.
The Ricker wavelet has the shape of the second derivative of a Gaussian.
******************************************************************************
Author:  Dave Hale, Colorado School of Mines, 04/29/90
******************************************************************************/
{
	float x,xx;
	
	x = PI*fpeak*t;
	xx = x*x;
	/* return (-6.0+24.0*xx-8.0*xx*xx)*exp(-xx); */
	/* return PI*fpeak*(4.0*xx*x-6.0*x)*exp(-xx); */
	return exp(-xx)*(1.0-2.0*xx);
}

void ricker_wavelet(int nt, float dt, float peak, float *s){
	int it;
	for(it = 0; it < nt; it++){
		s[it] = ricker(it*dt - 1.0/peak, peak);
	}
}
