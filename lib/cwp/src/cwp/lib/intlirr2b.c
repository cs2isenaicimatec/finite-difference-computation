/* Copyright (c) Colorado School of Mines, 2011.*/
/* All rights reserved.                       */

/*********************** self documentation **********************/
/*****************************************************************************
INTLIRR2B - bilinear interpolation of a 2-D array of bytes

intlirr2b	bilinear interpolation of a 2-D array of bytes
                where x spacings are irregular

******************************************************************************
Function Prototype:
void intlirr2b (int nxin, float *xin,
	int nyin, float dyin, float fyin, unsigned char *zin,
	int nxout, float dxout, float fxout,
	int nyout, float dyout, float fyout, unsigned char *zout);

******************************************************************************
Input:
nxin		number of x samples input (fast dimension of zin)
xin		array of (irregular) input x-coordinates
nyin		number of y samples input (slow dimension of zin)
dyin		y sampling interval input
fyin		first y sample input
zin		array[nyin][nxin] of input samples (see notes)
nxout		number of x samples output (fast dimension of zout)
dxout		x sampling interval output
fxout		first x sample output
nyout		number of y samples output (slow dimension of zout)
dyout		y sampling interval output
fyout		first y sample output

Output:
zout		array[nyout][nxout] of output samples (see notes)

******************************************************************************
Notes:
The arrays zin and zout must passed as pointers to the first element of
a two-dimensional contiguous array of unsigned char values.

Constant extrapolation of zin is used to compute zout for
output x and y outside the range of input x and y.
 
For efficiency, this function builds a table of interpolation
coefficents pre-multiplied by byte values.  To keep the table
reasonably small, the interpolation does not distinguish
between x and y values that differ by less than dxin/ICMAX
and dyin/ICMAX, respectively, where ICMAX is a parameter
defined above.

******************************************************************************
Author:  Dave Hale, Colorado School of Mines, c. 1989-1991.
*****************************************************************************/
/**************** end self doc ********************************/

#include "cwp.h"

/* number of tabulated interpolation coefficients */
#define ICMAX 99 /* must be odd, so that ICMAC-ic!=ic, for ic=0 to ICMAX/2! */
#define NTABLE (ICMAX+1)

/* functions defined and used internally */
static void intl2bx(int, int*, int*, int,
	unsigned char[][256], unsigned char*, unsigned char*);
static void intl2by(int, int, int, unsigned char[][256],
	unsigned char*, unsigned char*, unsigned char*);

void intlirr2b(int nxin, float *xin,
	int nyin, float dyin, float fyin, unsigned char *zin,
	int nxout, float dxout, float fxout,
	int nyout, float dyout, float fyout, unsigned char *zout)
/*****************************************************************************
bilinear interpolation of a 2-D array of bytes
******************************************************************************
Input:
nxin		number of x samples input (fast dimension of zin)
xin		array of (irregular) input x-coordinates
nyin		number of y samples input (slow dimension of zin)
dyin		y sampling interval input
fyin		first y sample input
zin		array[nyin][nxin] of input samples (see notes)
nxout		number of x samples output (fast dimension of zout)
dxout		x sampling interval output
fxout		first x sample output
nyout		number of y samples output (slow dimension of zout)
dyout		y sampling interval output
fyout		first y sample output

Output:
zout		array[nyout][nxout] of output samples (see notes)
******************************************************************************
Notes:
The arrays zin and zout must passed as pointers to the first element of
a two-dimensional contiguous array of unsigned char values.

Constant extrapolation of zin is used to compute zout for
output x and y outside the range of input x and y.
 
For efficiency, this function builds a table of interpolation
coefficents pre-multiplied by byte values.  To keep the table
reasonably small, the interpolation does not distinguish
between x and y values that differ by less than dxin/ICMAX
and dyin/ICMAX, respectively, where ICMAX is a parameter
defined above.
******************************************************************************
Author:  Dave Hale, Colorado School of Mines, 07/02/89
Modified:  Dave Hale, Colorado School of Mines, 05/30/90
	Changed function to interpolate unsigned char 
	instead of signed char, since many color tables and 
	image processing functions (e.g., PostScript) require 
	bytes with a maximum range of 0 to 255.
Modified:  Dave Hale, Colorado School of Mines, 06/01/91
	Changed computation of coefficient table to avoid
	errors due to truncation in float to fix.  Old code
	sometimes caused interpolated values to be less than
	the minimum of the byte values being interpolated or 
	greater than the maximum of the values being interpolated.
*****************************************************************************/
{         
	int ixout,iyout,ic,ib,iyin,iyinl=1;
	int found, il, im, ir;
	float xout,yout,ryin,frac=0.0;
	int *kzin,*kic;
	unsigned char *temp1,*temp2,*temp;
	static unsigned char table[NTABLE][256];
	static int tabled=0;
	
	/* if not already built, build byte multiplication table */
	if (!tabled) {
		for (ic=0; ic<=ICMAX/2; ++ic) {
			frac = (float)(ic)/(float)ICMAX;
			for (ib=0; ib<256; ++ib) {
				table[ic][ib] = frac*ib;
				table[ICMAX-ic][ib] = ib-table[ic][ib];
			}
		}
		tabled = 1;
	}                                              

	/* get workspace */
	kzin = (int*)malloc(nxout*sizeof(int));
	kic = (int*)malloc(nxout*sizeof(int));
	temp1 = (unsigned char*)malloc(nxout*sizeof(unsigned char));
	temp2 = (unsigned char*)malloc(nxout*sizeof(unsigned char));

	
	/* pre-compute indices for fast 1-D interpolation along x axis */
	for (ixout=0,xout=fxout; ixout<nxout; ixout++,xout+=dxout) {
	  if (xout<=xin[0]) {
	    kzin[ixout] = 0;
	    kic[ixout] = 0;
	  } else if (xout>=xin[nxin-1]) {
	    kzin[ixout] = nxin-2;
	    kic[ixout] = ICMAX*256;
	  } else {
	    /* search index */
	    found = 0;
	    il = 0;
	    im = nxin/2;
	    ir = nxin-1;
	    while (!found) {
	      if (xout >= xin[il] && xout <= xin[im]) {
		if (im-il == 1) {
		  kzin[ixout] = il;
		  frac = (xout-xin[il])/(xin[im]-xin[il]);
		  found = 1;
		} else {
		  ir = im;
		  im = (il+im)/2;
		}
	      } else {
		if (ir-im == 1) {
		  kzin[ixout] = im;
		  frac = (xout-xin[im])/(xin[ir]-xin[im]);
		  found = 1;
		} else {
		  il = im;
		  im = (im+ir)/2;
		}
	      }
	      ic = frac*ICMAX+0.5;
	      kic[ixout] = ic*256;
	    }
	  }
	}

	/* loop over output y */
	for (iyout=0,yout=fyout; iyout<nyout; iyout++,yout+=dyout) {

		/* compute index of input y, clipped to range of input y */
		ryin = MAX(0,MIN(nyin-1,(yout-fyin)/dyin));
		iyin = MAX(0,MIN(nyin-2,ryin));

		/* if output y is not between current input y */
		if (iyin!=iyinl || iyout==0) {

			/* if 2nd temporary vector is still useful */
			if (iyin==iyinl+1 && iyout!=0) {              

				/* swap 2nd and 1st temp; compute 2nd temp */
				temp = temp1;
				temp1 = temp2;
				temp2 = temp;
				intl2bx(nxout,kzin,kic,ICMAX,
					table,zin+(iyin+1)*nxin,temp2);

			/* else if 1st temporary vector is still useful */
			} else if (iyin==iyinl-1 && iyout!=0) {

				/* swap 1st and 2nd temp; compute 1st temp */
				temp = temp1;
				temp1 = temp2;
				temp2 = temp;
				intl2bx(nxout,kzin,kic,ICMAX,
					table,zin+iyin*nxin,temp1);

			/* else if neither 1st or 2nd temp is useful */
			} else {

				/* compute 1st and 2nd temporary vectors */
				intl2bx(nxout,kzin,kic,ICMAX,
					table,zin+iyin*nxin,temp1);
				intl2bx(nxout,kzin,kic,ICMAX,
					table,zin+(iyin+1)*nxin,temp2);
			}                 

			/* remember last index of input y */
			iyinl = iyin;
		}

		/* compute index of interpolation coefficient */
		frac = ryin-iyin;
		ic = frac*ICMAX+0.5;

		/* linearly interpolate output vector by table lookup */
		intl2by(nxout,ic,ICMAX,table,
  			temp1,temp2,zout+iyout*nxout);
	}                         

	/* free workspace before returning */
	free(kzin);
	free(kic);                     
	free(temp1);
	free(temp2);
}
   
static void intl2bx(int nxout, int *kzin, int *kic, int icmax, 
	unsigned char table[][256], unsigned char *zin, unsigned char *zout)
/****************************************************************************
 interpolate between input x values (FOR INTERNAL USE by intl2b) 
****************************************************************************/
{
	int ixout,jzin,jic;
  	unsigned char *tablel,*tableh;
	tablel = &table[0][0];
	tableh = &table[icmax][0];
	for (ixout=0; ixout<nxout; ixout++) {   
	        jzin = kzin[ixout];
		jic = kic[ixout];
	 	zout[ixout] = tableh[(int)zin[jzin]-jic] 
				+ tablel[(int)zin[jzin+1]+jic];
	}
}   

static void intl2by(int nxout, int ic, int icmax, unsigned char table[][256],
	unsigned char *temp1, unsigned char *temp2, unsigned char *zout)
/****************************************************************************
 interpolate between input y values (FOR INTERNAL USE by intl2b) 
****************************************************************************/
{
	int ixout;
	unsigned char *tablel, *tableh;
	tablel = &table[ic][0];
	tableh = &table[icmax-ic][0];
	for (ixout=0; ixout<nxout; ixout++)
		zout[ixout] = tableh[temp1[ixout]]+tablel[temp2[ixout]];
}

#ifdef TEST
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <math.h>
#define NXIN 25
#define NYIN 25
#define NXOUT 35
#define NYOUT 35
#define FAC 10000

int main()
{
	int nxin,nyin,nxout,nyout,i,j, ix0, ipos, jpos;
	float dyin,fyin,fxout,dxout,fyout,dyout,pi,s,tmp,amx;
	float xin[NXIN], bin[NYIN], bout[NYOUT];
	unsigned char zin[NYIN][NXIN], zout[NYOUT][NXOUT];
	FILE* fp;

	pi = 4. * atan(1.);

	nxin=NXIN;
	nyin=NYIN;
	nxout=NXOUT;  
	nyout=NYOUT;
	dyin=1.0;
	fyin=0.0;
	dyout=dyin*(nyin-1)/(nyout-1);
	fyout=0.0;

	ix0 = -8;
	s = 0.1;

    /* x-coordinates regular */
	/*
	for (i=0; i<nxin; i++) xin[i] = FAC*i/(nxin-1);
	for (i=0; i<nxin; i++) xin[i] = FAC*(i+ix0)/(nxin-1);
	*/
    /* x-coordinates irregular */
	/*
	for (i=0; i<nxin; i++) xin[i] = sin(pi/2.*i/(nxin-1))*FAC;
	*/
	for (i=0; i<nxin; i++) xin[i] = sin(-s*pi/2.+(1.+s)*pi/2.*i/(nxin-1))*FAC;

	fxout = xin[0];
	dxout = (xin[nxin-1]-xin[0])/(nxout-1);

	for (i=0; i<nxin; i++) fprintf(stderr,"i=%d xin=%f\n",i,xin[i]);
	for (i=0; i<nxout; i++) fprintf(stderr,"i=%d xout=%f\n",i,fxout+i*dxout);

    /* 2D Gaussian */
	for (j=0; j<nyin; j++) {
	  for (i=0; i<nxin; i++) {
	    zin[j][i] = 255*exp(-(pow(nxin/2-xin[i]/FAC*nxin,2)+pow(nyin/2-j,2))*0.02);
	  }
	}

	/* estimate location of maximum */
	amx = 0.;
	for (j=0; j<nyin; j++) {
	  for (i=0; i<nxin; i++) {
	    tmp = fabs(zin[j][i]);
	    if (tmp > amx) {
	      amx =tmp;
	      ipos = i;
	      jpos = j;
	    }
	  }
	}
	fprintf(stderr,"ipos=%d  xin_max=%f  jpos=%d\n",ipos,xin[ipos],jpos);

	fp = fopen("f1","w");
	for (i=0; i<nxin; i++) {
	  for (j=0; j<nyin; j++) bin[j] = zin[j][i];
	  fwrite(bin,nyin,sizeof(float),fp);
	}
	fclose(fp);

	intlirr2b(nxin,xin,
		  nyin,dyin,fyin,&zin[0][0],
		  nxout,dxout,fxout,
		  nyout,dyout,fyout,&zout[0][0]);

	/* estimate location of maximum */
	amx = 0.;
	for (j=0; j<nyin; j++) {
	  for (i=0; i<nxin; i++) {
	    tmp = fabs(zout[j][i]);
	    if (tmp > amx) {
	      amx =tmp;
	      ipos = i;
	      jpos = j;
	    }
	  }
	}
	fprintf(stderr,"ipos=%d  xout_max=%f  jpos=%d\n",ipos,fxout+ipos*dxout,jpos);

	fp = fopen("f2","w");
	for (i=0; i<nxout; i++) {
	  for (j=0; j<nyout; j++) bout[j] = zout[j][i];
	  fwrite(bout,nyout,sizeof(float),fp);
	}
	fclose(fp);
	return 0;
}
#endif
