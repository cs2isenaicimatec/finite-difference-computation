#include <stdio.h>
#include <time.h>
#include <CL/cl.h>
#include <omp.h>

#include "su.h"
#include "ptsrc.h"
#include "taper.h"
#include "fd.h"

float *calc_coefs(int order)
{
    float *coef;

    coef = calloc(order+1,sizeof(float));

    switch(order){
        case 2:
            coef[0] = 1.;
            coef[1] = -2.;
            coef[2] = 1.;
            break;
        case 4:
            coef[0] = -1./12.;
            coef[1] = 4./3.;
            coef[2] = -5./2.;
            coef[3] = 4./3.;
            coef[4] = -1./12.;
            break;
        case 6:
            coef[0] = 1./90.;
            coef[1] = -3./20.;
            coef[2] = 3./2.;
            coef[3] = -49./18.;
            coef[4] = 3./2.;
            coef[5] = -3./20.;
            coef[6] = 1./90.;
            break;
        case 8:
            coef[0] = -1./560.;
            coef[1] = 8./315.;
            coef[2] = -1./5.;
            coef[3] = 8./5.;
            coef[4] = -205./72.;
            coef[5] = 8./5.;
            coef[6] = -1./5.;
            coef[7] = 8./315.;
            coef[8] = -1./560.;
            break;
        default:
            makeo2(coef,order);
    }
    return coef;
}

void makeo2 (float *coef,int order){
    float h_beta, alpha1=0.0;
    float alpha2=0.0;
    float  central_term=0.0; 
    float coef_filt=0; 
    float arg=0.0; 
    float  coef_wind=0.0;
    int msign,ix; 
    float alpha = .54;
    float beta = 6.;

    h_beta = 0.5*beta;
    alpha1=2.*alpha-1.0;
    alpha2=2.*(1.0-alpha);
    central_term=0.0;

    msign=-1;

    for (ix=1; ix <= order/2; ix++){      
        msign=-msign ;            
        coef_filt = (2.*msign)/(ix*ix); 
        arg = PI*ix/(2.*(order/2+2));
        coef_wind=pow((alpha1+alpha2*cos(arg)*cos(arg)),h_beta);
        coef[order/2+ix] = coef_filt*coef_wind;
        central_term = central_term + coef[order/2+ix]; 
        coef[order/2-ix] = coef[order/2+ix]; 
    }
    
    coef[order/2]  = -2.*central_term;

    return; 
}