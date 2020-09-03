#include "cwp.h"
#include "fd.h"
#include "rtm.h"
#include "math.h"

#include <stdio.h> 

#define FREE_PTR(ptr)  if(ptr!=NULL) free(ptr)

///////////////////////////////////////////////////////////
static void makeo2 (float *coef,int order);
float *calc_coefs(int order);
void ftoa(float n, char *res, int afterpoint);
///////////////////////////////////////////////////////////

void fd_init(RTMExecParam * execParam){
	int i;
	float dx2inv = (1./execParam->dx)*(1./execParam->dx);
    float dz2inv = (1./execParam->dz)*(1./execParam->dz);
	float dt2 = execParam->dt*execParam->dt;

	float * coefs = calc_coefs(execParam->order);

	for(i=0;i<=execParam->order;i++) {
		execParam->coefsx[i] = (coefs[i]*dx2inv);
		execParam->coefsz[i] = (coefs[i]*dz2inv);
	}
	free(coefs);
	/////////////////////////////////////////////////////
	return;
}

void fd_reinit(int order, float dx, float dz, float dt, float * coefsx, float * coefsz){
		int i;
	float dx2inv = (1./dx)*(1./dx);
    float dz2inv = (1./dz)*(1./dz);
	float dt2 = dt*dt;

	float * coefs = calc_coefs(order);

	for(i=0;i<=order;i++) {
		coefsx[i] = (coefs[i]*dz2inv);
		coefsz[i] = (coefs[i]*dx2inv);
	}
	free(coefs);
	/////////////////////////////////////////////////////
	return;
}


void laplacian_float(int xstart, int zstart, int xend, int zend, int order, 
	float **P, float * coefsx, float * coefsz, float ** laplace, int it){
	int ix,iz,io;
	float acm_x,acm_z;
	acm_z = 0.f;
	acm_x = 0.f;
	int div = order/2;

    //#pragma omp parallel for reduction (+:acm_z,acm_x) private(ix, iz, io) schedule(static,1)
	for(ix=xstart+div;ix<xend-div;ix++){
		for(iz=zstart+div;iz<zend-div;iz++){
			for(io=0;io<=order;io++){
				acm_z += P[ix][iz+io-div]*coefsz[io];
				acm_x += P[ix+io-div][iz]*coefsx[io];
			}
			laplace[ix][iz] = (acm_z+acm_x);
			acm_z = 0.f;
			acm_x = 0.f;
		}
	}
	return;
}

void laplacian_dbg(int xstart, int zstart, int xend, int zend, int order, 
	float **P, float * coefsx, float * coefsz, float ** laplace, int it){
	int ix,iz,io;
	float acm_x,acm_z;
	acm_z = 0.f;
	acm_x = 0.f;
	int div = order/2;

	char fname[100];
	sprintf(fname, "debug/imloc/lapserial%d.txt",it);
	FILE * fp = fopen(fname, "w");

    //#pragma omp parallel for reduction (+:acm_z,acm_x) private(ix, iz, io) schedule(static,1)
	for(ix=xstart+div;ix<xend-div;ix++){
		for(iz=zstart+div;iz<zend-div;iz++){
			for(io=0;io<=order;io++){
				acm_z += P[ix][iz+io-div]*coefsz[io];
				acm_x += P[ix+io-div][iz]*coefsx[io];
				if(ix>=270 && ix<=290 && iz>=30 && iz<=50){
					fprintf(fp, "(%d)[%d][%d](%d)cz    = %.20f\n",it, ix, iz, io, coefsz[io] );
					fprintf(fp, "(%d)[%d][%d](%d)cx    = %.20f\n",it, ix, iz, io, coefsx[io] );
					fprintf(fp, "(%d)[%d][%d](%d)PZ    = %.20f\n",it, ix, iz, io, P[ix][iz+io-div]);
					fprintf(fp, "(%d)[%d][%d](%d)PX    = %.20f\n",it, ix, iz, io, P[ix+io-div][iz]);
					fprintf(fp, "(%d)[%d][%d](%d)acm_z = %.20f\n",it, ix, iz, io, acm_z );
					fprintf(fp, "(%d)[%d][%d](%d)acm_x = %.20f\n",it, ix, iz, io, acm_x);
					fprintf(fp, "(%d)[%d][%d](%d)LAP   = %.20f\n",it, ix, iz, io, acm_z+acm_x );
				}
			}
			laplace[ix][iz] = (acm_z+acm_x);
			acm_z = 0.f;
			acm_x = 0.f;
		}
	}
	fclose(fp);
	return;
}

void fd_step_float(RTMExecParam * exec, float ** P, float ** PP, float ** shotlaplace,
	int xstart, int xend, int zstart, int zend ){
	int ix,iz;
	float ** v2dt2 = exec->vel2dt2;

	//if(exec->exec_state==RTMEXEC_BWRD && exec->current_it==2){
	if(0){
		laplacian_dbg(xstart, zstart, xend, zend, 
			exec->order,P,exec->coefsx, exec->coefsz,
			shotlaplace, exec->current_it);
	}else{
		laplacian_float(xstart, zstart, xend, zend, 
			exec->order,P,exec->coefsx, exec->coefsz,
			shotlaplace, exec->current_it);
	}

	// char fname[100];
	// FILE * fp;
	// sprintf(fname, "debug/imloc/ppserial%d.txt",exec->current_it);
	// if(exec->exec_state==RTMEXEC_BWRD)
	// 	fp = fopen(fname, "w");
	for(ix=xstart;ix<xend;ix++){
		for(iz=zstart;iz<zend;iz++){
			float pp = PP[ix][iz];;
			float p =  P[ix][iz];
			float l = shotlaplace[ix][iz];
			float v = v2dt2[ix][iz];
			float npp = 2*p - pp + v*l;
			PP[ix][iz] = npp;		

			// if(exec->exec_state==RTMEXEC_BWRD){
			// fprintf(fp, "[%d][%d]prev_pp = %.30f \n",ix, iz, pp);
			// fprintf(fp, "[%d][%d]prev_p  = %.30f \n",ix, iz, p);
			// fprintf(fp, "[%d][%d]v2dt2   = %.30f \n",ix, iz, v);
			// fprintf(fp, "[%d][%d]laplace = %.30f \n",ix, iz, l);
			// fprintf(fp, "[%d][%d]newpp   = %.30f \n",ix, iz, npp);
			// }
		}
	}
	// if(exec->exec_state==RTMEXEC_BWRD)
	// 	fclose(fp);
}

void fd_destroy(){
	// free2float(laplace);
	return;
}

float *calc_coefs(int order){
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

static void makeo2 (float *coef,int order){
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

