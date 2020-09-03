#include "cwp.h"
#include "taper.h"
#include "rtm.h"

void extendvel(int nx,int nz,int nxb,int nzb,float *vel){
	int ix,iz;
	int rnz = nz+2.*nzb;

	for(ix=0;ix<nx;ix++){
		for(iz=0;iz<nzb;iz++)
			vel[(ix+nxb)*rnz+iz] = vel[(ix+nxb)*rnz+nzb];
		for(iz=nzb+nz;iz<nz+2*nzb;iz++)
			vel[(ix+nxb)*rnz+iz] = vel[(ix+nxb)*rnz+nz+nzb-1];
	}
	for(iz=0;iz<nz+2*nzb;iz++){
		for(ix=0;ix<nxb;ix++)
			vel[ix*rnz+iz] = vel[nxb*rnz+iz];
		for(ix=nxb+nx;ix<nx+2*nxb;ix++)
			vel[ix*rnz+iz] = vel[(nx+nxb-1)*rnz+iz];
	}
}


void extendvel_hyb(int nx,int nz,int nxb,int nzb,float **vel){
	int ix,iz;
	float v=0,v_ave=0,l_lim = 300.,delta = 200.;

	for(ix=0;ix<nx;ix++){
		for(iz=0;iz<nzb;iz++){ 
			/* borda superior */
			vel[ix+nxb][iz] = vel[ix+nxb][nzb];
			
			/* borda inferior */
			v = vel[ix+nxb][nzb+nz-1];
			v_ave = v - (v - l_lim)*(iz)/(nzb-1);
			vel[ix+nxb][nz+nzb+iz] = rand()%(int)(v + delta - (v_ave - delta) +1) + v_ave - delta;
		}
	}
	for(iz=0;iz<nz;iz++){
		for(ix=0;ix<nxb;ix++){									
			/* borda esquerda */
			v = vel[nxb][nzb+iz];	
			v_ave = v - (v - l_lim)*(ix)/(nxb-1);
			vel[nxb-1-ix][nzb+iz] = rand()%(int)(v + delta - (v_ave - delta) +1) + v_ave - delta;
			
			/* borda direita */
			v = vel[nxb+nx-1][nzb+iz];	
			v_ave = v - (v - l_lim)*(ix)/(nxb-1);
			vel[nxb+nx+ix][nzb+iz] = rand()%(int)(v + delta - (v_ave - delta) +1) + v_ave - delta;			
		}
	}

	/* Canto superior esquerdo e direito */
	for(iz=0;iz<nzb;iz++){
		for(ix=0;ix<nxb;ix++){
			vel[ix][iz] = vel[nxb][iz];
			vel[nxb+nx+ix][iz]= vel[nxb+nx-1][iz];
		}
	}

	/* Canto inferior esquerdo */
	for(iz=0;iz<nzb;iz++){
		for(ix=0;ix<=iz;ix++){
			v = vel[nxb][nzb+nz-1];
			v_ave = v - (v - l_lim)*(nxb-1-ix)/(nzb-1);
			vel[ix][nz+2*nzb-1-iz] = rand()%(int)(v + delta - (v_ave - delta) +1) + v_ave - delta;
			vel[iz][nz+2*nzb-1-ix] = rand()%(int)(v + delta - (v_ave - delta) +1) + v_ave - delta;
		}
	}

	/* Canto inferior direito */
	for(iz=0;iz<nzb;iz++){
		for(ix=0;ix<=iz;ix++){
			v = vel[nxb+nx-1][nzb+nz-1];
			v_ave = v - (v - l_lim)*(nxb-1-ix)/(nzb-1);
			vel[nx+2*nxb-1-ix][nz+2*nzb-1-iz] = rand()%(int)(v + delta - (v_ave - delta) +1) + v_ave - delta;
			vel[nx+2*nxb-1-iz][nz+2*nzb-1-ix] = rand()%(int)(v + delta - (v_ave - delta) +1) + v_ave - delta;
		}
	}
}

void taper_filldata(float * taperx, float *taperz, int nxb, int nzb, float F){
	int i;
	float dfrac;
	
	dfrac = sqrt(-log(F))/(1.*nxb);

	for(i=0;i<nxb;i++){
		taperx[i] = exp(-pow((dfrac*(nxb-i)),2));
	}

	dfrac = sqrt(-log(F))/(1.*nzb);

	for(i=0;i<nzb;i++){
		taperz[i] = exp(-pow((dfrac*(nzb-i)),2));
	}
	return;
}

void taper_init(RTMExecParam * execParam){

	if(execParam->taperx==NULL)
		alloc1float(execParam->nxb);
	if(execParam->taperz==NULL)
		alloc1float(execParam->nzb);

	taper_filldata(execParam->taperx, execParam->taperz, 
		execParam->nxb, execParam->nzb, execParam->fac);
}

void taper_apply(float **pp,int nx, int nz, int nxb, int nzb, float * tpx, float *tpz){
	int itz, itx,i;

	for(itx=0;itx<nx+2*nxb;itx++){
		for(itz=0;itz<nzb;itz++){
			pp[itx][itz] *= tpz[itz];
		}
		for(itz=nzb-1,i=0;itz>-1;itz--,i++){
			pp[itx][nz+nzb+i] *= tpz[itz];
		}
	}
	for(itz=0;itz<nz+2*nzb;itz++){
		for(itx=0;itx<nxb;itx++){
			pp[itx][itz] *= tpx[itx];
		}
		for(itx=nxb-1,i=0;itx>-1;itx--,i++){
			pp[nx+nxb+i][itz] *= tpx[itx];
		}
	}
	return;
}

void taper_apply_hyb_float(float **PP, int nx, int nz, int nxb, int nzb, 
	float * tpx, float *tpz){
	int itz, itx, itxr, ix;
	
	// for(itx=0;itx<nx+2*nxb;itx++){
	// 	for(itz=0;itz<nzb;itz++){
	// 		// Floating point calc
	// 		PP[itx][itz] *= tpz[itz];

	// 	}
	// }
	// for(itx=0,itxr=nx+2*nxb-1;itx<nxb;itx++,itxr--){
	// 	for(itz=0;itz<nzb;itz++){
	// 		PP[itx][itz] *= tpx[itx];
	// 		PP[itxr][itz] *= tpx[itx];
	// 	}
	// }
	float tz, txr, txl;
	for (itx=0; itx<nx+2*nxb; itx++){
		for(itz=0;itz<nzb;itz++){
			tz = tpz[itz];
			if(itx>=0 && itx<nxb){
				txl = tpx[itx];
			}else{
				txl = 1.;
			}

			if(itx<=nx+2*nxb-1 && itx>=nx+nzb){
				itxr = nxb - 1 - (itx - (nx+nxb));
				txr = tpx[itxr];
			}else{
				txr = 1.;
			}
			float prevp=PP[itx][itz];
			PP[itx][itz] *= tz;
			PP[itx][itz] *= txr;
			PP[itx][itz] *= txl;
		}
	}


	return;
}

void taper_apply_hyb(float **PP, int nx, int nz, int nxb, int nzb, 
	float * tpx, float *tpz){
	int itz, itx, itxr;
	taper_apply_hyb_float(PP, nx, nz, nxb, nzb, tpx, tpz);
	//taper_apply(PP, nx, nz, nxb, nzb, tpx, tpz);
	return;
}
void taper_destroy(){
	return;
}
