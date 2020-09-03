#include "taper.h"
#include "cwp.h"

static float *taperx=NULL,*taperz=NULL;
/*static void damp(int nbw, float abso, float *bw);*/

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

void extendvel_gauss(int nx,int nz,int nxb,int nzb,float **vel){
	int ix,iz;
	float v,l_lim = 300.,delta = 200., linear_incr, gauss, gauss_decay;

	for(ix=0;ix<nx;ix++){
		for(iz=0;iz<nzb;iz++){ 
			/* borda superior */
			v = vel[nxb+ix][nzb];
			gauss = 1 - exp((-120)*pow((float)(nzb-1-iz)/(float)(nzb-1),2));
			linear_incr = (v-l_lim)*(nzb-1-iz)/(nzb-1);
			gauss_decay = v - (v-l_lim)*gauss;
			vel[ix+nxb][iz] = rand()%(int)(v + delta - (gauss_decay - delta) - linear_incr +1) + gauss_decay - delta;
			
			/* borda inferior */ 
			v = vel[ix+nxb][nz+nzb-1];
			linear_incr = (v-l_lim)*(iz)/(nzb-1);
			gauss = 1 - exp((-120)*pow((float)(iz)/(float)(nzb-1),2));
			gauss_decay = v - (v - l_lim)*gauss;
			vel[ix+nxb][nz+nzb+iz] = rand()%(int)(v + delta - (gauss_decay - delta) - linear_incr +1) + gauss_decay - delta;
		}
	}
	for(iz=0;iz<nz;iz++){
		for(ix=0;ix<nxb;ix++){									
			/* borda esquerda */
			v = vel[nxb][nzb+iz];
			gauss = 1 - exp((-120)*pow((float)(nxb-1-ix)/(float)(nxb-1),2));
			linear_incr = (v-l_lim)*(nxb-1-ix)/(nxb-1);
			gauss_decay = v-(v-l_lim)*gauss;
			vel[ix][nzb+iz] = rand()%(int)(v + delta - (gauss_decay - delta) -linear_incr +1) + gauss_decay - delta;
			
			/* borda direita */	
			v = vel[nxb+nx-1][nzb+iz];
			gauss = 1 - exp((-120)*pow((float)1.*(ix)/(float)(nxb-1),2));
			linear_incr = (v-l_lim)*(float)1.*(ix)/(float)(nxb-1);
			gauss_decay = v-(v-l_lim)*gauss;
			vel[nxb+nx+ix][nzb+iz] = rand()%(int)(v + delta - (gauss_decay - delta) -linear_incr +1) + gauss_decay - delta;
		}
	}
	/* Canto superior esquerdo */
	for(iz=0;iz<nzb;iz++){
		for(ix=0;ix<=iz;ix++){
			v = vel[nxb][nzb];
			gauss = 1 - exp((-120)*pow((float)(nxb-1-ix)/(float)(nxb-1),2));
			linear_incr = (v-l_lim)*(nxb-1-ix)/(nxb-1);
			gauss_decay = v - (v - l_lim)*gauss;
			vel[ix][iz] = rand()%(int)(v + delta - (gauss_decay - delta) -linear_incr +1) + gauss_decay - delta;
			vel[iz][ix] = rand()%(int)(v + delta - (gauss_decay - delta) -linear_incr +1) + gauss_decay - delta;
		}
	}
	/* Canto superior direito */
	for(iz=0;iz<nzb;iz++){
		for(ix=0;ix<=iz;ix++){
			v = vel[nxb+nx-1][nzb];
			gauss = 1 - exp((-120)*pow((float)(nxb-1-ix)/(float)(nxb-1),2));
			linear_incr = (v-l_lim)*(nxb-1-ix)/(nxb-1);
			gauss_decay = v - (v - l_lim)*gauss;
			vel[nx+2*nxb-1-ix][iz] = rand()%(int)(v + delta - (gauss_decay - delta) -linear_incr +1) + gauss_decay - delta;
			vel[nx+2*nxb-1-iz][ix] = rand()%(int)(v + delta - (gauss_decay - delta) -linear_incr +1) + gauss_decay - delta;
		}
	}
	/* Canto inferior esquerdo */
	for(iz=0;iz<nzb;iz++){
		for(ix=0;ix<=iz;ix++){
			v = vel[nxb][nzb+nz-1];
			gauss = 1 - exp((-120)*pow((float)(nxb-1-ix)/(float)(nxb-1),2));
			linear_incr = (v-l_lim)*(nxb-1-ix)/(nxb-1);
			gauss_decay = v - (v - l_lim)*gauss;
			vel[ix][nz+2*nzb-1-iz] = rand()%(int)(v + delta - (gauss_decay - delta) -linear_incr +1) + gauss_decay - delta;
			vel[iz][nz+2*nzb-1-ix] = rand()%(int)(v + delta - (gauss_decay - delta) -linear_incr +1) + gauss_decay - delta;
		}
	}
	/* Canto inferior direito */
	for(iz=0;iz<nzb;iz++){
		for(ix=0;ix<=iz;ix++){
			v = vel[nxb+nx-1][nzb+nz-1];
			gauss = 1 - exp((-120)*pow((float)(nxb-1-ix)/(float)(nxb-1),2));
			linear_incr = (v-l_lim)*(nxb-1-ix)/(nxb-1);
			gauss_decay = v - (v - l_lim)*gauss;
			vel[nx+2*nxb-1-ix][nz+2*nzb-1-iz] = rand()%(int)(v + delta - (gauss_decay - delta) -linear_incr +1) + gauss_decay - delta;
			vel[nx+2*nxb-1-iz][nz+2*nzb-1-ix] = rand()%(int)(v + delta - (gauss_decay - delta) -linear_incr +1) + gauss_decay - delta;
		}
	}
}

void extendvel_linear(int nx,int nz,int nxb,int nzb,float **vel){
	int ix,iz;
	float v=0,v_ave=0,l_lim = 300.,delta = 200.;
	//fprintf(stdout,"\nRAND_MAX = %d\n",RAND_MAX); // Check max randomic number RAND_MAX

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

void extendvel_3(int nx,int nz,int nxb,int nzb,float **vel){
	int ix,iz;
	float v_ave[nxb],l_lim = 300.,delta = 200.;

	for(ix=0;ix<nx;ix++){
		for(iz=0;iz<nzb;iz++){ 
			/* borda superior */
			vel[ix+nxb][iz] = vel[ix+nxb][nzb];
			
			/* borda inferior */ 
			v_ave[iz] = vel[ix+nxb][nzb+nz-1] - (vel[ix+nxb][nz+nzb-1]-l_lim)*(iz)/(nzb-1);
			vel[ix+nxb][nz+nzb+iz] = rand()%(int)(vel[ix+nxb][nz+nzb-1] + delta - (v_ave[iz] - delta) +1) + v_ave[iz] - delta;
			
		}
	}
	for(iz=0;iz<nz;iz++){
		for(ix=0;ix<nxb;ix++){									
			/* borda esquerda */
			v_ave[ix] = vel[nxb][nzb+iz] - (vel[nxb][nzb+iz]-l_lim)*(ix)/(nxb-1);
			vel[ix][nzb+iz] = rand()%(int)(vel[nxb][nzb+iz] + delta - (v_ave[nxb-1-ix] - delta) +1) + v_ave[nxb-1-ix] - delta;
			
			/* borda direita */			
			v_ave[ix] = vel[nxb+nx-1][nzb+iz] - (vel[nxb+nx-1][nzb+iz]-l_lim)*(ix)/(nxb-1);
			vel[nxb+nx+ix][nzb+iz] = rand()%(int)(vel[nxb+nx-1][nzb+iz] + delta - (v_ave[ix] - delta) +1) + v_ave[ix] - delta;			
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
			v_ave[iz] = vel[nxb][nzb+nz-1] - (vel[nxb][nzb+nz-1]-l_lim)*(iz)/(nzb-1);
			vel[ix][nz+2*nzb-1-iz] = rand()%(int)(vel[nxb][nzb+nz-1] + delta - (v_ave[nxb-1-ix] - delta) +1) + v_ave[nxb-1-ix] - delta;
			vel[iz][nz+2*nzb-1-ix] = rand()%(int)(vel[nxb][nzb+nz-1] + delta - (v_ave[nxb-1-ix] - delta) +1) + v_ave[nxb-1-ix] - delta;
		}
	}

	/* Canto inferior direito */
	for(iz=0;iz<nzb;iz++){
		for(ix=0;ix<=iz;ix++){
			v_ave[iz] = vel[nxb+nx-1][nzb+nz-1] - (vel[nxb+nx-1][nzb+nz-1]-l_lim)*(iz)/(nzb-1);
			vel[nx+2*nxb-1-ix][nz+2*nzb-1-iz] = rand()%(int)(vel[nxb+nx-1][nzb+nz-1] + delta - (v_ave[nxb-1-ix] - delta) +1) + v_ave[nxb-1-ix] - delta;
			vel[nx+2*nxb-1-iz][nz+2*nzb-1-ix] = rand()%(int)(vel[nxb+nx-1][nzb+nz-1] + delta - (v_ave[nxb-1-ix] - delta) +1) + v_ave[nxb-1-ix] - delta;
		}
	}
}

void extendvel_4(int nx,int nz,int nxb,int nzb,float *vel){
	int ix,iz;
	float v_ave[nxb],l_lim = 300.,delta = 200.;
	int rnz = nz+2.*nzb;

	for(ix=0;ix<nx;ix++){
		for(iz=0;iz<nzb;iz++){ 
			/* borda superior */
			vel[(ix+nxb)*rnz+iz] = vel[(ix+nxb)*rnz+nzb];
			/* borda inferior */ 
			v_ave[iz] = vel[(ix+nxb)*rnz+nz+nzb-1] - (vel[(ix+nxb)*rnz+nz+nzb-1] - l_lim)*(iz)/(nzb-1);
			vel[(ix+nxb)*rnz+iz+nzb+nz] = rand()%(int)(vel[(ix+nxb)*rnz+nz+nzb-1] + delta - (v_ave[iz] - delta) +1) + v_ave[iz] - delta;
		}
	}
	for(iz=0;iz<nz;iz++){
		for(ix=0;ix<nxb;ix++){									
			/* borda esquerda */
			v_ave[ix] = vel[nxb*rnz+iz] - (vel[nxb*rnz+iz]-l_lim)*(ix)/(nxb-1);
			vel[ix*rnz+iz] = rand()%(int)(vel[nxb*rnz+iz] + delta - (v_ave[nxb-1-ix] - delta) +1) + v_ave[nxb-1-ix] - delta;
			/* borda direita */
			v_ave[ix] = vel[(nxb+nx-1)*rnz+iz+nzb] - (vel[(nxb+nx-1)*rnz+iz+nzb]-l_lim)*(ix)/(nxb-1);
			vel[(nx+nxb+ix)*rnz+iz+nzb] = rand()%(int)(vel[(nxb+nx-1)*rnz+iz+nzb] + delta - (v_ave[ix] - delta) +1) + v_ave[ix] - delta;			
		}
	}
	/* Canto superior esquerdo e direito */
	for(iz=0;iz<nzb;iz++){
		for(ix=0;ix<nxb;ix++){
			vel[ix*rnz+iz] = vel[nxb*rnz+iz];
			vel[(nxb+nx+ix)*rnz+iz] = vel[(nxb+nx-1)*rnz+iz];
		}
	}
	/* Canto inferior esquerdo */
	for(iz=0;iz<nzb;iz++){
		for(ix=0;ix<=iz;ix++){
			v_ave[iz] = vel[(nxb)*rnz+nz+nzb-1] - (vel[(nxb)*rnz+nz+nzb-1]-l_lim)*(iz)/(nzb-1);
			vel[ix*rnz+rnz-1-iz] = rand()%(int)(vel[(nxb)*rnz+nz+nzb-1] + delta - (vel[(nxb)*rnz+nz+nzb-1] - delta) +1) + v_ave[nxb-1-ix] - delta;
			vel[iz*rnz+rnz-1-ix] = rand()%(int)(vel[(nxb)*rnz+nz+nzb-1] + delta - (v_ave[nxb-1-ix] - delta) +1) + v_ave[nxb-1-ix] - delta;
		}
	}
	/* Canto inferior direito */
	for(iz=0;iz<nzb;iz++){
		for(ix=0;ix<=iz;ix++){
			v_ave[iz] = vel[(nxb+nx-1)*rnz+nz+nzb-1] - (vel[(nxb+nx-1)*rnz+nz+nzb-1]-l_lim)*(iz)/(nzb-1);
			vel[((nx+2*nxb)-1-ix)*rnz+rnz-1-iz] = rand()%(int)(vel[(nxb+nx-1)*rnz+nz+nzb-1] + delta - (v_ave[nxb-1-ix] - delta) +1) + v_ave[nxb-1-ix] - delta;
			vel[((nx+2*nxb)-1-iz)*rnz+rnz-1-ix] = rand()%(int)(vel[(nxb+nx-1)*rnz+nz+nzb-1] + delta - (v_ave[nxb-1-ix] - delta) +1) + v_ave[nxb-1-ix] - delta;
		}
	}
}

/*
void taper_init(int nxb,int nzb,float F){
	int i;
	float dfrac;
	taperx = alloc1float(nxb);
	taperz = alloc1float(nzb);

	//dfrac = sqrt(-log(F))/(1.*nxb);

	for(i=0;i<nxb;i++){
		//taperx[i] = exp(-pow((dfrac*(nxb-i)),2));
		taperx[i] = exp(-pow((F*(nxb-i)),2));
	}

	//dfrac = sqrt(-log(F))/(1.*nzb);

	for(i=0;i<nzb;i++){
		taperz[i] = exp(-pow((F*(nzb-i)),2));
		//printf("%d , %f \n",i,taperz[i]);
	}
	return;
}
*/

void taper_init(int nxb,int nzb,float F){
	int i;
	float dfrac;
	taperx = alloc1float(nxb);
	taperz = alloc1float(nzb);

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

void taper_apply(float **pp,int nx, int nz, int nxb, int nzb){
	int itz, itx,i;
	
	for(itx=0;itx<nx+2*nxb;itx++){
		for(itz=0;itz<nzb;itz++){
			pp[itx][itz] *= taperz[itz];
		}
		for(itz=nzb-1,i=0;itz>-1;itz--,i++){
			pp[itx][nz+nzb+i] *= taperz[itz];
		}
	}
	for(itz=0;itz<nz+2*nzb;itz++){
		for(itx=0;itx<nxb;itx++){
			pp[itx][itz] *= taperx[itx];
		}
		for(itx=nxb-1,i=0;itx>-1;itx--,i++){
			pp[nx+nxb+i][itz] *= taperx[itx];
		}
	}
	return;
}

void taper_apply2(float **pp,int nx, int nz, int nxb, int nzb){
	int itz, itx, itxr;

	for(itx=0;itx<nx+2*nxb;itx++){
		for(itz=0;itz<nzb;itz++){
			pp[itx][itz] *= taperz[itz];
		}
	}
	for(itx=0,itxr=nx+2*nxb-1;itx<nxb;itx++,itxr--){
		for(itz=0;itz<nzb;itz++){
			pp[itx][itz]  *= taperx[itx];
			pp[itxr][itz] *= taperx[itx];
		}
	}
	return;
}

void taper_destroy(){
	free1float(taperx);
	free1float(taperz);
	return;
}

/*static void damp(int nbw, float abso, float *bw)
{
	int i;
	float pi, delta;

	pi = 4. * atan(1.);
	delta = pi / nbw;

	for (i=0; i<nbw; i++) {
		bw[i] = 1.0 - abso * (1.0 + cos(i*delta)) * 0.5;
	}

	return;
}*/

