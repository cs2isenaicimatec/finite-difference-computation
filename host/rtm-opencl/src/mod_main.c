/* Acoustic wavefield modeling using finite-difference method
Leonardo Gómez Bernal, Salvador BA, Brazil
August, 2016 */

#include <stdio.h>
#include <time.h>
#include <omp.h>

#include "su.h"
#include "fd.h"

#include "cwp.h"

#include "ptsrc.h"
#include "taper.h"

char *sdoc[] = {	/* self documentation */
	" Seismic modeling using acoustic wave equation ",
	"				               ",
	NULL};
/* global variables */

/* file names */
#ifdef OPENCL	
char *kernelPath = NULL, *tmpdir = NULL, *vpfile = NULL, *datfile = NULL, *vel_ext_file = NULL, file[100];
#else
char *tmpdir = NULL, *vpfile = NULL, *datfile = NULL, *vel_ext_file = NULL, file[100];
#endif

/* size */
int nz, nx, nt;
float dz, dx, dt;

/* adquisition geometry */
int ns, sz, fsx, ds, gz;

/* boundary */
int nxb, nzb, nxe, nze;
float fac;

/* propagation */
int order; 
float fpeak;

/* arrays */
int *sx;

/* prototypes */

int main (int argc, char **argv){
	/* model file and data pointers */
	FILE *fsource = NULL, *fvel_ext = NULL, *fd_obs = NULL, *fvp = NULL, 
		*fsns = NULL,*fsns2 = NULL, *fsnr = NULL, *fimg = NULL, *flim = NULL, *fimg_lap = NULL;
	int elapsed;
	clock_t begin, end;
	long int time_spent;
 	begin = clock();
	float execTime;
	struct timeval st, et;
	gettimeofday(&st, NULL);
	/* iteration variables */
	int iz, ix, it, is;

	/* auxiliar */
	int iss, rnd, vel_ext_flag=0;

	/* arrays */
	float *srce;
	float **vp = NULL, **vpe = NULL, **vpex = NULL;

	/* propagation variables */
	float **PP,**P,**PPR,**PR,**tmp;
	float ***swf, ***upb, ***snaps, **vel2, ***d_obs, ***vel_ext_rnd;
	float **imloc, **img, **img_lap;

	/* initialization admiting self documentation */
	initargs(argc, argv);
	requestdoc(1);

	/* read parameters */
#ifdef OPENCL
	MUSTGETPARSTRING("kernel",&kernelPath);		// path to kernel file
#endif
	MUSTGETPARSTRING("tmpdir",&tmpdir);		// directory for data
	MUSTGETPARSTRING("vpfile",&vpfile);		// vp model
	MUSTGETPARSTRING("datfile",&datfile);	// observed data (seismogram)
	MUSTGETPARINT("nz",&nz); 				// number of samples in z
	MUSTGETPARINT("nx",&nx); 				// number of samples in x
	MUSTGETPARINT("nt",&nt); 				// number of time steps
	MUSTGETPARFLOAT("dz",&dz); 				// sampling interval in z
	MUSTGETPARFLOAT("dx",&dx); 				// sampling interval in x
	MUSTGETPARFLOAT("dt",&dt); 				// sampling interval in t
	MUSTGETPARFLOAT("fpeak",&fpeak); 		// souce peak frequency

	if(getparstring("vel_ext_file",&vel_ext_file)) vel_ext_flag = 1;
	if(!getparint("iss",&iss)) iss = 0;	 	// save snaps of this source
	if(!getparint("ns",&ns)) ns = 1;	 	// number of sources
	if(!getparint("sz",&sz)) sz = 0; 		// source depth
	if(!getparint("fsx",&fsx)) fsx = 0; 	// first source position
	if(!getparint("ds",&ds)) ds = 1; 		// source interval
	if(!getparint("gz",&gz)) gz = 0; 		// receivor depth

	if(!getparint("order",&order)) order = 8;	// FD order
	if(!getparint("nzb",&nzb)) nzb = 40;		// z border size
	if(!getparint("nxb",&nxb)) nxb = 40;		// x border size
	if(!getparfloat("fac",&fac)) fac = 0.7;		// damping factor
	// if(!getparint("rnd",&rnd)) rnd = 1;		    // random vel. border

	fprintf(stdout,"**************************************************\n");
	fprintf(stdout,"* vp = %s\n*d_obs = %s\n* vel_ext_file = %s\n* vel_ext_flag = %d \n",vpfile,datfile,vel_ext_file,vel_ext_flag);
	fprintf(stdout,"* nz = %d, nx = %d, nt = %d \n",nz,nx,nt);
	fprintf(stdout,"* dz = %f, dx = %f, dt = %f \n",dz,dx,dt);
	fprintf(stdout,"* ns = %d, sz = %d, fsx = %d, ds = %d, gz = %d \n",ns,sz,fsx,ds,gz);
	fprintf(stdout,"* order = %d, nzb = %d, nxb = %d, F = %f, rnd = %d \n",order,nzb,nxb,fac,rnd);

	/* create source vector  */
	srce = alloc1float(nt);
	ricker_wavelet(nt, dt, fpeak, srce);

	sx = alloc1int(ns);
	for(is=0; is<ns; is++){
		sx[is] = fsx + is*ds + nxb;
	}
	sz += nzb;
	gz += nzb;

	/* add boundary to models */
	nze = nz + 2 * nzb;
	nxe = nx + 2 * nxb;

	/*read randomic vel. models (per source) */
	if(vel_ext_flag){
		vel_ext_rnd = alloc3float(nze,nxe,ns);
		memset(**vel_ext_rnd,0,nze*nxe*ns*sizeof(float));
		fvel_ext = fopen(vel_ext_file,"r");
		fread(**vel_ext_rnd,sizeof(float),nze*nxe*ns,fvel_ext);
		fclose(fvel_ext);
	}

	/*read observed data (seism.) */
	d_obs = alloc3float(nt,nx,ns);
	memset(**d_obs,0,nt*nx*ns*sizeof(float));
	fd_obs = fopen(datfile,"r");
	fread(**d_obs,sizeof(float),nt*nx*ns,fd_obs);
	fclose(fd_obs);

	float **d_obs_aux=(float**)malloc(ns*sizeof(float*));
	for(int i=0; i<ns; i++) 
		d_obs_aux[i] = (float*)malloc((nt*nx)*sizeof(float)); 
	
	for(int i=0; i<ns; i++){
		for(int j=0; j<nx; j++){
			for(int k=0; k<nt; k++)
				d_obs_aux[i][j*nt+k] = d_obs[i][j][k]; 
		}
	}
	
	/* read parameter models */
	vp = alloc2float(nz,nx);
	memset(*vp,0,nz*nx*sizeof(float));
	fvp = fopen(vpfile,"r");
	fread(vp[0],sizeof(float),nz*nx,fvp);
	fclose(fvp);
	/* vp size estended to vpe */
	vpe = alloc2float(nze,nxe);
	vpex = vpe;

	for(ix=0; ix<nx; ix++){
		for(iz=0; iz<nz; iz++){
			vpe[ix+nxb][iz+nzb] = vp[ix][iz]; 
		}
	}
	/* allocate vel2 for vpe^2 */
	PP = alloc2float(nze,nxe);
	P = alloc2float(nze,nxe);
	PPR = alloc2float(nze,nxe);
	PR = alloc2float(nze,nxe);
	upb = alloc3float(order/2,nxe,nt);
	// swf = alloc3float(nz,nx,nt);
	snaps = alloc3float(nze,nxe,2);
	imloc = alloc2float(nz,nx);
	img = alloc2float(nz,nx);
	img_lap = alloc2float(nz,nx);

	vel2 = alloc2float(nze,nxe);

	/* initialize wave propagation */
#ifdef OPENCL
	fd_init(order,nxe,nze,nxb,nzb,nt,ns,fac,dx,dz,dt,kernelPath);
#else
	fd_init(order,nxe,nze,dx,dz,dt);
#endif

	taper_init(nxb,nzb,fac);

	char filePath [100];
	sprintf(filePath, "%s/dir.snaps", tmpdir);
	fsns = fopen(filePath,"w");
	sprintf(filePath, "%s/dir.snaps_rec", tmpdir);
	fsns2 = fopen(filePath,"w");
	sprintf(filePath, "%s/dir.snapr", tmpdir);
	fsnr = fopen(filePath,"w");
	sprintf(filePath, "%s/dir.image", tmpdir);
	fimg = fopen(filePath,"w");
	sprintf(filePath, "%s/dir.image_lap", tmpdir);	
	fimg_lap = fopen(filePath,"w");
	
	memset(*img,0,nz*nx*sizeof(float));
	memset(*img_lap,0,nz*nx*sizeof(float));
	for(is=0; is<ns; is++){
		fprintf(stdout,"** source %d, at (%d,%d) \n",is+1,sx[is]-nxb,sz-nzb);
		/* Calc (or load) velocity model border */
		if (vel_ext_flag){
			vpe = vel_ext_rnd[is];					// load hybrid border vpe from file
		}else{
			extendvel_linear(nx,nz,nxb,nzb,vpe); 	// hybrid border (linear randomic)
		}

		/* vel2 = vpe^2 */
		for(ix=0; ix<nx+2*nxb; ix++){
			for(iz=0; iz<nz+2*nzb; iz++){
				vel2[ix][iz] = vpe[ix][iz]*vpe[ix][iz];
			}
		}

		memset(*PP,0,nze*nxe*sizeof(float));
		memset(*P,0,nze*nxe*sizeof(float));

		/* Forward propagation */
		fd_step_forward(order,P,PP,vel2,upb,nze,nxe,nt,is,sz,sx,srce);
		
		fprintf(stdout,"\n");

		for(iz=0; iz<nze; iz++){
			for(ix=0; ix<nxe; ix++){
				snaps[0][ix][iz] = P[ix][iz];
				snaps[1][ix][iz] = PP[ix][iz];
			}
		}

		fprintf(stdout,"** backward propagation %d, at (%d,%d) \n",is+1,sx[is]-nxb,sz-nzb);

		memset(*PP,0,nze*nxe*sizeof(float));
		memset(*P,0,nze*nxe*sizeof(float));
		memset(*PPR,0,nze*nxe*sizeof(float));
		memset(*PR,0,nze*nxe*sizeof(float));
		memset(*imloc,0,nz*nx*sizeof(float));

		/* Reverse propagation */
    	           
		fd_step_back(order,P,PP,PR,PPR,vel2,upb,sx,nze,nxe,nt,is,it,sz,gz,snaps,imloc,d_obs_aux,srce);

		fprintf(stdout,"\n");

		/* stack migrated images */
		for(iz=0; iz<nz; iz++){
			for(ix=0; ix<nx; ix++){
				img[ix][iz] += imloc[ix][iz];
			}
		}
	}
	
#ifdef  PERF_COUNTERS
	fd_print_report(nxe, nze);
	gettimeofday(&et, NULL);
   	elapsed = ((et.tv_sec - st.tv_sec) * 1000000) + (et.tv_usec - st.tv_usec);
   	execTime += (elapsed*1.0);
   	printf("> Exec Time    = %.1f (us)\n",execTime);
	printf("> ================================================ \n\n");
#endif
	fwrite(*img,sizeof(float),nz*nx,fimg);

	// fwrite(*img_lap,sizeof(float),nz*nx,fimg_lap);

	fclose(fsns);
	fclose(fsns2);
	fclose(fsnr);
	fclose(fimg);
	// fclose(fimg_lap);

    /* release memory */
    fd_destroy();
	taper_destroy();
	free1int(sx);
	free1float(srce);
	free2float(vp);
	free2float(P);
	free2float(PP);
	free2float(PR);
	free2float(PPR);
	// free3float(swf);
	free3float(snaps);
	free2float(imloc);
	free2float(img);
	free2float(img_lap);
	free2float(vpex);
	free2float(vel2);
	free3float(upb);
	free3float(d_obs);
	if(vel_ext_flag) free3float(vel_ext_rnd);
	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Execution time - Min %d Seg %d\n",(time_spent/60),(time_spent%60));

	return(CWP_Exit());
}
