/* Acoustic wavefield modeling using finite-difference method
Leonardo Gómez Bernal, Salvador BA, Brazil
August, 2016 */

#include<stdio.h>
#include"su.h"

#include "fd.h"
#include "ptsrc.h"
#include "taper.h"
#include "fpga.h"

char *sdoc[] = {	/* self documentation */
	" Seismic modeling using acoustic wave equation ",
	"				               ",
	NULL};

/* global variables */

/* file names */
char *tmpdir = NULL, *vpfile = NULL, file[100];

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



int rtm (int argc, char **argv){
	/* model file and data pointers */
	FILE *fvp = NULL, *fsns = NULL, *fsnr = NULL, *fdat = NULL, *fimg = NULL, *flim = NULL;

	/* iteration variables */
	int iz, ix, it, is;

	/* arrays */
	float *srce;
	float **vp = NULL;

	/* propagation variables */
	float **PP,**P,**tmp;
	float ***swf, **vel2, **data;
	float **imloc, **img;

	/* initialization admiting self documentation */
	initargs(argc, argv);
	requestdoc(1);

	/*initargs(argc, argv);
	srand(time(0));*/

	/* read parameters */
	MUSTGETPARSTRING("tmpdir",&tmpdir);		// directory for data
	MUSTGETPARSTRING("vpfile",&vpfile);		// vp model
	MUSTGETPARINT("nz",&nz); 			// number of samples in z
	MUSTGETPARINT("nx",&nx); 			// number of samples in x
	MUSTGETPARINT("nt",&nt); 			// number of time steps
	MUSTGETPARFLOAT("dz",&dz); 			// sampling interval in z
	MUSTGETPARFLOAT("dx",&dx); 			// sampling interval in x
	MUSTGETPARFLOAT("dt",&dt); 			// sampling interval in t
	MUSTGETPARFLOAT("fpeak",&fpeak); 		// souce peak frequency

	if(!getparint("ns",&ns)) ns = 1;	 	// number of sources
	if(!getparint("sz",&sz)) sz = 0; 		// source depth
	if(!getparint("fsx",&fsx)) fsx = 0; 		// first source position
	if(!getparint("ds",&ds)) ds = 1; 		// source interval
	if(!getparint("gz",&gz)) gz = 0; 		// receivor depth

	if(!getparint("order",&order)) order = 8;	// FD order
	if(!getparint("nzb",&nzb)) nzb = 40;		// z border size
	if(!getparint("nxb",&nxb)) nxb = 40;		// x border size
	if(!getparfloat("fac",&fac)) fac = 0.7;		// damping factor

	// fprintf(stdout,"## vp = %s\n",vpfile);
	// fprintf(stdout,"## nz = %d, nx = %d, nt = %d\n",nz,nx,nt);
	// fprintf(stdout,"## dz = %f, dx = %f, dt = %f\n",dz,dx,dt);
	// fprintf(stdout,"## ns = %d, sz = %d, fsx = %d, ds = %d, gz = %d\n",ns,sz,fsx,ds,gz);
	// fprintf(stdout,"## order = %d, nzb = %d, nxb = %d, F = %f\n",order,nzb,nxb,fac);

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

	/* read parameter models */
	vp = alloc2float(nz,nx);
	memset(*vp,0,nz*nx*sizeof(float));

	fvp = fopen(vpfile,"r");
	/*for(ix=0; ix<nx; ix++){
		fread(&vp[ix+nxb][nzb],sizeof(float),nz,fvp);
	}*/
	fread(vp[0],sizeof(float),nz*nx,fvp);
	fclose(fvp);

	/* initialize velocity */
	vel2 = alloc2float(nze,nxe);

	for(ix=0; ix<nx; ix++){
		for(iz=0; iz<nz; iz++){
			vel2[ix+nxb][iz+nzb] = vp[ix][iz]*vp[ix][iz];
		}
	}

	extendvel(nx,nz,nxb,nzb,*vel2);

	/* initialize wave propagation */
	fd_init(order,nxe,nze,dx,dz,dt);
	taper_init(nxb,nzb,fac);

	PP = alloc2float(nze,nxe);
	P = alloc2float(nze,nxe);
	data = alloc2float(nt,nx);
	swf = alloc3float(nz,nx,nt);
	imloc = alloc2float(nz,nx);
	img = alloc2float(nz,nx);

	char path [50];
	sprintf (path, "%s/%s", tmpdir, "dir.snaps");
	fsns = fopen(path,"w");
	sprintf (path, "%s/%s", tmpdir, "dir.snapr");
	fsnr = fopen(path,"w");
	sprintf (path, "%s/%s", tmpdir, "dir.data");
	fdat = fopen(path,"w");
	sprintf (path, "%s/%s", tmpdir, "dir.locimg");
	flim = fopen(path,"w");
	sprintf (path, "%s/%s", tmpdir, "dir.image");
	fimg = fopen(path,"w");

	memset(*img,0,nz*nx*sizeof(float));

	///////////////////////////////////////////////////////////////////////////////////////
	ssize_t fpga_fid;
#ifdef LCORE_FPGA
	fpga_fid = fpga_open(LCORE_DEVICE_DEFAULT_NODE);
	if (fpga_fid < 0){
		printf ("\nError. Could not open FPGA device at %s. Abort! \n", LCORE_DEVICE_DEFAULT_NODE);
		exit (-1);
	}
	fpga_config(fpga_fid, nxe, nze, LCORE_DEFAULT_D2XINV, LCORE_DEFAULT_D2ZINV,
	LCORE_DEFAULT_IMGSTART_ADDR, LCORE_DEFAULT_LAPLSTART_ADDR);
#endif
	///////////////////////////////////////////////////////////////////////////////////////

	for(is=0; is<ns; is++){
		////fprintf(stdout,"\n** source %d, at (%d,%d)\n",is+1,sx[is]-nxb,sz-nzb);

		memset(*PP,0,nze*nxe*sizeof(float));
		memset(*P,0,nze*nxe*sizeof(float));
		memset(*data,0,nx*nt*sizeof(float));
		memset(**swf,0,nz*nx*nt*sizeof(float));

		for(it=0; it<nt; it++){
			/* propagate to t+dt */
			fd_step(fpga_fid, order,P,PP,vel2,nze,nxe, 0);

			/* add source */
			//PP[sx[is]][sz] += srce[it];
			ptsrc(sx[is],sz,nxe,nze,srce[it],PP);

			/* boundary conditions */
			taper_apply(PP,nx,nz,nxb,nzb);
			taper_apply(P,nx,nz,nxb,nzb);

			/* save data and source wavefield */
			for(ix=0; ix<nx; ix++){
				data[ix][it] = P[ix+nxb][gz];
			}

			for(iz=0; iz<nz; iz++){
				for(ix=0; ix<nx; ix++){
					swf[it][ix][iz] = P[ix+nxb][iz+nzb];
				}
			}

			if(it%10 == 0 && is == (int)ns/2-1){
				//fwrite(*P,sizeof(float),nxe*nze,fsnap);
				fwrite(*swf[it],sizeof(float),nx*nz,fsns);
			}
			//if((it+1)%100 == 0){fprintf(stdout,"\r* it = %d / %d (%d%)",it+1,nt,(100*(it+1)/nt));fflush(stdout);}
			// if(it%100 == 0)fprintf(stdout,"* it = %d / %d \n",it,nt);

			tmp = PP;
			PP = P;
			P = tmp;
		}
		fwrite(*data,sizeof(float),nt*nx,fdat);

		//fprintf(stdout,"\n** backward propagation %d, at (%d,%d) \n",is+1,sx[is]-nxb,sz-nzb);

		memset(*PP,0,nze*nxe*sizeof(float));
		memset(*P,0,nze*nxe*sizeof(float));
		memset(*imloc,0,nz*nx*sizeof(float));

		for(it=0; it<nt; it++){
			/* propagate to t-dt */
			fd_step(fpga_fid, order,P,PP,vel2,nze,nxe, it == 650 && is==2);

			/* add seismograms */
			for(ix=0; ix<nx; ix++){
				PP[ix+nzb][gz] += data[ix][nt-it];
			}

			/* boundary conditions */
			taper_apply(PP,nx,nz,nxb,nzb);
			taper_apply(P,nx,nz,nxb,nzb);

			/* save receivers wavefield */
			if(it%10 == 0 && is == (int)ns/2-1){
				for(ix=0; ix<nx; ix++){
					for(iz=0; iz<nz; iz++){
						fwrite(&P[ix+nxb][iz+nzb],sizeof(float),1,fsnr);
						//fwrite(&P[ix][iz],sizeof(float),1,fsnr);
					}
				}
			}
			//if((it+1)%100 == 0){fprintf(stdout,"\r* it = %d / %d (%d%)",it+1,nt,(100*(it+1)/nt));fflush(stdout);}
			

			/* apply imaging condition */
			for(iz=0; iz<nz; iz++){
				for(ix=0; ix<nx; ix++){
					imloc[ix][iz] += swf[nt-it-1][ix][iz] * P[ix+nxb][iz+nzb];
				}
			}

			tmp = PP;
			PP = P;
			P = tmp;
		}
		/* save local image */
		fwrite(*imloc,sizeof(float),nz*nx,flim);

		/* stack migrated images */
		for(iz=0; iz<nz; iz++){
			for(ix=0; ix<nx; ix++){
				img[ix][iz] += imloc[ix][iz];
			}
		}
	}

	/* save stacked image */
	fwrite(*img,sizeof(float),nz*nx,fimg);

	fclose(fsns);
	fclose(fsnr);
	fclose(fdat);
	fclose(fimg);
	fclose(flim);

	fd_destroy();
	taper_destroy();

	/* release memory */
	free1int(sx);
	free1float(srce);
	free2float(vp);
	free2float(P);
	free2float(PP);
	free2float(data);
	free3float(swf);
	free2float(imloc);
	free2float(img);

	fd_print_report(nxe, nze);

#ifdef LCORE_FPGA
	close(fpga_fid);
#endif
	printf ("Fuck yeah! \n");

	return(CWP_Exit());
}
