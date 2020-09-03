#ifdef USE_MPI
#include <mpi.h>
#endif
#include <sys/time.h>
#include <stdio.h>
#include <float.h>
#include "rtm.h"
#include "su.h"
#include "fd.h"
#include "ptsrc.h"
#include "taper.h"
#include "math.h"

#define TAG_SHIFT 4

/**
* rtm_swapptr
* swap 2D img matrices.
*/
void rtm_swapptr(void *** ptrA, void *** ptrB){
	void ** swap = *ptrB;
	*ptrB = *ptrA;
	*ptrA = swap;
}

/**
* rtm_process
*/
#ifdef USE_MPI
void rtm_process(RTMExecParam * execParam, RTMShot * rtmShots){
	
	MPI_Status status;
	int curShot=0, j=0;
	int shotsLeft;
	int procs;
	int pending;
	char * freeProcesses;
	int * curShotPerProcess;
	int source;
	int probeFlag;
	int imgPerProc = 0;

	imgPerProc = 1;

	procs = execParam->numberOfProcesses;
	freeProcesses = (char*)malloc(procs*sizeof(char));
	curShotPerProcess = (int*)malloc(procs*sizeof(int));		

	if (procs==1){ // single process
		printf("> RTM %s Process (Nprocs= %d).", 
			execParam->modeling?"Modeling":"Migration",procs); fflush(stdout);
		// run all shots on this process
		for (curShot=0; curShot< execParam->ns; curShot++){
			printf ("\n> Shot %d \n", curShot);
			fflush(stdout);
			if (execParam->modeling==0){
				rtm_migration(execParam, &rtmShots[curShot]);
			}else{
				
				rtm_modeling(execParam, &rtmShots[curShot]);
			}
		}
	}else{ // multi process: master controls shot distribution
		pending = 1;
		procs = (execParam->numberOfProcesses);
		memset(freeProcesses, 0, procs*sizeof(char));
		memset(curShotPerProcess, 0, procs*sizeof(int));
		if (execParam->processId==0){ // schedule shots
			printf("> Running RTM process. Type= %s Nprocs= %d", 
			execParam->modeling?"Modeling":"Migration",procs); fflush(stdout);
			shotsLeft = execParam->ns;
			while (shotsLeft > 0 || pending!=0){
				// distribute shots for free processes
				for (j=1; j<procs && shotsLeft > 0; j++){
					if (freeProcesses[j]==0){
						// sends lefting process to another node
						curShot=(--shotsLeft);
						printf ("\n> Sending shot %d to proc %d \n",
							curShot, j); fflush(stdout);
						MPI_Send( 	&curShot,
								    1,
								    MPI_INT,
								    j,
								    1,
								    MPI_COMM_WORLD);
						freeProcesses[j]=imgPerProc;
						curShotPerProcess[j]=curShot;
					}
				}
				// wait for finished shots
				MPI_Iprobe( MPI_ANY_SOURCE,
					    MPI_ANY_TAG,
					    MPI_COMM_WORLD,
					    &probeFlag,
					    &status);
				if (status.MPI_ERROR==0 && probeFlag){
					source = status.MPI_SOURCE;
					curShot = curShotPerProcess[source];
					if(execParam->modeling==0){
						MPI_Recv(
							    rtmShots[curShot].sfloat.imloc[0],
							    execParam->nz*execParam->nx,
							    MPI_FLOAT,
							    source,
							    RTMSHOT_FLOAT,
							    MPI_COMM_WORLD,
							    &status);
						freeProcesses[source]--;
						printf("> Received Shot %d from proc %d \n", curShot, source);
					}else{
						MPI_Recv(
						    rtmShots[curShot].dobs[0],
						    execParam->nt*execParam->nx,
						    MPI_FLOAT,
						    source,
						    MPI_ANY_TAG,
						    MPI_COMM_WORLD,
						    &status);
						freeProcesses[source] = 0;
					}
				}
				pending = 0;
				for (j=1; j<procs; j++){
					if (freeProcesses[j]!=0){
						pending = 1;
					}
				}

			}
			printf ("\n> Sending FINISH cmds \n"); fflush(stdout);
			curShot = -1;
			for (j=1; j<procs; j++){
				// sends lefting process to another node
				MPI_Send( 	&curShot,
						    1,
						    MPI_INT,
						    j,
						    RTMSHOT_FLOAT,
						    MPI_COMM_WORLD);				
			}
		}else{ // multi process: slaves run migration or modeling
			int myShot = 0;
			while(1){ // slave procs keep waiting for new shots
				// read number of myShot
				myShot = 0;
				MPI_Recv(   &myShot,
						    1,
						    MPI_INT,
						    0,
						    1,
						    MPI_COMM_WORLD,
						    &status);
				if (status.MPI_ERROR || myShot >= execParam->ns){
					printf ("> P[%d] reported a communication error. Aborting... \n", 
						execParam->processId);fflush(stdout);
					MPI_Finalize();
					exit(0);
				}else if(myShot == -1){ // end of shots
					// no more shots for you
					printf ("> P[%d] finished all its shots. \n", 
						execParam->processId);fflush(stdout);
					return;
				}else{
					printf ("> P[%d] received shot %d \n", 
						execParam->processId, myShot);fflush(stdout);
					if (execParam->modeling==0){
						rtm_migration(execParam, &rtmShots[myShot]);
						MPI_Send( rtmShots[myShot].sfloat.imloc[0],
								    execParam->nz*execParam->nx,
								    MPI_FLOAT,
								    0,
								    RTMSHOT_FLOAT,
								    MPI_COMM_WORLD);
						
					}else{
						//printf ("> Running modeling process %d \n", execParam->processId);
						rtm_modeling(execParam, &rtmShots[myShot]);
						MPI_Send( rtmShots[myShot].dobs[0],
								  execParam->nt*execParam->nx,
								  MPI_FLOAT,
								  0,
								  RTMSHOT_FLOAT,
								  MPI_COMM_WORLD);
					}

				}
			}

		}
	}
	free(freeProcesses);
	free(curShotPerProcess);
}
#else

void rtm_process(RTMExecParam * execParam, RTMShot * rtmShots){
	int curShot=0;
	char * freeProcesses;
	int * curShotPerProcess;
	int source;

	printf("> RTM %s Process (Nprocs= %d).", 
		execParam->modeling?"Modeling":"Migration",0); fflush(stdout);
	// run all shots on this process
	for (curShot=0; curShot< execParam->ns; curShot++){
		printf ("\n> Shot: %d \n", curShot);
		fflush(stdout);
		if (execParam->modeling==0){
			rtm_migration(execParam, &rtmShots[curShot]);
		}else{
			rtm_modeling(execParam, &rtmShots[curShot]);
		}
	}
}
#endif

/**
* rtm_modeling
*/
void rtm_modeling(RTMExecParam * execParam, RTMShot * rtmShot){
	rtm_extvel(execParam);
	execParam->exec_state = RTMEXEC_FWRD;
	rtm_mod_forward(rtmShot, execParam);
	if (execParam->processId==1){
		printf("\n");fflush(stdout);
	}
}
/**
* rtm_migration
*/
void rtm_migration(RTMExecParam * execParam, RTMShot * rtmShot){
	struct timeval st, et;
	int it, ix, iz, nt, nx, is;

	//rtm_extvel_hyb(execParam, rtmShot->shotNumber /*execParam->processId*/);
	rtm_extvel_hyb(execParam, execParam->processId);

	nt = execParam->nt;
	nx = execParam->nx;
	is = rtmShot->shotNumber;
	FILE * fp=fopen("debug/imloc/dobsserial.txt", "w");
	for(it=0; it<nt; it++){
		for(ix=0; ix<nx; ix++){
			int ptr = (ix)*nt + ((nt-1) - it);
			fprintf(fp, "it=%d ix=%d[ptr=%d] %.20f \n",
				it, ix, ptr, 
				(execParam->d_obs[is][ix][nt-1-it]));
		}
	}
	fclose(fp);


#ifdef OPENCLRTM
	printf ("> Forward Propagation: \n"); fflush(stdout);
	execParam->exec_state = RTMEXEC_FWRD;
	printf("> Serial \n");
	gettimeofday(&st, NULL);
	rtm_mig_forward(rtmShot, execParam);
	gettimeofday(&et, NULL);
	rtmShot->report.fwSerialAVGTime += elapsedTimeMs(st, et);
	printf("> OpenCl \n");
	// getchar();
	// int k=10000;
	// printf("Running... \n");
	// while(--k)
	// gettimeofday(&st, NULL);
	//rtmcl_mig_forward(rtmShot, execParam);
	// gettimeofday(&et, NULL);
	// rtmShot->report.fwDeviceAVGTime += elapsedTimeMs(st, et);
	printf ("> Backward Propagation: <\n"); fflush(stdout);

	printf("> Serial \n");
	gettimeofday(&st, NULL);
	execParam->exec_state = RTMEXEC_BWRD;
	rtm_mig_backward(rtmShot, execParam);
	gettimeofday(&et, NULL);
	rtmShot->report.bwSerialAVGTime += elapsedTimeMs(st, et);

	printf("> OpenCl \n");
	gettimeofday(&st, NULL);
	rtmcl_mig_backward(rtmShot, execParam);
	gettimeofday(&et, NULL);
	rtmShot->report.bwDeviceAVGTime += elapsedTimeMs(st, et);
#else
	printf ("> Forward Propagation: \n");
	execParam->exec_state = RTMEXEC_FWRD;
	gettimeofday(&st, NULL);
	rtm_mig_forward(rtmShot, execParam);
	gettimeofday(&et, NULL);
	rtmShot->report.fwSerialAVGTime += elapsedTimeMs(st, et);
	printf ("> Backward Propagation: \n");
	execParam->exec_state = RTMEXEC_BWRD;
	gettimeofday(&st, NULL);
	rtm_mig_backward(rtmShot, execParam);
	gettimeofday(&et, NULL);
	rtmShot->report.bwSerialAVGTime += elapsedTimeMs(st, et);
#endif
	rtmShot->report.serialAVGTime = rtmShot->report.fwSerialAVGTime 
			+ rtmShot->report.bwSerialAVGTime;
	rtmShot->report.deviceAVGTime = rtmShot->report.fwDeviceAVGTime 
			+ rtmShot->report.bwDeviceAVGTime;

	char fname[100];
#ifdef 	OPENCLRTM
	sprintf(fname, "debug/imloc/imloccl%d.txt", execParam->nt);
#else
	sprintf(fname, "debug/imloc/imlocserial%d.txt", execParam->nt);
#endif	
	FILE * fimloc = fopen(fname, "w");
	for(ix=0; ix<execParam->nx; ix++){
		for(iz=0; iz<execParam->nz; iz++){
			float s0 = rtmShot->sfloat.imloc[ix][iz];
			fprintf(fimloc, "(%d,%d) %.20f \n",
				ix, iz, s0);
		}
	}
	fclose(fimloc);
	/**************************************************/
}

void rtm_mod_forward(RTMShot * shot, RTMExecParam * execParam){

	int it, ix, iz, is;
	int nx, nz,nxe, nze, nxb, nzb, nt, order, gz;
	int32_t tmp32_0;

	nx  = execParam->nx;
	nz  = execParam->nz;
	nxe = execParam->nxe;
	nze = execParam->nze;
	nxb = execParam->nxb;
	nzb = execParam->nzb;
	gz = execParam->gz;
	order = execParam->order;
	nt = execParam->nt;
	is = shot->shotNumber;

	memset(*shot->sfloat.PP,0,nze*nxe*sizeof(float));
	memset(*shot->sfloat.P,0,nze*nxe*sizeof(float));
	memset(*shot->sfloat.laplace,0,nze*nxe*sizeof(float));
	memset(*shot->dobs,0,nx*nt*sizeof(float));

	/* Forward propagation */
	for(it=0; it<nt; it++){
		rtm_swapptr(&shot->sfloat.P, &shot->sfloat.PP);

		taper_apply(shot->sfloat.PP,nx,nz,nxb,nzb, execParam->taperx, execParam->taperz);
		taper_apply(shot->sfloat.P,nx,nz,nxb,nzb, execParam->taperx, execParam->taperz);

		/* propagate to t+dt */
		fd_step_float(execParam,shot->sfloat.P, shot->sfloat.PP,shot->sfloat.laplace,
				shot->start_x, shot->end_x,shot->start_z, shot->end_z);

		/* add source */
		ptsrc(shot->sx,shot->sz,
			execParam->srce_wavelet[it],
			shot->sfloat.PP);

		/* save data and source wavefield */
		for(ix=0; ix<nx; ix++){
			shot->dobs[ix][it] = shot->sfloat.PP[ix+nxb][gz];
		}			
		if (execParam->processId==1 || execParam->processId==0){		
			if(it%100 == 0){ 
				printf("\r* MOD[%d]: it = %d / %d",
					execParam->processId, it,nt);
				fflush(stdout);
			}
		}
	}
}

/**
* rtm_mig_forward
* Runs forward propagation
*/
void rtm_mig_forward(RTMShot * shot, RTMExecParam * execParam){

	int it, ix, iz;
	int nx, nz, nxe, nze, nxb, nzb, nt, order;
	char logPrinted = 0;
	nx  = execParam->nx;
	nz  = execParam->nz;
	nxe = execParam->nxe;
	nze = execParam->nze;
	nxb = execParam->nxb;
	nzb = execParam->nzb;
	order = execParam->order;
	nt = execParam->nt;
	memset(*shot->sfloat.PP,0,nze*nxe*sizeof(float));
	memset(*shot->sfloat.P,0,nze*nxe*sizeof(float));
	memset(*shot->sfloat.laplace,0,nze*nxe*sizeof(float));
	/* Forward propagation */
	for(it=0; it<nt; it++){
		execParam->current_it = it;
		rtm_swapptr((void ***) &shot->sfloat.P,(void ***)&shot->sfloat.PP);
			/* boundary conditions */
		taper_apply_hyb(shot->sfloat.P,nx,nz,nxb,nzb, 
			execParam->taperx, execParam->taperz);
			taper_apply_hyb(shot->sfloat.PP,nx,nz,nxb,nzb, 
			execParam->taperx, execParam->taperz);
				/* propagate to t+dt */
		fd_step_float(execParam,shot->sfloat.P, shot->sfloat.PP,shot->sfloat.laplace,
				shot->start_x, shot->end_x,shot->start_z, shot->end_z);
		/* add source */
		ptsrc(shot->sx,shot->sz,
			execParam->srce_wavelet[it],
			shot->sfloat.PP);
		/* save upper border */
		for(ix=0; ix<nxe; ix++){
			for(iz=nzb-order/2; iz<nzb; iz++){
				shot->sfloat.upb[it][ix][iz-(nzb-order/2)] = shot->sfloat.PP[ix][iz];
			}
		}
		if (execParam->processId==1 || execParam->processId==0){
			if((it+1)%100 == 0){
				printf("\r* FRWRD[%d]: it = %d / %d (%d %%)",
				execParam->processId,
				it+1,nt,(100*(it+1)/nt));fflush(stdout);
				logPrinted=1;
			}
		}
	}
	// Saving Snaps
	rtm_savesnaps_float(shot->sfloat.snaps, 
		shot->sfloat.PP, 
		shot->sfloat.P, 
		nxe, nze);
	if (execParam->processId==0 && logPrinted){
		printf("\n");
	}
}

/**
* rtm_mig_backward
* Runs backward propagation
*/
void rtm_mig_backward(RTMShot * shot, RTMExecParam * execParam){
	int it, ix, iz, is;
	int nx, nz,nxe, nze, nxb, nzb, nt, order, gz;
	char logPrinted=0;

	nx  = execParam->nx;
	nz  = execParam->nz;
	nxe = execParam->nxe;
	nze = execParam->nze;
	nxb = execParam->nxb;
	nzb = execParam->nzb;
	gz = execParam->gz;
	order = execParam->order;
	nt = execParam->nt;
	is = shot->shotNumber;

	memset(*shot->sfloat.PP,0,nze*nxe*sizeof(float));
	memset(*shot->sfloat.P,0,nze*nxe*sizeof(float));
	memset(*shot->sfloat.PPR,0,nze*nxe*sizeof(float));
	memset(*shot->sfloat.PR,0,nze*nxe*sizeof(float));
	memset(*shot->sfloat.laplace,0,nze*nxe*sizeof(float));
	memset(*shot->sfloat.imloc,0,nz*nx*sizeof(float));
	
	/* Reverse propagation */
	for(it=0; it<nt; it++){
		execParam->current_it = it;
		/* Reconstruct source wavefield */
		if(it==0 || it==1){
			for(ix=0; ix<nxe; ix++){
				for(iz=0; iz<nze; iz++){
					shot->sfloat.PP[ix][iz] = shot->sfloat.snaps[1-it][ix][iz];
				}
			}
		}else{
			/* propagate to t+dt (actually t-dt)*/
			fd_step_float(execParam,shot->sfloat.P, shot->sfloat.PP,shot->sfloat.laplace,
					shot->start_x, shot->end_x,shot->start_z, shot->end_z);
			/* set upb */
			for(ix=0; ix<nxe; ix++){
				for(iz=nzb-order/2; iz<nzb; iz++){
					shot->sfloat.PP[ix][iz] = shot->sfloat.upb[nt-1-it][ix][iz-(nzb-order/2)];
				}
			}
		}
		rtm_swapptr((void ***) &shot->sfloat.P,(void ***)&shot->sfloat.PP);		

		/* boundary conditions */
		taper_apply_hyb(shot->sfloat.PPR,nx,nz,nxb,nzb, 
			execParam->taperx, execParam->taperz);
		taper_apply_hyb(shot->sfloat.PR,nx,nz,nxb,nzb, 
			execParam->taperx, execParam->taperz);

		/* propagate to t-dt */
		fd_step_float(execParam,shot->sfloat.PR, shot->sfloat.PPR,shot->sfloat.laplace,
				shot->start_x, shot->end_x,shot->start_z, shot->end_z);

		/* add seismograms */
		for(ix=0; ix<nx; ix++){
			// Floating calc.
			/**
            if(it==1){
            printf("[%d,%d] dcnt=%d pp=%.20f dobs=%.20f \n", 
                ix, 
                gz,
                ix*nt + ((nt-1)-it),
                shot->sfloat.PPR[ix+nxb][gz],
                execParam->d_obs[is][ix][nt-1-it]
                );
            }
            /**/
			shot->sfloat.PPR[ix+nxb][gz] += (execParam->d_obs[is][ix][nt-1-it]);
		}
			
		/* apply imaging condition */
		for(ix=0; ix<nx; ix++){
			for(iz=0; iz<nz; iz++){
					// Floating point calc
				shot->sfloat.imloc[ix][iz] += 
				shot->sfloat.P[ix+nxb][iz+nzb] * 
				shot->sfloat.PPR[ix+nxb][iz+nzb];

				/**
                if(execParam->current_it==2){
                printf("it=%d [%d,%d](%d) p=%.20f pr=%.20f \n",
                    execParam->current_it, 
                    ix+execParam->nxb, 
                    iz+execParam->nzb,
                    ix*execParam->nz + iz,
                    shot->sfloat.P[ix+nxb][iz+nzb],
                    shot->sfloat.PPR[ix+nxb][iz+nzb]
                    );
                }
                /**/
			}
		}
		rtm_swapptr((void ***) &shot->sfloat.PR,(void ***)&shot->sfloat.PPR);

		/**********************************************************/
		int flag_now = 0;
		if (execParam->processId==1 || execParam->processId==0){
			if((it+1)%100 == 0){
				printf("\r* BKWRD[%d]: it = %d / %d (%d%)",
				execParam->processId,
				it+1,nt,(100*(it+1)/nt));fflush(stdout);
				flag_now = (100*(it+1)/nt)==99;
				logPrinted=1;
			}
		}
		/**********************************************************/
	}
	if (execParam->processId==0 && logPrinted){
		printf("\n");
	}
}


/**
* rtm_savesnaps_float
* Saves snapshots of PP and P floating point images
*/
void rtm_savesnaps_float(float ***	snaps, float ** PP, float ** P, int nxe, int nze){
	int ix, iz;
	for(iz=0; iz<nze; iz++){
		for(ix=0; ix<nxe; ix++){
			snaps[0][ix][iz] = P[ix][iz];
			snaps[1][ix][iz] = PP[ix][iz];
		}
	}
}

/**
* rtm_extvel
* Extends velocity vector
*/
void rtm_extvel(RTMExecParam * execParam){
	
	int ix, iz, nx, nz, nxb, nzb;
	nx = execParam->nx,
	nz = execParam->nz,
	nxb = execParam->nxb,
	nzb = execParam->nzb,

	extendvel(nx,nz,nxb,nzb,
			execParam->vpe[0]); // border ABC

	for(ix=0; ix<nx+2*nxb; ix++){
		for(iz=0; iz<nz+2*nzb; iz++){
			//vel2[ix][iz] = vpe[ix][iz]*vpe[ix][iz];
			execParam->vel2dt2[ix][iz] = 
			execParam->vpe[ix][iz]*execParam->vpe[ix][iz]*execParam->dt*execParam->dt;
		}
	}
}

/**
* rtm_extvel_hyb
* Extends velocity vector
*/
void rtm_extvel_hyb(RTMExecParam * execParam, int is){

	int it, ix, iz;
	int nx, nz, nxe, nze, nxb, nzb, nt, order;
	FILE * fvpe;
	nx = execParam->nx;
	nz = execParam->nz;
	nxe = execParam->nxe;
	nze = execParam->nze;
	nxb = execParam->nxb;
	nzb = execParam->nzb;
	order = execParam->order;

	/* Calc (or load) velocity model border */
	if (execParam->vel_ext_flag){
		execParam->vpe = execParam->vel_ext_rnd[is];					// load hybrid border vpe from file
	}else{
		extendvel_hyb(nx,nz,nxb,nzb,execParam->vpe); 	// hybrid border (linear randomic)
		fvpe = fopen(execParam->vpfile,"r");
		if(fvpe!=NULL){
			fwrite(*execParam->vpe,sizeof(float),nze*nxe,fvpe);// Saves vpe with hybrid border
			fclose(fvpe);
		}else{
			printf("> Error! Couldn't open VP file at %s \n",
				execParam->vpfile);
		}
	}

	/* vel2 = vpe^2 */
	for(ix=0; ix<nx+2*nxb; ix++){
		for(iz=0; iz<nz+2*nzb; iz++){
			execParam->vel2dt2[ix][iz] = 
			execParam->vpe[ix][iz]*execParam->vpe[ix][iz]*execParam->dt*execParam->dt;
		}
	}
}

/**
* rtm_stackimg
*/
void rtm_stackimg(RTMExecParam * execParam, RTMShot * rtmShots){
	int is=0, ix=0, iz=0;
	float 	** tmpImg 	 ;

	tmpImg 	 = alloc2float(execParam->nz,execParam->nx);
	memset(*tmpImg,0,execParam->nz*execParam->nx*sizeof(float));
	memset(*execParam->img_float,0,execParam->nz*execParam->nx*sizeof(float));
	/************************************************/
	for(is=0; is<execParam->ns; is++){
		/* stack migrated images */
		for(ix=0; ix<execParam->nx; ix++){
			for(iz=0; iz< execParam->nz; iz++){
				// Stacking floating point calc local images:
				tmpImg[ix][iz] += rtmShots[is].sfloat.imloc[ix][iz];
				//execParam->img_float[ix][iz] += rtmShots[is].sfloat.imloc[ix][iz];
			}
		}
	}
	/************************************************/
	
	//// laplacian filter for low freq noise reduction ////
	fd_reinit(LAPLACIAN_FILTER_ORDER, execParam->dx, execParam->dz, 
		execParam->dt, execParam->coefsx, execParam->coefsz);

	printf("> Filtering float-point img. \n");
	laplacian_float(0, 0, execParam->nx, execParam->nz, 
		LAPLACIAN_FILTER_ORDER, tmpImg, execParam->coefsx, execParam->coefsz, 
		execParam->img_float, 0);
	free2float(tmpImg);
}

/**
* rtm_loadparams
* Loads execution parameters
*/
void rtm_loadparams(RTMExecParam * exec){
/* read parameters */
	// MUSTGETPARSTRING("tmpdir",&tmpdir);		// directory for data

	MUSTGETPARSTRING("vpfile",&exec->vpfile);		// vp model
	MUSTGETPARSTRING("datfile",&exec->datfile);	// observed data (seismogram)
	MUSTGETPARINT("nz",&exec->nz); 				// number of samples in z
	MUSTGETPARINT("nx",&exec->nx); 				// number of samples in x
	MUSTGETPARINT("nt",&exec->nt); 				// number of time steps
	MUSTGETPARFLOAT("dz",&exec->dz); 				// sampling interval in z
	MUSTGETPARFLOAT("dx",&exec->dx); 				// sampling interval in x
	MUSTGETPARFLOAT("dt",&exec->dt); 				// sampling interval in t
	MUSTGETPARFLOAT("fpeak",&exec->fpeak); 		// souce peak frequency

	if(!getparstring("outdir",&exec->outputpath)) exec->outputpath = 0;
	if(!getparstring("vel_ext_file",&exec->vel_ext_file)) exec->vel_ext_flag = 0;
	if(!getparint("iss",&exec->iss)) exec->iss = 0;	 	// save snaps of this source
	if(!getparint("ns",&exec->ns)) exec->ns = 1;	 	// number of sources
	if(!getparint("sz",&exec->sz)) exec->sz = 0; 		// source depth
	if(!getparint("fsx",&exec->fsx)) exec->fsx = 0; 	// first source position
	if(!getparint("ds",&exec->ds)) exec->ds = 1; 		// source interval
	if(!getparint("gz",&exec->gz)) exec->gz = 0; 		// receivor depth
	if(!getparint("modeling",&exec->modeling)) exec->modeling = 0;
	if(!getparint("order",&exec->order)) exec->order = 8;	// FD order
	if(!getparint("nzb",&exec->nzb)) exec->nzb = 40;		// z border size
	if(!getparint("nxb",&exec->nxb)) exec->nxb = 40;		// x border size
	if(!getparfloat("fac",&exec->fac)) exec->fac = 0.7;		// damping factor
	if(!getparint("rnd",&exec->rnd)) exec->rnd = 0.;		    // random vel. border
	if(!getparint("swindow",&exec->swindow)) exec->swindow = 4000;  // shot window
	if(!getparint("floatrun",&exec->floatrun)) exec->floatrun = 1;	
	if(!getparint("fixrun",&exec->fixrun)) exec->fixrun = 1;
	if(!getparint("nbitsfrac",&exec->nbitsfrac)) exec->nbitsfrac = 23;
	if(!getparint("nbitsint",&exec->nbitsint)) exec->nbitsint = 0;	

#ifdef OPENCLRTM
	if(!getparint("platformid",&exec->clEnv.platform_id)) exec->clEnv.platform_id = 0;
	if(!getparstring("kernelfile",&exec->kernelPath)) exec->kernelPath = 0;
#endif

	//if(0){
	if(exec->processId==0){// only master will print
		printf("## vp = %s, d_obs = %s, vel_ext_flag = %d \n",
			exec->vpfile,exec->datfile,exec->vel_ext_flag);
		fflush(stdout);
		printf("## nz = %d, nx = %d, nt = %d \n",exec->nz,exec->nx,exec->nt);
		printf("## dz = %f, dx = %f, dt = %f \n",exec->dz,exec->dx,exec->dt);
		printf("## ns = %d, sz = %d, fsx = %d, ds = %d, gz = %d \n",
			exec->ns,exec->sz,exec->fsx,exec->ds,exec->gz);
		fflush(stdout);
		printf("## order = %d, nzb = %d, nxb = %d\n",
			exec->order,exec->nzb,exec->nxb);
		printf("## swindow=%d \n",exec->swindow);
		printf("## kernelpath=%s \n", exec->kernelPath);
		printf("## Exec type: %s \n", exec->modeling?"Modeling":"Migration");
		fflush(stdout);
	}
}

/**
* rtm_initparams
* Init execution parameters
*/
void rtm_initparams(RTMExecParam * execParam){
	FILE *fvel_ext = NULL, *fd_obs = NULL, *fvp = NULL;
	int i0, is, ix, iz, it;

	/* allocate memory */
	execParam->sz += execParam->nzb;
	execParam->gz += execParam->nzb;
	/* add boundary to models */
	execParam->nze = execParam->nz + 2 * execParam->nzb;
	execParam->nxe = execParam->nx + 2 * execParam->nxb;

	execParam->coefsx = alloc1float(execParam->order+1);
	execParam->coefsz = alloc1float(execParam->order+1);
	execParam->taperx = alloc1float(execParam->nxb);
	execParam->taperz = alloc1float(execParam->nzb);
	execParam->srce_wavelet = alloc1float(execParam->nt);
	ricker_wavelet(execParam->nt, execParam->dt, execParam->fpeak, execParam->srce_wavelet);

	execParam->sx = alloc1int(execParam->ns);
	for(is=0; is<execParam->ns; is++){
		execParam->sx[is] = execParam->fsx + is*execParam->ds + execParam->nxb;
		if (execParam->sx[is] >= execParam->nxe){
			printf("> Error! Shot %d at position %d exceeds matrix dimensions (%d) \n",
				is, execParam->sx[is], execParam->nxe);
			exit(0);
		}
	}
	/*read randomic vel. models (per source) */
	if(execParam->vel_ext_flag){
		execParam->vel_ext_rnd = alloc3float(execParam->nze,execParam->nxe,execParam->ns);
		memset(**execParam->vel_ext_rnd,0,execParam->nze*execParam->nxe*execParam->ns*sizeof(float));
		fvel_ext = fopen(execParam->vel_ext_file,"r");
		fread(**execParam->vel_ext_rnd,sizeof(float),
			execParam->nze*execParam->nxe*execParam->ns,fvel_ext);
		fclose(fvel_ext);
	}

	/*read observed data (seism.) */
	execParam->d_obs = alloc3float(execParam->nt,execParam->nx,execParam->ns);
	
	if (execParam->modeling==0){
		//memset(**execParam->d_obs,0,execParam->nt*execParam->nx*execParam->ns*sizeof(float));
		fd_obs = fopen(execParam->datfile,"r");
		if (fd_obs!=NULL){
			fread(**execParam->d_obs,sizeof(float),
				execParam->nt*execParam->nx*execParam->ns,fd_obs);
			fclose(fd_obs);
		}else {
			printf("> Error! Couldn't open DOBS file at %s \n", execParam->datfile);
			exit(-1);
		}
	}

	/* read parameter models */
	execParam->vp = alloc2float(execParam->nz,execParam->nx);
	memset(*execParam->vp,0,execParam->nz*execParam->nx*sizeof(float));
	fvp = fopen(execParam->vpfile,"r");
	fread(execParam->vp[0],sizeof(float),execParam->nz*execParam->nx,fvp);
	fclose(fvp);

	/* vp size estended to vpe */
	execParam->vpe = alloc2float(execParam->nze,execParam->nxe);
	for(ix=0; ix<execParam->nx; ix++){
		for(iz=0; iz<execParam->nz; iz++){
			execParam->vpe[ix+execParam->nxb][iz+execParam->nzb] = execParam->vp[ix][iz]; 
		}
	}
	/* allocate vel2 for vpe^2 */
	execParam->vel2dt2 = alloc2float(execParam->nze,execParam->nxe);

	execParam->taperx = alloc1float(execParam->nxb);
	execParam->taperz = alloc1float(execParam->nzb);

	execParam->report.wrTransferCnt = 0;
	execParam->report.rdTransferCnt = 0;
	execParam->report.serialAVGTime = 0;
	execParam->report.kernelAVGTime = 0;
	execParam->report.wrAVGTime = 0;
	execParam->report.rdAVGTime = 0;
	execParam->report.deviceAVGTime = 0; // rd+wr+kernel
	execParam->report.speedup = 0;
	execParam->report.execTime = 0;
}

/**
* rtm_loadshots
* Allocates shot vectors
*/
RTMShot * rtm_loadshots(RTMExecParam execParam){
	int is;

	RTMShot * rtmShots = malloc(execParam.ns*sizeof(RTMShot));
	for (is=0; is< execParam.ns; is++){
		rtmShots[is].shotNumber = is;
		rtmShots[is].sx = execParam.sx[is];
		rtmShots[is].sz = execParam.sz;

		rtmShots[is].report.wrTransferCnt = 0;
		rtmShots[is].report.rdTransferCnt = 0;
		rtmShots[is].report.serialAVGTime = 0;
		rtmShots[is].report.kernelAVGTime = 0;
		rtmShots[is].report.wrAVGTime = 0;
		rtmShots[is].report.rdAVGTime = 0;
		rtmShots[is].report.deviceAVGTime = 0; // rd+wr+kernel
		rtmShots[is].report.speedup = 0;
		rtmShots[is].report.execTime = 0;

		int sw_half = execParam.swindow/2;
		rtmShots[is].start_x = rtmShots[is].sx-sw_half >= 0? rtmShots[is].sx-sw_half:0;
		rtmShots[is].end_x = rtmShots[is].sx+sw_half <= execParam.nxe?rtmShots[is].sx+sw_half:execParam.nxe;
		rtmShots[is].start_z = 0;
		rtmShots[is].end_z = execParam.nze;
		rtmShots[is].dobs = alloc2float(execParam.nt,execParam.nx);


		rtmShots[is].sfloat.PP 		= alloc2float(execParam.nze,execParam.nxe); 
		rtmShots[is].sfloat.P 		= alloc2float(execParam.nze,execParam.nxe);
		rtmShots[is].sfloat.PPR 	= alloc2float(execParam.nze,execParam.nxe);
		rtmShots[is].sfloat.PR 		= alloc2float(execParam.nze,execParam.nxe);
		rtmShots[is].sfloat.laplace = alloc2float(execParam.nze,execParam.nxe);
		rtmShots[is].sfloat.imloc 	= alloc2float(execParam.nz,execParam.nx);
		rtmShots[is].sfloat.upb 	= alloc3float(execParam.order/2,execParam.nxe,execParam.nt);
		rtmShots[is].sfloat.snaps 	= alloc3float(execParam.nze,execParam.nxe,2);

		memset(*rtmShots[is].sfloat.imloc,0,execParam.nz*execParam.nx*sizeof(float));
		memset(*rtmShots[is].sfloat.laplace,0,execParam.nze*execParam.nxe*sizeof(float));
	}
	return rtmShots;
}

/**
* rtm_freeshots
* Free shot vectors
*/
void rtm_freeshots(RTMExecParam execParam, RTMShot * rtmShots){
	int is;

	free2float(rtmShots[is].dobs);
	for (is=0; is< execParam.ns; is++){
			free2float(rtmShots[is].sfloat.PP);
			free2float(rtmShots[is].sfloat.P);
			free2float(rtmShots[is].sfloat.laplace);
			free2float(rtmShots[is].sfloat.PPR);
			free2float(rtmShots[is].sfloat.PR);
			free3float(rtmShots[is].sfloat.upb);
			free3float(rtmShots[is].sfloat.snaps);
	}
}

void rtm_freeExecParams(RTMExecParam * execParam){
	if(execParam->taperx!=NULL)free1float(execParam->taperx);
	if(execParam->taperz!=NULL)free1float(execParam->taperz);
	if(execParam->coefsx!=NULL)free1float(execParam->coefsx);
	if(execParam->coefsz!=NULL)free1float(execParam->coefsz);
	if(execParam->srce_wavelet)free1float(execParam->srce_wavelet);
	if(execParam->vp)free2float(execParam->vp);
	if(execParam->d_obs)free3float(execParam->d_obs);	
	if(execParam->vel_ext_flag) free3float(execParam->vel_ext_rnd);
}