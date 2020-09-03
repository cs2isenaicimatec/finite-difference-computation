#ifdef USE_MPI
#include <mpi.h>
#endif
#include <stdio.h>
#include "su.h"

#include "fd.h"
#include "ptsrc.h"
#include "taper.h"
#include "rtm.h"
#include "rtmcl.h"
#include "math.h"

char *sdoc[] = {	/* self documentation */
	" Seismic modeling using acoustic wave equation ",
	"				               ",
	NULL};

/* prototypes */
int main (int argc, char **argv){

	/* model file and data pointers */
	struct timeval st, et;
	char imgFileName[100];
	FILE *fimg_fl = NULL, * f_mod_dobs=NULL;
	RTMExecParam execParam;	
	RTMShot * rtmShots;
	int iz, ix, is;
    int numberOfProcesses, processId;

    gettimeofday(&st, NULL);
	/* propagation variables */
	/* initialization admiting self documentation */
	initargs(argc, argv);
	requestdoc(1);
    // Get the rank of the process
#ifdef USE_MPI 
	// Initialize the MPI environment 
	MPI_Init(NULL, NULL);   
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
#else
    numberOfProcesses = 1;
    processId = 0;
#endif
	/*initargs(argc, argv);*/
	execParam.numberOfProcesses = numberOfProcesses;
	execParam.processId = processId;
	rtm_loadparams(&execParam); // load parameters from input.dat file
	rtm_initparams(&execParam); // init vectors
	
	/* initialize wave propagation */
	/** allocate img matrices.*/
	if (execParam.processId==0){ // only the master generates img
		execParam.img_float = alloc2float(execParam.nz,execParam.nx);
		memset(*execParam.img_float,0,execParam.nz*execParam.nx*sizeof(float));
	}
	rtmShots = rtm_loadshots(execParam);
	taper_init(&execParam);
	fd_init(&execParam);


	/*********************************************/
	if (execParam.processId==0 /*&& execParam.numberOfProcesses>1*/){ 
		// this MUST be done at least once for the process
		// that will perform the img stacking. 
		// Must be done after initiaziation
		rtm_extvel_hyb(&execParam, execParam.processId);
	}
	/*********************************************/

#ifdef OPENCLRTM
	rtmcl_init(&execParam);
#endif	
	
	/*********************************************/
	// run rtm modeling or migration processes	
	rtm_process(&execParam, rtmShots);
	/*********************************************/
	
	/*********************************************/
	// image stacking (process 0 only)
	if (execParam.processId==0){
		int window = execParam.swindow > execParam.nz?0:execParam.swindow;
		if (execParam.modeling==0){
			printf ("> Stacking images.... \n"); fflush(stdout);
			rtm_stackimg(&execParam, rtmShots);
			sprintf(imgFileName, "%s/imgfloat_s%d_w%d.bin", execParam.outputpath, 
				execParam.ns, window);
			
			printf ("> Saving img file at %s \n", imgFileName); fflush(stdout);
			fimg_fl = fopen(imgFileName,"w");
			if (fimg_fl==NULL){
				printf("> Error! Could not open output file at %s ", imgFileName);
			}else{
				fwrite(*execParam.img_float,sizeof(float),execParam.nz*execParam.nx,fimg_fl);
				fclose(fimg_fl);
				free2float(execParam.img_float);
				for (is = 0; is<execParam.ns; is++){
					execParam.report.wrTransferCnt 	+=	rtmShots[is].report.wrTransferCnt;
					execParam.report.rdTransferCnt 	+=	rtmShots[is].report.rdTransferCnt;
					execParam.report.serialAVGTime 	+=	rtmShots[is].report.serialAVGTime;
					execParam.report.kernelAVGTime 	+=	rtmShots[is].report.kernelAVGTime;
					execParam.report.wrAVGTime 		+=	rtmShots[is].report.wrAVGTime;
					execParam.report.rdAVGTime 		+=	rtmShots[is].report.rdAVGTime;
					execParam.report.deviceAVGTime 	+= 	rtmShots[is].report.deviceAVGTime;

					execParam.report.fwSerialAVGTime +=	rtmShots[is].report.fwSerialAVGTime;
					execParam.report.fwDeviceAVGTime +=	rtmShots[is].report.fwDeviceAVGTime;
					execParam.report.bwSerialAVGTime +=	rtmShots[is].report.bwSerialAVGTime;
					execParam.report.bwDeviceAVGTime += rtmShots[is].report.bwDeviceAVGTime;
				}
			}
		}else{
			printf ("> Saving DOBS file.... \n"); 
			sprintf(imgFileName, "%s/../dobs/dobs_%dshots.bin", execParam.outputpath, 
					execParam.ns);
			f_mod_dobs = fopen(imgFileName,"w");
			for(is=0; is<execParam.ns; is++){ 
				fwrite(rtmShots[is].dobs[0],sizeof(float),execParam.nx*execParam.nt,f_mod_dobs);
			}
			fclose(f_mod_dobs);
		}
	}
	/*********************************************/
    /* release memory */
    fd_destroy();
	taper_destroy();
	rtm_freeshots(execParam, rtmShots);
	free1int(execParam.sx);

#ifdef USE_MPI	
	MPI_Finalize();
#endif
	gettimeofday(&et, NULL);
	execParam.report.execTime = elapsedTimeMs(st, et);
	if (execParam.processId==0){
		print_time_report(execParam);
	}
	rtm_freeExecParams(&execParam);
	return 0;
}

void print_time_report(RTMExecParam execParam){

	printf("> ================================================\n");
	printf(">       Exec Time Report (NX = %d NZ= %d)      <\n", execParam.nxe, execParam.nze);
	printf("> General:\n");
	printf("> Serial       = %10.1f (ms)\n", execParam.report.serialAVGTime);
	printf("> Device       = %10.1f (ms)\n", execParam.report.deviceAVGTime);
	printf("> Environment  = %10.1f (ms)\n", 
		execParam.report.execTime-(execParam.report.serialAVGTime+execParam.report.fwDeviceAVGTime));
	printf("> Total        = %10.1f (ms)\n", execParam.report.execTime);
	printf("> Speedup      = %10.1f\n", 
		execParam.report.serialAVGTime/execParam.report.deviceAVGTime);
	printf("> \n");
	printf("> WR TransfCnt = %10d \n", execParam.report.wrTransferCnt);
	printf("> Write        = %10.1f (%.2f%%)(ms)\n", execParam.report.wrAVGTime, 
		(100.0*execParam.report.wrAVGTime/execParam.report.deviceAVGTime));
	printf("> RD TransfCnt = %10d \n", execParam.report.rdTransferCnt);
	printf("> Read         = %10.1f (%.2f%%)(ms)\n", execParam.report.rdAVGTime,
		(100.0*execParam.report.rdAVGTime/execParam.report.deviceAVGTime));

	printf("> Kernel (Only)= %10.1f (%.2f%%)(ms)\n", execParam.report.kernelAVGTime,
		(100.0*execParam.report.kernelAVGTime/execParam.report.deviceAVGTime));

	printf("> ================================================ \n");
	printf("> Forward and Backward Propagations only:\n");
	printf("> Serial Fwrd  = %10.1f (ms)\n", execParam.report.fwSerialAVGTime);
	printf("> Device Fwrd  = %10.1f (ms)\n", execParam.report.fwDeviceAVGTime);
	printf("> Fwrd Speedup = %10.1f\n", 
		execParam.report.fwSerialAVGTime/execParam.report.fwDeviceAVGTime);
	printf("> \n");
	printf("> Serial Bwrd  = %10.1f (ms)\n", execParam.report.bwSerialAVGTime);
	printf("> Device Bwrd  = %10.1f (ms)\n", execParam.report.bwDeviceAVGTime);
	printf("> Bwrd Speedup = %10.1f (ms)\n", 
		execParam.report.bwSerialAVGTime/execParam.report.bwDeviceAVGTime);
	printf("> ================================================ \n\n");

}
