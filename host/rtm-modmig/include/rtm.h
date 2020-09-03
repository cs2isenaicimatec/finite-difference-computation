#ifndef RTM_H
#define RTM_H
#include <time.h>

#define RTMEXEC_FWRD    0
#define RTMEXEC_BWRD    1

#define LAPLACIAN_FILTER_ORDER 6
///////////////////////////////////////////////////////////////////////
// opencl related data types
#ifdef OPENCLRTM
#include <CL/cl.h> 
typedef struct RMTCLForwardKernel{
	char 		* kname;
	int 		srcx;
	int 		srcz;
	long srcBufferSize;
	long v2dt2BufferSize;
	long imgBufferSize;
	long tpxBufferSize;
	long tpzBufferSize;
	long upbBufferSize;
	long coefBufferSize;
	cl_mem 		bfV2DT2;
	cl_mem 		bfCoefx;
	cl_mem 		bfCoefz;
	cl_mem 		bfP;
	cl_mem 		bfPP;
	cl_mem 		bfTaperX;
	cl_mem 		bfTaperZ;
	cl_mem 		bfSrcwavelet;
	cl_mem 		bfUPB;
	cl_kernel 	kernel;
}RMTCLForwardKernel;

typedef struct RMTCLBackwardKernel{
	char 		* kname;
	int 		dobsz;
	long coefBufferSize;
	long v2dt2BufferSize;
	long tpxBufferSize;
	long tpzBufferSize;
	long dobsBufferSize;
	long upbBufferSize;
	long snapsBufferSize;
	long imgBufferSize;
	long imlocBufferSize;
	cl_mem 		bfCoefx;
	cl_mem 		bfCoefz;
	cl_mem 		bfV2DT2;
	cl_mem 		bfTaperX;
	cl_mem 		bfTaperZ;
	cl_mem 		bfDobs;
	cl_mem 		bfUPB;
	cl_mem 		bfSnaps0;
	cl_mem 		bfSnaps1;
	cl_mem 		bfPR;
	cl_mem 		bfPPR;
	cl_mem 		bfP;
	cl_mem 		bfPP;
	cl_mem 		bfImloc;
	cl_kernel 	kernel;
}RMTCLBackwardKernel;

typedef struct RTMCLEnv{
	RMTCLBackwardKernel kbackward;
	RMTCLForwardKernel 	kforward;
	cl_context 			context; 
	cl_program 			program;
	cl_command_queue 	cmdQueue;
	char * 				kernelPath;
	int 				platform_id;
	size_t 				singleWork[2];   
	size_t 				localWorkSize[2]; 
	size_t 				globalWorkSize[2]; 
	size_t 				globalWorkSizeUpb[2];    
	size_t 				globalWorkSizeTaper[2]; 
}RTMCLEnv;
#endif
///////////////////////////////////////////////////////////////////////

typedef struct RTMTimeReport{
	int wrTransferCnt;
	int rdTransferCnt;
	float fwSerialAVGTime;
	float fwDeviceAVGTime;
	float bwSerialAVGTime;
	float bwDeviceAVGTime;
	float serialAVGTime;
	float kernelAVGTime;
	float wrAVGTime;
	float rdAVGTime;
	float deviceAVGTime; // rd+wr+kernel
	float speedup;
	float execTime;
}RTMTimeReport;

typedef struct RTMShotParamFloat{
	float ** 	PP;
	float ** 	P;
	float ** 	PPR;
	float ** 	PR;
	float ** 	laplace;
	float **	imloc;
	float **  	swap;
	float ***	upb;
	float ***	snaps;
}RTMShotParamFloat;

typedef struct RTMShot{
	int 		shotNumber;
	int 		sx;
	int 		sz;
	int 		start_x;
	int 		start_z;
	int 		end_x;
	int 		end_z;
	float 		** dobs; // modeling 
	RTMShotParamFloat sfloat;
	RTMTimeReport report;
}RTMShot;

typedef struct RTMExecParam{
	int 		processId;
	int 		numberOfProcesses;
	int 		modeling;
	int 		fixrun;
	int 		floatrun;
	int 		nbitsfrac;
	int 		nbitsint;
	int 		nx;
	int 		nz;
	int 		nxe;
	int 		nze;
	int 		nxb;
	int 		nzb;
	int 		nt;
	int 		order;
	int 		iss;
	int 		ns;
	int 		fsx;
	int 		ds;
	int 		sz;
	int 		gz;
	int 		swindow;
	float 		dz;
	float 		dx;
	float 		dt;
	float 		fpeak;
	float 		fac;
	float 		rnd;
	float ** 	vel2dt2;
	float * 	taperx;
	float * 	taperz;
	float * 	coefsx;
	float * 	coefsz;
	float *		srce_wavelet; // source signature
	int   *		sx; // shot positions
	float *** 	d_obs;
	float *** 	vel_ext_rnd;
	char 		vel_ext_flag;
	char  * 	outputpath;
	char  *  	vpfile;
	char  *  	datfile;
	char  *  	vel_ext_file;
	char  *		tmpdir;
	float **	vp ;
	float ** 	vpe;
	float **	img_float;
	float **	img_float_from_fix;
	int   **	img_fix;
	char  *		kernelPath; // for opencl
	int current_it;
#ifdef OPENCLRTM
	RTMCLEnv	clEnv;
#endif	
	int 		exec_state;
	RTMTimeReport report;
}RTMExecParam;

void rtm_swapptr(void *** ptrA, void *** ptrB);
void rtm_savesnaps_float(float ***	snaps, float ** PP, float ** P, int nxe, int nze);
void rtm_forward(RTMShot * shot, RTMExecParam * execParams);
void rtm_backward(RTMShot * shot, RTMExecParam * execParams);
void rtm_extvel(RTMExecParam * execParam);
void rtm_extvel_hyb(RTMExecParam * exec, int is);
void rtm_stackimg(RTMExecParam * execParam, RTMShot * rtmShots);
void rtm_loadparams(RTMExecParam * exec);
void rtm_initparams(RTMExecParam * exec);
void rtm_freeshots(RTMExecParam execParam, RTMShot * rtmShots);
void rtm_freeExecParams(RTMExecParam * execParam);
void rtm_migration(RTMExecParam * execParam, RTMShot * rtmShot);
void rtm_modeling(RTMExecParam * execParam, RTMShot * rtmShot);
void rtm_process(RTMExecParam * execParam, RTMShot * rtmShots);
void rtm_mig_gather(RTMExecParam * execParam, RTMShot * rtmShots);
void rtm_mod_gather(RTMExecParam * execParam, RTMShot * rtmShots);
RTMShot * rtm_loadshots(RTMExecParam execParam);

float elapsedTimeUs(struct timeval st, struct timeval et);
float elapsedTimeMs(struct timeval st, struct timeval et);
float elapsedTimeS(struct timeval st, struct timeval et);



#define myprintf(A) 	printf(A);fflush(stdout);

#endif
