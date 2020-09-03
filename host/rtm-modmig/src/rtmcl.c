#ifdef OPENCLRTM
#ifdef USE_MPI
#include <mpi.h>
#endif

#include <stdio.h>
#include <CL/cl.h>
#include <float.h>
#include <time.h>
#include "su.h"
#include "fd.h"
#include "ptsrc.h"
#include "taper.h"
#include "math.h"
#include "rtmcl.h"

void rtmcl_enqueue_fkernel(RTMCLEnv * clEnv){
	cl_int status;
#if defined(USE_GPU) || defined(USE_CPU)
    status = clEnqueueNDRangeKernel(clEnv->cmdQueue, 
        clEnv->kforward.kernel, 2, NULL, 
    	clEnv->globalWorkSize, clEnv->localWorkSize, 
        0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to run the kernel on NDRange %d\n", status);
        exit(1);
    } 
#else
    cl_event kernel_event; 
    int error = clEnqueueTask(clEnv->cmdQueue, 
        clEnv->kforward.kernel, 0, NULL, kernel_event); 
    if(error)
        printf("> Error Code: %d \n", error); 
#endif
    clFinish(clEnv->cmdQueue);
}

void rtmcl_enqueue_bkernel(RTMCLEnv * clEnv){
    cl_int status;
#if defined(USE_GPU) || defined(USE_CPU)
    status = clEnqueueNDRangeKernel(clEnv->cmdQueue, 
        clEnv->kbackward.kernel, 2, NULL, 
        clEnv->globalWorkSize, clEnv->localWorkSize, 
        0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to run the kernel on NDRange %d\n", 
            status);
        exit(1);
    } 
#else
    cl_event kernel_event; 
    int error = clEnqueueTask(clEnv->cmdQueue, 
        clEnv->kbackward.kernel, 0, NULL, kernel_event); 
    if(error)
        printf("> Error Code: %d \n", error);
#endif
    clFinish(clEnv->cmdQueue);
}

void rtmcl_read_forward_output(RTMShot * shot, RTMExecParam * execParam){
	cl_int status;

	status = clEnqueueReadBuffer(execParam->clEnv.cmdQueue, execParam->clEnv.kforward.bfUPB, CL_TRUE, 0, 
		execParam->clEnv.kforward.upbBufferSize, shot->sfloat.upb[0][0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to read the UPB buffer status:%d \n", status);
        exit(1);
    }

    status = clEnqueueReadBuffer(execParam->clEnv.cmdQueue, execParam->clEnv.kforward.bfP, CL_TRUE, 0, 
		execParam->clEnv.kforward.imgBufferSize, shot->sfloat.P[0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to read the P buffer status:%d \n", status);
        exit(1);
    }

    status = clEnqueueReadBuffer(execParam->clEnv.cmdQueue, execParam->clEnv.kforward.bfPP, CL_TRUE, 0, 
		execParam->clEnv.kforward.imgBufferSize, shot->sfloat.PP[0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to read the PP buffer status:%d \n", status);
        exit(1);
    }

    int ix, iz;
	for(iz=0; iz<execParam->nze; iz++){
		for(ix=0; ix<execParam->nxe; ix++){
			shot->sfloat.snaps[1][ix][iz] = shot->sfloat.PP[ix][iz];
			shot->sfloat.snaps[0][ix][iz] = shot->sfloat.P[ix][iz];
		}
	}
}

void rtmcl_write_forward_params(RTMShot * shot, RTMExecParam * execParam){
	cl_int status; 

	/////////////////////////////////////////////////////////////////////////
    // write buffers
    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
    	execParam->clEnv.kforward.bfP, CL_TRUE, 0, execParam->clEnv.kforward.imgBufferSize, 
    	shot->sfloat.P[0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy P to buffer\n");
        exit(1);
    }

    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
        execParam->clEnv.kforward.bfCoefx, CL_TRUE, 0, execParam->clEnv.kforward.coefBufferSize, 
        execParam->coefsx, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy COEFX to buffer\n");
        exit(1);
    }

    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
        execParam->clEnv.kforward.bfCoefz, CL_TRUE, 0, execParam->clEnv.kforward.coefBufferSize, 
        execParam->coefsz, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy COEFZ to buffer\n");
        exit(1);
    }

    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
    	execParam->clEnv.kforward.bfPP, CL_TRUE, 0, execParam->clEnv.kforward.imgBufferSize, 
    	shot->sfloat.PP[0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy PP to buffer\n");
        exit(1);
    }

    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
    	execParam->clEnv.kforward.bfV2DT2, CL_TRUE, 0, execParam->clEnv.kforward.v2dt2BufferSize, 
    	execParam->vel2dt2[0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy V2DT2 to buffer\n");
        exit(1);
    }

    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
    	execParam->clEnv.kforward.bfTaperX, CL_TRUE, 0, execParam->clEnv.kforward.tpxBufferSize, 
    	execParam->taperx, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy TAPER_X to buffer\n");
        exit(1);
    }


    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
    	execParam->clEnv.kforward.bfTaperZ, CL_TRUE, 0, execParam->clEnv.kforward.tpzBufferSize, 
    	execParam->taperz, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy TAPER_Z to buffer\n");
        exit(1);
    }

    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
    	execParam->clEnv.kforward.bfSrcwavelet, CL_TRUE, 0, execParam->clEnv.kforward.srcBufferSize, 
    	execParam->srce_wavelet, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy SRCWAVELET to buffer\n");
        exit(1);
    }

    /////////////////////////////////////////////////////////////////////////
    // set args
    execParam->clEnv.kforward.srcx = shot->sx;
	execParam->clEnv.kforward.srcz = shot->sz;
	status = clSetKernelArg(execParam->clEnv.kforward.kernel, 0, sizeof(int), &execParam->clEnv.kforward.srcx);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 1st kforward img argument %d\n",status);
        exit(1);
    }


    status = clSetKernelArg(execParam->clEnv.kforward.kernel, 1, sizeof(int), &execParam->clEnv.kforward.srcz);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 2nd kforward img argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(execParam->clEnv.kforward.kernel, 2, sizeof(cl_mem), &execParam->clEnv.kforward.bfCoefx);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 3rd kforward argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(execParam->clEnv.kforward.kernel, 3, sizeof(cl_mem), &execParam->clEnv.kforward.bfCoefz);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 4th kforward argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(execParam->clEnv.kforward.kernel, 4, sizeof(cl_mem), &execParam->clEnv.kforward.bfV2DT2);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 5th kforward argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(execParam->clEnv.kforward.kernel, 5, sizeof(cl_mem), &execParam->clEnv.kforward.bfTaperX);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 6th kforward argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(execParam->clEnv.kforward.kernel, 6, sizeof(cl_mem), &execParam->clEnv.kforward.bfTaperZ);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 7th kforward argument %d\n",status);
        exit(1);
    }
    status = clSetKernelArg(execParam->clEnv.kforward.kernel, 7, sizeof(cl_mem), &execParam->clEnv.kforward.bfSrcwavelet);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 8th kforward argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(execParam->clEnv.kforward.kernel, 8, sizeof(cl_mem), &execParam->clEnv.kforward.bfP);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 9th kforward argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(execParam->clEnv.kforward.kernel, 9, sizeof(cl_mem), &execParam->clEnv.kforward.bfPP);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 10th kforward argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(execParam->clEnv.kforward.kernel, 10, sizeof(cl_mem), &execParam->clEnv.kforward.bfUPB);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 11th kforward argument %d\n",status);
        exit(1);
    }
}

void rtmcl_load_forward_kernel(RTMExecParam * execParam){

	cl_int status; 
	execParam->clEnv.kforward.srcBufferSize    = execParam->nt*sizeof(float);
	execParam->clEnv.kforward.v2dt2BufferSize  = execParam->nxe*execParam->nze*sizeof(float);
	execParam->clEnv.kforward.imgBufferSize    = execParam->nxe*execParam->nze*sizeof(float);
	execParam->clEnv.kforward.tpxBufferSize    = execParam->nxb*sizeof(float);
	execParam->clEnv.kforward.tpzBufferSize    = execParam->nzb*sizeof(float);
	execParam->clEnv.kforward.upbBufferSize    = execParam->nt*execParam->nxe*(execParam->order/2)*sizeof(float);
    execParam->clEnv.kforward.coefBufferSize   = (execParam->order+1)*sizeof(float);

	execParam->clEnv.kforward.srcx = 0;
	execParam->clEnv.kforward.srcz = 0;

	///////////////////////////////////////////////////////////////////////////////////
	execParam->clEnv.kforward.bfP = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_READ_WRITE, 
            execParam->clEnv.kforward.imgBufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for P\n");
        exit(1);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    execParam->clEnv.kforward.bfPP = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_READ_WRITE, 
            execParam->clEnv.kforward.imgBufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for PP\n");
        exit(1);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    execParam->clEnv.kforward.bfCoefx = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_READ_ONLY, 
            execParam->clEnv.kforward.coefBufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for COEFX\n");
        exit(1);
    }

    execParam->clEnv.kforward.bfCoefz = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_READ_ONLY, 
            execParam->clEnv.kforward.coefBufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for COEFZ\n");
        exit(1);
    }

	///////////////////////////////////////////////////////////////////////////////////
    execParam->clEnv.kforward.bfTaperX = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_READ_ONLY, 
            execParam->clEnv.kforward.tpxBufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for TAPER_X\n");
        exit(1);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    execParam->clEnv.kforward.bfTaperZ = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_READ_ONLY, 
            execParam->clEnv.kforward.tpzBufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for TAPER_X\n");
        exit(1);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    execParam->clEnv.kforward.bfV2DT2 = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_READ_ONLY, 
            execParam->clEnv.kforward.v2dt2BufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for V2DT2\n");
        exit(1);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    execParam->clEnv.kforward.bfSrcwavelet = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_READ_ONLY, 
            execParam->clEnv.kforward.srcBufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for SRCWAVELET\n");
        exit(1);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    execParam->clEnv.kforward.bfUPB = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_WRITE_ONLY, 
            execParam->clEnv.kforward.upbBufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for UPBOARD\n");
        exit(1);
    }


    execParam->clEnv.kforward.kname = RTMCL_FORWARD_NAME;
}

void rtmcl_unload_forward_kernel(RTMCLEnv * clEnv){
   
    clReleaseKernel         (clEnv->kforward.kernel);
    clReleaseMemObject      (clEnv->kforward.bfV2DT2);
    clReleaseMemObject      (clEnv->kforward.bfP);
    clReleaseMemObject      (clEnv->kforward.bfPP); 
    clReleaseMemObject      (clEnv->kforward.bfTaperX); 
    clReleaseMemObject      (clEnv->kforward.bfTaperZ); 
    clReleaseMemObject      (clEnv->kforward.bfSrcwavelet);
    clReleaseMemObject      (clEnv->kforward.bfUPB); 
    return;
}

void rtmcl_write_backward_params(RTMShot * shot, 
    RTMExecParam * execParam){
    cl_int status; 

    /////////////////////////////////////////////////////////////////////////
    // write buffers
    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
        execParam->clEnv.kbackward.bfCoefx, CL_TRUE, 0, 
        execParam->clEnv.kbackward.coefBufferSize, 
        execParam->coefsx, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy COEFX to buffer\n");
        exit(1);
    }

    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
        execParam->clEnv.kbackward.bfCoefz, CL_TRUE, 0, 
        execParam->clEnv.kbackward.coefBufferSize, 
        execParam->coefsz, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy COEFZ to buffer\n");
        exit(1);
    }

    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
        execParam->clEnv.kbackward.bfV2DT2, CL_TRUE, 0, 
        execParam->clEnv.kbackward.v2dt2BufferSize, 
        execParam->vel2dt2[0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy V2DT2 to buffer\n");
        exit(1);
    }

    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
        execParam->clEnv.kbackward.bfTaperX, CL_TRUE, 0, 
        execParam->clEnv.kbackward.tpxBufferSize, 
        execParam->taperx, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy TAPER_X to buffer\n");
        exit(1);
    }


    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
        execParam->clEnv.kbackward.bfTaperZ, CL_TRUE, 0, 
        execParam->clEnv.kbackward.tpzBufferSize, 
        execParam->taperz, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy TAPER_Z to buffer\n");
        exit(1);
    }

    int shotId = shot->shotNumber;
    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
        execParam->clEnv.kbackward.bfDobs, CL_TRUE, 0, 
        execParam->clEnv.kbackward.dobsBufferSize, 
        execParam->d_obs[shotId][0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy DOBS to buffer\n");
        exit(1);
    }

    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
        execParam->clEnv.kbackward.bfUPB, CL_TRUE, 0, 
        execParam->clEnv.kbackward.upbBufferSize, 
        shot->sfloat.upb[0][0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy DOBS to buffer\n");
        exit(1);
    }

    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
        execParam->clEnv.kbackward.bfSnaps0, CL_TRUE, 0, 
        execParam->clEnv.kbackward.snapsBufferSize, 
        shot->sfloat.snaps[0][0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy SNAPS0 to buffer\n");
        exit(1);
    }

    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
        execParam->clEnv.kbackward.bfSnaps1, CL_TRUE, 0, 
        execParam->clEnv.kbackward.snapsBufferSize, 
        shot->sfloat.snaps[1][0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy SNAPS1 to buffer\n");
        exit(1);
    }

    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
        execParam->clEnv.kbackward.bfPR, CL_TRUE, 0, 
        execParam->clEnv.kbackward.imgBufferSize, 
        shot->sfloat.PR[0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy PR to buffer\n");
        exit(1);
    }

    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
        execParam->clEnv.kbackward.bfPPR, CL_TRUE, 0, 
        execParam->clEnv.kbackward.imgBufferSize, 
        shot->sfloat.PPR[0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy PPR to buffer\n");
        exit(1);
    }

    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
        execParam->clEnv.kbackward.bfP, CL_TRUE, 0, 
        execParam->clEnv.kbackward.imgBufferSize, 
        shot->sfloat.P[0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy P to buffer\n");
        exit(1);
    }

    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
        execParam->clEnv.kbackward.bfPP, CL_TRUE, 0, 
        execParam->clEnv.kbackward.imgBufferSize, 
        shot->sfloat.PP[0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy PP to buffer\n");
        exit(1);
    }

    status = clEnqueueWriteBuffer(execParam->clEnv.cmdQueue, 
        execParam->clEnv.kbackward.bfImloc, CL_TRUE, 0, 
        execParam->clEnv.kbackward.imlocBufferSize, 
        shot->sfloat.imloc[0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy PP to buffer\n");
        exit(1);
    }

    /////////////////////////////////////////////////////////////////////////
    // set args
    execParam->clEnv.kbackward.dobsz = execParam->gz;
    status = clSetKernelArg(execParam->clEnv.kbackward.kernel, 0, 
        sizeof(int), &execParam->clEnv.kbackward.dobsz);
    if (status != CL_SUCCESS) {
        printf ("Unable to set dobsz kbackward argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(execParam->clEnv.kbackward.kernel, 1, 
        sizeof(cl_mem), &execParam->clEnv.kbackward.bfCoefx);
    if (status != CL_SUCCESS) {
        printf ("Unable to set coefsx kbackward argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(execParam->clEnv.kbackward.kernel, 2, 
        sizeof(cl_mem), &execParam->clEnv.kbackward.bfCoefz);
    if (status != CL_SUCCESS) {
        printf ("Unable to set coefsz kbackward argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(execParam->clEnv.kbackward.kernel, 3, 
        sizeof(cl_mem), &execParam->clEnv.kbackward.bfV2DT2);
    if (status != CL_SUCCESS) {
        printf ("Unable to set v2dt2 kbackward argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(execParam->clEnv.kbackward.kernel, 4, 
        sizeof(cl_mem), &execParam->clEnv.kbackward.bfTaperX);
    if (status != CL_SUCCESS) {
        printf ("Unable to set taperx kbackward argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(execParam->clEnv.kbackward.kernel, 5, 
        sizeof(cl_mem), &execParam->clEnv.kbackward.bfTaperZ);
    if (status != CL_SUCCESS) {
        printf ("Unable to set taperz kbackward argument %d\n",status);
        exit(1);
    }
    status = clSetKernelArg(execParam->clEnv.kbackward.kernel, 6, 
        sizeof(cl_mem), &execParam->clEnv.kbackward.bfDobs);
    if (status != CL_SUCCESS) {
        printf ("Unable to set dobs kbackward argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(execParam->clEnv.kbackward.kernel, 7, 
        sizeof(cl_mem), &execParam->clEnv.kbackward.bfUPB);
    if (status != CL_SUCCESS) {
        printf ("Unable to set UPB kbackward argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(execParam->clEnv.kbackward.kernel, 8, 
        sizeof(cl_mem), &execParam->clEnv.kbackward.bfSnaps0);
    if (status != CL_SUCCESS) {
        printf ("Unable to set snaps0 kbackward argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(execParam->clEnv.kbackward.kernel, 9, 
        sizeof(cl_mem), &execParam->clEnv.kbackward.bfSnaps1);
    if (status != CL_SUCCESS) {
        printf ("Unable to set snaps1 kbackward argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(execParam->clEnv.kbackward.kernel, 10, 
        sizeof(cl_mem), &execParam->clEnv.kbackward.bfPR);
    if (status != CL_SUCCESS) {
        printf ("Unable to set PR kbackward argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(execParam->clEnv.kbackward.kernel, 11, 
        sizeof(cl_mem), &execParam->clEnv.kbackward.bfPPR);
    if (status != CL_SUCCESS) {
        printf ("Unable to set PPR kbackward argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(execParam->clEnv.kbackward.kernel, 12, 
        sizeof(cl_mem), &execParam->clEnv.kbackward.bfP);
    if (status != CL_SUCCESS) {
        printf ("Unable to set P kbackward argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(execParam->clEnv.kbackward.kernel, 13, 
        sizeof(cl_mem), &execParam->clEnv.kbackward.bfPP);
    if (status != CL_SUCCESS) {
        printf ("Unable to set PP kbackward argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(execParam->clEnv.kbackward.kernel, 14, 
        sizeof(cl_mem), &execParam->clEnv.kbackward.bfImloc);
    if (status != CL_SUCCESS) {
        printf ("Unable to set imloc kbackward argument %d\n",status);
        exit(1);
    }
}

void rtmcl_load_backward_kernel(RTMExecParam * execParam){

    cl_int status; 
    
    execParam->clEnv.kbackward.coefBufferSize   = (execParam->order+1)*sizeof(float);
    execParam->clEnv.kbackward.v2dt2BufferSize  = execParam->nxe*execParam->nze*sizeof(float);
    execParam->clEnv.kbackward.tpxBufferSize    = execParam->nxb*sizeof(float);
    execParam->clEnv.kbackward.tpzBufferSize    = execParam->nzb*sizeof(float);
    execParam->clEnv.kbackward.dobsBufferSize   = execParam->nt*execParam->nx*sizeof(float);
    execParam->clEnv.kbackward.upbBufferSize    = execParam->nt*execParam->nxe*
                                                    (execParam->order/2)*sizeof(float);
    execParam->clEnv.kbackward.snapsBufferSize  = execParam->nxe*execParam->nze*sizeof(float);
    execParam->clEnv.kbackward.imgBufferSize    = execParam->nxe*execParam->nze*sizeof(float);
    execParam->clEnv.kbackward.imlocBufferSize  = execParam->nx*execParam->nz*sizeof(float);

    execParam->clEnv.kbackward.dobsz = execParam->gz;

    ///////////////////////////////////////////////////////////////////////////////////
    execParam->clEnv.kbackward.bfCoefx = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_READ_ONLY, 
            execParam->clEnv.kbackward.coefBufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for COEFX\n");
        exit(1);
    }

    execParam->clEnv.kbackward.bfCoefz = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_READ_ONLY, 
            execParam->clEnv.kbackward.coefBufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for COEFZ\n");
        exit(1);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    execParam->clEnv.kbackward.bfV2DT2 = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_READ_ONLY, 
            execParam->clEnv.kbackward.v2dt2BufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for V2DT2\n");
        exit(1);
    }
    ///////////////////////////////////////////////////////////////////////////////////
    execParam->clEnv.kbackward.bfTaperX = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_READ_ONLY, 
            execParam->clEnv.kbackward.tpxBufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for TAPER_X\n");
        exit(1);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    execParam->clEnv.kbackward.bfTaperZ = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_READ_ONLY, 
            execParam->clEnv.kbackward.tpzBufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for TAPER_X\n");
        exit(1);
    }

    execParam->clEnv.kbackward.bfDobs = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_READ_WRITE, 
            execParam->clEnv.kbackward.dobsBufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for DOBS\n");
        exit(1);
    }
    ///////////////////////////////////////////////////////////////////////////////////
    execParam->clEnv.kbackward.bfUPB = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_WRITE_ONLY, 
            execParam->clEnv.kbackward.upbBufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for UPB\n");
        exit(1);
    }

    execParam->clEnv.kbackward.bfSnaps0 = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_READ_ONLY, 
            execParam->clEnv.kbackward.snapsBufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for SNAPS0\n");
        exit(1);
    }
    execParam->clEnv.kbackward.bfSnaps1 = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_READ_ONLY, 
            execParam->clEnv.kbackward.snapsBufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for SNAPS1\n");
        exit(1);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    execParam->clEnv.kbackward.bfPR = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_READ_WRITE, 
            execParam->clEnv.kbackward.imgBufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for PR\n");
        exit(1);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    execParam->clEnv.kbackward.bfPPR = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_READ_WRITE, 
            execParam->clEnv.kbackward.imgBufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for PPR\n");
        exit(1);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    execParam->clEnv.kbackward.bfP = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_READ_WRITE, 
            execParam->clEnv.kbackward.imgBufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for P\n");
        exit(1);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    execParam->clEnv.kbackward.bfPP = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_READ_WRITE, 
            execParam->clEnv.kbackward.imgBufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for PP\n");
        exit(1);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    execParam->clEnv.kbackward.bfImloc = 
        clCreateBuffer(execParam->clEnv.context, CL_MEM_READ_WRITE, 
            execParam->clEnv.kbackward.imlocBufferSize, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for IMGLOC\n");
        exit(1);
    }

    execParam->clEnv.kbackward.kname = RTMCL_BACKWARD_NAME;
}

void rtmcl_unload_backward_kernel(RTMCLEnv * clEnv){
   
    clReleaseKernel         (clEnv->kbackward.kernel);
    clReleaseMemObject      (clEnv->kbackward.bfV2DT2);
    clReleaseMemObject      (clEnv->kbackward.bfP);
    clReleaseMemObject      (clEnv->kbackward.bfPP); 
    clReleaseMemObject      (clEnv->kbackward.bfPR);
    clReleaseMemObject      (clEnv->kbackward.bfPPR); 
    clReleaseMemObject      (clEnv->kbackward.bfTaperX); 
    clReleaseMemObject      (clEnv->kbackward.bfTaperZ); 
    clReleaseMemObject      (clEnv->kbackward.bfDobs);
    clReleaseMemObject      (clEnv->kbackward.bfUPB); 
    clReleaseMemObject      (clEnv->kbackward.bfSnaps0); 
    clReleaseMemObject      (clEnv->kbackward.bfSnaps1); 
    clReleaseMemObject      (clEnv->kbackward.bfImloc); 
    return;
}

void rtmcl_read_backward_output(RTMShot * shot, 
    RTMExecParam * execParam){
    cl_int status;

    status = clEnqueueReadBuffer(execParam->clEnv.cmdQueue, 
        execParam->clEnv.kbackward.bfImloc, CL_TRUE, 0, 
        execParam->clEnv.kbackward.imlocBufferSize, 
        shot->sfloat.imloc[0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to read the UPB buffer status:%d \n", status);
        exit(1);
    }

}

void rtmcl_init(RTMExecParam * execParam){
	//-----------------------------------------------------
    // STEP 1: Read Kernel File
    //----------------------------------------------------- 
    cl_int status; 
    execParam->clEnv.kernelPath = execParam->kernelPath;
    size_t source_size;
    char *source_str; 
    source_str = clmiscReadFileStr(execParam->clEnv.kernelPath, &source_size);       
    //-----------------------------------------------------
    // STEP 2: Initialize Plataforms and Devices
    //----------------------------------------------------- 
    
    cl_uint numDevices;  
    cl_platform_id  *   platforms   = NULL;
    cl_device_id    *   devices     = 
        clmiscGetPlatformAndDevice(platforms, execParam->clEnv.platform_id, &numDevices); 
    
    //-----------------------------------------------------
    // STEP 3: Create Context
    //----------------------------------------------------- 
    // Criar contexto usando clCreateContext() e
    execParam->clEnv.context = 
        clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable create a context %d\n", status);
        exit(1);
    }

    //-----------------------------------------------------
    // STEP 4: Create Comand Queue
    //----------------------------------------------------- 
    execParam->clEnv.cmdQueue = 
        clCreateCommandQueue(execParam->clEnv.context, devices[0], 0, &status);
    if(status != CL_SUCCESS){
        printf ("Unable create a command queue\n");
        exit(1);
    }

    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    rtmcl_load_forward_kernel(execParam);
    rtmcl_load_backward_kernel(execParam);
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    cl_int bin_status; 
#ifndef USE_FPGA
        execParam->clEnv.program = clCreateProgramWithSource(execParam->clEnv.context, 1, (const char**)&source_str,  NULL, &status);
        if (status != CL_SUCCESS) {
            printf ("Unable to create a program from source\n");
            exit(1);
        }
        
        char buildFlags[100];
        sprintf(buildFlags, "-DNT=%d -DNX=%d -DNZ=%d -DNXB=%d -DNZB=%d", 
            execParam->nt,  execParam->nx,  execParam->nz, execParam->nxb, execParam->nzb);
        printf("> Build Flags= %s \n", buildFlags);
        status = clBuildProgram(execParam->clEnv.program, numDevices, devices, buildFlags, NULL, NULL); 
        if (status != CL_SUCCESS) {
            char logBuffer[10240];
            clGetProgramBuildInfo(execParam->clEnv.program, devices[0], CL_PROGRAM_BUILD_LOG, sizeof(logBuffer), logBuffer, NULL);
            printf ("Unable to build a program, %d\n", status);
            printf("CL Compilation failed:\n%s", logBuffer);
            exit(1);
        }
        //-----------------------------------------------------
        // STEP 7: Configure the work-item structure
        // Define an index space (global work size) 
        // of work items for execution.
        //----------------------------------------------------- 
        int tx = ((execParam->nxe - 1) / 32 + 1) * 32;
        int tz = ((execParam->nze - 1) / 32 + 1) * 32;
        int tx_b = ((execParam->nxb - 1) / 32 + 1) * 32; 
        int tz_b = ((execParam->nzb - 1) / 32 + 1) * 32; 
        execParam->clEnv.globalWorkSize[0] = tx;
        execParam->clEnv.globalWorkSize[1] = tz;
        execParam->clEnv.globalWorkSizeUpb[0] = tx;
        execParam->clEnv.globalWorkSizeUpb[1] = 32;
        execParam->clEnv.globalWorkSizeTaper[0] = tx;
        execParam->clEnv.globalWorkSizeTaper[1] = tx_b;

        execParam->clEnv.singleWork[0] = 1;
        execParam->clEnv.singleWork[1] = 1;

        execParam->clEnv.localWorkSize[0] = BSIZE;
        execParam->clEnv.localWorkSize[1] = BSIZE;
#else
        // -------------------------------------------------
        // STEP 6: Create program for offline compiler
        // ------------------------------------------------- 
        execParam->clEnv.program = clCreateProgramWithBinary(execParam->clEnv.context, 1, devices, &source_size,
            (const unsigned char**)&source_str, &bin_status, &status);
        if (status != CL_SUCCESS){
            printf ("Unable to create a program from binary (%d) \n", status);
            exit(1);
        }
#endif
    execParam->clEnv.kforward.kernel = clCreateKernel(execParam->clEnv.program, RTMCL_FORWARD_NAME, &status);
    if (status != CL_SUCCESS){
        printf ("clCreateKernel: Unable to set a kernel '%s' from program! Status=%d \n", 
        	RTMCL_FORWARD_NAME,status );
        exit(1);
    }

    execParam->clEnv.kbackward.kernel = clCreateKernel(execParam->clEnv.program, RTMCL_BACKWARD_NAME, &status);
    if (status != CL_SUCCESS){
        printf ("clCreateKernel: Unable to set a kernel '%s' from program! Status=%d \n", 
            RTMCL_BACKWARD_NAME,status );
        exit(1);
    }
}

void rtmcl_destroy( RTMCLEnv * clEnv){
	rtmcl_unload_forward_kernel(clEnv);
    rtmcl_unload_backward_kernel(clEnv);
	clReleaseProgram        (clEnv->program);
    clReleaseCommandQueue   (clEnv->cmdQueue);
    clReleaseContext        (clEnv->context);
}

void rtmcl_mig_forward(RTMShot * shot, RTMExecParam * execParam){

    struct timeval st, et;
    memset(*shot->sfloat.PP,0,execParam->nze*execParam->nxe*sizeof(float));
    memset(*shot->sfloat.P,0,execParam->nze*execParam->nxe*sizeof(float));
    memset(*shot->sfloat.laplace,0,execParam->nze*execParam->nxe*sizeof(float));

    // write all parameters;
    gettimeofday(&st, NULL);
	rtmcl_write_forward_params(shot, execParam);
    gettimeofday(&et, NULL);
    shot->report.wrAVGTime += elapsedTimeMs(st, et);
    shot->report.wrTransferCnt++;

	// issue kernel
    gettimeofday(&st, NULL);
	rtmcl_enqueue_fkernel(&execParam->clEnv);
    gettimeofday(&et, NULL);
    shot->report.kernelAVGTime += elapsedTimeMs(st, et);
	// read results
    gettimeofday(&st, NULL);
	rtmcl_read_forward_output(shot,execParam);
    gettimeofday(&et, NULL);
    shot->report.rdAVGTime += elapsedTimeMs(st, et);
    shot->report.rdTransferCnt++;
}

void rtmcl_mig_backward(RTMShot * shot, RTMExecParam * execParam){
    struct timeval st, et;
    memset(*shot->sfloat.PP,0,execParam->nze*execParam->nxe*sizeof(float));
    memset(*shot->sfloat.P,0,execParam->nze*execParam->nxe*sizeof(float));
    memset(*shot->sfloat.PPR,0,execParam->nze*execParam->nxe*sizeof(float));
    memset(*shot->sfloat.PR,0,execParam->nze*execParam->nxe*sizeof(float));
    memset(*shot->sfloat.laplace,0,execParam->nze*execParam->nxe*sizeof(float));
    memset(*shot->sfloat.imloc,0,execParam->nz*execParam->nx*sizeof(float));

    // write all parameters;
    gettimeofday(&st, NULL);
    rtmcl_write_backward_params(shot, execParam);
    gettimeofday(&et, NULL);
    shot->report.wrAVGTime += elapsedTimeMs(st, et);
    shot->report.wrTransferCnt++;

    // issue kernel
    gettimeofday(&st, NULL);
    rtmcl_enqueue_bkernel(&execParam->clEnv);
    gettimeofday(&et, NULL);
    shot->report.kernelAVGTime += elapsedTimeMs(st, et);
    // read results
    gettimeofday(&st, NULL);
    rtmcl_read_backward_output(shot,execParam);
    gettimeofday(&et, NULL);
    shot->report.rdAVGTime += elapsedTimeMs(st, et);
    shot->report.rdTransferCnt++;
   
}
#endif //OPENCLRTM