#ifdef UPDATE_PP
#include <stdio.h>
#include <time.h>
#include <CL/cl.h>
#include <omp.h>
#include <math.h>

#include "su.h"
#include "ptsrc.h"
#include "taper.h"

#include "fd.h"
#include "clmisc.h"

float ** laplace;
float *  coefs;
float *  coefs_z;
float *  coefs_x;
float dx2inv,dz2inv,dt2;

#ifdef PERF_COUNTERS
    int wrTransferCnt;
    int rdTransferCnt;
    float fwAVGTime;
    float bwAVGTime;
    // float fwDeviceAVGTime;
    // float bwDeviceAVGTime;
    float kernelAVGTime;
    float wrAVGTime;
    float rdAVGTime;
    float deviceAVGTime; // rd+wr+kernel

void fd_print_report(int nx, int nz) {
    printf("> ================================================ \n");
    printf("> Exec Time Report (NX = %d NZ= %d):\n",nx,nz);
    printf("> Device       = %.1f (us)\n",deviceAVGTime);
    // printf("> Total        = %.1f (us)\n",execTime);
    printf("> WR TransfCnt = %d \n",wrTransferCnt);
    printf("> Write        = %.1f (%.2f%%)(us)\n",wrAVGTime, 
        (100.0*wrAVGTime/deviceAVGTime));
    printf("> RD TransfCnt = %d \n",rdTransferCnt);
    printf("> Read         = %.1f (%.2f%%)(us)\n",rdAVGTime,
        (100.0*rdAVGTime/deviceAVGTime));
    printf("> Device Fwrd  = %.1f (us)\n",fwAVGTime);
    printf("> Device Bwrd  = %.1f (us)\n",bwAVGTime);
    // printf("> ================================================ \n\n");
}
#endif

//////////////////////////////////////////
// OpenCL environment global variables 
size_t mtxBufferLength, brdBufferLength;
size_t imgBufferLength, sisBufferLength; 
size_t coefsBufferLength, upbBufferLength; 

size_t singleWork[2];   
size_t localWorkSize[2]; 
size_t globalWorkSize[2]; 
size_t globalWorkSizeUpb[2];    
size_t globalWorkSizeTaper[2]; 
// auxiliar variables 
int nxbin, nzbin; 
float *taperx, *taperz;        

// Laplacian and Time 
cl_mem bufferP, bufferPR;
cl_mem bufferPP, bufferPP_IN;
cl_mem bufferPP_OUT, bufferPPR;
cl_mem bufferV2, bufferLaplace; 
cl_mem bufferCoefs_x, bufferCoefs_z; 
// Tapper Apply 
cl_mem bufferTaper_x, bufferTaper_z;

// Save Uper Board
cl_mem bufferUPB; 
// Add Sismogram 
cl_mem bufferObs; 
// Img Condition 
cl_mem bufferImg; 
// Aux buffer 
cl_mem bufferSwap;
// OpenCL defines 
clEnvironment clEnv;

void fd_init_opencl(int order, int nxe, int nze, int nxb, int nzb, 
     int nt, int ns, float fac, char *kernelPath){

    //-----------------------------------------------------
    // STEP 0: Initialize OPENCL features
    // Matrix size with padding 
    //-----------------------------------------------------
    float dfrac;
    cl_int status; 
    nxbin=nxb; nzbin=nzb;  
    taperx = alloc1float(nxb);
    taperz = alloc1float(nzb);
   
    int tx = ((nxe - 1) / 32 + 1) * 32;
    int tz = ((nze - 1) / 32 + 1) * 32;
    int tx_b = ((nxb - 1) / 32 + 1) * 32; 
    int tz_b = ((nzb - 1) / 32 + 1) * 32; 

    sisBufferLength = nt*(nxe-(2*nxb))*sizeof(float);
    brdBufferLength = nxb*sizeof(float);
    mtxBufferLength = (nxe*nze)*sizeof(float);
    coefsBufferLength = (order+1)*sizeof(float);
    upbBufferLength = nt*nxe*(order/2)*sizeof(float);
    imgBufferLength = (nxe-(2*nxb))*(nze-(2*nzb))*sizeof(float);
    
    taperx = alloc1float(nxb);
    taperz = alloc1float(nzb);
      
    dfrac = sqrt(-log(fac))/(1.*nxb);
    for(int i=0;i<nxb;i++)
        taperx[i] = exp(-pow((dfrac*(nxb-i)),2));
    

    dfrac = sqrt(-log(fac))/(1.*nzb);
    for(int i=0;i<nzb;i++)
        taperz[i] = exp(-pow((dfrac*(nzb-i)),2));

    
    //-----------------------------------------------------
    // STEP 1: Read Kernel File
    //----------------------------------------------------- 
    size_t source_size;
    char *source_str; 
    source_str = clmiscReadFileStr(kernelPath, &source_size);       
    //-----------------------------------------------------
    // STEP 2: Initialize Plataforms and Devices
    //----------------------------------------------------- 
    
    cl_uint numDevices;  
    cl_platform_id  *   platforms   = NULL;
    cl_device_id    *   devices     = 
        clmiscGetPlatformAndDevice(platforms, PLATFORM_ID, &numDevices); 
    
    //-----------------------------------------------------
    // STEP 3: Create Context
    //----------------------------------------------------- 
    // Criar contexto usando clCreateContext() e
    clEnv.context = 
        clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable create a context %d\n", status);
        exit(1);
    }

    //-----------------------------------------------------
    // STEP 4: Create Comand Queue
    //----------------------------------------------------- 
    clEnv.cmdQueue = 
        clCreateCommandQueue(clEnv.context, devices[0], 0, &status);
    if(status != CL_SUCCESS){
        printf ("Unable create a command queue\n");
        exit(1);
    }

    //-----------------------------------------------------
    // STEP 5: Create buffers for the devices.
    //----------------------------------------------------- 
    bufferPP_IN = 
        clCreateBuffer(clEnv.context, CL_MEM_READ_ONLY, mtxBufferLength, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for pp_in\n");
        exit(1);
    }

    bufferPP_OUT = 
        clCreateBuffer(clEnv.context, CL_MEM_WRITE_ONLY, mtxBufferLength, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for pp_out\n");
        exit(1);
    }
    
    bufferP = 
        clCreateBuffer(clEnv.context, CL_MEM_READ_WRITE, mtxBufferLength, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for p %d \n",status);
        exit(1);
    }

    bufferPR = 
        clCreateBuffer(clEnv.context, CL_MEM_READ_WRITE, mtxBufferLength, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for pr %d \n",status);
        exit(1);
    }

    bufferPP = 
        clCreateBuffer(clEnv.context, CL_MEM_READ_WRITE, mtxBufferLength, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for pp\n");
        exit(1);
    }

    bufferPPR = 
        clCreateBuffer(clEnv.context, CL_MEM_READ_WRITE, mtxBufferLength, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for ppr %d \n",status);
        exit(1);
    }

    bufferSwap = 
        clCreateBuffer(clEnv.context, CL_MEM_READ_WRITE, mtxBufferLength, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for swap %d \n",status);
        exit(1);
    }

    bufferObs = 
        clCreateBuffer(clEnv.context, CL_MEM_READ_WRITE, sisBufferLength, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for obs %d \n",status);
        exit(1);
    }
    
    bufferUPB = 
        clCreateBuffer(clEnv.context, CL_MEM_READ_WRITE, upbBufferLength, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for upb %d \n",status);
        exit(1);
    }

    bufferLaplace = 
        clCreateBuffer(clEnv.context, CL_MEM_READ_WRITE, mtxBufferLength, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for laplace\n");
        exit(1);
    }

    bufferV2 = 
        clCreateBuffer(clEnv.context, CL_MEM_READ_WRITE, mtxBufferLength, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for V2\n");
        exit(1);
    }

    bufferTaper_x = 
        clCreateBuffer(clEnv.context, CL_MEM_READ_WRITE, brdBufferLength, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for TaperX\n");
        exit(1);
    }

    bufferTaper_z = 
        clCreateBuffer(clEnv.context, CL_MEM_READ_WRITE, brdBufferLength, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for TaperZ\n");
        exit(1);
    }
    
    bufferCoefs_x = 
        clCreateBuffer(clEnv.context, CL_MEM_READ_WRITE, coefsBufferLength, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for coefs x\n");
        exit(1);
    }

    bufferCoefs_z = 
        clCreateBuffer(clEnv.context, CL_MEM_READ_WRITE, coefsBufferLength, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for coefs z\n");
        exit(1);
    }

    bufferImg = 
        clCreateBuffer(clEnv.context, CL_MEM_READ_WRITE, imgBufferLength, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for img\n");
        exit(1);
    }

    // -----------------------------------------------------
    // STEP 6: Create, compile and build the program
    // ----------------------------------------------------- 
    cl_int bin_status; 
#ifndef USE_FPGA
        clEnv.program = clCreateProgramWithSource(clEnv.context, 1, (const char**)&source_str,  NULL, &status);
        if (status != CL_SUCCESS) {
            printf ("Unable to create a program from source\n");
            exit(1);
        }
    
        status = clBuildProgram(clEnv.program, numDevices, devices, NULL, NULL, NULL); 
        if (status != CL_SUCCESS) {
            char logBuffer[10240];
            clGetProgramBuildInfo(clEnv.program, devices[0], CL_PROGRAM_BUILD_LOG, sizeof(logBuffer), logBuffer, NULL);
            printf ("Unable to build a program, %d\n", status);
            printf("CL Compilation failed:\n%s", logBuffer);
            exit(1);
        }
        //-----------------------------------------------------
        // STEP 7: Configure the work-item structure
        // Define an index space (global work size) 
        // of work items for execution.
        //----------------------------------------------------- 
        globalWorkSize[0] = tx;
        globalWorkSize[1] = tz;
        globalWorkSizeUpb[0] = tx;
        globalWorkSizeUpb[1] = 32;
        globalWorkSizeTaper[0] = tx;
        globalWorkSizeTaper[1] = tx_b;

        singleWork[0] = 1;
        singleWork[1] = 1;

        localWorkSize[0] = BSIZE;
        localWorkSize[1] = BSIZE;
#else
        // -------------------------------------------------
        // STEP 6: Create program for offline compiler
        // ------------------------------------------------- 
        clEnv.program = clCreateProgramWithBinary(clEnv.context, 1, devices, &source_size,
            (const unsigned char**)&source_str, &bin_status, &status);
        if (status != CL_SUCCESS){
            printf ("Unable to create a program from binary (%d) \n", status);
            exit(1);
        }
#endif

    //-----------------------------------------------------
    // STEP 8: Create the kernel
    //  Use clCreateKernel() to create a kernel from the 
    //----------------------------------------------------- 
    clEnv.kernel_taper = clCreateKernel(clEnv.program, "taper_apply", &status);
    if (status != CL_SUCCESS){
        printf ("clCreateKernel: Unable to set a kernel tapper from program! Status=%d \n", status);
        exit(1);
    }

    clEnv.kernel_lap = clCreateKernel(clEnv.program, "fd_step", &status);
    if (status != CL_SUCCESS){
        printf ("clCreateKernel: Unable to set a kernel laplacian from program! Status=%d \n", status);
        exit(1);
    }

    clEnv.kernel_time = clCreateKernel(clEnv.program, "fd_time", &status);
    if (status != CL_SUCCESS){
        printf ("clCreateKernel: Unable to set a kernel time from program! Status=%d \n", status);
        exit(1);
    }

    clEnv.kernel_src = clCreateKernel(clEnv.program, "ptsrc", &status);
    if (status != CL_SUCCESS){
        printf ("clCreateKernel: Unable to set a kernel src from program! Status=%d \n", status);
        exit(1);
    }
    
    clEnv.kernel_upb = clCreateKernel(clEnv.program, "upb", &status);
    if (status != CL_SUCCESS){
        printf ("clCreateKernel: Unable to set a kernel upb from program! Status=%d \n", status);
        exit(1);
    }

    clEnv.kernel_upb_r = clCreateKernel(clEnv.program, "upb_reverse", &status);
    if (status != CL_SUCCESS){
        printf ("clCreateKernel: Unable to set a kernel upb r from program! Status=%d \n", status);
        exit(1);
    }

    clEnv.kernel_sism = clCreateKernel(clEnv.program, "add_sism", &status);
    if (status != CL_SUCCESS){
        printf ("clCreateKernel: Unable to set a kernel sism from program! Status=%d \n", status);
        exit(1);
    }

    clEnv.kernel_img = clCreateKernel(clEnv.program, "img_cond", &status);
    if (status != CL_SUCCESS){
        printf ("clCreateKernel: Unable to set a kernel img cond from program! Status=%d \n", status);
        exit(1);
    }

    //-------------------------------------------------------
    // End OPENCL Initialize
    //-------------------------------------------------------
}

void fd_init(int order, int nx, int nz, int nxb, int nzb, int nt, int ns,
    float fac, float dx, float dz, float dt, char * kernelPath){
    int io;
    dx2inv = (1./dx)*(1./dx);
    dz2inv = (1./dz)*(1./dz);
    dt2 = dt*dt;
    coefs = calc_coefs(order);
    laplace = alloc2float(nz,nx);

    coefs_z = calc_coefs(order);
    coefs_x = calc_coefs(order);

    memset(*laplace,0,nz*nx*sizeof(float));
    fd_init_opencl(order,nx,nz,nxb,nzb,nt,ns,fac,kernelPath); 

    wrTransferCnt=0;
    rdTransferCnt=0;
    kernelAVGTime=0.;
    deviceAVGTime=0.;
    wrAVGTime=0.;
    rdAVGTime=0.;
    fwAVGTime=0.;
    bwAVGTime=0.;

    // pre calc coefs 8 d2 inv
    for (io = 0; io <= order; io++) {
        coefs_z[io] = dz2inv * coefs[io];
        coefs_x[io] = dx2inv * coefs[io];
    } 
    return;
}

void fd_reinit(int order, int nx, int nz){
    // todo: free coefs or realloc
    int io;
    coefs = calc_coefs(order);
    coefs_z = calc_coefs(order);
    coefs_x = calc_coefs(order);
    // pre calc coefs 8 d2inv
    for (io = 0; io <= order; io++) {
        coefs_z[io] = dz2inv * coefs[io];
        coefs_x[io] = dx2inv * coefs[io];
    }
    #ifdef PERF_COUNTERS
        wrTransferCnt=0;
        rdTransferCnt=0;
        kernelAVGTime=0.;
        deviceAVGTime=0.;
        wrAVGTime=0.;
        rdAVGTime=0.;
        fwAVGTime=0.;
        bwAVGTime=0.;
    #endif  
    // todo: free laplace or realloc
    laplace = alloc2float(nz,nx);
    memset(*laplace,0,nz*nx*sizeof(float));
    return;
}

// Release Allocated Buffers
void fd_destroy(){
    free2float(laplace);
    free1float(coefs);
    
    #ifdef OPENCL
        clReleaseKernel         (clEnv.kernel_lap);
        clReleaseKernel         (clEnv.kernel_img);
        clReleaseKernel         (clEnv.kernel_upb);
        clReleaseKernel         (clEnv.kernel_src);
        clReleaseKernel         (clEnv.kernel_sism);
        clReleaseKernel         (clEnv.kernel_time);
        clReleaseKernel         (clEnv.kernel_taper);
        clReleaseKernel         (clEnv.kernel_upb_r);
        clReleaseProgram        (clEnv.program);
        clReleaseCommandQueue   (clEnv.cmdQueue);
        clReleaseContext        (clEnv.context);
        
        clReleaseMemObject      (bufferP);
        clReleaseMemObject      (bufferPR);
        clReleaseMemObject      (bufferPP); 
        clReleaseMemObject      (bufferPP_IN); 
        clReleaseMemObject      (bufferPP_OUT); 
        clReleaseMemObject      (bufferPPR);
        clReleaseMemObject      (bufferV2); 
        clReleaseMemObject      (bufferLaplace); 
        clReleaseMemObject      (bufferCoefs_x); 
        clReleaseMemObject      (bufferCoefs_z); 
        clReleaseMemObject      (bufferTaper_x);
        clReleaseMemObject      (bufferTaper_z);
        clReleaseMemObject      (bufferUPB); 
        clReleaseMemObject      (bufferObs); 
        clReleaseMemObject      (bufferImg); 
        // clReleaseMemObject      (bufferSwap);
    #endif

    return;
}

void write_buffers(float **p, float **pp, float **v2, float *srce, float ***upb, 
    float *taperx, float *taperz, float **d_obs, float **imloc, int *sx, int is, int flag){
    
    cl_int status; 
    if(flag==0){
        status = clEnqueueWriteBuffer(clEnv.cmdQueue, bufferP, CL_TRUE, 0, mtxBufferLength, p[0], 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to copy P to buffer\n");
            exit(1);
        }
    
        status = clEnqueueWriteBuffer(clEnv.cmdQueue, bufferPP, CL_TRUE, 0, mtxBufferLength, pp[0], 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to copy P to buffer\n");
            exit(1);
        }

        status = clEnqueueWriteBuffer(clEnv.cmdQueue, bufferV2, CL_TRUE, 0, mtxBufferLength, v2[0], 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to copy V2 to buffer\n");
            exit(1);
        }

        status = clEnqueueWriteBuffer(clEnv.cmdQueue, bufferCoefs_x, CL_TRUE, 0, coefsBufferLength, coefs_x, 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to copy coefs x to buffer\n");
            exit(1);
        }

        status = clEnqueueWriteBuffer(clEnv.cmdQueue, bufferCoefs_z, CL_TRUE, 0, coefsBufferLength, coefs_z, 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to copy coefs z to buffer\n");
            exit(1);
        }
        
        status = clEnqueueWriteBuffer(clEnv.cmdQueue, bufferTaper_x, CL_TRUE, 0, brdBufferLength, taperx, 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to copy taper x to buffer\n");
            exit(1);
        }

        status = clEnqueueWriteBuffer(clEnv.cmdQueue, bufferTaper_z, CL_TRUE, 0, brdBufferLength, taperz, 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to copy taper z to buffer\n");
            exit(1);
        }
  
        status = clEnqueueWriteBuffer(clEnv.cmdQueue, bufferUPB, CL_TRUE, 0, upbBufferLength, upb[0][0], 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to copy srce to buffer\n");
            exit(1);
        }
    }else{
        status = clEnqueueWriteBuffer(clEnv.cmdQueue, bufferPR, CL_TRUE, 0, mtxBufferLength, p[0], 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to copy PR to buffer\n");
            exit(1);
        }
    
        status = clEnqueueWriteBuffer(clEnv.cmdQueue, bufferPPR, CL_TRUE, 0, mtxBufferLength, pp[0], 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to copy PPR to buffer\n");
            exit(1);
        }
        status = clEnqueueWriteBuffer(clEnv.cmdQueue, bufferObs, CL_TRUE, 0, sisBufferLength, d_obs[is], 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to copy d_obs to buffer\n");
            exit(1);
        }

        status = clEnqueueWriteBuffer(clEnv.cmdQueue, bufferImg, CL_TRUE, 0, imgBufferLength, imloc[0], 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to copy img to buffer\n");
            exit(1);
        }
    }
}

void set_args(int order, int nx, int nz, int nt, float dt2, 
    int *sx, int is, int sz, float *srce, int it, int gz, int flag){
    cl_int status;
    
    //------------------Kernel Laplacian ----------------------- 
    status = clSetKernelArg(clEnv.kernel_lap, 0, sizeof(int), &order);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 0 kernel laplacian argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(clEnv.kernel_lap, 1, sizeof(int), &nx);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 1 kernel laplacian argument %d\n",status);
        exit(1);
    }    

    status = clSetKernelArg(clEnv.kernel_lap, 2, sizeof(int), &nz);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 2 kernel laplacian argument %d\n",status);
        exit(1);
    }            

    if(flag==0||flag==1)
        status = clSetKernelArg(clEnv.kernel_lap, 3, sizeof(cl_mem), &bufferP);
    else
        status = clSetKernelArg(clEnv.kernel_lap, 3, sizeof(cl_mem), &bufferPR);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 3 kernel laplacian argument %d\n",status);
        exit(1);
    }    
    
    status = clSetKernelArg(clEnv.kernel_lap, 4, sizeof(cl_mem), &bufferLaplace);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 4 kernel laplacian argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(clEnv.kernel_lap, 5, sizeof(cl_mem), &bufferCoefs_x);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 5 kernel laplacian argument %d\n",status);
        exit(1);
    }

    status = clSetKernelArg(clEnv.kernel_lap, 6, sizeof(cl_mem), &bufferCoefs_z);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 6 kernel laplacian argument %d\n",status);
        exit(1);
    }               
    // ---------------------- Kernel Time ------------------------ 
    status = clSetKernelArg(clEnv.kernel_time, 0, sizeof(int), &nx);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 0 kernel_time argument %d\n",status);
        exit(1);
    }    

    status = clSetKernelArg(clEnv.kernel_time, 1, sizeof(int), &nz);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 1 kernel_time argument %d\n",status);
        exit(1);
    }    

    if(flag==0||flag==1)
        status = clSetKernelArg(clEnv.kernel_time, 2, sizeof(cl_mem), &bufferP);
    else
        status = clSetKernelArg(clEnv.kernel_time, 2, sizeof(cl_mem), &bufferPR);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 2 kernel_time argument %d\n",status);
        exit(1);
    }

    if(flag==0||flag==1)
        status = clSetKernelArg(clEnv.kernel_time, 3, sizeof(cl_mem), &bufferPP);
    else
        status = clSetKernelArg(clEnv.kernel_time, 3, sizeof(cl_mem), &bufferPPR);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 3 kernel_time argument %d\n",status);
        exit(1);
    }       

    status = clSetKernelArg(clEnv.kernel_time, 4, sizeof(cl_mem), &bufferV2);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 4 kernel_time argument %d\n",status);
        exit(1);
    }       

    status = clSetKernelArg(clEnv.kernel_time, 5, sizeof(cl_mem), &bufferLaplace);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 5 kernel_time argument %d\n",status);
        exit(1);
    }  

    status = clSetKernelArg(clEnv.kernel_time, 6, sizeof(float), &dt2);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 6 kernel_time argument %d\n",status);
        exit(1);
    }

    // ------------------- Kernel Taper ------------------------    
    status = clSetKernelArg(clEnv.kernel_taper, 0, sizeof(int), &nx);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 0 kernel tapper argument %d\n",status);
        exit(1);
    }   
    status = clSetKernelArg(clEnv.kernel_taper, 1, sizeof(int), &nz);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 1 kernel tapper argument %d\n",status);
        exit(1);
    }   
    status = clSetKernelArg(clEnv.kernel_taper, 2, sizeof(int), &nxbin);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 2 kernel tapper argument %d\n",status);
        exit(1);
    }   
    status = clSetKernelArg(clEnv.kernel_taper, 3, sizeof(int), &nzbin);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 3 kernel tapper argument %d\n",status);
        exit(1);
    }   

    if(flag==0||flag==1)
        status = clSetKernelArg(clEnv.kernel_taper, 4, sizeof(cl_mem), &bufferP);
    else
        status = clSetKernelArg(clEnv.kernel_taper, 4, sizeof(cl_mem), &bufferPR);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 4 kernel tapper argument %d\n",status);
        exit(1);
    }   

    if(flag==0||flag==1)
        status = clSetKernelArg(clEnv.kernel_taper, 5, sizeof(cl_mem), &bufferPP);
    else
        status = clSetKernelArg(clEnv.kernel_taper, 5, sizeof(cl_mem), &bufferPPR);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 5 kernel tapper argument %d\n",status);
        exit(1);
    } 

    status = clSetKernelArg(clEnv.kernel_taper, 6, sizeof(cl_mem), &bufferTaper_x);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 6 kernel tapper argument %d\n",status);
        exit(1);
    }    

    status = clSetKernelArg(clEnv.kernel_taper, 7, sizeof(cl_mem), &bufferTaper_z);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 7 kernel tapper argument %d\n",status);
        exit(1);
    }  

    if(flag==0||flag==1){
        // ------------------- Kernel Src ------------------------ 
        status = clSetKernelArg(clEnv.kernel_src, 0, sizeof(int), &nz);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 0 kernel argument %d\n",status);
            exit(1);
        }

        status = clSetKernelArg(clEnv.kernel_src, 1, sizeof(cl_mem), &bufferPP);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 1 kernel argument %d\n",status);
            exit(1);
        }   
        
        status = clSetKernelArg(clEnv.kernel_src, 2, sizeof(int), &sx[is]);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 2 kernel argument %d\n",status);
            exit(1);
        } 

        status = clSetKernelArg(clEnv.kernel_src, 3, sizeof(int), &sz);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 3 kernel argument %d\n",status);
            exit(1);
        }    

        status = clSetKernelArg(clEnv.kernel_src, 4, sizeof(float), &srce[it]);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 4 kernel argument %d\n",status);
            exit(1);
        }

        // ------------------- Kernel Up Board ------------------------ 
        status = clSetKernelArg(clEnv.kernel_upb, 0, sizeof(int), &order);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 0 kernel upb argument %d\n",status);
            exit(1);
        }   

        status = clSetKernelArg(clEnv.kernel_upb, 1, sizeof(int), &nx);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 1 kernel upb argument %d\n",status);
            exit(1);
        }   

        status = clSetKernelArg(clEnv.kernel_upb, 2, sizeof(int), &nz);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 2 kernel upb argument %d\n",status);
            exit(1);
        }   

        status = clSetKernelArg(clEnv.kernel_upb, 3, sizeof(int), &nzbin);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 3 kernel upb argument %d\n",status);
            exit(1);
        }   
        status = clSetKernelArg(clEnv.kernel_upb, 4, sizeof(cl_mem), &bufferPP);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 4 kernel upb argument %d\n",status);
            exit(1);
        }   
        
        status = clSetKernelArg(clEnv.kernel_upb, 5, sizeof(cl_mem), &bufferUPB);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 5 kernel upb argument %d\n",status);
            exit(1);
        }

        status = clSetKernelArg(clEnv.kernel_upb, 6, sizeof(int), &it);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 6 kernel upb argument %d\n",status);
            exit(1);
        }
    }

    if(flag==1){
        // ------------------- Kernel Up Reverse Board ------------------------ 
        status = clSetKernelArg(clEnv.kernel_upb_r, 0, sizeof(int), &order);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 0 kernel upb reverse argument %d\n",status);
            exit(1);
        }   

        status = clSetKernelArg(clEnv.kernel_upb_r, 1, sizeof(int), &nx);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 1 kernel upb reverse argument %d\n",status);
            exit(1);
        }   

        status = clSetKernelArg(clEnv.kernel_upb_r, 2, sizeof(int), &nz);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 2 kernel upb reverse argument %d\n",status);
            exit(1);
        } 

        status = clSetKernelArg(clEnv.kernel_upb_r, 3, sizeof(int), &nzbin);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 3 kernel upb reverse argument %d\n",status);
            exit(1);
        }     

        status = clSetKernelArg(clEnv.kernel_upb_r, 4, sizeof(int), &nt);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 4 kernel upb reverse argument %d\n",status);
            exit(1);
        }   
        status = clSetKernelArg(clEnv.kernel_upb_r, 5, sizeof(cl_mem), &bufferPP);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 5 kernel upb reverse argument %d\n",status);
            exit(1);
        }   
        
        status = clSetKernelArg(clEnv.kernel_upb_r, 6, sizeof(cl_mem), &bufferUPB);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 6 kernel upb reverse argument %d\n",status);
            exit(1);
        }

        status = clSetKernelArg(clEnv.kernel_upb_r, 7, sizeof(int), &it);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 7 kernel upb reverse argument %d\n",status);
            exit(1);
        }
    }
    if(flag == 2){
        //---------------------- Add Sism ------------------------ 
        status = clSetKernelArg(clEnv.kernel_sism, 0, sizeof(int), &nx);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 0 kernel sism argument %d\n",status);
            exit(1);
        }

        status = clSetKernelArg(clEnv.kernel_sism, 1, sizeof(int), &nz);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 1 kernel sism argument %d\n",status);
            exit(1);
        }

        status = clSetKernelArg(clEnv.kernel_sism, 2, sizeof(int), &nxbin);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 2 kernel sism argument %d\n",status);
            exit(1);
        }
        
        status = clSetKernelArg(clEnv.kernel_sism, 3, sizeof(int), &nt);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 3 kernel sism argument %d\n",status);
            exit(1);
        }

        status = clSetKernelArg(clEnv.kernel_sism, 4, sizeof(int), &is);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 4 kernel sism argument %d\n",status);
            exit(1);
        }

        status = clSetKernelArg(clEnv.kernel_sism, 5, sizeof(int), &it);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 5 kernel sism argument %d\n",status);
            exit(1);
        }
                
        status = clSetKernelArg(clEnv.kernel_sism, 6, sizeof(int), &gz);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 6 kernel sism argument %d\n",status);
            exit(1);
        }

        status = clSetKernelArg(clEnv.kernel_sism, 7, sizeof(cl_mem), &bufferObs);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 7 kernel sism argument %d\n",status);
            exit(1);
        }

        status = clSetKernelArg(clEnv.kernel_sism, 8, sizeof(cl_mem), &bufferPPR);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 8 kernel sism argument %d\n",status);
            exit(1);
        }

        // -------------------------- imloc ------------------------------
        status = clSetKernelArg(clEnv.kernel_img, 0, sizeof(int), &nx);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 0 kernel img argument %d\n",status);
            exit(1);
        }

        status = clSetKernelArg(clEnv.kernel_img, 1, sizeof(int), &nz);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 1 kernel img argument %d\n",status);
            exit(1);
        }

        status = clSetKernelArg(clEnv.kernel_img, 2, sizeof(int), &nxbin);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 2 kernel img argument %d\n",status);
            exit(1);
        }

        status = clSetKernelArg(clEnv.kernel_img, 3, sizeof(int), &nzbin);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 3 kernel img argument %d\n",status);
            exit(1);
        }

        status = clSetKernelArg(clEnv.kernel_img, 4, sizeof(cl_mem), &bufferImg);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 4 kernel img argument %d\n",status);
            exit(1);
        }

        status = clSetKernelArg(clEnv.kernel_img, 5, sizeof(cl_mem), &bufferP);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 5 kernel img argument %d\n",status);
            exit(1);
        }

        status = clSetKernelArg(clEnv.kernel_img, 6, sizeof(cl_mem), &bufferPPR);
        if (status != CL_SUCCESS) {
            printf ("Unable to set 6 kernel img argument %d\n",status);
            exit(1);
        }
    }
}

void fd_step_forward(int order, float **p, float **pp, float **v2, float ***upb, 
    int nz, int nx, int nt, int is, int sz, int *sx, float *srce){
    cl_int status;
    int ix,iz,elapsed;
    struct timeval st, et, stCR, etCR, stCW, etCW, stK, etK;
    //Start total time 
    gettimeofday(&st, NULL);

    //-----------------------------------------------------
    // STEP 6: Write data for the devices buffers
    //----------------------------------------------------- 
    // Start write time
    gettimeofday(&stCW, NULL);
    write_buffers(p,pp,v2,srce,upb,taperx,taperz,NULL,NULL,sx,is,0); 
    // Calc avg write time  
    gettimeofday(&etCW, NULL);
    elapsed = ((etCW.tv_sec - stCW.tv_sec) * 1000000) + (etCW.tv_usec - stCW.tv_usec);
    wrAVGTime += (elapsed*1.0);
    wrTransferCnt++;
    //-----------------------------------------------------
    // STEP 9: Set the kernel arguments
    //  Associate the input and output buffers 
    //-----------------------------------------------------     
    // start Kernel time  
    gettimeofday(&stK, NULL);
    for (int it = 0; it < nt; it++){
        bufferSwap = bufferPP; 
        bufferPP = bufferP; 
        bufferP = bufferSwap; 
       
        set_args(order,nx,nz,nt,dt2,sx,is,sz,srce,it,NULL,0);
       
        //-----------------------------------------------------
        // STEP 11: Enqueue the kernel for execution
        //-----------------------------------------------------  
        status = 
            clEnqueueNDRangeKernel(clEnv.cmdQueue, clEnv.kernel_taper, 2, NULL, globalWorkSizeTaper, NULL, 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to run the kernel on NDRange Taper %d\n", status);
            exit(1);
        }
        clFinish(clEnv.cmdQueue);     

        status = 
            clEnqueueNDRangeKernel(clEnv.cmdQueue, clEnv.kernel_lap, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to run the kernel on NDRange Laplacian %d\n", status);
            exit(1);
        }
        clFinish(clEnv.cmdQueue); 
        
        status = 
            clEnqueueNDRangeKernel(clEnv.cmdQueue, clEnv.kernel_time, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to run the kernel_time on NDRange Time %d\n", status);
            exit(1);
        }
        clFinish(clEnv.cmdQueue);  
        
        status = 
            clEnqueueNDRangeKernel(clEnv.cmdQueue, clEnv.kernel_src, 1, NULL, singleWork, NULL, 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to run the kernel_src on NDRange src %d\n", status);
            exit(1);
        }
        clFinish(clEnv.cmdQueue); 

        status = 
            clEnqueueNDRangeKernel(clEnv.cmdQueue, clEnv.kernel_upb, 2, NULL, globalWorkSizeUpb, NULL, 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to run the kernel_upb on NDRange upb %d\n", status);
            exit(1);
        }
        clFinish(clEnv.cmdQueue);   
        if((it+1)%100 == 0){fprintf(stdout,"\r* it = %d / %d (%d%)",it+1,nt,(100*(it+1)/nt));fflush(stdout);}
    }  
    // Calc avg exec time  
    gettimeofday(&etK, NULL);
    elapsed = ((etK.tv_sec - stK.tv_sec) * 1000000) + (etK.tv_usec - stK.tv_usec);
    kernelAVGTime += (elapsed*1.0);

    //-----------------------------------------------------
    // STEP 12: Read the output buffer back to the host
    //----------------------------------------------------- 
    
    gettimeofday(&stCR, NULL);
    status = clEnqueueReadBuffer(clEnv.cmdQueue, bufferPP, CL_TRUE, 0, mtxBufferLength, pp[0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to read the PPF buffer status:%d \n", status);
        exit(1);
    }

    status = clEnqueueReadBuffer(clEnv.cmdQueue, bufferP, CL_TRUE, 0, mtxBufferLength, p[0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to read the PF buffer status:%d \n", status);
        exit(1);
    }

    status = clEnqueueReadBuffer(clEnv.cmdQueue, bufferUPB, CL_TRUE, 0, upbBufferLength, upb[0][0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to read the UPBF buffer status:%d \n", status);
        exit(1);
    }

    // Calc avg read time  
    gettimeofday(&etCR, NULL);
    elapsed = ((etCR.tv_sec - stCR.tv_sec) * 1000000) + (etCR.tv_usec - stCR.tv_usec);
    rdAVGTime += (elapsed*1.0);
    rdTransferCnt++;
    
    // Calc avg total time  
    gettimeofday(&et, NULL);
    elapsed = ((et.tv_sec - st.tv_sec) * 1000000) + (et.tv_usec - st.tv_usec);
    fwAVGTime += (elapsed*1.0);
    // laplacian_counter++;  
}

void fd_step_back(int order, float **p, float **pp, float **pr, float **ppr, float **v2, float ***upb, int *sx,
    int nz, int nx, int nt, int is, int it, int sz, int gz, float ***snaps, float **imloc, float **d_obs, float *srce){
    int ix,iz,elapsed; 
    cl_int status;
    struct timeval st, et, stCR, etCR, stCW, etCW, stK, etK;
    gettimeofday(&st, NULL);

    gettimeofday(&stCW, NULL);
    write_buffers(p,pp,v2,srce,upb,taperx,taperz,d_obs,imloc,sx,is,0); 
    write_buffers(pr,ppr,v2,srce,upb,taperx,taperz,d_obs,imloc,sx,is,1); 
    gettimeofday(&etCW, NULL);
    elapsed = ((etCW.tv_sec - stCW.tv_sec) * 1000000) + (etCW.tv_usec - stCW.tv_usec);
    wrAVGTime += (elapsed*1.0);
    wrTransferCnt++; 
        // start Kernel time  
    gettimeofday(&stK, NULL);
    for(it=0; it<nt; it++){
        if(it==0 || it==1){
            for(ix=0; ix<nx; ix++){
                for(iz=0; iz<nz; iz++){
                    pp[ix][iz] = snaps[1-it][ix][iz];                       
                }
            }
            status = 
                clEnqueueWriteBuffer(clEnv.cmdQueue, bufferPP, CL_TRUE, 0, mtxBufferLength, pp[0], 0, NULL, NULL);
            if (status != CL_SUCCESS) {
                printf ("Unable to copy P to buffer\n");
                exit(1);
            }
        }else{
            //-----------------------------------------------------
            // STEP 6: Write data for the devices buffers
            //-----------------------------------------------------   
            set_args(order,nx,nz,nt,dt2,sx,is,sz,srce,it,gz,1);
            
            status = 
                clEnqueueNDRangeKernel(clEnv.cmdQueue, clEnv.kernel_lap, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
            if (status != CL_SUCCESS) {
                printf ("Unable to run the kernel on NDRange Laplacian %d\n", status);
                exit(1);
            }
            clFinish(clEnv.cmdQueue); 
            
            status = 
                clEnqueueNDRangeKernel(clEnv.cmdQueue, clEnv.kernel_time, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
            if (status != CL_SUCCESS) {
                printf ("Unable to run the kernel_time on NDRange Time %d\n", status);
                exit(1);
            }
            clFinish(clEnv.cmdQueue);   

            status = 
                clEnqueueNDRangeKernel(clEnv.cmdQueue, clEnv.kernel_upb_r, 2, NULL, globalWorkSizeUpb, NULL, 0, NULL, NULL);
            if (status != CL_SUCCESS) {
                printf ("Unable to run the kernel_time on NDRange kernel_upb_reverse %d\n", status);
                exit(1);
            }
            clFinish(clEnv.cmdQueue);  
        }

        bufferSwap = bufferPP;
        bufferPP = bufferP;
        bufferP = bufferSwap;

        set_args(order,nx,nz,nt,dt2,sx,is,sz,srce,it,gz,2);
            
        status = 
            clEnqueueNDRangeKernel(clEnv.cmdQueue, clEnv.kernel_taper, 2, NULL, globalWorkSizeTaper, NULL, 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to run the kernel on NDRange Taper %d\n", status);
            exit(1);
        }
        clFinish(clEnv.cmdQueue); 

        status = 
            clEnqueueNDRangeKernel(clEnv.cmdQueue, clEnv.kernel_lap, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to run the kernel on NDRange Laplacian %d\n", status);
            exit(1);
        }
        clFinish(clEnv.cmdQueue); 
        
        status = 
            clEnqueueNDRangeKernel(clEnv.cmdQueue, clEnv.kernel_time, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to run the kernel_time on NDRange Time %d\n", status);
            exit(1);
        }
        clFinish(clEnv.cmdQueue);   

        status = 
            clEnqueueNDRangeKernel(clEnv.cmdQueue, clEnv.kernel_sism, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to run the kernel_time on NDRange Time %d\n", status);
            exit(1);
        }

        status = 
            clEnqueueNDRangeKernel(clEnv.cmdQueue, clEnv.kernel_img, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        if (status != CL_SUCCESS) {
            printf ("Unable to run the kernel_time on NDRange Time %d\n", status);
            exit(1);
        }
        clFinish(clEnv.cmdQueue);   

        //-----------------------------------------------------
        // STEP 12: Read the output buffer back to the host
        //----------------------------------------------------- 
       
        bufferSwap = bufferPPR;
        bufferPPR = bufferPR;
        bufferPR = bufferSwap;

        if((it+1)%100 == 0){fprintf(stdout,"\r* it = %d / %d (%d%)",it+1,nt,(100*(it+1)/nt));fflush(stdout);}
    } 
    gettimeofday(&etK, NULL);
    elapsed = ((etK.tv_sec - stK.tv_sec) * 1000000) + (etK.tv_usec - stK.tv_usec);
    kernelAVGTime += (elapsed*1.0);

    gettimeofday(&stCR, NULL);
    status = clEnqueueReadBuffer(clEnv.cmdQueue, bufferImg, CL_TRUE, 0, imgBufferLength, imloc[0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to read the img buffer status:%d \n", status);
        exit(1);
    }
    gettimeofday(&etCR, NULL);
    elapsed = ((etCR.tv_sec - stCR.tv_sec) * 1000000) + (etCR.tv_usec - stCR.tv_usec);
    rdAVGTime += (elapsed*1.0);
    rdTransferCnt++;
    gettimeofday(&et, NULL);
    elapsed = ((et.tv_sec - st.tv_sec) * 1000000) + (et.tv_usec - st.tv_usec);
    bwAVGTime += (elapsed*1.0);
    deviceAVGTime = fwAVGTime + bwAVGTime; 
}

void fd_laplacian(int nz, int nx, float **p, float **laplace, int order){
    int ix,iz,io;
    float acmx = 0, acmz = 0;

    for(ix=order/2;ix<nx-order/2;ix++){
        for(iz=order/2;iz<nz-order/2;iz++){
            for(io=0;io<=order;io++){
                acmz += p[ix][iz+io-order/2]*coefs[io];
                acmx += p[ix+io-order/2][iz]*coefs[io];
            }
            laplace[ix][iz] = acmz*dz2inv + acmx*dx2inv;
            acmx = 0.0;
            acmz = 0.0;
        }
    }
}

#endif // UPDATE_PP