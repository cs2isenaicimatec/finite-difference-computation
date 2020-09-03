#ifndef UPDATE_PP
#include <stdio.h>
#include <time.h>
#include <CL/cl.h>
#include <omp.h>

#include "su.h"
#include "ptsrc.h"
#include "taper.h"

#include "fd.h"
#include "clmisc.h"


static float ** laplace = NULL;
static float *  coefs = NULL;
static float *  coefs_z = NULL;
static float *  coefs_x = NULL;
static float dx2inv,dz2inv,dt2;
static int NX, NZ, ORDER;

// ---OPENCL Variables---  
clEnvironment clEnv;


cl_mem bufferP;
cl_mem bufferLAPLACE; 

size_t imgBufferLength; 
// --------- End ---------  

size_t localWorkSize[2]; 
size_t globalWorkSize[2];    

// profiling counters
unsigned int laplacian_counter;
float avgSerial;
float avgDevice;
float avgWriteCL;
float avgReadCL;
float avgKernel;


#ifdef OPENCL
void fd_init_opencl(int nxe, int nze, char * kernelPath){

    //-----------------------------------------------------
    // STEP 0: Initialize OPENCL features
    // Matrix size with padding 
    //-----------------------------------------------------
    cl_int status; 
    int tx = ((nxe - 1) / 32 + 1) * 32;
    int tz = ((nze - 1) / 32 + 1) * 32; 
    imgBufferLength = (nxe*nze)*sizeof(float);


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
    bufferP = 
        clCreateBuffer(clEnv.context, CL_MEM_READ_ONLY, imgBufferLength, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for p %d\n", status);
        exit(1);
    }
    bufferLAPLACE = 
        clCreateBuffer(clEnv.context, CL_MEM_WRITE_ONLY, imgBufferLength, NULL, &status);
    if(status != CL_SUCCESS){
        printf ("Unable to create buffer for laplace\n");
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
    clEnv.kernel = clCreateKernel(clEnv.program, KERNEL_FUNCTION_NAME, &status);
    if (status != CL_SUCCESS){
        printf ("clCreateKernel: Unable to set a kernel from program! Status=%d \n", status);
        exit(1);
    }
    //-------------------------------------------------------
    // End OPENCL Initialize
    //-------------------------------------------------------
}
#endif

#ifdef OPENCL
void fd_init(int order, int nx, int nz, float dx, float dz, float dt, float **p, char * kernelPath)
#else 
void fd_init(int order, int nx, int nz, float dx, float dz, float dt)
#endif
{
    int io;
    dx2inv = (1./dx)*(1./dx);
    dz2inv = (1./dz)*(1./dz);
    dt2 = dt*dt;

    coefs = calc_coefs(order);
    laplace = alloc2float(nz,nx);

    coefs_z = calc_coefs(order);
    coefs_x = calc_coefs(order);

    // pre calc coefs 8 d2 inv
    for (io = 0; io <= order; io++) {
        coefs_z[io] = dz2inv * coefs[io];
        coefs_x[io] = dx2inv * coefs[io];
    }

    
    memset(*laplace,0,nz*nx*sizeof(float));
#ifdef OPENCL
    fd_init_opencl(nx, nz, kernelPath); 
#endif
        
#ifdef PERF_COUNTERS
    avgSerial = 4306.42;
    avgDevice = 0;
    avgWriteCL = 0;
    avgReadCL = 0;
    avgKernel = 0;
    laplacian_counter = 0;
#endif  
    NX = nx;
    NZ = nz;
    ORDER = order;
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
        avgSerial = 0;
        avgDevice = 0;
        avgWriteCL = 0;
        avgReadCL = 0;
        avgKernel = 0;
        laplacian_counter = 0;
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
        clReleaseKernel         (clEnv.kernel);
        clReleaseProgram        (clEnv.program);
        clReleaseCommandQueue   (clEnv.cmdQueue);
        clReleaseContext        (clEnv.context);
        
        clReleaseMemObject      (bufferP); 
        clReleaseMemObject      (bufferLAPLACE);
    #endif

    return;
}

void fd_step(float **p, float **pp, float **v2){

    int ix,iz, elapsed;
    cl_int status;
    struct timeval st, et, stCR, etCR, stCW, etCW, stK, etK;

    //-----------------------------------------------------
    // STEP 6: Write data for the devices buffers
    //----------------------------------------------------- 
  
    //Start total time 
    gettimeofday(&st, NULL);
    
    // Start write time
    gettimeofday(&stCW, NULL);
    status = clEnqueueWriteBuffer(clEnv.cmdQueue, bufferP, CL_TRUE, 
        0,imgBufferLength, p[0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to copy P to buffer\n");
        exit(1);
    }
   
    //-----------------------------------------------------
    // STEP 9: Set the kernel arguments
    //  Associate the input and output buffers 
    //-----------------------------------------------------     
    status = clSetKernelArg(clEnv.kernel, 0, sizeof(cl_mem), &bufferP);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 2 kernel argument\n");
        exit(1);
    } 

    status = clSetKernelArg(clEnv.kernel, 1, sizeof(cl_mem), &bufferLAPLACE);
    if (status != CL_SUCCESS) {
        printf ("Unable to set 6 kernel argument\n");
        exit(1);
    }

    // Calc avg write time  
    gettimeofday(&etCW, NULL);
    elapsed = ((etCW.tv_sec - stCW.tv_sec) * 1000000) + (etCW.tv_usec - stCW.tv_usec);
    avgWriteCL += (elapsed*1.0);

    //-----------------------------------------------------
    // STEP 11: Enqueue the kernel for execution
    //----------------------------------------------------- 
    
    // start Kernel time  
    gettimeofday(&stK, NULL);

#if defined(USE_GPU) || defined(USE_CPU)
    status = clEnqueueNDRangeKernel(clEnv.cmdQueue, clEnv.kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to run the kernel on NDRange %d\n", status);
        exit(1);
    } 
#else
    cl_event kernel_event; 
    clEnqueueTask(clEnv.cmdQueue, clEnv.kernel, 0, NULL, kernel_event); 
#endif
    clFinish(clEnv.cmdQueue); 
    // Calc avg write time  
    gettimeofday(&etK, NULL);
    elapsed = ((etK.tv_sec - stK.tv_sec) * 1000000) + (etK.tv_usec - stK.tv_usec);
    avgKernel += (elapsed*1.0);

    //-----------------------------------------------------
    // STEP 12: Read the output buffer back to the host
    //----------------------------------------------------- 
    // start read time 
    gettimeofday(&stCR, NULL);
    status = clEnqueueReadBuffer(clEnv.cmdQueue, bufferLAPLACE, CL_TRUE, 0, imgBufferLength, laplace[0], 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf ("Unable to read the pp buffer\n");
        exit(1);
    }
    // Calc avg read time  
    gettimeofday(&etCR, NULL);
    elapsed = ((etCR.tv_sec - stCR.tv_sec) * 1000000) + (etCR.tv_usec - stCR.tv_usec);
    avgReadCL += (elapsed*1.0);
    
    
    // Calc avg total time  
    gettimeofday(&et, NULL);
    elapsed = ((et.tv_sec - st.tv_sec) * 1000000) + (et.tv_usec - st.tv_usec);
    avgDevice += (elapsed*1.0);
    laplacian_counter++;

    int div = ORDER/2;
    for(ix=0;ix<NX;ix++){
        for(iz=0;iz<NZ;iz++){
            if(ix>=div && iz >= div && ix < NX-div && iz < NZ-div)
                pp[ix][iz] = 2.*p[ix][iz] - pp[ix][iz] + v2[ix][iz]*dt2*laplace[ix][iz];
            else
                pp[ix][iz] = 2.*p[ix][iz] - pp[ix][iz] + 0.f;
        }
    }
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
void fd_print_report(int nx, int nz) {

    float total_bytes, rd_perc, wr_perc, calc_perc, speedup;
    avgSerial = avgSerial ;/// (laplacian_counter * 1.0);
    avgDevice = avgDevice / (laplacian_counter * 1.0);
    avgKernel = avgKernel / (laplacian_counter * 1.0);
    avgWriteCL = avgWriteCL / (laplacian_counter * 1.0);
    avgReadCL = avgReadCL / (laplacian_counter * 1.0);

    total_bytes = (nx * nz * sizeof(float)) * 1.0 / 1000.0;
    rd_perc = (avgReadCL / avgDevice) * 100.0;
    wr_perc = (avgWriteCL / avgDevice) * 100.0;
    calc_perc = (avgKernel / avgDevice) * 100.0;
    speedup = avgSerial / avgDevice;

    printf("\n**************************************************\n");
    printf("  Size:                 nx=%d, nz=%d  \n", nx, nz);
    printf("  Total Interactions:   %d \n", laplacian_counter);
    printf("  Bytes/Interaction:    %.3f KB \n", total_bytes);
    printf("  Serial                %.6f  \n", avgSerial);
    printf("  Device                %.6f  \n", avgDevice);
    printf("  Send Data             %.6f us (%03.2f %%) \n", avgWriteCL,
            wr_perc);
    printf("  Read Data             %.6f us (%03.2f %%) \n", avgReadCL,
            rd_perc);
    printf("  Kernel                %.6f us (%03.2f %%) \n", avgKernel, calc_perc);
    printf("  Speedup               %.2f \n", speedup);
    printf("**************************************************\n");

}
#endif // UPDATE_PP