#ifndef CLMISC_H
#define CLMISC_H

#include <CL/cl.h> 
#define PERF_COUNTERS
#define OPENCL

#define BSIZE 32
#define PLATFORM_ID 0
//#define USE_FPGA 0

#ifdef USE_FPGA
#define PLATFORM_ID 0
#elif  USE_CPU
#define PLATFORM_ID 1
#elif  DEFAULT
#define PLATFORM_ID 0
#else
#define PLATFORM_ID 0
#endif

char 			* 	clmiscReadFileStr(char* fileLocation, size_t *source_size);
cl_device_id 	*	clmiscGetPlatformAndDevice(cl_platform_id *platforms, int qualPlat, 
												cl_uint *numDevices); 

typedef struct clEnvironment {
	cl_kernel kernel_lap;
	cl_kernel kernel_img; 
	cl_kernel kernel_upb;  
	cl_kernel kernel_src;
	cl_kernel kernel_sism;  
	cl_kernel kernel_time; 
	cl_kernel kernel_taper; 
	cl_kernel kernel_upb_r;
	cl_context context; 
	cl_program program;
	cl_command_queue cmdQueue;
}clEnvironment;

#endif
