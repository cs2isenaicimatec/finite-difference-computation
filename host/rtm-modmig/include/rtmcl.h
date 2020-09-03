#ifdef OPENCLRTM
#ifndef CLMISC_H
#define CLMISC_H

#include <CL/cl.h> 
#define PERF_COUNTERS
#define OPENCL

#define BSIZE 32
#define PLATFORM_ID 0
//#define USE_FPGA 0

// #ifdef USE_FPGA
// #define PLATFORM_ID 0
// #elif  USE_CPU
// #define PLATFORM_ID 1
// #elif  DEFAULT
// #define PLATFORM_ID 0
// #else
// #define PLATFORM_ID 0
// #endif

#include "rtm.h"

#define RTMCL_FORWARD_NAME  "rtmforward"
#define RTMCL_BACKWARD_NAME "rtmbackward"


char 			* 	clmiscReadFileStr(char* fileLocation, size_t *source_size);
cl_device_id 	*	clmiscGetPlatformAndDevice(cl_platform_id *platforms, int qualPlat, 
												cl_uint *numDevices); 
void rtmcl_enqueue_kernel(RTMCLEnv * clEnv);
void rtmcl_read_forward_output(RTMShot * shot, RTMExecParam * execParam);
void rtmcl_write_forward_params(RTMShot * shot, RTMExecParam * execParam);
void rtmcl_load_forward_kernel(RTMExecParam * execParam);
void rtmcl_unload_forward_kernel(RTMCLEnv * clEnv);
void rtmcl_init(RTMExecParam * execParam);
void rtmcl_mig_forward(RTMShot * shot, RTMExecParam * execParam);
void rtmcl_mig_backward(RTMShot * shot, RTMExecParam * execParam);
void rtmcl_destroy( RTMCLEnv * clEnv);

#endif // CLMISC_H

#endif // OPENCLRTM
