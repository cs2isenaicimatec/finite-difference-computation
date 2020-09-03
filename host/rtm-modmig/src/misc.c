#include <stdio.h>
#include <sys/time.h>
#include <CL/cl.h>
#include <omp.h>

#include <CL/cl.h> 
#include "rtmcl.h"

// ------- Auxiliar Functions --------- //  
cl_device_id *clmiscGetPlatformAndDevice(cl_platform_id *platforms, int platformId, cl_uint *numDev){
    cl_int status;
    cl_uint numPlatforms = 0;
    cl_device_id *devices;
    //-----------------------------------------------------
    // STEP 1: Discover and initialize Platforms 
    //-----------------------------------------------------
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS) {
        printf ("Unable to query the number of platforms\n");
        exit(1);
    }
    platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
    
    //Preenche as plataformas com  clGetPlatformIDs().
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    
    //-----------------------------------------------------
    // STEP 2: Descobrir e inicializar os devices.
    //----------------------------------------------------- 
    cl_uint numDevices = 0;
    status = clGetDeviceIDs(platforms[platformId], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);

    if (status != CL_SUCCESS) {
        printf ("Unable to query the number of devices\n");
        exit(1);
    }
    devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));

    status = clGetDeviceIDs(platforms[platformId], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
    *numDev = numDevices; 
    return devices; 
}



char *clmiscReadFileStr(char* fileLocation, size_t *source_size){
    FILE *fp;
    char *source_str;
    
    // Load kernel file 
    fp = fopen(fileLocation, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }

    fseek(fp, 0, SEEK_END);
    long fSize = ftell(fp);
    rewind(fp);
    
    source_str = (char*)malloc((fSize+1));
    *source_size = fread( source_str, 1, (fSize+1)*sizeof(char), fp);
    source_str[fSize] = '\0';
    fclose( fp );

    return source_str; 
}

float elapsedTimeUs(struct timeval st, struct timeval et){
    float elapsedTime;
    elapsedTime = ((et.tv_sec - st.tv_sec) * 1000000.0) + (et.tv_usec - st.tv_usec)*1.0;
    return elapsedTime;
}

float elapsedTimeMs(struct timeval st, struct timeval et){
    float elapsed = elapsedTimeUs(st, et)*1.0;
    return elapsed/1000.0;
}

float elapsedTimeS(struct timeval st, struct timeval et){
    float elapsed = elapsedTimeUs(st, et)*1.0;
    return elapsed/1000000.0;
}
//