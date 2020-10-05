#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <string.h>

void fd_init(int order, int nx, int nz, float dx, float dz);
void fd_init_cuda(int order, int nxe, int nze);
float *calc_coefs(int order);
static void makeo2 (float *coef,int order);
void read_input(char *file);
void *alloc1 (size_t n1, size_t size);
void **alloc2 (size_t n1, size_t n2, size_t size);
float *alloc1float(size_t n1);
float **alloc2float(size_t n1, size_t n2);
void free1 (void *p);
void free2 (void **p);
void free1float(float *p);
void free2float(float **p);

#define sizeblock 32
#define PI (3.141592653589793)

/* file names */
char *tmpdir = NULL, *vpfile = NULL, *datfile = NULL, *vel_ext_file = NULL, file[100];

float *d_p;
float *d_laplace, *d_coefs_x, *d_coefs_z;

size_t mtxBufferLength, coefsBufferLength;

int gridx, gridz;
/* size */
int nz, nx, nt;
float dz, dx, dt;

/* adquisition geometry */
int ns = -1, sz = -1, fsx = -1, ds = -1, gz = -1;

/* boundary */
int nxb = -1, nzb = -1, nxe, nze;
float fac = -1.0;

/* propagation */
int order = -1; 
float fpeak;

/* arrays */
int *sx;

/*aux*/
int iss = -1, rnd, vel_ext_flag=0;

static float dx2inv, dz2inv;
static float *coefs = NULL;
static float *coefs_z = NULL;
static float *coefs_x = NULL;

void read_input(char *file)
{
        FILE *fp;
        fp = fopen(file, "r");
        char *line = NULL;
        size_t len = 0;
        if (fp == NULL)
                exit(EXIT_FAILURE);
        while (getline(&line, &len, fp) != -1) {
                if(strstr(line,"tmpdir") != NULL)
                {
                        char *tok;
                        tok = strtok(line, "=");
                        tok = strtok(NULL,"=");
                        tok[strlen(tok) - 1] = '\0';
                        tmpdir = strdup(tok);
                }
                if(strstr(line,"datfile") != NULL)
                {
                        char *tok;
                        tok = strtok(line, "=");
                        tok = strtok(NULL,"=");
                        tok[strlen(tok) - 1] = '\0';
                        datfile = strdup(tok);
                }
                if(strstr(line,"vpfile") != NULL)
                {
                        char *tok;
                        tok = strtok(line, "=");
                        tok = strtok(NULL,"=");
                        tok[strlen(tok) - 1] = '\0';
                        vpfile = strdup(tok);
                }
                if(strstr(line,"vel_ext_file") != NULL)
                {
                        char *tok;
                        tok = strtok(line, "=");
                        tok = strtok(NULL,"=");
                        tok[strlen(tok) - 1] = '\0';
                        vel_ext_file = strdup(tok);
                        vel_ext_flag = 1;
                }
                if(strstr(line,"fpeak") != NULL)
                {
                        char *fpeak_char;
                        fpeak_char = strtok(line, "=");
                        fpeak_char = strtok(NULL,"=");
                        fpeak = atof(fpeak_char);
                }
                if(strstr(line,"nt") != NULL)
                {
                        char *nt_char;
                        nt_char = strtok(line, "=");
                        nt_char = strtok(NULL,"=");
                        nt = atoi(nt_char);
                }
                if(strstr(line,"dt") != NULL)
                {
                        char *dt_char;
                        dt_char = strtok(line, "=");
                        dt_char = strtok(NULL,"=");
                        dt = atof(dt_char);
                }
                if(strstr(line,"ns") != NULL)
                {
                        char *ns_char;
                        ns_char = strtok(line, "=");
                        ns_char = strtok(NULL,"=");
                        ns = atoi(ns_char);
                }
                if(strstr(line,"iss") != NULL)
                {
                        char *iss_char;
                        iss_char = strtok(line, "=");
                        iss_char = strtok(NULL,"=");
                        iss = atoi(iss_char);
                }
                if(strstr(line,"sz") != NULL)
                {
                        char *sz_char;
                        sz_char = strtok(line, "=");
                        sz_char = strtok(NULL,"=");
                        sz = atoi(sz_char);
                }
                if(strstr(line,"fsx") != NULL)
                {
                        char *fsx_char;
                        fsx_char = strtok(line, "=");
                        fsx_char = strtok(NULL,"=");
                        fsx = atoi(fsx_char);
                }
                if(strstr(line,"ds") != NULL)
                {
                        char *ds_char;
                        ds_char = strtok(line, "=");
                        ds_char = strtok(NULL,"=");
                        ds = atoi(ds_char);
                }
                if(strstr(line,"gz") != NULL)
                {
                        char *gz_char;
                        gz_char = strtok(line, "=");
                        gz_char = strtok(NULL,"=");
                        gz = atoi(gz_char);
                }
                if(strstr(line,"nzb") != NULL)
                {
                        char *nzb_char;
                        nzb_char = strtok(line, "=");
                        nzb_char = strtok(NULL,"=");
                        nzb = atoi(nzb_char);
                }
                if(strstr(line,"nxb") != NULL)
                {
                        char *nxb_char;
                        nxb_char = strtok(line, "=");
                        nxb_char = strtok(NULL,"=");
                        nxb = atoi(nxb_char);
                }
                if(strstr(line,"rnd") != NULL)
                {
                        char *rnd_char;
                        rnd_char = strtok(line, "=");
                        rnd_char = strtok(NULL,"=");
                        rnd = atoi(rnd_char);
                }
                if(strstr(line,"nz") != NULL)
                {
                        char *nz_char;
                        nz_char = strtok(line, "=");
                        if (strlen(nz_char) <= 2)
                        {
                                nz_char = strtok(NULL,"=");
                                nz = atoi(nz_char);
                        }
                }
                if(strstr(line,"nx") != NULL)
                {
                        char *nx_char;
                        nx_char = strtok(line, "=");
                        if (strlen(nx_char) <= 2)
                        {
                                nx_char = strtok(NULL,"=");
                                nx = atoi(nx_char);
                        }
                }
                if(strstr(line,"dz") != NULL)
                {
                        char *dz_char;
                        dz_char = strtok(line, "=");
                        dz_char = strtok(NULL,"=");
                        dz = atof(dz_char);
                }
                if(strstr(line,"dx") != NULL)
                {
                        char *dx_char;
                        dx_char = strtok(line, "=");
                        dx_char = strtok(NULL,"=");
                        dx = atof(dx_char);
                }
                if(strstr(line,"fac") != NULL)
                {
                        char *fac_char;
                        fac_char = strtok(line, "=");
                        fac_char = strtok(NULL,"=");
                        fac = atof(fac_char);
                }
                if(strstr(line,"order") != NULL)
                {
                        char *order_char;
                        order_char = strtok(line, "=");
                        order_char = strtok(NULL,"=");
                        order = atoi(order_char);
                }
        }
	if(iss == -1 ) iss = 0;	 	// save snaps of this source
	if(ns == -1) ns = 1;	 	// number of sources
	if(sz == -1) sz = 0; 		// source depth
	if(fsx == -1) fsx = 0; 	// first source position
	if(ds == -1) ds = 1; 		// source interval
	if(gz == -1) gz = 0; 		// receivor depth
	if(order == -1) order = 8;	// FD order
	if(nzb == -1) nzb = 40;		// z border size
	if(nxb == -1) nxb = 40;		// x border size
	if(fac == -1.0) fac = 0.7;	
}

__global__ void kernel_lap(int order, int nx, int nz, float * __restrict__ p, float * __restrict__ lap, float * __restrict__ coefsx, float * __restrict__ coefsz)
{
        int half_order=order/2;
        int i =  half_order + blockIdx.x * blockDim.x + threadIdx.x; // Global row index
        int j =  half_order + blockIdx.y * blockDim.y + threadIdx.y; // Global column index
        int mult = i*nz;
        int aux;
        float acmx = 0, acmz = 0;

        if(i<nx - half_order)
        {
                if(j<nz - half_order)
                {
                        for(int io=0;io<=order;io++)
                        {
                                aux = io-half_order;
                                acmz += p[mult + j+aux]*coefsz[io];
                                acmx += p[(i+aux)*nz + j]*coefsx[io];
                        }
                        lap[mult +j] = acmz + acmx;
                        acmx = 0.0;
                        acmz = 0.0;
                }
        }

}

float *calc_coefs(int order)
{
        float *coef;

        coef = (float *)calloc(order+1,sizeof(float));
        switch(order)
        {
                case 2:
                        coef[0] = 1.;
                        coef[1] = -2.;
                        coef[2] = 1.;
                        break;
                case 4:
                        coef[0] = -1./12.;
                        coef[1] = 4./3.;
                        coef[2] = -5./2.;
                        coef[3] = 4./3.;
                        coef[4] = -1./12.;
                        break;
                case 6:
                        coef[0] = 1./90.;
                        coef[1] = -3./20.;
                        coef[2] = 3./2.;
                        coef[3] = -49./18.;
                        coef[4] = 3./2.;
                        coef[5] = -3./20.;
                        coef[6] = 1./90.;
                        break;
                case 8:

                        coef[0] = -1./560.;
                        coef[1] = 8./315.;
                        coef[2] = -1./5.;
                        coef[3] = 8./5.;
                        coef[4] = -205./72.;
                        coef[5] = 8./5.;
                        coef[6] = -1./5.;
                        coef[7] = 8./315.;
                        coef[8] = -1./560.;
                        break;
                default:
                        makeo2(coef,order);
        }

        return coef;
}

static void makeo2 (float *coef,int order)
{
        float h_beta, alpha1=0.0;
        float alpha2=0.0;
        float  central_term=0.0;
        float coef_filt=0;
        float arg=0.0;
        float  coef_wind=0.0;
        int msign,ix;

        float alpha = .54;
        float beta = 6.;
        h_beta = 0.5*beta;
        alpha1=2.*alpha-1.0;
        alpha2=2.*(1.0-alpha);
        central_term=0.0;

        msign=-1;

        for (ix=1; ix <= order/2; ix++){
                msign=-msign ;
                coef_filt = (2.*msign)/(ix*ix);
                arg = PI*ix/(2.*(order/2+2));
                coef_wind=pow((alpha1+alpha2*cos(arg)*cos(arg)),h_beta);
                coef[order/2+ix] = coef_filt*coef_wind;
                central_term = central_term + coef[order/2+ix];
                coef[order/2-ix] = coef[order/2+ix];
        }

        coef[order/2]  = -2.*central_term;

        return;
}

void *alloc1 (size_t n1, size_t size)
{
	void *p;

	if ((p=malloc(n1*size))==NULL)
		return NULL;
	return p;
}

void **alloc2 (size_t n1, size_t n2, size_t size)
{
	size_t i2;
	void **p;

	if ((p=(void**)malloc(n2*sizeof(void*)))==NULL) 
		return NULL;
	if ((p[0]=(void*)malloc(n2*n1*size))==NULL) {
		free(p);
		return NULL;
	}
	for (i2=0; i2<n2; i2++)
		p[i2] = (char*)p[0]+size*n1*i2;
	return p;
}

float *alloc1float(size_t n1)
{
	return (float*)alloc1(n1,sizeof(float));
}

float **alloc2float(size_t n1, size_t n2)
{
	return (float**)alloc2(n1,n2,sizeof(float));
}

void free1 (void *p)
{
	free(p);
}

void free2 (void **p)
{
	free(p[0]);
	free(p);
}

void free1float(float *p)
{
	free1(p);
}

void free2float(float **p)
{
	free2((void**)p);
}

void fd_init_cuda(int order, int nxe, int nze)
{
        mtxBufferLength = (nxe*nze)*sizeof(float);
        coefsBufferLength = (order+1)*sizeof(float);

        // Create a Device pointers
        cudaMalloc(&d_p, mtxBufferLength);
        cudaMalloc(&d_laplace, mtxBufferLength);
        cudaMalloc(&d_coefs_x, coefsBufferLength);
        cudaMalloc(&d_coefs_z, coefsBufferLength);

        int div_x, div_z;
        // Set a Grid for the execution on the device
        int tx = ((nxe - 1) / 32 + 1) * 32;
        int tz = ((nze - 1) / 32 + 1) * 32;

        div_x = (float) tx/(float) sizeblock;
        div_z = (float) tz/(float) sizeblock;

        gridx = (int) ceil(div_x);
        gridz = (int) ceil(div_z);
}

void fd_init(int order, int nx, int nz, float dx, float dz)
{
        int io;
        dx2inv = (1./dx)*(1./dx);
        dz2inv = (1./dz)*(1./dz);

        coefs = calc_coefs(order);

        coefs_z = calc_coefs(order);
        coefs_x = calc_coefs(order);

        // pre calc coefs 8 d2 inv
        for (io = 0; io <= order; io++)
        {
                coefs_z[io] = dz2inv * coefs[io];
                coefs_x[io] = dx2inv * coefs[io];
        }

        fd_init_cuda(order,nx,nz);

        return;
}

int main (int argc, char **argv)
{
        /* model file and data pointers */
	FILE *fsource = NULL, *fvel_ext = NULL, *fd_obs = NULL, *fvp = NULL, *fsns = NULL,*fsns2 = NULL, *fsnr = NULL, *fimg = NULL, *flim = NULL, *fimg_lap = NULL;

	/* iteration variables */
	int iz, ix, it, is;

	/* arrays */
	float *srce;
	float **vp = NULL, **vpe = NULL, **vpex = NULL;

	/* propagation variables */
	float **PP,**P,**PPR,**PR,**tmp;
	float ***swf, ***upb, ***snaps, **vel2, ***d_obs, ***vel_ext_rnd;
	float **imloc, **img, **img_lap;
        read_input(argv[1]);

        printf("## vp = %s, d_obs = %s, vel_ext_file = %s, vel_ext_flag = %d \n",vpfile,datfile,vel_ext_file,vel_ext_flag);
	printf("## nz = %d, nx = %d, nt = %d \n",nz,nx,nt);
	printf("## dz = %f, dx = %f, dt = %f \n",dz,dx,dt);
	printf("## ns = %d, sz = %d, fsx = %d, ds = %d, gz = %d \n",ns,sz,fsx,ds,gz);
	printf("## order = %d, nzb = %d, nxb = %d, F = %f, rnd = %d \n",order,nzb,nxb,fac,rnd);

        // nxe = nx + 2 * nxb;
        // nze = nz + 2 * nzb;
        // // inicialização
        // fd_init(order,nxe,nze,dx,dz);

        // dim3 dimGrid(gridx, gridz);
        // dim3 dimBlock(sizeblock, sizeblock);
        // FILE *finput;
        // float *input_data;

        // if((finput = fopen(path_file, "rb")) == NULL)
        //         printf("Unable to open file!\n");
        // else
        //         printf("Opened input successfully for read.\n");
        
        // input_data = (float*)malloc(mtxBufferLength);
        // if(!input_data)
        //         printf("input Memory allocation error!\n");
        // else 
        //         printf("input Memory allocation successful.\n");

        // memset(input_data, 0, mtxBufferLength);
        
        // if( fread(input_data, sizeof(float), nze*nxe, finput) != nze*nxe)
        //         printf("input Read error!\n");
        
        // else 
        //         printf("input Read was successful.\n");
        // fclose(finput);

        // // utilização do kernel
        // cudaMemcpy(d_p, input_data, mtxBufferLength, cudaMemcpyHostToDevice);
        // cudaMemcpy(d_coefs_x, coefs_x, coefsBufferLength, cudaMemcpyHostToDevice);
        // cudaMemcpy(d_coefs_z, coefs_z, coefsBufferLength, cudaMemcpyHostToDevice);

        // kernel_lap<<<dimGrid, dimBlock>>>(order,nxe,nze,d_p,d_laplace,d_coefs_x,d_coefs_z);

        // float *output_data;
        // output_data = (float*)malloc(mtxBufferLength);
        // if(!output_data)
        //         printf("output Memory allocation error!\n");
        // else 
        //         printf("output Memory allocation successful.\n");
        // memset(output_data, 0, mtxBufferLength);
        // cudaMemcpy(output_data, d_laplace, mtxBufferLength, cudaMemcpyDeviceToHost);

        // // salvando a saída
        // FILE *foutput;
        // if((foutput = fopen("output_cuda.bin", "wb")) == NULL)
        //         printf("Unable to open file!\n");
        // else
        //         printf("Opened output successfully for write.\n");
        
        // if( fwrite(output_data, sizeof(float), nze*nxe, foutput) != nze*nxe)
        //         printf("output Write error!\n");
        
        // else 
        //         printf("output Write was successful.\n");
        // fclose(foutput);

        // // free memory device
        // free(input_data);
        // free(output_data);
        // cudaFree(d_p);
        // cudaFree(d_laplace);
        // cudaFree(d_coefs_x);
        // cudaFree(d_coefs_z);
        return 0;
}

//==============================================================================================================================================
/* Acoustic wavefield modeling using finite-difference method
Leonardo Gómez Bernal, Salvador BA, Brazil
August, 2016 */

// #include <stdio.h>
// #include <cuda.h>
// #include <time.h>
// #include "fd.h"
// #include <sys/time.h>
// #include <cuda_runtime.h>
// #include <cuda_profiler_api.h>
// extern "C" {
// 	#include "cwp.h"
// 	#include "su.h"
// 	#include "ptsrc.h"
// 	#include "taper.h"
// }

// char *sdoc[] = {	/* self documentation */
// 	" Seismic modeling using acoustic wave equation ",
// 	"				               ",
// 	NULL};
// /* global variables */


// /* prototypes */

// int main (int argc, char **argv){
//   cudaProfilerStart();
// 		struct timeval st, et;
//     int elapsed;
// 	float execTime;
// 	clock_t begin, end;
// 	long int time_spent;
//  	begin = clock();
// 	gettimeofday(&st, NULL);
	
	

// 	/* initialization admiting self documentation */
// 	initargs(argc, argv);
// 	requestdoc(1);

// 	/* read parameters */
// 	MUSTGETPARSTRING("tmpdir",&tmpdir);		// directory for data
// 	MUSTGETPARSTRING("vpfile",&vpfile);		// vp model
// 	MUSTGETPARSTRING("datfile",&datfile);	// observed data (seismogram)
// 	MUSTGETPARINT("nz",&nz); 				// number of samples in z
// 	MUSTGETPARINT("nx",&nx); 				// number of samples in x
// 	MUSTGETPARINT("nt",&nt); 				// number of time steps
// 	MUSTGETPARFLOAT("dz",&dz); 				// sampling interval in z
// 	MUSTGETPARFLOAT("dx",&dx); 				// sampling interval in x
// 	MUSTGETPARFLOAT("dt",&dt); 				// sampling interval in t
// 	MUSTGETPARFLOAT("fpeak",&fpeak); 		// souce peak frequency

// 	if(getparstring("vel_ext_file",&vel_ext_file)) vel_ext_flag = 1;
// 	if(!getparint("iss",&iss)) iss = 0;	 	// save snaps of this source
// 	if(!getparint("ns",&ns)) ns = 1;	 	// number of sources
// 	if(!getparint("sz",&sz)) sz = 0; 		// source depth
// 	if(!getparint("fsx",&fsx)) fsx = 0; 	// first source position
// 	if(!getparint("ds",&ds)) ds = 1; 		// source interval
// 	if(!getparint("gz",&gz)) gz = 0; 		// receivor depth

// 	if(!getparint("order",&order)) order = 8;	// FD order
// 	if(!getparint("nzb",&nzb)) nzb = 40;		// z border size
// 	if(!getparint("nxb",&nxb)) nxb = 40;		// x border size
// 	if(!getparfloat("fac",&fac)) fac = 0.7;		// damping factor
// 	// if(!getparint("rnd",&rnd)) rnd = 1;		    // random vel. border

// 	fprintf(stdout,"## vp = %s, d_obs = %s, vel_ext_file = %s, vel_ext_flag = %d \n",vpfile,datfile,vel_ext_file,vel_ext_flag);
// 	fprintf(stdout,"## nz = %d, nx = %d, nt = %d \n",nz,nx,nt);
// 	fprintf(stdout,"## dz = %f, dx = %f, dt = %f \n",dz,dx,dt);
// 	fprintf(stdout,"## ns = %d, sz = %d, fsx = %d, ds = %d, gz = %d \n",ns,sz,fsx,ds,gz);
// 	fprintf(stdout,"## order = %d, nzb = %d, nxb = %d, F = %f, rnd = %d \n",order,nzb,nxb,fac,rnd);
// 	/* create source vector  */
// 	srce = alloc1float(nt);
// 	ricker_wavelet(nt, dt, fpeak, srce);
// 	sx = alloc1int(ns);
// 	for(is=0; is<ns; is++){
// 		sx[is] = fsx + is*ds + nxb;
// 	}
// 	sz += nzb;
// 	gz += nzb;
// 	/* add boundary to models */
// 	nze = nz + 2 * nzb;
// 	nxe = nx + 2 * nxb;
// 	/*read randomic vel. models (per source) */
// 	if(vel_ext_flag){
// 		vel_ext_rnd = alloc3float(nze,nxe,ns);
// 		memset(**vel_ext_rnd,0,nze*nxe*ns*sizeof(float));
// 		fvel_ext = fopen(vel_ext_file,"r");
// 		fread(**vel_ext_rnd,sizeof(float),nze*nxe*ns,fvel_ext);
// 		fclose(fvel_ext);
// 	}

// 	/*read observed data (seism.) */
// 	d_obs = alloc3float(nt,nx,ns);
// 	memset(**d_obs,0,nt*nx*ns*sizeof(float));
// 	fd_obs = fopen(datfile,"r");
// 	fread(**d_obs,sizeof(float),nt*nx*ns,fd_obs);
// 	fclose(fd_obs);

// 	float **d_obs_aux=(float**)malloc(ns*sizeof(float*));
// 	for(int i=0; i<ns; i++) 
// 		d_obs_aux[i] = (float*)malloc((nt*nx)*sizeof(float)); 
	
// 	for(int i=0; i<ns; i++){
// 		for(int j=0; j<nx; j++){
// 			for(int k=0; k<nt; k++)
// 				d_obs_aux[i][j*nt+k] = d_obs[i][j][k]; 
// 		}
// 	}

// 	/* read parameter models */
// 	vp = alloc2float(nz,nx);
// 	memset(*vp,0,nz*nx*sizeof(float));
// 	fvp = fopen(vpfile,"r");
// 	fread(vp[0],sizeof(float),nz*nx,fvp);
// 	fclose(fvp);

// 	/* vp size estended to vpe */
// 	vpe = alloc2float(nze,nxe);
// 	vpex = vpe;

// 	for(ix=0; ix<nx; ix++){
// 		for(iz=0; iz<nz; iz++){
// 			vpe[ix+nxb][iz+nzb] = vp[ix][iz]; 
// 		}
// 	}

// 	/* allocate vel2 for vpe^2 */
// 	vel2 = alloc2float(nze,nxe);

// 	/* initialize wave propagation */
// 	fd_init(order,nxe,nze,nxb,nzb,nt,ns,fac,dx,dz,dt);
// 	taper_init(nxb,nzb,fac);

// 	PP = alloc2float(nze,nxe);
// 	P = alloc2float(nze,nxe);
// 	PPR = alloc2float(nze,nxe);
// 	PR = alloc2float(nze,nxe);
// 	upb = alloc3float(order/2,nxe,nt);
// 	// swf = alloc3float(nz,nx,nt);
// 	snaps = alloc3float(nze,nxe,2);
// 	imloc = alloc2float(nz,nx);
// 	img = alloc2float(nz,nx);
// 	img_lap = alloc2float(nz,nx);

// 	// fsns = fopen("output/dir.snaps","w");
// 	// fsns2 = fopen("output/dir.snaps_rec","w");
// 	// fsnr = fopen("output/dir.snapr","w");
// 	// fimg = fopen("output/dir.image","w");	
// 	// fimg_lap = fopen("output/dir.image_lap","w");

// 	char filepath [100];
// 	sprintf(filepath, "%s/dir.snaps", tmpdir);
// 	fsns = fopen(filepath,"w");
// 	sprintf(filepath, "%s/dir.snaps_rec", tmpdir);
// 	fsns2 = fopen(filepath,"w");
// 	sprintf(filepath, "%s/dir.snapr", tmpdir);
// 	fsnr = fopen(filepath,"w");
// 	sprintf(filepath, "%s/dir.image", tmpdir);
// 	fimg = fopen(filepath,"w");
// 	sprintf(filepath, "%s/dir.image_lap", tmpdir);	
// 	fimg_lap = fopen(filepath,"w");
	
// 	memset(*img,0,nz*nx*sizeof(float));
// 	memset(*img_lap,0,nz*nx*sizeof(float));

// 	for(is=0; is<ns; is++){
// 		fprintf(stdout,"** source %d, at (%d,%d) \n",is+1,sx[is]-nxb,sz-nzb);
// 		/* Calc (or load) velocity model border */
// 		if (vel_ext_flag){
// 			vpe = vel_ext_rnd[is];					// load hybrid border vpe from file
// 		}else{
// 			extendvel_linear(nx,nz,nxb,nzb,vpe); 	// hybrid border (linear randomic)
// 		}

// 		/* vel2 = vpe^2 */
// 		for(ix=0; ix<nx+2*nxb; ix++){
// 			for(iz=0; iz<nz+2*nzb; iz++){
// 				vel2[ix][iz] = vpe[ix][iz]*vpe[ix][iz];
// 			}
// 		}

// 		memset(*PP,0,nze*nxe*sizeof(float));
// 		memset(*P,0,nze*nxe*sizeof(float));
    
// 		cudaProfilerStart();
// 		fd_forward(order,P,PP,vel2,upb,nze,nxe,nt,is,sz,sx,srce, is);
// 		fprintf(stdout,"\n");

// 		for(iz=0; iz<nze; iz++){
// 			for(ix=0; ix<nxe; ix++){
// 				snaps[0][ix][iz] = P[ix][iz];
// 				snaps[1][ix][iz] = PP[ix][iz];
// 			}
// 		}

// 		fprintf(stdout,"** backward propagation %d, at (%d,%d) \n",is+1,sx[is]-nxb,sz-nzb);

// 		memset(*PP,0,nze*nxe*sizeof(float));
// 		memset(*P,0,nze*nxe*sizeof(float));
// 		memset(*PPR,0,nze*nxe*sizeof(float));
// 		memset(*PR,0,nze*nxe*sizeof(float));
// 		memset(*imloc,0,nz*nx*sizeof(float));

// 		/* Reverse propagation */
// 		fd_back(order,P,PP,PR,PPR,vel2,upb,nze,nxe,nt,is,sz,gz,snaps,imloc,d_obs_aux);
// 		fprintf(stdout,"\n");
//     cudaProfilerStop();
		
// 		/* stack migrated images */
// 		for(iz=0; iz<nz; iz++){
// 			for(ix=0; ix<nx; ix++){
// 				img[ix][iz] += imloc[ix][iz];
// 			}
// 		}
// 	}
	
// 	cudaProfilerStop();
// 	// cudaDeviceReset();
// #ifdef  PERF_COUNTERS
// 	fd_print_report(nxe, nze);
// 	gettimeofday(&et, NULL);
//    	elapsed = ((et.tv_sec - st.tv_sec) * 1000000) + (et.tv_usec - st.tv_usec);
//    	execTime += (elapsed*1.0);
//    	printf("> Exec Time    = %.1f (s)\n",execTime/1000000.0);
// 	printf("> ================================================ \n\n");
// #endif
// 	fwrite(*img,sizeof(float),nz*nx,fimg);

// 	fwrite(*img_lap,sizeof(float),nz*nx,fimg_lap);

// 	fclose(fsns);
// 	fclose(fsns2);
// 	fclose(fsnr);
// 	fclose(fimg);
// 	fclose(fimg_lap);

//     /* release memory */
//   fd_destroy();
// 	taper_destroy();
// 	free1int(sx);
// 	free1float(srce);
// 	free2float(vp);
// 	free2float(P);
// 	free2float(PP);
// 	free2float(PR);
// 	free2float(PPR);
// 	// free3float(swf);
// 	free3float(snaps);
// 	free2float(imloc);
// 	free2float(img);
// 	free2float(img_lap);
// 	free2float(vpex);
// 	free2float(vel2);
// 	free3float(upb);
// 	free3float(d_obs);
// 	if(vel_ext_flag) free3float(vel_ext_rnd);
// 	end = clock();
// 	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
// 	return(CWP_Exit());
// }
