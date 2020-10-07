#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <string.h>

void fd_init(int order, int nx, int nz, int nxb, int nzb, int nt, int ns, float fac, float dx, float dz, float dt);
void fd_init_cuda(int order, int nxe, int nze, int nxb, int nzb, int nt, int ns, float fac);
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
char *tmpdir = NULL, *vpfile = NULL, *datfile = NULL, *vel_ext_file = NULL;
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

float *d_p, *d_pr, *d_pp, *d_ppr, *d_swap;
float *d_laplace, *d_v2, *d_coefs_x, *d_coefs_z;
float *d_taperx, *d_taperz, *d_upb, *d_sis, *d_img;

size_t mtxBufferLength, brdBufferLength;
size_t imgBufferLength, obsBufferLength;
size_t coefsBufferLength, upbBufferLength;

float *taper_x, *taper_z;
int nxbin, nzbin;

int gridx, gridz, gridupb;
int gridBorder_x, gridBorder_z;

static float dx2inv,dz2inv,dt2;
static float **laplace = NULL;
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
        free(line);
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
// ============================ Kernels ============================
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

__global__ void kernel_lap(int order, int nx, int nz, float * __restrict__ p, float * __restrict__ lap, float * __restrict__ coefsx, float * __restrict__ coefsz)
{

   int half_order=order/2;
  	int i =  half_order + blockIdx.x * blockDim.x + threadIdx.x; // Global row index
  	int j =  half_order + blockIdx.y * blockDim.y + threadIdx.y; // Global column index
  	int mult = i*nz;
  	int aux;
	float acmx = 0, acmz = 0;

  	if(i<nx - half_order){
  		if(j<nz - half_order){
			for(int io=0;io<=order;io++){
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

__global__ void kernel_time(int nx, int nz, float *__restrict__ p, float *__restrict__ pp, float *__restrict__ v2, float *__restrict__ lap, float dt2)
{

  	int i =  blockIdx.x * blockDim.x + threadIdx.x; // Global row index
  	int j =  blockIdx.y * blockDim.y + threadIdx.y; // Global column index
  	int mult = i*nz;

  	if(i<nx){
  		if(j<nz){
			 pp[mult+j] = 2.*p[mult+j] - pp[mult+j] + v2[mult+j]*dt2*lap[mult+j];
		}
  	}
}

__global__ void kernel_tapper(int nx, int nz, int nxb, int nzb, float *__restrict__ p, float *__restrict__ pp, float *__restrict__ taperx, float *__restrict__ taperz)
{

	int i =  blockIdx.x * blockDim.x + threadIdx.x; // nx index
	int j =  blockIdx.y * blockDim.y + threadIdx.y; // nzb index
	int itxr = nx - 1, mult = i*nz;

	if(i<nx){
		if(j<nzb){
			p[mult+j] *= taperz[j];
			pp[mult+j] *= taperz[j];
		}
	}

	if(i<nxb){
		if(j<nzb){
			p[mult+j] *= taperx[i];
			pp[mult+j] *= taperx[i];

			p[(itxr-i)*nz+j] *= taperx[i];
			pp[(itxr-i)*nz+j] *= taperx[i];
		}
	}
}

__global__ void kernel_src(int nz, float * __restrict__ pp, int sx, int sz, float srce)
{
 	pp[sx*nz+sz] += srce;
}

__global__ void kernel_upb(int order, int nx, int nz, int nzb, int nt, float *__restrict__ pp, float *__restrict__ upb, int it, int flag)
{
	int half_order = order/2;
	int i = blockIdx.x * blockDim.x + threadIdx.x; //nx index

 	if(i<nx){
		for(int j=nzb-order/2;j<nzb;j++)
    		if(flag == 0)
    			upb[(it*nx*half_order)+(i*half_order)+(j-(nzb-half_order))] = pp[i*nz+j];
        	else
	        	pp[i*nz+j] = upb[((nt-1-it)*nx*half_order)+(i*half_order)+(j-(nzb-half_order))];
  	}
}

__global__ void kernel_sism(int nx, int nz, int nxb, int nt, int is, int it, int gz, float *__restrict__ d_obs, float *__restrict__ ppr)
{
 	int size = nx-(2*nxb);
	int i = blockIdx.x * blockDim.x + threadIdx.x; //nx index
 	if(i<size)
 		ppr[((i+nxb)*nz) + gz] += d_obs[i*nt + (nt-1-it)];

}

__global__ void kernel_img(int nx, int nz, int nxb, int nzb, float * __restrict__ imloc, float * __restrict__ p, float * __restrict__ ppr)
{
 	int size_x = nx-(2*nxb);
 	int size_z = nz-(2*nzb);
	int i =  blockIdx.x * blockDim.x + threadIdx.x; // Global row index
  	int j =  blockIdx.y * blockDim.y + threadIdx.y; // Global column index
 	if(j<size_z){
      if(i<size_x){
        imloc[i*size_z+j] += p[(i+nxb)*nz+(j+nzb)] * ppr[(i+nxb)*nz+(j+nzb)];
      }
    }
}
// ============================ Aux ============================
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

void ***alloc3 (size_t n1, size_t n2, size_t n3, size_t size)
{
	size_t i3,i2;
	void ***p;

	if ((p=(void***)malloc(n3*sizeof(void**)))==NULL)
		return NULL;
	if ((p[0]=(void**)malloc(n3*n2*sizeof(void*)))==NULL) {
		free(p);
		return NULL;
	}
	if ((p[0][0]=(void*)malloc(n3*n2*n1*size))==NULL) {
		free(p[0]);
		free(p);
		return NULL;
	}

	for (i3=0; i3<n3; i3++) {
		p[i3] = p[0]+n2*i3;
		for (i2=0; i2<n2; i2++)
			p[i3][i2] = (char*)p[0][0]+size*n1*(i2+n2*i3);
	}
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

float ***alloc3float(size_t n1, size_t n2, size_t n3)
{
	return (float***)alloc3(n1,n2,n3,sizeof(float));
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

void free3 (void ***p)
{
	free(p[0][0]);
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

void free3float(float ***p)
{
	free3((void***)p);
}
// ============================ Init ============================
void fd_init_cuda(int order, int nxe, int nze, int nxb, int nzb, int nt, int ns, float fac)
{
        float dfrac;
	// cudaProfilerSart();
   	nxbin=nxb; nzbin=nzb;
   	brdBufferLength = nxb*sizeof(float);
   	mtxBufferLength = (nxe*nze)*sizeof(float);
   	coefsBufferLength = (order+1)*sizeof(float);
   	upbBufferLength = nt*nxe*(order/2)*sizeof(float);
	obsBufferLength = nt*(nxe-(2*nxb))*sizeof(float);
   	imgBufferLength = (nxe-(2*nxb))*(nze-(2*nzb))*sizeof(float);

	taper_x = alloc1float(nxb);
	taper_z = alloc1float(nzb);

	dfrac = sqrt(-log(fac))/(1.*nxb);
	for(int i=0;i<nxb;i++)
	  taper_x[i] = exp(-pow((dfrac*(nxb-i)),2));


	dfrac = sqrt(-log(fac))/(1.*nzb);
	for(int i=0;i<nzb;i++)
	  taper_z[i] = exp(-pow((dfrac*(nzb-i)),2));


	// Create a Device pointers
	cudaMalloc((void **) &d_v2, mtxBufferLength);
	cudaMalloc((void **) &d_p, mtxBufferLength);
	cudaMalloc((void **) &d_pp, mtxBufferLength);
	cudaMalloc((void **) &d_pr, mtxBufferLength);
	cudaMalloc((void **) &d_ppr, mtxBufferLength);
	cudaMalloc((void **) &d_swap, mtxBufferLength);
	cudaMalloc((void **) &d_laplace, mtxBufferLength);

	cudaMalloc((void **) &d_upb, upbBufferLength);
	cudaMalloc((void **) &d_sis, obsBufferLength);
	cudaMalloc((void **) &d_img, imgBufferLength);
	cudaMalloc((void **) &d_coefs_x, coefsBufferLength);
	cudaMalloc((void **) &d_coefs_z, coefsBufferLength);
	cudaMalloc((void **) &d_taperx, brdBufferLength);
	cudaMalloc((void **) &d_taperz, brdBufferLength);

	int div_x, div_z;
	// Set a Grid for the execution on the device
	div_x = (float) nxe/(float) sizeblock;
	div_z = (float) nze/(float) sizeblock;
	gridx = (int) ceil(div_x);
	gridz = (int) ceil(div_z);

	div_x = (float) nxb/(float) sizeblock;
	div_z = (float) nzb/(float) sizeblock;
	gridBorder_x = (int) ceil(div_x);
	gridBorder_z = (int) ceil(div_z);

	div_x = (float) 8/(float) sizeblock;
	gridupb = (int) ceil(div_x);
}

void fd_init(int order, int nx, int nz, int nxb, int nzb, int nt, int ns, float fac, float dx, float dz, float dt)
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

        fd_init_cuda(order,nx,nz,nxb,nzb,nt,ns,fac);

        return;
}

void write_buffers(float **p, float **pp, float **v2, float ***upb, float *taperx, float *taperz, float **d_obs, float **imloc, int is, int flag)
{
    
        if(flag == 0){
                cudaMemcpy(d_p, p[0], mtxBufferLength, cudaMemcpyHostToDevice);
                cudaMemcpy(d_pp, pp[0], mtxBufferLength, cudaMemcpyHostToDevice);
                cudaMemcpy(d_v2, v2[0], mtxBufferLength, cudaMemcpyHostToDevice);
                cudaMemcpy(d_coefs_x, coefs_x, coefsBufferLength, cudaMemcpyHostToDevice);
                cudaMemcpy(d_coefs_z, coefs_z, coefsBufferLength, cudaMemcpyHostToDevice);
                cudaMemcpy(d_taperx, taperx, brdBufferLength, cudaMemcpyHostToDevice);
                cudaMemcpy(d_taperz, taperz, brdBufferLength, cudaMemcpyHostToDevice);
                cudaMemcpy(d_upb, upb[0][0], upbBufferLength, cudaMemcpyHostToDevice);
        }

        if(flag == 1){
                cudaMemcpy(d_pr, p[0], mtxBufferLength, cudaMemcpyHostToDevice);
                cudaMemcpy(d_ppr, pp[0], mtxBufferLength, cudaMemcpyHostToDevice);
                cudaMemcpy(d_sis, d_obs[is], obsBufferLength, cudaMemcpyHostToDevice);
                cudaMemcpy(d_img, imloc[0], imgBufferLength, cudaMemcpyHostToDevice);
        }
}
// ============================ Propagation ============================
void fd_forward(int order, float **p, float **pp, float **v2, float ***upb, int nz, int nx, int nt, int is, int sz, int *sx, float *srce, int propag)
{
 	dim3 dimGrid(gridx, gridz);
  	dim3 dimGridTaper(gridx, gridBorder_z);

  	dim3 dimGridSingle(1,1);
  	dim3 dimGridUpb(gridx,1);

  	dim3 dimBlock(sizeblock, sizeblock);
  	
	write_buffers(p,pp,v2,upb,taper_x, taper_z,NULL, NULL,is,0);
	   	
   	for (int it = 0; it < nt; it++){
	 	d_swap  = d_pp;
	 	d_pp = d_p;
	 	d_p = d_swap;

	 	kernel_tapper<<<dimGridTaper, dimBlock>>>(nx,nz,nxbin,nzbin,d_p,d_pp,d_taperx,d_taperz);
	 	kernel_lap<<<dimGrid, dimBlock>>>(order,nx,nz,d_p,d_laplace,d_coefs_x,d_coefs_z);
	 	kernel_time<<<dimGrid, dimBlock>>>(nx,nz,d_p,d_pp,d_v2,d_laplace,dt2);
	 	kernel_src<<<dimGridSingle, dimBlock>>>(nz,d_pp,sx[is],sz,srce[it]);
	 	kernel_upb<<<dimGridUpb, dimBlock>>>(order,nx,nz,nzbin,nt,d_pp,d_upb,it,0);
		cudaCheck();

     	        if((it+1)%100 == 0){fprintf(stdout,"\r* it = %d / %d (%d%)",it+1,nt,(100*(it+1)/nt));fflush(stdout);}
 	}
 	cudaMemcpy(p[0], d_p, mtxBufferLength, cudaMemcpyDeviceToHost);
 	cudaMemcpy(pp[0], d_pp, mtxBufferLength, cudaMemcpyDeviceToHost);
 	cudaMemcpy(upb[0][0], d_upb, upbBufferLength, cudaMemcpyDeviceToHost);
}

void fd_back(int order, float **p, float **pp, float **pr, float **ppr, float **v2, float ***upb, int nz, int nx, int nt, int is, int sz, int gz, float ***snaps, float **imloc, float **d_obs)
{
	int ix, iz, it;

	dim3 dimGrid(gridx, gridz);
  	dim3 dimGridTaper(gridx, gridBorder_z);
  	dim3 dimGridUpb(gridx,1);

  	dim3 dimBlock(sizeblock, sizeblock);
	write_buffers(p,pp,v2,upb,taper_x, taper_z,d_obs,imloc,is,0);
	write_buffers(pr,ppr,v2,upb,taper_x,taper_z,d_obs,imloc,is,1);
	
        for(it=0; it<nt; it++)
        {
                if(it==0 || it==1)
                {
                        for(ix=0; ix<nx; ix++)
                        {
                                for(iz=0; iz<nz; iz++)
                                {
                                        pp[ix][iz] = snaps[1-it][ix][iz];
                                }
                        }
                        cudaMemcpy(d_pp, pp[0], mtxBufferLength, cudaMemcpyHostToDevice);
                }
                else
                {
                        kernel_lap<<<dimGrid, dimBlock>>>(order,nx,nz,d_p,d_laplace,d_coefs_x,d_coefs_z);
                        kernel_time<<<dimGrid, dimBlock>>>(nx,nz,d_p,d_pp,d_v2,d_laplace,dt2);
                        kernel_upb<<<dimGridUpb, dimBlock>>>(order,nx,nz,nzbin,nt,d_pp,d_upb,it,1);
                }

                d_swap = d_pp;
                d_pp = d_p;
                d_p = d_swap;

                kernel_tapper<<<dimGridTaper, dimBlock>>>(nx,nz,nxbin,nzbin,d_pr,d_ppr,d_taperx,d_taperz);
                kernel_lap<<<dimGrid, dimBlock>>>(order,nx,nz,d_pr,d_laplace, d_coefs_x, d_coefs_z);
                kernel_time<<<dimGrid, dimBlock>>>(nx,nz,d_pr,d_ppr,d_v2,d_laplace,dt2);
                kernel_sism<<<dimGridUpb, dimBlock>>>(nx,nz,nxbin,nt,is,it,gz,d_sis,d_ppr);
                kernel_img<<<dimGrid, dimBlock>>>(nx,nz,nxbin,nzbin,d_img,d_p,d_ppr);

                d_swap = d_ppr;
                d_ppr = d_pr;
                d_pr = d_swap;

                if((it+1)%100 == 0)
                {
                        fprintf(stdout,"\r* it = %d / %d (%d%)",it+1,nt,(100*(it+1)/nt));fflush(stdout);
                }
	}
}

int main (int argc, char **argv)
{
	FILE *fsource = NULL, *fvel_ext = NULL, *fd_obs = NULL, *fvp = NULL, *fsns = NULL,*fsns2 = NULL, *fsnr = NULL, *fimg = NULL, *flim = NULL, *fimg_lap = NULL;

	int iz, ix, it, is;

	float *srce;
	float **vp = NULL, **vpe = NULL, **vpex = NULL;

	float **PP,**P,**PPR,**PR,**tmp;
	float ***swf, ***upb, ***snaps, **vel2, ***d_obs, ***vel_ext_rnd;
	float **imloc, **img, **img_lap;
        read_input(argv[1]);

        printf("## vp = %s, d_obs = %s, vel_ext_file = %s, vel_ext_flag = %d \n",vpfile,datfile,vel_ext_file,vel_ext_flag);
	printf("## nz = %d, nx = %d, nt = %d \n",nz,nx,nt);
	printf("## dz = %f, dx = %f, dt = %f \n",dz,dx,dt);
	printf("## ns = %d, sz = %d, fsx = %d, ds = %d, gz = %d \n",ns,sz,fsx,ds,gz);
	printf("## order = %d, nzb = %d, nxb = %d, F = %f, rnd = %d \n",order,nzb,nxb,fac,rnd);
        srce = alloc1float(nt);
        //ricker_wavelet(nt, dt, fpeak, srce);
	sx = alloc1int(ns);
	for(is=0; is<ns; is++){
		sx[is] = fsx + is*ds + nxb;
	}
	sz += nzb;
	gz += nzb;
	nze = nz + 2 * nzb;
	nxe = nx + 2 * nxb;
	if(vel_ext_flag){
		vel_ext_rnd = alloc3float(nze,nxe,ns);
		memset(**vel_ext_rnd,0,nze*nxe*ns*sizeof(float));
		fvel_ext = fopen(vel_ext_file,"r");
		fread(**vel_ext_rnd,sizeof(float),nze*nxe*ns,fvel_ext);
		fclose(fvel_ext);
	}

	d_obs = alloc3float(nt,nx,ns);
	memset(**d_obs,0,nt*nx*ns*sizeof(float));
	fd_obs = fopen(datfile,"r");
	fread(**d_obs,sizeof(float),nt*nx*ns,fd_obs);
	fclose(fd_obs);

	float **d_obs_aux=(float**)malloc(ns*sizeof(float*));
	for(int i=0; i<ns; i++) 
		d_obs_aux[i] = (float*)malloc((nt*nx)*sizeof(float)); 
	
	for(int i=0; i<ns; i++){
		for(int j=0; j<nx; j++){
			for(int k=0; k<nt; k++)
				d_obs_aux[i][j*nt+k] = d_obs[i][j][k]; 
		}
	}

	vp = alloc2float(nz,nx);
	memset(*vp,0,nz*nx*sizeof(float));
	fvp = fopen(vpfile,"r");
	fread(vp[0],sizeof(float),nz*nx,fvp);
	fclose(fvp);

	vpe = alloc2float(nze,nxe);
	vpex = vpe;

	for(ix=0; ix<nx; ix++){
		for(iz=0; iz<nz; iz++){
			vpe[ix+nxb][iz+nzb] = vp[ix][iz]; 
		}
	}

	vel2 = alloc2float(nze,nxe);
        fd_init(order,nxe,nze,nxb,nzb,nt,ns,fac,dx,dz,dt);
	//taper_init(nxb,nzb,fac);

        PP = alloc2float(nze,nxe);
	P = alloc2float(nze,nxe);
	PPR = alloc2float(nze,nxe);
	PR = alloc2float(nze,nxe);
	upb = alloc3float(order/2,nxe,nt);
	snaps = alloc3float(nze,nxe,2);
	imloc = alloc2float(nz,nx);
	img = alloc2float(nz,nx);
	img_lap = alloc2float(nz,nx);

	char filepath [100];
	sprintf(filepath, "%s/dir.snaps", tmpdir);
	fsns = fopen(filepath,"w");
	sprintf(filepath, "%s/dir.snaps_rec", tmpdir);
	fsns2 = fopen(filepath,"w");
	sprintf(filepath, "%s/dir.snapr", tmpdir);
	fsnr = fopen(filepath,"w");
	sprintf(filepath, "%s/dir.image", tmpdir);
	fimg = fopen(filepath,"w");
	sprintf(filepath, "%s/dir.image_lap", tmpdir);	
	fimg_lap = fopen(filepath,"w");
	
	memset(*img,0,nz*nx*sizeof(float));
        memset(*img_lap,0,nz*nx*sizeof(float));
        
        for(is=0; is<ns; is++){
		fprintf(stdout,"** source %d, at (%d,%d) \n",is+1,sx[is]-nxb,sz-nzb);

		if (vel_ext_flag){
			vpe = vel_ext_rnd[is];					// load hybrid border vpe from file
		}else{
			extendvel_linear(nx,nz,nxb,nzb,vpe); 	// hybrid border (linear randomic)
		}


		for(ix=0; ix<nx+2*nxb; ix++){
			for(iz=0; iz<nz+2*nzb; iz++){
				vel2[ix][iz] = vpe[ix][iz]*vpe[ix][iz];
			}
		}

		memset(*PP,0,nze*nxe*sizeof(float));
		memset(*P,0,nze*nxe*sizeof(float));
    
		fd_forward(order,P,PP,vel2,upb,nze,nxe,nt,is,sz,sx,srce, is);
		fprintf(stdout,"\n");

		for(iz=0; iz<nze; iz++){
			for(ix=0; ix<nxe; ix++){
				snaps[0][ix][iz] = P[ix][iz];
				snaps[1][ix][iz] = PP[ix][iz];
			}
		}

		fprintf(stdout,"** backward propagation %d, at (%d,%d) \n",is+1,sx[is]-nxb,sz-nzb);

		memset(*PP,0,nze*nxe*sizeof(float));
		memset(*P,0,nze*nxe*sizeof(float));
		memset(*PPR,0,nze*nxe*sizeof(float));
		memset(*PR,0,nze*nxe*sizeof(float));
		memset(*imloc,0,nz*nx*sizeof(float));


		fd_back(order,P,PP,PR,PPR,vel2,upb,nze,nxe,nt,is,sz,gz,snaps,imloc,d_obs_aux);
		fprintf(stdout,"\n");
		

		for(iz=0; iz<nz; iz++){
			for(ix=0; ix<nx; ix++){
				img[ix][iz] += imloc[ix][iz];
			}
		}
	}
	fwrite(*img,sizeof(float),nz*nx,fimg);

	fwrite(*img_lap,sizeof(float),nz*nx,fimg_lap);

	fclose(fsns);
	fclose(fsns2);
	fclose(fsnr);
	fclose(fimg);
	fclose(fimg_lap);
        
        // free memory device
        // taper_destroy();
        free1float(coefs);
	free1int(sx);
	free1float(srce);
        free2float(laplace);
	free2float(vp);
	free2float(P);
	free2float(PP);
	free2float(PR);
	free2float(PPR);
	free3float(snaps);
	free2float(imloc);
	free2float(img);
	free2float(img_lap);
	free2float(vpex);
	free2float(vel2);
	free3float(upb);
        free3float(d_obs);
        if(vel_ext_flag) free3float(vel_ext_rnd);
        cudaFree(d_p);
        cudaFree(d_pp);
        cudaFree(d_pr);
        cudaFree(d_ppr);
        cudaFree(d_v2);
        cudaFree(d_laplace);
        cudaFree(d_coefs_z);
        cudaFree(d_coefs_x);

        cudaFree(d_taperx);
        cudaFree(d_taperz);

        cudaFree(d_sis);
        cudaFree(d_img);
        cudaFree(d_upb);
        return 0;
}
