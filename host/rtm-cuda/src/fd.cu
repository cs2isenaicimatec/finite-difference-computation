#include "fd.h"
#include "time.h"
#include "misc.h"

extern "C" {
	#include "cwp.h"
}
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <math.h>
#define sizeblock 8 

void cudaCheck(){
    cudaError_t err=cudaGetLastError();
    if(err!=cudaSuccess){
      printf("%s\n", cudaGetErrorString(err));
      exit(1);
    }
}


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
	printf("> Device       = %.1f (s)\n",deviceAVGTime/1000000.0);
	// printf("> Total        = %.1f (s)\n",execTime/1000000.0);
	// printf("> WR TransfCnt = %d \n",wrTransferCnt/1000000.0);
	// printf("> Write        = %.1f (%.2f%%)(s)\n",wrAVGTime/1000000.0, 
	// 	(100.0*wrAVGTime/deviceAVGTime));
	// printf("> RD TransfCnt = %d \n",rdTransferCnt/1000000.0);
	// printf("> Read         = %.1f (%.2f%%)(s)\n",rdAVGTime/1000000.0,
	// 	(100.0*rdAVGTime/deviceAVGTime));
	printf("> Device Fwrd  = %.1f (s)\n",fwAVGTime/1000000.0);
	printf("> Device Bwrd  = %.1f (s)\n",bwAVGTime/1000000.0);
}
#endif
//////////////////////////////////////////
// Cuda enviroroment global variables  
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
static void makeo2 (float *coef,int order);

float *calc_coefs(int order);

void fd_init_cuda(int order, int nxe, int nze, 
	int nxb, int nzb, int nt, int ns, float fac){
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

void fd_init(int order, int nx, int nz, int nxb, int nzb, 
	int nt, int ns, float fac, float dx, float dz, float dt){
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

 	#ifdef CUDA
        fd_init_cuda(order,nx,nz,nxb,nzb,nt,ns,fac); 
    #endif

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
	// todo: free laplace or realloc
	laplace = alloc2float(nz,nx);
	memset(*laplace,0,nz*nx*sizeof(float));
	return;
}

void fd_destroy(){
	free2float(laplace);
	free1float(coefs);
	#ifdef CUDA
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

	#endif
	return;
}

float *calc_coefs(int order){
	float *coef;

	coef = (float *)calloc(order+1,sizeof(float));

	switch(order){
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

static void makeo2 (float *coef,int order){
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

__global__ void kernel_lap(int order, int nx, int nz, float * __restrict__ p, 
	float * __restrict__ lap, float * __restrict__ coefsx, float * __restrict__ coefsz){  	

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

__global__ void kernel_time(int nx, int nz, float *__restrict__ p, float *__restrict__ pp,
	float *__restrict__ v2, float *__restrict__ lap, float dt2){  	

  	int i =  blockIdx.x * blockDim.x + threadIdx.x; // Global row index    
  	int j =  blockIdx.y * blockDim.y + threadIdx.y; // Global column index
  	int mult = i*nz; 
	
  	if(i<nx){
  		if(j<nz){
			 pp[mult+j] = 2.*p[mult+j] - pp[mult+j] + v2[mult+j]*dt2*lap[mult+j];		
		}
  	}  
}	

__global__ void kernel_tapper(int nx, int nz, int nxb, int nzb, 
	float *__restrict__ p, float *__restrict__ pp, float *__restrict__ taperx, float *__restrict__ taperz){  	

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

__global__ void kernel_src(int nz, float * __restrict__ pp, int sx, int sz, float srce){
 	pp[sx*nz+sz] += srce;
}

__global__ void kernel_upb(int order, int nx, int nz, int nzb, int nt, float *__restrict__ pp,
	float *__restrict__ upb, int it, int flag){
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

__global__ void kernel_sism(int nx, int nz, int nxb,
	int nt, int is, int it, int gz, float *__restrict__ d_obs, float *__restrict__ ppr){
 	int size = nx-(2*nxb); 
	int i = blockIdx.x * blockDim.x + threadIdx.x; //nx index    
 	if(i<size)
 		ppr[((i+nxb)*nz) + gz] += d_obs[i*nt + (nt-1-it)]; 

}

__global__ void kernel_img(int nx, int nz, int nxb, int nzb,
	float * __restrict__ imloc, float * __restrict__ p, float * __restrict__ ppr){
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

void write_buffers(float **p, float **pp, float **v2, float ***upb, 
    float *taperx, float *taperz, float **d_obs, float **imloc, int is, int flag){
	
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

void fd_forward(int order, float **p, float **pp, float **v2, 
   float ***upb, int nz, int nx, int nt, int is, int sz, int *sx, float *srce, int propag){
	int elapsed;
    struct timeval st, et, stCR, etCR, stCW, etCW, stK, etK;
    //Start total time 
    gettimeofday(&st, NULL);

	dim3 dimGrid(gridx, gridz);	
  	dim3 dimGridTaper(gridx, gridBorder_z); 
  	
  	dim3 dimGridSingle(1,1); 
  	dim3 dimGridUpb(gridx,1); 
  	
  	dim3 dimBlock(sizeblock, sizeblock);
  	// Start write time
   	gettimeofday(&stCW, NULL);
	write_buffers(p,pp,v2,upb,taper_x, taper_z,NULL, NULL,is,0);
	// Calc avg write time  
   	gettimeofday(&etCW, NULL);
   	elapsed = ((etCW.tv_sec - stCW.tv_sec) * 1000000) + (etCW.tv_usec - stCW.tv_usec);
   	wrAVGTime += (elapsed*1.0);
   	wrTransferCnt++; 

   	// start Kernel time  
   	gettimeofday(&stK, NULL);
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
     	// op_L_counter++;
 	} 
 	// Calc avg kernel time  
	gettimeofday(&etK, NULL);
	elapsed = ((etK.tv_sec - stK.tv_sec) * 1000000) + (etK.tv_usec - stK.tv_usec);
	kernelAVGTime += (elapsed*1.0);

	// start read time 
	gettimeofday(&stCR, NULL);
	if(propag == 5){
		float input[mtxBufferLength], output[mtxBufferLength];
		FILE *finput;
		finput = fopen("./input.bin", "wb");
		cudaMemcpy(input, d_p, mtxBufferLength, cudaMemcpyDeviceToHost);
		cudaMemcpy(output, d_laplace, mtxBufferLength, cudaMemcpyDeviceToHost);
		fwrite(input,sizeof(input),1,finput);
		printf("\n=== input: ===\n");
		for(int i = 1321; i < 1341; i++){
				printf("%.15f\n", input[i]);
		}
		printf("\n=== output: ===\n");
		for(int i = 1321; i < 1341; i++){
				printf("%.15f\n", output[i]);
		}
		fclose(finput);
	}
 	cudaMemcpy(p[0], d_p, mtxBufferLength, cudaMemcpyDeviceToHost);
 	cudaMemcpy(pp[0], d_pp, mtxBufferLength, cudaMemcpyDeviceToHost);
 	cudaMemcpy(upb[0][0], d_upb, upbBufferLength, cudaMemcpyDeviceToHost);
	// Calc avg read time  
	gettimeofday(&etCR, NULL);
	elapsed = ((etCR.tv_sec - stCR.tv_sec) * 1000000) + (etCR.tv_usec - stCR.tv_usec);
	rdAVGTime += (elapsed*1.0);
    rdTransferCnt++;	
 	gettimeofday(&et, NULL);
	elapsed = ((et.tv_sec - st.tv_sec) * 1000000) + (et.tv_usec - st.tv_usec);
	fwAVGTime += (elapsed*1.0);    
}

void fd_back(int order, float **p, float **pp, float **pr, float **ppr, float **v2, float ***upb,
   int nz, int nx, int nt, int is, int sz, int gz, float ***snaps, float **imloc, float **d_obs){
	int ix, iz, it, elapsed;
   	struct timeval st, et, stCR, etCR, stCW, etCW, stK, etK;
	//Start total time 
	gettimeofday(&st, NULL);
    
	dim3 dimGrid(gridx, gridz);	
  	dim3 dimGridTaper(gridx, gridBorder_z); 	
  	dim3 dimGridUpb(gridx,1); 
  	
  	dim3 dimBlock(sizeblock, sizeblock);
  	// Start write time	
	gettimeofday(&stCW, NULL);
	write_buffers(p,pp,v2,upb,taper_x, taper_z,d_obs,imloc,is,0);
	write_buffers(pr,ppr,v2,upb,taper_x,taper_z,d_obs,imloc,is,1);
	// Calc avg write time  
	gettimeofday(&etCW, NULL);
   	elapsed = ((etCW.tv_sec - stCW.tv_sec) * 1000000) + (etCW.tv_usec - stCW.tv_usec);
  	wrAVGTime += (elapsed*1.0);
	wrTransferCnt++; 
    // start Kernel time  
	gettimeofday(&stK, NULL);
   	for(it=0; it<nt; it++){
		gettimeofday(&etCW, NULL);
    	if(it==0 || it==1){
         for(ix=0; ix<nx; ix++){
            for(iz=0; iz<nz; iz++){
               pp[ix][iz] = snaps[1-it][ix][iz];                       
            }
         }
			cudaMemcpy(d_pp, pp[0], mtxBufferLength, cudaMemcpyHostToDevice);
      	}else{ 
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

		if((it+1)%100 == 0){fprintf(stdout,"\r* it = %d / %d (%d%)",it+1,nt,(100*(it+1)/nt));fflush(stdout);}
	}
	gettimeofday(&etK, NULL);
	elapsed = ((etK.tv_sec - stK.tv_sec) * 1000000) + (etK.tv_usec - stK.tv_usec);
	kernelAVGTime += (elapsed*1.0);
 	
 	gettimeofday(&stCR, NULL);
 	// Calc avg read time  
 	cudaMemcpy(imloc[0], d_img, imgBufferLength, cudaMemcpyDeviceToHost);
	gettimeofday(&etCR, NULL);
	elapsed = ((etCR.tv_sec - stCR.tv_sec) * 1000000) + (etCR.tv_usec - stCR.tv_usec);
	rdAVGTime += (elapsed*1.0);
	rdTransferCnt++;
	//bw avg time
	gettimeofday(&et, NULL);
	elapsed = ((et.tv_sec - st.tv_sec) * 1000000) + (et.tv_usec - st.tv_usec);
	bwAVGTime += (elapsed*1.0);
 	deviceAVGTime = fwAVGTime + bwAVGTime; 

}

