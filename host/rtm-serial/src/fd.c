#include "cwp.h"
#include "fd.h"
#include "time.h"
#include  <sys/time.h>
#include "misc.h"

#ifdef PERF_COUNTERS
unsigned int op_L_counter;
float avg_serial;
float avg_fpga;
float avg_send_data;
float avg_read_data;
float avg_calc;
void fd_print_report(int nx, int nz) {

	// float total_bytes, rd_perc, wr_perc, calc_perc, speedup;
	// avg_serial = avg_serial / (op_L_counter * 1.0);
	// avg_fpga = avg_fpga / (op_L_counter * 1.0);
	// avg_calc = avg_calc / (op_L_counter * 1.0);
	// avg_send_data = avg_send_data / (op_L_counter * 1.0);
	// avg_read_data = avg_read_data / (op_L_counter * 1.0);

	// total_bytes = (nx * nz * sizeof(float)) * 1.0 / 1000.0;
	// rd_perc = (avg_read_data / avg_fpga) * 100.0;
	// wr_perc = (avg_send_data / avg_fpga) * 100.0;
	// calc_perc = (avg_calc / avg_fpga) * 100.0;
	// speedup = avg_serial / avg_fpga;

	// printf("\n************************************************** \n");
	// printf("  Size:                 nx=%d, nz=%d  \n", nx, nz);
	// printf("  Total Interactions:   %d \n", op_L_counter);
	// printf("  Bytes/Interaction:    %.3f KB \n", total_bytes);
	// printf("  Serial                %.6f  \n", avg_serial);
	// printf("  FPGA                  %.6f  \n", avg_fpga);
	// printf("  Send Data             %.6f us (%03.2f %%) \n", avg_send_data,
	// 		wr_perc);
	// printf("  Read Data             %.6f us (%03.2f %%) \n", avg_read_data,
	// 		rd_perc);
	// printf("  Kernel                %.6f us (%03.2f %%) \n", avg_calc, calc_perc);
	// printf("  Speedup               %.2f \n", speedup);
	// printf("**************************************************\n");

}
#endif

static float dx2inv,dz2inv,dt2;
static float **laplace = NULL;
static float *coefs = NULL;
static float *coefs_z = NULL;
static float *coefs_x = NULL;
static void makeo2 (float *coef,int order);

float *calc_coefs(int order);

void op_L(int nz, int nx, float **p, float **laplace, int order){
	int ix,iz,io;
	float acmx = 0., acmz = 0.;

	for(ix=order/2;ix<nx-order/2;ix++){
		for(iz=order/2;iz<nz-order/2;iz++){
			for(io=0;io<=order;io++){
				//acmz += p[ix][iz+io-order/2]*coefs[io];
				//acmx += p[ix+io-order/2][iz]*coefs[io];
				acmz += p[ix][iz+io-order/2]*coefs_z[io];
				acmx += p[ix+io-order/2][iz]*coefs_x[io];
			}
			//laplace[ix][iz] = acmz*dz2inv + acmx*dx2inv;
			laplace[ix][iz] = acmz + acmx;
			acmx = 0.0;
			acmz = 0.0;
		}
	}
}

void fd_init(int order, int nx, int nz, float dx, float dz, float dt){
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

#ifdef PERF_COUNTERS
	avg_serial = 0;
	avg_fpga = 0;
	avg_send_data = 0;
	avg_read_data = 0;
	avg_calc = 0;
	op_L_counter = 0;
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

void fd_step(int order, float **p, float **pp, float **v2, int nz, int nx){
	int ix,iz;

#ifdef PERF_COUNTERS
	struct timeval st, et;
	gettimeofday(&st, NULL);
#endif	
	op_L(nz, nx, p, laplace, order);
#ifdef PERF_COUNTERS
	gettimeofday(&et, NULL);
	avg_serial += (__elapsed_time(st, et) * 1.0);
	op_L_counter++;
#endif

	for(ix=0;ix<nx;ix++){
		for(iz=0;iz<nz;iz++){
			pp[ix][iz] = 2.*p[ix][iz] - pp[ix][iz] + v2[ix][iz]*dt2*laplace[ix][iz];
		}
	}

	return;
}

void fd_destroy(){
	free2float(laplace);
	free1float(coefs);
	return;
}

float *calc_coefs(int order){
	float *coef;

	coef = calloc(order+1,sizeof(float));

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
