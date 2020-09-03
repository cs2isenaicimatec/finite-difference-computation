#include "cwp.h"
#include "fd.h"
#include "time.h"
#include "fpga.h"

#include  <sys/time.h>

static float dx2inv, dz2inv, dt2;
static float **laplace_serial = NULL;
static float **laplace_fpga = NULL;
static float *coefs = NULL;
static float *coefs_z = NULL;
static float *coefs_x = NULL;
static void makeo2(float *coef, int order);

#define SEL_MAX(ref, val) 		(val > ref? val : ref)
#define SEL_MIN(ref, val) 		(val < ref? val : ref)
#define SWAP_MAX(ref, val)  ref = SEL_MAX(ref, val)
#define SWAP_MIN(ref, val)  ref = SEL_MIN(ref, val)

#define MULT_FACTOR			1000.0
#define NORM(out, in)		out = (in*MULT_FACTOR)
#define DENORM(out, in)		out = (float) ((in*1.0)/MULT_FACTOR)

float *calc_coefs(int order);

float avg_serial;
float avg_fpga;
float avg_send_data;
float avg_read_data;
float avg_calc;
int counter;

void fd_print_report(int nx, int nz) {

	float total_bytes, rd_perc, wr_perc, calc_perc, speedup;
	// avg_serial = avg_serial / (counter * 1.0);
	avg_fpga = avg_fpga / (counter * 1.0);
	avg_calc = avg_calc / (counter * 1.0);
	avg_send_data = avg_send_data / (counter * 1.0);
	avg_read_data = avg_read_data / (counter * 1.0);

	// total_bytes = (nx * nz * sizeof(float)) * 1.0 / 1000.0;
	rd_perc = (avg_read_data / avg_fpga) * 100.0;
	wr_perc = (avg_send_data / avg_fpga) * 100.0;
	calc_perc = (avg_calc / avg_fpga) * 100.0;
	// speedup = avg_serial / avg_fpga;

	// printf("\n**************************************************\n");
	// printf("  Size:                 nx=%d, nz=%d  \n", nx, nz);
	// printf("  Total Interactions:   %d \n", counter);
	// printf("  Bytes/Interaction:    %.3f KB \n", total_bytes);
	// printf("  Serial                %.6f  \n", avg_serial);
	printf("  Device                  %.6f  \n", avg_fpga);
	printf("  Send Data             %.6f us (%03.2f %%) \n", avg_send_data,
			wr_perc);
	printf("  Read Data             %.6f us (%03.2f %%) \n", avg_read_data,
			rd_perc);
	printf("  Kernel                %.6f us (%03.2f %%) \n", avg_calc, calc_perc);
	// printf("  Speedup               %.2f \n", speedup);
	// printf("**************************************************\n");

}

void fd_init(int order, int nx, int nz, float dx, float dz, float dt) {
	int io;
	unsigned ulapx, ulapz;
	dx2inv = (1. / dx) * (1. / dx);
	dz2inv = (1. / dz) * (1. / dz);
	dt2 = dt * dt;

	coefs = calc_coefs(order);
	coefs_z = calc_coefs(order);
	coefs_x = calc_coefs(order);

	// pre calc coefs 8 d2 inv
	for (io = 0; io <= order; io++) {
		coefs_z[io] = dz2inv * coefs[io];
		coefs_x[io] = dx2inv * coefs[io];
		ulapx = *(unsigned int *) &coefs_x[io];
		ulapz = *(unsigned int *) &coefs_z[io];
//		printf ("coefs_x[%d] = 32'h%x (%f)\n", io, ulapx, coefs_x[io]);
//		printf ("coefs_z[%d] = 32'h%x (%f)\n", io, ulapz, coefs_z[io]);
	}
	laplace_serial = alloc2float(nz, nx);
	laplace_fpga = alloc2float(nz, nx);

	// REPORT
	avg_serial = 0;
	avg_fpga = 0;
	avg_send_data = 0;
	avg_read_data = 0;
	avg_calc = 0;
	counter = 0;

	memset(*laplace_serial, 0, nz * nx * sizeof(float));
	memset(*laplace_fpga, 0, nz * nx * sizeof(float));
	return;
}

void fd_serial_laplacian(int order, float **p, float ** laplace, int nz, int nx) {
	int ix, iz, io;
	float acm = 0.0;
	float res0[9], res1[9], x[9], z[9], cx[9], cz[9];
	unsigned int ures0[9], ures1[9], uacm[9], ux[9], uz[9], ucx[9], ucz[9];
	float result_0 = 0.0, result_1 = 0.0, result = 0.0;
	unsigned int uresult;

	for (ix = order / 2; ix < nx - (order / 2); ix++) {
		for (iz = order / 2; iz < nz - (order / 2); iz++) {
			for (io = 0; io <= order; io++) {
				//acm += p[ix][iz + io - order / 2] * coefs[io] * dz2inv;
				//acm += p[ix + io - order / 2][iz] * coefs[io] * dx2inv;
				// coefs*d2inv where previously calculated
				// acm += p[ix][iz + io - order / 2] * coefs_z[io];
				// acm += p[ix + io - order / 2][iz] * coefs_x[io];
				result_0 += p[ix][iz + io - order / 2] * coefs_z[io];
				result_1 += p[ix + io - order / 2][iz] * coefs_x[io];
			}
			acm = result_0 + result_1;
			laplace[ix][iz] = acm;
			acm = 0.0;
			result_0 = 0.0;
			result_1 = 0.0;
		}
	}
//	for (ix = order / 2; ix < nx - (order / 2); ix++) {
//		for (iz = order / 2; iz < nz - (order / 2); iz++) {
//			for (io = 0; io <= order; io++) {
//				//acm += p[ix][iz + io - order / 2] * coefs[io] * dz2inv;
//				//acm += p[ix + io - order / 2][iz] * coefs[io] * dx2inv;
//				// coefs*d2inv where previously calculated
//				result_0 += p[ix][iz + io - order / 2] * coefs_z[io];
//				acm += p[ix][iz + io - order / 2] * coefs_z[io];
//				//ures0[io] = *(unsigned int *) &acm;
//				//res0[io] = acm;
//				ures0[io] = *(unsigned int *) &result_0;
//				res0[io] = result_0;
//
//				acm += p[ix + io - order / 2][iz] * coefs_x[io];
//				result_1 += p[ix + io - order / 2][iz] * coefs_x[io];
//				ures1[io] = *(unsigned int *) &result_1;
//				res1[io] = result_1;
//
//				//ures1[io] = *(unsigned int *) &acm;
//				//res1[io] = acm;
//
//				x[io] = p[ix + io - order / 2][iz];
//				z[io] = p[ix][iz + io - order / 2];
//				cx[io] = coefs_x[io];
//				cz[io] = coefs_z[io];
//
//				uacm[io] = *(unsigned int *) &acm;
//				ux[io] = *(unsigned int *) &p[ix + io - order / 2][iz];
//				uz[io] = *(unsigned int *) &p[ix][iz + io - order / 2];
//				ucx[io] = *(unsigned int *) &coefs_x[io];
//				ucz[io] = *(unsigned int *) &coefs_z[io];
//			}
//			result = result_0 + result_1;
//			uresult = *(unsigned int *) &result;
//			laplace[ix][iz] = acm;
//			acm = 0.0;
//			result_0 = 0.0;
//			result_1 = 0.0;
//			result = 0.0;
//		}
//	}
}

void fd_fpga_laplacian(ssize_t fid, int order, float **p, float ** laplace,
		int nz, int nx) {

	struct timeval st, et;
	unsigned int timeout = LCORE_DEFAULT_TIMEOUT;

	//////////////////////////////////////////////////////////////////////
	// send data
	gettimeofday(&st, NULL);
	fpga_write(fid, p, nx * nz, LCORE_DEFAULT_IMGSTART_ADDR);
	fpga_run(fid);
	gettimeofday(&et, NULL);
	avg_send_data += (__elapsed_time(st, et) * 1.0);

	//////////////////////////////////////////////////////////////////////
	// calc lapl
	gettimeofday(&st, NULL);
	do {
		if (--timeout == 0) {
			break;
		}
		usleep(10);
	} while (fpga_isbusy(fid));
	//usleep(800);

	if (timeout == 0) {
		printf("Error. Could not get a response from FPGA. Aborting... \n");
		close(fid);
		exit(-1);
	}
	gettimeofday(&et, NULL);
	avg_calc += (__elapsed_time(st, et) * 1.0);

	//////////////////////////////////////////////////////////////////////
	// read data
	gettimeofday(&st, NULL);
	fpga_read(fid, laplace, nx * nz, LCORE_DEFAULT_LAPLSTART_ADDR);
	//fpga_read(fid, (float *) laplace[0], nx * nz, LCORE_DEFAULT_LAPLSTART_ADDR);
	gettimeofday(&et, NULL);
	avg_read_data += (__elapsed_time(st, et) * 1.0);
}

void fd_step(ssize_t fid, int order, float **p, float **pp, float **v2, int nz,
		int nx, int log) {
	int ix, iz;
	int elapsed_serial, elapsed_fpga;
	FILE *fidLapl, *fidFPGALapl, *fidInput;
	unsigned int ulap;
	struct timeval st, et;
	char fileName[200];
#ifdef LCORE_FPGA
	// if (log) {
	// 	sprintf(fileName, "data/inputdata_%dx%d.hex", nx, nz);
	// 	fidInput = fopen(fileName, "w");
	// 	for (ix = 0; ix < nx; ix++) {
	// 		for (iz = 0; iz < nz; iz++) {
	// 			ulap = *(unsigned int *) &p[ix][iz];
	// 			fprintf(fidInput, "%08x\n", ulap);
	// 		}
	// 	}
	// 	fclose(fidInput);
	// }
#endif

#ifdef LCORE_FPGA
	gettimeofday(&st, NULL);
	fd_fpga_laplacian(fid, order, p, laplace_fpga, nz, nx);
	gettimeofday(&et, NULL);
	elapsed_fpga = __elapsed_time(st, et);
	//gettimeofday(&st, NULL);
	//fd_serial_laplacian(order, p, laplace_serial, nz, nx);
	//gettimeofday(&et, NULL);
	//elapsed_serial = __elapsed_time(st, et);

	//avg_serial += (elapsed_serial * 1.0);
	avg_fpga += (elapsed_fpga * 1.0);
	counter++;
	//arria_print_driver_status(fid);
#else
	gettimeofday(&st, NULL);
	fd_serial_laplacian(order, p, laplace_serial, nz, nx);
	gettimeofday(&et, NULL);
	elapsed_serial = __elapsed_time(st, et);
	avg_serial += (elapsed_serial * 1.0);
	counter++;
#endif

	for (ix = 0; ix < nx; ix++) {
		for (iz = 0; iz < nz; iz++) {
//			pp[ix][iz] = 2. * p[ix][iz] - pp[ix][iz]
//					+ v2[ix][iz] * dt2 * laplace_serial[ix][iz];
#ifdef LCORE_FPGA
			pp[ix][iz] = 2. * p[ix][iz] - pp[ix][iz]
					+ v2[ix][iz] * dt2 * laplace_fpga[ix][iz];
#else
			pp[ix][iz] = 2. * p[ix][iz] - pp[ix][iz]
			+ v2[ix][iz] * dt2 * laplace_serial[ix][iz];
#endif
		}
	}
#ifdef LCORE_FPGA
// 	if (log) {
// 		sprintf(fileName, "data/laplace_%dx%d_serial.hex", nx, nz);
// 		fidLapl = fopen(fileName, "w");
// 		sprintf(fileName, "data/laplace_%dx%d_fpga.hex", nx, nz);
// 		fidFPGALapl = fopen(fileName, "w");
// 		for (ix = 0; ix < nx; ix++) {
// 			for (iz = 0; iz < nz; iz++) {
// 				ulap = *(unsigned int *) &laplace_serial[ix][iz];
// 				fprintf(fidLapl, "%08x\n", ulap);
// 				ulap = *(unsigned int *) &laplace_fpga[ix][iz];
// 				fprintf(fidFPGALapl, "%08x\n", ulap);
// 			}
// 		}
// 		fclose(fidLapl);
// 		fclose(fidFPGALapl);
// //		for (ix = 0; ix < nx; ix ++) {
// //
// //			printf("\n ADDR laplace_serial[%d] = %p \n", ix,
// //					laplace_serial[ix]);
// //			printf(" ADDR laplace_fpga[%d] = %p \n", ix,
// //					laplace_fpga[ix]);
// //		}
// 	}
#endif
	return;
}

void fd_destroy() {
	free2float(laplace_serial);
	free1float(coefs);
	return;
}

float *calc_coefs(int order) {
	float *coef;

	coef = calloc(order + 1, sizeof(float));

	switch (order) {
	case 2:
		coef[0] = 1.;
		coef[1] = -2.;
		coef[2] = 1.;
		break;
	case 4:
		coef[0] = -1. / 12.;
		coef[1] = 4. / 3.;
		coef[2] = -5. / 2.;
		coef[3] = 4. / 3.;
		coef[4] = -1. / 12.;
		break;
	case 6:
		coef[0] = 1. / 90.;
		coef[1] = -3. / 20.;
		coef[2] = 3. / 2.;
		coef[3] = -49. / 18.;
		coef[4] = 3. / 2.;
		coef[5] = -3. / 20.;
		coef[6] = 1. / 90.;
		break;
	case 8:
		coef[0] = -1. / 560.;
		coef[1] = 8. / 315.;
		coef[2] = -1. / 5.;
		coef[3] = 8. / 5.;
		coef[4] = -205. / 72.;
		coef[5] = 8. / 5.;
		coef[6] = -1. / 5.;
		coef[7] = 8. / 315.;
		coef[8] = -1. / 560.;
		break;
	default:
		makeo2(coef, order);
	}

	return coef;
}

static void makeo2(float *coef, int order) {
	float h_beta, alpha1 = 0.0;
	float alpha2 = 0.0;
	float central_term = 0.0;
	float coef_filt = 0;
	float arg = 0.0;
	float coef_wind = 0.0;
	int msign, ix;
	float alpha = .54;
	float beta = 6.;

	h_beta = 0.5 * beta;
	alpha1 = 2. * alpha - 1.0;
	alpha2 = 2. * (1.0 - alpha);
	central_term = 0.0;

	msign = -1;

	for (ix = 1; ix <= order / 2; ix++) {
		msign = -msign;
		coef_filt = (2. * msign) / (ix * ix);
		arg = PI * ix / (2. * (order / 2 + 2));
		coef_wind = pow((alpha1 + alpha2 * cos(arg) * cos(arg)), h_beta);
		coef[order / 2 + ix] = coef_filt * coef_wind;
		central_term = central_term + coef[order / 2 + ix];
		coef[order / 2 - ix] = coef[order / 2 + ix];
	}

	coef[order / 2] = -2. * central_term;

	return;
}

