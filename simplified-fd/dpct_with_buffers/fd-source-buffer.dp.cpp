#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <math.h>

void fd_init(int order, int nx, int nz, float dx, float dz);
void fd_init_cuda(int order, int nxe, int nze);
float *calc_coefs(int order);
static void makeo2 (float *coef,int order);

#define sizeblock 16
#define PI (3.141592653589793)

float *d_p;
float *d_laplace, *d_coefs_x, *d_coefs_z;

size_t mtxBufferLength, coefsBufferLength;

int gridx, gridz;

static float dx2inv, dz2inv;
static float *coefs = NULL;
static float *coefs_z = NULL;
static float *coefs_x = NULL;

typedef const cl::sycl::accessor<float, 1,
	cl::sycl::access::mode::read_write,
	cl::sycl::access::target::global_buffer> acc_float;

void kernel_lap(int order, int nx, int nz, acc_float p, acc_float lap,
								acc_float coefsx, acc_float coefsz, sycl::nd_item<3> item_ct1)
{
        int half_order=order/2;
        int i = half_order +
                item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                item_ct1.get_local_id(2); // Global row index
        int j = half_order +
                item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
                item_ct1.get_local_id(1); // Global column index
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
                coef_wind =
                    pow((alpha1 + alpha2 * cos(arg) * cos(arg)), h_beta);
                coef[order/2+ix] = coef_filt*coef_wind;
                central_term = central_term + coef[order/2+ix];
                coef[order/2-ix] = coef[order/2+ix];
        }

        coef[order/2]  = -2.*central_term;

        return;
}

void fd_init_cuda(int order, int nxe, int nze)
{
        dpct::device_ext &dev_ct1 = dpct::get_current_device();
        sycl::queue &q_ct1 = dev_ct1.default_queue();
        mtxBufferLength = (nxe*nze)*sizeof(float);
        coefsBufferLength = (order+1)*sizeof(float);

        // Create a Device pointers
				/*
        d_p = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);
        d_laplace = (float *)sycl::malloc_device(mtxBufferLength, q_ct1);
        d_coefs_x = (float *)sycl::malloc_device(coefsBufferLength, q_ct1);
        d_coefs_z = (float *)sycl::malloc_device(coefsBufferLength, q_ct1);
				*/
        int div_x, div_z;
        // Set a Grid for the execution on the device
        int tx = ((nxe - 1) / 32 + 1) * 32;
        int tz = ((nze - 1) / 32 + 1) * 32;

        div_x = (float) tx/(float) sizeblock;
        div_z = (float) tz/(float) sizeblock;

        gridx = (int)ceil(div_x);
        gridz = (int)ceil(div_z);
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
        dpct::device_ext &dev_ct1 = dpct::get_current_device();
        sycl::queue &q_ct1 = dev_ct1.default_queue();
        // constantes
        int nz = 195, nx = 315, nxb = 50, nzb = 50, nxe, nze, order = 8;
        float dz = 10.000000, dx = 10.000000;

        nxe = nx + 2 * nxb;
        nze = nz + 2 * nzb;
        // inicialização
        fd_init(order,nxe,nze,dx,dz);

        sycl::range<3> dimGrid(gridx, gridz, 1);
        sycl::range<3> dimBlock(sizeblock, sizeblock, 1);

        FILE *finput;
        // leitura do input
        finput = fopen("input.bin", "rb");

        float *input_data, *output_data;
        input_data = (float*)malloc(mtxBufferLength);
				output_data = (float*)malloc(mtxBufferLength);
				printf("lendo arquivo...\n");
        fread(input_data, sizeof(float), nxe*nze, finput);
        fclose(finput);
        // utilização do kernel
				/*
				q_ct1.memcpy(d_coefs_x, coefs_x, coefsBufferLength).wait();
        q_ct1.memcpy(d_p, input_data, mtxBufferLength).wait();
				q_ct1.memcpy(d_coefs_z, coefs_z, coefsBufferLength).wait();
				*/
				{
					sycl::buffer<float, 1> buf_input(input_data, sycl::range<1>(nxe*nze));
					sycl::buffer<float, 1> buf_coefsx(coefs_x, sycl::range<1>(order+1));
					sycl::buffer<float, 1> buf_coefsz(coefs_z, sycl::range<1>(order+1));
					sycl::buffer<float, 1> buf_output(output_data, sycl::range<1>(nxe*nze));
					/*
					DPCT1049:0: The workgroup size passed to the SYCL kernel may
					* exceed the limit. To get the device limit, query
					* info::device::max_work_group_size. Adjust the workgroup size if
					* needed.
					*/
					q_ct1.submit([&](sycl::handler &cgh) {
						auto dpct_global_range = dimGrid * dimBlock;
						auto A_input = buf_input.get_access<sycl::access::mode::read_write>(cgh);
						auto A_coefsx = buf_coefsx.get_access<sycl::access::mode::read_write>(cgh);
						auto A_coefsz = buf_coefsz.get_access<sycl::access::mode::read_write>(cgh);
						auto A_output = buf_output.get_access<sycl::access::mode::read_write>(cgh);
						sycl::stream out(1024, 256, cgh);
						/*auto d_p_ct3 = d_p;
						auto d_laplace_ct4 = d_laplace;
						auto d_coefs_x_ct5 = d_coefs_x;
						auto d_coefs_z_ct6 = d_coefs_z;*/
						cgh.parallel_for(
							sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
							dpct_global_range.get(1),
							dpct_global_range.get(0)),
							sycl::range<3>(dimBlock.get(2),
							dimBlock.get(1),
							dimBlock.get(0))),
							[=](sycl::nd_item<3> item_ct1) {
								kernel_lap(order, nxe, nze, A_input, A_output,
									A_coefsx, A_coefsz, item_ct1);
								});
							});
				} //scope to destroy buffers


        //q_ct1.memcpy(output_data, d_laplace, mtxBufferLength).wait();

        // salvando a saída
        FILE *foutput;
        printf("salvando saída...\n");
        foutput = fopen("output_teste.bin", "wb");
        fwrite(output_data, sizeof(float), nxe*nze, foutput);
        fclose(foutput);
				for(int i = 1321; i < 1341; i++){
                printf("%.15f\n", output_data[i]);
        }
        // free memory device
        free(input_data);
        free(output_data);
        // sycl::free(d_p, q_ct1);
        // sycl::free(d_laplace, q_ct1);
        // sycl::free(d_coefs_x, q_ct1);
        // sycl::free(d_coefs_z, q_ct1);
        return 0;
}
