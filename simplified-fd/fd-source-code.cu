#include <stdio.h>
#include <cuda.h>

void fd_init(int order, int nx, int nz, float dx, float dz);
void fd_init_cuda(int order, int nxe, int nze);
float *calc_coefs(int order);

#define sizeblock 32

float *d_p;
float *d_laplace, *d_coefs_x, *d_coefs_z;

size_t mtxBufferLength, coefsBufferLength;

int gridx, gridz;

static float dx2inv, dz2inv;
static float *coefs = NULL;
static float *coefs_z = NULL;
static float *coefs_x = NULL;

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
        }

        return coef;
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
        // constantes
        int nz = 195, nx = 315, nxb = 50, nzb = 50, nxe, nze, order = 8;
        float dz = 10.00000, dx = 10.000000;



        nxe = nx + 2 * nxb;
        nze = nz + 2 * nzb;
        // inicialização
        fd_init(order,nxe,nze,dx,dz);
        dim3 dimGrid(gridx, gridz);
        dim3 dimBlock(sizeblock, sizeblock);


        // arquivos
        FILE *finput;
        FILE *foutput;
        // leitura do input
        finput = fopen("./input.bin", "rb");

        float input_data[mtxBufferLength], output_data[mtxBufferLength];
        printf("lendo arquivo...\n");
        fread(input_data, sizeof(input_data), 1, finput);
        printf("%.15f\n", input_data[1341]);
        fclose(finput);
        // utilização do kernel
        cudaMemcpy(d_p, input_data, mtxBufferLength, cudaMemcpyHostToDevice);


        kernel_lap<<<dimGrid, dimBlock>>>(order,nx,nz,d_p,d_laplace,d_coefs_x,d_coefs_z);

        cudaMemcpy(output_data, d_laplace, mtxBufferLength, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess)
        {
                // print the CUDA error message and exit
                printf("CUDA error: %s\n", cudaGetErrorString(error));
                exit(-1);
        }

        // salvando a saída
        printf("salvando saída...\n");
        foutput = fopen("output_teste.bin", "wb");
        printf("%.15f\n", output_data[1341]);
        printf("Esperado: 0.000010451854905\n");
        printf("escrevendo arquivo\n");
        fwrite(output_data, sizeof(output_data), 1, foutput);
        fclose(foutput);
        // free memory device

        cudaFree(d_p);
        cudaFree(d_laplace);
        cudaFree(d_coefs_x);
        cudaFree(d_coefs_z);
        return 0;
}
