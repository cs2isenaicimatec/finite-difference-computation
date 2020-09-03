/**********************************************************************/
/************************* FAILED *************************************/
__kernel void fd_step(int order, __global float *restrict p, int nz, int nx,
    __global float *restrict coefs_x, __global float *restrict coefs_z,  
        __global float *restrict laplace){
    int div = order/2;
    int i, j;
    __local int mult;
    __local int aux, io;  

    float acmz_vec[9];
    float acmx_vec[9];

    float tmp_x;
    float tmp_z;

    for (i=div;i < (nx - div);i++ ){
        mult = i*nz;
        #pragma unroll 16
        for (j=div;j < (nz - div);j++ ){
            acmz_vec[0] = p[(mult) + (j-4)]*coefs_z[0];
            acmz_vec[1] = p[(mult) + (j-3)]*coefs_z[1];
            acmz_vec[2] = p[(mult) + (j-2)]*coefs_z[2];
            acmz_vec[3] = p[(mult) + (j-1)]*coefs_z[3];
            acmz_vec[4] = p[(mult) + (j+0)]*coefs_z[4];
            acmz_vec[5] = p[(mult) + (j+1)]*coefs_z[5];
            acmz_vec[6] = p[(mult) + (j+2)]*coefs_z[6];
            acmz_vec[7] = p[(mult) + (j+3)]*coefs_z[7];
            acmz_vec[8] = p[(mult) + (j+4)]*coefs_z[8];


            acmx_vec[0] = p[((i-4)*nz) + j]*coefs_x[0];
            acmx_vec[1] = p[((i-3)*nz) + j]*coefs_x[1];
            acmx_vec[2] = p[((i-2)*nz) + j]*coefs_x[2];
            acmx_vec[3] = p[((i-1)*nz) + j]*coefs_x[3];
            acmx_vec[4] = p[((i+0)*nz) + j]*coefs_x[4];
            acmx_vec[5] = p[((i+1)*nz) + j]*coefs_x[5];
            acmx_vec[6] = p[((i+2)*nz) + j]*coefs_x[6];
            acmx_vec[7] = p[((i+3)*nz) + j]*coefs_x[7];
            acmx_vec[8] = p[((i+4)*nz) + j]*coefs_x[8];

            tmp_z = (acmz_vec[0]+acmz_vec[1]+acmz_vec[2]+acmz_vec[3]+acmz_vec[4]+acmz_vec[5]+acmz_vec[6]+acmz_vec[7]+acmz_vec[8]);
            tmp_x = (acmx_vec[0]+acmx_vec[1]+acmx_vec[2]+acmx_vec[3]+acmx_vec[4]+acmx_vec[5]+acmx_vec[6]+acmx_vec[7]+acmx_vec[8]);

            laplace[mult+j] = tmp_x + tmp_z;
        }
    }
}

