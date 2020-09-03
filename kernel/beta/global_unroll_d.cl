#define NZ              295
#define NX              415
#define ORDER           8
#define DIV             (ORDER/2)
#define ORDER_LENGTH    (ORDER +1)
#define PREFETCH_SIZE   (ORDER_LENGTH*NZ)
#define NX_START        (DIV)
#define NZ_START        (DIV)
#define NX_END          (NX-DIV)
#define NZ_END          (NZ-DIV)

__kernel void fd_step(int order, __global float *restrict p, int nz, int nx,
    __global float *restrict coefs_x, __global float *restrict coefs_z,  
        __global float *restrict laplace){

    int i, j, k;
    int mult0, mult1, mult2;
    int aux, io;   
    float acmz_vec[ORDER_LENGTH],acmx_vec[ORDER_LENGTH];
    float tmp_p[PREFETCH_SIZE], tmp_coefs_z[ORDER_LENGTH],tmp_coefs_x[ORDER_LENGTH]; 
    float tmp_x, tmp_z;

    #pragma ivdep array(tmp_coefs_z, tmp_coefs_x)
    for(i=0;i<=ORDER;i++){
        tmp_coefs_z[i] = coefs_z[i];
        tmp_coefs_x[i] = coefs_x[i];
    }
    mult2 = DIV*NZ;
    for (i=DIV;i < NX_END;i++ ){
        mult0 = i*NZ;
        mult1 = (i-DIV)*NZ;
        #pragma ivdep
        for(k=0; k<PREFETCH_SIZE; k++){    
            tmp_p[k] = p[mult1 + k];
        }
        #pragma unroll 7
        for (j=DIV;j < NZ_END;j++ ){
                acmz_vec[0] = tmp_p[mult2 + (j-4)]*tmp_coefs_z[0];
                acmz_vec[1] = tmp_p[mult2 + (j-3)]*tmp_coefs_z[1];
                acmz_vec[2] = tmp_p[mult2 + (j-2)]*tmp_coefs_z[2];
                acmz_vec[3] = tmp_p[mult2 + (j-1)]*tmp_coefs_z[3];
                acmz_vec[4] = tmp_p[mult2 + (j+0)]*tmp_coefs_z[4];
                acmz_vec[5] = tmp_p[mult2 + (j+1)]*tmp_coefs_z[5];
                acmz_vec[6] = tmp_p[mult2 + (j+2)]*tmp_coefs_z[6];
                acmz_vec[7] = tmp_p[mult2 + (j+3)]*tmp_coefs_z[7];
                acmz_vec[8] = tmp_p[mult2 + (j+4)]*tmp_coefs_z[8];
                
                acmx_vec[0] = tmp_p[((0)*NZ) + j]*tmp_coefs_x[0];
                acmx_vec[1] = tmp_p[((1)*NZ) + j]*tmp_coefs_x[1];
                acmx_vec[2] = tmp_p[((2)*NZ) + j]*tmp_coefs_x[2];
                acmx_vec[3] = tmp_p[((3)*NZ) + j]*tmp_coefs_x[3];
                acmx_vec[4] = tmp_p[((4)*NZ) + j]*tmp_coefs_x[4];
                acmx_vec[5] = tmp_p[((5)*NZ) + j]*tmp_coefs_x[5];
                acmx_vec[6] = tmp_p[((6)*NZ) + j]*tmp_coefs_x[6];
                acmx_vec[7] = tmp_p[((7)*NZ) + j]*tmp_coefs_x[7];
                acmx_vec[8] = tmp_p[((8)*NZ) + j]*tmp_coefs_x[8];
                
                tmp_z = (acmz_vec[0]+acmz_vec[1]+acmz_vec[2]+acmz_vec[3]+
                    acmz_vec[4]+acmz_vec[5]+acmz_vec[6]+acmz_vec[7]+acmz_vec[8]);
                tmp_x = (acmx_vec[0]+acmx_vec[1]+acmx_vec[2]+acmx_vec[3]+
                    acmx_vec[4]+acmx_vec[5]+acmx_vec[6]+acmx_vec[7]+acmx_vec[8]);

                laplace[mult0+j] = tmp_x + tmp_z;
        }
    }
}

