#define NZ              295
#define NX              415
#define ORDER           8
#define HALF_ORDER      (ORDER/2)
#define ORDER_LENGTH    (ORDER +1)
#define PREFETCH_SIZE   (ORDER_LENGTH*NZ)
#define NX_START        (HALF_ORDER)
#define NZ_START        (HALF_ORDER)
#define NX_END          (NX-HALF_ORDER)
#define NZ_END          (NZ-HALF_ORDER)
#define NXNZ            (NX*NZ) // total points on the img
#define NLAPLACE        8


/*---------------- Coefs Declaration ----------------*/ 
#define FLOAT_CONV(SRC, DEST, VAL) SRC=VAL;(DEST=*((float*)&SRC));

__kernel void fd_step(__global float *restrict p, __global float *restrict laplace,
     __global float *restrict pp, __global float *restrict v2, float dt2
    ){
    int float_conv_v; 
    float coefs[ORDER_LENGTH]; 
    FLOAT_CONV(float_conv_v, coefs[0],0xB795CBEC);
    FLOAT_CONV(float_conv_v, coefs[1],0x3985270B);
    FLOAT_CONV(float_conv_v, coefs[2],0xBB03126F);
    FLOAT_CONV(float_conv_v, coefs[3],0x3C83126F);
    FLOAT_CONV(float_conv_v, coefs[4],0xBCE93E94);
    FLOAT_CONV(float_conv_v, coefs[5],0x3C83126F);
    FLOAT_CONV(float_conv_v, coefs[6],0xBB03126F);
    FLOAT_CONV(float_conv_v, coefs[7],0x3985270B);
    FLOAT_CONV(float_conv_v, coefs[8],0xB795CBEC);
    int i, j;
    int mult;
    int aux, io;  

    float acmz_vec[9];
    float acmx_vec[9];

    float tmp_x, tmp_z, tmp_pp;

    for (i=HALF_ORDER;i < (NX - HALF_ORDER);i++ ){
        mult = i*NZ;
        #pragma unroll 16
        for (j=HALF_ORDER;j < (NZ - HALF_ORDER);j++ ){
            acmz_vec[0] = p[(mult) + (j-4)]*coefs[0];
            acmz_vec[1] = p[(mult) + (j-3)]*coefs[1];
            acmz_vec[2] = p[(mult) + (j-2)]*coefs[2];
            acmz_vec[3] = p[(mult) + (j-1)]*coefs[3];
            acmz_vec[4] = p[(mult) + (j+0)]*coefs[4];
            acmz_vec[5] = p[(mult) + (j+1)]*coefs[5];
            acmz_vec[6] = p[(mult) + (j+2)]*coefs[6];
            acmz_vec[7] = p[(mult) + (j+3)]*coefs[7];
            acmz_vec[8] = p[(mult) + (j+4)]*coefs[8];


            acmx_vec[0] = p[((i-4)*NZ) + j]*coefs[0];
            acmx_vec[1] = p[((i-3)*NZ) + j]*coefs[1];
            acmx_vec[2] = p[((i-2)*NZ) + j]*coefs[2];
            acmx_vec[3] = p[((i-1)*NZ) + j]*coefs[3];
            acmx_vec[4] = p[((i+0)*NZ) + j]*coefs[4];
            acmx_vec[5] = p[((i+1)*NZ) + j]*coefs[5];
            acmx_vec[6] = p[((i+2)*NZ) + j]*coefs[6];
            acmx_vec[7] = p[((i+3)*NZ) + j]*coefs[7];
            acmx_vec[8] = p[((i+4)*NZ) + j]*coefs[8];

            tmp_z = (acmz_vec[0]+acmz_vec[1]+acmz_vec[2]+acmz_vec[3]+acmz_vec[4]+acmz_vec[5]+acmz_vec[6]+acmz_vec[7]+acmz_vec[8]);
            tmp_x = (acmx_vec[0]+acmx_vec[1]+acmx_vec[2]+acmx_vec[3]+acmx_vec[4]+acmx_vec[5]+acmx_vec[6]+acmx_vec[7]+acmx_vec[8]);

            laplace[mult+j] = tmp_x + tmp_z;
        }
    }

    for(i=0;i<NX;i++){
        mult = i*NZ;
            for(j=0;j<NZ;j++){
            tmp_pp = 2.*p[mult+j] - pp[mult+j]; 
            pp[mult+j] = tmp_pp + v2[mult+j]*dt2*laplace[mult+j];
        }
    }

}

