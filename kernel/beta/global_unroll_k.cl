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
#define NXNZ            (NX*NZ) // total points on the img
#define NLAPLACE        8


/*---------------- Coefs Declaration ----------------*/ 
#define FLOAT_CONV(SRC, DEST, VAL) SRC=VAL;(DEST=*((float*)&SRC));

__kernel void fd_step(__global float *restrict p, __global float *restrict laplace){
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
    int i0, j0, j1, k0, k1, x0, z0;
    int zidx, xidx, lidx;
    
    //Shift reg contains the equivalent of 9 lines of the image
    float shiftreg[PREFETCH_SIZE];
    float acmz[ORDER_LENGTH],acmx[ORDER_LENGTH];
    float accx, accz, accout;

    x0 = NX_START;
    z0 = NZ_START;

    #pragma ivdep
    for (i0=0; i0<(NXNZ)+NZ-ORDER_LENGTH; i0++){
        // update shifreg
        #pragma unroll
        for (j0=0; j0<PREFETCH_SIZE-1; j0++){
            shiftreg[j0] = shiftreg[j0+1];
        }
        shiftreg[PREFETCH_SIZE-1] = i0 < NXNZ? p[i0]:0.f;

        #pragma unroll 9
        for (j1=0; j1<ORDER_LENGTH; j1++){
            acmx[j1] = shiftreg[j1*NZ + NZ_START] * coefs[j1];
            acmz[j1] = shiftreg[NX_START*NZ + j1] * coefs[j1];
        }
        
        if (i0 >= PREFETCH_SIZE-1){
            if (x0 >= NX_START && x0 < NX_END && z0 >= NZ_START && z0 < NZ_END){
                accz = (acmz[0]+acmz[1]+acmz[2]+acmz[3]+acmz[4]+acmz[5]+acmz[6]+acmz[7]+acmz[8]);
                accx = (acmx[0]+acmx[1]+acmx[2]+acmx[3]+acmx[4]+acmx[5]+acmx[6]+acmx[7]+acmx[8]);
                lidx = x0*NZ+z0;
                laplace[lidx] = accx+accz;
            }
            if(z0>=NZ-1){
                z0 = 0;
                x0 = x0>=NX-1?NX_START:x0+1;
            } else{
                z0++;
            }
        }
        // /printf ("> i0=%d x0=%d z0=%d value: %0.5f\n", i0, x0, z0, accout);
    }
}

