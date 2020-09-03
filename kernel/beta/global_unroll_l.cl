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
#define NXNZ_HALF       (NX_HALF*NZ)
#define NSHIFTREGS       11

#define SLICE_LENGTH    (((NX-ORDER)/NSHIFTREGS))  // integer HALF_ORDERision
#define SHIFT_START     0
#define SHIFT_END       ((SLICE_LENGTH+ORDER_LENGTH)*NZ)//((SLICE_LENGTH+ORDER)*NZ)

#define X_START         (HALF_ORDER)
#define X_END           (SLICE_LENGTH)          


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
    int i0, i1, j0, j1, k0, k1;
    int x[NSHIFTREGS], z[NSHIFTREGS], xstart[NSHIFTREGS], xend[NSHIFTREGS];

    int sidx, lidx[NSHIFTREGS];
    
    //Shift reg contains the equivalent of 9 lines of the image
    float shiftreg[NSHIFTREGS][PREFETCH_SIZE];
    
    float acmz[NSHIFTREGS][ORDER_LENGTH],acmx[NSHIFTREGS][ORDER_LENGTH];
    float accx[NSHIFTREGS], accz[NSHIFTREGS];

    #pragma unroll NSHIFTREGS
    for (k1=0; k1<NSHIFTREGS; k1++){
        z[k1] = NZ_START;
        xstart[k1]  = HALF_ORDER + k1*SLICE_LENGTH;
        xend[k1]    = (HALF_ORDER + k1*SLICE_LENGTH) + SLICE_LENGTH + HALF_ORDER-1;
        x[k1]       = HALF_ORDER + k1*SLICE_LENGTH;
        // printf("PG=%d xst[%d] = %2d xend[%d] = %2d diff=%2d pstart=%3d pend=%3d \n", 
        //     SLICE_LENGTH, k1, xstart[k1], k1, xend[k1], xend[k1]-xstart[k1],
        //     xstart[k1]*NZ, (xend[k1])*NZ-1);
    }
    #pragma ivdep
    for (i0=SHIFT_START; i0<(SHIFT_END+(NZ)-ORDER_LENGTH); i0++){
        // update shifreg
            
        #pragma unroll NSHIFTREGS
        for (i1=0; i1<NSHIFTREGS; i1++){
            
            #pragma unroll
            for (j0=0; j0<PREFETCH_SIZE-1; j0++){
                shiftreg[i1][j0] = shiftreg[i1][j0+1];
            }
            sidx = (xstart[i1] - HALF_ORDER)*NZ + i0;

            if (sidx < (NXNZ)){
                shiftreg[i1][PREFETCH_SIZE-1] = p[sidx];
            }else{
                shiftreg[i1][PREFETCH_SIZE-1] = 0.f;
            }

            #pragma unroll 9
            for (j1=0; j1<ORDER_LENGTH; j1++){
                acmx[i1][j1] = shiftreg[i1][j1*NZ + NZ_START] * coefs[j1];
                acmz[i1][j1] = shiftreg[i1][NX_START*NZ + j1] * coefs[j1];
            }
            if (i0 >= PREFETCH_SIZE-1){
                if (x[i1] >= xstart[i1] && x[i1] <= (xend[i1]-HALF_ORDER) 
                    && z[i1] >= NZ_START && z[i1] < NZ_END){
                    accz[i1] = (acmz[i1][0]+acmz[i1][1]+acmz[i1][2]+acmz[i1][3]+acmz[i1][4]+acmz[i1][5]+acmz[i1][6]+acmz[i1][7]+acmz[i1][8]);
                    accx[i1] = (acmx[i1][0]+acmx[i1][1]+acmx[i1][2]+acmx[i1][3]+acmx[i1][4]+acmx[i1][5]+acmx[i1][6]+acmx[i1][7]+acmx[i1][8]);
                    lidx[i1] = x[i1]*NZ+z[i1];
                    laplace[lidx[i1]] = accx[i1]+accz[i1];
                }
                if(z[i1]>=NZ-1){
                    z[i1] = 0;
                    x[i1] = x[i1]>=xend[i1]-1?x[i1]:x[i1]+1;
                }else{
                    z[i1]++;
                }
            }
        }
    }
}

