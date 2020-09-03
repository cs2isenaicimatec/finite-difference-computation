#define NZ              295
#define NX              415
#define ORDER           8
#define HALF_ORDER      (ORDER/2)
#define ORDER_LENGTH    9//(ORDER +1)
#define NX_START        (HALF_ORDER)
#define NZ_START        (HALF_ORDER)
#define NX_END          (NX-HALF_ORDER)
#define NZ_END          (NZ-HALF_ORDER)
#define NXNZ            (NX*NZ) // total points on the img
#define NXNZ_HALF       (NX_HALF*NZ)
#define NSHIFTREGS      37
#define PREFETCH_LENGTH (ORDER_LENGTH*NZ)
#define POSFETCH_LENGTH NSHIFTREGS

#define SLICE_LENGTH    (((NX-ORDER)/NSHIFTREGS))
#define SHIFT_START     0
#define SHIFT_END       ((SLICE_LENGTH+ORDER_LENGTH)*NZ)

#define X_START         (HALF_ORDER)
#define X_END           (SLICE_LENGTH) 

#define FLOAT_CONV(SRC, DEST, VAL) SRC=VAL;(DEST=*((float*)&SRC));         

__kernel void laplacian (   __global const float  * restrict p,  
                            __global float *restrict laplace){

    int i0, i1, i2, i3, j0, j1, j2, k0, k1, k2,k3;
    int x[NSHIFTREGS], z[NSHIFTREGS], xstart[NSHIFTREGS], xend[NSHIFTREGS];
    int sidx, lidx[NSHIFTREGS];
    
    //Shift reg contains the equivalent of 9 lines of the image
    float shiftreg_in[NSHIFTREGS][PREFETCH_LENGTH];
    __global float * laplaces_out[NSHIFTREGS];

    float acmz[NSHIFTREGS][ORDER_LENGTH],acmx[NSHIFTREGS][ORDER_LENGTH];
    float accx[NSHIFTREGS], accz[NSHIFTREGS], accout[NSHIFTREGS];

    /////////////////////////////////////////////////////////////
    int float_conv_v;
    #pragma unroll NSHIFTREGS
    for (k0=0; k0<NSHIFTREGS; k0++){
        FLOAT_CONV(float_conv_v, acmz[k0][0],0xB795CBEC);
        FLOAT_CONV(float_conv_v, acmz[k0][1],0x3985270B);
        FLOAT_CONV(float_conv_v, acmz[k0][2],0xBB03126F);
        FLOAT_CONV(float_conv_v, acmz[k0][3],0x3C83126F);
        FLOAT_CONV(float_conv_v, acmz[k0][4],0xBCE93E94);
        FLOAT_CONV(float_conv_v, acmz[k0][5],0x3C83126F);
        FLOAT_CONV(float_conv_v, acmz[k0][6],0xBB03126F);
        FLOAT_CONV(float_conv_v, acmz[k0][7],0x3985270B);
        FLOAT_CONV(float_conv_v, acmz[k0][8],0xB795CBEC);
        FLOAT_CONV(float_conv_v, acmx[k0][0],0xB795CBEC);
        FLOAT_CONV(float_conv_v, acmx[k0][1],0x3985270B);
        FLOAT_CONV(float_conv_v, acmx[k0][2],0xBB03126F);
        FLOAT_CONV(float_conv_v, acmx[k0][3],0x3C83126F);
        FLOAT_CONV(float_conv_v, acmx[k0][4],0xBCE93E94);
        FLOAT_CONV(float_conv_v, acmx[k0][5],0x3C83126F);
        FLOAT_CONV(float_conv_v, acmx[k0][6],0xBB03126F);
        FLOAT_CONV(float_conv_v, acmx[k0][7],0x3985270B);
        FLOAT_CONV(float_conv_v, acmx[k0][8],0xB795CBEC);
    }
    
    #pragma unroll NSHIFTREGS
    for (k1=0; k1<NSHIFTREGS; k1++){
        z[k1]       = NZ_START;
        xstart[k1]  = HALF_ORDER + k1*SLICE_LENGTH;
        xend[k1]    = (HALF_ORDER + k1*SLICE_LENGTH) + SLICE_LENGTH + HALF_ORDER-1;
        x[k1]       = HALF_ORDER + k1*SLICE_LENGTH;
        laplaces_out[k1] = &laplace[xstart[k1]*NZ];
        accz[k1] = 0.f;
        accx[k1] = 0.f;
    }

    #pragma ivdep
    for (i0=SHIFT_START; i0<(SHIFT_END+(NZ-ORDER_LENGTH)); i0++){
        #pragma unroll NSHIFTREGS
        for (i1=0; i1<NSHIFTREGS; i1++){
            #pragma unroll
            for (j0=0; j0<PREFETCH_LENGTH-1; j0++){
                shiftreg_in[i1][j0] = shiftreg_in[i1][j0+1];
            }
            sidx = (xstart[i1] - HALF_ORDER)*NZ + i0;
            if (sidx < (NXNZ)){
                shiftreg_in[i1][PREFETCH_LENGTH-1] = p[sidx];
            }else{
                shiftreg_in[i1][PREFETCH_LENGTH-1] = 0.f;
            }
            #pragma unroll ORDER_LENGTH
            for (j1=0; j1<ORDER_LENGTH;j1++){
                accz[i1] += acmz[i1][j1]*shiftreg_in[i1][NX_START*NZ + j1];
                accx[i1] += acmx[i1][j1]*shiftreg_in[i1][j1*NZ + NZ_START];
            }
            accout[i1] = accx[i1]+accz[i1];
            if (i0 >= PREFETCH_LENGTH-1){
                if (x[i1] >= xstart[i1] && x[i1] <= (xend[i1]-HALF_ORDER) 
                    && z[i1] >= NZ_START && z[i1] < NZ_END)
                {
                    laplaces_out[i1][(i0-(PREFETCH_LENGTH-1) + HALF_ORDER)] = accout[i1];
                }
                if(z[i1]>=NZ-1){
                    z[i1] = 0;
                    x[i1] = x[i1]>=xend[i1]-1?x[i1]:x[i1]+1;
                }else{
                    z[i1]++;
                }
            }
            accz[i1] = 0.f;
            accx[i1] = 0.f;
        }
    }
}

