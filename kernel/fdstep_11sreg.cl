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
#define NSHIFTREGS      11
#define PREFETCH_LENGTH (ORDER_LENGTH*NZ)
#define POSFETCH_LENGTH NSHIFTREGS
#define LPBUFFER_LENGTH (HALF_ORDER*NZ + HALF_ORDER +1)

#define SLICE_LENGTH    (((NX-ORDER)/NSHIFTREGS))
#define SHIFT_START     0
#define SHIFT_END       ((SLICE_LENGTH+ORDER_LENGTH)*NZ)

#define X_START         (HALF_ORDER)
#define X_END           (SLICE_LENGTH)          

#define FLOAT_CONV(SRC, DEST, VAL) SRC=VAL;(DEST=*((float*)&SRC));

__kernel void fdstep(   __global const float   * restrict p, 
                        __global const float   * restrict v2,
                        __global const float   * restrict pp_in,
                        const float dt2, 
                        __global float *restrict pp_out
                        )
{

    int i0, i1, i2, i3, j0, j1, j2, k0, k1, k2,k3;
    int x[NSHIFTREGS], z[NSHIFTREGS]; 
    int xstart[NSHIFTREGS], xend[NSHIFTREGS];
    int sidx[NSHIFTREGS], lidx[NSHIFTREGS];
    
    //Shift reg contains the equivalent of 9 lines of the image
    float pin_sreg      [NSHIFTREGS][PREFETCH_LENGTH];
    float v2_sreg       [NSHIFTREGS][PREFETCH_LENGTH];
    float pp_sreg       [NSHIFTREGS][PREFETCH_LENGTH];
    float laplace_sreg  [NSHIFTREGS][LPBUFFER_LENGTH];
    float accx[NSHIFTREGS], accz[NSHIFTREGS];
    float ch_coefs_x[NSHIFTREGS][ORDER_LENGTH];
    float ch_coefs_z[NSHIFTREGS][ORDER_LENGTH];
    __global float * pp_out_ptrs[NSHIFTREGS];

    /////////////////////////////////////////////////////////////
    int float_conv_v;
    #pragma unroll NSHIFTREGS
    for (k0 = 0; k0 < NSHIFTREGS; k0++)
    {   
        FLOAT_CONV(float_conv_v, ch_coefs_z[k0][0],0xB795CBEC);
        FLOAT_CONV(float_conv_v, ch_coefs_z[k0][1],0x3985270B);
        FLOAT_CONV(float_conv_v, ch_coefs_z[k0][2],0xBB03126F);
        FLOAT_CONV(float_conv_v, ch_coefs_z[k0][3],0x3C83126F);
        FLOAT_CONV(float_conv_v, ch_coefs_z[k0][4],0xBCE93E94);
        FLOAT_CONV(float_conv_v, ch_coefs_z[k0][5],0x3C83126F);
        FLOAT_CONV(float_conv_v, ch_coefs_z[k0][6],0xBB03126F);
        FLOAT_CONV(float_conv_v, ch_coefs_z[k0][7],0x3985270B);
        FLOAT_CONV(float_conv_v, ch_coefs_z[k0][8],0xB795CBEC);
        FLOAT_CONV(float_conv_v, ch_coefs_x[k0][0],0xB795CBEC);
        FLOAT_CONV(float_conv_v, ch_coefs_x[k0][1],0x3985270B);
        FLOAT_CONV(float_conv_v, ch_coefs_x[k0][2],0xBB03126F);
        FLOAT_CONV(float_conv_v, ch_coefs_x[k0][3],0x3C83126F);
        FLOAT_CONV(float_conv_v, ch_coefs_x[k0][4],0xBCE93E94);
        FLOAT_CONV(float_conv_v, ch_coefs_x[k0][5],0x3C83126F);
        FLOAT_CONV(float_conv_v, ch_coefs_x[k0][6],0xBB03126F);
        FLOAT_CONV(float_conv_v, ch_coefs_x[k0][7],0x3985270B);
        FLOAT_CONV(float_conv_v, ch_coefs_x[k0][8],0xB795CBEC);
        
        z[k0]       = NZ_START;
        xstart[k0]  = HALF_ORDER +  k0*SLICE_LENGTH;
        xend[k0]    = (HALF_ORDER + k0*SLICE_LENGTH) + SLICE_LENGTH + HALF_ORDER-1;
        x[k0]       = HALF_ORDER +  k0*SLICE_LENGTH;
        accz[k0]    = 0.f;
        accx[k0]    = 0.f;
        pp_out_ptrs[k0] = &pp_out[(xstart[k0] - HALF_ORDER)*NZ];
    }
    
    #pragma ivdep
    for (i0=SHIFT_START; 
        i0<((SHIFT_END+(NZ-ORDER_LENGTH))+((ORDER-1)*NZ + ORDER)); 
        i0++){

        #pragma unroll NSHIFTREGS
        for (i1=0; i1<NSHIFTREGS; i1++){
            //////////////////////////////////////////////////////////////
            // Shift regs left
            #pragma unroll
            for (j0=0; j0<PREFETCH_LENGTH-1; j0++){
                pin_sreg[i1][j0]  = pin_sreg[i1][j0+1];
                v2_sreg [i1][j0]  = v2_sreg [i1][j0+1];
                pp_sreg [i1][j0]  = pp_sreg [i1][j0+1];
            }
            #pragma unroll
            for (j2=0; j2<(LPBUFFER_LENGTH-1); j2++){
                laplace_sreg[i1][j2] = laplace_sreg[i1][j2+1];
            }
            //////////////////////////////////////////////////////////////
            // insert inputs P and V2
            sidx[i1] = (xstart[i1] - HALF_ORDER)*NZ + i0;
            if (sidx[i1] < (NXNZ)){
               pin_sreg[i1][PREFETCH_LENGTH-1]  = p[sidx[i1]];
               v2_sreg [i1][PREFETCH_LENGTH-1] = v2[sidx[i1]];
               pp_sreg [i1][PREFETCH_LENGTH-1] = pp_in[sidx[i1]];
            }else{
                pin_sreg[i1][PREFETCH_LENGTH-1]  = 0.f;
                v2_sreg [i1][PREFETCH_LENGTH-1] = 0.f;
                pp_sreg [i1][PREFETCH_LENGTH-1] = 0.f;
            }
            //////////////////////////////////////////////////////////////

            //////////////////////////////////////////////////////////////
            // calc laplace_sreg    
            #pragma unroll ORDER_LENGTH
            for (j1=0; j1<ORDER_LENGTH;j1++){
                volatile float tmp_az = (ch_coefs_z[i1][j1]*pin_sreg[i1][NX_START*NZ + j1]);
                volatile float tmp_ax = (ch_coefs_x[i1][j1]*pin_sreg[i1][j1*NZ + NZ_START]);
                accz[i1] = accz[i1] + tmp_az;
                accx[i1] = accx[i1] + tmp_ax;
            }
            //////////////////////////////////////////////////////////////
            if (i0 >= PREFETCH_LENGTH-1){
                if (x[i1] >= xstart[i1] && x[i1] <= (xend[i1]-HALF_ORDER) 
                && z[i1] >= NZ_START && z[i1] < NZ_END)
                {
                    laplace_sreg [i1][(LPBUFFER_LENGTH-1)] = accz[i1] + accx[i1];
                }else if (x[i1] > (xend[i1]-HALF_ORDER)){
                    laplace_sreg [i1][(LPBUFFER_LENGTH-1)] = INFINITY;
                }else{
                    laplace_sreg [i1][(LPBUFFER_LENGTH-1)] = 0.f;
                }
                if(z[i1] < NZ-1){
                    z[i1]++;
                }else{
                    x[i1] = x[i1]>=xend[i1]-1?x[i1]:x[i1]+1;
                    z[i1] = 0;
                }
            }else {
               laplace_sreg[i1][(LPBUFFER_LENGTH-1)] = 0.f; 
            }
            accz[i1] = 0.f;
            accx[i1] = 0.f;
            //////////////////////////////////////////////////////////////
            // update pp
            if (i0 >= PREFETCH_LENGTH-1 && (i0-PREFETCH_LENGTH-1)< NXNZ){
                volatile float prev_dt2  = dt2;
                volatile float prev_v2dt = v2_sreg[i1][0]*prev_dt2;
                volatile float prev_p    = 2*pin_sreg[i1][0];
                volatile float prev_lapl = prev_v2dt*laplace_sreg[i1][0];
                volatile float prev_pp   = pp_sreg[i1][0];
                volatile float new_pp0   = prev_p - prev_pp;
                volatile float new_pp1 = new_pp0  + prev_lapl;

                if (laplace_sreg[i1][0] != INFINITY){
                    pp_out_ptrs[i1][(i0-(PREFETCH_LENGTH-1))]= new_pp1;
                }
            }
        }
    }
}
