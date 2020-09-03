#ifndef NZ
#define NZ                  375
#endif
#ifndef NX
#define NX                  369
#endif
#ifndef NXB
#define NXB                 40
#endif
#ifndef NZB
#define NZB                 40
#endif

#define NZE                 (NZ+2*NZB) // 375 + 80  
#define NXE                 (NX+2*NXB) // 369 + 80

#define ORDER           8
#define HALF_ORDER      (ORDER/2)
#define ORDER_LENGTH    9//(ORDER +1)
#define NXE_START        (HALF_ORDER)
#define NZE_START        (HALF_ORDER)
#define NXE_END          (NXE-HALF_ORDER)
#define NZE_END          (NZE-HALF_ORDER)
#define NXENZE            (NXE*NZE) // total points on the img
#define NXENZE_HALF       (NXE_HALF*NZE)
#define NSHIFTREGS      1
#define PREFETCH_LENGTH (ORDER_LENGTH*NZE)
#define POSFETCH_LENGTH NSHIFTREGS
#define LPBUFFER_LENGTH (HALF_ORDER*NZE + HALF_ORDER +1)

#define SLICE_LENGTH    (((NXE-ORDER)/NSHIFTREGS))
#define SHIFT_START     0
#define SHIFT_END       ((SLICE_LENGTH+ORDER_LENGTH)*NZE)

#define X_START         (HALF_ORDER)
#define X_END           (SLICE_LENGTH) 

#define FLOAT_CONV(SRC, DEST, VAL) SRC=VAL;(DEST=*((float*)&SRC));         

__kernel void fdstep(  __global const float   * restrict p, 
                        __global const float   * restrict v2,
                        __global const float   * restrict pp_in,
                        const float dt2, 
                        __global float *restrict pp_out
                        ){

    int i0, i1, i2, i3, j0, j1, j2, k0, k1, k2,k3;
    int x, z, xstart, xend;
    int sidx, lidx;
    
    //Shift reg contains the equivalent of 9 lines of the image
    float pin_sreg   [PREFETCH_LENGTH];
    float v2_sreg    [PREFETCH_LENGTH];
    float laplace_sreg[LPBUFFER_LENGTH];

    
    float accx, accz;
    float ch_coefs_x[ORDER_LENGTH], ch_coefs_z[ORDER_LENGTH];

    /////////////////////////////////////////////////////////////
    int float_conv_v;
    FLOAT_CONV(float_conv_v, ch_coefs_z[0],0xB795CBEC);
    FLOAT_CONV(float_conv_v, ch_coefs_z[1],0x3985270B);
    FLOAT_CONV(float_conv_v, ch_coefs_z[2],0xBB03126F);
    FLOAT_CONV(float_conv_v, ch_coefs_z[3],0x3C83126F);
    FLOAT_CONV(float_conv_v, ch_coefs_z[4],0xBCE93E94);
    FLOAT_CONV(float_conv_v, ch_coefs_z[5],0x3C83126F);
    FLOAT_CONV(float_conv_v, ch_coefs_z[6],0xBB03126F);
    FLOAT_CONV(float_conv_v, ch_coefs_z[7],0x3985270B);
    FLOAT_CONV(float_conv_v, ch_coefs_z[8],0xB795CBEC);
    FLOAT_CONV(float_conv_v, ch_coefs_x[0],0xB795CBEC);
    FLOAT_CONV(float_conv_v, ch_coefs_x[1],0x3985270B);
    FLOAT_CONV(float_conv_v, ch_coefs_x[2],0xBB03126F);
    FLOAT_CONV(float_conv_v, ch_coefs_x[3],0x3C83126F);
    FLOAT_CONV(float_conv_v, ch_coefs_x[4],0xBCE93E94);
    FLOAT_CONV(float_conv_v, ch_coefs_x[5],0x3C83126F);
    FLOAT_CONV(float_conv_v, ch_coefs_x[6],0xBB03126F);
    FLOAT_CONV(float_conv_v, ch_coefs_x[7],0x3985270B);
    FLOAT_CONV(float_conv_v, ch_coefs_x[8],0xB795CBEC);
    
    z       = NZE_START;
    xstart  = HALF_ORDER;
    xend    = (HALF_ORDER) + SLICE_LENGTH + HALF_ORDER-1;
    x       = HALF_ORDER;
    accz    = 0.f;
    accx    = 0.f;

    #pragma ivdep
    for (i0=SHIFT_START; 
        i0<((SHIFT_END+(NZE-ORDER_LENGTH))+((ORDER-1)*NZE + ORDER)); 
        i0++){

        //////////////////////////////////////////////////////////////
        // Shift regs left
        #pragma unroll
        for (j0=0; j0<PREFETCH_LENGTH-1; j0++){
            pin_sreg[j0]  = pin_sreg[j0+1];
            v2_sreg[j0] = v2_sreg[j0+1];
        }
        #pragma unroll
        for (j2=0; j2<(LPBUFFER_LENGTH-1); j2++){
            laplace_sreg[j2] = laplace_sreg[j2+1];
        }
        //////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////
        // insert inputs P and V2
        sidx = (xstart - HALF_ORDER)*NZE + i0;
        if (sidx < (NXENZE)){
           pin_sreg[PREFETCH_LENGTH-1]  = p[sidx];
           v2_sreg[PREFETCH_LENGTH-1] = v2[sidx];
            //pin_sreg[PREFETCH_LENGTH-1]  = sidx;
            //v2_sreg[PREFETCH_LENGTH-1] = sidx;
        }else{
            pin_sreg[PREFETCH_LENGTH-1]  = 0.f;
            v2_sreg[PREFETCH_LENGTH-1] = 0.f;
        }
        //////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////
        // calc laplace_sreg    
        #pragma unroll ORDER_LENGTH
        for (j1=0; j1<ORDER_LENGTH;j1++){
            volatile float tmp_az = (ch_coefs_z[j1]*pin_sreg[NXE_START*NZE + j1]);
            volatile float tmp_ax = (ch_coefs_x[j1]*pin_sreg[j1*NZE + NZE_START]);

            accz = accz + tmp_az;
            accx = accx + tmp_ax;
        }
        if (i0 >= PREFETCH_LENGTH-1){
            if (x >= xstart && x < (NXE_END) 
            && z >= NZE_START && z < NZE_END)
            {
                laplace_sreg [(LPBUFFER_LENGTH-1)] = accz + accx;
            }else{
                laplace_sreg[(LPBUFFER_LENGTH-1)] = 0.f;
            }
            if(z < NZE-1){
                z++;
            }else{
                x = x< NXE-1? x+1:0;
                z = 0;
            }
        }else {
           laplace_sreg[(LPBUFFER_LENGTH-1)] = 0.f; 
        }
        accz = 0.f;
        accx = 0.f;
        //////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////
        // update pp
        if (i0 >= PREFETCH_LENGTH-1 && (i0-PREFETCH_LENGTH-1)< NXENZE){
            volatile float prev_dt2  = dt2;
            volatile float prev_v2dt = v2_sreg[0]*prev_dt2;
            volatile float prev_p    = 2*pin_sreg[0];
            volatile float prev_lapl = prev_v2dt*laplace_sreg[0];
            volatile float prev_pp   = pp_in[(i0-(PREFETCH_LENGTH-1))];
            volatile float new_pp0   = prev_p - prev_pp;
            volatile float new_pp1 = new_pp0  + prev_lapl;
            //if (new_pp1 == new_pp1)
            pp_out[(i0-(PREFETCH_LENGTH-1))]= new_pp1;
        }
    }
}