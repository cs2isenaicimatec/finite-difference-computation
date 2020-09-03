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

#ifndef NT
#define NT                  3004
#endif

#define NZE                 (NZ+2*NZB) // 375 + 80  
#define NXE                 (NX+2*NXB) // 369 + 80
#define ORDER               8
#define HALF_ORDER          (ORDER/2)
#define ORDER_LENGTH        9//(ORDER +1)
#define NXENZE              (NXE*NZE) // total points on the img
#define NXENZE_HALF         (NXE_HALF*NZE)
#define NSHIFTREGS          1
#define PREFETCH_LENGTH     (ORDER_LENGTH*NZE)
#define POSFETCH_LENGTH     NSHIFTREGS
#define LPBUFFER_LENGTH     (HALF_ORDER*NZE + HALF_ORDER +1)
#define UPBBUFFER_LENGTH    (NT*NXE*HALF_ORDER)

#define SLICE_LENGTH        (((NXE-ORDER)/NSHIFTREGS))
#define SHIFT_END           ((SLICE_LENGTH+ORDER_LENGTH)*NZE)

#define XLAP_START           (HALF_ORDER)
#define XLAP_END             (NXE-HALF_ORDER)
#define ZLAP_START           (HALF_ORDER)
#define ZLAP_END             (NZE-HALF_ORDER)

// time iteration limit
#define IT_END              ((SHIFT_END+(NZE-ORDER_LENGTH))+((ORDER-1)*NZE + ORDER))

#define NULL                0
#define FALSE               0
#define TRUE                1
#define END_SHIFT_VAL       0.0f

#define FLOAT_CONV(SRC, DEST, VAL) SRC=VAL;(DEST=*((float*)&SRC)); 

float getTaperxr(int x, int z, const float * ch_taperx){
    if(z >= 0 && z < NZB){
        if (x >= 0 && x < NXB){
            return 1.;
        }else if(x >= NXE-NXB && x < NXE){
            volatile int k = NXB - 1 - (x - (NXE-NXB)); 
            return ch_taperx[k];
        }else{
           return 1.;
        }
    }else{
        return 1.;
    }
}

float getTaperxl(int x, int z, const float * ch_taperx){
    if(z >= 0 && z < NZB){
        if (x >= 0 && x < NXB){
            return ch_taperx[x];
        }else{
            return 1.;
        }
    }else{
        return 1.;
    }
}

float getTaperz(int x, int z, const float * ch_taperz){
    if(z >= 0 && z < NZB){
        return ch_taperz[z];
    }else {
        return 1.;
    }
}
float taper_apply(
    float PP, 
    const float ch_taperxl,
    const float ch_taperxr,  
    const float ch_taperz)
{
    volatile  float p0;
    volatile  float p1;
    volatile  float p2;
    p0  = PP*ch_taperz;
    p1  = p0*ch_taperxl;
    p2  = p1*ch_taperxr;
    return p2;
} 

void inc_xz(volatile int *x, volatile int *z)
{
    volatile int xv = *x;
    volatile int zv = *z;
    if(zv < NZE-1){
        zv++;
    }else{
        xv = xv< NXE-1? xv+1:0;
        zv = 0;
    }
    *x = xv;
    *z = zv;
}

float update_pp(volatile float pp,
                volatile float pin,
                volatile float v2dt2,
                volatile float lapl)
{
    ////////////////////////////////////////////////////////////////
    volatile float npp = 2*pin - pp + v2dt2*lapl;
    return npp;
    ////////////////////////////////////////////////////////////////
}
  
__kernel void rtmforward(   const int                         srcx,
                            const int                         srcz,
                            __global const float   * restrict coefx,
                            __global const float   * restrict coefz,
                            __global const float   * restrict v2dt2,
                            __global const float   * restrict taperx,
                            __global const float   * restrict taperz,
                            __global const float   * restrict srcwavelet,
                            __global float         * restrict p, 
                            __global float         * restrict pp,
                            __global float         * restrict upb
                        )
{

    ///////////////////////////////////////
    // var declarations
    
    // counters
    int gcnt, icnt, tcnt, ucnt, i0, i1, i2;
    int i3, j0, j1, j2, k0, k1, k2,k3;
    volatile int xin, zin, xlap, zlap, xout, zout;
    volatile int sidx, lidx, oidx;
    
    //Shift regs 
    float pin_sreg      [PREFETCH_LENGTH];
    float pp_sreg       [PREFETCH_LENGTH];
    float v2dt2_sreg    [PREFETCH_LENGTH];
    float laplace_sreg  [LPBUFFER_LENGTH];

    // laplacian
    float accx, accz;
    float ch_coefs_x[ORDER_LENGTH], ch_coefs_z[ORDER_LENGTH];
    float ch_taperx[NXB], ch_taperz[NXB];

    // pp update
    volatile float pinVal;
    volatile float v2dt2Val;
    volatile float ppVal;
    volatile float srcVal;
    volatile char  savePPFlag;
    volatile char  saveUPBFlag;

    volatile float tapered_pp;
    volatile float new_pp0;
    volatile float ptpxr, ptpxl, ptpz;
    volatile float pptpxr, pptpxl, pptpz;

    /////////////////////////////////////////////////////////////
    // cache coeficients and taper values
    #pragma unroll ORDER_LENGTH
    for (i0=0; i0<ORDER_LENGTH; i0++){
        ch_coefs_x[i0] = coefx[i0];
        ch_coefs_z[i0] = coefz[i0];
    }

    #pragma unroll NXB
    for (i1=0; i1<NXB; i1++){
        ch_taperx[i1] = taperx[i1];
        ch_taperz[i1] = taperz[i1];
    }
    
    // initialize vars
    zlap=ZLAP_START; xlap=XLAP_START;
    xin=0;zin=0; xout=0;zout=0;
    accz=0.f;accx=0.f;lidx=0;sidx=0, oidx=0;
    icnt=0;tcnt=0;gcnt= 0;ucnt=0;
    savePPFlag  = 0;
    saveUPBFlag = 0;
    /////////////////////////////////////////////////////////////

    #pragma ivdep
    for (gcnt=0; gcnt<IT_END*NT; gcnt++){
        ///////////////////////////////////////////////////////////
        // LOADING INPUTS (optimizing mem access pattern)
        if(icnt < NXENZE){
            if(tcnt & 0x01){ // swap operation
                pinVal     = pp[icnt];
                ppVal      = p[icnt];
            }else{
                pinVal     = p[icnt];
                ppVal      = pp[icnt];
            }
            v2dt2Val   = v2dt2[icnt];
        }else{
            pinVal     = 0.f;
            ppVal      = 0.f;
        }
        srcVal     = srcwavelet[tcnt];  
        // get taper values for the current P and PP points
        ptpxl  = getTaperxl(xin, zin, ch_taperx);
        ptpxr  = getTaperxr(xin, zin, ch_taperx);
        ptpz   = getTaperz(xin, zin, ch_taperz);
        pptpxl = getTaperxl(xout, zout, ch_taperx);
        pptpxr = getTaperxr(xout, zout, ch_taperx);
        pptpz  = getTaperz(xout, zout, ch_taperz);

        //////////////////////////////////////////////////////////////
        // Shift regs left
        #pragma unroll
        for (j0=0; j0<PREFETCH_LENGTH-1; j0++){
            pin_sreg[j0]       = pin_sreg[j0+1];
            pp_sreg[j0]        = pp_sreg[j0+1];
            v2dt2_sreg[j0]     = v2dt2_sreg[j0+1];
        }
        #pragma unroll
        for (j2=0; j2<(LPBUFFER_LENGTH-1); j2++){
            laplace_sreg[j2] = laplace_sreg[j2+1];
        }
        //////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////
        // insert inputs P and V2
        if (icnt < (NXENZE)){
            pin_sreg    [PREFETCH_LENGTH-1]   = taper_apply(pinVal, ptpxl, ptpxr, ptpz);
            v2dt2_sreg  [PREFETCH_LENGTH-1]   = v2dt2Val;
            pp_sreg     [PREFETCH_LENGTH-1]   = ppVal;
            inc_xz(&xin, &zin);
        }else{
            xin = 0; zin = 0;
            pin_sreg    [PREFETCH_LENGTH-1]   = END_SHIFT_VAL;
            v2dt2_sreg  [PREFETCH_LENGTH-1]   = END_SHIFT_VAL;
            pp_sreg     [PREFETCH_LENGTH-1]   = END_SHIFT_VAL;
        }
        //////////////////////////////////////////////////////////////
        #pragma unroll ORDER_LENGTH
        for (j1=0; j1<ORDER_LENGTH;j1++){
            volatile float tmp_az = (ch_coefs_z[j1]*pin_sreg[XLAP_START*NZE + j1]);
            volatile float tmp_ax = (ch_coefs_x[j1]*pin_sreg[j1*NZE + ZLAP_START]);
            accz = accz + tmp_az;
            accx = accx + tmp_ax;
        }
        //////////////////////////////////////////////////////////////
        // calc laplace_sreg    
        if (icnt >= PREFETCH_LENGTH-1){
            if (xlap >= XLAP_START && xlap < (XLAP_END) 
            && zlap >= ZLAP_START && zlap < ZLAP_END)
            {
                laplace_sreg [(LPBUFFER_LENGTH-1)] = accz + accx;
            }else{
                laplace_sreg [(LPBUFFER_LENGTH-1)] = END_SHIFT_VAL;
            }
            inc_xz(&xlap, &zlap);
        }else {
           laplace_sreg[(LPBUFFER_LENGTH-1)] = END_SHIFT_VAL; 
        }
        accz = 0.f;
        accx = 0.f;

        //////////////////////////////////////////////////////////////
        if (icnt >= PREFETCH_LENGTH-1 && (icnt-PREFETCH_LENGTH-1)< NXENZE){
            oidx = (icnt-(PREFETCH_LENGTH-1)); // output idx
            ///////////////////////////////////////////////////////////
            // tapper Z
            tapered_pp  = taper_apply(pp_sreg[0], pptpxl, pptpxr, pptpz);
            /////////////////////////////////////////////////////////
            // update pp
            new_pp0     = update_pp(tapered_pp, pin_sreg[0], 
                                     v2dt2_sreg[0], laplace_sreg[0]);
            //////////////////////////////////////////////////////////
            // apply source
            if (xout==srcx && zout==srcz){
                new_pp0 += srcVal;//srcwavelet[tcnt];
            }
            // // save up board
            if (zout >= NZB-HALF_ORDER && zout < NZB){
                saveUPBFlag = 1;
            }else {
                saveUPBFlag = 0;
            }
            savePPFlag = 1;
        }else{
            savePPFlag = 0;
            saveUPBFlag = 0;
            oidx = 0;
        }

        ///////////////////////////////////////////////////////////
        // STORING RESULTS
        if (saveUPBFlag){
            upb[ucnt] = new_pp0;
            ucnt = ucnt < UPBBUFFER_LENGTH-1?ucnt+1:0;
        }
        if (savePPFlag){
            // store tapered updated PP
            // store tapered P
            if(tcnt & 0x01){
                p[oidx]   = new_pp0; 
                pp[oidx]  = pin_sreg[0]; 
            }else{
                pp[oidx]  = new_pp0;
                p[oidx]   = pin_sreg[0];
            }
            // update xout zout and it
            inc_xz(&xout, &zout);
            if (xout==0 && zout==0){
                // printf("* CLFRWRD: it = %d / %d (%d %%) \n",
                // tcnt+1,NT,(100*(tcnt+1)/NT));
                if (tcnt < NT-1){
                    tcnt++; // inc it
                }else{
                    tcnt=0;
                }
            }
            savePPFlag = 0;
        }

        if (icnt < IT_END -1){
            icnt++;
        }else{
            icnt = 0;
        }
    }
}
