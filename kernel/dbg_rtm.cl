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

/* REMEMBER: NSHIFTREGS < (NXE/ORDER_LENGTH) */
#define NSHIFTREGS          4
#define NTFIXED             ((((int)NT/NSHIFTREGS)+1)*NSHIFTREGS)
#define NTSTEP              ((int)NTFIXED/NSHIFTREGS)
#define NZE                 (NZ+2*NZB) // 375 + 80  
#define NXE                 (NX+2*NXB) // 369 + 80
#define ORDER               8
#define HALF_ORDER          (ORDER/2)
#define ORDER_LENGTH        9//(ORDER +1)
#define NXENZE              (NXE*NZE) // total points on the img
#define NXENZE_HALF         (NXE_HALF*NZE)


#define PREFECTH_LOOPS      (((((int)NXE/ORDER_LENGTH)+1)*ORDER_LENGTH)/ORDER_LENGTH)
#define PREFETCH_LENGTH     (ORDER_LENGTH*NZE)
#define LPBUFFER_LENGTH     (HALF_ORDER*NZE + HALF_ORDER +1)
#define UPBBUFFER_LENGTH    (NT*NXE*HALF_ORDER)
#define SREG_LOOP_LIMIT     ((PREFETCH_LENGTH*PREFECTH_LOOPS)+PREFETCH_LENGTH)
#define GLOOP_LIMIT         (SREG_LOOP_LIMIT*NTSTEP)

#define XLAP_START          (HALF_ORDER)
#define XLAP_END            (NXE-HALF_ORDER)
#define ZLAP_START          (HALF_ORDER)
#define ZLAP_END            (NZE-HALF_ORDER)

#define NULL                0
#define FALSE               0
#define TRUE                1
#define END_SHIFT_VAL       0.0f

#define SREGSTATE_ON     0x01
#define SREGSTATE_OFF    0x00

typedef struct DataPathValues{
    float pVal    ;
    float v2dt2Val;
    float ppVal   ;
    float dobsVal  ;
    float outpVal ;
    float outppVal;
    float lapVal  ;
    float newppVal;
    float srcVal;
    float tapered_pp;
    float accx;
    float accz;

    float ptpxr;
    float ptpxl;
    float ptpz;
    float pptpxr;
    float pptpxl;
    float pptpz;
}DataPathValues;

typedef struct ControlFlags{
    char savePPFlag;
    char saveUPBFlag;
    char sregStateInput;
    char sregStateOutput;
}ControlFlags;

typedef struct MemAccessPTR{
    unsigned int pcnt; 
    unsigned int dcnt;
    unsigned int ucntbase;
    unsigned int ucnt;
    unsigned int itcnt;
    unsigned int oidx;
    unsigned int imcnt;
}MemAccessPTR;

typedef struct XZCounters{
    int  xin;
    int  zin;
    int  xlap;
    int  zlap;
    int  xout;
    int  zout;
}XZCounters;

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
    volatile  float p0, p1, p2;
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
    int io, i0, i1, i2, j0, j1, j2, k0, k1, sregId;
    volatile unsigned int gcnt;
    // circular shift regs structure
    float pin_sreg      [NSHIFTREGS][PREFETCH_LENGTH];
    float pp_sreg       [NSHIFTREGS][PREFETCH_LENGTH];
    float v2dt2_sreg    [NSHIFTREGS][PREFETCH_LENGTH];
    float laplace_sreg  [NSHIFTREGS][LPBUFFER_LENGTH];

    float ch_coefs_x[ORDER_LENGTH], ch_coefs_z[ORDER_LENGTH];
    float ch_taperx[NXB], ch_taperz[NXB];

    DataPathValues dirValues  [NSHIFTREGS];
    ControlFlags   dirFlags   [NSHIFTREGS];
    MemAccessPTR   dirPtr     [NSHIFTREGS];
    XZCounters     dirXZ      [NSHIFTREGS];

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

    #pragma unroll NSHIFTREGS
    for (i2=0; i2<NSHIFTREGS; i2++){
        dirPtr[i2].itcnt = i2; // each sreg starts on a different IT
        dirPtr[i2].pcnt  = 0; 
        dirPtr[i2].ucnt  = 0;
        dirPtr[i2].oidx  = 0;
        
        dirXZ[i2].xin    = 0;
        dirXZ[i2].xout   = 0;
        dirXZ[i2].xlap   = XLAP_START;
        dirXZ[i2].zin    = 0;
        dirXZ[i2].zout   = 0;
        dirXZ[i2].zlap   = ZLAP_START;
        
        dirValues[i2].outppVal = 0.f;
        dirValues[i2].outpVal  = 0.f;
        dirValues[i2].accx = 0.f;
        dirValues[i2].accz = 0.f;
        
        dirFlags[i2].savePPFlag = 0;
        dirFlags[i2].saveUPBFlag = 0;
        if(i2==0){
            dirFlags[i2].sregStateInput  = SREGSTATE_ON;
        }else{
            dirFlags[i2].sregStateInput  = SREGSTATE_OFF;
        }
        dirFlags[i2].sregStateOutput = SREGSTATE_OFF;

    }    
    #pragma ivdep
    for (gcnt=0; gcnt<GLOOP_LIMIT; gcnt++){
        
        #pragma unroll NSHIFTREGS
        for (sregId=0; sregId<NSHIFTREGS; sregId++){

            #pragma unroll
            for (k0=0; k0<(LPBUFFER_LENGTH-1); k0++){
                laplace_sreg[sregId][k0] = laplace_sreg[sregId][k0+1];
            }
            #pragma unroll
            for (k1=0; k1<PREFETCH_LENGTH-1; k1++){
                pin_sreg    [sregId][k1] = pin_sreg[sregId][k1+1];
                pp_sreg     [sregId][k1] = pp_sreg[sregId][k1+1];
                v2dt2_sreg  [sregId][k1] = v2dt2_sreg[sregId][k1+1];
            }
            dirPtr[sregId].pcnt      = dirXZ[sregId].xin*NZE 
                                        + dirXZ[sregId].zin;
            dirPtr[sregId].ucntbase = dirPtr[sregId].itcnt*(NXE*HALF_ORDER) 
                                        + dirXZ[sregId].xout*HALF_ORDER; 
            dirPtr[sregId].ucnt      = dirPtr[sregId].ucntbase + 
                                        (dirXZ[sregId].zout - 
                                            (NZB-HALF_ORDER));
            dirPtr[sregId].oidx      = dirXZ[sregId].xout*NZE 
                                        + dirXZ[sregId].zout;
            /*****************************************************/
            /* sreg inputs */
            if(dirFlags[sregId].sregStateInput==SREGSTATE_ON){
                if(sregId==0){ 
                    /*first sreg always gets from memory*/
                    dirValues[sregId].pVal  = p [dirPtr[sregId].pcnt];
                    dirValues[sregId].ppVal = pp[dirPtr[sregId].pcnt];
                }else{
                    dirValues[sregId].pVal  = dirValues[sregId-1].outppVal;
                    dirValues[sregId].ppVal = dirValues[sregId-1].outpVal;
                }
                dirValues[sregId].v2dt2Val  = v2dt2[dirPtr[sregId].pcnt];
            }else{
                dirValues[sregId].pVal    = END_SHIFT_VAL;
                dirValues[sregId].ppVal   = END_SHIFT_VAL;
                dirValues[sregId].v2dt2Val= END_SHIFT_VAL;
            }
            /*****************************************************/
            // taper apply
            if(dirFlags[sregId].sregStateInput==SREGSTATE_ON){
                dirValues[sregId].ptpxl = getTaperxl(dirXZ[sregId].xin, 
                    dirXZ[sregId].zin, ch_taperx);
                dirValues[sregId].ptpxr = getTaperxr(dirXZ[sregId].xin, 
                    dirXZ[sregId].zin, ch_taperx);
                dirValues[sregId].ptpz  = getTaperz(dirXZ[sregId].xin, 
                    dirXZ[sregId].zin, ch_taperz);
            }else{
                dirValues[sregId].ptpxl = 1.0;
                dirValues[sregId].ptpxr = 1.0;
                dirValues[sregId].ptpz  = 1.0;
            }

            pin_sreg[sregId][PREFETCH_LENGTH-1] = taper_apply(
                                                dirValues[sregId].pVal, 
                                                dirValues[sregId].ptpxl, 
                                                dirValues[sregId].ptpxr, 
                                                dirValues[sregId].ptpz);
            v2dt2_sreg[sregId][PREFETCH_LENGTH-1] = dirValues[sregId].v2dt2Val;
            pp_sreg   [sregId][PREFETCH_LENGTH-1] = dirValues[sregId].ppVal;

            /*****************************************************/
            #pragma unroll ORDER_LENGTH
            for (io=0; io<ORDER_LENGTH;io++){
                volatile float tmp_az = (ch_coefs_z[io]*pin_sreg[sregId][XLAP_START*NZE + io]);
                volatile float tmp_ax = (ch_coefs_x[io]*pin_sreg[sregId][io*NZE + ZLAP_START]);
                dirValues[sregId].accz = dirValues[sregId].accz + tmp_az;
                dirValues[sregId].accx = dirValues[sregId].accx + tmp_ax;
            }
            dirValues[sregId].lapVal = dirValues[sregId].accz
                 +  dirValues[sregId].accx;
            /*****************************************************/
            /* calc laplace_sreg    */
            if (dirFlags[sregId].sregStateOutput == SREGSTATE_ON){
                if(dirPtr[sregId].itcnt<NT){
                    dirValues[sregId].srcVal = 
                    srcwavelet[dirPtr[sregId].itcnt];
                }else{
                    dirValues[sregId].srcVal  = 0.;
                }
                if (dirXZ[sregId].xlap >= XLAP_START 
                    && dirXZ[sregId].xlap <  (XLAP_END) 
                    && dirXZ[sregId].zlap >= ZLAP_START 
                    && dirXZ[sregId].zlap <  ZLAP_END)
                {
                    laplace_sreg [sregId][(LPBUFFER_LENGTH-1)] = dirValues[sregId].lapVal;
                }else{
                    laplace_sreg [sregId][(LPBUFFER_LENGTH-1)] = END_SHIFT_VAL;
                }
                dirValues[sregId].pptpxl = getTaperxl(dirXZ[sregId].xout,
                                            dirXZ[sregId].zout,
                                            ch_taperx);
                dirValues[sregId].pptpxr = getTaperxr(dirXZ[sregId].xout,
                                            dirXZ[sregId].zout,
                                            ch_taperx);
                dirValues[sregId].pptpz  = getTaperz(dirXZ[sregId].xout,
                                            dirXZ[sregId].zout,
                                            ch_taperz);
                /*****************************************************/
                //tapper Z
                dirValues[sregId].tapered_pp  = taper_apply(pp_sreg[sregId][0], 
                                   dirValues[sregId].pptpxl, 
                                   dirValues[sregId].pptpxr, 
                                   dirValues[sregId].pptpz);

                /*****************************************************/
                // update pp
                dirValues[sregId].newppVal = update_pp(dirValues[sregId].tapered_pp, 
                                        pin_sreg[sregId][0], 
                                        v2dt2_sreg[sregId][0], 
                                        laplace_sreg[sregId][0]);
                /*****************************************************/
                // apply source
                if (dirXZ[sregId].xout==srcx && dirXZ[sregId].zout==srcz){
                    dirValues[sregId].newppVal += dirValues[sregId].srcVal;
                }
                // save up board
                if (dirXZ[sregId].zout >= NZB-HALF_ORDER && dirXZ[sregId].zout < NZB){
                    dirFlags[sregId].saveUPBFlag = TRUE;
                }
                dirFlags[sregId].savePPFlag = TRUE;
            }else{
                dirFlags[sregId].savePPFlag = 0;
                dirFlags[sregId].saveUPBFlag = 0;
                dirValues[sregId].pptpxl = 1.0;
                dirValues[sregId].pptpxr = 1.0;
                dirValues[sregId].pptpz  = 1.0;
                dirValues[sregId].newppVal = END_SHIFT_VAL;
                laplace_sreg[sregId][(LPBUFFER_LENGTH-1)] = END_SHIFT_VAL;
            }
            /*****************************************************/

            /*****************************************************/
            if (dirFlags[sregId].saveUPBFlag && dirPtr[sregId].itcnt < NT &&  dirPtr[sregId].ucnt < UPBBUFFER_LENGTH){
                upb[dirPtr[sregId].ucnt] = dirValues[sregId].newppVal;
                dirFlags[sregId].saveUPBFlag = FALSE;
            }
            if (dirFlags[sregId].savePPFlag){
                /* Stores in DDR if it's the last iteration,
                 * or last sreg on a valid IT*/
                if(dirPtr[sregId].itcnt == (NT-1)){ 
                    // last IT saves to memory
                    pp[dirPtr[sregId].oidx]  = dirValues[sregId].newppVal;
                    p[dirPtr[sregId].oidx]   = pin_sreg[sregId][0];
                }else if((sregId==(NSHIFTREGS-1))&&(dirPtr[sregId].itcnt < NT)){
                    // last sreg saves swapped ptrs 
                    // to memory if its IT < NT
                   p[dirPtr[sregId].oidx]  = dirValues[sregId].newppVal;
                   pp[dirPtr[sregId].oidx] = pin_sreg[sregId][0];
                }else{
                    dirValues[sregId].outpVal = pin_sreg[sregId][0]; // saves p for next it
                    dirValues[sregId].outppVal= dirValues[sregId].newppVal;
                }
                dirFlags[sregId].savePPFlag = FALSE;
            }
            /*****************************************************/
            /* update img counters */
            /* update img counters */
            if(sregId==0){
                inc_xz(&dirXZ[sregId].xin, &dirXZ[sregId].zin);
            }else{
                dirFlags[sregId].sregStateInput = dirFlags[sregId-1].sregStateOutput;
                dirXZ[sregId].xin = dirXZ[sregId-1].xout;
                dirXZ[sregId].zin = dirXZ[sregId-1].zout;
            }
            if(dirFlags[sregId].sregStateOutput==SREGSTATE_OFF){
                dirFlags[sregId].sregStateOutput = 
                            dirXZ[sregId].xin==ORDER && dirXZ[sregId].zin==(NZE-1)? 
                            SREGSTATE_ON:SREGSTATE_OFF;
            }else{
                inc_xz(&dirXZ[sregId].xlap, &dirXZ[sregId].zlap);
                inc_xz(&dirXZ[sregId].xout, &dirXZ[sregId].zout);
                if(dirXZ[sregId].xout==0 && dirXZ[sregId].zout==0){
                    // end of production
                    printf("> sreg=%d it=%d/%d\n",
                        sregId, dirPtr[sregId].itcnt, NT);
                    if(dirPtr[sregId].itcnt!=(NT-1)){
                        dirFlags[sregId].sregStateOutput = 
                            dirXZ[sregId].xin==ORDER && dirXZ[sregId].zin==(NZE-1)? 
                            SREGSTATE_ON:SREGSTATE_OFF;
                        dirPtr[sregId].itcnt += NSHIFTREGS;
                    }else{
                        return;
                    }
                }
            }
            dirValues[sregId].accx = 0.f;
            dirValues[sregId].accz = 0.f;
        }
    }
}
  
__kernel void rtmbackward(   const int                        dobsz,
                            __global const float   * restrict coefx,
                            __global const float   * restrict coefz,
                            __global const float   * restrict v2dt2,
                            __global const float   * restrict taperx,
                            __global const float   * restrict taperz,
                            __global const float   * restrict dobs,
                            __global const float   * restrict upb,
                            __global float         * restrict snaps0, 
                            __global float         * restrict snaps1, 
                            __global float         * restrict pr, 
                            __global float         * restrict ppr,
                            __global float         * restrict p,
                            __global float         * restrict pp,
                            __global float         * restrict imloc
                        )
{

    unsigned int gcnt, i0, i1, i2, j0, j1, k0, k1, io0, io1;
    unsigned int sregId;
    float v2dt2_sreg    [NSHIFTREGS][PREFETCH_LENGTH];
    float p_sreg        [NSHIFTREGS][PREFETCH_LENGTH];
    float pp_sreg       [NSHIFTREGS][PREFETCH_LENGTH];
    float lap_sreg      [NSHIFTREGS][LPBUFFER_LENGTH];

    float pr_sreg       [NSHIFTREGS][PREFETCH_LENGTH];
    float ppr_sreg      [NSHIFTREGS][PREFETCH_LENGTH];
    float lapr_sreg     [NSHIFTREGS][LPBUFFER_LENGTH];

    float ch_coefs_x[ORDER_LENGTH], ch_coefs_z[ORDER_LENGTH];
    float ch_taperx[NXB], ch_taperz[NXB];

    DataPathValues dirValues [NSHIFTREGS];
    DataPathValues revValues [NSHIFTREGS];
    ControlFlags  dirFlags   [NSHIFTREGS];
    ControlFlags  revFlags   [NSHIFTREGS];
    MemAccessPTR  dirPtr     [NSHIFTREGS];
    MemAccessPTR  revPtr     [NSHIFTREGS];
    XZCounters    dirXZ      [NSHIFTREGS];
    XZCounters    revXZ      [NSHIFTREGS];

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

     #pragma unroll NSHIFTREGS
    for (i2=0; i2<NSHIFTREGS; i2++){
        dirPtr[i2].itcnt = i2; // each sreg starts on a different IT
        dirPtr[i2].pcnt  = 0; 
        dirPtr[i2].oidx  = 0;
        dirPtr[i2].dcnt  = 0;
        dirXZ[i2].xin   = 0;
        dirXZ[i2].xlap  = XLAP_START;
        dirXZ[i2].xout  = 0;
        dirXZ[i2].zin   = 0;
        dirXZ[i2].zlap  = ZLAP_START;
        dirXZ[i2].zout  = 0;
        
        dirValues[i2].outppVal = 0.f;
        dirValues[i2].outpVal  = 0.f;
        
        dirValues[i2].accx = 0.f;
        dirValues[i2].accz = 0.f;
        
        dirFlags[i2].savePPFlag = 0;
        if(i2==0){
            dirFlags[i2].sregStateInput  = SREGSTATE_ON;
        }else{
            dirFlags[i2].sregStateInput  = SREGSTATE_OFF;
        }
        dirFlags[i2].sregStateOutput = SREGSTATE_OFF;


        revPtr[i2].itcnt = i2; // each sreg starts on a different IT
        revPtr[i2].pcnt  = 0; 
        revPtr[i2].oidx  = 0;
        revPtr[i2].dcnt  = 0;
        revXZ[i2].xin   = 0;
        revXZ[i2].xlap  = XLAP_START;
        revXZ[i2].xout  = 0;
        revXZ[i2].zin   = 0;
        revXZ[i2].zlap  = ZLAP_START;
        revXZ[i2].zout  = 0;
        
        revValues[i2].outppVal = 0.f;
        revValues[i2].outpVal  = 0.f;
        
        revValues[i2].accx = 0.f;
        revValues[i2].accz = 0.f;
        
        revFlags[i2].savePPFlag = 0;
        if(i2==0){
            revFlags[i2].sregStateInput  = SREGSTATE_ON;
        }else{
            revFlags[i2].sregStateInput  = SREGSTATE_OFF;
        }
        revFlags[i2].sregStateOutput = SREGSTATE_OFF;
    }

    #pragma ivdep
    for (gcnt=0; gcnt<GLOOP_LIMIT; gcnt++){

        #pragma unroll NSHIFTREGS
        for (sregId=0; sregId<NSHIFTREGS; sregId++){
            #pragma unroll
            for (k0=0; k0<(LPBUFFER_LENGTH-1); k0++){
                lap_sreg[sregId][k0]  = lap_sreg[sregId][k0+1];
                lapr_sreg[sregId][k0] = lapr_sreg[sregId][k0+1];
            }
            #pragma unroll
            for (k1=0; k1<PREFETCH_LENGTH-1; k1++){
                p_sreg      [sregId][k1] = p_sreg[sregId][k1+1];
                pp_sreg     [sregId][k1] = pp_sreg[sregId][k1+1];
                pr_sreg     [sregId][k1] = pr_sreg[sregId][k1+1];
                ppr_sreg    [sregId][k1] = ppr_sreg[sregId][k1+1];
                v2dt2_sreg  [sregId][k1] = v2dt2_sreg[sregId][k1+1];
            }

            /******************************************************/
            /* mem access pointers update */
            {
                dirPtr[sregId].pcnt  = dirXZ[sregId].xin*NZE + 
                                        dirXZ[sregId].zin; 
                dirPtr[sregId].oidx  = dirXZ[sregId].xout*NZE + 
                                        dirXZ[sregId].zout;
                dirPtr[sregId].ucntbase = ((NT-1)-dirPtr[sregId].itcnt)*(NXE*HALF_ORDER)
                    + dirXZ[sregId].xout*HALF_ORDER; 
                dirPtr[sregId].ucnt = dirPtr[sregId].ucntbase
                    + (dirXZ[sregId].zout - (NZB-HALF_ORDER));



                revPtr[sregId].pcnt  = revXZ[sregId].xin*NZE + 
                                        revXZ[sregId].zin;
                revPtr[sregId].dcnt  = (revXZ[sregId].xout-NXB)*NT + ((NT-1) - 
                                        revPtr[sregId].itcnt);
                revPtr[sregId].oidx  = revXZ[sregId].xout*NZE + 
                                        revXZ[sregId].zout;
                revPtr[sregId].imcnt = (revXZ[sregId].xout-NXB)*NZ + 
                        (revXZ[sregId].zout-NZB);
            }
            /******************************************************/
            /* input data */
            {
                /* load direct input values */
                if(dirFlags[sregId].sregStateInput==SREGSTATE_ON){
                    if(sregId==0){
                        if(dirPtr[sregId].itcnt==0){// first iteration
                            dirValues[sregId].pVal = END_SHIFT_VAL;
                            dirValues[sregId].ppVal = snaps1[dirPtr[sregId].pcnt];
                        }else{
                            dirValues[sregId].pVal  = p[dirPtr[sregId].pcnt];
                            dirValues[sregId].ppVal = pp[dirPtr[sregId].pcnt];
                        }
                    }else{
                        if(dirPtr[sregId].itcnt==1){
                            dirValues[sregId].ppVal  = snaps0[dirPtr[sregId].pcnt];
                            dirValues[sregId].pVal   = dirValues[sregId-1].outpVal;
                        }else{
                            dirValues[sregId].ppVal  = dirValues[sregId-1].outpVal;
                            dirValues[sregId].pVal   = dirValues[sregId-1].outppVal;
                        }
                    }
                }else{
                    dirValues[sregId].pVal  = END_SHIFT_VAL;
                    dirValues[sregId].ppVal = END_SHIFT_VAL;
                }
                /* load reverse input values + vel2dt2 */
                if(revFlags[sregId].sregStateInput==SREGSTATE_ON){
                    if(sregId==0){
                        revValues[sregId].pVal  = pr[revPtr[sregId].pcnt];
                        revValues[sregId].ppVal = ppr[revPtr[sregId].pcnt];
                    }else{
                        revValues[sregId].pVal  = revValues[sregId-1].outppVal;
                        revValues[sregId].ppVal = revValues[sregId-1].outpVal;
                    }
                    revValues[sregId].v2dt2Val  = v2dt2[revPtr[sregId].pcnt];
                }else{
                    revValues[sregId].pVal  = END_SHIFT_VAL;
                    revValues[sregId].ppVal = END_SHIFT_VAL;
                    revValues[sregId].v2dt2Val = END_SHIFT_VAL;
                }
                //dirValues[sregId].v2dt2Val = revValues[sregId].v2dt2Val;
            }
            /* tapper input */
            {
                // taper apply
                if(revFlags[sregId].sregStateInput==SREGSTATE_ON){
                    revValues[sregId].ptpxl  = getTaperxl(revXZ[sregId].xin, 
                        revXZ[sregId].zin, ch_taperx);
                    revValues[sregId].ptpxr  = getTaperxr(revXZ[sregId].xin, 
                        revXZ[sregId].zin, ch_taperx);
                    revValues[sregId].ptpz   = getTaperz(revXZ[sregId].xin, 
                        revXZ[sregId].zin, ch_taperz);
                }else{
                    revValues[sregId].ptpxl  = 1.0;
                    revValues[sregId].ptpxr  = 1.0;
                    revValues[sregId].ptpz   = 1.0;
                }
                p_sreg[sregId][PREFETCH_LENGTH-1]   = dirValues[sregId].pVal;
                pp_sreg[sregId][PREFETCH_LENGTH-1]  = dirValues[sregId].ppVal;

                // tapper is applied only for reverse propagation
                pr_sreg[sregId][PREFETCH_LENGTH-1] = taper_apply(revValues[sregId].pVal, 
                                                           revValues[sregId].ptpxl, 
                                                           revValues[sregId].ptpxr, 
                                                           revValues[sregId].ptpz);
                ppr_sreg  [sregId][PREFETCH_LENGTH-1] = revValues[sregId].ppVal;
                v2dt2_sreg[sregId][PREFETCH_LENGTH-1] = revValues[sregId].v2dt2Val;
            }
            /* calc. laplacian */
            {
                #pragma unroll ORDER_LENGTH
                for (io0=0; io0<ORDER_LENGTH;io0++){
                    volatile float tmp_az = (ch_coefs_z[io0]*
                        p_sreg[sregId][XLAP_START*NZE + io0]);
                    volatile float tmp_ax = (ch_coefs_x[io0]*
                        p_sreg[sregId][io0*NZE + ZLAP_START]);
                    dirValues[sregId].accz = dirValues[sregId].accz + tmp_az;
                    dirValues[sregId].accx = dirValues[sregId].accx + tmp_ax;

                    // int ix = dirXZ[sregId].xlap;
                    // int iz = dirXZ[sregId].zlap;
                    // int it = dirPtr[sregId].itcnt;
                    // float pz = p_sreg[sregId][XLAP_START*NZE + io0];
                    // float px = p_sreg[sregId][io0*NZE + ZLAP_START];
                    // if(ix>=270 && ix<=290 && iz>=30 && iz<=50 && it==3){
                    //     printf("(%d)[%d][%d](%d)cz    = %.20f\n",it, ix, iz, io0, ch_coefs_z[io0] );
                    //     printf("(%d)[%d][%d](%d)cx    = %.20f\n",it, ix, iz, io0, ch_coefs_x[io0] );
                    //     printf("(%d)[%d][%d](%d)PZ    = %.20f\n",it, ix, iz, io0, pz);
                    //     printf("(%d)[%d][%d](%d)PX    = %.20f\n",it, ix, iz, io0, px);
                    //     printf("(%d)[%d][%d](%d)acm_z = %.20f\n",it, ix, iz, io0, dirValues[sregId].accz );
                    //     printf("(%d)[%d][%d](%d)acm_x = %.20f\n",it, ix, iz, io0, dirValues[sregId].accx);
                    //     printf("(%d)[%d][%d](%d)LAP   = %.20f\n",it, ix, iz, io0, dirValues[sregId].accz +  dirValues[sregId].accx);
                    // }
                }
                dirValues[sregId].lapVal = 
                    dirValues[sregId].accz +  dirValues[sregId].accx;

                #pragma unroll ORDER_LENGTH
                for (io1=0; io1<ORDER_LENGTH;io1++){
                    volatile float tmp_az = (ch_coefs_z[io1]*
                        pr_sreg[sregId][XLAP_START*NZE + io1]);
                    volatile float tmp_ax = (ch_coefs_x[io1]*
                        pr_sreg[sregId][io1*NZE + ZLAP_START]);
                    revValues[sregId].accz = revValues[sregId].accz + tmp_az;
                    revValues[sregId].accx = revValues[sregId].accx + tmp_ax;
                }
                revValues[sregId].lapVal = 
                    revValues[sregId].accz +  revValues[sregId].accx;
            }

            /* update output */
            {
                if (dirFlags[sregId].sregStateOutput == SREGSTATE_ON){
                    if (dirXZ[sregId].xlap >= XLAP_START 
                        && dirXZ[sregId].xlap <  (XLAP_END) 
                        && dirXZ[sregId].zlap >= ZLAP_START 
                        && dirXZ[sregId].zlap <  ZLAP_END)
                    {
                        lap_sreg [sregId][(LPBUFFER_LENGTH-1)] = dirValues[sregId].lapVal;
                    }else{
                        lap_sreg [sregId][(LPBUFFER_LENGTH-1)] = END_SHIFT_VAL;
                    }

                    /*****************************************************/
                    // update pp
                    if(dirPtr[sregId].itcnt > 1){
                        if (dirXZ[sregId].zout >= NZB-HALF_ORDER 
                            && dirXZ[sregId].zout < NZB &&
                            dirPtr[sregId].itcnt < NT){
                            dirValues[sregId].newppVal = upb[dirPtr[sregId].ucnt];
                        }else{
                            dirValues[sregId].newppVal = update_pp(pp_sreg[sregId][0], 
                                                    p_sreg[sregId][0], 
                                                    v2dt2_sreg[sregId][0], 
                                                    lap_sreg[sregId][0]);
                        }
                    }else{
                        dirValues[sregId].newppVal = pp_sreg[sregId][0];
                    }
                    // if(dirPtr[sregId].itcnt==1){
                    //     int ix = dirXZ[sregId].xout;
                    //     int iz = dirXZ[sregId].zout;
                    //     dirValues[sregId].tapered_pp = pp_sreg[sregId][0];
                    //     printf("[%d][%d]prev_pp = %.30f \n",ix, iz, dirValues[sregId].tapered_pp);
                    //     printf("[%d][%d]prev_p  = %.30f \n",ix, iz, p_sreg[sregId][0]);
                    //     printf("[%d][%d]v2dt2   = %.30f \n",ix, iz, v2dt2_sreg[sregId][0]);
                    //     printf("[%d][%d]laplace = %.30f \n",ix, iz, lap_sreg[sregId][0]);
                    //     printf("[%d][%d]newpp   = %.30f \n",ix, iz, dirValues[sregId].newppVal);
                    // }
                    /*****************************************************/
                    dirFlags[sregId].savePPFlag = TRUE;
                }else{
                    dirFlags[sregId].savePPFlag = FALSE;
                    dirValues[sregId].newppVal  = END_SHIFT_VAL;
                    lap_sreg [sregId][(LPBUFFER_LENGTH-1)] = END_SHIFT_VAL;
                }

                if (revFlags[sregId].sregStateOutput == SREGSTATE_ON){
                    if (revXZ[sregId].xlap >= XLAP_START 
                        && revXZ[sregId].xlap <  (XLAP_END) 
                        && revXZ[sregId].zlap >= ZLAP_START 
                        && revXZ[sregId].zlap <  ZLAP_END)
                    {
                        lapr_sreg [sregId][(LPBUFFER_LENGTH-1)] = revValues[sregId].lapVal;
                    }else{
                        lapr_sreg [sregId][(LPBUFFER_LENGTH-1)] = END_SHIFT_VAL;
                    }
                    revValues[sregId].pptpxl = getTaperxl(revXZ[sregId].xout, 
                        revXZ[sregId].zout, ch_taperx);
                    revValues[sregId].pptpxr = getTaperxr(revXZ[sregId].xout, 
                        revXZ[sregId].zout, ch_taperx);
                    revValues[sregId].pptpz  = getTaperz(revXZ[sregId].xout, 
                        revXZ[sregId].zout, ch_taperz);
                    /*****************************************************/
                    //tapper Z
                    revValues[sregId].tapered_pp = taper_apply(
                                        ppr_sreg[sregId][0], 
                                        revValues[sregId].pptpxl, 
                                        revValues[sregId].pptpxr, 
                                        revValues[sregId].pptpz);

                    /*****************************************************/
                    // update pp
                    revValues[sregId].newppVal = update_pp(
                                            revValues[sregId].tapered_pp, 
                                            pr_sreg[sregId][0], 
                                            v2dt2_sreg[sregId][0], 
                                            lapr_sreg[sregId][0]);
                    /*****************************************************/
                    /* apply dobs */
                    if (revXZ[sregId].xout>=NXB 
                        && revXZ[sregId].xout < NXE-NXB
                        && revXZ[sregId].zout== dobsz
                        && revPtr[sregId].itcnt < NT){
                        revValues[sregId].newppVal += dobs[revPtr[sregId].dcnt];
                    }
                    revFlags[sregId].savePPFlag = TRUE;
                }else{
                    revFlags[sregId].savePPFlag = FALSE;
                    revValues[sregId].pptpxl =  1.0;
                    revValues[sregId].pptpxr =  1.0;
                    revValues[sregId].pptpz  =  1.0;
                    revValues[sregId].newppVal = END_SHIFT_VAL;
                    lapr_sreg [sregId][(LPBUFFER_LENGTH-1)] = END_SHIFT_VAL;
                }
            }

            /* save results to memory */
            {
                if (dirFlags[sregId].savePPFlag==TRUE){
                    /* Stores in DDR if it's the last iteration,
                     * or last sreg on a valid IT*/
                    if((sregId==(NSHIFTREGS-1))&&(dirPtr[sregId].itcnt < NT)){
                        // last sreg saves swapped ptrs 
                        // to memory if its IT < NT
                       p[dirPtr[sregId].oidx]  = dirValues[sregId].newppVal;
                       pp[dirPtr[sregId].oidx] = p_sreg[sregId][0];
                    }else{
                        if(dirPtr[sregId].itcnt==0){
                            dirValues[sregId].outpVal  = pp_sreg[sregId][0];
                            dirValues[sregId].outppVal = p_sreg[sregId][0];
                        }else{
                            dirValues[sregId].outpVal  = p_sreg[sregId][0];
                            dirValues[sregId].outppVal = dirValues[sregId].newppVal;
                        }
                    }
                    dirFlags[sregId].savePPFlag = FALSE;
                }

                if (revFlags[sregId].savePPFlag==TRUE){
                    /* Stores in DDR if it's the last iteration,
                     * or last sreg on a valid IT*/
                    if((sregId==(NSHIFTREGS-1))&&(revPtr[sregId].itcnt < NT)){
                        // last sreg saves swapped ptrs 
                        // to memory if its IT < NT
                       pr[revPtr[sregId].oidx]  = revValues[sregId].newppVal;
                       ppr[revPtr[sregId].oidx] = pr_sreg[sregId][0];
                    }else{
                        revValues[sregId].outpVal  = pr_sreg[sregId][0]; // saves p for next it
                        revValues[sregId].outppVal = revValues[sregId].newppVal;
                    }
                    revFlags[sregId].savePPFlag = FALSE;
                }
            }

            /* correlation */
            {
                if(revFlags[sregId].sregStateOutput == SREGSTATE_ON){
                    if(revXZ[sregId].xout >= NXB 
                        && revXZ[sregId].xout < NXE-NXB
                        && revXZ[sregId].zout >= NZB 
                        && revXZ[sregId].zout < NZE-NZB
                        && revPtr[sregId].itcnt < NT){
                        volatile float imlocVal = 
                                revValues[sregId].newppVal * 
                                dirValues[sregId].newppVal;

                        /**
                        if(revPtr[sregId].itcnt==2){
                        printf("it=%d [%d,%d](%d) p=%.20f pr=%.20f \n",
                            revPtr[sregId].itcnt, 
                            revXZ[sregId].xout, 
                            revXZ[sregId].zout,
                            revPtr[sregId].imcnt,
                            dirValues[sregId].newppVal,
                            revValues[sregId].newppVal
                            );
                        }
                        /**/
                        imloc[revPtr[sregId].imcnt] +=  imlocVal;
                    }
                }
            }

            /*update xz coordinates*/
            {
                if(sregId==0){
                    inc_xz(&dirXZ[sregId].xin, &dirXZ[sregId].zin);
                    inc_xz(&revXZ[sregId].xin, &revXZ[sregId].zin);
                }else{
                    dirFlags[sregId].sregStateInput = 
                        dirFlags[sregId-1].sregStateOutput;
                    dirXZ[sregId].xin = dirXZ[sregId-1].xout;
                    dirXZ[sregId].zin = dirXZ[sregId-1].zout;

                    revFlags[sregId].sregStateInput = 
                        revFlags[sregId-1].sregStateOutput;
                    revXZ[sregId].xin = revXZ[sregId-1].xout;
                    revXZ[sregId].zin = revXZ[sregId-1].zout;
                }
                
                if(dirFlags[sregId].sregStateOutput==SREGSTATE_OFF){
                    dirFlags[sregId].sregStateOutput = 
                                dirXZ[sregId].xin==ORDER 
                                && dirXZ[sregId].zin==(NZE-1)? 
                                SREGSTATE_ON:SREGSTATE_OFF;
                }else{
                    inc_xz(&dirXZ[sregId].xlap, &dirXZ[sregId].zlap);
                    inc_xz(&dirXZ[sregId].xout, &dirXZ[sregId].zout);
                    if(dirXZ[sregId].xout==0 && dirXZ[sregId].zout==0){
                        // end of production
                        // printf("> sreg=%d it=%d/%d\n",
                        //     sregId, dirPtr[sregId].itcnt, NT);
                        if(dirPtr[sregId].itcnt!=(NT-1)){
                            dirFlags[sregId].sregStateOutput = 
                                dirXZ[sregId].xin==ORDER 
                                && dirXZ[sregId].zin==(NZE-1)? 
                                SREGSTATE_ON:SREGSTATE_OFF;
                            dirPtr[sregId].itcnt += NSHIFTREGS;
                        }
                    }
                }
                if(revFlags[sregId].sregStateOutput==SREGSTATE_OFF){
                    revFlags[sregId].sregStateOutput = 
                                revXZ[sregId].xin==ORDER 
                                && revXZ[sregId].zin==(NZE-1)? 
                                SREGSTATE_ON:SREGSTATE_OFF;
                }else{
                    inc_xz(&revXZ[sregId].xlap, &revXZ[sregId].zlap);
                    inc_xz(&revXZ[sregId].xout, &revXZ[sregId].zout);
                    if(revXZ[sregId].xout==0 && revXZ[sregId].zout==0){
                        // end of production
                        printf("> sreg=%d it=%d/%d\n",
                            sregId, dirPtr[sregId].itcnt, NT);
                        if(revPtr[sregId].itcnt!=(NT-1)){
                            revFlags[sregId].sregStateOutput = 
                                revXZ[sregId].xin==ORDER 
                                && revXZ[sregId].zin==(NZE-1)? 
                                SREGSTATE_ON:SREGSTATE_OFF;
                            revPtr[sregId].itcnt += NSHIFTREGS;
                        }else{
                            return;
                        }
                    }
                }

                dirValues[sregId].accx = 0.f;
                dirValues[sregId].accz = 0.f;
                revValues[sregId].accx = 0.f;
                revValues[sregId].accz = 0.f;
            }
        }
    }
}
