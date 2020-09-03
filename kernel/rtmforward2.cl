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
    volatile unsigned int ucnt_base[NSHIFTREGS], pcnt[NSHIFTREGS]; 
    volatile unsigned int ucnt[NSHIFTREGS], oidx[NSHIFTREGS];
    volatile unsigned int gcnt;
    // circular shift regs structure
    float pin_sreg      [NSHIFTREGS][PREFETCH_LENGTH];
    float pp_sreg       [NSHIFTREGS][PREFETCH_LENGTH];
    float v2dt2_sreg    [NSHIFTREGS][PREFETCH_LENGTH];
    float laplace_sreg  [NSHIFTREGS][LPBUFFER_LENGTH];

    float ch_coefs_x[ORDER_LENGTH], ch_coefs_z[ORDER_LENGTH];
    float ch_taperx[NXB], ch_taperz[NXB];

    volatile float accx[NSHIFTREGS], accz[NSHIFTREGS];
    
    volatile float pVal     [NSHIFTREGS];
    volatile float v2dt2Val [NSHIFTREGS];
    volatile float ppVal    [NSHIFTREGS];
    volatile float srcVal   [NSHIFTREGS];
    volatile float outpVal  [NSHIFTREGS];
    volatile float outppVal [NSHIFTREGS];
    volatile int  itcnt[NSHIFTREGS];
    volatile int  xin[NSHIFTREGS], zin[NSHIFTREGS];
    volatile int  xlap[NSHIFTREGS], zlap[NSHIFTREGS];
    volatile int  xout[NSHIFTREGS], zout[NSHIFTREGS];
    volatile int  savePPFlag[NSHIFTREGS];
    volatile int  saveUPBFlag[NSHIFTREGS];
    volatile int  sregStateInput[NSHIFTREGS];
    volatile int  sregStateOutput[NSHIFTREGS];
    volatile float lapVal[NSHIFTREGS];
    volatile float tapered_pp[NSHIFTREGS];
    volatile float new_pp[NSHIFTREGS];
    volatile float ptpxr[NSHIFTREGS], ptpxl[NSHIFTREGS], ptpz[NSHIFTREGS];
    volatile float pptpxr[NSHIFTREGS], pptpxl[NSHIFTREGS], pptpz[NSHIFTREGS];

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
        itcnt [i2] = i2; // each sreg starts on a different IT
        pcnt  [i2] = 0; 
        xin   [i2] = 0;
        xlap  [i2] = XLAP_START;
        xout  [i2] = 0;
        zin   [i2] = 0;
        zlap  [i2] = ZLAP_START;
        zout  [i2] = 0;
        oidx  [i2] = 0;
        outppVal[i2] = 0.f;
        outpVal [i2] = 0.f;
        accx[i2] = 0.f;
        accz[i2] = 0.f;
        ucnt[i2]  = 0;
        savePPFlag[i2] = 0;
        saveUPBFlag[i2] = 0;
        if(i2==0){
            sregStateInput[i2]  = SREGSTATE_ON;
        }else{
            sregStateInput[i2]  = SREGSTATE_OFF;
        }
        sregStateOutput[i2] = SREGSTATE_OFF;

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
            pcnt[sregId] 	  = xin[sregId]*NZE + zin[sregId];
    		ucnt_base[sregId] = itcnt[sregId]*(NXE*HALF_ORDER) + xout[sregId]*HALF_ORDER; 
    		ucnt[sregId]      = ucnt_base[sregId] + (zout[sregId] - (NZB-HALF_ORDER));
            oidx[sregId]      = xout[sregId]*NZE + zout[sregId];
            /*****************************************************/
            /* sreg inputs */
		    if(sregStateInput[sregId]==SREGSTATE_ON){
		        if(sregId==0){ 
		            /*first sreg always gets from memory*/
		            pVal  [sregId] = p [pcnt[sregId]];
		            ppVal [sregId] = pp[pcnt[sregId]];
		        }else{
		            pVal  [sregId] = outppVal[sregId-1];
		            ppVal [sregId] = outpVal[sregId-1];
		        }
		        v2dt2Val[sregId]  = v2dt2[pcnt[sregId]];
		    }else{
		        pVal  [sregId]   = END_SHIFT_VAL;
		        ppVal [sregId]   = END_SHIFT_VAL;
		        v2dt2Val[sregId] = END_SHIFT_VAL;
		    }
            /*****************************************************/
            // taper apply
            if(sregStateInput[sregId]==SREGSTATE_ON){
                ptpxl [sregId] = getTaperxl(xin[sregId], zin[sregId], ch_taperx);
                ptpxr [sregId] = getTaperxr(xin[sregId], zin[sregId], ch_taperx);
                ptpz  [sregId] = getTaperz(xin[sregId], zin[sregId], ch_taperz);
            }else{
                ptpxl [sregId] = 1.0;
                ptpxr [sregId] = 1.0;
                ptpz  [sregId] = 1.0;
            }

            pin_sreg[sregId][PREFETCH_LENGTH-1] = taper_apply(pVal[sregId], 
                                                       ptpxl[sregId], 
                                                       ptpxr[sregId], 
                                                       ptpz[sregId]);
            v2dt2_sreg[sregId][PREFETCH_LENGTH-1] = v2dt2Val[sregId];
            pp_sreg   [sregId][PREFETCH_LENGTH-1] = ppVal[sregId];

            /*****************************************************/
            #pragma unroll ORDER_LENGTH
            for (io=0; io<ORDER_LENGTH;io++){
                volatile float tmp_az = (ch_coefs_z[io]*pin_sreg[sregId][XLAP_START*NZE + io]);
                volatile float tmp_ax = (ch_coefs_x[io]*pin_sreg[sregId][io*NZE + ZLAP_START]);
                accz[sregId] = accz[sregId] + tmp_az;
                accx[sregId] = accx[sregId] + tmp_ax;
            }
            lapVal[sregId] = accz[sregId] +  accx[sregId];
            /*****************************************************/
            /* calc laplace_sreg    */
            if (sregStateOutput[sregId] == SREGSTATE_ON){
                if(itcnt[sregId]<NT){
                    srcVal [sregId]  = srcwavelet[itcnt[sregId]];
                }else{
                    srcVal [sregId]  = 0.;
                }
                if (xlap[sregId] >= XLAP_START 
                    && xlap[sregId] <  (XLAP_END) 
                    && zlap[sregId] >= ZLAP_START 
                    && zlap[sregId] <  ZLAP_END)
                {
                    laplace_sreg [sregId][(LPBUFFER_LENGTH-1)] = lapVal[sregId];
                }else{
                    laplace_sreg [sregId][(LPBUFFER_LENGTH-1)] = END_SHIFT_VAL;
                }
                pptpxl[sregId] = getTaperxl(xout[sregId], zout[sregId], ch_taperx);
                pptpxr[sregId] = getTaperxr(xout[sregId], zout[sregId], ch_taperx);
                pptpz [sregId] = getTaperz(xout[sregId], zout[sregId], ch_taperz);
                /*****************************************************/
                //tapper Z
                tapered_pp[sregId]  = taper_apply(pp_sreg[sregId][0], 
                                   pptpxl[sregId], pptpxr[sregId], 
                                   pptpz[sregId]);

                /*****************************************************/
                // update pp
                new_pp[sregId]     = update_pp(tapered_pp[sregId], 
                                        pin_sreg[sregId][0], 
                                        v2dt2_sreg[sregId][0], 
                                        laplace_sreg[sregId][0]);
                /*****************************************************/
                // apply source
                if (xout[sregId]==srcx && zout[sregId]==srcz){
                    new_pp[sregId] += srcVal[sregId];
                }
                // save up board
                if (zout[sregId] >= NZB-HALF_ORDER && zout[sregId] < NZB){
                    saveUPBFlag[sregId] = TRUE;
                }
                savePPFlag[sregId] = TRUE;
            }else{
                savePPFlag[sregId] = 0;
                saveUPBFlag[sregId] = 0;
                pptpxl[sregId] = 1.0;
                pptpxr[sregId] = 1.0;
                pptpz [sregId] = 1.0;
                new_pp[sregId] = END_SHIFT_VAL;
                laplace_sreg[sregId][(LPBUFFER_LENGTH-1)] = END_SHIFT_VAL;
            }
            /*****************************************************/

            /*****************************************************/
            if (saveUPBFlag[sregId] && itcnt[sregId] < NT &&  ucnt[sregId] < UPBBUFFER_LENGTH){
                upb[ucnt[sregId]] = new_pp[sregId];
                saveUPBFlag[sregId] = FALSE;
            }
            if (savePPFlag[sregId]){
                /* Stores in DDR if it's the last iteration,
                 * or last sreg on a valid IT*/
                if(itcnt[sregId] == (NT-1)){ 
                    // last IT saves to memory
                    pp[oidx[sregId]]  = new_pp[sregId];
                    p[oidx[sregId]]   = pin_sreg[sregId][0];
                }else if((sregId==(NSHIFTREGS-1))&&(itcnt[sregId] < NT)){
                    // last sreg saves swapped ptrs 
                    // to memory if its IT < NT
                   p[oidx[sregId]]  = new_pp[sregId];
                   pp[oidx[sregId]] = pin_sreg[sregId][0];
                }else{
                    outpVal  [sregId] = pin_sreg[sregId][0]; // saves p for next it
                    outppVal [sregId] = new_pp[sregId];
                }
                savePPFlag[sregId] = FALSE;
            }
            /*****************************************************/
            /* update img counters */
            /* update img counters */
            if(sregId==0){
                inc_xz(&xin[sregId], &zin[sregId]);
            }else{
                sregStateInput[sregId] = sregStateOutput[sregId-1];
                xin[sregId] = xout[sregId-1];
                zin[sregId] = zout[sregId-1];
            }
            if(sregStateOutput[sregId]==SREGSTATE_OFF){
                sregStateOutput[sregId] = 
                            xin[sregId]==ORDER && zin[sregId]==(NZE-1)? 
                            SREGSTATE_ON:SREGSTATE_OFF;
            }else{
                inc_xz(&xlap[sregId], &zlap[sregId]);
                inc_xz(&xout[sregId], &zout[sregId]);
                if(xout[sregId]==0 && zout[sregId]==0){
                    // end of production
                    //printf("> sreg=%d it=%d/%d\n",sregId, itcnt[sregId], NT);
                    if(itcnt[sregId]!=(NT-1)){
                        sregStateOutput[sregId] = 
                            xin[sregId]==ORDER && zin[sregId]==(NZE-1)? 
                            SREGSTATE_ON:SREGSTATE_OFF;
                        itcnt[sregId] += NSHIFTREGS;
                    }else{
                        return;
                    }
                }
            }
            accx[sregId] = 0.f;
            accz[sregId] = 0.f;
        }
    }
}