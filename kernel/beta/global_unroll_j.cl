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
#define NXNZ            (NX*NZ)
#define NLAPLACE        8

__kernel void fd_step(  int order, 
                        __global float *restrict p, 
                        int nz, 
                        int nx,
                        __global float *restrict coefs_x, 
                        __global float *restrict coefs_z,  
                        __global float *restrict laplace){

    int i0, j0, k0, k1, k3, l0;
    int zidx, xidx, lidx;
    int x[NLAPLACE], z[NLAPLACE], offset_x[NLAPLACE];
    float acmz[NLAPLACE],acmx[NLAPLACE];
    float accx[NLAPLACE], accz[NLAPLACE];// acc_out[NLAPLACE];
    float ch_coefs_z[ORDER_LENGTH],ch_coefs_x[ORDER_LENGTH]; 
    float ch_p[NXNZ];


    #pragma unroll 9
    for(k0=0;k0<=ORDER;k0++){
        ch_coefs_z[k0] = coefs_z[k0];
        ch_coefs_x[k0] = coefs_x[k0];
    }
    #pragma ivdep
    for (k1=0; k1<NXNZ; k1++){
        ch_p[k1] = p[k1];
    }
    #pragma unroll NLAPLACE
    for (k3=0; k3<NLAPLACE; k3++){
        z[k3] = NZ_START; x[k3] = NX_START+k3;
    }
    //#pragma unroll 1
    #pragma ivdep
    for (i0=0; i0<(NX-ORDER)*(NZ-ORDER); i0+=NLAPLACE){
        
        #pragma unroll NLAPLACE
        for (l0=0; l0<NLAPLACE; l0++)
        {
        ///////////////////////////////////////////////////////////////////////////
            offset_x[l0] = x[l0]*NZ;
            lidx = offset_x[l0]+z[l0];
            acmz[l0] = 0.f;
            acmx[l0] = 0.f;
            //accx[l0]=0.0; accz[l0]=0.0;
            #pragma unroll 9
            for (j0=0; j0<=ORDER; j0++){
                zidx = offset_x[l0] + z[l0]+j0-DIV;
                xidx = ((x[l0]+j0-DIV)*nz) + z[l0];
                acmz[l0]+= ch_p[zidx] *ch_coefs_z[j0];
                acmx[l0]+= ch_p[xidx] *ch_coefs_x[j0];;
            }
            laplace[lidx] = acmz[l0]+acmx[l0];
            //laplace[offset_x+z] = accx + accz;
            if (z[l0]>=NZ-1){
                z[l0] = NZ_START;
                x[l0] = x[l0]>=NX-NLAPLACE? NX_START+l0:x[l0]+NLAPLACE;
            }else{
                z[l0]+=1;
            }
        ///////////////////////////////////////////////////////////////////////////
        }
        
    }

}

