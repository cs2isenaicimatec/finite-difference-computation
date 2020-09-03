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

/*******************************************************
FAILED
*******************************************************/
__kernel void fd_step(  int order, 
                        __global float *restrict p, 
                        int nz, 
                        int nx,
                        __global float *restrict coefs_x, 
                        __global float *restrict coefs_z,  
                        __global float *restrict laplace){

    int i0, j0, k0, k1;
    int x, z, offset_x;
    float acmz[ORDER_LENGTH],acmx[ORDER_LENGTH];
    float accx, accz, acc_out;
    float ch_p[NXNZ], ch_coefs_z[ORDER_LENGTH],ch_coefs_x[ORDER_LENGTH]; 
    z = NZ_START; x = NX_START;
    offset_x = 0;

    #pragma unroll 9
    for(k0=0;k0<=ORDER;k0++){
        ch_coefs_z[k0] = coefs_z[k0];
        ch_coefs_x[k0] = coefs_x[k0];
    }
    for (k1=0; k1<NXNZ; k1++){
        ch_p[k1] = p[k1];
    }
    // #pragma unroll 1
    //#pragma ivdep
    for (i0=0; i0<(NX-ORDER)*(NZ-ORDER); i0++){
        accx=0.0; accz=0.0;
        #pragma unroll 9
        for (j0=0; j0<=ORDER; j0++){
            acmz[j0] = ch_p[offset_x + (z+j0-DIV)] *ch_coefs_z[j0];
            acmx[j0] = ch_p[((x+j0-DIV)*nz) + z]   *ch_coefs_x[j0];
            // accz += ch_p[offset_x + (z+j0-DIV)] *ch_coefs_z[j0];
            // accx += ch_p[((x+j0-DIV)*nz) + z]   *ch_coefs_x[j0];
        }
        accz = (acmz[0]+acmz[1]+acmz[2]+acmz[3]+acmz[4]+acmz[5]+acmz[6]+acmz[7]+acmz[8]);
        accx = (acmx[0]+acmx[1]+acmx[2]+acmx[3]+acmx[4]+acmx[5]+acmx[6]+acmx[7]+acmx[8]);
        acc_out = accx+accz;
        laplace[offset_x+z] = acc_out;
        //laplace[offset_x+z] = accx + accz;
        if (z==NZ-1){
            z = NZ_START;
            if (x==NX-1){
                x= NX_START;
            }else{
                x++;
            }
        }else{
            z++;
        }
        offset_x = x*NZ;
    }

}

