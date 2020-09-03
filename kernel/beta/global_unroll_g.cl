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

__kernel void fd_step(  int order, 
                        __global float *restrict p, 
                        int nz, 
                        int nx,
                        __global float *restrict coefs_x, 
                        __global float *restrict coefs_z,  
                        __global float *restrict laplace){

    float ch_p[NXNZ], ch_coefs_z[ORDER_LENGTH],ch_coefs_x[ORDER_LENGTH]; 
    int i0, j0, j1, k0, k1;
    int x1, z1, offset_x1;
    int x2, z2, offset_x2;
    float l1_acmz[ORDER_LENGTH],l1_acmx[ORDER_LENGTH];
    float l1_accx, l1_accz, l1_acc_out;
    float l2_acmz[ORDER_LENGTH],l2_acmx[ORDER_LENGTH];
    float l2_accx, l2_accz, l2_acc_out;
    z1 = NZ_START;      x1 = NX_START;
    z2 = NZ_START+1;    x2 = NX_START+1;
    offset_x1 = 0;
    offset_x2 = 0;

    #pragma unroll 9
    for(k0=0;k0<=ORDER;k0++){
        ch_coefs_z[k0] = coefs_z[k0];
        ch_coefs_x[k0] = coefs_x[k0];
    }
    for (k1=0; k1<NXNZ; k1++){
        ch_p[k1] = p[k1];
    }
    // #pragma unroll 1
    #pragma ivdep
    for (i0=0; i0<(NX-ORDER)*(NZ-ORDER); i0+=2){
        offset_x1 = x1*NZ;
        offset_x2 = x2*NZ;
        //////////////////////////////////////////////////////////////////////////////
        l1_accx=0.0; l1_accz=0.0;
        #pragma unroll 9
        for (j0=0; j0<=ORDER; j0++){
            l1_acmz[j0] = ch_p[offset_x1 + (z1+j0-DIV)] *ch_coefs_z[j0];
            l1_acmx[j0] = ch_p[((x1+j0-DIV)*nz) + z1]   *ch_coefs_x[j0];
            // l1_accz += ch_p[offset_x + (z+j0-DIV)] *ch_coefs_z[j0];
            // l1_accx += ch_p[((x+j0-DIV)*nz) + z]   *ch_coefs_x[j0];
        }
        l1_accz = (l1_acmz[0]+l1_acmz[1]+l1_acmz[2]+l1_acmz[3]+l1_acmz[4]+l1_acmz[5]+l1_acmz[6]+l1_acmz[7]+l1_acmz[8]);
        l1_accx = (l1_acmx[0]+l1_acmx[1]+l1_acmx[2]+l1_acmx[3]+l1_acmx[4]+l1_acmx[5]+l1_acmx[6]+l1_acmx[7]+l1_acmx[8]);
        l1_acc_out = l1_accx+l1_accz;
        laplace[offset_x1+z1] = l1_acc_out;
        //////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////
        l2_accx=0.0; l2_accz=0.0;
        #pragma unroll 9
        for (j1=0; j1<=ORDER; j1++){
            l2_acmz[j1] = ch_p[offset_x2 + (z2+j1-DIV)] *ch_coefs_z[j1];
            l2_acmx[j1] = ch_p[((x2+j1-DIV)*nz) + z2]   *ch_coefs_x[j1];
            // l2_accz += ch_p[offset_x + (z+j0-DIV)] *ch_coefs_z[j0];
            // l2_accx += ch_p[((x+j0-DIV)*nz) + z]   *ch_coefs_x[j0];
        }
        l2_accz = (l2_acmz[0]+l2_acmz[1]+l2_acmz[2]+l2_acmz[3]+l2_acmz[4]+l2_acmz[5]+l2_acmz[6]+l2_acmz[7]+l2_acmz[8]);
        l2_accx = (l2_acmx[0]+l2_acmx[1]+l2_acmx[2]+l2_acmx[3]+l2_acmx[4]+l2_acmx[5]+l2_acmx[6]+l2_acmx[7]+l2_acmx[8]);
        l2_acc_out = l2_accx+l2_accz;
        laplace[offset_x2+z2] = l2_acc_out;
        //////////////////////////////////////////////////////////////////////////////
        
        if (z1==NZ-1){
            z1 = NZ_START;
            z2 = NZ_START;
            if (x1==NX-2){
                x1= NX_START;
                x2= NX_START+1;
            }else{
                x1+=2;
                x2+=2;
            }
        }else{
            z1++;
            z2++;
        }
        // printf("x1=%d z1=%d offset_x1=%d \n", x1, z1, offset_x1);
        // printf("x2=%d z2=%d offset_x2=%d \n", x2, z2, offset_x2);
    }

}

