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

    float ch_p[NXNZ], ch_coefs_z[ORDER_LENGTH],ch_coefs_x[ORDER_LENGTH]; 
    int i0, j1, j2, j3, j4, j5, j6, j7, j8; 
    int k0, k1;
    int x1, z1, offset_x1;
    int x2, z2, offset_x2;
    int x3, z3, offset_x3;
    int x4, z4, offset_x4;
    int x5, z5, offset_x5;
    int x6, z6, offset_x6;
    int x7, z7, offset_x7;
    int x8, z8, offset_x8;

    float l1_acmz[ORDER_LENGTH],l1_acmx[ORDER_LENGTH];
    float l2_acmz[ORDER_LENGTH],l2_acmx[ORDER_LENGTH];
    float l3_acmz[ORDER_LENGTH],l3_acmx[ORDER_LENGTH];
    float l4_acmz[ORDER_LENGTH],l4_acmx[ORDER_LENGTH];
    float l5_acmz[ORDER_LENGTH],l5_acmx[ORDER_LENGTH];
    float l6_acmz[ORDER_LENGTH],l6_acmx[ORDER_LENGTH];
    float l7_acmz[ORDER_LENGTH],l7_acmx[ORDER_LENGTH];
    float l8_acmz[ORDER_LENGTH],l8_acmx[ORDER_LENGTH];

    float l1_accx, l1_accz, l1_acc_out;
    float l2_accx, l2_accz, l2_acc_out;
    float l3_accx, l3_accz, l3_acc_out;
    float l4_accx, l4_accz, l4_acc_out;
    float l5_accx, l5_accz, l5_acc_out;
    float l6_accx, l6_accz, l6_acc_out;
    float l7_accx, l7_accz, l7_acc_out;
    float l8_accx, l8_accz, l8_acc_out;

    z1 = NZ_START;      x1 = NX_START;
    z2 = NZ_START;      x2 = NX_START+1;
    z3 = NZ_START;      x3 = NX_START+2;
    z4 = NZ_START;      x4 = NX_START+3;
    z5 = NZ_START;      x5 = NX_START+4;
    z6 = NZ_START;      x6 = NX_START+5;
    z7 = NZ_START;      x7 = NX_START+6;
    z8 = NZ_START;      x8 = NX_START+7;

    offset_x1 = 0;
    offset_x2 = 0;
    offset_x3 = 0;
    offset_x4 = 0;
    offset_x5 = 0;
    offset_x6 = 0;
    offset_x7 = 0;
    offset_x8 = 0;

    #pragma unroll 9
    for(k0=0;k0<=ORDER;k0++){
        ch_coefs_z[k0] = coefs_z[k0];
        ch_coefs_x[k0] = coefs_x[k0];
    }
    #pragma ivdep
    for (k1=0; k1<NXNZ; k1++){
        ch_p[k1] = p[k1];
    }
    // #pragma unroll 1
    #pragma ivdep
    for (i0=0; i0<(NX-ORDER)*(NZ-ORDER); i0+=NLAPLACE){
        //////////////////////////////////////////////////////////////////////////////
        offset_x1 = x1*NZ;
        offset_x2 = x2*NZ;
        offset_x3 = x3*NZ;
        offset_x4 = x4*NZ;
        offset_x5 = x5*NZ;
        offset_x6 = x6*NZ;
        offset_x7 = x7*NZ;
        offset_x8 = x8*NZ;

        //////////////////////////////////////////////////////////////////////////////
        l1_accx=0.0; l1_accz=0.0;
        #pragma unroll 9
        for (j1=0; j1< ORDER_LENGTH; j1++){
            l1_acmz[j1] = ch_p[offset_x1 + (z1+j1-DIV)] *ch_coefs_z[j1];
            l1_acmx[j1] = ch_p[((x1+j1-DIV)*nz) + z1]   *ch_coefs_x[j1];
        }
        l1_accz = (l1_acmz[0]+l1_acmz[1]+l1_acmz[2]+l1_acmz[3]+l1_acmz[4]+l1_acmz[5]+l1_acmz[6]+l1_acmz[7]+l1_acmz[8]);
        l1_accx = (l1_acmx[0]+l1_acmx[1]+l1_acmx[2]+l1_acmx[3]+l1_acmx[4]+l1_acmx[5]+l1_acmx[6]+l1_acmx[7]+l1_acmx[8]);
        l1_acc_out = l1_accx+l1_accz;
        //////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////
        l2_accx=0.0; l2_accz=0.0;
        #pragma unroll 9
        for (j2=0; j2< ORDER_LENGTH; j2++){
            l2_acmz[j2] = ch_p[offset_x2 + (z2+j2-DIV)] *ch_coefs_z[j2];
            l2_acmx[j2] = ch_p[((x2+j2-DIV)*nz) + z2]   *ch_coefs_x[j2];
        }
        l2_accz = (l2_acmz[0]+l2_acmz[1]+l2_acmz[2]+l2_acmz[3]+l2_acmz[4]+l2_acmz[5]+l2_acmz[6]+l2_acmz[7]+l2_acmz[8]);
        l2_accx = (l2_acmx[0]+l2_acmx[1]+l2_acmx[2]+l2_acmx[3]+l2_acmx[4]+l2_acmx[5]+l2_acmx[6]+l2_acmx[7]+l2_acmx[8]);
        l2_acc_out = l2_accx+l2_accz;
        //////////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////////
        l3_accx=0.0; l3_accz=0.0;
        #pragma unroll 9
        for (j3=0; j3< ORDER_LENGTH; j3++){
            l3_acmz[j3] = ch_p[offset_x3 + (z3+j3-DIV)] *ch_coefs_z[j3];
            l3_acmx[j3] = ch_p[((x3+j3-DIV)*nz) + z3]   *ch_coefs_x[j3];
        }
        l3_accz = (l3_acmz[0]+l3_acmz[1]+l3_acmz[2]+l3_acmz[3]+l3_acmz[4]+l3_acmz[5]+l3_acmz[6]+l3_acmz[7]+l3_acmz[8]);
        l3_accx = (l3_acmx[0]+l3_acmx[1]+l3_acmx[2]+l3_acmx[3]+l3_acmx[4]+l3_acmx[5]+l3_acmx[6]+l3_acmx[7]+l3_acmx[8]);
        l3_acc_out = l3_accx+l3_accz;
        //////////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////////
        l4_accx=0.0; l4_accz=0.0;
        #pragma unroll 9
        for (j4=0; j4< ORDER_LENGTH; j4++){
            l4_acmz[j4] = ch_p[offset_x4 + (z4+j4-DIV)] *ch_coefs_z[j4];
            l4_acmx[j4] = ch_p[((x4+j4-DIV)*nz) + z4]   *ch_coefs_x[j4];
        }
        l4_accz = (l4_acmz[0]+l4_acmz[1]+l4_acmz[2]+l4_acmz[3]+l4_acmz[4]+l4_acmz[5]+l4_acmz[6]+l4_acmz[7]+l4_acmz[8]);
        l4_accx = (l4_acmx[0]+l4_acmx[1]+l4_acmx[2]+l4_acmx[3]+l4_acmx[4]+l4_acmx[5]+l4_acmx[6]+l4_acmx[7]+l4_acmx[8]);
        l4_acc_out = l4_accx+l4_accz;
        //////////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////////
        l5_accx=0.0; l5_accz=0.0;
        #pragma unroll 9
        for (j5=0; j5< ORDER_LENGTH; j5++){
            l5_acmz[j5] = ch_p[offset_x5 + (z5+j5-DIV)] *ch_coefs_z[j5];
            l5_acmx[j5] = ch_p[((x5+j5-DIV)*nz) + z5]   *ch_coefs_x[j5];
        }
        l5_accz = (l5_acmz[0]+l5_acmz[1]+l5_acmz[2]+l5_acmz[3]+l5_acmz[4]+l5_acmz[5]+l5_acmz[6]+l5_acmz[7]+l5_acmz[8]);
        l5_accx = (l5_acmx[0]+l5_acmx[1]+l5_acmx[2]+l5_acmx[3]+l5_acmx[4]+l5_acmx[5]+l5_acmx[6]+l5_acmx[7]+l5_acmx[8]);
        l5_acc_out = l5_accx+l5_accz;
        //////////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////////
        l6_accx=0.0; l6_accz=0.0;
        #pragma unroll 9
        for (j6=0; j6< ORDER_LENGTH; j6++){
            l6_acmz[j6] = ch_p[offset_x6 + (z6+j6-DIV)] *ch_coefs_z[j6];
            l6_acmx[j6] = ch_p[((x6+j6-DIV)*nz) + z6]   *ch_coefs_x[j6];
        }
        l6_accz = (l6_acmz[0]+l6_acmz[1]+l6_acmz[2]+l6_acmz[3]+l6_acmz[4]+l6_acmz[5]+l6_acmz[6]+l6_acmz[7]+l6_acmz[8]);
        l6_accx = (l6_acmx[0]+l6_acmx[1]+l6_acmx[2]+l6_acmx[3]+l6_acmx[4]+l6_acmx[5]+l6_acmx[6]+l6_acmx[7]+l6_acmx[8]);
        l6_acc_out = l6_accx+l6_accz;
        //////////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////////
        l7_accx=0.0; l7_accz=0.0;
        #pragma unroll 9
        for (j7=0; j7< ORDER_LENGTH; j7++){
            l7_acmz[j7] = ch_p[offset_x7 + (z7+j7-DIV)] *ch_coefs_z[j7];
            l7_acmx[j7] = ch_p[((x7+j7-DIV)*nz) + z7]   *ch_coefs_x[j7];
        }
        l7_accz = (l7_acmz[0]+l7_acmz[1]+l7_acmz[2]+l7_acmz[3]+l7_acmz[4]+l7_acmz[5]+l7_acmz[6]+l7_acmz[7]+l7_acmz[8]);
        l7_accx = (l7_acmx[0]+l7_acmx[1]+l7_acmx[2]+l7_acmx[3]+l7_acmx[4]+l7_acmx[5]+l7_acmx[6]+l7_acmx[7]+l7_acmx[8]);
        l7_acc_out = l7_accx+l7_accz;
        //////////////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////////
        l8_accx=0.0; l8_accz=0.0;
        #pragma unroll 9
        for (j8=0; j8< ORDER_LENGTH; j8++){
            l8_acmz[j8] = ch_p[offset_x8 + (z8+j8-DIV)] *ch_coefs_z[j8];
            l8_acmx[j8] = ch_p[((x8+j8-DIV)*nz) + z8]   *ch_coefs_x[j8];
        }
        l8_accz = (l8_acmz[0]+l8_acmz[1]+l8_acmz[2]+l8_acmz[3]+l8_acmz[4]+l8_acmz[5]+l8_acmz[6]+l8_acmz[7]+l8_acmz[8]);
        l8_accx = (l8_acmx[0]+l8_acmx[1]+l8_acmx[2]+l8_acmx[3]+l8_acmx[4]+l8_acmx[5]+l8_acmx[6]+l8_acmx[7]+l8_acmx[8]);
        l8_acc_out = l8_accx+l8_accz;
        //////////////////////////////////////////////////////////////////////////////
        laplace[offset_x1+z1] = l1_acc_out;
        laplace[offset_x2+z2] = l2_acc_out;
        laplace[offset_x3+z3] = l3_acc_out;
        laplace[offset_x4+z4] = l4_acc_out;
        laplace[offset_x5+z5] = l5_acc_out;
        laplace[offset_x6+z6] = l6_acc_out;
        laplace[offset_x7+z7] = l7_acc_out;
        laplace[offset_x8+z8] = l8_acc_out;
        
        if (z1>=NZ-1){
            z1 = NZ_START;
            z2 = NZ_START;
            z3 = NZ_START;
            z4 = NZ_START;
            z5 = NZ_START;
            z6 = NZ_START;
            z7 = NZ_START;
            z8 = NZ_START;
            if (x1>=NX-NLAPLACE){
                x1 = NX_START;
                x2 = NX_START+1;
                x3 = NX_START+2;
                x4 = NX_START+3;
                x5 = NX_START+4;
                x6 = NX_START+5;
                x7 = NX_START+6;
                x8 = NX_START+7;
            }else{
                x1+=NLAPLACE;
                x2+=NLAPLACE;
                x3+=NLAPLACE;
                x4+=NLAPLACE;
                x5+=NLAPLACE;
                x6+=NLAPLACE;
                x7+=NLAPLACE;
                x8+=NLAPLACE;
            }
        }else{
            z1++;
            z2++;
            z3++;
            z4++;
            z5++;
            z6++;
            z7++;
            z8++;
        }
        // printf("x1=%d z1=%d offset_x1=%d \n", x1, z1, offset_x1);
        // printf("x2=%d z2=%d offset_x2=%d \n", x2, z2, offset_x2);
        // printf("x3=%d z3=%d offset_x3=%d \n", x3, z3, offset_x3);
        // printf("x4=%d z4=%d offset_x4=%d \n", x4, z4, offset_x4);
        // printf("x5=%d z5=%d offset_x5=%d \n", x5, z5, offset_x5);
        // printf("x6=%d z6=%d offset_x6=%d \n", x6, z6, offset_x6);
        // printf("x7=%d z7=%d offset_x7=%d \n", x7, z7, offset_x7);
        // printf("x8=%d z8=%d offset_x8=%d \n", x8, z8, offset_x8);
    }

}

