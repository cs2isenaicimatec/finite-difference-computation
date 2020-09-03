#define ORDER           8
#define HALF_ORDER      (ORDER/2)
#define ORDER_LENGTH    (ORDER+1)
#define NX_START        (HALF_ORDER)
#define NZ_START        (HALF_ORDER)
#define NX_END          (NX-HALF_ORDER)
#define NZ_END          (NZ-HALF_ORDER)
#define NXNZ            (NX*NZ) // total points on the img
#define NXNZ_HALF       (NX_HALF*NZ)
#define NSHIFTREGS      11
#define PREFETCH_LENGTH (ORDER_LENGTH*NZ)
#define POSFETCH_LENGTH NSHIFTREGS-1

#define SLICE_LENGTH    (((NX-ORDER)/NSHIFTREGS))
#define SHIFT_START     0
#define SHIFT_END       ((SLICE_LENGTH+ORDER_LENGTH)*NZ)

#define X_START         (HALF_ORDER)
#define X_END           (SLICE_LENGTH)

__kernel void 
__attribute__((task))
 fd_step(const int order, __global const float *restrict p, const int nz, const int nx,
    __global const float *restrict coefs_x, __global const float *restrict coefs_z,
        __global float *restrict laplace)
{
    //float shiftreg[8*295+10];
    float shiftreg[2370];
    //int div = order/2;
    int div = 4;
    //const int loop_iteration = nz*nx-div-1;
    const int loop_iteration = 122420;
    //const int gap = div*(nz+1);
    const int gap = 1184;
    int k, k0, ind, count = 0, idx = 0;
    float resultx, resultz;
    float ce, le0, ri0, to0, bo0, le1, ri1, to1, bo1, le2, ri2, to2, bo2, le3, ri3, to3, bo3;
	int i, j;
    float ch_coefs_z[ORDER_LENGTH],ch_coefs_x[ORDER_LENGTH]; 

    for(k0=0;k0<=ORDER;k0++){
    	ch_coefs_z[k0] = coefs_z[k0];
        ch_coefs_x[k0] = coefs_x[k0];
    }
    #pragma unroll 4
	for(count = 0; count < loop_iteration; count ++){
        #pragma unroll
        //for (k=(8*295+9); k>0; --k) {
        for (k=(2369); k>0; --k) {
            shiftreg[k] = shiftreg[k-1];
        }
		if (count==idx*(295)+4)
		{
			idx++;
		}

        shiftreg[0] = (count < loop_iteration) ? p[count] : 0.0f; 
        //if(count >= 8*295+9){
        if(count >= 2369){
            to3 = shiftreg[0*295+4];//top()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
            to2 = shiftreg[1*295+4];
            to1 = shiftreg[2*295+4];
            to0 = shiftreg[3*295+4];
            le3 = shiftreg[4*295+0];//left
            le2 = shiftreg[4*295+1];
            le1 = shiftreg[4*295+2];                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
            le0 = shiftreg[4*295+3];
            ce = shiftreg[4*295+4];//center
            ri0 = shiftreg[4*295+5];//right
            ri1 = shiftreg[4*295+6];
            ri2 = shiftreg[4*295+7];
            ri3 = shiftreg[4*295+8];
            bo0 = shiftreg[5*295+4];//bottom
            bo1 = shiftreg[6*295+4];
            bo2 = shiftreg[7*295+4];
            bo3 = shiftreg[8*295+4];


            if ((count>=idx*295-1)&&(count<=(idx*295+3)))
            {
                resultz = 0.0f;
                resultx = 0.0f;

            } else {
                    resultx = coefs_x[3]*to0 + coefs_x[2]*to1 + coefs_x[1]*to2 + coefs_x[0]*to3 + coefs_x[4]*ce 
                    + coefs_x[5]*bo0 + coefs_x[6]*bo1 + coefs_x[7]*bo2 + coefs_x[8]*bo3;
                    
                    resultz = coefs_z[3]*le0 + coefs_z[2]*le1 + coefs_z[1]*le2 + coefs_z[0]*le3+ coefs_z[4]*ce 
                    + coefs_z[5]*ri0 + coefs_z[6]*ri1 + coefs_z[7]*ri2 + coefs_z[8]*ri3;
            }
            ind=count-gap;
            laplace[ind] = resultx + resultz;
        }
    }
}

