__kernel void fd_step(int order, int nx, int nz,
    __global float *restrict p, __global float *restrict laplace, 
    __global float *restrict coefs_x, __global float *restrict coefs_z){
  
  int order_length = order+1;
  int half_order = order/2;   
  int i = half_order + get_global_id(0);
  int j = half_order + get_global_id(1);;
  int mult= i*nz;
  int aux, io;  
  float acmx = 0.f, acmz = 0.f; 
  if (i < nx - half_order){
    if(j < nz - half_order){
      for(io=0;io<=order;io++){
        aux = io - half_order;  
        acmz += p[(mult) + (j+aux)]*coefs_z[io];
        acmx += p[((i+aux)*nz) + j]*coefs_x[io];
      }
      laplace[mult+j] = acmz + acmx;
      acmx = acmz = 0.f;
    }
  }
}

__kernel void fd_time(int nx, int nz, 
                      __global float *restrict p, __global float *restrict pp,
                      __global float *restrict v2, __global float *restrict laplace, float dt2){
  int ix = get_global_id(0);
  int iz = get_global_id(1);
  int mult = ix * nz;
  float acm;    
  
  if(ix<nx){
    if(iz<nz){
        acm = 2.*p[mult+iz] - pp[mult+iz] + v2[mult+iz]*dt2*laplace[mult+iz];
        pp[mult+iz] = acm; 
    }
  }
}

__kernel void taper_apply(int nx, int nz, int nxb, int nzb, 
                          __global float *restrict p, __global float *restrict pp,
                          __global float *restrict taperx, __global float *restrict taperz){
  
  int itx = get_global_id(0); // Size NX 
  int itz = get_global_id(1); // Size NZ_border
  int itxr = nx - 1, mult = itx*nz; 

  if(itx<nx){
      if(itz<nzb){
      p[mult+itz] *= taperz[itz];
      pp[mult+itz] *= taperz[itz];
    }
  }

  if(itx<nxb){
    if(itz<nzb){
      p[mult+itz] *= taperx[itx];
      pp[mult+itz] *= taperx[itx];
          
      p[(itxr-itx)*nz+itz] *= taperx[itx];
      pp[(itxr-itx)*nz+itz] *= taperx[itx];
    }
  }
}

__kernel void ptsrc(int nz, __global float *restrict pp, int sx_is, int sz, float srce_it){
    pp[sx_is*nz+sz] += srce_it;
}

__kernel void upb(int order, int nx, int nz, int nzb, 
    __global float *pp, __global float *upb, int it){
  int half_order = order/2; 
  int ix = get_global_id(0); 
  int iz = (nzb-half_order) + get_global_id(1); 
  if(ix<nx){
      if(iz<nzb){
        upb[(it*nx*half_order)+(ix*half_order)+(iz-(nzb-half_order))] = pp[ix*nz+iz];
    }
  }
}

__kernel void upb_reverse(int order, int nx, int nz, int nzb, int nt,  
    __global float *restrict pp, __global float *restrict upb, int it){
  int half_order = order/2; 
  int ix = get_global_id(0); 
  int iz = (nzb-half_order) + get_global_id(1); 
  
  if(ix<nx){
    if(iz<nzb)
      pp[ix*nz+iz] = upb[((nt-1-it)*nx*half_order)+(ix*half_order)+(iz-(nzb-half_order))];
  }
}

__kernel void add_sism(int nx, int nz, int nxb, int nt, int is, int it, 
        int gz, __global float *restrict d_obs, __global float *restrict ppr){
  int size = nx-(2*nxb); 
  int ix = get_global_id(0); 
  if(ix<size){
    ppr[((ix+nxb)*nz) + gz] += d_obs[ix*nt + (nt-1-it)]; 
  }
}

__kernel void img_cond(int nx, int nz, int nxb, int nzb, 
      __global float *restrict imloc, __global float *restrict p, __global float *restrict ppr){
    int size_x = nx-(2*nxb); 
    int size_z = nz-(2*nzb); 
    int ix = get_global_id(0);
    int iz = get_global_id(1);
    if(iz<size_z){
      if(ix<size_x){
        imloc[ix*size_z+iz] += p[(ix+nxb)*nz+(iz+nzb)] * ppr[(ix+nxb)*nz+(iz+nzb)];          
      }
    }
}