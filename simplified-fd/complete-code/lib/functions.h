#include <stdio.h>

void read_input(char *file);
void fd_init(int order, int nx, int nz, int nxb, int nzb, int nt, int ns, float fac, float dx, float dz, float dt);
void fd_init_cuda(int order, int nxe, int nze, int nxb, int nzb, int nt, int ns, float fac);
float *calc_coefs(int order);
static void makeo2 (float *coef,int order);
void *alloc1 (size_t n1, size_t size);
void **alloc2 (size_t n1, size_t n2, size_t size);
float *alloc1float(size_t n1);
float **alloc2float(size_t n1, size_t n2);
void free1 (void *p);
void free2 (void **p);
void free1float(float *p);
void free2float(float **p);