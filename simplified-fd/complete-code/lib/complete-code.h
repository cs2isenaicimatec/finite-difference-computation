#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "functions.h"

#define sizeblock 32
#define PI (3.141592653589793)

/* file names */
char *tmpdir = NULL, *vpfile = NULL, *datfile = NULL, *vel_ext_file = NULL;
/* size */
int nz, nx, nt;
float dz, dx, dt;

/* adquisition geometry */
int ns = -1, sz = -1, fsx = -1, ds = -1, gz = -1;

/* boundary */
int nxb = -1, nzb = -1, nxe, nze;
float fac = -1.0;

/* propagation */
int order = -1; 
float fpeak;

/* arrays */
int *sx;

/*aux*/
int iss = -1, rnd, vel_ext_flag=0;

void read_input(char *file)
{
        FILE *fp;
        fp = fopen(file, "r");
        char *line = NULL;
        size_t len = 0;
        if (fp == NULL)
                exit(EXIT_FAILURE);
        while (getline(&line, &len, fp) != -1) {
                if(strstr(line,"tmpdir") != NULL)
                {
                        char *tok;
                        tok = strtok(line, "=");
                        tok = strtok(NULL,"=");
                        tok[strlen(tok) - 1] = '\0';
                        tmpdir = strdup(tok);
                }
                if(strstr(line,"datfile") != NULL)
                {
                        char *tok;
                        tok = strtok(line, "=");
                        tok = strtok(NULL,"=");
                        tok[strlen(tok) - 1] = '\0';
                        datfile = strdup(tok);
                }
                if(strstr(line,"vpfile") != NULL)
                {
                        char *tok;
                        tok = strtok(line, "=");
                        tok = strtok(NULL,"=");
                        tok[strlen(tok) - 1] = '\0';
                        vpfile = strdup(tok);
                }
                if(strstr(line,"vel_ext_file") != NULL)
                {
                        char *tok;
                        tok = strtok(line, "=");
                        tok = strtok(NULL,"=");
                        tok[strlen(tok) - 1] = '\0';
                        vel_ext_file = strdup(tok);
                        vel_ext_flag = 1;
                }
                if(strstr(line,"fpeak") != NULL)
                {
                        char *fpeak_char;
                        fpeak_char = strtok(line, "=");
                        fpeak_char = strtok(NULL,"=");
                        fpeak = atof(fpeak_char);
                }
                if(strstr(line,"nt") != NULL)
                {
                        char *nt_char;
                        nt_char = strtok(line, "=");
                        nt_char = strtok(NULL,"=");
                        nt = atoi(nt_char);
                }
                if(strstr(line,"dt") != NULL)
                {
                        char *dt_char;
                        dt_char = strtok(line, "=");
                        dt_char = strtok(NULL,"=");
                        dt = atof(dt_char);
                }
                if(strstr(line,"ns") != NULL)
                {
                        char *ns_char;
                        ns_char = strtok(line, "=");
                        ns_char = strtok(NULL,"=");
                        ns = atoi(ns_char);
                }
                if(strstr(line,"iss") != NULL)
                {
                        char *iss_char;
                        iss_char = strtok(line, "=");
                        iss_char = strtok(NULL,"=");
                        iss = atoi(iss_char);
                }
                if(strstr(line,"sz") != NULL)
                {
                        char *sz_char;
                        sz_char = strtok(line, "=");
                        sz_char = strtok(NULL,"=");
                        sz = atoi(sz_char);
                }
                if(strstr(line,"fsx") != NULL)
                {
                        char *fsx_char;
                        fsx_char = strtok(line, "=");
                        fsx_char = strtok(NULL,"=");
                        fsx = atoi(fsx_char);
                }
                if(strstr(line,"ds") != NULL)
                {
                        char *ds_char;
                        ds_char = strtok(line, "=");
                        ds_char = strtok(NULL,"=");
                        ds = atoi(ds_char);
                }
                if(strstr(line,"gz") != NULL)
                {
                        char *gz_char;
                        gz_char = strtok(line, "=");
                        gz_char = strtok(NULL,"=");
                        gz = atoi(gz_char);
                }
                if(strstr(line,"nzb") != NULL)
                {
                        char *nzb_char;
                        nzb_char = strtok(line, "=");
                        nzb_char = strtok(NULL,"=");
                        nzb = atoi(nzb_char);
                }
                if(strstr(line,"nxb") != NULL)
                {
                        char *nxb_char;
                        nxb_char = strtok(line, "=");
                        nxb_char = strtok(NULL,"=");
                        nxb = atoi(nxb_char);
                }
                if(strstr(line,"rnd") != NULL)
                {
                        char *rnd_char;
                        rnd_char = strtok(line, "=");
                        rnd_char = strtok(NULL,"=");
                        rnd = atoi(rnd_char);
                }
                if(strstr(line,"nz") != NULL)
                {
                        char *nz_char;
                        nz_char = strtok(line, "=");
                        if (strlen(nz_char) <= 2)
                        {
                                nz_char = strtok(NULL,"=");
                                nz = atoi(nz_char);
                        }
                }
                if(strstr(line,"nx") != NULL)
                {
                        char *nx_char;
                        nx_char = strtok(line, "=");
                        if (strlen(nx_char) <= 2)
                        {
                                nx_char = strtok(NULL,"=");
                                nx = atoi(nx_char);
                        }
                }
                if(strstr(line,"dz") != NULL)
                {
                        char *dz_char;
                        dz_char = strtok(line, "=");
                        dz_char = strtok(NULL,"=");
                        dz = atof(dz_char);
                }
                if(strstr(line,"dx") != NULL)
                {
                        char *dx_char;
                        dx_char = strtok(line, "=");
                        dx_char = strtok(NULL,"=");
                        dx = atof(dx_char);
                }
                if(strstr(line,"fac") != NULL)
                {
                        char *fac_char;
                        fac_char = strtok(line, "=");
                        fac_char = strtok(NULL,"=");
                        fac = atof(fac_char);
                }
                if(strstr(line,"order") != NULL)
                {
                        char *order_char;
                        order_char = strtok(line, "=");
                        order_char = strtok(NULL,"=");
                        order = atoi(order_char);
                }
        }
        free(line);
	if(iss == -1 ) iss = 0;	 	// save snaps of this source
	if(ns == -1) ns = 1;	 	// number of sources
	if(sz == -1) sz = 0; 		// source depth
	if(fsx == -1) fsx = 0; 	// first source position
	if(ds == -1) ds = 1; 		// source interval
	if(gz == -1) gz = 0; 		// receivor depth
	if(order == -1) order = 8;	// FD order
	if(nzb == -1) nzb = 40;		// z border size
	if(nxb == -1) nxb = 40;		// x border size
	if(fac == -1.0) fac = 0.7;	
}

// ============================ Aux ============================
float *calc_coefs(int order)
{
        float *coef;

        coef = (float *)calloc(order+1,sizeof(float));
        switch(order)
        {
                case 2:
                        coef[0] = 1.;
                        coef[1] = -2.;
                        coef[2] = 1.;
                        break;
                case 4:
                        coef[0] = -1./12.;
                        coef[1] = 4./3.;
                        coef[2] = -5./2.;
                        coef[3] = 4./3.;
                        coef[4] = -1./12.;
                        break;
                case 6:
                        coef[0] = 1./90.;
                        coef[1] = -3./20.;
                        coef[2] = 3./2.;
                        coef[3] = -49./18.;
                        coef[4] = 3./2.;
                        coef[5] = -3./20.;
                        coef[6] = 1./90.;
                        break;
                case 8:

                        coef[0] = -1./560.;
                        coef[1] = 8./315.;
                        coef[2] = -1./5.;
                        coef[3] = 8./5.;
                        coef[4] = -205./72.;
                        coef[5] = 8./5.;
                        coef[6] = -1./5.;
                        coef[7] = 8./315.;
                        coef[8] = -1./560.;
                        break;
                default:
                        makeo2(coef,order);
        }

        return coef;
}

static void makeo2 (float *coef,int order)
{
        float h_beta, alpha1=0.0;
        float alpha2=0.0;
        float  central_term=0.0;
        float coef_filt=0;
        float arg=0.0;
        float  coef_wind=0.0;
        int msign,ix;

        float alpha = .54;
        float beta = 6.;
        h_beta = 0.5*beta;
        alpha1=2.*alpha-1.0;
        alpha2=2.*(1.0-alpha);
        central_term=0.0;

        msign=-1;

        for (ix=1; ix <= order/2; ix++){
                msign=-msign ;
                coef_filt = (2.*msign)/(ix*ix);
                arg = PI*ix/(2.*(order/2+2));
                coef_wind=pow((alpha1+alpha2*cos(arg)*cos(arg)),h_beta);
                coef[order/2+ix] = coef_filt*coef_wind;
                central_term = central_term + coef[order/2+ix];
                coef[order/2-ix] = coef[order/2+ix];
        }

        coef[order/2]  = -2.*central_term;

        return;
}

void *alloc1 (size_t n1, size_t size)
{
	void *p;

	if ((p=malloc(n1*size))==NULL)
		return NULL;
	return p;
}

void **alloc2 (size_t n1, size_t n2, size_t size)
{
	size_t i2;
	void **p;

	if ((p=(void**)malloc(n2*sizeof(void*)))==NULL) 
		return NULL;
	if ((p[0]=(void*)malloc(n2*n1*size))==NULL) {
		free(p);
		return NULL;
	}
	for (i2=0; i2<n2; i2++)
		p[i2] = (char*)p[0]+size*n1*i2;
	return p;
}

void ***alloc3 (size_t n1, size_t n2, size_t n3, size_t size)
{
	size_t i3,i2;
	void ***p;

	if ((p=(void***)malloc(n3*sizeof(void**)))==NULL)
		return NULL;
	if ((p[0]=(void**)malloc(n3*n2*sizeof(void*)))==NULL) {
		free(p);
		return NULL;
	}
	if ((p[0][0]=(void*)malloc(n3*n2*n1*size))==NULL) {
		free(p[0]);
		free(p);
		return NULL;
	}

	for (i3=0; i3<n3; i3++) {
		p[i3] = p[0]+n2*i3;
		for (i2=0; i2<n2; i2++)
			p[i3][i2] = (char*)p[0][0]+size*n1*(i2+n2*i3);
	}
	return p;
}

float *alloc1float(size_t n1)
{
	return (float*)alloc1(n1,sizeof(float));
}

float **alloc2float(size_t n1, size_t n2)
{
	return (float**)alloc2(n1,n2,sizeof(float));
}

float ***alloc3float(size_t n1, size_t n2, size_t n3)
{
	return (float***)alloc3(n1,n2,n3,sizeof(float));
}

void free1 (void *p)
{
	free(p);
}

void free2 (void **p)
{
	free(p[0]);
	free(p);
}

void free3 (void ***p)
{
	free(p[0][0]);
	free(p[0]);
	free(p);
}

void free1float(float *p)
{
	free1(p);
}

void free2float(float **p)
{
	free2((void**)p);
}

void free3float(float ***p)
{
	free3((void***)p);
}

