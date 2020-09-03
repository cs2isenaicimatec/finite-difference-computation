#ifndef PTSRC_H
#define PTSRC_H

void ricker_wavelet(int ns, float dt, float peak, float *s);
float ricker (float t, float fpeak);
void ptsrc (int xs, int zs, float ts, float **s);

#endif
