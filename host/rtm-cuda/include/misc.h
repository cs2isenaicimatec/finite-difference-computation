
#ifndef MISC_H
#define MISC_H

#include "time.h"
#include  <sys/time.h>

static inline int __elapsed_time(struct timeval st, struct timeval et) {
	return ((et.tv_sec - st.tv_sec) * 1000000) + (et.tv_usec - st.tv_usec);
}


#endif