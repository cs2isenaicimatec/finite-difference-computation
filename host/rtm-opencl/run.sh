#!/bin/bash

for n in `seq $1 $2`;
do
	echo "Running test number $n"
	#touch perf.data
	perf record -e "cpu-clock, instructions, cache-references, cache-misses, L1-dcache-loads, L1-dcache-load-misses, LLC-loads, LLC-load-misses" ./rtm_opencl par=../models/new_mod/input.dat
	chmod 777 perf.data
	
	mv 'perf.data' "s$n-perf.data"
	rm "perf.data.old"
	ximage n1=195 < output/dir.image perc=98
done

