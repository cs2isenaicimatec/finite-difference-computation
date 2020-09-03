#!/bin/bash

for n in `seq $1 $2`;
do
	echo "Running test number $n"
	#touch perf.data
	#perf record --freq=14000 -e "cpu-clock, instructions, cache-references, cache-misses, L1-dcache-loads, L1-dcache-load-misses, LLC-loads, LLC-load-misses" ./rtm_serial par=../models/new_mod/input.dat
	#perf record -e "cpu-clock, instructions, cache-references, cache-misses, L1-dcache-loads, L1-dcache-load-misses, LLC-loads, LLC-load-misses" ./rtm_serial par=../models/new_mod/input.dat
	perf record -e "cache-references, cache-misses" ./rtm_serial par=../models/new_mod/input.dat
	chmod 777 perf.data
	
	mv 'perf.data' "s$n-perf.data"
	rm "perf.data.old"
done