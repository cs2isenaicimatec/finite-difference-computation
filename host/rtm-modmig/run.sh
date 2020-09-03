#!/bin/sh
#rm -f dir.*
#export OMP_NUM_THREADS=80
#você vai ter que coisar aí na mão até 79
numThreads=20
export KMP_AFFINITY="granularity=thread,proclist=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39],explicit"
export OMP_NUM_THREADS=$numThreads
#time ../mod_main par=input.dat
#time mpirun -N $1 ../mod_main par=input.dat
echo "> SRUN $1 par=$2"
time srun $1 par=$2

#sbatch -p standard -x c006 -N 6 -o output.txt -c 20 ./run.sh
#sbatch -p standard -x c006,c044 -N 30 -o output.txt -c 40 ./run.sh
