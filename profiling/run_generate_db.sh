#!/bin/bash

#
#	Perf db generation
#	
#	usage:   ./run_dbgenerate.sh serial 16
#	This script generates a set of perf.db
# 	files for all executions for every execModel 
#	in input/inputlist file.
#
#	@author: Anderson
if [ "$#" -ne 1 ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "! ERROR     : Illegal number of parameters        !"
    echo "! Usage     : ./run_dbgenerate.sh ExecType        !"
    echo "! ExecType  : serial, fpga, cuda, openmp,         !"
    echo "!             opencl_pac, opencl_gpu, opencl_cpu  !"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit
fi

currentDir=`pwd`
repoRoot="`pwd`/.."
inputListFile="$repoRoot/profiling/input/inputlist"
execPoint="$repoRoot"
nModels=$(cat $inputListFile | wc -l)
execType="$1"
execsPerModel=$2
perfEvents="cpu-clock, instructions, cache-references, cache-misses, L1-dcache-loads, L1-dcache-load-misses, LLC-loads, LLC-load-misses"
perfEventsArray=($(echo "$perfEvents" | tr ',' ' '))
logOuput="exec_log.log"


echo "==================================================="
echo "   PERFORMANCE TEST SUIT FOR RTM IMPLEMENTATIONS   "
echo "==================================================="
echo "> Summary:"
echo "   Execution Type  : $execType" # serial, opencl, cuda ...
echo "   Execution Point : $execPoint"
echo "   Number of Models: $nModels"
echo "   Perf Events     : "
for n in ${perfEventsArray[@]};do
	echo -e "\t\t     $n"
done
echo "==================================================="
echo "> Generating DB for all input models... "
cd $execPoint

while read currentModel; do
	
	# limit program eventual outputs
	echo "==================================================="
	echo "> Current Model: $currentModel "
	echo "> Generating DB: "
	perfDataPath="$repoRoot/profiling/perf/$currentModel/$execType"
	cd $perfDataPath
	perfDataFiles=($(ls "$perfDataPath/"*perf.data))
	cd $execPoint
	count=0;
	for perfData in ${perfDataFiles[@]};do
		STARTTIME=$(date +%s)
		dbName="${currentModel}_${execType}_db$count"
		runTime="`cat ${perfData}_runtime`"
		#touch "perf.db"
		echo -ne "\t     $perfData => $dbName "
		#sleep 1
		perf script -i "$perfData" -s "$repoRoot/script/export-to-postgresql.py" "$dbName" "$runTime" > /dev/null 2>&1
		ENDTIME=$(date +%s)
		echo "($(($ENDTIME - $STARTTIME)) s)"
		#mv "perf.db" "$perfDataPath/$perfData.db"
		#touch "perf.data.old"
		#rm "perf.data.old"
		count=$((count+1))
	done
	echo "==================================================="
done <$inputListFile


#finish up
cd $currentDir