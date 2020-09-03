#!/bin/bash

#	analysis script
#	
#	@author: Anderson


currentDir=`pwd`
repoRoot="`pwd`/.."


cd $repoRoot
echo -n " Exec point: "
pwd
python3 script/analyze_perf_db.py "profiling/report"

#finish up
cd $currentDir