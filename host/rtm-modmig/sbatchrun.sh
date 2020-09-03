#!/bin/sh
#rm -f dir.*
#export OMP_NUM_THREADS=80
function runModeling(){

	maxProcs=40
	modelPath=$1
	binFileOrig="./mod_main"
	binFile="./mod_main"
	nThreads=$2
	nTests=$3
	cc=$4
	nint=0

	shots=(375)
	rm mod_main*
	rm ${modelPath}/input*.dat
	rm ${modelPath}/output*.txt
	for ns in ${shots[@]}; do
		nProcs=$((ns+1))
		#nProcs=1
		nfrac=20
		if [ "$nProcs" -gt $maxProcs ]; then
			nProcs=$maxProcs
		fi
		echo "> Running Modeling for NS=$ns (Procs=$nProcs)"
		./make.sh $nfrac $nint $cc > /dev/null 2>&1
		if [ ! -f $binFileOrig ]; then
			echo ""
		    echo "> ERROR: Compilation failed. Abort!"
		    exit 1
		fi
		binFile="${binFileOrig}"
		outfile="${modelPath}/output.txt"
		inputfile="${modelPath}/input.dat"
		touch $outfile
		touch $inputfile
		rm $outfile
		rm $inputfile
		cp ${modelPath}/input.orig $inputfile
		floatrun=1;
		fixrun=0;
		echo "" >> $inputfile
		echo "ns=$ns" >> $inputfile
		echo "floatrun=$floatrun" >> $inputfile
		echo "fixrun=$fixrun" >> $inputfile
		echo "modeling=1" >> $inputfile
		# echo "nbitsfrac=$nfrac" >> $inputfile
		# echo "nbitsint=$nint"  >> $inputfile
		echo "> Job $inputfile ==> $outfile "
		sbatch -p standard -x c006,c044 -N $nProcs -o $outfile -c $nThreads ./run.sh $binFile $inputfile
		# wait until all jobs are done

		user=`whoami`
		jobcnt=`squeue -u $user | grep run.sh | wc -l`
		echo "> Waiting for jobs to finish..."
		while [ $jobcnt -gt 0 ]
		do
			jobcnt=`squeue -u $user | grep run.sh | wc -l`
			sleep 1;
		done
		rm mod_main*
	done

}
function runMigration(){
	maxProcs=40
	startPrecision=20
	modelPath=$1
	binFileOrig="./mod_main"
	binFile="./mod_main"
	nThreads=$2
	nTests=$3
	cc=$4
	nint=0

	shots=(1 2 4 8 16 32 64 128 256 375)
	#shots=(375)
	#shots=(2 32)
	#shots=(4)
	rm mod_main*
	rm ${modelPath}/input*.dat
	rm ${modelPath}/output*.txt
	for ns in ${shots[@]}; do
		nProcs=$((ns+1))
		nfrac=$startPrecision
		if [ "$nProcs" -gt $maxProcs ]; then
			nProcs=$maxProcs
		fi
		echo "> Running Migration tests for NS=$ns (Procs=$nProcs nTests=$nTests)"
		for k in `seq 1 $nTests`; do
			./make.sh $nfrac $nint $cc > /dev/null 2>&1
			if [ ! -f $binFileOrig ]; then
				echo ""
			    echo "> ERROR: Compilation failed. Abort!"
			    exit 1
			fi
			binFile="${binFileOrig}_${nfrac}bits"
			mv $binFileOrig "$binFile"
			outfile="${modelPath}/output$k.txt"
			inputfile="${modelPath}/input$k.dat"
			touch $outfile
			touch $inputfile
			rm $outfile
			rm $inputfile
			cp ${modelPath}/input.orig $inputfile
			floatrun=1;
			fixrun=1;
			if [ $k -gt 1 ]; then
				# will run float only on first exec
				floatrun=0;
			fi
			echo "ns=$ns" >> $inputfile
			echo "floatrun=$floatrun" >> $inputfile
			echo "fixrun=$fixrun" >> $inputfile
			echo "modeling=0" >> $inputfile
			# echo "nbitsfrac=$nfrac" >> $inputfile
			# echo "nbitsint=$nint"  >> $inputfile
			echo "> Job $k $inputfile ==> $outfile "
			sbatch -p standard -x c006,c044 -N $nProcs -o $outfile -c $nThreads ./run.sh $binFile $inputfile
			nfrac=$((nfrac+1))
		done
		# wait until all jobs are done

		user=`whoami`
		jobcnt=`squeue -u $user | grep run.sh | wc -l`
		echo "> Waiting for jobs to finish..."
		while [ $jobcnt -gt 0 ]
		do
			jobcnt=`squeue -u $user | grep run.sh | wc -l`
			sleep 1;
		done
		rm mod_main*
	done
}


if [ "$#" -lt 4 ]; then
	echo ""
	echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "> Illegal parameter count! Please, define <path_to_model>, <nThreads>,"
	echo ">"
	echo "> Usage:"
	echo "> ./sbatchrun.sh <path_to_model> <nThreads> <nTests> <compiler>"
    echo ">  <nTests> and <compiler> parameters"
	echo ">"
    echo "> Parameters: "
    echo "> Model: path to the seismic model to be migrated"
    echo ">   nThreads: number of OpenMP threads per process"
    echo ">   nTests  : number of different nBits to be tested "
    echo ">             (starting from 21 bits)"
    echo ">   compiler: 'mpicc' for MPI Run or 'gcc' for serial run"
    echo "> Example:"
    echo ">  ./sbatchrun.sh 'build/spluto' $maxProcs 10 mpicc"
    echo ">   Will run migration/modeling for '10' different bitlengths, "
    echo ">   starting in 20, using the 'build/spluto' model and '$maxProcs' "
    echo ">   OpenMP threads per node" 
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo ""
    exit
fi

modeling=0;
if [ "$#" -eq 5 ]; then
	modeling=$5;
fi

if [ "$modeling" -eq 1 ]; then
	runModeling $1 $2 $3 $4
else
	runMigration $1 $2 $3 $4
fi

#sbatch -p standard -x c006,c044 -N 30 -o output.txt -c $maxProcs ./run.sh $inputfile
#sbatch -p standard -x c006,c044 -N 1 -o output.txt -c $maxProcs ./run.sh $inputfile
