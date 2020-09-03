#!/bin/bash


#kernels=("fdstep_1sreg.cl" "fdstep_11sreg.cl" "laplacian_1sreg.cl" "laplacian_11sreg.cl" "laplacian_37sreg.cl")
#kernels=( "fdstep_11sreg.cl" "laplacian_37sreg.cl")
kernels=($1)
board=$2
flags=$3
for k in ${kernels[@]}; do
	echo "****************************************************"
	echo "> Building kernel $k... "
	folderName="${k%.*}"
	outDir="build/$folderName"
	rm -rf $outDir
	mkdir "$outDir"
	STARTTIME=$(date +%s)
	echo "> Started at: $STARTTIME "
	echo "aoc -board=$board $k -o $outDir/${folderName}.aocx -v -report -no-interleaving=default -fp-relaxed $flags"
	aoc -board=$board "$k" -o "$outDir/${folderName}.aocx" -v -report -no-interleaving=default -fp-relaxed $flags
	ENDTIME=$(date +%s)
	echo "> Finished at: $ENDTIME "
	elapsed_time="$(($ENDTIME - $STARTTIME))"
	echo "> Kernel '$k' took $elapsed_time s to compile."

	echo "> Copying files... "
	cp -R "$outDir/${folderName}/reports/" "$outDir/"
	cp "$outDir/${folderName}/build/output_files/afu_fit.gbs" "$outDir/"
	cp "$outDir/${folderName}/acl_quartus_report.txt" "$outDir/"
	#touch "$outDir/${folderName}.aocx"

done;
echo "> Finished build kernels"
echo "****************************************************"
echo ""