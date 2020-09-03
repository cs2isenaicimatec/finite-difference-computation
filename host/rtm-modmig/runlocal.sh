#!/bin/bash
# $1 cpu or pac
# $2 model_path
#shots=(16 32 64 128 256 375)


if [ "$#" -lt 2 ]; then
	#echo ""
	#echo "> Usage: "
	#echo "> ./runlocal.sh < cpu|pac > < path_to_model > "
	#echo ""
	exit
fi

shots=(1)
#NTs=(10 20 40 80 160 320 740 1480 3004)
NTs=(2048)
binfile=./rtm_modmig_$1
modelpath=$2
#rm rtm_modmig_*
rm ${modelpath}/input*.dat
rm ${modelpath}/output*.txt
#echo "> Building rtm-modmig... "
./build.sh $1 gcc > build.log 2>&1
if [ ! -f $binfile ]; then
	#echo ""
    #echo "> ERROR: Compilation failed. Abort!"
    exit 1
fi
rm build.log

#echo "> Bin File  : $binfile"
#echo "> Model Path: $modelpath"
#for ns in ${shots[@]}; do
for nt in ${NTs[@]}; do

	ns=1;
	echo "> ============================ < "
	echo "> Running for NT=$nt"
	outfile="${modelpath}/output${nt}.txt"
	inputfile="${modelpath}/input${nt}.dat"
	touch $outfile
	touch $inputfile
	rm $outfile
	rm $inputfile
	cp ${modelpath}/input.orig $inputfile
	floatrun=1;
	echo "ns=$ns" >> $inputfile
	echo "nt=$nt" >> $inputfile
	echo "modeling=0" >> $inputfile
	#echo "floatrun=$floatrun" >> $inputfile
	#echo "fixrun=$fixrun" >> $inputfile

	#echo "> Running ./$1 par=$inputfile (NS=$ns OUTFILE=$outfile)"
	time $binfile par=$inputfile #> $outfile

done
echo "> ============================ < "
echo ""
#echo "> Cleaning binary files... "
rm $binfile
#echo "Finished"
#echo ""