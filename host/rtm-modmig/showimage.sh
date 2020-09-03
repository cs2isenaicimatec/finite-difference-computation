#!/bin/bash

function cvtImg {

	# n1=$1 # nz dimension
	# input="$2"
	# output="$3"
	# title="$4"

	# d1num=1 # z-axis ticks
	# d2num=3 # x-axis ticks
	# label1="Depth (km)" # z-axis name
	# label2="Distance (km)" # x-axis name
	# labelfont="Helvetica"
	# labelsize=33 # size of label font
	# d1s=1
	# d2s=1

	# # Title parameters
	# titlefont="Helvetica-Bold"
	# titlesize=32

	# # Legend parameters
	# legend=1
	# lstyle="horibottom" # where legend is: horibottom, vertleft, etc
	# units="Velocity (m/s)" # legend units
	# #lbeg=1500 # starting val for truevel
	# lbeg=0 # starting val for veldif
	# lend=5500 # ending val for truevel
	# #lend=2000 # ending val
	# #ldnum=2000 # legend tick
	# ldnum=1000 # legend tick
	# #lwidth=0.2 # width of lbox in cm
	# #lheight=5 # height of lbox in cm


	# # Color, clipping

	# # # Blue green red
	# # wrgb="0,0,1.0"
	# # grgb="0,1,0.4"
	# # brgb="1,0,0"

	# # # Blue white brown
	# # wrgb="0.0,0.0,1.0" 
	# # grgb="1.0,1.0,1.0" 
	# # brgb="0.52,0.34,0.13"

	# # # Blue white red
	# wrgb="0.0,0.0,1.0" 
	# grgb="1,1,1" 
	# brgb="0.8,0.0,0.0"

	# #wclip=1500 for truevel
	# wclip=0 # for veldif
	# #bclip=5500 for truevel
	# bclip=1000 # for veldif

	# # Figure dimensions
	# width=10
	# height=7
	# bps=24 # image resolution, either 12 or 24

	# #wclip=1500 for truevel
	# wclip=0 # for veldif
	# bclip=5500 #for truevel
	# #bclip=1000 # for veldif

	# # Figure dimensions
	# bps=24 # image resolution, either 12 or 24


	# #Full
	# psimage n1=$n1 perc=98 label1="$label1" label2="$title" labelfont="$labelfont" labelsize=$labelsize \
	# legend=$legend lstyle=$lstyle lbeg=$lbeg lend=$lend ldnum=$ldnum lwidth=$lwidth lheight=$lheight units="$units" \
	# bps=$bps<$input >dummy.ps

	n1=$1 # nz dimension
	input="$2"
	output="$3"
	title="$4"
	psimage n1=$NZ perc=97 <$infile >dummy.ps
	#wrgb="$wrgb" grgb="$grgb" brgb="$brgb"  \
	
	
	convert -trim -geometry 100% dummy.ps $output
	#convert -trim -density 550 -geometry 100% dummy.ps $output
	rm dummy.ps

}

function calcIndices {

	DIR=$1
	IQISCRIPT=$2 # path to IQISCRIPT octave script
	NZ=$3
	shots=(1 2 4 8 16 32 64 128 256 375)
	nTests=11
	logFile="iqi_snr.log"
	rm $logFile
	touch $logFile
	for ns in ${shots[@]}; do
		nfrac=20
		nint=0
		
		echo "> Calculating img indices for NS=$ns"
		for k in `seq 1 $nTests`; do
			flBase="imgfloat_s${ns}_w0.bin.png"
			fxBase="imgfix_s${ns}_w0_b${nint}x${nfrac}.bin.png"
			flFile="$DIR/$flBase"
			fxFile="$DIR/$fxBase"

			if [ ! -f $fxFile ]; then
				echo "No $fxFile!"
				nfrac=$((nfrac+1))
			    continue
			fi

			echo ">>> Calc $flFile $fxFile"
			snr=`octave $IQISCRIPT $NZ $flFile $fxFile | grep SNR | cut -d'=' -f2`
			iqi=`octave $IQISCRIPT $NZ $flFile $fxFile | grep IQI | cut -d'=' -f2`

			echo "$flBase x $fxBase: SNR=$snr IQI=$iqi" >> $logFile
			nfrac=$((nfrac+1))
		done
		echo "#######################################################################"
		echo ""
	done
}

function genImg {
	# $1 = nz
	# $2 = model folder
	NZ=$1
	INPUTDIR=$2

	IQISCRIPT="../img-iqi/imgIQI.m"

	#ximage title=$3 n1=$1 < $2 perc=98 &
	echo "> Converting files in $INPUTDIR"
	for k in "$INPUTDIR/imgf"*.bin;
	do
		infile="$k"
		outfile="$k.png"
		base=`basename $infile`
		rm $outfile
		echo "Convert: $infile to $outfile"

		#psimage n1=$NZ perc=98 title=$base <$infile >dummy.ps
		cvtImg $NZ $infile $outfile $base

	#ximage title=$k n1=$NZ < $k perc=98 &
	done
}

function cpyImgs(){
	INPUTDIR=$1
	OUTDIR=$2
	echo "> Copying files to $OUTDIR"
	shots=(1 2 4 8 16 32 64 128 256 375)
	#shots=(1)
	for ns in ${shots[@]}; do
		mkdir "$OUTDIR/${ns}shots"
		cp "$INPUTDIR/imgf"*_s${ns}_* "$OUTDIR/${ns}shots/"
	done
}

NZ=$1
INPUTDIR=$2
OUTDIR=$3
IQISCRIPT="../img-iqi/imgIQI.m"

genImg $NZ $INPUTDIR
cpyImgs $INPUTDIR $OUTDIR
calcIndices $INPUTDIR $IQISCRIPT $NZ
