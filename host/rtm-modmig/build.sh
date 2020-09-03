#!/bin/bash

binName="rtm_modmig"
binNameNew="rtm_modmig"
defUseFPGA="DONT_USE_FPGA"
defCC="mpicc"

if [[ $1 = *"--help"* ]]; then
	# save bts into PAC
	echo "Usage   :  ./build.sh execType  cc"
	echo "ExecType:  serial, cpu, pac, gpu"
fi

if [ "$#" -ge 1 ]; then
    binNameNew="rtm_modmig_$1"
	
    if [[ $1 = *"pac"* ]]; then
		defUseFPGA="USE_FPGA"
	fi

	defCC=$2
fi


#rm ${binName}_*
make clean
make USE_FPGA="$defUseFPGA" CC="$defCC" || exit 1

mv $binName $binNameNew
chmod 777 $binNameNew
exit 0