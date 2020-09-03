#!/bin/bash

binName="rtm_opencl"
binNameNew="rtm_opencl"
defUseFPGA=""
defUpdatePP=""

if [[ $1 = *"--help"* ]]; then
	# save bts into PAC
	echo "Usage   :  ./build.sh execType  USE_FPGA"
	echo "ExecType:  opencl_cpu, opencl_pac, opencl_gpu"
fi

if [ "$#" -ge 1 ]; then
    binNameNew="rtm_$1"
	defUseFPGA="$2"
	defUpdatePP=$3
fi



rm rtm_opencl*
make clean
make USE_FPGA="$defUseFPGA" UPDATE_PP="$defUpdatePP" || exit 1

mv $binName $binNameNew
chmod 777 $binNameNew
exit 0