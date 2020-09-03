#!/bin/sh

build_path=$(pwd)
rm build/mod_main
cd src/
#make allclean
make QMN_F_SIZE=$1 QMN_I_SIZE=$2 CC=$3
echo moving executable to $build_path
mv mod_main $build_path
cd ../
