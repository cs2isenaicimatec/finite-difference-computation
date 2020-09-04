#!/bin/bash

export CWPROOT=`pwd`
cd src
make clean
make install

cd ..
