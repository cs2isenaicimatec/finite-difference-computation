#!/bin/bash

# $1 = nz
# $2 = model folder
OUTPUTDIR=$2
ximage n1=$1 < $OUTPUTDIR/dir.image perc=98 &
