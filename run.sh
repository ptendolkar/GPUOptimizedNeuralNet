#!/bin/sh

. /opt/modules/init/bash
module load cuda

module load gsl-gnu4
export LD_LIBRARY_PATH=/opt/gsl/1.15/gnu4/lib:$LD_LIBRARY_PATH

./test.o > output.txt

