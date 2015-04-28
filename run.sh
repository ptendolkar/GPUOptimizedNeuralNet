#!/bin/sh
# -- our name ---
#$ -N gpu_neuron_project
#$ -S /bin/sh
# Make sure that the .e and .o file arrive in the
#working directory
#$ -cwd
#Merge the standard out and standard error to one file
#$ -j y
# Send mail at submission and completion of script
#$ -M pt39@njit.edu
# Request a gpu
#$ -l gpu=1

. /opt/modules/init/bash
module load cuda

#module load gsl-gnu4
#export LD_LIBRARY_PATH=/opt/gsl/1.15/gnu4/lib:$LD_LIBRARY_PATH

$HOME/nnet_cuda/test > output.txt

