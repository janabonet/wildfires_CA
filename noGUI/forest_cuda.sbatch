#!/bin/sh
#SBATCH --output=sortida-%j.out
#SBATCH --error=error-%j.out
#SBATCH --job-name="cuda"
#SBATCH -A cuda
#SBATCH -p cuda
#SBATCH --gres=gpu:1

# Info: file to execute cuda code easily. 
# Inputs:
#    $1: number of steps
#    $2: block size, direction x
#    $3: block size, direction y
#    $4: Matrix dimension (square)
if [ $# -eq 5 ]; then
 	outputfile=$1
	steps=$2
	bs_x=$3
 	bs_y=$4
  	size=$5
else
  	outputfile=OUTPUT_CUDA.txt
	steps=1000
	bs_x=10
  	bs_y=10
  	size=1000
	echo Executing with default parameters
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
./forest_cuda_nogui $outputfile $steps $bs_x $bs_y $size 
