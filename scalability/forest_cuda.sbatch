#!/bin/sh
#SBATCH --chdir=/scratch/nas/1/siri1010/wildfires_CA/scalability/
#SBATCH --output=/scratch/nas/1/siri1010/wildfires_CA/scalability/sortida-%j.out
#SBATCH --error=/scratch/nas/1/siri1010/wildfires_CA/scalability/error-%j.out
#SBATCH --job-name="cuda"
#SBATCH -A cuda
#SBATCH -p cuda
#SBATCH --gres=gpu:1

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export OMP_NUM_THREADS=40

echo començant serial amb
/usr/bin/time --format=%e ./forest_serial_nogui $1 $2  > output_serial.txt 2>&1 

echo començant omp amb $OMP_NUM_THREADS

/usr/bin/time --format=%e ./forest_omp_nogui $1 $2  > output_omp.txt 2>&1 

echo començant cuda
/usr/bin/time --format=%e ./forest_cuda_nogui $1 11 11 $2 > output_cuda.txt 2>&1 
