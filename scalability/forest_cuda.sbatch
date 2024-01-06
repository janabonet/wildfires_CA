#!/bin/sh
#SBATCH --chdir=/scratch/nas/1/siri1010/wildfires_CA/scalability/
#SBATCH --output=/scratch/nas/1/siri1010/wildfires_CA/scalability/sortida-%j.out
#SBATCH --error=/scratch/nas/1/siri1010/wildfires_CA/scalability/error-%j.out
#SBATCH --job-name="cuda"
#SBATCH -A cuda
#SBATCH -p cuda
#SBATCH --gres=gpu:1

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
/usr/bin/time --format=%e ./forest_cuda_nogui 1000 10 10 1000 > output.txt 2>&1 
