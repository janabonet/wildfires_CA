#!/bin/sh
#SBATCH --chdir=/scratch/nas/1/siri1010/wildfires_CA/noGUI
#SBATCH --output=/scratch/nas/1/siri1010/wildfires_CA/noGUI/sortida-%j.out
#SBATCH --error=/scratch/nas/1/siri1010/wildfires_CA/noGUI/error-%j.out
#SBATCH --job-name="cuda"
#SBATCH -A cuda
#SBATCH -p cuda
#SBATCH --gres=gpu:1

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
./forest_curand output_curand.txt $1 $2 $3 $4 
