#!/bin/sh
#SBATCH --chdir=/scratch/nas/1/siri1010/wildfires_CA/
#SBATCH --output=/scratch/nas/1/siri1010/wildfires_CA/sortida-%j.out
#SBATCH --error=/scratch/nas/1/siri1010/wildfires_CA/-error-%j.out
#SBATCH --job-name="cuda"
#SBATCH -A cuda
#SBATCH -p cuda
#SBATCH --gres=gpu:1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
LD_LIBRARY_PATH=/.libraries/allegro/lib ./forest_serial


