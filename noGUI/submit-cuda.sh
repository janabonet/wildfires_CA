#!/bin/bash
#SBATCH --chdir=/scratch/nas/1/siri1010/wildfires_CA/noGUI/
#SBATCH --output=/scratch/nas/1/siri1010/wildfires_CA/noGUI/sortida-%j.out
#SBATCH --error=/scratch/nas/1/siri1010/wildfires_CA/noGUI/error-%j.out
#SBATCH --job-name="cuda"
#SBATCH -A cuda
#SBATCH -p cuda
#SBATCH --gres=gpu:1

USAGE="\n USAGE: submit-extrae.sh prog [options] \n
		prog		-> Program name \n
		options		-> matrix_size num_steps num_procs (default values 1000 1000 8 \n" 

# Parameters
if [ $# -eq 5 ]; then
    size=$2
	steps=$3
	bs_x=$4
	bs_y=$5
else
	size=1000
	steps=500
	bs_x=10
	bs_y=10
echo Executing with default parameters
fi

if (test $# -ne 1 && test $# -ne 5)
then
        echo -e $USAGE
		        exit 0
fi



# Compile file
make $1

export KMP_AFFINITY=scatter

export LD_PRELOAD=${EXTRAE_HOME}/lib/libomptrace.so
./$1 temp_output.txt $steps $bs_x $bs_y $size
unset LD_PRELOAD
rm -f temp_output.txt

mpi2prv -f TRACE.mpits -o $1.prv -e $1 -paraver
rm -rf  TRACE.mpits set-0 >& /dev/null

