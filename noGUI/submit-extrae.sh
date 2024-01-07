#!/bin/bash
#SBATCH --job-name=submit-extrae.sh
#SBATCH -D .
#SBATCH --output=submit-extrae.sh.o%j
#SBATCH --error=submit-extrae.sh.e%j

# Parameters
if [ $# -eq 2 ]; then
    size=$2
	steps=$3
	nprocs=$4
else
	size=1000
	steps=1000
	nprocs=8
echo Executing with default parameters
fi

HOST=$(echo $HOSTNAME | cut -f 1 -d'.')

if [ ${HOST} = 'boada-6' ] || [ ${HOST} = 'boada-7' ] || [ ${HOST} == 'boada-8' ]
then
	echo "Use sbatch to execute this script"
	exit 0
fi


# Compile file
make $1

export OMP_NUM_THREADS=$num_procs
export KMP_AFFINITY=scatter

export LD_PRELOAD=${EXTRAE_HOME}/lib/libomptrace.so
./$1 $output_file $steps $size
unset LD_PRELOAD

mpi2prv -f TRACE.mpits -o $1.prv -e $1 -paraver
rm -rf  TRACE.mpits set-0 >& /dev/null

