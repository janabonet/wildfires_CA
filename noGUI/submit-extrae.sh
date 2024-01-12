#!/bin/bash
#SBATCH --job-name=submit-extrae.sh
#SBATCH -D .
#SBATCH --output=output-%j.out
#SBATCH --error=error-%j.out

USAGE="\n USAGE: submit-extrae.sh prog [options] \n
		prog		-> Program name \n
		options		-> matrix_size num_steps num_procs (default values 1000 1000 8 \n" 

# Parameters
if [ $# -eq 4 ]; then
    size=$2
	steps=$3
	nprocs=$4
else
	size=1000
	steps=500
	nprocs=8
echo Executing with default parameters
fi

if (test $# -ne 1 && test $# -ne 4)
then
        echo -e $USAGE
		        exit 0
fi


HOST=$(echo $HOSTNAME | cut -f 1 -d'.')

if [ ${HOST} = 'boada-6' ] || [ ${HOST} = 'boada-7' ] || [ ${HOST} == 'boada-8' ]
then
	echo "Use sbatch to execute this script"
	exit 0
fi



# Compile file
echo Compiling file
make $1

export OMP_NUM_THREADS=$nprocs
export KMP_AFFINITY=scatter

echo abans de correrho
export LD_PRELOAD=${EXTRAE_HOME}/lib/libomptrace.so
./$1 temp_output.txt $steps $size
unset LD_PRELOAD
rm -f temp_output.txt

mpi2prv -f TRACE.mpits -o $1.prv -e $1 -paraver
rm -rf  TRACE.mpits set-0 >& /dev/null

