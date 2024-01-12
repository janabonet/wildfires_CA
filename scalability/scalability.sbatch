#!/bin/bash
#SBATCH --chdir=/scratch/nas/1/siri1010/wildfires_CA/scalability
#SBATCH --output=/scratch/nas/1/siri1010/wildfires_CA/scalability/sortida-strong-%j.out
#SBATCH --error=/scratch/nas/1/siri1010/wildfires_CA/scalability/error-strong-%j.out

# Script to measure the execution time and speedup of the OMP code
# It essentially runs the code with a different number of threads,
# measures the elapsed time with the /usr/bin/time utility and
# outputs the result in a .txt file. 

# Description of program usage
USAGE="\n Usage: sbatch scalability.sbatch [options] \n
		options		-> size: Size of the matrix (default 1000) \n
					-> steps: Number of time steps (default 1000) \n
					-> np_NMAX: Maximum number of OMP threads (default 20)\n"


# File names
SEQ_SRC=forest_ser_nogui.cpp
OMP_SRC=forest_omp_nogui.cpp
# Executable names
SEQ_EXEC=forest_serial_nogui
OMP_EXEC=forest_omp_nogui

# Parameters
if [ $# -eq 3 ]; then
	size=$1
	steps=$2
	np_NMAX=$3
else
	size=10000
	steps=100
	np_NMAX=20
	echo Executing with default parameters
fi
np_NMIN=1
N=3 # number of executions

out=/tmp/out.$$

# Output file creation
outputpath=./elapsed.txt
outputpath2=./speedup.txt
rm -rf $outputpath 2> /dev/null
rm -rf $outputpath2 2> /dev/null

# Compile binaries just in case
echo Compiling
make serial
make omp

# Sequential execution
echo Executing $SEQ sequentially

min_elapsed=1000
i=0
while (test $i -lt $N)
		do
			/usr/bin/time --format=%e ./$SEQ_EXEC $steps $size > $out 2>&1 

			time=$(< $out)

			st=`echo "$time < $min_elapsed" | bc`
			if [ $st -eq 1 ]; then
				min_elapsed=$time
			fi

			rm -f $out		
			rm -f $aux
			rm -f OUTPUT.txt
			i=`expr $i + 1`
		done

echo -n ELAPSED TIME MIN OF $N EXECUTIONS =
sequential=`echo $min_elapsed`
echo $sequential


# Parallel execution
i=0
echo "Starting OpenMP executions..."

PARS=$np_NMIN
while (test $PARS -le $np_NMAX)
	do
		echo Executing $OMP_EXEC with $PARS threads
		min_elapsed=1000

		while (test $i -lt $N)
			do
				export OMP_NUM_THREADS=$PARS
					/usr/bin/time --format=%e ./$OMP_EXEC $steps $size > $out 2>&1 

				time=$(< $out)

				st=`echo "$time < $min_elapsed" | bc`
				if [ $st -eq 1 ]; then 
					min_elapsed=$time;
				fi

				rm -f $out
				i=`expr $i + 1`
			done
		
		echo -n ELAPSED TIME MIN OF $N EXECUTIONS = 
		min=`echo $min_elapsed`
		result=`echo $sequential/$min|bc -l`
		echo $min
		echo
		i=0

		# Output num threads & elapsed time
		echo -n $PARS >> $outputpath
		echo -n "   " >> $outputpath
		echo $min	  >> $outputpath

		# Output numthreads & speedup 
		echo -n $PARS >> $outputpath2
		echo -n "   " >> $outputpath2
		echo $result  >> $outputpath2

		# increase parameter
		PARS=`expr $PARS + 1`
done

#Save results to file 
mv $outputpath ./resultats/elapsed-$OMP_EXEC-$size-$PARS.txt
mv $outputpath2 ./resultats/speedup-$OMP_EXEC-$size-$PARS.txt
