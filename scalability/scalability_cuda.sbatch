#!/bin/bash
#SBATCH --chdir=/scratch/nas/1/siri1010/wildfires_CA/scalability/
#SBATCH --output=/scratch/nas/1/siri1010/wildfires_CA/scalability/sortida-strong-%j.out
#SBATCH --error=/scratch/nas/1/siri1010/wildfires_CA/scalability/error-strong-%j.out

#SBATCH --job-name="cuda"
#SBATCH -A cuda
#SBATCH -p cuda
#SBATCH --gres=gpu:1

# Script to measure the execution time and speedup of the CUDA code
# It essentially runs the code with a different blocksize, measures
# the elapsed time with the /usr/bin/time utility and outputs the 
# result in a .txt file. 


# Description of program usage
USAGE="\n Usage: sbatch scalability.sbatch [options] \n
		options		-> size: Size of the matrix (default 1000) \n
					-> steps: Number of time steps (default 1000) \n
					-> bs_NMAX: Maximum number of block size (default 100)\n"


# File names
SEQ_SRC=forest_ser_nogui.cpp
CUDA_SRC=forest_curand.cu
# Executable names
SEQ_EXEC=forest_serial_nogui
CUDA_EXEC=forest_cuda_nogui

# Parameters
if [ $# -eq 3 ]; then
	size=$1
	steps=$2
	bs_NMAX=$3
else
	size=1000
	steps=1000
	bs_NMAX=32
	echo Executing with default parameters
fi
bs_NMIN=1
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
#make cuda

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
			i=`expr $i + 1`
		done

echo -n ELAPSED TIME MIN OF $N EXECUTIONS =
sequential=`echo $min_elapsed`
echo $sequential


# Parallel execution
i=0
echo "Starting CUDA executions..."

bs=$bs_NMIN
while (test $bs -le $bs_NMAX)
	do
		echo Executing $CUDA_EXEC with blocksize $bs 
		min_elapsed=1000

		while (test $i -lt $N)
			do
					/usr/bin/time --format=%e ./$CUDA_EXEC $steps $bs $bs $size > $out 2>&1 

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

		#output el num threads i elapsed time
		echo -n $bs  >> $outputpath
		echo -n "   " >> $outputpath
		echo $min	  >> $outputpath

		#output numthreads i speedup 
		echo -n $bs >> $outputpath2
		echo -n "   " >> $outputpath2
		echo $result  >> $outputpath2

		# increase parameter
		bs=`expr $bs + 1`
done


mv $outputpath ./resultats/elapsed-$CUDA_EXEC-$size-$bs.txt
mv $outputpath2 ./resultats/speedup-$CUDA_EXEC-$size-$bs.txt
