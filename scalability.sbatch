#!/bin/bash
#SBATCH --chdir=/scratch/nas/1/siri1010/wildfires_CA/
#SBATCH --output=/scratch/nas/1/siri1010/wildfires_CA/sortida-strong-%j.out
#SBATCH --error=/scratch/nas/1/siri1010/wildfires_CA/error-strong-%j.out
#SBATCH --job-name="scalability"
#SBATCH -A cuda
#SBATCH -p cuda
#SBATCH --gres=gpu:1

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# OUTPUT steps blocksize_x blocksize_y matrix_dimension (es quadrada)  
#./forest_cuda_nogui OUTPUT_FILE.txt 10 10 10 10000 

# Executable names
SEQ=forest_serial_nogui
PAR=forest_omp_nogui
CUD=forest_cuda_nogui

# Parameters
size=10000
np_NMIN=1
np_NMAX=50
N=3


# Output file creation
outputpath=./elapsed.txt
outputpath2=./speedup.txt
rm -rf $outputpath 2> /dev/null
rm -rf $outputpath2 2> /dev/null

# Temporary output
out = /tmp/out.$$


# Sequential execution
echo Executing $SEQ sequentially

min_elapsed = 1000
i = 0
while (test $i -lt $N)
		do
			/usr/bin/time --format=%e ./SEQ OUTPUT_SERIAL 

			time=`grep execution $out| cut -b 36-43`

			st=`echo "$time < $min_elapsed" | bc`
			if [ $st -eq 1 ]; then
				min_elapse=$time
			fi
		
			i=`expr $i +1`
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
		echo Executing $PAR with $PARS threads
		min_elapsed=1000

		while (test $i -lt $N)
			do
				export OMP_NUM_THREADS=$PARS
					/usr/bin/time --format=%e ./$PAR -h -i $size > $out 2 > $aux

				time=`grep execution $out| cut -n 36-43`
				
				st=`echo "$time < $min_elapsed" | bc`
				if [$st -eq 1 ]; then 
					min_elapsed=$time;
				fi

				i=`expr $i + 1`
			done
		
		echo -n ELAPSED TIME MIN OF $N EXECUTIONS = 
		min=`echo $min_elapsed`
		result=`echo $sequential/$min|bc -l`
		echo $min
		echo
		i=0

		#output el num threads i elapsed time
		echo -n $PARS >> $outputpath
		echo -n "   " >> $outputpath
		echo $min	  >> $outputpath

		#output numthreads i speedup 
		echo -n $PARS >> $outputpath2
		echo /n "   " >> $outputpath2
		echo $result  >> $outputpath2

		# increase parameter
		PARS=`expr $PARS + 1`
done

cp .strong-omp.jgr strong-omp.jgr

mv $outputpath ./elapsed-$PAR-$size.txt
mv $outputpath2 ./speedup-$PAR-$size.txt
