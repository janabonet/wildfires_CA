# Makefile for all the codes. 
# Usage:
#1. Compile either the serial code, the parallelized code with OpenMP or the parallelized one with CUDA doing : make serial/omp/cuda.
#2. Execute the compiled code using make run_serial / run_omp / run_cuda

ifndef CPPC
	CPPC = g++
endif



SRC_SERIAL = forest_serial.cpp
SRC_OMP = forest_omp.cpp
SRC_CUDA = forest_cuda.cu

EXEC_SERIAL = forest_serial
EXEC_OMP = forest_omp
EXEC_CUDA = forest_cuda

serial:
	$(CPPC) $(SRC_SERIAL) -I./libraries/allegro/include -L./libraries/allegro/lib -lalleg -o  $(EXEC_SERIAL) -O3

omp:
	$(CPPC) -fopenmp $(SRC_OMP) -I./libraries/allegro/include -L./libraries/allegro/lib -lalleg -o  $(EXEC_OMP) -O3

cuda:
	nvcc $(SRC_CUDA) -I./libraries/allegro/include -L./libraries/allegro/lib -lalleg -o $(EXEC_CUDA)

run_serial:
	LD_LIBRARY_PATH=./libraries/allegro/lib ./$(EXEC_SERIAL)

run_omp:
	LD_LIBRARY_PATH=./libraries/allegro/lib OMP_NUM_THREADS=8OMP_NUM_THREADS=8 ./$(EXEC_OMP)

run_cuda:
	LD_LIBRARY_PATH=./libraries/allegro/lib ./$(EXEC_CUDA)


