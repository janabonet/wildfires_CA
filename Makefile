# Makefile for all the codes. 
# Usage:
#1. Compile either the serial code, the parallelized code with OpenMP or the parallelized one with CUDA doing : make serial/omp/cuda.
#2. Execute the compiled code using make run_serial / run_omp / run_cuda

ifndef CPPC
	CPPC = g++
endif



SRC_SERIAL = forest_serial.cpp
SRC_SERIAL_NOGUI = forest_serial_nogui.cpp

SRC_OMP = forest_omp.cpp
SRC_OMP_NOGUI = forest_omp_nogui.cpp

SRC_CUDA = forest_cuda.cu
SRC_CUDA_NOGUI = forest_cuda_nogui.cu

EXEC_SERIAL = forest_serial
EXEC_SERIAL_NOGUI = forest_serial_nogui

EXEC_OMP = forest_omp
EXEC_OMP_NOGUI = forest_omp_nogui

EXEC_CUDA = forest_cuda
EXEC_CUDA_GUI = forest_cuda_nogui


serial:
	$(CPPC) $(SRC_SERIAL) -I./libraries/allegro/include -L./libraries/allegro/lib -lalleg -o  $(EXEC_SERIAL) -O3
	$(CPPC) $(SRC_SERIAL_NOGUI) -o $(EXEC_SERIAL_NOGUI) -O3
omp:
	$(CPPC) -fopenmp $(SRC_OMP) -I./libraries/allegro/include -L./libraries/allegro/lib -lalleg -o  $(EXEC_OMP) -O3
	$(CPPC) -fopenmp $(SRC_OMP_NOGUI) -o $(EXEC_OMP_NOGUI) -O3
cuda:
	nvcc $(SRC_CUDA) -I./libraries/allegro/include -L./libraries/allegro/lib -lalleg -o $(EXEC_CUDA)
	nvcc $(SRC_CUDA_NOGUI) -o $(EXEC_CUDA_NOGUI)



run_serial:
	LD_LIBRARY_PATH=./libraries/allegro/lib ./$(EXEC_SERIAL)

run_serial_nogui:
	./$(EXEC_SERIAL_NOGUI)

run_omp:
	LD_LIBRARY_PATH=./libraries/allegro/lib OMP_NUM_THREADS=8 ./$(EXEC_OMP)

run_omp_nogui:
	OMP_NUM_THREADS=8 ./$(EXEC_OMP_NOGUI)

run_cuda:
	LD_LIBRARY_PATH=./libraries/allegro/lib ./$(EXEC_CUDA)

run_cuda_nogui:
	./$(EXEC_CUDA_NOGUI)

