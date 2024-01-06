# Makefile for all the codes. 
# Usage:
#1. Compile either the serial code, the parallelized code with OpenMP or the parallelized one with CUDA doing : make serial/omp/cuda.
#2. Execute the compiled code using make run_serial / run_omp / run_cuda

ifndef CPPC
	CPPC = g++
endif



SRC_SERIAL = withGUI/forest_serial.cpp
SRC_SERIAL_NOGUI = noGUI/forest_serial_nogui.cpp

SRC_OMP = withGUI/forest_omp.cpp
SRC_OMP_NOGUI = noGUI/forest_omp_nogui.cpp
SRC_CUDA_NOGUI = noGUI/forest_cuda_nogui.cu
SRC_OMP_TRY = prova.cpp

EXEC_SERIAL = forest_serial
EXEC_SERIAL_NOGUI = forest_serial_nogui

EXEC_OMP = forest_omp
EXEC_OMP_NOGUI = forest_omp_nogui
EXEC_OMP_TRY = prova

EXEC_CUDA_GUI = forest_cuda_nogui

SEEDS_FILE = SEED_SEQUENCE.txt
SRC_SEEDS = create_seed_matrix.cpp
EXEC_SEEDS = seeds_executable


serial:
	$(CPPC) $(SRC_SERIAL) -I./libraries/allegro/include -L./libraries/allegro/lib -lalleg -o  $(EXEC_SERIAL) -O3
	$(CPPC) $(SRC_SERIAL_NOGUI) -o $(EXEC_SERIAL_NOGUI) -O3
omp:
	$(CPPC) -fopenmp $(SRC_OMP) -I./libraries/allegro/include -L./libraries/allegro/lib -lalleg -o  $(EXEC_OMP) -O3
	$(CPPC) -fopenmp $(SRC_OMP_NOGUI) -o $(EXEC_OMP_NOGUI) -O3
	$(CPPC) -fopenmp $(SRC_OMP_TRY) -o $(EXEC_OMP_TRY) -O3
	
cuda:
	nvcc $(SRC_CUDA) -I./libraries/allegro/include -L./libraries/allegro/lib -lalleg -o $(EXEC_CUDA)
	nvcc -o $(EXEC_CUDA_NOGUI) $(SRC_CUDA_NOGUI) -O3 

seeds:
	g++ -o $(EXEC_SEEDS) -fopenmp $(SRC_SEEDS) -O3
	./$(EXEC_SEEDS) $(SEEDS_FILE) 1000 1000
	rm $(EXEC_SEEDS)

run_serial:
	LD_LIBRARY_PATH=./libraries/allegro/lib ./$(EXEC_SERIAL)

run_serial_nogui:
	./$(EXEC_SERIAL_NOGUI) OUTPUT_SER.txt 1000 1000

run_omp:
	LD_LIBRARY_PATH=./libraries/allegro/lib OMP_NUM_THREADS=8 ./$(EXEC_OMP)

run_omp_nogui:
	OMP_NUM_THREADS=8 ./$(EXEC_OMP_NOGUI) OUTPUT_OMP.txt 1000 1000 

run_try:
	OMP_NUM_THREADS=8 ./$(EXEC_OMP_TRY) OUTPUT_OMP_TRY.txt 1000 1000 $(SEEDS_FILE)

run_cuda_nogui:
	./$(EXEC_CUDA_NOGUI)

clean: 
	rm *.out
	
