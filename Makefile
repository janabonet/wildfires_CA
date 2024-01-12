# Makefile for all the codes. 
# Usage:
#1. Compile either the serial code, the parallelized code with OpenMP or the parallelized one with CUDA doing : make serial/omp/cuda.
#2. Execute the compiled code using make run_serial / run_omp / run_cuda (_nogui for the no GUI version).

ifndef CPPC
	CPPC = g++
endif


# Fitxers de codi
SRC_SERIAL = withGUI/forest_serial.cpp
SRC_SERIAL_NOGUI = noGUI/forest_serial_nogui.cpp
SRC_WIND = withGUI/forest_serial_wind.cpp

SRC_OMP = withGUI/forest_omp.cpp
SRC_OMP_NOGUI = noGUI/forest_omp_nogui.cpp
SRC_CUDA = noGUI/forest_curand.cu

# Executables
EXEC_SERIAL = forest_serial
EXEC_SERIAL_NOGUI = forest_serial_nogui
EXEC_WIND = forest_wind

EXEC_OMP = forest_omp
EXEC_OMP_NOGUI = forest_omp_nogui

EXEC_CUDA = forest_cuda_nogui


# To compile
serial:
	$(CPPC) $(SRC_SERIAL) -I./libraries/allegro/include -L./libraries/allegro/lib -lalleg -o  $(EXEC_SERIAL) -O3
	$(CPPC) $(SRC_SERIAL_NOGUI) -o $(EXEC_SERIAL_NOGUI) -O3
	$(CPPC) $(SRC_WIND) -I./libraries/allegro/include -L./libraries/allegro/lib -lalleg -o  $(EXEC_WIND) -O3
	
omp:
	$(CPPC) -fopenmp $(SRC_OMP) -I./libraries/allegro/include -L./libraries/allegro/lib -lalleg -o  $(EXEC_OMP) -O3
	 $(CPPC) -fopenmp $(SRC_OMP_NOGUI) -o $(EXEC_OMP_NOGUI) -O3
	
cuda:
	nvcc -o $(EXEC_CUDA) $(SRC_CUDA) -O3 

# To execute
run_serial:
	LD_LIBRARY_PATH=./libraries/allegro/lib ./$(EXEC_SERIAL)

run_serial_nogui:
	./$(EXEC_SERIAL_NOGUI) OUTPUT_SER.txt 1000 1000

run_wind:
	LD_LIBRARY_PATH=./libraries/allegro/lib ./$(EXEC_WIND)

run_omp:
	LD_LIBRARY_PATH=./libraries/allegro/lib OMP_NUM_THREADS=8 ./$(EXEC_OMP)

run_omp_nogui:
	OMP_NUM_THREADS=8 ./$(EXEC_OMP_NOGUI) OUTPUT_OMP.txt 1000 1000 

run_cuda:
	sbatch noGUI/forest_cuda.sbatch OUTPUT_CUDA.txt 1000 10 10 1000

clean: 
	find . .maxdepth 1 -type f -executable -delete
