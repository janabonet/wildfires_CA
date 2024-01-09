#Makefile for the trace script use only
SEQ_SRC=forest_serial_nogui.cpp
OMP_SRC=forest_omp_nogui.cpp
SRC_CUDA = forest_nogui.cu
SRC_CURAND = forest_curand.cu

# Executable names
SEQ_EXEC=forest_serial_nogui
OMP_EXEC=forest_omp_nogui
EXEC_CUDA = forest_cuda_nogui
EXEC_CURAND = forest_curand

forest_serial_nogui:  
	g++ $(SEQ_SRC) -o $(SEQ_EXEC) -O3
forest_omp_nogui:
	g++ -fopenmp $(OMP_SRC) -o $(OMP_EXEC) -O3
forest_cuda_nogui:
	nvcc -o $(EXEC_CUDA) $(SRC_CUDA) -O3
	nvcc -o $(EXEC_CURAND) $(SRC_CURAND) -O3

clean:
	rm $(SEQ_EXEC)
	rm $(OMP_EXEC)
	rm *.out *.txt

