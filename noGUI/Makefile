#Makefile to use only fore the trace script. To compile normally, use the Makefile in ../

SEQ_SRC=forest_serial_nogui.cpp
OMP_SRC=forest_omp_nogui.cpp
CUDA_SRC=forest_curand.cu

# Executable names
SEQ_EXEC=forest_serial_nogui
OMP_EXEC=forest_omp_nogui
CUDA_EXEC=forest_curand

forest_serial_nogui:  
	g++ $(SEQ_SRC) -o $(SEQ_EXEC) -O3
forest_omp_nogui:
	g++ -fopenmp $(OMP_SRC) -o $(OMP_EXEC) -O3
forest_cuda_nogui:
	nvcc -o $(CUDA_EXEC) $(CUDA_SRC) -O3

clean:
	rm $(SEQ_EXEC)
	rm $(OMP_EXEC)
	rm $(CUDA_EXEC
	rm *.out *.txt

