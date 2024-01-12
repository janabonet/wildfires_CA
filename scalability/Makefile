# Makefile to use only in the scalability.sbatch script.

SEQ_SRC=forest_ser_nogui.cpp
OMP_SRC=forest_omp_nogui.cpp
CUDA_SRC=forest_nogui.cu
# Executable names
SEQ_EXEC=forest_serial_nogui
OMP_EXEC=forest_omp_nogui
CUDA_EXEC=forest_cuda_nogui

 
serial:  
	g++ $(SEQ_SRC) -o $(SEQ_EXEC) -O3
omp:
	g++ -fopenmp $(OMP_SRC) -o $(OMP_EXEC) -O3
cuda:
	nvcc -o $(CUDA_EXEC) $(CUDA_SRC) -O3
clean: 
	rm $(SEQ_EXEC)
	rm $(OMP_EXEC)
	rm $(CUDA_EXEC)
	rm *.out *.txt
