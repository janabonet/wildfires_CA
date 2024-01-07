#Makefile for the trace script use only
SEQ_SRC=forest_serial_nogui.cpp
OMP_SRC=forest_omp_nogui.cpp
# Executable names
SEQ_EXEC=forest_serial_nogui
OMP_EXEC=forest_omp_nogui
 
forest_serial_nogui:  
	g++ $(SEQ_SRC) -o $(SEQ_EXEC) -O3
forest_omp_nogui:
	g++ -fopenmp $(OMP_SRC) -o $(OMP_EXEC) -O3
clean:
	rm $(SEQ_EXEC)
	rm $(OMP_EXEC)
	rm *.out *.txt

