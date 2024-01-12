// Code for a simulation of a wildfire parallelized with OMP.
// Compile with "make omp", execute with "make run_omp_nogui"
// Results are in OUTPUT.txt

// C libraries
#include <stdio.h>
#include <stdlib.h>
// C++ libraries
#include <random>
#include <ctime>

// I/O parameters used to index argv[]
// Name of output file
#define OUTPUT_PATH_ID 1
// Number of steps in simulation
#define STEPS_ID 2
// Size of the grid
#define MATRIX_SIZE 3

// Variable used in saveGrid2Dr
#define STRLEN 256

// Class for clearer syntax of random numbers
class SerialRNG{
    public:
        SerialRNG(){
            generator.seed(time(NULL));
        }
        int getBinNumber(double p){
            std::binomial_distribution<int> distribution(1,p);
            return distribution(generator);
        }
    private:
    std::default_random_engine generator;
};

// Function to save last configuration
bool saveGrid2Dr(int *M, int d, char *path){
	FILE *f;
	f = fopen(path,"w");

	if (!f)
		return false;

	char str[STRLEN];
	for (int i = 0; i < d; i++){
		for (int j = 0; j < d; j++){
			sprintf(str,"%d ",M[i*d+j]);
			fprintf(f,"%s ", str);
		}
		fprintf(f,"\n");
	}
	fclose(f);

	return true;
}

// Function for Periodic Boundary Conditions
int getToroidal(int i, int size){
	if(i < 0){
		return i+size;
	}else{
		if(i > size-1){
			return i-size;
		}
	}
	return i;
}

// Principal function. Applies transition function to grid
void transition_function(int d, int total_steps, int *read_matrix, int *write_matrix, SerialRNG rng){
	int sum;
	for (int y = 0; y < d; ++y) {
		for (int x = 0; x < d; ++x) {	
		switch(read_matrix[y*d+x]){
			// Cell is not burnable
			case 0: 
				write_matrix[y*d+x] = 0; 
				break;
			// Cell is burnable
			case 1:
				sum = 0;
				// Count burning neighbours (with PBC)
				for (int i = -1; i <= 1; i++){
					for (int j = -1; j <= 1; j++){
						if (!(i == 0 && j == 0)){
							int indexi = getToroidal(y+i,d);
							int indexj = getToroidal(x+j,d);
							if (read_matrix[indexi*d+indexj] == 2)
								sum += 1;
						}
					}
				}
				// Cell has more at least one burning neighbour, pass or not to burning
				if (sum > 0){
					float prob = 0.2/7.0*sum + 5.4/7.0;
					write_matrix[y*d+x] = rng.getBinNumber(prob) + 1;
				}
				// No burning neighbours
				else 
					write_matrix[y*d+x] = 1;
				break;
			//Cell is burning
			case 2:
				write_matrix[y*d+x] = 3;
				break;
			// Cell is burnt
			case 3:
				write_matrix[y*d+x] = 3;
				break;
			}
		}
	}
}

// Function to copy data from write_matrix to read_matrix
void swap(int d, int *read_matrix, int *write_matrix){
	for (int y = 0; y < d; ++y) {
		for (int x = 0; x < d; ++x) {
			read_matrix[y*d+x] = write_matrix[y*d+x];
		}
	}
}

/*
Function to generate initial condition. It assigns the state 
"not burnable" (0) to a cell to simulate a rock and state 
"burnable" (1) to a cell to simulate a tree.
We introduce a burning cell in the middle of the grid
*/
void initForest(int d, int *read_matrix, int *write_matrix){
	for (int y = 0; y < d; ++y) {
		for (int x = 0; x < d; ++x) {
			int state = rand()%2; 
			read_matrix[y*d+x]=state;
			write_matrix[y*d+x]=state;
		}
	}
	// Introduce a burning cell
	int index_middle = d/2 * d + d/2;
	read_matrix[index_middle] = 2;
	write_matrix[index_middle] = 2;
}

// MAIN FUNCTION
int main(int argc, char **argv) {
	// Initialize seeds
	srand(1); // For the initial conditions
	SerialRNG rng; // For the transition function
	
	// Memory allocation
	int d = atoi(argv[MATRIX_SIZE]);
	int size = d*d*sizeof(int);
	int total_steps = atoi(argv[STEPS_ID]);

	printf("d = %d\n",d);
	printf("total steps = %d\n",total_steps);

	// Matrices of CA
	int *read_matrix;
	int *write_matrix;
	read_matrix = (int *)malloc(size);
	write_matrix = (int *)malloc(size);
	
	// Fill read_matrix with initial conditions
	initForest(d, read_matrix, write_matrix);

	printf("Starting simulation ...\n");
	for (int timestep = 0; timestep < total_steps; timestep++){		
		transition_function(d, total_steps, read_matrix, write_matrix, rng);
		swap(d,read_matrix,write_matrix);
	}

	printf("Saving data to file...\n");
	saveGrid2Dr(write_matrix, d, argv[OUTPUT_PATH_ID]);

	// Memory releasing
	delete [] read_matrix;
	delete [] write_matrix;
	
	return 0;
}
