// ./forest_omp_nogui OUTPUT_FILE_OMP.txt #steps Matrix_size(square)
#include <stdlib.h>
#include <unistd.h>
#include <random>
#include <omp.h>
//using namespace std;
// #include <iostream>
#include <stdio.h>

// I/O parameters used to index argv[]
#define OUTPUT_PATH_ID 1
#define STEPS_ID 2
#define MATRIX_SIZE 3

#define STRLEN 256

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


// Assure Periodic Boundary Conditions
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

// void transiction_function(int d, int *read_matrix, int *write_matrix, std:: default_random_engine generator_b2b){
void transition_function(int d, int total_steps, int *read_matrix, int *write_matrix, int *seed_matrix){
	int sum;
	for (int y = 0; y < d; ++y) {
		for (int x = 0; x < d; ++x) {	
		switch(read_matrix[y*d+x]){
			case 0: 
				write_matrix[y*d+x] = 0; 
				break;
			case 1:
				sum = 0;
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

				if (sum > 0){
					// seed_matrix[y*d+x] = y+x +5;
					std::default_random_engine generator_b2b;
					generator_b2b.seed(seed_matrix[total_steps*x*y+y*d+x]);
						 
					float prob = 0.2/7.0*sum + 5.4/7.0;
					std::binomial_distribution<int> dist_b2b(1,prob);
					write_matrix[y*d+x] = dist_b2b(generator_b2b) + 1;
				}
				else 
					write_matrix[y*d+x] = 1;
				break;
				
			case 2:
				write_matrix[y*d+x] = 3;
				break;

			case 3:
				write_matrix[y*d+x] = 3;
				break;
			}
		}
	}
}

void swap(int d, int *read_matrix, int *write_matrix){
	for (int y = 0; y < d; ++y) {
		for (int x = 0; x < d; ++x) {
			read_matrix[y*d+x] = write_matrix[y*d+x];
		}
	}
}


void random_seed_matrix(int total_steps, int d, int *seed_array){
	for (int i = 0; i < total_steps*d*d; i++){
		seed_array[i] = rand()%100;	
	}
}


// This function generates the forest (grid) and assigns each cell one of the two possible states: rock (not burnable) or tree (burnable)
void initForest(int d, int *read_matrix, int *write_matrix){
	for (int y = 0; y < d; ++y) {
		for (int x = 0; x < d; ++x) {
			int state = rand()%2; 
			read_matrix[y*d+x]=state;
			write_matrix[y*d+x]=state;
		}
	}
// introduce a burning cell
	read_matrix[250*d+250] = 2;
	write_matrix[250*d+250] = 2;
}



int main(int argc, char **argv) {
	// Starting seeds
	printf("principi\n");
	srand(1);
	printf("srandejat\n");
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

	printf("Allocated readwrite\n");
	// Generate seeds (one matrix for each timestep) 
	int *seed_matrix;
	seed_matrix = (int *)malloc(size*total_steps);
	random_seed_matrix(total_steps, d, seed_matrix);
	printf("Allocated seedmatrix\n");


	printf("Starting simulation ...\n");
	for (int timestep = 0; timestep < total_steps; timestep++){
		
		transition_function(d, total_steps, read_matrix, write_matrix, seed_matrix);
		swap(d,read_matrix,write_matrix);
	}

	printf("Saving data to file...\n");
	saveGrid2Dr(write_matrix, d, argv[OUTPUT_PATH_ID]);

	delete [] read_matrix;
	delete [] write_matrix;
	delete [] seed_matrix;
	
	return 0;
}
