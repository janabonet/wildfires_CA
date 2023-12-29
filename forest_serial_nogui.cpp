#include <stdlib.h>
#include <unistd.h>
#include <random>
#include <malloc.h>
//using namespace std;

//maybe
#include <iostream>
#include <stdio.h>

// I/O parameters used to index argv[]
#define STEPS_ID		1
#define BLOCK_SIZE_X	2
#define BLOCK_SIZE_Y	3
#define MATRIX_SIZE		4
#define OUTPUT_PATH_ID	5 
#define STRLEN			256


// Function to save the last iteration matrix

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


// Kernel for periodic boundary conditions
__device__ int getToroidal(int i, int size){
	if(i < 0){
		return i+size;
	}else{
		if(i > size-1){
			return i-size;
		}
	}
	return i;
}


// TODO: Arreglar generador random
__global__ void transition_function(int d, int *read_matrix, int *write_matrix){
	int x = (blockDim.x*blockIdx.x + threadIdx.x) + 1;
	int y = (blockDim.y*blockIdx.y + threadIdx.y) + 1;

	int sum;

	if (x < d && y < d){	
		switch(read_matrix[y*d+x]){ 
			case 0: 
				write_matrix[y*d+x] = 0; 
				break;
			
			case 1:
				sum = 0;
				for (int i = -1; i <= 1; i++){
					for (int j = -1; i <= 1; j++){
						if (!(i == 0 && j == 0)){
							int indexi = getToroidal(y+i,d);
							int indexj = getToroidal(x+j,d);
							if (read_matrix[indexi*d+indexj] == 2) 
								sum += 1;
						}
					}
				}

				if (sum > 0){
					//float p = 0.8;
					//float prob = (-p+1.0)/7.0*sum + (8.0*p-1.0)/7.0;
					//std::binomial_distribution<int> BinDist(1,prob);
					//int new_state = BinDist(generator_binomial);
					//write_matrix[y*d+x] = new_state+1;
					write_matrix[y*d+x] = 2;
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

__global__ void swap(int d, int *read_matrix, int *write_matrix){
	int x = (blockDim.x*blockIdx.x + threadIdx.x) + 1;
	int y = (blockDim.y*blockIdx.y + threadIdx.y) + 1;
	if (x < d && y < d){
		read_matrix[y*d+x] = write_matrix[y*d+x];
	}
}


void initForest(int d, int *read_matrix, int *write_matrix){
// This function generates the forest (grid) and assigns each cell one of the two possible states: rock (not burnable) or tree (burnable)
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


//---------------------------------------------------------//
//---------------		MAIN FUNCTION ---------------------//
//---------------------------------------------------------//

int main(int argc, char **argv) {
	//srand(1);
	//	std::default_random_engine generator_binomial;
	//generator_binomial.seed(1);

	int d = atoi(argv[MATRIX_SIZE]);
	int size = d * d * sizeof(int);
	
	int *read_matrix; 
	int *write_matrix;
	//read_matrix = new int[d][d];
	//write_matrix = new int[d][d];
	read_matrix = (int *)malloc(size);
	write_matrix = (int *)malloc(size);


	int total_steps = atoi(argv[STEPS_ID]);
	
	// Block size and number of blocks
	int bs_x, bs_y;
	bs_x = atoi(argv[BLOCK_SIZE_X]);
	bs_y = atoi(argv[BLOCK_SIZE_Y]);

	dim3 block_size(bs_x, bs_y, 1);
	dim3 block_number(ceil((d)/(float)block_size.x), ceil((d)/(float)block_size.y),1);
	
	printf("Files: %d, columnes: %d\n",d,d);
	printf("blocksize_x: %d, blocksize_y: %d\n",bs_x, bs_y);
	printf("Number of blocks (x): %d, Number of blocks (y): %d \n",block_number.x, block_number.y);
	
	// Fill read_matrix with initial conditions	
	initForest(d, read_matrix, write_matrix);
	
	// Simulation 
	for (int timestep = 0; timestep < total_steps; timestep++){
		transition_function(d, read_matrix, write_matrix);
		swap(d, read_matrix, write_matrix);
	}

	printf("Saving data...");

	// Copy data to file
	saveGrid2Dr(write_matrix,d,argv[OUTPUT_PATH_ID]);
	
	printf("Releasing memory...\n");
	delete [] read_matrix;
	delete [] write_matrix;
	return 0;
}
