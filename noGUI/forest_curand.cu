#include <stdlib.h>
#include <unistd.h>
#include <random>
#include <malloc.h>
#include <cuda.h>
#include <time.h>
#include <curand_kernel.h>
//using namespace std;

//maybe
#include <iostream>
#include <stdio.h>

// I/O parameters used to index argv[]
#define OUTPUT_PATH_ID	1
#define STEPS_ID		2
#define BLOCK_SIZE_X	3
#define BLOCK_SIZE_Y	4
#define MATRIX_SIZE		5

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

// Initialize seeds for threads
__global__ void setup_kernel(curandState *state, int d){
	int idx =blockDim.x * blockIdx.x + threadIdx.x;
	int idy =blockDim.y * blockIdx.y + threadIdx.y;
	int id = idy*d+idx;
	curand_init(1234, id, 0, &state[id]);
}


__global__ void transition_function(int d, int total_steps, int *read_matrix, int *write_matrix, curandState *seedStates){
	int x = (blockDim.x*blockIdx.x + threadIdx.x);
	int y = (blockDim.y*blockIdx.y + threadIdx.y);

	int sum;

	if (x < d && y < d){	
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
					float prob = 0.2/7.0*sum + 5.4/7.0;
					curandState localState = seedStates[y*d+x];
					float uniformVal = curand_uniform(&localState);
					seedStates[y*d+x] = localState;
					int bin_num = 0;
					if (uniformVal > 1-prob)
						bin_num = 1;
					//int bin_num = (uniformVal < 1 - prob)? 0 : 1; // uniform to binomial
					write_matrix[y*d+x] = bin_num + 1;
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
	int x = (blockDim.x*blockIdx.x + threadIdx.x);
	int y = (blockDim.y*blockIdx.y + threadIdx.y);
	if (x < d && y < d){
		read_matrix[y*d+x] = write_matrix[y*d+x];
	}
}

void random_seed_matrix(int total_steps, int d, int *seed_array){
	for (int i = 0; i< total_steps*d*d; i++)
		seed_array[i] = rand()%100;
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
	int index_middle = d/2 * d + d/2;
	// introduce a burning cell
	read_matrix[index_middle] = 2;
	write_matrix[index_middle] = 2;
}


//---------------------------------------------------------//
//---------------		MAIN FUNCTION ---------------------//
//---------------------------------------------------------//

int main(int argc, char **argv) {
	srand(1);
	
	// Allocate CPU memory
	int d = atoi(argv[MATRIX_SIZE]);
	int size = d * d * sizeof(int);	
	int total_steps = atoi(argv[STEPS_ID]);

	printf("Dimensio: %d",d);

	int *read_matrix, *write_matrix; 
	read_matrix = (int *)malloc(size);
	write_matrix = (int *)malloc(size);	

	// Block size and number of blocks
	int bs_x, bs_y;
	bs_x = atoi(argv[BLOCK_SIZE_X]);
	bs_y = atoi(argv[BLOCK_SIZE_Y]);


	dim3 block_size(bs_x, bs_y, 1);
	dim3 block_number(ceil((d)/(float)block_size.x), ceil((d)/(float)block_size.y),1);
	
	printf("Files: %d, columnes: %d\n",d,d);
	printf("blocksize_x: %d, blocksize_y: %d\n",bs_x, bs_y);
	printf("Number of blocks (x): %d, Number of blocks (y): %d \n",block_number.x, block_number.y);
	printf("Number of steps: %d",total_steps);

	// Setup seeds
	curandState *seedStates;
	cudaMalloc((void**) &seedStates, d*d*sizeof(curandState));
	setup_kernel<<<block_number,block_size>>>(seedStates,d);

	// Fill read_matrix with initial conditions	
	initForest(d, read_matrix, write_matrix);
	
	// Allocate memory in GPU and copy data  
	int *d_read_matrix, *d_write_matrix;
	
	cudaMalloc((void**) &d_read_matrix, size);
	cudaMalloc((void**) &d_write_matrix, size);

	cudaMemcpy(d_read_matrix, read_matrix, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_write_matrix, write_matrix, size, cudaMemcpyHostToDevice);

	// Simulation 
	for (int timestep = 0; timestep < total_steps; timestep++){
		// Apply transition function
		transition_function<<<block_number, block_size>>>(d, total_steps, d_read_matrix, d_write_matrix, seedStates);

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess){
			printf("CUDA Error in transition_function(): %s\n",cudaGetErrorString(err));
		}
		// Swap read and write matrix
		swap<<<block_number, block_size>>>(d, d_read_matrix, d_write_matrix);	
		err = cudaGetLastError();
		if (err != cudaSuccess){
			printf("CUDA Error in swap(): %s\n",cudaGetErrorString(err));
		}
	}

	printf("Saving data...");
	// Copy data from GPU to CPU
	cudaMemcpy(read_matrix, d_read_matrix, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(write_matrix, d_write_matrix, size, cudaMemcpyDeviceToHost);

	
	// Copy data to file
	saveGrid2Dr(write_matrix,d,argv[OUTPUT_PATH_ID]);
	
	printf("Releasing memory...\n");
	delete [] read_matrix;
	delete [] write_matrix;
	cudaFree(seedStates);
	cudaFree(d_read_matrix);
	cudaFree(d_write_matrix);
	return 0;
}
