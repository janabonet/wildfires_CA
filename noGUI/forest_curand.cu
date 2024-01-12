// Code for a simulation of a wildfire parallelized with CUDA.
// Compile with "make cuda", execute by sending the "forest_cuda.sbatch" script through sbatch.
// Results are in OUTPUT_PATH_ID

// C libraries
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>

// I/O parameters used to index argv[]
// Name of the output file
#define OUTPUT_PATH_ID 1
// Number of steps in simulation
#define STEPS_ID 2
// Block size, dimension x
#define BLOCK_SIZE_X 3
// Block size, dimension y
#define BLOCK_SIZE_Y 4
// Size of the grid
#define MATRIX_SIZE 5

// Variable used in saveGrid2Dr
#define STRLEN 256


// Function to save the last configuration to file
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

// Function to initialize a different rng seed to all threads
__global__ void setup_kernel(curandState *state, int d){
	int idx =blockDim.x * blockIdx.x + threadIdx.x;
	int idy =blockDim.y * blockIdx.y + threadIdx.y;
	int id = idy*d+idx;
	if (idx < d && idy < d)
		curand_init(1234, id, 0, &state[id]);
}

// Principal function. Applies transition function to grid.
__global__ void transition_function(int d, int total_steps, int *read_matrix, int *write_matrix, curandState *seedStates){
	// Fetch grid index
	int x = (blockDim.x*blockIdx.x + threadIdx.x);
	int y = (blockDim.y*blockIdx.y + threadIdx.y);

	int sum;
	if (x < d && y < d){	// Make sure index is inbounds
		switch(read_matrix[y*d+x]){ 
			// Cell is not burnable
			case 0: 
				write_matrix[y*d+x] = 0; 
				break;
			// Cell is burnable
			case 1:
				sum = 0;
				for (int i = -1; i <= 1; i++){
					for (int j = -1; j <= 1; j++){
						if (!(i == 0 && j == 0)){
							// Count burning neighbours (with PBC)
							int indexi = getToroidal(y+i,d);
							int indexj = getToroidal(x+j,d);
							if (read_matrix[indexi*d+indexj] == 2) 
								sum += 1;
						}
					}
				}
				// Cell has at least one burning neighbour. Pass or not to burning 
				if (sum > 0){
					// Calculate parameter p, prob of burning
					float prob = 0.2/7.0*sum + 5.4/7.0;
					// Fetch seed at thread and generate u ~ Unif(0,1)
					curandState localState = seedStates[y*d+x];
					float uniformVal = curand_uniform(&localState);
					seedStates[y*d+x] = localState;
					// Apply inversion algorithm to get x ~ Bin(1,prob)
					int bin_num = (uniformVal < 1 - prob)? 0 : 1; // uniform to binomial
					write_matrix[y*d+x] = bin_num + 1;
				}
				// No burning neighbours
				else 
					write_matrix[y*d+x] = 1;
				break;	
			// Cell is burning
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

// Kernel to copy data from write_matrix to read_matrix
__global__ void swap(int d, int *read_matrix, int *write_matrix){
	int x = (blockDim.x*blockIdx.x + threadIdx.x);
	int y = (blockDim.y*blockIdx.y + threadIdx.y);
	if (x < d && y < d){
		read_matrix[y*d+x] = write_matrix[y*d+x];
	}
}

/*
Function to generate initial condition. It assigns the state 
"not burnable" (0) to a cell to simulate a rock and state 
"burnable" (1) to a cell to simulate a tree.
We introduce a burning cell in the middle of the grid
*/
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


// MAIN FUNCTION
int main(int argc, char **argv) {
	// Initialize seed for initial conditions
	srand(1);
	
	// Declare and allocate CPU memory
	int d = atoi(argv[MATRIX_SIZE]);
	int size = d * d * sizeof(int);	
	int total_steps = atoi(argv[STEPS_ID]);

	int *read_matrix, *write_matrix; 
	read_matrix = (int *)malloc(size);
	write_matrix = (int *)malloc(size);	

	// Define block size and number of blocks such that block_size*block_number = grid_size
	int bs_x, bs_y;
	bs_x = atoi(argv[BLOCK_SIZE_X]);
	bs_y = atoi(argv[BLOCK_SIZE_Y]);

	dim3 block_size(bs_x, bs_y, 1);
	dim3 block_number(ceil((d)/(float)block_size.x), ceil((d)/(float)block_size.y),1);
	
	// Information messages
	printf("Files: %d, columnes: %d\n",d,d);
	printf("blocksize_x: %d, blocksize_y: %d\n",bs_x, bs_y);
	printf("Number of blocks (x): %d, Number of blocks (y): %d \n",block_number.x, block_number.y);
	printf("Number of steps: %d \n",total_steps);

	// Declare rng at each thread and initialize different seed.
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

	// Copy data from GPU to CPU
	printf("Saving data...\n");
	cudaMemcpy(read_matrix, d_read_matrix, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(write_matrix, d_write_matrix, size, cudaMemcpyDeviceToHost);

	// Copy data to file
	saveGrid2Dr(write_matrix,d,argv[OUTPUT_PATH_ID]);

	// Release memory	
	printf("Releasing memory...\n");
	delete [] read_matrix;
	delete [] write_matrix;
	cudaFree(seedStates);
	cudaFree(d_read_matrix);
	cudaFree(d_write_matrix);
	return 0;
}
