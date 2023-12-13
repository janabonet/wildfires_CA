#include <stdlib.h>
#include <unistd.h>
#include <random>
#include <cuda.h>
//using namespace std;

// I/O parameters used to index argv[]
#define STEPS_ID 1
#define BLOCK_SIZE_X 2
#define BLOCK_SIZE_Y 3
#define OUTPUT_PATH_ID 4 

__constant__ int d;

struct cell{
	int i;
	int j;
};


//STATES -------
int write_matrix[d][d];  // Write Matrix
int read_matrix[d][d];

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


// TODO: Passar funcio a CUDA (canviar fors per index)
__global__ void transiction_function(std::default_random_engine generator_binomial, int d, int *read_matrix, int *write_matrix){
	int x = (blockDim.x*blockIdx.x + threadIdx.x) + 1;
	int y = (blockDim.x*blockIdx.y + threadIdx.y) + 1;

	int sum;

	if (x < d && y < d){	
		switch(read_matrix[y][x]){
			case 0: 
				write_matrix[y][x] = 0; 
				break;
			case 1:
				sum = 0;
				for (int i = -1; i <= 1; i++){
					for (int j = -1; i <= 1; j++){
						if (!(i == 0 && j == 0)){
							int indexi = getToroidal(y+y,d);
							int indexj = getToroidal(x+j,d);
							if (read_matrix[indexi][indexj] == 2)
								sum += 1;
						}
					}
				}

				if (sum > 0){
					float p = 0.8;
					float prob = (-p+1.0)/7.0*sum + (8.0*p-1.0)/7.0;
					std::binomial_distribution<int> BinDist(1,prob);
					new_state_BurnableToBurning[y][x] = BinDist(generator_binomial);
					write_matrix[y][x] = new_state_BurnableToBurning+1;
				}
				else 
					write_matrix[y][x] = 1;
				break;
				
			case 2:
				write_matrix[y][x] = 3;
				break;

			case 3:
				write_matrix[y][x] = 3;
				break;
			}
		}
}

void swap(){
	for (int y = 0; y < d; ++y) {
		for (int x = 0; x < d; ++x) {
			read_matrix[y][x] = write_matrix[y][x];
		}
	}
}

void global_transiction_function(std::default_random_engine generator_binomial){
	transiction_function(generator_binomial);
	swap();
}

void initForest()
{
// This function generates the forest (grid) and assigns each cell one of the two possible states: rock (not burnable) or tree (burnable)
	for (int y = 0; y < d; ++y) {
		for (int x = 0; x < d; ++x) {
			int state = rand()%2; 
			read_matrix[y][x]=state;
			write_matrix[y][x]=state;
		}
	}
// introduce a burning cell
	read_matrix[250][250] = 2;
	write_matrix[250][250] = 2;
}



int main(int argc, char **argv) {
	// RNG initialization
	srand(1);
	std::default_random_engine generator_binomial;
	generator_binomial.seed(1);

	// Allocate CPU memory
	int d;	
	
	int write_matrix[d][d];
	int read_matrix[d][d]; 
	
	int total_steps = atoi(argv[STEPS_ID]);


	// Block size and number of blocks
	int bs_x, bs_y;
	bs_x = atoi(argv[BLOCK_SIZE_X]);
	bs_y = atoi(argv[BLOCK_SIZE_Y]);

	dim3 block_size(bs_x, bs_y, 1);
	dim3 number_of_blocks(ceil((d-1)/(float)block_size.x), ceil((d-1)/(float)block_size.y),1);
	
	printf("Files: %d, columnes: %d\n",d,d);
	printf("blocksize_x: %d, blocksize_y: %d\n",bs_x, bs_y);
	printf("Number of blocks (x): %d, Number of blocks (y): %d",number_of_blocks.x, number_of_blocks.y);
	

	// Fill read_matrix with initial conditions	
	initForest();
	
	// Allocate memory in GPU and copy data  
	int *d_read_matrix, *d_write_matrix;
	int size = d*d*sizeof(int);
	cudaMalloc((void**) &d_read_matrix, size);
	cudaMalloc((void**) &d_write_matrix, size);

	cudaMemcpy(d_read_matrix, read_matrix, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_write_matrix, write_matrix, size, cudaMemcpyHostToDevice);


	// Simulation 
	for (int timestep = 0; timestep < total_steps; timestep++){
		global_transition_function<<<number_of_blocks, block_size>>>(generator_binomial, d_read_matrix, d_write_matrix);

		// Check for CUDA errors
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess){
			printf("CUDA Error: %s\n",cudaErrorString(err));
		}
	}

	// Copy data from GPU to CPU
	cudaMemcpy(read_matrix, d_read_matrix, cudaMemcpyDeviceToHost);
	cudaMemcpy(write_matrix, d_write_matrix, cudaMemcpyDeviceToHost);

	
	printf("Releasing memory...\n");
	delete [] read_matrix;
	delete [] write_matrix;
	cudaFree(d_read_matrix);
	cudaFree(d_write_matrix);
	return 0;
}
