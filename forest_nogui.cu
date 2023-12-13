#include <stdlib.h>
#include <unistd.h>
#include <random>
#include <malloc.h>
#include <cuda.h>
//using namespace std;

// I/O parameters used to index argv[]
#define STEPS_ID 1
#define BLOCK_SIZE_X 2
#define BLOCK_SIZE_Y 3
#define OUTPUT_PATH_ID 4 


// Macros linearizing buffer 2D indices
#define SET(M,columns,i,j,value) (M[(i)*columns) + j] = value)
#define GET(M,columns,i,j) (M[i*columns + j])
#define BUF_SET(M,rows,columns,n,i,j,value) ( M[n*rows*columns + i*columns + j])
#define BUF_GET(M,rows,columns, n,i,j) ( M[n*rows*columns + i*columns + j])


struct cell{
	int i;
	int j;
};

// Kernel for periodic boundary conditions
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


// TODO: Passar funcio a CUDA
__global__ void transiction_function(std::default_random_engine generator_binomial, int d, int *read_matrix, int *write_matrix){
	int x = (blockDim.x*blockIdx.x + threadIdx.x) + 1;
	int y = (blockDim.y*blockIdx.y + threadIdx.y) + 1;

	int sum;

	if (x < d && y < d){	
		switch(read_matrix[y*d+x]){ // or read_matrix[(n*r*c + i*c) + j]? 
			case 0: 
				write_matrix[y*d+x] = 0; // or write_matrix[(n*r*c + i*c) + j]
				break;
			
			case 1:
				sum = 0;
				for (int i = -1; i <= 1; i++){
					for (int j = -1; i <= 1; j++){
						if (!(i == 0 && j == 0)){
							int indexi = getToroidal(y+i,d);
							int indexj = getToroidal(x+j,d);
							if (read_matrix[indexi*d+indexj] == 2) //(read_matrix[indexi][indexj] == 2)
								sum += 1;
						}
					}
				}

				if (sum > 0){
					float p = 0.8;
					float prob = (-p+1.0)/7.0*sum + (8.0*p-1.0)/7.0;
					std::binomial_distribution<int> BinDist(1,prob);
					int new_state = BinDist(generator_binomial);
					write_matrix[y*d+x] = new_state+1;
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
	if (x < d && y < d)
		read_matrix[y*d+x] = write_matrix[y*d+x];
	}
}


void initForest(int d, int *read_matrix, int *write_matrix){
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
	int d = 500;	
	
	int *write_matrix[d][d];
	int *read_matrix[d][d]; 
	read_matrix = new int[d][d];
	write_matrix = new int[d][d];

	
	int total_steps = atoi(argv[STEPS_ID]);
	
	// Block size and number of blocks
	int bs_x, bs_y;
	bs_x = atoi(argv[BLOCK_SIZE_X]);
	bs_y = atoi(argv[BLOCK_SIZE_Y]);

	dim3 block_size(bs_x, bs_y, 1);
	dim3 number_of_blocks(ceil((d-1)/(float)block_size.x), ceil((d-1)/(float)block_size.y),1);
	
	printf("Files: %d, columnes: %d\n",dim,dim);
	printf("blocksize_x: %d, blocksize_y: %d\n",bs_x, bs_y);
	printf("Number of blocks (x): %d, Number of blocks (y): %d",number_of_blocks.x, number_of_blocks.y);
	

	// Fill read_matrix with initial conditions	
	initForest(d, read_matrix, write_matrix);
	
	// Allocate memory in GPU and copy data  
	int *d_read_matrix, *d_write_matrix;
	
	int size = d*d*sizeof(int);
	cudaMalloc((void**) &d_read_matrix, size);
	cudaMalloc((void**) &d_write_matrix, size);

	cudaMemcpy(d_read_matrix, read_matrix, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_write_matrix, write_matrix, size, cudaMemcpyHostToDevice);


	// Simulation 
	for (int timestep = 0; timestep < total_steps; timestep++){
			transition_function<<<block_number, block_size>>>(generator_binomial, d, d_read_matrix, d_write_matrix);
			// Check for CUDA errors
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess){
				printf("CUDA Error: %s\n",cudaErrorString(err));
			}
			
		swap<<<block_number, block_size>>>(d, d_read_matrix, d_write_matrix);	
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
