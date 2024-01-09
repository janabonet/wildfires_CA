// ./forest_omp_nogui OUTPUT_FILE_OMP.txt #steps Matrix_size(square)
// C libraries
#include <omp.h>
#include <stdio.h>

// C++ libraries
#include <random>
#include <ctime>
// I/O parameters used to index argv[]
#define OUTPUT_PATH_ID 1
#define STEPS_ID 2
#define MATRIX_SIZE 3

#define STRLEN 256

// Class for thread-safe random numbers
class ThreadSafeRNG{
public:
	ThreadSafeRNG(){
		// Each thread gets a seed
		unsigned int seed = static_cast<unsigned int>(omp_get_thread_num() + time(NULL));
		generator.seed(seed);
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
void transition_function(int d, int total_steps, int *read_matrix, int *write_matrix, ThreadSafeRNG rng){
	int sum;
	#pragma omp for schedule(dynamic)
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
						float prob = 0.2/7.0*sum + 5.4/7.0;
						write_matrix[y*d+x] = rng.getBinNumber(prob) + 1;
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
	// Starting seed for grid initial conditions
	srand(1);

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
	
	#pragma omp parallel
	{
		ThreadSafeRNG rng;
		for (int timestep = 0; timestep < total_steps; timestep++){	
			transition_function(d, total_steps, read_matrix, write_matrix, rng);
			swap(d,read_matrix,write_matrix);
		}
	}
	printf("Saving data to file...\n");
	saveGrid2Dr(write_matrix, d, argv[OUTPUT_PATH_ID]);

	delete [] read_matrix;
	delete [] write_matrix;	
	return 0;
}
