#include <algorithm>
#include <stdlib.h>
#include <unistd.h>
#include <random>
//using namespace std;

const int b_screen = 500;
const int h_screen = 500;
const int d = 500;
int step = 0;

struct cell{
	int i;
	int j;
};

const int neighborhood_size=9;

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

void transiction_function(std::default_random_engine generator_binomial){
	//std::default_random_engine generator_BurnableToBurning;
	//std::binomial_distribution<int> distribution_BurnableToBurning(1,0.95); //p =0.3 de passar de burnable a burning
	int sum;
	for (int y = 0; y < d; ++y) {
		for (int x = 0; x < d; ++x) {	
		switch(read_matrix[y][x]){
			case 0: 
				write_matrix[y][x] = 0; 
				break;
			case 1:
				sum = 0;
				for (i = -1; i <= 1; i++){
					for (j = -1; i <= 1; j++){
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



int main() {
	srand(1);
	std::default_random_engine generator_binomial;
	generator_binomial.seed(1);
	
	initForest();

	for (int timestep = 0; timestep < 1000; timestep++){
		global_transiction_function(generator_binomial);
	}
	return 0;
}
