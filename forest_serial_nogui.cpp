#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
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
cell vicinato[neighborhood_size];

//STATES -------
int write_matrix[d][d];  // Write Matrix
int read_matrix[d][d];

//Moore neighbourhood
void create_neighborhood(int i, int j){
	vicinato[0].i = i;
	vicinato[0].j = j;

	vicinato[1].i = i-1;
	vicinato[1].j = j;

	vicinato[2].i = i;
	vicinato[2].j = j-1;

	vicinato[3].i = i;
	vicinato[3].j = j+1;

	vicinato[4].i = i+1;
	vicinato[4].j = j;

	vicinato[5].i = i-1;
	vicinato[5].j = j-1;

	vicinato[6].i = i+1;
	vicinato[6].j = j-1;

	vicinato[7].i = i+1;
	vicinato[7].j = j+1;

	vicinato[8].i = i-1;
	vicinato[8].j = j+1;
}

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

void transiction_function(){
	std::default_random_engine generator_BurnableToBurning;
	std::binomial_distribution<int> distribution_BurnableToBurning(1,0.95); //p =0.3 de passar de burnable a burning
	int sum;
	for (int y = 0; y < d; ++y) {
		for (int x = 0; x < d; ++x) {	
		switch(read_matrix[y][x]){
			case 0: 
				write_matrix[y][x] = 0; 
				break;

			case 1:
				create_neighborhood(y,x); //agafem les veïnes de la cel·la observadda
				sum = 0;
				for (int n = 1; n < neighborhood_size; ++n) {
					int indexi = getToroidal(vicinato[n].i,d);
					int indexj = getToroidal(vicinato[n].j,d);

					if (read_matrix[indexi][indexj] == 2) { //si es detectava que hi ha un burning, sumem 1
						sum += 1; //read_matrix[indexi][indexj];
					}
				}
				
				if(sum>0){ // si s'ha detectat almenys una cel·la veïna burning, tirar número random i comparar amb probabilitat de passar a estat burning
					int new_state_BurnableToBurning = distribution_BurnableToBurning(generator_BurnableToBurning); //dona 0 o 1
					
					write_matrix[y][x] = new_state_BurnableToBurning+1;
					// PER FER: calcular la probabilitat de passar a burning en funció del número de veïnes burning (més probable de passar 
					// a burning si hi ha més veïnes cremant) i implementar vent
				}
				else
					write_matrix[y][x] =1;
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

void global_transiction_function(){
	transiction_function();
	swap();
	step++;
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
	initForest();

	global_transiction_function();

	return 0;
}
