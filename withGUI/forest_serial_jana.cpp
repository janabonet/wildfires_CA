#include <allegro.h>
//#include <conio.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <random>
//using namespace std;

const int b_screen = 1000;
const int h_screen = 1000;
const int d = 1000;
int zoom=1;
BITMAP* buffer;
int step = 0;

struct cell{
	int i;
	int j;
};

const int neighborhood_size=9;
cell vicinato[neighborhood_size];

//STATES -------

int read_matrix[d][d];   // Read  Matrix
int write_matrix[d][d];  // Write Matrix

//ALLEGRO COLOR -----

int green; //tree
int white; //not burnable
int yellow; // burnable
int orange; //burning	
int black; //burned
int grey;

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
    int vent = 7;
	int sum,indexi,indexj;
    int indexi_left,indexj_left,indexi_right,indexj_right,indexi_middle,indexj_middle;
	for (int y = 0; y < d; ++y) {
		for (int x = 0; x < d; ++x) {	
            switch(read_matrix[y][x]){
                case 0: 
                    write_matrix[y][x] = 0; 
                    break;
                case 1: // burnable
                    // create_neighborhood(y,x); //agafem les veïnes de la cel·la observada
                    switch(vent) {
                        case 0: //no vent
                            sum = 0;
                            for (int i = -1; i <= 1; i++){
                            for (int j = -1; j <= 1; j++){
                                if (!(i == 0 && j == 0)){
                                    indexi = getToroidal(y+i,d);
                                    indexj = getToroidal(x+j,d);
                                    if (read_matrix[indexi][indexj] == 2)//si es detectava que hi ha un burning, sumem 1
                                        sum += 1;
                                } 
                            }
                            }
                            
                            if(sum>0){ 
                                float p = 0.8;
                                float prob = (-p+1.0)/7.0*sum + (8.0*p-1.0)/7.0;
                                std::binomial_distribution<int> distribution_BurnableToBurning(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurning(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }
                            else
                                write_matrix[y][x] =1;
                        break;

                        case 1: //N 
                            indexi_middle = getToroidal(y,d);
                            indexj_middle = getToroidal(x+1,d);
                            indexi_right = getToroidal(y+1,d);
                            indexj_right = getToroidal(x+1,d);
                            indexi_left = getToroidal(y-1,d);
                            indexj_left = getToroidal(x+1,d);

                            sum = 8;

                            if (read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_right][indexj_right] == 2 && read_matrix[indexi_left][indexj_left]) {
                                float prob = 1.0;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }

                            else if ((read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_right][indexj_right] == 2)||((read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_left][indexj_left] == 2))) {
                                float prob = 0.9;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }                    
                            
                            else if (read_matrix[indexi_middle][indexj_middle] == 2) {
                                float prob = 0.7;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }

                            else if ((read_matrix[indexi_left][indexj_left] == 2) ||(read_matrix[indexi_right][indexj_right] == 2)) {
                                float prob = 0.65;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }
                                 
                            else
                                write_matrix[y][x] =1;

                        break;

                        case 2: // S
                            indexi_middle = getToroidal(y,d);
                            indexj_middle = getToroidal(x-1,d);
                            indexi_right = getToroidal(y+1,d);
                            indexj_right = getToroidal(x+1,d);
                            indexi_left = getToroidal(y-1,d);
                            indexj_left = getToroidal(x-1,d);

                            sum = 8;

                            if (read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_right][indexj_right] == 2 && read_matrix[indexi_left][indexj_left]) {
                                float prob = 1.0;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }

                            else if ((read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_right][indexj_right] == 2)||((read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_left][indexj_left] == 2))) {
                                float prob = 0.9;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }                    
                            
                            else if (read_matrix[indexi_middle][indexj_middle] == 2) {
                                float prob = 0.7;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }

                            else if ((read_matrix[indexi_left][indexj_left] == 2)||(read_matrix[indexi_right][indexj_right] == 2)) {
                                float prob = 0.65;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }
                                 
                            else
                                write_matrix[y][x] =1;
                        break;

                        case 3: // O
                            indexi_middle = getToroidal(y+1,d);
                            indexj_middle = getToroidal(x,d);
                            indexi_right = getToroidal(y+1,d);
                            indexj_right = getToroidal(x-1,d);
                            indexi_left = getToroidal(y+1,d);
                            indexj_left = getToroidal(x+1,d);

                            sum = 8;

                            if (read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_right][indexj_right] == 2 && read_matrix[indexi_left][indexj_left]) {
                                float prob = 1.0;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }

                            else if ((read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_right][indexj_right] == 2)||((read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_left][indexj_left] == 2))) {
                                float prob = 0.9;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }                    
                            
                            else if (read_matrix[indexi_middle][indexj_middle] == 2) {
                                float prob = 0.7;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }

                            else if ((read_matrix[indexi_left][indexj_left] == 2)||(read_matrix[indexi_right][indexj_right] == 2)) {
                                float prob = 0.65;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }
                                 
                            else
                                write_matrix[y][x] =1;
                        break;

                        case 4: // E
                            indexi_middle = getToroidal(y-1,d);
                            indexj_middle = getToroidal(x,d);
                            indexi_right = getToroidal(y-1,d);
                            indexj_right = getToroidal(x+1,d);
                            indexi_left = getToroidal(y-1,d);
                            indexj_left = getToroidal(x-1,d);

                            sum = 8;

                            if (read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_right][indexj_right] == 2 && read_matrix[indexi_left][indexj_left]) {
                                float prob = 1.0;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }

                            else if ((read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_right][indexj_right] == 2)||((read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_left][indexj_left] == 2))) {
                                float prob = 0.9;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }                    
                            
                            else if (read_matrix[indexi_middle][indexj_middle] == 2) {
                                float prob = 0.7;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }

                            else if ((read_matrix[indexi_left][indexj_left] == 2)||(read_matrix[indexi_right][indexj_right] == 2)) {
                                float prob = 0.65;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }
                                 
                            else
                                write_matrix[y][x] =1;
                        break;

                        case 5: // SO
                            indexi_middle = getToroidal(y+1,d);
                            indexj_middle = getToroidal(x-1,d);
                            indexi_right = getToroidal(y,d);
                            indexj_right = getToroidal(x-1,d);
                            indexi_left = getToroidal(y+1,d);
                            indexj_left = getToroidal(x,d);

                            sum = 8;

                            if (read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_right][indexj_right] == 2 && read_matrix[indexi_left][indexj_left]) {
                                float prob = 1.0;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }

                            else if ((read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_right][indexj_right] == 2)||((read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_left][indexj_left] == 2))) {
                                float prob = 0.9;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }                    
                            
                            else if (read_matrix[indexi_middle][indexj_middle] == 2) {
                                float prob = 0.7;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }

                            else if ((read_matrix[indexi_left][indexj_left] == 2)||(read_matrix[indexi_right][indexj_right] == 2)) {
                                float prob = 0.65;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }
                                 
                            else
                                write_matrix[y][x] =1;
                        break;

                        case 6: //NO
                            indexi_middle = getToroidal(y+1,d);
                            indexj_middle = getToroidal(x+1,d);
                            indexi_right = getToroidal(y+1,d);
                            indexj_right = getToroidal(x,d);
                            indexi_left = getToroidal(y,d);
                            indexj_left = getToroidal(x+1,d);

                            sum = 8;

                            if (read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_right][indexj_right] == 2 && read_matrix[indexi_left][indexj_left]) {
                                float prob = 1.0;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }

                            else if ((read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_right][indexj_right] == 2)||((read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_left][indexj_left] == 2))) {
                                float prob = 0.9;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }                    
                            
                            else if (read_matrix[indexi_middle][indexj_middle] == 2) {
                                float prob = 0.7;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }

                            else if ((read_matrix[indexi_left][indexj_left] == 2)||(read_matrix[indexi_right][indexj_right] == 2)) {
                                float prob = 0.65;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }
                                 
                            else
                                write_matrix[y][x] =1;
                        break;

                        case 7: //SE 
                            indexi_middle = getToroidal(y-1,d);
                            indexj_middle = getToroidal(x-1,d);
                            indexi_right = getToroidal(y-1,d);
                            indexj_right = getToroidal(x,d);
                            indexi_left = getToroidal(y,d);
                            indexj_left = getToroidal(x-1,d);

                            sum = 8;

                            if (read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_right][indexj_right] == 2 && read_matrix[indexi_left][indexj_left]) {
                                float prob = 1.0;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }

                            else if ((read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_right][indexj_right] == 2)||((read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_left][indexj_left] == 2))) {
                                float prob = 0.9;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }                    
                            
                            else if (read_matrix[indexi_middle][indexj_middle] == 2) {
                                float prob = 0.7;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }

                            else if ((read_matrix[indexi_left][indexj_left] == 2)||(read_matrix[indexi_right][indexj_right] == 2)) {
                                float prob = 0.65;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }
                                 
                            else
                                write_matrix[y][x] =1;
                        break;
                        
                        case 8: //NE
                            indexi_middle = getToroidal(y-1,d);
                            indexj_middle = getToroidal(x+1,d);
                            indexi_right = getToroidal(y,d);
                            indexj_right = getToroidal(x+1,d);
                            indexi_left = getToroidal(y-1,d);
                            indexj_left = getToroidal(x,d);

                            sum = 8;

                            if (read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_right][indexj_right] == 2 && read_matrix[indexi_left][indexj_left]) {
                                float prob = 1.0;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }

                            else if ((read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_right][indexj_right] == 2)||((read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_left][indexj_left] == 2))) {
                                float prob = 0.9;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }                    
                            
                            else if (read_matrix[indexi_middle][indexj_middle] == 2) {
                                float prob = 0.7;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }

                            else if ((read_matrix[indexi_left][indexj_left] == 2)||(read_matrix[indexi_right][indexj_right] == 2)) {
                                float prob = 0.65;
                                std::binomial_distribution<int> distribution_BurnableToBurningN(1,prob);
                                int new_state_BurnableToBurning = distribution_BurnableToBurningN(generator_BurnableToBurning); //dona 0 o 1
                                write_matrix[y][x] = new_state_BurnableToBurning+1;
                            }
                                 
                            else
                                write_matrix[y][x] =1;
                        break;
                    }
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
    // int vent = 0;
	transiction_function();
	swap();
	step++;
}

void initForest()
{
    std::default_random_engine generator_treerock;
	std::binomial_distribution<int> distribution_treerock(1,0.95); 
    
// This function generates the forest (grid) and assigns each cell one of the two possible states: rock (not burnable) or tree (burnable)
	for (int y = 0; y < d; ++y) {
		for (int x = 0; x < d; ++x) {
            std::binomial_distribution<int> distribution_treerock(1,0.95);
            int state = distribution_treerock(generator_treerock); //dona 0 o 1
			// int state = rand()%2; 
            read_matrix[y][x]=state;
			write_matrix[y][x]=state;
            
			// read_matrix[y][x]=1;
			// write_matrix[y][x]=1;
		}
	}
// introduce a burning cell

    for (int i = -1; i <= 1; i++){
        for (int j = -1; j <= 1; j++){
            if (!(i == 0 && j == 0)){
                int indexi = getToroidal(500+i,d);
                int indexj = getToroidal(500+j,d);
                read_matrix[indexi][indexj] = 2;
                write_matrix[indexi][indexj] = 2;
                
            } 
        }
    }
}

// INIT ALLEGRO
void initAllegro(){
	allegro_init();
	install_keyboard();
	install_mouse();
	set_gfx_mode(GFX_AUTODETECT_WINDOWED, b_screen, h_screen, 0 ,0 );
	buffer = create_bitmap(b_screen,h_screen);
	//red = makecol(255,0,0);
	green = makecol(50,205,50); // tree (burnable)
	white = makecol(255,255,255); // rock (not burnable)
	orange = makecol(255,165,0); // burning
	black = makecol(0,0,0); //burned
	grey = makecol(192,192,192);
}

//DRAW
void drawwithAllegro(){
	for (int y = 0; y < d; ++y) {
		for (int x = 0; x < d; ++x) {
			switch(read_matrix[y][x]){
			case 0: // not burnable
				rectfill(buffer,y*zoom,x*zoom,y*zoom+zoom,x*zoom+zoom,grey);
				break;

			case 1: // burnable
				rectfill(buffer,y*zoom,x*zoom,y*zoom+zoom,x*zoom+zoom,green);
				break;

			case 2: //burning
				rectfill(buffer,y*zoom,x*zoom,y*zoom+zoom,x*zoom+zoom,orange);
				break;

			case 3: //burned
				rectfill(buffer,y*zoom,x*zoom,y*zoom+zoom,x*zoom+zoom,black);
			}
		}
	}

	textprintf_ex(buffer, font, 0 ,0, white, black, "step %d ", step);
	blit(buffer,screen, 0, 0,0,0,b_screen, h_screen);
}

int main() {
	initAllegro();
	initForest();
	drawwithAllegro();
	bool pressed_p_button=false;
	int microsec = 100000;

	while(!key[KEY_ESC]){
		
		if(key[KEY_P])
			pressed_p_button=true;

		if(key[KEY_R])
			pressed_p_button=false;

		if(!pressed_p_button)
			global_transiction_function();

		drawwithAllegro();
	}

	return 0;
}
END_OF_MAIN()