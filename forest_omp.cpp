#include <allegro.h>
//#include <conio.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <random>
#include <omp.h>
//using namespace std;

const int b_screen = 500;
const int h_screen = 500;
const int d = 500;
int zoom=1;
BITMAP* buffer;
int step = 0;

struct cell{
	int i;
	int j;
};

const int neighborhood_size=9;
//cell vicinato[neighborhood_size]; moved to inside the for loop, as it is different for each cell

//STATES -------

int read_matrix[d][d];   // Read  Matrix
int write_matrix[d][d];  // Write Matrix

//ALLEGRO COLOR -----

//int green; //tree
//int white; //not burnable
//int yellow; // burnable
//int orange; //burning	
//int black; //burned

int grey; // Not burnable (rock) 
int green; // burnable (tree)
int orange; // burning
int black; //burned
int white;
//Moore neighbourhood
void create_neighborhood(int i, int j, cell* vicinato){
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
	std::binomial_distribution<int> distribution_BurnableToBurning(1,0.85); //p =0.3 de passar de burnable a burning

	#pragma omp parallel for collapse(2)
	for (int y = 0; y < d; ++y) {
		for (int x = 0; x < d; ++x) {
		cell vicinato[neighborhood_size];
// casos: cada cel·la pot ser burnable (green) o not burnable(white), depen d'una probabilitat --> assignar quan es crea la graella inicial
// 1r cas: burnable --> es mira si té algun veí burning i, en aquest cas, es compara probabilitat de (burnable-->burning) amb un rand
//2n cas: not burnable --> es reescriu igual a la matriu del següent pas de temps (no cal mirar neighbourhood)
			if (read_matrix[y][x] == 0) //(not burnable)
			{
				write_matrix[y][x] = 0; // cell remains not burnable
			}
			else if (read_matrix[y][x] == 1) //burnable)
				{
				create_neighborhood(y,x, vicinato); //agafem les veïnes de la cel·la observadda
				int sum = 0;
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
					// a burning si hi ha més veïnes cremant)
				}
				else
					write_matrix[y][x] =1;
			}
			else if (read_matrix[y][x] == 2) { //burning
				write_matrix[y][x] = 3;
			}		
			}
	}
}

void swap(){
	# pragma omp for collapse(2)
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

// INIT ALLEGRO
void initAllegro(){
	allegro_init();
	install_keyboard();
	install_mouse();
	set_gfx_mode(GFX_AUTODETECT_WINDOWED, b_screen, h_screen, 0 ,0 );
	buffer = create_bitmap(b_screen,h_screen);
	
	green = makecol(50,205,50); // tree (burnable)
	grey = makecol(192, 192, 192); // rock (not burnable)
	orange = makecol(255,165,0); // burning
	black = makecol(0,0,0); //burned
	white = makecol(255,255,255);
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
