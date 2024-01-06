#include <allegro.h>
#include <stdlib.h>
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

void transiction_function(std::default_random_engine generator_BurnableToBurning){
	// std::binomial_distribution<int> distribution_BurnableToBurning(1,0.85); //p =0.3 de passar de burnable a burning

#pragma omp parallel
	{
	int sum;
	int i, j;		
	int new_state_BurnableToBurning;	
	#pragma omp for private(sum,i,j,new_state_BurnableToBurning,generator_BurnableToBurning)
	for (int y = 0; y < d; ++y) {
		for (int x = 0; x < d; ++x) {
			if (read_matrix[y][x] == 0) //(not burnable)
			{
				write_matrix[y][x] = 0; // cell remains not burnable
			}
			else if (read_matrix[y][x] == 1) //burnable)
				{
					sum = 0;
						for (i = -1; i <= 1; i++){
						for (j = -1; j <= 1; j++){
							if (!(i == 0 && j == 0)){
								int indexi = getToroidal(y+i,d);
								int indexj = getToroidal(x+j,d);
								if (read_matrix[indexi][indexj] == 2)
									sum += 1;
							} 
						}
					}				
			      
				
				if(sum>0){ 
					float p = 0.8;
					float prob = (-p+1.0)/7.0*sum + (8.0*p-1.0)/7.0;
					std::binomial_distribution<int> distribution_BurnableToBurning(1,prob);
					new_state_BurnableToBurning = distribution_BurnableToBurning(generator_BurnableToBurning);
					write_matrix[y][x] = new_state_BurnableToBurning+1;
					}
				else
					write_matrix[y][x] =1;
			}
			else if (read_matrix[y][x] == 2) {
				write_matrix[y][x] = 3;
			}		
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

void global_transiction_function(std::default_random_engine generator_BurnableToBurning){
	transiction_function(generator_BurnableToBurning);
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
	srand(1);
	std::default_random_engine generator_BurnableToBurning;
	generator_BurnableToBurning.seed(1);
	
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
			global_transiction_function(generator_BurnableToBurning);

		drawwithAllegro();
	}

	return 0;
}
END_OF_MAIN()
