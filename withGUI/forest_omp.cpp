// C libraries
#include <allegro.h>
#include <omp.h>

// C++ libraries
#include <random>
#include <ctime>

// Dimension of the grid
const int d = 500;

// Allegro parameters
const int b_screen = 500;
const int h_screen = 500;
int zoom=1;
BITMAP* buffer;

// Allegro colors
int grey; // Not burnable (rock) 
int green; // burnable (tree)
int orange; // burning
int black; //burned
int white;

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

// Function for periodic boundary conditions
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


// Transition function for one timestep
void transition_function(int read_matrix[d][d], int write_matrix[d][d], ThreadSafeRNG rng){
	int sum;
	float p = 0.8; // minimum value so that the model evolvedd more than a few timesteps
	#pragma omp for schedule(dynamic)
	for (int y = 0; y < d; ++y) {
		for (int x = 0; x < d; ++x) {
			switch(read_matrix[y][x]){ 
				case 0: //(not burnable)
					write_matrix[y][x] = 0; // cell remains not burnable
					break;
				case 1: 
					sum = 0; //counter of burning neighbours
					for (int i = -1; i <= 1; i++){
						for (int j = -1; j <= 1; j++){
							if (!(i == 0 && j == 0)){
								int indexi = getToroidal(y+i,d);
								int indexj = getToroidal(x+j,d);
								if (read_matrix[indexi][indexj] == 2)
									sum += 1;
							} 
						}
					}				
			      
					if(sum>0){ 
						float prob = (-p+1.0)/7.0*sum + (8.0*p-1.0)/7.0;
						write_matrix[y][x] = rng.getBinNumber(p)+1;
					}
					else
						write_matrix[y][x] =1;
			
					break;
			
				case 2:
					write_matrix[y][x] = 3;
					break;
				// case 3:
					// write_matrix[y][x] = 3;
					// break;
			}		
		}
	}
}

void swap(int read_matrix[d][d], int write_matrix[d][d]){
	#pragma omp for
	for (int y = 0; y < d; ++y) {
		for (int x = 0; x < d; ++x) {
			read_matrix[y][x] = write_matrix[y][x];
		}
	}
}


void initForest(int read_matrix[d][d], int write_matrix[d][d])
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
	read_matrix[d/2][d/2] = 2;
	write_matrix[d/2][d/2] = 2;
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
void drawwithAllegro(int read_matrix[d][d], int step){
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
	// States matrix
	int read_matrix[d][d];   // Read  Matrix
	int write_matrix[d][d];  // Write Matrix
	
	int step = 0;
	
	initAllegro();
	initForest(read_matrix, write_matrix);
	drawwithAllegro(read_matrix, step);
	
	bool pressed_p_button=false;
	int microsec = 100000;

	#pragma omp parallel
	{
		ThreadSafeRNG rng;
		while(!key[KEY_ESC]){
			if(key[KEY_P])
				pressed_p_button=true;
				
				if(key[KEY_R])
					pressed_p_button=false;
				
				if(!pressed_p_button){
					transition_function(read_matrix, write_matrix, rng);
					swap(read_matrix, write_matrix);
					step++;
				
					#pragma omp critical
					drawwithAllegro(read_matrix, step);
				}
		}
	}
	return 0;
}
END_OF_MAIN()
