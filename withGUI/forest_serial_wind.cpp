// C libraries
#include <allegro.h>
// C++ libraries
#include <random>
#include <ctime>

const int d = 1000;
// int step = 0;

const int b_screen = 1000;
const int h_screen = 1000;
int zoom=1;
BITMAP* buffer;

// int read_matrix[d][d];   // Read  Matrix
// int write_matrix[d][d];  // Write Matrix

//ALLEGRO COLOR -----

int green; //tree
int white; //not burnable
int yellow; // burnable
int orange; //burning	
int black; //burned
int grey;

// Class for clearer syntax of random numbers
class SerialRNG{
    public:
        SerialRNG(){
            generator.seed(time(NULL));
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

// Funtion that defines transition behaviour if there is wind present (used in transition_function)
void transition_with_wind(int x, int y, int indexi_left,int indexj_left, int indexi_right, int indexj_right, int indexi_middle, int indexj_middle,
                            int read_matrix[d][d], int write_matrix[d][d], SerialRNG changeState){
    if (read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_right][indexj_right] == 2 && read_matrix[indexi_left][indexj_left] == 2) //three relevant neighbours are burning
        write_matrix[y][x] = changeState.getBinNumber(0.95)+1; //0.95

    else if ((read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_right][indexj_right] == 2)||((read_matrix[indexi_middle][indexj_middle] == 2 && read_matrix[indexi_left][indexj_left] == 2))) //middle and one corner neighbours are burning
        write_matrix[y][x] = changeState.getBinNumber(0.9)+1;     // 0.9               

    else if (read_matrix[indexi_middle][indexj_middle] == 2) //middle neighbour is burning
        write_matrix[y][x] = changeState.getBinNumber(0.7)+1; // 0.7

    else if ((read_matrix[indexi_left][indexj_left] == 2) ||(read_matrix[indexi_right][indexj_right] == 2)) // one of the corner neighbours is burning
        write_matrix[y][x] = changeState.getBinNumber(0.65)+1; //0.65

    else
        write_matrix[y][x] =1;
}

// Transition function for one timestep
void transition_function(int read_matrix[d][d], int write_matrix[d][d], SerialRNG changeState){
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
                    switch(vent) {
                        case 0: //no wind
                            sum = 0;
                            for (int i = -1; i <= 1; i++){
                                for (int j = -1; j <= 1; j++){
                                    if (!(i == 0 && j == 0)){
                                        indexi = getToroidal(y+i,d);
                                        indexj = getToroidal(x+j,d);
                                        if (read_matrix[indexi][indexj] == 2) // check if the neighbours are burning and count them
                                            sum += 1;
                                    } 
                                }
                            }
                            
                            if(sum>0){ 
                                float p = 0.8;
                                float prob = (-p+1.0)/7.0*sum + (8.0*p-1.0)/7.0; 
                                write_matrix[y][x] = changeState.getBinNumber(prob) +1;
                            }
                            else
                                write_matrix[y][x] =1;
                        break;
                        case 1: //South wind

                        // Define the indices of the relevant neighbours
                            indexi_middle = getToroidal(y,d);
                            indexj_middle = getToroidal(x+1,d);
                            indexi_right = getToroidal(y+1,d);
                            indexj_right = getToroidal(x+1,d);
                            indexi_left = getToroidal(y-1,d);
                            indexj_left = getToroidal(x+1,d);

                            transition_with_wind(x, y, indexi_left, indexj_left, indexi_right, indexj_right, indexi_middle, indexj_middle, read_matrix, write_matrix, changeState);
                        break;

                        case 2: // North wind
                            indexi_middle = getToroidal(y,d);
                            indexj_middle = getToroidal(x-1,d);
                            indexi_right = getToroidal(y+1,d);
                            indexj_right = getToroidal(x+1,d);
                            indexi_left = getToroidal(y-1,d);
                            indexj_left = getToroidal(x-1,d);

                            transition_with_wind(x, y, indexi_left, indexj_left, indexi_right, indexj_right, indexi_middle, indexj_middle, read_matrix, write_matrix, changeState);
                        break;

                        case 3: // East wind
                            indexi_middle = getToroidal(y+1,d);
                            indexj_middle = getToroidal(x,d);
                            indexi_right = getToroidal(y+1,d);
                            indexj_right = getToroidal(x-1,d);
                            indexi_left = getToroidal(y+1,d);
                            indexj_left = getToroidal(x+1,d);

                            transition_with_wind(x, y, indexi_left, indexj_left, indexi_right, indexj_right, indexi_middle, indexj_middle, read_matrix, write_matrix, changeState);
                        break;

                        case 4: // West wind
                            indexi_middle = getToroidal(y-1,d);
                            indexj_middle = getToroidal(x,d);
                            indexi_right = getToroidal(y-1,d);
                            indexj_right = getToroidal(x+1,d);
                            indexi_left = getToroidal(y-1,d);
                            indexj_left = getToroidal(x-1,d);

                            transition_with_wind(x, y, indexi_left, indexj_left, indexi_right, indexj_right, indexi_middle, indexj_middle, read_matrix, write_matrix, changeState);
                        break;

                        case 5: // Northeast wind
                            indexi_middle = getToroidal(y+1,d);
                            indexj_middle = getToroidal(x-1,d);
                            indexi_right = getToroidal(y,d);
                            indexj_right = getToroidal(x-1,d);
                            indexi_left = getToroidal(y+1,d);
                            indexj_left = getToroidal(x,d);

                            transition_with_wind(x, y, indexi_left, indexj_left, indexi_right, indexj_right, indexi_middle, indexj_middle, read_matrix, write_matrix, changeState);
                        break;

                        case 6: //Southeast wind
                            indexi_middle = getToroidal(y+1,d);
                            indexj_middle = getToroidal(x+1,d);
                            indexi_right = getToroidal(y+1,d);
                            indexj_right = getToroidal(x,d);
                            indexi_left = getToroidal(y,d);
                            indexj_left = getToroidal(x+1,d);

                            transition_with_wind(x, y, indexi_left, indexj_left, indexi_right, indexj_right, indexi_middle, indexj_middle, read_matrix, write_matrix, changeState);
                        break;

                        case 7: //NorthWest wind 
                            indexi_middle = getToroidal(y-1,d);
                            indexj_middle = getToroidal(x-1,d);
                            indexi_right = getToroidal(y-1,d);
                            indexj_right = getToroidal(x,d);
                            indexi_left = getToroidal(y,d);
                            indexj_left = getToroidal(x-1,d);

                            transition_with_wind(x, y, indexi_left, indexj_left, indexi_right, indexj_right, indexi_middle, indexj_middle, read_matrix, write_matrix, changeState);
                        break;
                        
                        case 8: //Southwest wind
                            indexi_middle = getToroidal(y-1,d);
                            indexj_middle = getToroidal(x+1,d);
                            indexi_right = getToroidal(y,d);
                            indexj_right = getToroidal(x+1,d);
                            indexi_left = getToroidal(y-1,d);
                            indexj_left = getToroidal(x,d);

                            transition_with_wind(x, y, indexi_left, indexj_left, indexi_right, indexj_right, indexi_middle, indexj_middle, read_matrix, write_matrix, changeState);
                        break;
                    }
                    break;
                case 2: // burning
                    write_matrix[y][x] = 3;
                    break;
            }
		}
	}
}

// Copies write matrix to read matrix
void swap(int read_matrix[d][d], int write_matrix[d][d]){
	for (int y = 0; y < d; ++y) {
		for (int x = 0; x < d; ++x) {
			read_matrix[y][x] = write_matrix[y][x];
		}
	}
}

// Initial conditions
void initForest(int read_matrix[d][d], int write_matrix[d][d], SerialRNG treeRock){
  // This function generates the forest (grid) and assigns each cell one of the two possible states: rock (not burnable) or tree (burnable)
	for (int y = 0; y < d; ++y) {
	    for (int x = 0; x < d; ++x) {
        int state = treeRock.getBinNumber(0.8);
        read_matrix[y][x] = state;
		write_matrix[y][x] = state;            
		}
	}
  // introduce a square of burning cells
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

// Initializes Allegro
void initAllegro(){
	allegro_init();
	install_keyboard();
	install_mouse();
	set_gfx_mode(GFX_AUTODETECT_WINDOWED, b_screen, h_screen, 0 ,0 );
	buffer = create_bitmap(b_screen,h_screen);

	green = makecol(50,205,50); // tree (burnable)
	white = makecol(255,255,255); // rock (not burnable)
	orange = makecol(255,165,0); // burning
	black = makecol(0,0,0); //burned
	grey = makecol(192,192,192);
}

// Function to draw grid
void drawwithAllegro(int read_matrix[d][d],int step){
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


int main(){
  // States matrices
  int read_matrix[d][d];
  int write_matrix[d][d];

  int step = 0;

    // Initialize rng seeds
    SerialRNG treeRock;
    SerialRNG changeState;

  // Initialize allegro
	initAllegro();
	initForest(read_matrix, write_matrix, treeRock);
	drawwithAllegro(read_matrix, step);

	bool pressed_p_button=false;
	int microsec = 100000;


	while(!key[KEY_ESC]){
		
		if(key[KEY_P])
			pressed_p_button=true;

		if(key[KEY_R])
			pressed_p_button=false;

		if(!pressed_p_button){
    	transition_function(read_matrix, write_matrix, changeState);
    	swap(read_matrix, write_matrix);
    	step++;
	  	drawwithAllegro(read_matrix, step);
	  }
  }

	return 0;
}
END_OF_MAIN()
