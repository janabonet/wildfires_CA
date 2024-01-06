#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <stdio.h>

#define OUTPUT_PATH_ID 1
#define STEPS_ID 2
#define MATRIX_SIZE 3

#define STRLEN 256

  bool saveGrid2Dr(int *M, int d, char *path){
		FILE *f;

		// Check if file exists and delete it if so
		f = fopen(path,"a");
		if (!f)
			return false;
		char str[STRLEN];
		for (int i = 0; i < d; i++){
			for (int j = 0; j < d; j++){
				sprintf(str,"%d ",M[i*d+j]);
				fprintf(f,"%s", str);
			}
		}
		fclose(f);

	return true;
}

int main(int argc, char **argv){
	srand(1);
	int d = atoi(argv[MATRIX_SIZE]);
	int size = d*d*sizeof(int);
	int total_steps = atoi(argv[STEPS_ID]);

	int *seed_array;
	seed_array = (int *)malloc(size);

	// Clear file if it exists
	FILE *f;	
	f = fopen(argv[OUTPUT_PATH_ID],"w");
	
	for (int step = 0; step < total_steps; step++){
		printf("STEP %d\n",step);
		for (int i = 0; i < d*d; i++){
			seed_array[i] = rand()%100;
		}
			saveGrid2Dr(seed_array, d, argv[OUTPUT_PATH_ID]);
	}
}
