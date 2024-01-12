# Simulating wildfires with Cellular automata
We simulate wildfires with cellular automata with C/C++. We include an OpenMP and a CUDA implementation.

## Usage

We have grouped all the serial codes under the target `serial`, all the OpenMP codes under `omp` and the CUDA one in `cuda`.
To compile them, run: 
```
make <target>
```
We have included other targets in the Makefile for easily executing the codes, so if you want to execute the serial code with a GUI, for example, you can run:
```
make run_serial
```
Look in the Makefile for the other targets.

## Contents
### WithGUI

Implementations with graphical visualisation of the evolution of the wildfire. 

+ `forest_serial.cpp`: Serial implementation of the basic model.

+ `forest_serial_wind.cpp`: Serial implementation of the model incorporating constant wind.

+ `forest_omp.cpp`: Parallel implementation of the model using OpenMP.

### noGUI

Implementations without graphical visualisation of the wildfire. 

+ `forest_serial_nogui.cpp`: Serial implementation of the basic model. 

+ `forest_omp_nogui.cpp`: OpenMP implementation of the model.

+ `forest_curand.cu`: CUDA implementation of the model, using cuRAND for the rng.

+ `forest_cuda.sbatch`: Bash script for running the cuda code with slurm's sbatch.

+ `submit-extrae.sh`: Bash script for generating a trace of the OpenMP implementation.

**Note:** There is a Makefile in this directory which is only to be used in the `submit-extrae.sh` script.

### scalability 

+ We include 3 files which are copies of the `nogui` ones but without randomness. 

+ `scalability.sbatch`: Bash script to test the execution time and speedup of the OpenMP implementation with a different number of threads. To be executed with slurm's sbatch.
 
+ `scalability_cuda.sbatch`: Same bash script but for the CUDA implementation with different block size. 


**Note:** There is a Makefile in this directory which is only to be used in the `scalability.sbatch` script.

## Dependencies
+ The codes which include a GUI need the Allegro library to work. We include it in the `libraries` directory for a Linux system. 
+ To compile and execute the CUDA file, access to a system with an NVIDIA GPU and the CUDA library is required.
+ If one intends to run the CUDA file as is expected in the Makefile and the other bash scripts, access to a system with the Slurm workload Manager is required.
