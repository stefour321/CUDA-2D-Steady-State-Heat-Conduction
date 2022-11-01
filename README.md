# CUDA-2D-Steady-State-Heat-Conduction
This program uses CUDA to determine the steady state heat distribution 
in a thin metal plate using synchronous iteration on a GPU. Using the finite difference method
to solve Laplace's equation.

## Thin Plate Set Up
A perfectly insulted thin plate with the sides held at 20 °C and a short segment on one side is 
held at 100 °C is shown below<br /><br />
![image](https://user-images.githubusercontent.com/117101758/199339632-d869810c-33bd-41fb-b5d3-4838cdaf9d87.png)

## Command Line Arguments
`./main -n 256 -I 10000`
1. -n 255 - the number of interior points.  
2. -I 10000 – the number of iterations 

## Run CUDA on Windows
`module load gcc/9.2.0 cuda/11.1`
`nvcc main.cu`
`./a.out`


