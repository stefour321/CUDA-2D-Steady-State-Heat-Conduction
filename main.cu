/*
Author: Stella Fournier
Class: ECE4122 A
Last Date Modified: 11/1/2022

Description:

This is the CUDA program for the thin plates calculation over number of iterations.
It takes in the whole grid size and the number of iterations from the command line and outputs
a csv text file of the new temperatures of the whole grid.

*/
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <string>
#include <stdio.h>
#include <chrono>

using namespace std;

// Iterate thin plates temperature in GPU using CUDA
__global__ void iterateTemp(double *myPlate, double *myNewPlate, int n)
{
    int num = blockIdx.x * blockDim.x + threadIdx.x;
    int row = num % n;
    int col = num / n;
    if (num < (n * n) && (col > 0 && col < n - 1) && (row > 0 && row < n - 1))
    {
        myNewPlate[n * row + col] = 0.25 * (myPlate[n * (row - 1) + col] + myPlate[n * (row + 1) + col] + myPlate[n * row + (col - 1)] + myPlate[n * row + (col + 1)]);
    }
}

// Global Functions
// Initializing the matrix plate temperatures
void initializePlateTemp(double myPlate[], double myNewPlate[], int n)
{
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            myPlate[n * row + col] = 0.0;
            myNewPlate[n * row + col] = 0.0;
            // Edges for 20 Celcius degree
            if (row == 0 || row == n - 1 || col == 0 || col == n - 1)
            {
                myPlate[n * row + col] = 20.0;
                myNewPlate[n * row + col] = 20.0;
            }
            // The 4ft 100 Celcius degree edge on the top thin plate
            if (col > 0.3 * (n - 1) && col < 0.7 * (n - 1) && row == 0)
            {
                myPlate[n * row + col] = 100.0;
                myNewPlate[n * row + col] = 100.0;
            }
        }
    }
}
// Outfile to a csv file
void outFile(double myNewPlate[], int n)
{
    fstream outputFile;
    outputFile.open("finalTemperatures.csv", ios::out | ios::app);

    outputFile << fixed << setprecision(6);

    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            outputFile << fixed << setw(n);
            if (col < n - 1)
            {
                outputFile << myNewPlate[n * row + col] << ",";
            }
            else
            {
                outputFile << myNewPlate[n * row + col];
            }
        }
        outputFile << endl;
    }
    outputFile.close();
}
// Checks if input is valid or not
bool validOrNot(string myString)
{
    if (myString.empty())
    {
        return false;
    }
    if (myString.length() > 1 && myString[0] == '0')
    {
        return false;
    }
    // checks if it is a digit or if it has alphabets
    for (int i = 0; i < myString.length(); i++)
    {
        if (!isdigit(myString[i]) || isalpha(myString[i]))
        {
            return false;
        }
        else if (' ' == myString[i])
        {
            return false;
        }
    }
    return true;
}

int main(int argc, const char *argv[])
{
    // checks if input is the correct format or not
    if (argc < 4 || argc > 5)
    {
        cout << "Invalid parameters, please check your values." << endl;
        return 1;
    }
    // checks if input is valid or not
    if (validOrNot(argv[2]) && validOrNot(argv[4]))
    {
        int dimensions;
        int innerDimensions;
        long iterations;
        innerDimensions = sqrt(stoi(argv[2], nullptr, 10));
        iterations = stoi(argv[4], nullptr, 10);
        dimensions = innerDimensions + 2;
        // Checks if number of dimensions and iterations is good or not
        if (dimensions < 0 || iterations < 0)
        {
            cout << "Invalid parameters, please check your values." << endl;
            return 1;
        }

        // Initialize new and old plates
        int size = (dimensions * dimensions) * sizeof(double);
        double *currentPlate;
        double *newPlate;
        cudaMallocManaged(&currentPlate, size);
        cudaMallocManaged(&newPlate, size);
        initializePlateTemp(currentPlate, newPlate, dimensions);

        // iterate using GPU loop
        int dev = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        int numThreads = deviceProp.maxThreadsPerBlock;
        int blockSize = (((dimensions * dimensions) + numThreads - 1) / numThreads);

        // timer
        float time;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        for (int i = 0; i < iterations; i++)
        {
            iterateTemp<<<blockSize, numThreads>>>(currentPlate, newPlate, dimensions);
            cudaDeviceSynchronize(); // wait for GPU threads to finish
            cudaMemcpy(currentPlate, newPlate, size, cudaMemcpyDeviceToDevice);
        }

        // finishing timer
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        cout << "\nThin plate calculation took " << fixed << setprecision(2) << time << " milliseconds." << endl;

        // output file
        outFile(newPlate, dimensions);

        // free gpu
        cudaFree(currentPlate);
        cudaFree(newPlate);
    }
    else
    {
        cout << "Invalid parameters, please check your values." << endl;
        return 1;
    }

    return 0;
}