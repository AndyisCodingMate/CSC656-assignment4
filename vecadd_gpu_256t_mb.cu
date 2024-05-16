#include <iostream>
#include <chrono>
#include <unistd.h>
#include <cuda_runtime.h>

// Run kernel on GPU
#define THREADSPerBLOCK 256
int numBlocks = (N + THREADSPerBLOCK - 1) / THREADSPerBLOCK;

__global__ void add (int n, float *x, float *y) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (index < n) {
        y[i] = x[i] + y[i];
    }
}
int main(void){
   int N =  1<<29; 
   float *x, *y;
   
   int size = N * sizeof(float);
   cudaMallocManaged((void**)&x, size);
   cudaMallocManaged((void**)&y, size);

    // initialize x and y on the CPU
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int numBlocks = (N + THREADSPerBLOCK - 1) / THREADSPerBLOCK;
    printf("numBlocks: %d\n", numBlocks);
    
    // Run on 512M elements on the GPU
    add<<<numBlocks,THREADSPerBLOCK>>>(N, x, y);
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x); cudaFree(y);
    return 0;
}

