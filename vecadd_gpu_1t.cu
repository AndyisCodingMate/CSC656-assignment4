#include <iostream>
#include <chrono>
#include <unistd.h>
#include <cuda_runtime.h>


__global__ void add (int n, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}
int main(int ac, char *av[]){
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

    // Run on 512M elements on the GPU
    add<<<1,1>>>(N, x, y);
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