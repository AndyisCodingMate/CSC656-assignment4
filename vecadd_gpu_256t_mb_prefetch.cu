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
   cudaMallocHost((void**)&x, N * sizeof(float));
   cudaMallocHost((void**)&y, N * sizeof(float));
    // initialize x and y on the CPU
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int size = N * sizeof(float);
    float * d_x, * d_y;
    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_y, size);
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
    
    int numBlocks = (N + THREADSPerBLOCK - 1) / THREADSPerBLOCK;
    int deviceID = 0;
    cudaMemPrefetchAsync((void *)x, N*sizeof(float), deviceID) ;
    cudaMemPrefetchAsync((void *)y, N*sizeof(float), deviceID) ;
    
    // Run Kernel on 512M elements on the GPU
    add<<<numBlocks,THREADSPerBLOCK>>>(N, d_x, d_y);
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_x); cudaFree(d_y);
    delete [] x; delete [] y;
    return 0;
}