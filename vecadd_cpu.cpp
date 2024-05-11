#include <iostream>
#include <chrono>
#include <unistd.h>

void add (int n, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}
int main(void){
   int N =  1<<29; 
   float *x = new float[N];
   float *y = new float[N];
    // initialize x and y on the CPU
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
    // Run on 512M elements on the CPU
    add(N, x, y);
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << " Elapsed time is : " << elapsed.count() << " " << std::endl;
    // Free memory
    delete [] x;
    delete [] y;
    return 0;
}