#include <iostream>

__global__ void helloFromGPU() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    std::cout << "Running CUDA test...\n";
    helloFromGPU<<<1, 5>>>();  // Launch 5 threads
    cudaDeviceSynchronize();   // Wait for GPU to finish
    return 0;
}
