#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

__global__ void gpu_vec_add(double *a, double *b, double *c, int N) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) 
        c[i] = a[i] + b[i];
} 

void cpu_vec_add(double *a, double *b, double *c, const int &N) {
    for (int i = 0; i < N; ++i)
        c[i] = a[i] + b[i];
}


int main(int argc, char **argv){

    size_t N = 1e8;
    if (argc > 1)
        N = atoi(argv[1]);
    size_t size = N * sizeof(double);

    printf("Total required memory: %1.3f GBytes\n", 3 * size * 1e-9);

    // allocate memory in host RAM
    double *h_a, *h_b, *h_c;
    CUDA_SAFE_CALL(cudaMallocHost(&h_a, size));
    CUDA_SAFE_CALL(cudaMallocHost(&h_b, size));
    CUDA_SAFE_CALL(cudaMallocHost(&h_c, size));

    for (size_t i = 0; i < N; ++i){
        h_a[i] = 1.0;
        h_b[i] = 2.0;
    }

    // CUDA events to count the execution time
    cudaEvent_t start, stop, start_c, stop_c;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    CUDA_SAFE_CALL(cudaEventCreate(&start_c));
    CUDA_SAFE_CALL(cudaEventCreate(&stop_c));

    // start to count execution time
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(start));

    // Allocate memory space on the device 
    double *d_a, *d_b, *d_c;
    CUDA_SAFE_CALL(cudaMalloc(&d_a, size));
    CUDA_SAFE_CALL(cudaMalloc(&d_b, size));
    CUDA_SAFE_CALL(cudaMalloc(&d_c, size));

    // copy matrix A and B from host to device memory
    CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaEventRecord(start_c, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(start_c));

    dim3 threads(128, 1, 1);
    dim3 blocks((N - 1) / threads.x + 1, 1, 1);
    gpu_vec_add<<<blocks, threads>>>(d_a, d_b, d_c, N);

    CUDA_SAFE_CALL(cudaEventRecord(stop_c, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop_c));

    // Transefr results from device to host 
    CUDA_SAFE_CALL(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // time counting terminate
    CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));

    // compute time elapse on GPU computing
    float gpu_elapsed_time_ms, gpu_comp_elapsed_time_ms;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&gpu_comp_elapsed_time_ms, start_c, stop_c));
    printf("Time: %f ms.\n", gpu_elapsed_time_ms);
    printf("Time computation: %f ms.\n", gpu_comp_elapsed_time_ms);
    printf("print the first value of c (should be 3.0 if you do not change the values within the vectors) : \n %1.1f\n", h_c[0]);
    
    // free memory
    CUDA_SAFE_CALL(cudaFree(d_a));
    CUDA_SAFE_CALL(cudaFree(d_b));
    CUDA_SAFE_CALL(cudaFree(d_c));
    CUDA_SAFE_CALL(cudaFreeHost(h_a));
    CUDA_SAFE_CALL(cudaFreeHost(h_b));
    CUDA_SAFE_CALL(cudaFreeHost(h_c));

    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));
    CUDA_SAFE_CALL(cudaEventDestroy(start_c));
    CUDA_SAFE_CALL(cudaEventDestroy(stop_c));

    return 0;

}
