#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

//TODO add the necessary CUDA headers

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

//TODO copy the cpu version of the function and translate it into CUDA

void cpu_matmul(double *a, double *b, double *c, const int &N) {
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += a[k + j * N] * b[i + k * N];
            c[i + j * N] = sum;
        }
}


int main(int argc, char **argv){

    size_t N = 1e4;
    if (argc > 1)
        N = atoi(argv[1]);
    size_t N2 = N * N;
    size_t size = N2 * sizeof(double);

    printf("Total required memory: %1.3f GBytes\n", 3.0 * double(size) * 1e-9);

    //TODO allocate memory in host RAM using the CUDA function
    double *h_a, *h_b, *h_c;
    h_a = (double *)malloc(size);
    h_b = (double *)malloc(size);
    h_c = (double *)malloc(size);

    for (size_t i = 0; i < N2; ++i){
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
    
    //TODO Allocate memory space on the device 

    //TODO copy NEEDED matrices from host to device memory

    CUDA_SAFE_CALL(cudaEventRecord(start_c, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(start_c));

    //TODO Use the CUDA function, remember to define blocks and threads.
    cpu_matmul(h_a, h_b, h_c, N);

    CUDA_SAFE_CALL(cudaEventRecord(stop_c, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop_c));

    //TODO Transfer results from device to host 

    // time counting terminate
    CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));

    // compute time elapse on GPU computing
    float gpu_elapsed_time_ms, gpu_comp_elapsed_time_ms;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&gpu_comp_elapsed_time_ms, start_c, stop_c));
    printf("Time: %f ms.\n", gpu_elapsed_time_ms);
    printf("Time computation: %f ms.\n", gpu_comp_elapsed_time_ms);
    printf("print the first value of c (should be %d if you do not change the values within the vectors) : \n %1.1f\n", N*2 , h_c[0]);
    
    //TODO free memory (both host and device) with the CUDA function
    CUDA_SAFE_CALL(cudaFree(d_a));
    CUDA_SAFE_CALL(cudaFree(d_b));
    CUDA_SAFE_CALL(cudaFree(d_c));
    CUDA_SAFE_CALL(cudaFreeHost(h_a));
    CUDA_SAFE_CALL(cudaFreeHost(h_b));
    CUDA_SAFE_CALL(cudaFreeHost(h_c));

    return 0;

}
