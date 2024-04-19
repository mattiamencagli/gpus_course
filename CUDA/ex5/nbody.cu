#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>

//TODO add the necessary MPI headers
#include <cuda.h>
#include <cuda_runtime.h>

#define SOFTENING 1e-9f
#define NTHREADS 256

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        int MyRank;
	    MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
        fprintf(stderr, "GPUassert on rank %d: %s %s %d\n", MyRank, cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// Each body contains x, y, and z coordinate positions, as well as velocities in the x, y, and z directions.
typedef struct { float x, y, z, vx, vy, vz; } Body;
typedef struct { float x, y, z; } floats3;

void read_values_from_file(const char * file, Body * data, size_t size) {
    std::ifstream values(file, std::ios::binary);
    values.read(reinterpret_cast<char*>(data), size);
    values.close();
}

void write_values_to_file(const char * file, Body * data, size_t size) {
    std::ofstream values(file, std::ios::binary);
    values.write(reinterpret_cast<char*>(data), size);
    values.close();
}

void check_correctness(const char * file_out, const char * file_sol, size_t size, size_t nBodies){

    Body *out = (Body *)malloc(size);
    Body *sol = (Body *)malloc(size);
    std::ifstream values_output(file_out, std::ios::binary);
    std::ifstream values_solution(file_sol, std::ios::binary);
    values_output.read(reinterpret_cast<char*>(out), size);
    values_solution.read(reinterpret_cast<char*>(sol), size);

    for(int i=0; i<nBodies; ++i)
        if(out[i].x != sol[i].x ){
            printf("\n\e[01;31m YOUR OUTPUT IS WRONG!\e[0;37m :(\n\n");
            printf("output body %d    : %f %f %f %f %f %f\n", i, out[i].x, out[i].y, out[i].z, out[i].vx, out[i].vy, out[i].vz);
            printf("solution body %d  : %f %f %f %f %f %f\n", i, sol[i].x, sol[i].y, sol[i].z, sol[i].vx, sol[i].vy, sol[i].vz);
            exit(1);
        }

    values_output.close();
    values_solution.close();

    free(out);
    free(sol);

    printf("\n\e[01;32m YOUR OUTPUT IS CORRECT!\e[0;37m :D\n\n");

}



__inline__ __device__ float warpReduce (float val){
    for(int k=16; k>0; k>>=1) // bit-wise version of k/=2
        val += __shfl_down_sync(0xFFFFFFFF, val, k, 32); 
    return val;
}

__device__ float blockReduce(float val){
    static __shared__ float shared[32]; // Shared mem for 32 partial sums
    int threads_localwarp_id = threadIdx.x % 32; // Lane
    int warp_id = threadIdx.x / 32;
    val = warpReduce(val); // Each warp performs partial reduction
    if (threads_localwarp_id == 0)
        shared[warp_id] = val; // Write reduced value to shared memory
    __syncthreads(); // Wait for all partial reductions
    //read from shared memory
    val = (threadIdx.x < blockDim.x / 32) ? shared[threads_localwarp_id] : 0.0;
    if (warp_id == 0)
        val = warpReduce(val); //Final reduce within first warp
    return val;
}

template<int j_stride>
__global__ void bodyForce(Body * p, floats3 * F, float dt, int n, int j_off, int j_max) {

    // in the case each block has multiple i_stars. So in the case you are using less blocks then bodies.
    const int stride_i = gridDim.x;
    for (int i = blockIdx.x; i < n; i += stride_i) { 
        float Fx = 0.0f; 
        float Fy = 0.0f; 
        float Fz = 0.0f;
        const float pix = p[i].x;
        const float piy = p[i].y;
        const float piz = p[i].z;

        // TODO: fix the loop range for multiple gpus using j_off and j_max
        for (int j = threadIdx.x; j < n; j += j_stride) {
            const float dx = p[j].x - pix;
            const float dy = p[j].y - piy;
            const float dz = p[j].z - piz;
            const float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            const float invDist = rsqrtf(distSqr);
            const float invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3; 
            Fy += dy * invDist3; 
            Fz += dz * invDist3;
        }

        //then we reduce the forces on the first thread of the block
        Fx = blockReduce(Fx);
        Fy = blockReduce(Fy);
        Fz = blockReduce(Fz);
        __syncthreads();
        if(threadIdx.x==0){
            F[i].x = Fx; 
            F[i].y = Fy; 
            F[i].z = Fz;
        }

    }
}

__global__ void integratePosition(Body * p, floats3 * F, float dt, int n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        p[i].vx += F[i].x * dt;
        p[i].vy += F[i].y * dt;
        p[i].vz += F[i].z * dt;
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}


int main(int argc, char** argv) {

    // the operator << is the left bit shift operator. 
    // 1 in binary is still "1", if you shift it by 12 position you add 12 zeros obtaining: "100000000000", that is 2^12=4096.

    int nBodies = 1<<12; //the other choice is 1<<16
    if (argc > 1) 
        nBodies = 1<<atoi(argv[1]);
    int size = nBodies * sizeof(Body);
    int sizeF = nBodies * sizeof(floats3);

    //TODO: initialize MPI
    int MyRank, NumberOfProcessors;

    const char *initialized_values, *output_values, *solution_values;

    if (nBodies == 1<<12) {
        initialized_values = "../files/initialized_4096";
        output_values = "../files/output_4096";
        solution_values = "../files/solution_4096";
    } else if (nBodies == 1<<16) {
        initialized_values = "../files/initialized_65536";
        output_values = "../files/output_65536";
        solution_values = "../files/solution_65536";
    } else {
        printf("ERROR: you must choose 12 or 16 for 4096 or 65536 bodies respectively!\n");
        exit(1);
    }

    cudaEvent_t start, stop, start_i, stop_i, start_writing, stop_writing, start_reading, stop_reading;
    CUDA_SAFE_CALL(cudaSetDevice(MyRank));
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    CUDA_SAFE_CALL(cudaEventCreate(&start_i));
    CUDA_SAFE_CALL(cudaEventCreate(&stop_i));
    CUDA_SAFE_CALL(cudaEventCreate(&start_writing));
    CUDA_SAFE_CALL(cudaEventCreate(&stop_writing));
    CUDA_SAFE_CALL(cudaEventCreate(&start_reading));
    CUDA_SAFE_CALL(cudaEventCreate(&stop_reading));

    Body *bodies;
    floats3 *F;
    //TODO allocate properly the new array of forces!
    CUDA_SAFE_CALL(cudaMallocManaged(&bodies, size));
    CUDA_SAFE_CALL(cudaSetDevice(MyRank));
    CUDA_SAFE_CALL(cudaEventRecord(start_reading));
    CUDA_SAFE_CALL(cudaEventSynchronize(start_reading));
    read_values_from_file(initialized_values, bodies, size);
    CUDA_SAFE_CALL(cudaSetDevice(MyRank));
    CUDA_SAFE_CALL(cudaEventRecord(stop_reading));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop_reading));  

    CUDA_SAFE_CALL(cudaSetDevice(MyRank));
    CUDA_SAFE_CALL(cudaMemPrefetchAsync(bodies, size, MyRank));

    // TODO: compute the offsets for particles (you should have 0 for gpu 0 and nbodies/2 for the second gpu)
    int gpu_offsets = 0;
    // TODO: compute the max j index for particles (you should have nbodies/2 for gpu 0 and nbodies for the second gpu)
    int gpu_j_max = 0;
    printf("rank %d. gpu_offset and j_max : %d %d\n", MyRank, gpu_offsets, gpu_j_max);

    dim3 threads(NTHREADS, 1, 1);
    dim3 blocks(nBodies, 1, 1);

    const float dt = 0.01f;  // Time step
    const int nIters = 10;  // Simulation iterations
    /*
    * This simulation will run for 10 cycles of time, calculating gravitational
    * interaction amongst bodies, and adjusting their positions according to their new velocities.
    */
    CUDA_SAFE_CALL(cudaSetDevice(MyRank));
    CUDA_SAFE_CALL(cudaEventRecord(start));
    CUDA_SAFE_CALL(cudaEventSynchronize(start));
    for (int iter = 0; iter < nIters; iter++) {

        CUDA_SAFE_CALL(cudaSetDevice(MyRank));
        CUDA_SAFE_CALL(cudaEventRecord(start_i));
        CUDA_SAFE_CALL(cudaEventSynchronize(start_i));

        CUDA_SAFE_CALL(cudaSetDevice(MyRank));
        bodyForce<NTHREADS><<<blocks, threads>>>(bodies, F, dt, nBodies, gpu_offsets, gpu_j_max); // compute interbody forces
        CUDA_SAFE_CALL(cudaGetLastError());
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        //TODO: take care of the MPI reduction for the Forces!
        MPI_Allreduce(..., ..., ..., ..., ..., ...);

        CUDA_SAFE_CALL(cudaSetDevice(MyRank));
        integratePosition<<<blocks, threads>>>(bodies, F, dt, nBodies);
        CUDA_SAFE_CALL(cudaGetLastError());
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        CUDA_SAFE_CALL(cudaSetDevice(MyRank));
        CUDA_SAFE_CALL(cudaEventRecord(stop_i));
        CUDA_SAFE_CALL(cudaEventSynchronize(stop_i));
        float time_iter_ms;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&time_iter_ms, start_i, stop_i));
        printf("rank %d.  time iter %d : %f ms\n", MyRank, iter, time_iter_ms);

    }
    CUDA_SAFE_CALL(cudaSetDevice(MyRank));
    CUDA_SAFE_CALL(cudaEventRecord(stop));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    MPI_Barrier(MPI_COMM_WORLD);

    if(!MyRank){
        CUDA_SAFE_CALL(cudaSetDevice(MyRank));
        CUDA_SAFE_CALL(cudaEventRecord(start_writing));
        CUDA_SAFE_CALL(cudaEventSynchronize(start_writing));  
        write_values_to_file(output_values, bodies, size);
        CUDA_SAFE_CALL(cudaSetDevice(MyRank));
        CUDA_SAFE_CALL(cudaEventRecord(stop_writing));
        CUDA_SAFE_CALL(cudaEventSynchronize(stop_writing)); 
    }  
    
    float totalTime_loop_ms, time_writing_ms, time_reading_ms;
    CUDA_SAFE_CALL(cudaSetDevice(MyRank));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&totalTime_loop_ms, start, stop));
    CUDA_SAFE_CALL(cudaSetDevice(MyRank));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&time_reading_ms, start_reading, stop_reading));    
    printf("\nrank %d.  time reading : %f ms\n", MyRank, time_reading_ms);
    if(!MyRank){
        CUDA_SAFE_CALL(cudaSetDevice(MyRank));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&time_writing_ms, start_writing, stop_writing));
        printf("rank %d.  time writing : %f ms\n\n", MyRank, time_writing_ms);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    float avgTime_ms = totalTime_loop_ms / nIters;
    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / (avgTime_ms * 1e-3 * NumberOfProcessors);
    printf("rank %d.  %0.3f Billion Interactions / second\n", MyRank, billionsOfOpsPerSecond);
    printf("rank %d.  TOT time loop : %f ms \n", MyRank, totalTime_loop_ms);

    CUDA_SAFE_CALL(cudaSetDevice(MyRank));
    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));
    CUDA_SAFE_CALL(cudaEventDestroy(start_i));
    CUDA_SAFE_CALL(cudaEventDestroy(stop_i));
    CUDA_SAFE_CALL(cudaEventDestroy(start_writing));
    CUDA_SAFE_CALL(cudaEventDestroy(stop_writing));
    CUDA_SAFE_CALL(cudaEventDestroy(start_reading));
    CUDA_SAFE_CALL(cudaEventDestroy(stop_reading));

    cudaFree(bodies);
    cudaFree(F);

    if(!MyRank){
        check_correctness(output_values, solution_values, size, nBodies);
    }

    MPI_Finalize();

    return 0;

}
