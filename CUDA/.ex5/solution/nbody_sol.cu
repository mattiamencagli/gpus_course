#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>

#include <mpi.h>
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
typedef struct { float x, y, z; } floats3;

void read_values_from_file(const char * file, floats3 * datap, floats3 * datav, size_t nBodies) {
    std::ifstream values(file, std::ios::binary);
    size_t size3 = sizeof(floats3);
    for(size_t i=0; i<nBodies; ++i){
        values.read(reinterpret_cast<char*>(datap+i), size3);
        values.read(reinterpret_cast<char*>(datav+i), size3);
        //printf("body %d    : %f %f %f %f %f %f\n", i/3, datap[i].x, datap[i].y, datap[i].z, datav[i].x, datav[i].y, datav[i].z);
    }
    values.close();
}

void write_values_to_file(const char * file, floats3 * datap, floats3 * datav, size_t nBodies) {
    std::ofstream values(file, std::ios::binary);
    size_t size3 = sizeof(floats3);
    for(size_t i=0; i<nBodies; ++i){
        values.write(reinterpret_cast<char*>(datap+i), size3);
        values.write(reinterpret_cast<char*>(datav+i), size3);
    }
    values.close();
}

void check_correctness(const char * file_out, const char * file_sol, size_t size, size_t nBodies){

    floats3 *out_p = (floats3 *)malloc(size);
    floats3 *out_v = (floats3 *)malloc(size);
    floats3 *sol_p = (floats3 *)malloc(size);
    floats3 *sol_v = (floats3 *)malloc(size);
    read_values_from_file(file_out, out_p, out_v, nBodies);
    read_values_from_file(file_sol, sol_p, sol_v, nBodies);

    for(int i=0; i<nBodies; ++i)
        if(out_p[i].x != sol_p[i].x ){
            printf("\n\e[01;31m YOUR OUTPUT IS WRONG!\e[0;37m :(\n\n");
            printf("output body %d    : %f %f %f %f %f %f\n", i, out_p[i].x, out_p[i].y, out_p[i].z, out_v[i].x, out_v[i].y, out_v[i].z);
            printf("solution body %d  : %f %f %f %f %f %f\n", i, sol_p[i].x, sol_p[i].y, sol_p[i].z, sol_v[i].x, sol_v[i].y, sol_v[i].z);
            exit(1);
        }

    free(out_p);
    free(sol_p);
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
__global__ void bodyForce(floats3 * pos, floats3 * vel, float dt, int n, int j_off, int j_max) {

    // in the case each block has multiple i_stars. So in the case you are using less blocks then bodies.
    const int stride_i = gridDim.x;
    for (int i = blockIdx.x; i < n; i += stride_i) { 
        float Fx = 0.0f; 
        float Fy = 0.0f; 
        float Fz = 0.0f;
        const float pix = pos[i].x;
        const float piy = pos[i].y;
        const float piz = pos[i].z;

        // each i-star will have its own block, each thread will take care of a number of interactions (j_max-off)/j_stride
        for (int j = threadIdx.x + j_off; j < j_max; j += j_stride) {
            const float dx = pos[j].x - pix;
            const float dy = pos[j].y - piy;
            const float dz = pos[j].z - piz;
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
        if(threadIdx.x==0){ //only the first thread of each block has the entire block reduction.
            vel[i].x += dt*Fx;
            vel[i].y += dt*Fy;
            vel[i].z += dt*Fz;
        }

    }
}

__global__ void integratePosition(floats3 * pos, floats3 * vel, float dt, int n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        pos[i].x += vel[i].x * dt;
        pos[i].y += vel[i].y * dt;
        pos[i].z += vel[i].z * dt;
        //if(i==0)
            //printf("vel[%d] = %f %f %f\npos[%d] = %f %f %f\n", i, vel[i].x, vel[i].y, vel[i].z, i, pos[i].x, pos[i].y, pos[i].z);
    }
}


int main(int argc, char** argv) {

    // the operator << is the left bit shift operator. 
    // 1 in binary is still "1", if you shift it by 12 position you add 12 zeros obtaining: "100000000000", that is 2^12=4096.

    int nBodies = 1<<12; //the other choice is 1<<16
    if (argc > 1) 
        nBodies = 1<<atoi(argv[1]);
    int size = nBodies * sizeof(floats3);

	//MPI Intialization
    int MyRank, NumberOfProcessors;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
	MPI_Comm_size(MPI_COMM_WORLD, &NumberOfProcessors);
    //Number of GPUs (I culd also use NumberOfProcessors since I assign 1 GPU to each MPI task).
    int Ngpus; 
    CUDA_SAFE_CALL(cudaGetDeviceCount(&Ngpus));

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

    floats3 *p, *v;
    CUDA_SAFE_CALL(cudaMallocManaged(&p, size));
    CUDA_SAFE_CALL(cudaMallocManaged(&v, size));
    CUDA_SAFE_CALL(cudaSetDevice(MyRank));
    CUDA_SAFE_CALL(cudaEventRecord(start_reading));
    CUDA_SAFE_CALL(cudaEventSynchronize(start_reading));
    read_values_from_file(initialized_values, p, v, nBodies);
    CUDA_SAFE_CALL(cudaSetDevice(MyRank));
    CUDA_SAFE_CALL(cudaEventRecord(stop_reading));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop_reading));  

    CUDA_SAFE_CALL(cudaSetDevice(MyRank));
    CUDA_SAFE_CALL(cudaMemPrefetchAsync(p, size, MyRank));
    CUDA_SAFE_CALL(cudaMemPrefetchAsync(v, size, MyRank));

    // Works for this simple example for 2 GPUs. 
    int gpu_offsets = MyRank ? (nBodies/NumberOfProcessors + nBodies*(MyRank-1)) : 0;
    int gpu_j_max = gpu_offsets + nBodies/NumberOfProcessors;
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
        bodyForce<NTHREADS><<<blocks, threads>>>(p, v, dt, nBodies, gpu_offsets, gpu_j_max); // compute interbody forces
        CUDA_SAFE_CALL(cudaGetLastError());
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        //MPI reduction in place on the bodies' velocities (possible because all the bodies have the same dt)
        if(NumberOfProcessors > 1)
            MPI_Allreduce(MPI_IN_PLACE, v, nBodies*3, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        CUDA_SAFE_CALL(cudaSetDevice(MyRank));
        integratePosition<<<blocks, threads>>>(p, v, dt, nBodies);
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
        write_values_to_file(output_values, p, v, nBodies);
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

    cudaFree(p);
    cudaFree(v);

    if(!MyRank){
        check_correctness(output_values, solution_values, size, nBodies);
    }

    MPI_Finalize();

    return 0;

}
