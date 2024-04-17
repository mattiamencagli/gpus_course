#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>

//TODO add the necessary CUDA headers

#define SOFTENING 1e-9f

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// Each body contains x, y, and z coordinate positions, as well as velocities in the x, y, and z directions.
typedef struct { float x, y, z, vx, vy, vz; } Body;

void read_values_from_file(const char * file, Body * data, size_t size) {
    std::ifstream values(file, std::ios::binary);
    if(values.fail()){
        printf("fail trying to open the file: %s\n", file);
        exit(1);
    }
    values.read(reinterpret_cast<char*>(data), size);
    values.close();
}

void write_values_to_file(const char * file, Body * data, size_t size) {
    std::ofstream values(file, std::ios::binary);
    if(values.fail()){
        printf("fail trying to open the file: %s\n", file);
        exit(1);
    }
    values.write(reinterpret_cast<char*>(data), size);
    values.close();
}

void check_correctness(const char * file_out, const char * file_sol, size_t size, size_t nBodies){

    Body *out = (Body *)malloc(size);
    Body *sol = (Body *)malloc(size);
    std::ifstream values_output(file_out, std::ios::binary);
    std::ifstream values_solution(file_sol, std::ios::binary);    
    if(values_output.fail()){
        printf("fail trying to open the file: %s\n", file_out);
        exit(1);
    } else if(values_output.fail()) {
        printf("fail trying to open the file: %s\n", file_sol);
        exit(1);
    }
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



//TODO copy the cpu version of the function and translate it into CUDA

void bodyForce(Body *p, float dt, int n) {
    for (int i = 0; i < n; ++i) {
        float Fx = 0.0f; 
        float Fy = 0.0f; 
        float Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3; 
            Fy += dy * invDist3; 
            Fz += dz * invDist3;
        }

        p[i].vx += dt*Fx; 
        p[i].vy += dt*Fy; 
        p[i].vz += dt*Fz;
    }
}


//TODO copy the cpu version of the function and translate it into CUDA

void integratePosition(Body* p, float dt, int n) {

    for (int i = 0 ; i < n; i++) {
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }

}


int main(int argc, char** argv) {

    // the operator << is the left bit shift operator. 
    // 1 in binary is still "1", if you shift it by 12 position you add 12 zeros obtaining: "100000000000", that is 2^12=4096.

    int nBodies = 1<<12; //the other choice is 1<<16
    if (argc > 1) 
        nBodies = 1<<atoi(argv[1]);
    int size = nBodies * sizeof(Body);

    const char *initialized_values, *output_values, *solution_values;

    if (nBodies == 1<<12) {
        initialized_values = "files/initialized_4096";
        output_values = "files/output_4096";
        solution_values = "files/solution_4096";
    } else if (nBodies == 1<<16) {
        initialized_values = "files/initialized_65536";
        output_values = "files/output_65536";
        solution_values = "files/solution_65536";
    } else {
        printf("ERROR: you must choose 12 or 16 for 4096 or 65536 bodies respectively!\n");
        exit(1);
    }

    cudaEvent_t start, stop, start_i, stop_i, start_writing, stop_writing, start_reading, stop_reading;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    CUDA_SAFE_CALL(cudaEventCreate(&start_i));
    CUDA_SAFE_CALL(cudaEventCreate(&stop_i));
    CUDA_SAFE_CALL(cudaEventCreate(&start_writing));
    CUDA_SAFE_CALL(cudaEventCreate(&stop_writing));
    CUDA_SAFE_CALL(cudaEventCreate(&start_reading));
    CUDA_SAFE_CALL(cudaEventCreate(&stop_reading));


    //TODO get the device Id and th number of Streaminbg multiprocessors

    //TODO Allocate memory space on the device using the managed memory (or not, if you prefer to menage the memory by yourself)
    Body *bodies;
    bodies = (Body *)malloc(size);

    CUDA_SAFE_CALL(cudaEventRecord(start_reading, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(start_reading));   
    read_values_from_file(initialized_values, bodies, size);
    CUDA_SAFE_CALL(cudaEventRecord(stop_reading, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop_reading));   

    //TODO Prefetch the memory to optimize data transfer

    //TODO Define blocks and threads.

    const float dt = 0.01f;  // Time step
    const int nIters = 10;  // Simulation iterations
    /*
    * This simulation will run for 10 cycles of time, calculating gravitational
    * interaction amongst bodies, and adjusting their positions according to their new velocities.
    */
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(start));
    for (int iter = 0; iter < nIters; iter++) {

        CUDA_SAFE_CALL(cudaEventRecord(start_i, 0));
        CUDA_SAFE_CALL(cudaEventSynchronize(start_i));

        //TODO Use the CUDA function
        bodyForce(bodies, dt, nBodies);
        
        //TODO remeber to wait for the GPU to finish before going to the next kernel.

        //TODO Use the CUDA function
        integratePosition(bodies, dt, nBodies);

        //TODO remeber to wait for the GPU to finish before going to the next kernel.
   
        CUDA_SAFE_CALL(cudaEventRecord(stop_i, 0));
        CUDA_SAFE_CALL(cudaEventSynchronize(stop_i));

        float time_iter_ms;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&time_iter_ms, start_i, stop_i));
        printf("time iter %d iteration : %f ms\n", iter, time_iter_ms);

    }
    CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));   
    
    //TODO If you DO NOT use the managed memory, remeber to transfer the results back on the host
    CUDA_SAFE_CALL(cudaEventRecord(start_writing, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(start_writing));   
    write_values_to_file(output_values, bodies, size);
    CUDA_SAFE_CALL(cudaEventRecord(stop_writing, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop_writing));   
    
    float totalTime_loop_ms, time_writing_ms, time_reading_ms;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&totalTime_loop_ms, start, stop));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&time_writing_ms, start_writing, stop_writing));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&time_reading_ms, start_reading, stop_reading));

    printf("\ntime reading : %f ms\n", time_reading_ms);
    printf("time writing : %f ms\n\n", time_writing_ms);

    float avgTime_ms = totalTime_loop_ms / nIters;
    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / (avgTime_ms * 1e-3);
    printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);
    printf("TOT time loop : %f ms \n", totalTime_loop_ms);
    //TODO free memory (both host and device) with the CUDA function
    free(bodies);

    check_correctness(output_values, solution_values, size, nBodies);

    return 0;

}
