#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include <chrono> 

#include <openacc.h>

#define SOFTENING 1e-9f

// Each body contains x, y, and z coordinate positions, as well as velocities in the x, y, and z directions.
typedef struct { float x, y, z, vx, vy, vz; } Body;

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


void bodyForce(Body *p, float dt, int n) {

    #pragma acc parallel loop present(p) gang
    for (int i = 0; i < n; ++i) {
        float Fx = 0.0f; 
        float Fy = 0.0f; 
        float Fz = 0.0f;

        #pragma acc loop vector reduction(+:Fx,Fy,Fz)
        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = 1.0f/sqrt(distSqr);
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

void integratePosition(Body* p, float dt, int n) {

    #pragma acc parallel loop present(p) 
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
    
    Body *bodies = (Body *)malloc(size);

    read_values_from_file(initialized_values, bodies, size);

    const float dt = 0.01f;  // Time step
    const int nIters = 10;  // Simulation iterations

    auto start = std::chrono::high_resolution_clock::now();
    #pragma acc enter data copyin(bodies[:nBodies])
    /*
    * This simulation will run for 10 cycles of time, calculating gravitational
    * interaction amongst bodies, and adjusting their positions according to their new velocities.
    */
    for (int iter = 0; iter < nIters; iter++) {

        auto starti = std::chrono::high_resolution_clock::now();

        //TODO Use the CUDA function
        bodyForce(bodies, dt, nBodies);
        
        //TODO remeber to wait for the GPU to finish before going to the next kernel.
        #pragma acc wait

        //TODO Use the CUDA function
        integratePosition(bodies, dt, nBodies);

        //TODO remeber to wait for the GPU to finish before going to the next kernel.
        #pragma acc wait

        auto stopi = std::chrono::high_resolution_clock::now();
        printf("time iter %d : %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(stopi-starti).count()*1e-3);

    }

    #pragma acc exit data copyout(bodies[:nBodies])
    auto stop = std::chrono::high_resolution_clock::now();
    
    //TODO If you DO NOT use the managed memory, remeber to transfer the results back on the host
    write_values_to_file(output_values, bodies, size); 
    
    float totalTime_loop_ms = std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count()*1e-3;
    float avgTime_ms = totalTime_loop_ms / nIters;
    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / (avgTime_ms * 1e-3);
    printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);
    printf("TOT time %d : %f ms\n", totalTime_loop_ms);
    
    //TODO free memory (both host and device) with the CUDA function
    free(bodies);

    check_correctness(output_values, solution_values, size, nBodies);

    return 0;

}
