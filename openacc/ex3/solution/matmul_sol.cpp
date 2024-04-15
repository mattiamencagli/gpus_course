#include <stdlib.h>
#include <stdio.h>
#include <chrono> 

#include <openacc.h>

void matmul(const double *M1, const double *M2, double *MS, const int &N) {

    #pragma acc parallel loop collapse(2) present(M1, M2, MS)
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            
            double sum = 0.0;
            #pragma acc loop reduction(+:sum)
            for (int k = 0; k < N; ++k)
                sum += M1[k + j * N] * M2[i + k * N];

            MS[i + j * N] = sum;

        }

}

void init_matrix(double *M, const int &N, const double val) {
    const int N2 = N * N;
    for (int i = 0; i < N2; ++i)
        M[i] = val;
}



int main( int argc, char  **argv){

    int N = 1e3;
	if (argc > 1)
		N = atoi(argv[1]);
    int N2 = N*N;

    double *A = new double[N2];
    double *B = new double[N2];
    double *C = new double[N2];

    init_matrix(A, N, 1.0);
    init_matrix(B, N, 2.0);

    auto start = std::chrono::high_resolution_clock::now();

    #pragma acc enter data copyin(A[:N2],B[:N2]) create(C[:N2])

    auto comp_start = std::chrono::high_resolution_clock::now();
    matmul(A,B,C,N);
    auto comp_stop = std::chrono::high_resolution_clock::now();

    #pragma acc exit data delete(A[:N2], B[:N2]) copyout(C[:N2])

    auto stop = std::chrono::high_resolution_clock::now();
	printf(" time : %1.5f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count()*1e-3);
	printf(" computation time : %1.5f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(comp_stop-comp_start).count()*1e-3);
	printf(" print the first value of C (should be %d if you do not change the values within the matrices) : \n %1.1f\n", N*2, C[0]);

    return 0;

}

