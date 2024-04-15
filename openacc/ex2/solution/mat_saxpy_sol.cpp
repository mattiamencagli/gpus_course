#include <stdlib.h>
#include <stdio.h>
#include <chrono>

#include <openacc.h>

void saxpy(int n, double a, double *x, double *y) {
	#pragma acc parallel loop collapse(2) present(x, y)
	for (int j = 0; j < n; ++j)
		for (int i = 0; i < n; ++i){
			int idx = i + n * j;
			y[idx] = a * x[idx] + y[idx];
		}
}


int main(int argc, char **argv) {

	int N = 1e4;
	if (argc > 1)
		N = atoi(argv[1]);
	int N2 = N*N;

	double *x = (double*)malloc(N2 * sizeof(double));
	double *y = (double*)malloc(N2 * sizeof(double));

	for (int i = 0; i < N2; ++i) {
		x[i] = 2.0;
		y[i] = 1.0;
	}
	
	auto start = std::chrono::high_resolution_clock::now();

	#pragma acc enter data copyin(x[:N2], y[:N2])

	auto comp_start = std::chrono::high_resolution_clock::now();
	saxpy(N, 3.0, x, y);
	auto comp_stop = std::chrono::high_resolution_clock::now();

	#pragma acc exit data delete(x[:N2]) copyout(y[:N2])

	auto stop = std::chrono::high_resolution_clock::now();
	printf(" time : %1.5f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count()*1e-3);	
	printf(" computation time : %1.5f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(comp_stop-comp_start).count()*1e-3);	
	printf(" print the first value of y (should be 7 if you do not change any number) : \n %1.1f\n", y[0]);

	return 0;

}

