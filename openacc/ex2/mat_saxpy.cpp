#include <stdlib.h>
#include <stdio.h>
#include <chrono>
//TODO: include the needed header

void mat_saxpy(int n, double a, double *x, double *y) {
	//TODO parallelize and optimize the loop
	for (int j = 0; j < n; ++j)
		for (int i = 0; i < n; ++i)
			y[i+n*j] = a * x[i+n*j] + y[i+n*j];
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

	//TODO move the data on GPU memory

	auto comp_start = std::chrono::high_resolution_clock::now();
	mat_saxpy(N, 3.0, x, y);
	auto comp_stop = std::chrono::high_resolution_clock::now();

	//TODO finalize the data: copy them back on CPU
	
	auto stop = std::chrono::high_resolution_clock::now();
	printf(" time : %1.5f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count()*1e-3);
	printf(" computation time : %1.5f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(comp_stop-comp_start).count()*1e-3);	
	printf(" print the first value of y (should be 7 if you do not change any number) : \n %1.1f\n", y[0]);

	return 0;

}

