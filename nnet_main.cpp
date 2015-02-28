#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

int main(int argc, char* argv[])
{
	size_t m = 8;
	size_t k = 6;
	size_t n = 1;

	Matrix A(m,k);
	Matrix B(k,n);
	Matrix C(m,n);

	srand(86456);

	double max_ceil = (double)RAND_MAX;

	for (int i=0; i<m; i++)
		for (int j=0; j<k; j++)
			A(i,j) = rand()/max_ceil;

	for (int i=0; i<k; i++)
		for (int j=0; j<n; j++)
			B(i,j) = rand()/max_ceil;

	double alpha = 1.0;
	double beta  = 0.0;

	dgemm("N", "N", alpha, A, B, beta, C);

	for (int i=0; i<m; i++)
	{
		int j = 0;
		std::cout << C(i,j) << std::endl;
	}

    return 0;
}
