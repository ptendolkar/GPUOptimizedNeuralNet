#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "neuron.h"

int main(int argc, char* argv[])
{
	std::vector<double> x(8);
	x[0] = x[1] = x[2] = x[3] = 1;
	x[4] = x[5] = x[6] = x[7] = 2;

	Matrix A(4,4,0);
	A(0,0) = A(1,1) = A(2,2) = A(3,3) = 1;

	Data test;
	test.X = x;
	test.n_feat = 4;

	Layer L0(0,4,4);
	L0.w_swp(A);

//	L0.push(1, test);

    return 0;
}
