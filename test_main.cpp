#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include "neuron.h"

int main(int argc, char* argv[])
{
	Matrix A(2,2);
	A(0,0) = 4.0;
	std::vector<double> B(4);
	B = (std::vector<double>)A;
	for (int i=0; i<4; i++)
	{
		std::cout << B[i] << std::endl;
	}
    return 0;
}
