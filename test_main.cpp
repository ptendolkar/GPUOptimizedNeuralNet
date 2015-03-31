#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "neuron.h"

double dtanh (double x) {
	return (1 - pow(tanh(x), 2));
}
double  sqloss(double x) {
	return pow(x, 2);
}
double dsqloss(double x) {
	return 2*x;
}

int main(int argc, char* argv[])
{
	Data d("training", ' ', 1);
	Data *train = &d;

	std::vector<size_t> dim(3);
	dim[0] = dim[1] = 2;
	dim[2] = 1;

	Network net(dim, (Funct *)NULL, (Funct *)NULL, train);
	
	Matrix C(5,5);
	C.initialize();

	C.writeToFile("testmatrix");

    return 0;
}
