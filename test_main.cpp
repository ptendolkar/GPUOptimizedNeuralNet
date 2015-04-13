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

	std::vector<size_t> dim(2);
	dim[0] = 2;
	dim[1] = 1;

	Funct L(&sqloss, &dsqloss);

	Network net(dim, (Funct *)NULL, &L, train);

    return 0;
}
