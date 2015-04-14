#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "neuron.h"

double dtanh (double x) {
	return (1 - pow(tanh(x), 2));
}
double  sqloss(double x) {
	return 0.5*pow(x, 2);
}
double dsqloss(double x) {
	return x;
}
double lact(double x)
{
	return x;
}
double lgrd(double x)
{
	return 1.0;
}

int main(int argc, char* argv[])
{
	Data d("training", ' ', 1);

	std::vector<size_t> dim(3);
	dim[0] = 2;
	dim[1] = 1;

	Funct L(&sqloss, &dsqloss);
	Funct Phi(&lact, &lgrd);

	Network net(dim, &Phi, &L, &d);

	std::vector<size_t> obs(4);
	obs[0] = 2;
	obs[1] = 1;
	obs[2] = 0;
	obs[3] = 3;

	net.initialize();
	
	net.print();
	net.train(0.1, obs, 100000);
	//net.print();
    return 0;
}
