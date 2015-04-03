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
/*	Matrix A(2,2);
	A.identity();
	A(0,1)=3;
	Matrix x(2,2);
	x(0,0) =-1;
	x(0,1) =-1;

	x(1,0) = 1;
	x(1,1) = 2;
	Matrix flux(2,1);
	
	std::cout << sizeof(double) << std::endl;	
	double* L = &x[0]+1 ;
	dgemv('N', 1.0, A, *L, x.nrow(), 0.0, *((std::vector<double>*)&flux), 1);

	flux.print();
	
*/
	Data d("training", ' ', 1);

	std::vector<size_t> dim(3);
	dim[0] = dim[1] = 2;
	dim[2] = 1;

	Funct g(&(nncuda::lact), &(nncuda::lgrd));
	Funct l(&sqloss, &dsqloss);

	Network net(dim, &g, &l, &d);
	
	std::vector<size_t> obs(4);
	obs[0] = 0;
	obs[1] = 1;
	obs[2] = 2;
	obs[3] = 3;
	
	net.initialize();
	net.train(.001, obs, 10);
	net.writeModelToFile();

    return 0;
}
