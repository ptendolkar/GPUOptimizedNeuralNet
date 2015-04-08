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
	//A(0,1)=3;
	Matrix x(2,2);
	x(0,0) =-1;
	x(0,1) =-1;

	x(1,0) = 1;
	x(1,1) = 2;
	Matrix flux(2,1);
	
	double* L = &x[0]+1 ;
	dgemv('T', 1.0, A, *L, x.nrow(), 0.0, *((std::vector<double>*)&flux), 1);

	flux.print();
*/	
/*
	Matrix v1(3,1);
	Matrix v2(3,1);
	Matrix p(3,3);

	v1[0] = 1;
	v1[1] = 2;
	v1[2] = 3;

	v2[0] = v2[2] = 2;
	v2[1] = -1;
	
	dger(1, *((double *) &v1[0]), 1, v2, 1, p); 
	p.print();
	*/

	Data d("training", ' ', 1);

	std::vector<size_t> dim(3);
	dim[0] = dim[1] = 2;
	dim[2] = 1;

	Funct g(&(nncuda::lact), &(nncuda::lgrd));
	Funct l(&sqloss, &dsqloss);

	Network net(dim, &g, &l, &d);
	
	std::vector<size_t> obs(1);
	obs[0] = 2;
//	obs[1] = 1;
//	obs[2] = 2;
//	obs[3] = 3;
	
	net.initialize();
	net.print();
	net.train(.01, obs, 100000);
	net.print();
	net.writeModelToFile();

	std::vector<double> tr1(2);
	tr1[0] = 0.01;
	tr1[1] = 0.99;

	(net.predict(tr1)).print();

    return 0;
}
