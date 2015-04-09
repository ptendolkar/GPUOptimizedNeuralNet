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
/*	Matrix X(3,1,1);
	Matrix y(2,1);

	Matrix z(2,1);
	y[0] =0.5;
	y[1] =1.5;*/
//	y[2] =2.5;
	/*double X[3] = {1.0,1.0,1.0};
	double y[2] = {0.5, 1.5};
	double z[2];*/

/*	Matrix X(3,1,.5);
	Matrix y(3,1);
	y[0] = 0.5;
	y[1] = 1.5;
	y[2] = -1 ;	
	Matrix z(3,1);

	int N = 3;
	int k = 0;
	int lda = 1;
	int inc_z = 1;
	int inc_y =1;
	double alpha = 1.0;
	double beta = 0.0;

	dsbmv_("L", &N, &k, &alpha, &*X.std::vector<double>::begin(), &lda, &*y.std::vector<double>::begin() ,&inc_y, &beta ,&*z.std::vector<double>::begin(),&inc_z );
	
	std::cout << z[0] << " " << z[1] << " " << z[2] <<  std::endl; */
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
	Matrix v1(1,3);
	Matrix v2(1,1);
	Matrix p(3,1);

	v1[0] = 1;
	v1[1] = 2;
	v1[2] = 3;

	v2[0] = 2;
	//v2[2] = 2;
	//v2[1] = -1;
	
	dger(1, *((double *) &v1[0]), 1, v2, 1, p); 
	p.print();
*/	

	Data d("training", ' ', 1);

	std::vector<size_t> dim(2);
	dim[0] = 2;
	dim[1] = 1;
	//dim[2] = 1;

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
	net.train(1, obs, 100);
	net.print();
	net.writeModelToFile();

	std::vector<double> tr1(2);
	tr1[0] = 0.01;
	tr1[1] = 0.99;

	(net.predict(tr1)).print();

    return 0;
}
