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
	dim[1] = 2;
	dim[2] = 1;

	Funct L(&sqloss, &dsqloss);
	Funct Phi(&lact, &lgrd);
	Funct Psi(&tanh, &dtanh);

	Network net(dim, &Psi, &L, &d);

	std::vector<size_t> obs(4);
	obs[0] = 0;
	obs[1] = 1;
	obs[2] = 2;
	obs[3] = 3;

	net.initialize();

	double a = atof(argv[1]);
	int n = atoi(argv[2]);
	
	net.print();
	net.train(a, obs, n);
/*	
	double del = .1;
	for(int i=-1; i < 1.1; i+= del)
	{
		for (int j = -1; j < 1.1; j+=del)
		{
			std::vector<double> xhat(2);
			xhat[0] = i;
			xhat[1] = j;
			std::cout<< net.predict(xhat)[0] << " ";
		}
		std::cout << std::endl;
	}*/
    return 0;
}
