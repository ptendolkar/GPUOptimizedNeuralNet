#include <stdio.h>
#include <stdint.h> //for uint apparently
#include "neuron.h"
#include <cmath>
#include <ctime>

#define BILLION 1000000000L
int localpid(void) 
{ 
	static int a[9] = { 0 }; 
	return a[0]; 
}

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
		
	uint64_t diff;
	struct timespec start, end;

	clock_gettime(CLOCK_MONOTONIC, &start); /* mark start time */
	Data d("data.csv", ',', 1);

	std::vector<size_t> dim(3);
	dim[0] = d.nfea();
	dim[1] = 15;
	dim[2] = 1;

	Funct L(&sqloss, &dsqloss);
	Funct Phi(&lact, &lgrd);
	Funct Psi(&tanh, &dtanh);

	Network net(dim, &Psi, &L, &d);
	
	int nobs =4;
	std::vector<size_t> obs(nobs);

	for(int i =0 ; i< obs.size(); i++)
	{
		obs[i] = i;
	}

	double a = atof(argv[1]);
	int n = atoi(argv[2]);
	
	net.initialize(1234);
	net.print();
	net.train(a, obs, n);
	net.print();

	clock_gettime(CLOCK_MONOTONIC, &end); /* mark start time */
	diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	printf("Seconds: %f\n", diff/1000000000.0);

    return 0;
}
