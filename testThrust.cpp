#include <thrust/random.h>
#include <iostream>
#include <iomanip>
#include <cmath>
int main(){
	thrust::default_random_engine rng(1234);
	thrust::uniform_real_distribution<double> X(0, 1);
	double x = X(rng);
	double x2 = X(rng);	
	std::cout << x<< " "<< x2 << std::endl;

	return 0;
}
