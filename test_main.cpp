#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "neuron.h"
//#include "matrix.h"

double dtanh (double x) {
	return (1 - pow(tanh(x), 2));
}
double sqloss(double x) {
	return pow(x, 2);
}
double dsqloss(double x) {
	return 2*x;
}

int main(int argc, char* argv[])
{
	Data d("training", ' ', 1);
	
	
    	return 0;
}
