#ifndef FUNCT_H
#define FUNCT_H
#include <cstddef>

class Funct 
{
	private:
		double (*fun)(double);
		double (*grd)(double);

	public:
		Funct(); 
		Funct(double (*)(double), double (*)(double));

		~Funct();


		double (*get_fun())(double);
		double (*get_grd())(double);
};

#endif
