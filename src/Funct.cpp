#include <Funct.h>

Funct::Funct() : fun(NULL), grd(NULL) {}
Funct::Funct(double (*f)(double), double (*g)(double)) : fun(f), grd(g) {}

Funct::~Funct()
{
	fun = NULL;
	grd = NULL;
};

double (*Funct::get_fun())(double) const { return fun; }
double (*Funct::get_grd())(double) const { return grd; }
