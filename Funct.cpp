#include "Funct.h"

__device__ Funct::Funct() : fun(NULL), grd(NULL){}
__device__ Funct::Funct(float (*f)(float), float (*g)(float)) : fun(f), grd(g) {}

__device__ Funct::~Funct()
{

	fun = NULL;
	grd = NULL;

}

__device__ float (*Funct::get_fun())(float)
{
	return fun;
}

__device__ float (*Funct::get_grd())(float)
{
	return grd;
}
