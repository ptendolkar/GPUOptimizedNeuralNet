#pragma once

class Funct
{
	private:
		float (*fun)(float);
		float (*grd)(float);
	
	public:
		__device__ Funct();
		__device__ Funct(float (*)(float), float (*)(float));
		
		__device__ ~Funct();
		
		__device__ float (*get_fun())(float);
		__device__ float (*get_grd())(float);
};
