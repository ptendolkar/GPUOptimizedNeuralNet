#include "cublas_interface.h"
#include "Funct.h"
#include "DevMatrix.h"
#include "DevData.h"

#pragma once

class Layer : public DevMatrix
{
	private:
		size_t iden;
		Layer *prev_lay_ptr;
		Layer *next_lay_ptr;

	public:
		DevMatrix bias;
		DevMatrix flux;
		DevMatrix actv;

		__device__ Layer();
		__device__ Layer(size_t, size_t, size_t);
		__device__ Layer(size_t, size_t, size_t, Layer *, Layer *);
		__device__ Layer(size_t, size_t, size_t, Layer *, Layer *, Funct *);

		__device__ size_t id();
		__device__ Layer * prev();
		__device__ Layer * next();

		__device__ float * w();
		__device__ float * b();
		__device__ float * z();
		__device__ float * a();
		
		__device__ float eval_f(float x);
		__device__ float eval_g(float x);
		
		__device__ void id(size_t);
		__device__ void prev(Layer *);
		__device__ void next(Layer *);
		__device__ void f(size_t, Funct *);

		__device__ ~Layer();

		__device__ void push(size_t, DevData *);
};	