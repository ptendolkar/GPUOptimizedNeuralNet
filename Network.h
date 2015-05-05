#pragma once 
#include "Layer.h"
#include "Funct.h"
#include "DevData.h"
#include "DevMatrix.h"

class Network 
{
	private:
		size_t           n_lay;
		Layer            *head_lay_ptr;
		Layer            *tail_lay_ptr;
		Funct            *loss;
		DevData          *data_ptr;
		cublasHandle_t   *handle;

		DevMatrix *del_curr;
		DevMatrix *del_past;

	public:

		 __device__ Network(); 

// Build network dynamically fowards (head to tail) from the output layer.  Single layer network (e.g. logistic regression) will have NULL input layer pointer,
// but all networks must have an output.  The first entry of the dimension array is the size of the covariate space, and the last entry is the size of the output space.

		 __device__ Network(int *, int, Funct *, Funct *, DevData *, cublasHandle_t *, int);

		 __device__ size_t depth(); 
		 __device__ Layer  *head();
		 __device__ Layer  *tail();
		 __device__ Funct  *lfun();
		 __device__ DevData *data();

		 __device__ ~Network();

		 __device__ void depth(size_t i); 

		 __device__ void clear();

		 __device__ void feed_forward(size_t);
		 __device__ void backprop(float, size_t);

	 	 __device__ void train(float, int *, int,  size_t);

		 __device__ void print();
		 __device__ void initialize(unsigned long, float, float);
};
