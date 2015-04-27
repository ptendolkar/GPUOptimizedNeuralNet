#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples
// CUDA runtime
#include <cuda_runtime.h>
//#include <cublas_v2.h>
// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>
#include "dmatrix.h"
#include<stdio.h>

__global__ void newmatrix(Matrix * M)
{
	printf("inside kernel\n");
	M = new Matrix(5,5);
	M->print();
	M->identity();
	M->print();
}

int main(int argc, char* argv[])
{
	int cuda_device = 0;

    	cuda_device = findCudaDevice(argc, (const char **)argv);

    	cudaDeviceProp deviceProp;
    	checkCudaErrors(cudaGetDevice(&cuda_device));
    	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

	//cudaDeviceSetLimit(cudaLimitMallocHeapSize, 512 * (1 << 20));
	
	std::cout << "before allocation" << std::endl;
	Matrix *d_M;
	cudaMalloc(&d_M, sizeof(Matrix *));
	newmatrix<<<1,1>>>(d_M);
	cudaDeviceSynchronize();

	std::cout << "completed" << std::endl;
		
	cudaDeviceReset();
	return 0;
}
