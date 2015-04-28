#include <iostream>
#include <math.h>
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublas_api.h>
// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

#include "dmatrix.h"
#include <stdio.h>

__global__ void newmatrix(Matrix * M, Matrix * N, Matrix *O )
{
	printf("inside kernel\n");
	M = new Matrix(3,3);
	M->print();
	printf("\n");
	M->initialize();

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
	Matrix *d_M, *d_N, *d_O;
	cudaMalloc(&d_M, sizeof(Matrix *));
	cudaMalloc(&d_N, sizeof(Matrix *));
	cudaMalloc(&d_O, sizeof(Matrix *));

	newmatrix<<<1,1>>>(d_M, d_N, d_O);
	cudaDeviceSynchronize();

	std::cout << "completed" << std::endl;
		
	//saxpy(1.0, *d_M->begin(), 1, *d_N->begin(), 1);

	cudaDeviceReset();
	return 0;

}
