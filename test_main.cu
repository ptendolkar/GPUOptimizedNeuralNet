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
#include "thrustMatrix.h"

double dtanh (double x) {
	return (1 - pow(tanh(x), 2));
}
double  sqloss(double x) {
	return 0.5*pow(x, 2);
}
double dsqloss(double x) {
	return x;
}
double lact(double x)
{
	return x;
}
double lgrd(double x)
{
	return 1.0;
}

/*__global__ void initializeMatrix(Matrix **M)
{

	}
*/
__global__ void newmatrix(Matrix *M){
	M = new Matrix(5,5);
}
int main(int argc, char* argv[])
{
	int cuda_device = 0;

    	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
    	cuda_device = findCudaDevice(argc, (const char **)argv);

    	cudaDeviceProp deviceProp;
    	checkCudaErrors(cudaGetDevice(&cuda_device));
    	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

	if (deviceProp.major < 2)
    	{
    	    printf("> This GPU with Compute Capability %d.%d does not meet minimum requirements.\n", deviceProp.major, deviceProp.minor);
    	    printf("> Test will not run.  Exiting.\n");
    	    exit(EXIT_SUCCESS);
    	}
	
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 512 * (1 << 20));
	
	Matrix *d_M;
	cudaMalloc(&d_M, sizeof(Matrix *));
	newmatrix<<<1,1>>>(d_M);
	
	std::cout << "completed" << std::endl;
		
	return 0;
}
