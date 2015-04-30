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
#include <stdio.h>

#include "dneuron.h"
#include <ctime>

__device__ float dtanh (float x) {
	return (1 - pow(tanh(x), 2));
}
__device__ float  sqloss(float x) {
	return 0.5*pow(x, 2);
}
__device__ float dsqloss(float x) {
	return x;
}
__device__ float lact(float x)
{
	return x;
}
__device__ float lgrd(float x)
{
	return 1.0;
}

__global__ void train( Network *net, DevData *dd, float *dX, int  n_row, int n_col, int n_rsp, int n_fea, float alpha, int iters)
{
	printf("in train kernel\n");
	for (int i= 0; i< n_row*n_col; i++){
		printf("%f ", dX[i]);
	}
	printf("\n");

	Funct L   ( &sqloss , &dsqloss);
	Funct Phi ( &lact   , &lgrd);
	Funct Psi ( &tanh   , &dtanh);

	dd = new DevData(dX, n_row, n_col, n_rsp, n_fea);
	int dim[3];
	    dim[0] = 2;
	    dim[1] = 2;
	    dim[2] = 1;
	
	int obs[4];
	obs[0] = 2;
	obs[1] = 0;
	obs[2] = 1;
	obs[3] = 3;

	net = new Network(dim, 3, &Psi, &L, dd);
	net->initialize();
	net->print();
	net->train(alpha, obs, 4,iters);
	net->print();
}

int main(int argc, char* argv[])
{
	time_t start, end;
	time(&start);
	
	int cuda_device = 0;
    	cuda_device = findCudaDevice(argc, (const char **)argv);

    	cudaDeviceProp deviceProp;
    	checkCudaErrors(cudaGetDevice(&cuda_device));
    	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

	//cudaDeviceSetLimit(cudaLimitMallocHeapSize, 512 * (1 << 20));
	
	std::cout << "before allocation" << std::endl;

	Data d("training", ' ', 2);
	DevData *dd;
	Network *net;
	
	cudaMalloc(&dd, sizeof(DevData *));
	cudaMalloc(&net, sizeof(Network *));

	float alpha = atof(argv[1]);
	int iters = atoi(argv[2]);
	train<<<1,1>>>(net, dd, thrust::raw_pointer_cast(&(d.X[0])), d.nrow(), d.ncol(), d.nrsp(), d.nfea(), alpha, iters);
	cudaDeviceSynchronize();
	time(&end);

	printf("Time: %f\n",difftime(end, start));
	//cudaDeviceReset();

	return 0;

}
