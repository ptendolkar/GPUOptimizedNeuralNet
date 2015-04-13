#include"matrix.h"
#include<iostream>
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>


void initializeCUDA(int argc, char **argv, int &devID)
{
    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    cudaError_t error;
    devID = 0;

    // get number of SMs on this GPU
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp deviceProp;

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

}


int main(int argc, char *argv[])
{
	int devID = 0;	
	initializeCUDA(argc, argv, devID);
	
	Matrix h_A(5,5);
	Matrix h_B(5,5);
	Matrix h_C(5,5);

	h_A.initialize();
	h_B.identity();
	h_B(4,3) = -0.3;
	
    	double *d_A, *d_B, *d_C;
	
	// allocate host memory for matrices A and B
   	unsigned int size_A = h_A.nrow() * h_A.ncol();
   	unsigned int mem_size_A = sizeof(float) * size_A;
    	
	unsigned int size_B = h_B.nrow() * h_B.ncol();
    	unsigned int mem_size_B = sizeof(float) * size_B;

	checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
    	checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
    	checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    	checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
    	checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));

        const float alpha = 1.0f;
        const float beta  = 0.0f;
        cublasHandle_t handle;

        checkCudaErrors(cublasCreate(&handle));

        checkCudaErrors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, h_A.nrow(), h_B.ncol(), h_A.ncol(), &alpha, d_A, h_A.nrow(), d_B, h_B.nrow(), &beta, d_C, h_A.nrow()));

        checkCudaErrors(cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost));

        checkCudaErrors(cublasDestroy(handle));
   
	// clean up memory
    	free(h_A);
   	free(h_B);
   	free(h_C);
    	
	checkCudaErrors(cudaFree(d_A));
    	checkCudaErrors(cudaFree(d_B));
    	checkCudaErrors(cudaFree(d_C));

	return 0;	
}
