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
	
//	double h_A[9] = {2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0};
//	double h_B[9] = {2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0};
	
	Matrix h_A(3,3);
	Matrix h_B(3,3,2);

	Matrix h_C(3,3);

	h_A.identity();
	
    	double *d_A, *d_B, *d_C;
	
	// allocate host memory for matrices A and B
   	unsigned int size_A = 9; 
   	unsigned int mem_size_A = sizeof(double) * size_A;
    	 
	unsigned int size_B = 9;
    	unsigned int mem_size_B = sizeof(double) * size_B;

	unsigned int mem_size_C = mem_size_A;
	double *h_CUBLAS = (double *) malloc(mem_size_C);

	checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
    	checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
    	checkCudaErrors(cudaMemcpy(d_A, &h_A.front(), mem_size_A, cudaMemcpyHostToDevice));
    	checkCudaErrors(cudaMemcpy(d_B, &h_B.front(), mem_size_B, cudaMemcpyHostToDevice));
    	checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));

        const double alpha = 1.0f;
        const double beta  = 0.0f;
        cublasHandle_t handle;

        checkCudaErrors(cublasCreate(&handle));

        checkCudaErrors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, 3, &alpha, d_A, 3, d_B, 3, &beta, d_C, 3));

        checkCudaErrors(cudaMemcpy(&h_C.front(), d_C, mem_size_C, cudaMemcpyDeviceToHost));

        checkCudaErrors(cublasDestroy(handle));
   
	checkCudaErrors(cudaFree(d_A));
    	checkCudaErrors(cudaFree(d_B));
    	checkCudaErrors(cudaFree(d_C));

	std::cout << "output" << std::endl;
	
/*	for (int i = 0; i < 9; i++)
		std::cout << h_CUBLAS[i] << " ";
	std::cout << std::endl;
*/
	h_C.print();
	
	std::cout << "test successful" << std::endl;
	return 0;	
}
