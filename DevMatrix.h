#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublas_api.h>
#include <curand_kernel.h>
#include <stdio.h>


class DevMatrix
{
	private:
                int n_row;
                int n_col;
                float* M;
                curandState *devStates;

        public:

                __device__ DevMatrix(); 
                __device__ DevMatrix(size_t, size_t );
                __device__ ~DevMatrix();

                __device__ void initialize(unsigned long, float, float);

                __device__ float * getM();

                __host__ void fill(float x);

                __host__ __device__ int size(); 
                __host__ __device__ int nrow(); 
                __host__ __device__ int ncol(); 

                __device__ float * begin();
		__device__ float * end();

                __device__ void write(size_t , size_t , float );
                __device__ float read(size_t , size_t );

                __device__ void print();

                __device__ void identity();

		__device__ void copy(DevMatrix &);
};


/*	CUBLAS INTERFACE 	*/

__device__ void saxpy(const float, DevMatrix &, const int, DevMatrix &, const int);

__device__ void saxpy(float, const float &, const int, DevMatrix &, const int);

__device__ void sgemv(cublasOperation_t, const float, DevMatrix &, DevMatrix &, int, const float, DevMatrix &, const int);

__device__ void sgemv(cublasOperation_t, const float, DevMatrix &, const float &, const int, const float, DevMatrix &, const int);

// k = 1 and LDA = 1 for a diagonal matrix (0 = n_super = n_lower), stored columnwise in a 1 x N vector where N is the number of columns of A

__device__ void ssbmv(cublasFillMode_t, const float, DevMatrix &, const int, DevMatrix &, const int, const float, DevMatrix &, const int);

//A := alpha*x*y**T + A 

__device__ void sger(const float, DevMatrix &, const int, const float &, const int, DevMatrix &);

//A := alpha*x*y**T + A 
__device__ void sger(const float, DevMatrix  &, const int, DevMatrix &y, const int, DevMatrix &A);

__device__ void sgemm(cublasOperation_t, cublasOperation_t, float, DevMatrix &, DevMatrix &, float, DevMatrix &);
