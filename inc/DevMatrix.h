#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublas_api.h>
#include <curand_kernel.h>

#pragma once

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
