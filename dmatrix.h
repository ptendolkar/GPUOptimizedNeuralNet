#include <iostream>
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublas_api.h>
// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>


__global__ void kernelFill(float x, float * mat)
{
	mat[threadIdx.x] = x;
}

class Matrix 
{
	private:
		int n_row;
		int n_col;	
		float* M;
	public:
		
		__device__ Matrix() : M(NULL), n_row(0), n_col(0) {;}
		__device__ Matrix(size_t m, size_t n) {
			n_row = m;
			n_col = n;
			M = new float[m*n];
		}
		__device__ ~Matrix()
		{
			n_row=n_col=0;
			delete[] M;
			M = (float *) NULL;
		}	
		
		__host__ __device__ float * getM(){ return M; }

		__host__ void fill(float x)
		{
			int jobs = n_row*n_col;
			kernelFill<<<1, jobs>>>(x, getM());
		}
		
		__host__ __device__ int size() { return n_row*n_col;}
		__host__ __device__ int nrow() { return n_row; }
		__host__ __device__ int ncol() { return n_col; }

		float * begin()
		{
			return M;
		}
		
		float * end()
		{
			return M + (n_row*n_col);
		}
		
		__device__ void write(size_t i, size_t j, float x)
		{
			M[i + j*n_row] = x;
		}
		__device__ float read(size_t i, size_t j)
		{
			return M[i + j*n_row];
		}

				
		__device__ void print(){

			if(n_row == 0 || n_col == 0){
				printf("empty matrix\n");
				return;
			}
			for(int i = 0; i < n_row; i++)
			{
				for(int j = 0; j < n_col; j++)
				{
					printf("%f ", read(i,j));	
				}
				printf("\n");
			}			
		}
		

		 __device__ void identity(){
			if( this->n_row != this->n_col){
				printf("empty matrix\n");
				return;
			}else{
				for(int i = 0; i < this->n_row; i++){
					for(int j = 0; j < this->n_col; j++){
						if(i == j)
							write(i, j, 1.0);
						else
							write(i, j, 0.0);
					}
				}
			}
		}
	
};

/*extern "C"
{
	// Level 1
	void daxpy_(const int *N, const float *ALPHA, const float *X, const int *INCX, float *Y, const int *INCY);

	// Level 2
	void dgemv_(const char *TRANSA, const int *M, const int *N, const float *ALPHA, const float *A, const int *LDA, const float *X, const int *INCX, const float *BETA, float *Y, const int *INCY);
	void dsbmv_(const char *UPLO, const int *N, const int *K, const float *ALPHA, const float *A, const int *LDA, const float *X, const int *INCX, const float *BETA, float *Y, const int *INCY);
	void dger_ (const int *M, const int *N, const float *ALPHA, const float *X, const int *INCX, const float *Y, const int *INCY, float *A, const int *LDA);
	void dgemm_(const char *TRANSA, const char *TRANSB, const int *M, const int *N, const int *K, const float *ALPHA, const float *A, const int *LDA, const float *B, const int *LDB, const float *BETA, const float *C, const int *LDC);
}
*/
//y  <--  alpha*x + y

/*void saxpy(const float alpha, const thrust::device_vector<float> &x, const int inc_x, thrust::device_vector<float> &y, const int inc_y)
{
	const int n = y.size();

	cublasSaxpy(cuHandle::handle, n, &alpha, thrust::raw_pointer_cast(&x[0]), inc_x, thrust::raw_pointer_cast(&y[0]), inc_y);
}
*//*
void daxpy(float alpha, const float &x, const int inc_x, thrust::device_vector<float> &y, const int inc_y)
{
	const int n = y.size();

	cublasDaxpy(cuHandle::handle, n, &alpha, &x, inc_x, &*y.begin(), inc_y);
}
*//*
void sgemv(cublasOperation_t trans, const float alpha, const Matrix &A, const thrust::device_vector<float> &x, const int inc_x, const float beta, thrust::device_vector<float> &y, const int inc_y)
{
	int M = A.nrow();
	int N = A.ncol();

	int LDA = A.nrow();

	cublasSgemv(cuHandle::handle, trans, M, N, &alpha, thrust::raw_pointer_cast(&A[0]), LDA, thrust::raw_pointer_cast(&x[0]) , inc_x, &beta, thrust::raw_pointer_cast(&y[0]), inc_y); 
}
*//*
void dgemv(cublasOperation_t trans, const float alpha, const Matrix &A, const float &x, const int inc_x, const float beta, thrust::device_vector<float> &y, const int inc_y)
{
	int M = A.nrow();
	int N = A.ncol();

	int LDA = A.nrow();

	cublasDgemv(cuHandle::handle, trans, M, N, &alpha, &*A.begin(), LDA, &x, inc_x, &beta, &*y.begin(), inc_y); 
}
*/
// k = 1 and LDA = 1 for a diagonal matrix (0 = n_super = n_lower), stored columnwise in a 1 x N vector where N is the number of columns of A

/*void dsbmv(const char UPLO, const float alpha, const Matrix &A, const int K, const thrust::device_vector<float> &x, const int inc_x, const float beta, thrust::device_vector<float> &y, const int inc_y)
{
	int N = A.nrow();
	int LDA = 1;

	dsbmv_(&UPLO, &N, &K, &alpha, &*A.thrust::device_vector<float>::begin(), &LDA, &*x.begin(), &inc_x, &beta, &*y.begin(), &inc_y); 
}*/

//A := alpha*x*y**T + A 
/*
void dger(const float alpha, const thrust::device_vector<float>  &x, const int inc_x, const float &y, const int inc_y, Matrix &A)
{
	int M  = A.nrow();
	int N  = A.ncol();

	int LDA = A.nrow();

	cublasDger(cuHandle::handle, M, N, &alpha, &*x.begin(), inc_x, &y, inc_y, &*A.thrust::device_vector<float>::begin(), LDA);
}

//A := alpha*x*y**T + A 
void dger(const float alpha, const thrust::device_vector<float> &x, const int inc_x, const thrust::device_vector<float> &y, const int inc_y, Matrix &A)
{
	int M  = A.nrow();
	int N  = A.ncol();

	int LDA = A.nrow();

	cublasDger(cuHandle::handle, M, N, &alpha, &*x.begin(), inc_x, &*y.begin(), inc_y, &*A.thrust::device_vector<float>::begin(), LDA);
}
void dgemm(cublasOperation_t TrA, cublasOperation_t TrB, float alpha, Matrix &A, Matrix &B, float beta, Matrix &C)

{
	int M;
	int N;
	int K;

	int LDA = A.nrow();
	int LDB = B.nrow();
	int LDC = C.nrow();

	switch(TrA)
	{
		case CUBLAS_OP_N:
		{
			switch(TrB)
			{
				case CUBLAS_OP_N:
				{
						M = A.nrow();
						N = B.ncol();
						K = B.nrow();
						break;	
				}
				case CUBLAS_OP_T:
				{
						M = A.nrow();
						N = B.nrow();
						K = B.ncol();
						break;
				}
			}
			break;
		}
		case CUBLAS_OP_T:
		{
			switch(TrB)
			{
				case CUBLAS_OP_N:
				{
						M = A.ncol();
						N = B.ncol();
						K = B.nrow();
						break;
				}
				case CUBLAS_OP_T:
				{
						M = A.ncol();
						N = B.nrow();
						K = B.ncol();
						break;
				}
			}
			break;
		}
	}


	cublasDgemm(cuHandle::handle,TrA, TrB, M, N, K, &alpha, &*A.thrust::device_vector<float>::begin(), LDA, &*B.thrust::device_vector<float>::begin(), LDB, &beta, &*C.thrust::device_vector<float>::begin(), LDC);
} 
*/
