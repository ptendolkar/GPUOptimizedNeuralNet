#include <cublas_v2.h>
#include <cublas_api.h>
#include "DevMatrix.h"

#pragma once
/*	CUBLAS INTERFACE 	*/

//y  <--  alpha*x + y

__device__ void saxpy(const float alpha, DevMatrix &x, const int inc_x, DevMatrix &y, const int inc_y)
{
	const int n = y.size();
	cublasHandle_t hdl;
 	cublasStatus_t status = cublasCreate_v2(&hdl);
	cublasSaxpy(hdl, n, &alpha, x.begin() , inc_x, y.begin(), inc_y);
	//printf("rf %d info %d\n", status);
	cublasDestroy_v2(hdl);
}

__device__ void saxpy(float alpha, const float &x, const int inc_x, DevMatrix &y, const int inc_y)
{
	const int n = y.nrow();

	cublasHandle_t hdl;
 	cublasStatus_t status = cublasCreate_v2(&hdl);
	status = cublasSaxpy(hdl, n, &alpha, &x, inc_x, y.begin(), inc_y);
	//__syncthreads();
 	//printf("rf %d info %d\n", status);
 	cublasDestroy_v2(hdl);
}

__device__ void sgemv(cublasOperation_t trans, const float alpha, DevMatrix &A, DevMatrix &x, int inc_x, const float beta, DevMatrix &y, const int inc_y)
{
	int M = A.nrow();
	int N = A.ncol();

	int LDA = A.nrow();

	cublasHandle_t hdl;
	cublasStatus_t status = cublasCreate_v2(&hdl);
	cublasSgemv(hdl, trans, M, N, &alpha, A.begin(), LDA, x.begin(), inc_x, &beta, y.begin(), inc_y); 
	//printf("rf %d info %d\n", status);
	cublasDestroy_v2(hdl);
}

__device__ void sgemv(cublasOperation_t trans, const float alpha, DevMatrix &A, const float &x, const int inc_x, const float beta, DevMatrix &y, const int inc_y)
{
	int M = A.nrow();
	int N = A.ncol();

	int LDA = A.nrow();

	cublasHandle_t hdl;
	cublasStatus_t status = cublasCreate_v2(&hdl);
	cublasSgemv(hdl, trans, M, N, &alpha, A.begin(), LDA, &x, inc_x, &beta, y.begin(), inc_y); 
	//printf("rf %d info %d\n", status);
	cublasDestroy_v2(hdl);
}

// k = 1 and LDA = 1 for a diagonal matrix (0 = n_super = n_lower), stored columnwise in a 1 x N vector where N is the number of columns of A

__device__ void ssbmv(cublasFillMode_t uplo, const float alpha, DevMatrix &A, const int K, DevMatrix &x, const int inc_x, const float beta, DevMatrix &y, const int inc_y)
{
//cublasFillmode_t literal example CUBLAS_FILL_MODE_LOWER
	int N = A.nrow();
	int LDA = 1;
	cublasHandle_t hdl;
	cublasStatus_t status = cublasCreate_v2(&hdl);
	cublasSsbmv(hdl, uplo, N, K, &alpha, A.begin(), LDA, x.begin(), inc_x, &beta, y.begin(), inc_y); 
	//printf("rd %d info %d\n", status);
	cublasDestroy_v2(hdl);
}

//A := alpha*x*y**T + A 

__device__ void sger(const float alpha, DevMatrix &x, const int inc_x, const float &y, const int inc_y, DevMatrix &A)
{
	int M  = A.nrow();
	int N  = A.ncol();

	int LDA = A.nrow();
	cublasHandle_t hdl;
	cublasStatus_t status = cublasCreate_v2(&hdl);
	cublasSger(hdl, M, N, &alpha, x.begin(), inc_x, &y, inc_y, A.begin(), LDA);
	cublasDestroy_v2(hdl);
}

//A := alpha*x*y**T + A 
__device__ void sger(const float alpha, DevMatrix  &x, const int inc_x, DevMatrix &y, const int inc_y, DevMatrix &A)
{
	int M  = A.nrow();
	int N  = A.ncol();

	int LDA = A.nrow();
	cublasHandle_t hdl;
	cublasStatus_t status = cublasCreate_v2(&hdl);
	cublasSger(hdl, M, N, &alpha, x.begin(), inc_x, y.begin(), inc_y, A.begin(), LDA);
	cublasDestroy_v2(hdl);
}

__device__ void sgemm(cublasOperation_t TrA, cublasOperation_t TrB, float alpha, DevMatrix &A, DevMatrix &B, float beta, DevMatrix &C)

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

	cublasHandle_t hdl;
 	cublasStatus_t status = cublasCreate_v2(&hdl);
	
	cublasSgemm(hdl,TrA, TrB, M, N, K, &alpha, &*A.begin(), LDA, &*B.begin(), LDB, &beta, &*C.begin(), LDC);
	
	cublasDestroy_v2(hdl);
	
}
