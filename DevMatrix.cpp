#include "DevMatrix.h"

__device__ DevMatrix::DevMatrix() : M(NULL), n_row(0), n_col(0) {;}
__device__ DevMatrix::DevMatrix(size_t m, size_t n) {
	n_row = m;
	n_col = n;
	devStates = new curandState[1];
	M = new float[m*n];
}
__device__ DevMatrix::~DevMatrix()
{
	n_row=n_col=0;
	delete[] M;
	delete devStates;
	devStates = (curandState *) NULL;
	M = (float *) NULL;
}

__host__ __device__ float * DevMatrix::getM(){ return M; }

__host__ void DevMatrix::fill(float x)
{
	//int jobs = n_row*n_col;
	//kernelFill<<<1, jobs>>>(x, getM());
}

__host__ __device__ int DevMatrix::size() { return n_row*n_col;}
__host__ __device__ int DevMatrix::nrow() { return n_row; }
__host__ __device__ int DevMatrix::ncol() { return n_col; }

__device__ float * DevMatrix::begin()
{
	return M;
}

 __device__ float * DevMatrix::end()
{
	return M + (n_row*n_col);
}

__device__ void DevMatrix::write(size_t i, size_t j, float x)
{
	M[i + j*n_row] = x;
}
__host__ __device__ float DevMatrix::read(size_t i, size_t j)
{
	return M[i + j*n_row];
}

__device__ void DevMatrix::print(){

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

__device__ void DevMatrix::identity()
{
	if( this->n_row != this->n_col){
		printf("empty matrix\n");
		return;
	}else{
		for(int i = 0; i < this->n_row; i++){
			for(int j = 0; j < this->n_col; j++){
				if(i == j)
					write(i, j, 2.0);
				else
					write(i, j, 0.0);
			}
		}
	}
}

__device__ void DevMatrix::copy(DevMatrix &X){
	cublasHandle_t hdl;
	cublasStatus_t status = cublasCreate_v2(&hdl);
	cublasScopy(hdl, X.nrow(), X.getM(), 1, getM(), 1);
	cublasDestroy_v2(hdl);
}

/* kernel used in random initialization of matrices */
__global__ void genNorm(unsigned long seed, float *M, float mean = 0, float std = 1)
{
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        //printf("%d\n", id);
        curandState state;
        curand_init(seed, id, 0, &state);
        /* Generate pseudo-random normals */
        M[id] = curand_normal( &state)*std + mean;
        //__syncthreads();
}

__device__ void DevMatrix::initialize(unsigned long seed= 1234, float mean = 0, float std = 1)
{
        //printf("in initialize\n");
        genNorm<<<1, size()>>>(seed, M, mean, std);
        cudaDeviceSynchronize();
}

/* cublas interface */

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
