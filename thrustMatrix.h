#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>

#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublas_api.h>
// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

class cuHandle
{
	public:
		static cublasHandle_t handle;

};
cublasHandle_t cuHandle::handle = NULL;

struct prg
{
    float a, b;

    __host__ __device__
    prg(float _a=0.f, float _b=1.f) : a(_a), b(_b) {};

    __host__ __device__
        float operator()(const unsigned int n) const
        {
            thrust::default_random_engine rng;
            thrust::uniform_real_distribution<float> dist(a, b);
            rng.discard(n);

            return dist(rng);
        }
};



class Matrix : public thrust::device_vector<float>
{
	private:
		size_t n_row;
		size_t n_col;	

	public:
		
		
		   Matrix() : thrust::device_vector<float>(NULL), n_row(0), n_col(0) {}
		   Matrix(size_t m, size_t n) : thrust::device_vector<float>(m*n), n_row(m), n_col(n) {}
		   Matrix(size_t m, size_t n, float entries) : thrust::device_vector<float>(m*n, entries), n_row(m), n_col(n) {}
		   Matrix(const Matrix &Y) : thrust::device_vector<float>(Y), n_row(Y.nrow()), n_col(Y.ncol()) {} 

		   void reserve(size_t m, size_t n)
			{thrust::device_vector<float>reserve(m*n);}
		   void resize(size_t m, size_t n)
			{n_row=m; n_col=n; thrust::device_vector<float>resize(m*n);}
		   void clear()
			{n_row=0; n_col=0; thrust::device_vector<float>clear();}

		   size_t nrow() const {return n_row;}
		   size_t ncol() const {return n_col;}

		   void nrow(size_t i){ this->n_row = i;}
		   void ncol(size_t j){ this->n_col = j;}

		   void write(size_t i, size_t j, float x)
		{
			(*this)[i + j*n_row] = x;
		};
		   float read(size_t i, size_t j)
		{
			return (*this)[i + j*n_row];
		};

		   void copy(const Matrix &Y)
		{
			std::copy(Y.thrust::device_vector<float>::begin(), Y.thrust::device_vector<float>::end(), this->thrust::device_vector<float>::begin());
			n_row = Y.nrow();
			n_col = Y.ncol();
		}

		   void swap(Matrix &Y)
		{
			thrust::swap(*this, Y);
			thrust::swap(n_row, Y.n_row);
			thrust::swap(n_col, Y.n_col);
		};
		
	/*	void writeToFile(std::string filename, int prec=5){
			std::ofstream myfile(filename.c_str(), std::ios::trunc);
			for(int i = 0; i < n_row; i++){
				for(int j = 0; j < n_col; j++){
					
					myfile << std::setprecision(prec) << (*this)(i,j); 
					if(j < n_col-1)
						myfile << " ";
					else
						myfile << "\n";
				}
			}		
			myfile.close();

		};*/
	
		  void print(){
			
			if(n_row == 0 || n_col == 0){
				std::cout << "not initialized" << std::endl;
				return;
			}
			for(int i = 0; i < n_row; i++){
				for(int j = 0; j < n_col; j++){
						
				//	std::cout.width(10);
				//	std::cout << std::fixed << std::showpoint;
					std::cout << /*std::left << std::setprecision(5)  <<*/ read(i,j) << " "; 	
				}
				std::cout << std::endl;
			}			
		};
		
		  void initialize(){
			
			thrust::counting_iterator<unsigned int> index_sequence_begin(0);
			thrust::transform(index_sequence_begin,
            		index_sequence_begin + this->size(),
            		this->begin(),
            		prg(-1.f,1.f));	
			
		};

		  void convertToColumnMajor(Matrix &X){
			Matrix A(X.nrow(), X.ncol());
			
			for( int i = 0; i< X.nrow(); i++){
				for( int j = 0 ; j < X.ncol(); j++){
					write(i, j, X[i*X.ncol() + j]);
				}
			}
			X.swap(A);
		};

		  void identity(){
			if( this->n_row != this->n_col){
				std::cout << " identity(): Not a square matrix" << std::endl;
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
		};
	
		  void clearMemory()
			{
			Matrix empty;
			swap(empty);
		};
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

void saxpy(const float alpha, const thrust::device_vector<float> &x, const int inc_x, thrust::device_vector<float> &y, const int inc_y)
{
	const int n = y.size();

	cublasSaxpy(cuHandle::handle, n, &alpha, thrust::raw_pointer_cast(&x[0]), inc_x, thrust::raw_pointer_cast(&y[0]), inc_y);
}
/*
void daxpy(float alpha, const float &x, const int inc_x, thrust::device_vector<float> &y, const int inc_y)
{
	const int n = y.size();

	cublasDaxpy(cuHandle::handle, n, &alpha, &x, inc_x, &*y.begin(), inc_y);
}
*/
void sgemv(cublasOperation_t trans, const float alpha, const Matrix &A, const thrust::device_vector<float> &x, const int inc_x, const float beta, thrust::device_vector<float> &y, const int inc_y)
{
	int M = A.nrow();
	int N = A.ncol();

	int LDA = A.nrow();

	cublasSgemv(cuHandle::handle, trans, M, N, &alpha, thrust::raw_pointer_cast(&A[0]), LDA, thrust::raw_pointer_cast(&x[0]) , inc_x, &beta, thrust::raw_pointer_cast(&y[0]), inc_y); 
}
/*
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
