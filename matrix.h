#include <vector>
#include <string.h>
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

class Matrix : public std::vector<double>
{
	private:
		size_t n_row;
		size_t n_col;	

	public:
		
		
		Matrix() : n_row(0), n_col(0) {}
		Matrix(size_t m, size_t n) : std::vector<double>(m*n), n_row(m), n_col(n) {}
		Matrix(size_t m, size_t n, double entries) : std::vector<double>(m*n, entries), n_row(m), n_col(n) {}

		void reserve(size_t m, size_t n)
			{std::vector<double>reserve(m*n);}
		void resize(size_t m, size_t n)
			{n_row=m; n_col=n; std::vector<double>resize(m*n);}
		void clear()
			{n_row=0; n_col=0; std::vector<double>clear();}

		size_t nrow() const {return n_row;}
		size_t ncol() const {return n_col;}

		void nrow(size_t i){ this->n_row = i;}
		void ncol(size_t j){ this->n_col = j;}

		double & operator()(size_t i, size_t j)
		{
			return operator[](i + j*n_row);
		};
		const double & operator()(size_t i, size_t j) const
		{
			return operator[](i + j*n_row);
		};

		void swap(Matrix &Y)
		{
			std::vector<double>::swap(Y);
			std::swap(n_row, Y.n_row);
			std::swap(n_col, Y.n_col);
		};
	
		void print(){
			
			if(n_row == 0 || n_col == 0){
				std::cout << "not initialized" << std::endl;
				return;
			}
			std::cout << "Rows: " << this->n_row << ", Cols: " << this->n_col << std::endl;
			for(int i = 0; i < n_row; i++){
				for(int j = 0; j < n_col; j++){
						
					std::cout.width(10);
					std::cout << std::fixed << std::showpoint;
					std::cout << std::left << std::setprecision(5)  << (*this)(i,j) << " "; 	
				}
				std::cout << std::endl;
			}			
		};
		
		void initialize(double mean = 0, double sigma = 1){

			const gsl_rng_type * T;
			gsl_rng * r;
			
			r = gsl_rng_alloc(gsl_rng_mt19937);

			for( int i = 0; i < n_row*n_col; i++ ){
				
		
				(*this)[i] = gsl_ran_gaussian(r, sigma);
				(*this)[i] = mean + (*this)[i];

			}
				gsl_rng_free(r);
		};

		void convertToColumnMajor(Matrix &X){
			Matrix A(X.nrow(), X.ncol());
			
			for( int i = 0; i< X.nrow(); i++){
				for( int j = 0 ; j < X.ncol(); j++){
					A(i,j) = X[i*X.ncol() + j];
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
							(*this)(i,j) = 1;
						else
							(*this)(i,j) = 0;
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

extern "C"

{
	void dgemm_(const char *TrA,const  char *TrB, size_t *m, size_t *n, size_t *k, double *alpha, double *A, size_t *LDA, double *B, size_t *LDB, double *beta, double *C, size_t *LDC);
	
	void dgemv_(const char *TrA, size_t *m, size_t *n, double *alpha, double *A, size_t *LDA, double *x, size_t *inc_x, double *beta, double *y, size_t *inc_y);

	void daxpy_(int *n, double *a, double *x, int *inc_x, double *y, int *inc_y);

	void dger_ (size_t *m, size_t *n, double *a, double *x, int *inc_x, double *y, int *inc_y, double *A, size_t *LDA);
	
	void dsbmv_(const char *uplo, const int *n, const int *k,const double *alpha, const double *a, const int *lda, const double *x, const int *incx, const double *beta,double *y,const int *incy);
}
// y := alpha*A*x + beta*y
void dsbmv(std::vector<double> &a, std::vector<double> &x, std::vector<double> &y){
	int 	k 	= 0;
	double 	alpha 	= 1.0;
	int 	lda 	= 1;
	int 	incx	= 1;
	double 	beta 	= 0.0;
	int 	incy	= 1;
 	int 	n	= a.size();
	
	dsbmv_("L", &n, &k, &alpha, &*a.begin() , &lda, &*x.begin(), &incx, &beta, &*y.begin(), &incy);
}

// A := alpha*x*y' + A
void dger(double alpha, std::vector<double> &x, int inc_x, std::vector<double> &y, int inc_y, Matrix &A)
{
	size_t M = A.nrow();
	size_t N = A.ncol();

	size_t LDA = A.nrow();

	dger_(&M, &N, &alpha, &*x.begin(), &inc_x, &*y.begin(), &inc_y, &*A.begin(), &LDA);

	return;
}

// y := alpha*x + y
void daxpy(double a, std::vector<double> &x, int inc_x, std::vector<double> &y, int inc_y){
	
	int n = x.size();
	
	daxpy_( &n, &a, &*x.begin(), &inc_x, &*y.begin(), &inc_y);

	return;
}



void dgemm(const char *TrA, const char *TrB, double alpha, Matrix &A, Matrix &B, double beta, Matrix &C)
{
	size_t M;
	size_t N;
	size_t K;

	size_t LDA = A.nrow();
	size_t LDB = B.nrow();
	size_t LDC = C.nrow();

	switch(*TrA)
	{
		case 'N':
		{
			switch(*TrB)
			{
				case 'N':
				{
						M = A.nrow();
						N = B.ncol();
						K = B.nrow();
				}
				case 'T':
				{
						M = A.nrow();
						N = B.nrow();
						K = B.ncol();
				}
			}
		}
		case 'T':
		{
			switch(*TrB)
			{
				case 'N':
				{
						M = A.ncol();
						N = B.ncol();
						K = B.nrow();
				}
				case 'T':
				{
						M = A.ncol();
						N = B.nrow();
						K = B.ncol();
				}
			}
		}
	}

	dgemm_(TrA, TrB, &M, &N, &K, &alpha, &*A.begin(), &LDA, &*B.begin(), &LDB, &beta, &*C.begin(), &LDC);

	return;
} 

// y := alpha*A*x + beta*y
void dgemv(const char *TrA, double alpha, Matrix &A, Matrix &x, double beta, Matrix &y)
{
	size_t M;
	size_t N;
	size_t LDA = A.nrow();
	size_t LDx = x.nrow();
	size_t LDy = y.nrow();

	switch(*TrA)
	{		
	case 'N':
	{
		M = A.nrow();
		N = A.ncol();
	}
	case 'T':
	{
		M = A.ncol();
		N = A.nrow();
	}
	}

	dgemv_(TrA, &M, &N, &alpha, &*A.begin(), &LDA, &*x.begin(), &LDx, &beta, &*y.begin(), &LDy);
	return;
}


