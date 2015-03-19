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
	void dgemm_(const char *TrA,const  char *TrB, int *m, int *n, int *k, double *alpha, double *A, int *LDA, double *B, int *LDB, double *beta, double *C, int *LDC);
}

void dgemm(const char *TrA,const char *TrB, double alpha, Matrix &A, Matrix &B, double beta, Matrix &C)
{
	int m = A.nrow();
	int k = A.ncol();
	int n = B.ncol();

	int LDA = A.nrow();
	int LDB = B.nrow();
	int LDC = C.nrow();
	
	if(*TrA == 'N' && *TrB == 'N'){
		dgemm_(TrA, TrB, &m, &n, &k, &alpha, &*A.begin(), &LDA, &*B.begin(), &LDB, &beta, &*C.begin(), &LDC);	
	}else if(*TrA == 'T' && *TrB == 'N'){
		dgemm_(TrA, TrB, &k, &n, &m, &alpha, &*A.begin(), &LDA, &*B.begin(), &LDB, &beta, &*C.begin(), &LDC);
	}else if(*TrA == 'N' && *TrB == 'T'){
		dgemm_(TrA, TrB, &m, &k, &k, &alpha, &*A.begin(), &LDA, &*B.begin(), &LDB, &beta, &*C.begin(), &LDC);
	}else if(*TrA == 'T' && *TrB == 'T'){
		dgemm_(TrA, TrB, &m, &n, &k, &alpha, &*A.begin(), &LDA, &*B.begin(), &LDB, &beta, &*C.begin(), &LDC);
	}else{
		std::cout << "dgemm(): use only \"N\" or \"T\" flags for TrA and TrB" << std::endl;
	}
	return;
} 
