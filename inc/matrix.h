#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <string.h>
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

class Matrix : public std::vector<double>
{
	private:
		size_t n_row;
		size_t n_col;	

	public:
		
		
		Matrix();
		Matrix(size_t, size_t);
		Matrix(size_t, size_t, double);
		Matrix(Matrix &);

		void reserve(size_t, size_t);
		void resize(size_t, size_t);
		void clear();

		size_t nrow();
		size_t ncol();

		void nrow(size_t);
		void ncol(size_t);

		double & operator()(size_t , size_t);

		//const double & operator()(size_t , size_t);

		void copy(Matrix &);

		void swap(Matrix &);
		
		void writeToFile(std::string, int prec);
	
		void print();
		
		void initialize(long, double , double );

		void convertToColumnMajor(Matrix &);

		void identity();
		void clearMemory();
};

extern "C"
{
	// Level 1
	void daxpy_(const int *N, const double *ALPHA, const double *X, const int *INCX, double *Y, const int *INCY);

	// Level 2
	void dgemv_(const char *TRANSA, const int *M, const int *N, const double *ALPHA, const double *A, const int *LDA, const double *X, const int *INCX, const double *BETA, double *Y, const int *INCY);
	void dsbmv_(const char *UPLO, const int *N, const int *K, const double *ALPHA, const double *A, const int *LDA, const double *X, const int *INCX, const double *BETA, double *Y, const int *INCY);
	void dger_ (const int *M, const int *N, const double *ALPHA, const double *X, const int *INCX, const double *Y, const int *INCY, double *A, const int *LDA);
	void dgemm_(const char *TRANSA, const char *TRANSB, const int *M, const int *N, const int *K, const double *ALPHA, const double *A, const int *LDA, const double *B, const int *LDB, const double *BETA, const double *C, const int *LDC);
}

//y  <--  alpha*x + y
void daxpy(const double, const std::vector<double> &, const int, std::vector<double> &, const int);

void daxpy(double alpha, const double &x, const int inc_x, std::vector<double> &y, const int inc_y);

void dgemv(const char, const double, Matrix &, const std::vector<double> &, const int, const double, std::vector<double> &, const int);

void dgemv(const char, const double, Matrix &, const double &, const int, const double, std::vector<double> &, const int);

// k = 1 and LDA = 1 for a diagonal matrix (0 = n_super = n_lower), stored columnwise in a 1 x N vector where N is the number of columns of A

void dsbmv(const char, const double, Matrix &, const int, const std::vector<double> &, const int, const double, std::vector<double> &, const int);

//A := alpha*x*y**T + A 
void dger(const double, const std::vector<double>  &, const int, const double &, const int, Matrix &);

//A := alpha*x*y**T + A 
void dger(const double, const std::vector<double> &, const int, const std::vector<double> &, const int, Matrix &);

void dgemm(const char, const char, double, Matrix &, Matrix &, double, Matrix &);

#endif

