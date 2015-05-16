#include <matrix.h>

Matrix::Matrix() : n_row(0), n_col(0) {}
Matrix::Matrix(size_t m, size_t n) : std::vector<double>(m*n), n_row(m), n_col(n) {}
Matrix::Matrix(size_t m, size_t n, double entries) : std::vector<double>(m*n, entries), n_row(m), n_col(n) {}
Matrix::Matrix(Matrix &Y) : std::vector<double>(Y), n_row(Y.nrow()), n_col(Y.ncol()) {} 

void Matrix::reserve(size_t m, size_t n)
	{std::vector<double>reserve(m*n);}
void Matrix::resize(size_t m, size_t n)
	{n_row=m; n_col=n; std::vector<double>resize(m*n);}
void Matrix::clear()
	{n_row=0; n_col=0; std::vector<double>clear();}

size_t Matrix::nrow() {return n_row;}
size_t Matrix::ncol() {return n_col;}

void Matrix::nrow(size_t i){ this->n_row = i;}
void Matrix::ncol(size_t j){ this->n_col = j;}

double & Matrix::operator()(size_t i, size_t j)
{
	return operator[](i + j*n_row);
};

/*const double & Matrix::operator()(size_t i, size_t j) const
{
	return operator[](i + j*n_row);
}
*/

void Matrix::copy(Matrix &Y)
{
	std::copy(Y.std::vector<double>::begin(), Y.std::vector<double>::end(), this->std::vector<double>::begin());
	n_row = Y.nrow();
	n_col = Y.ncol();
}

void Matrix::swap(Matrix &Y)
{
	std::vector<double>::swap(Y);
	std::swap(n_row, Y.n_row);
	std::swap(n_col, Y.n_col);
}

void Matrix::writeToFile(std::string filename, int prec=5){
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

}

void Matrix::print(){
	
	if(n_row == 0 || n_col == 0){
		std::cout << "not initialized" << std::endl;
		return;
	}
//	std::cout << "Rows: " << this->n_row << ", Cols: " << this->n_col << std::endl;
	for(int i = 0; i < n_row; i++){
		for(int j = 0; j < n_col; j++){
				
			std::cout.width(10);
			std::cout << std::fixed << std::showpoint;
			std::cout << std::left << std::setprecision(5)  << (*this)(i,j) << " "; 	
		}
		std::cout << std::endl;
	}			
}

void Matrix::initialize(long seed, double mean = 0, double sigma = 1){

	const gsl_rng_type * T;
	gsl_rng * r;
	
	r = gsl_rng_alloc(gsl_rng_mt19937);
	gsl_rng_set(r, seed);	
	for( int i = 0; i < n_row*n_col; i++ ){	

		(*this)[i] = gsl_ran_gaussian(r, sigma);
		(*this)[i] = mean + (*this)[i];

	}
		gsl_rng_free(r);
}

void Matrix::convertToColumnMajor(Matrix &X){
	Matrix A(X.nrow(), X.ncol());
	
	for( int i = 0; i< X.nrow(); i++){
		for( int j = 0 ; j < X.ncol(); j++){
			A(i,j) = X[i*X.ncol() + j];
		}
	}
	X.swap(A);
}

void Matrix::identity(){
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
}
void Matrix::clearMemory()
{
	Matrix empty;
	swap(empty);
}


//y  <--  alpha*x + y
void daxpy(const double alpha, const std::vector<double> &x, const int inc_x, std::vector<double> &y, const int inc_y)
{
	const int n = y.size();

	daxpy_(&n, &alpha, &*x.begin(), &inc_x, &*y.begin(), &inc_y);
}

void daxpy(double alpha, const double &x, const int inc_x, std::vector<double> &y, const int inc_y)
{
	const int n = y.size();

	daxpy_(&n, &alpha, &x, &inc_x, &*y.begin(), &inc_y);
}

void dgemv(const char TrA, const double alpha, Matrix &A, const std::vector<double> &x, const int inc_x, const double beta, std::vector<double> &y, const int inc_y)
{
	int M = A.nrow();
	int N = A.ncol();

	int LDA = A.nrow();

	dgemv_(&TrA, &M, &N, &alpha, &*A.std::vector<double>::begin(), &LDA, &*x.begin() , &inc_x, &beta, &*y.begin(), &inc_y); 
}

void dgemv(const char TrA, const double alpha, Matrix &A, const double &x, const int inc_x, const double beta, std::vector<double> &y, const int inc_y)
{
	int M = A.nrow();
	int N = A.ncol();

	int LDA = A.nrow();

	dgemv_(&TrA, &M, &N, &alpha, &*A.begin(), &LDA, &x, &inc_x, &beta, &*y.begin(), &inc_y); 
}

// k = 1 and LDA = 1 for a diagonal matrix (0 = n_super = n_lower), stored columnwise in a 1 x N vector where N is the number of columns of A

void dsbmv(const char UPLO, const double alpha, Matrix &A, const int K, const std::vector<double> &x, const int inc_x, const double beta, std::vector<double> &y, const int inc_y)
{
	int N = A.nrow();
	int LDA = 1;

	dsbmv_(&UPLO, &N, &K, &alpha, &*A.std::vector<double>::begin(), &LDA, &*x.begin(), &inc_x, &beta, &*y.begin(), &inc_y); 
}

//A := alpha*x*y**T + A 
void dger(const double alpha, const std::vector<double>  &x, const int inc_x, const double &y, const int inc_y, Matrix &A)
{
	int M  = A.nrow();
	int N  = A.ncol();

	int LDA = A.nrow();

	dger_(&M, &N, &alpha, &*x.begin(), &inc_x, &y, &inc_y, &*A.std::vector<double>::begin(), &LDA);
}

//A := alpha*x*y**T + A 
void dger(const double alpha, const std::vector<double> &x, const int inc_x, const std::vector<double> &y, const int inc_y, Matrix &A)
{
	int M  = A.nrow();
	int N  = A.ncol();

	int LDA = A.nrow();

	dger_(&M, &N, &alpha, &*x.begin(), &inc_x, &*y.begin(), &inc_y, &*A.std::vector<double>::begin(), &LDA);
}
void dgemm(const char TrA, const char TrB, double alpha, Matrix &A, Matrix &B, double beta, Matrix &C)

{
	int M;
	int N;
	int K;

	int LDA = A.nrow();
	int LDB = B.nrow();
	int LDC = C.nrow();

	switch(TrA)
	{
		case 'N':
		{
			switch(TrB)
			{
				case 'N':
				{
						M = A.nrow();
						N = B.ncol();
						K = B.nrow();
						break;	
				}
				case 'T':
				{
						M = A.nrow();
						N = B.nrow();
						K = B.ncol();
						break;
				}
			}
			break;
		}
		case 'T':
		{
			switch(TrB)
			{
				case 'N':
				{
						M = A.ncol();
						N = B.ncol();
						K = B.nrow();
						break;
				}
				case 'T':
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


	dgemm_(&TrA, &TrB, &M, &N, &K, &alpha, &*A.std::vector<double>::begin(), &LDA, &*B.std::vector<double>::begin(), &LDB, &beta, &*C.std::vector<double>::begin(), &LDC);
} 
