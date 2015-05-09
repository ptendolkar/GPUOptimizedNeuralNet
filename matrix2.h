#include <vector>

template <class T>
class Matrix : public std::vector<T>
{
	private:
		size_t n_row;
		size_t n_col;

	public:
		Matrix() : n_row(0), n_col(0) {}
		Matrix(size_t m, size_t n) : std::vector<T>(m*n), n_row(m), n_col(n) {}
		Matrix(size_t m, size_t n, T entries) : std::vector<T>(m*n, entries), n_row(m), n_col(n) {}
		Matrix(const Matrix &Y) : std::vector<T>(Y), n_row(Y.nrow()), n_col(Y.ncol()) {} 

		void reserve(size_t m, size_t n)
			{std::vector<T>reserve(m*n);}
		void resize(size_t m, size_t n)
			{n_row=m; n_col=n; std::vector<T>resize(m*n);}
		void clear()
			{n_row=0; n_col=0; std::vector<T>clear();}

		size_t nrow() const {return n_row;}
		size_t ncol() const {return n_col;}

		void nrow(size_t m) {n_row = m;}
		void ncol(size_t n) {n_col = n;}

		T & operator()(size_t i, size_t j)
		{
			return std::vector<T>::operator[](i + j*n_row);
		};
		const T & operator()(size_t i, size_t j) const
		{
			return std::vector<T>::operator[](i + j*n_row);
		};

		void copy(const Matrix<T> &Y)
		{
			std::copy(Y.std::vector<T>::begin(), Y.std::vector<T>::end(), this->std::vector<T>::begin());
			n_row = Y.nrow();
			n_col = Y.ncol();
		}

		void swap(Matrix<T> &Y)
		{
			std::vector<T>::swap(Y);
			std::swap(n_row, Y.n_row);
			std::swap(n_col, Y.n_col);
		};

		void clearMemory()
		{
			Matrix<T> empty;
			swap(empty);
		};
};

extern "C"
{
	// Level 1
	void daxpy_(const int *N, const float *ALPHA, const float *X, const int *INCX, float *Y, const int *INCY);

	// Level 2
	void dgemv_(const char *TRANSA, const int *M, const int *N, const float *ALPHA, const float *A, const int *LDA, const float *X, const int *INCX, const float *BETA, float *Y, const int *INCY);
	void dsbmv_(const char *UPLO, const int *N, const int *K, const float *ALPHA, const float *A, const int *LDA, const float *X, const int *INCX, const float *BETA, float *Y, const int *INCY);
	void dger_ (const int *M, const int *N, const float *ALPHA, const float *X, const int *INCX, const float *Y, const int *INCY, float *A, const int *LDA);
	void dgemm_(const char *TRANSA, const char *TRANSB, const int *M, const int *N, const int *K, const float *ALPHA, const float *A, const int *LDA, const float *B, const int *LDB, const float *BETA, const float *C, const int *LDC);
}

template <class T>
void daxpy(const float alpha, const std::vector<T> &x, const int inc_x, std::vector<T> &y, const int inc_y)
{
	const int N = x.size();

	daxpy_(&N, &alpha, (float *)&*x.begin(), &inc_x, (float *)&*y.begin(), &inc_y);
}

template <class T>
void dgemv(const char TrA, const float alpha, const Matrix<T> &A, const std::vector<T> &x, const int inc_x, const float beta, std::vector<T> &y, const int inc_y)
{
	int M = A.nrow();
	int N = A.ncol();

	int LDA = A.nrow();

	dgemv_(&TrA, &M, &N, &alpha, (float*)&*A.std::vector<T>::begin(), &LDA, (float*)&*x.begin(), &inc_x, &beta, (float*)&*y.begin(), &inc_y); 
}

// k = 1 and LDA = 1 for a diagonal matrix (0 = n_super = n_lower), stored columnwise in a 1 x N vector where N is the number of columns of A

template <class T>
void dsbmv(const char UPLO, const float alpha, const Matrix<T> &A, const int K, const std::vector<T> &x, const int inc_x, const float beta, std::vector<T> &y, const int inc_y)
{
	int N = A.ncol();

	int LDA = A.nrow();

	dsbmv_(&UPLO, &N, &K, &alpha, (float *)&*A.std::vector<T>::begin(), &LDA, (float *)&*x.begin(), &inc_x, &beta, (float *)&*y.begin(), &inc_y); 
}

template <class T>
void dger(const float alpha, const std::vector<T> &x, const int inc_x, const std::vector<T> &y, const int inc_y, Matrix<T> &A)
{
	int M  = A.nrow();
	int N  = A.ncol();

	int LDA = A.nrow();

	dger_(&M, &N, &alpha, (float *)&*x.begin(), &inc_x, (float *)&*y.begin(), &inc_y, (float *)&*A.std::vector<T>::begin(), &LDA);
}

template <class T>
void dgemm(const char TrA, const char TrB, const float alpha, const Matrix<T> &A, const Matrix<T> &B, const float beta, Matrix<T> &C)
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

	dgemm_(&TrA, &TrB, &M, &N, &K, &alpha, (float *)&*A.std::vector<T>::begin(), &LDA, (float *)&*B.std::vector<T>::begin(), &LDB, &beta, (float *)&*C.std::vector<T>::begin(), &LDC);
} 
