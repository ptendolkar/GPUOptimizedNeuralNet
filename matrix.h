#include <vector>

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

		void nrow(size_t m) {n_row = m;}
		void ncol(size_t n) {n_col = n;}

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

		void clearMemory()
		{
			Matrix empty;
			swap(empty);
		};
};

extern "C"
{
	void daxpy_(int *n, double *alpha, double *x, int inc_x, double *y, int *inc_y);
	void dgemv_(const char *TrA, int *m, int *n, double *alpha, double *A, int *LDA, double *x, int *inc_x, double *beta, double *y, int *inc_y);
	void dger_ (int *m, int *n, double *alpha, double *x, int *inc_x, double *y, int *inc_y, double *A, int *LDA);
	void dgemm_(const char *TrA, const char *TrB, int *m, int *n, int *k, double *alpha, double *A, int *LDA, double *B, int *LDB, double *beta, double *C, int *LDC);
}

void daxpy(double alpha, std::vector<double> &x, int inc_x, std::vector<double> &y, int inc_y)
{
	size_t n = x.size();

	daxpy_(&n, &alpha, &*x.begin(), &inc_x, &*y.begin(), &inc_y);

	return;
}

void dger(double alpha, std::vector<double> &x, int inc_x, std::vector<double> &y, int inc_y, Matrix &A)
{
	size_t M = A.nrow();
	size_t N = A.ncol();

	size_t LDA = A.nrow();

	dger_(&M, &N, &alpha, &*x.begin(), &inc_x, &*y.begin(), &inc_y, &*A.begin(), &LDA);

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
			switch(TrB)
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

void md_mult(const char *TrA, double alpha, Matrix &A, std::vector<double> &x, double beta, Matrix &y, size_t obs_id, size_t n_feat)
{
	size_t M;
	size_t N;

	size_t LDA = A.nrow();
	size_t LDB = n_feat;
	size_t LDC = C.nrow();

	switch(TrA)
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

	dgemv_(TrA, &M, &N, &alpha, &*A.begin(), &LDA, &*(B.begin() + obs_id*n_feat), &LDB, &beta, &*C.begin(), &LDC);

	return;
} 
