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
	void dgemm_(const char *TrA, const char *TrB, int *m, int *n, int *k, double *alpha, double *A, int *LDA, double *B, int *LDB, double *beta, double *C, int *LDC);
}

void dgemm(const char *TrA, const char *TrB, double alpha, Matrix &A, Matrix &B, double beta, Matrix &C)
{
	int m = A.nrow();
	int k = A.ncol();
	int n = B.ncol();

	int LDA = A.nrow();
	int LDB = B.nrow();
	int LDC = C.nrow();

	dgemm_(TrA, TrB, &m, &n, &k, &alpha, &*A.begin(), &LDA, &*B.begin(), &LDB, &beta, &*C.begin(), &LDC);

	return;
} 

void md_mult(const char *TrA, const char *TrB, double alpha, Matrix &A, std::vector<double> &B, double beta, Matrix &C, size_t obs_id, size_t n_feat)
{
	int m = A.nrow();
	int k = A.ncol();
	int n = 1;

	int LDA = A.nrow();
	int LDB = n_feat;
	int LDC = C.nrow();

	dgemm_(TrA, TrB, &m, &n, &k, &alpha, &*A.begin(), &LDA, &*(B.begin() + obs_id*n_feat), &LDB, &beta, &*C.begin(), &LDC);

	return;
} 
