#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include "neuron.h"

/* int read(std::string data_file, char delim, std::vector<double> &X, size_t &m, size_t &n)
{
	std::fstream input(data_file.c_str());
	std::string  line;
	int i = 0;
	int j = 0;

	while (std::getline(input, line))
	{
		double x;
		std::stringstream ss(line);
		std::string item;

		while (std::getline(ss, item, delim))
		{
			if (i == 0)
			{
				++j;
			}
			x = atof(item.c_str());
			X.std::vector<double>::push_back(x);
		}

		++i;
	}

	m = i;
	n = j;

	return 0;
};

int read(std::string data_file, char delim, std::vector<double> &X)
{
	std::fstream input(data_file.c_str());
	std::string  line;
	int i = 0;
	int j = 0;

	while (std::getline(input, line))
	{
		double x;
		std::stringstream ss(line);
		std::string item;

		while (std::getline(ss, item, delim))
		{
			if (i == 0)
			{
				++j;
			}
			x = atof(item.c_str());
			X.std::vector<double>::push_back(x);
		}

		++i;
	}

	return 0;
}; */

void dgemm(const char *TrA, const char *TrB, double alpha, Matrix &A, std::vector<double> &b, double beta, Matrix &C, size_t i, size_t n_feat)
{
	int m = A.nrow();
	int k = A.ncol();
	int n = B.ncol();

	int LDA = A.nrow();
	int LDB = B.nrow();
	int LDC = C.nrow();

	size_t j = i*n_feat;
	
	if(*TrA == 'N' && *TrB == 'N'){
		dgemm_(TrA, TrB, &m, &n, &k, &alpha, &*A.begin(), &LDA, &*(B.begin()+j), &LDB, &beta, &*C.begin(), &LDC);	
	}else if(*TrA == 'T' && *TrB == 'N'){
		dgemm_(TrA, TrB, &k, &n, &m, &alpha, &*A.begin(), &LDA, &*(B.begin()+j), &LDB, &beta, &*C.begin(), &LDC);
	}else if(*TrA == 'N' && *TrB == 'T'){
		dgemm_(TrA, TrB, &m, &k, &k, &alpha, &*A.begin(), &LDA, &*(B.begin()+j), &LDB, &beta, &*C.begin(), &LDC);
	}else if(*TrA == 'T' && *TrB == 'T'){
		dgemm_(TrA, TrB, &k, &m, &k, &alpha, &*A.begin(), &LDA, &*(B.begin()+j), &LDB, &beta, &*C.begin(), &LDC);
	}else{
		std::cout << "dgemm(): use only \"N\" or \"T\" flags for TrA and TrB" << std::endl;
	}
	return;
} 

/* class Data
{
	public:
		size_t n_data;
		size_t n_feat;
		std::vector<double> X;
		std::vector<double> y;
		Data() : X(), y(), n_data(0), n_feat(0) {}

		Data (std::string input_file, std::string label_file, char delim)
		{
			read(input_file, delim, X, n_data, n_feat);
			read(label_file, delim, y);
		}
}; */

double dtanh (double x) {
	return (1 - pow(tanh(x), 2));
}

double sqloss(double x) {
	return pow(x, 2);
}

double dsqloss(double x) {
	return 2*x;
}

int main(int argc, char* argv[])
{
	Funct Phi(&tanh, &dtanh);
	Funct L(&sqloss, &dsqloss);

	std::cout << (*(Phi.get_fun()))(2.0) << std::endl;
	
	
//	Data test("test_X.csv", "test_y.csv", ',');
	Matrix A(4,4);
	A.identity();
	
	std::vector<double> x(8);
	x[0] = x[1] = x[2] = x[3] = 1;
	x[4] = x[5] = x[6] = x[7] = 2;

	Data test;
	test.X = x;
	test.n_feat = 4;

	Layer L0(4,4);
	L0.w_swp(A);

	L0.push(1, test);	
	
	/*Matrix C(4,1);
	int m = 4;
	int n = 1;
	int k = 4;
	int LDA = 4;
	int LDB = 4;
	int LDC = 4;

	double alpha = 1.0;	

	dgemm_("N","N", &m, &n, &k, &alpha, &*A.begin(), &LDA, &*(x.begin()+4), &LDB, &alpha, &*C.begin(), &LDC);

	C.print();*/
	
/*	Layer L1(1, 4, 5);
	Layer L0(0, 5, 1);
	
	
	L1.f(0, &Phi);
	Matrix x1(5,1);
	x1.initialize();
	L0.a_swp(x1);

	Matrix x2(4,5);
	x2.initialize();
	L1.w_swp(x2);

	Matrix x3(4,1);
	x3.initialize();
	L1.b_swp(x3);
	
	L0.print();
	L1.push(L0);
	std::cout << std::endl;
	L1.print();*/

    return 0;
}
