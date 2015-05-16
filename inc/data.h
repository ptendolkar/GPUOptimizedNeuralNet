#ifndef DATA_H
#define DATA_H

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <stdlib.h>

/* Reads data matrix (observations along rows and features along columns) and stores in row-major format */

class Data
{
	private:
		size_t n_row;
		size_t n_col;
		size_t n_rsp;
		size_t n_fea;

		std::vector<double> X;

	public:
		Data();

		size_t nrsp();
		size_t nfea();
		size_t nrow();

		double* resp(size_t);
		double* feat(size_t);

		Data(std::string, char, size_t);

};

#endif
