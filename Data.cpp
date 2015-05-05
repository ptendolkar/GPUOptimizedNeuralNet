#include "Data.h"

/* Reads data matrix (observations along rows and features along columns) and stores in row-major format */

void read(std::string data_file, char delim, thrust::host_vector<float> &X, int &m, int &n)
{
	std::fstream input(data_file.c_str());
	std::string  line;
	int i = 0;
	int j = 0;

	while (std::getline(input, line))
	{
		float x;
		std::stringstream ss(line);
		std::string item;

		while (std::getline(ss, item, delim))
		{
			if (i == 0)
			{
				++j;
			}
			x = atof(item.c_str());
			X.push_back(x);
		}

		++i;
	}

	m = i;
	n = j;
};

Data::Data() : X(), n_row(0), n_col(0), n_rsp(0), n_fea(0) {}

Data::Data(std::string data_file, char delim, int d)
{
	read(data_file, delim, h_X, n_row, n_col);
	X = h_X;	
	

	n_rsp = d;
	n_fea = n_col - n_rsp;
}

int Data::nrsp() { return n_rsp; }
int Data::nfea() { return n_fea; }
int Data::nrow() { return n_row; }
int Data::ncol() { return n_col; }
