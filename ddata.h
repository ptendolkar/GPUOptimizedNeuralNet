#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <string>
#include <sstream>
#include <fstream>

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

class Data
{
	private:
		int n_row;
		int n_col;
		int n_rsp;
		int n_fea;

		thrust::host_vector<float> h_X;
	public:
		thrust::device_vector<float> X;
		Data() : X(), n_row(0), n_col(0), n_rsp(0), n_fea(0) {}

		Data(std::string data_file, char delim, int d)
		{
			read(data_file, delim, h_X, n_row, n_col);
			X = h_X;	
			
//			thrust::copy(X.begin(),X.end(), std::ostream_iterator<float>(std::cout, " "));
//			std::cout << std::endl;

			n_rsp = d;
			n_fea = n_col - n_rsp;
		}

		int nrsp() { return n_rsp; }
		int nfea() { return n_fea; }
		int nrow() { return n_row; }
		int ncol() { return n_col; }

};

class DevData
{
	private:
		int n_row;
		int n_col;
		int n_rsp;
		int n_fea;

		float *X;
	public:
		__device__ DevData() : X(), n_row(0), n_col(0), n_rsp(0), n_fea(0) {}

		__device__ int nrsp() { return n_rsp; }
		__device__ int nfea() { return n_fea; }
		__device__ int nrow() { return n_row; }
		__device__ int ncol() { return n_col; }

		__device__ float * resp(int obs_id) { return (&X[obs_id*n_col]); }
		__device__ float * feat(int obs_id) { return (&X[obs_id*n_col + n_rsp]); }

		__device__ DevData( float *dX, int row, int col, int rsp, int fea )
		{
			X = dX;
			n_row = row;
			n_col = col;
			n_rsp = rsp;
			n_fea = fea;	
		}

};
