#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <string>
#include <sstream>
#include <fstream>

#pragma once

void read(std::string, char, thrust::host_vector<float> &, int &, int &);

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

		Data(); 
		Data(std::string, char, int);

		int nrsp(); 
		int nfea();
		int nrow(); 
		int ncol(); 

};

