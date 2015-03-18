#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include "neuron.h"

int read(std::string data_file, char delim, Matrix &X, size_t &m, size_t &n)
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

int read(std::string data_file, char delim, Matrix &X)
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
};


class Data
{
	public:
		size_t n_data;
		size_t n_feat;
		Matrix X;
		Matrix y;
		Data() : X(), y(), n_data(0), n_feat(0) {}

		Data (std::string input_file, std::string label_file, char delim)
		{
			read(input_file, delim, X, n_data, n_feat);
			X.nrow(n_data);
			X.ncol(n_feat);
			read(label_file, delim, y);
			y.nrow(n_data);
			y.ncol(1);
		}
};

int main(int argc, char* argv[])
{
	
	Data test("test_X.csv", "test_y.csv", ',');
	std::cout << (test.X).std::vector<double>::size() << std::endl;

    return 0;
}
