#include <vector>
#include <string>
#include <sstream>
#include <fstream>

/* Reads data matrix (observations along rows and features along columns) and stores in row-major format */

void read(std::string data_file, char delim, std::vector<double> &X, size_t &m, size_t &n)
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
			X.push_back(x);
		}

		++i;
	}

	m = i;
	n = j;

	return;
};

class Data
{
	private:
		size_t n_row;
		size_t n_col;
		size_t n_rsp;
		size_t n_fea;

		std::vector<double> X;

	public:
		Data() : X(), n_row(0), n_col(0), n_rsp(0), n_fea(0) {}

		size_t nrsp() { return n_rsp; }
		size_t nfea() { return n_fea; }

		double* resp(size_t obs_id) { return &X[obs_id*n_col]; }
		double* feat(size_t obs_id) { return &X[obs_id*n_col + n_rsp]; }

		Data(std::string data_file, char delim, size_t d)
		{
			read(data_file, delim, X, n_row, n_col);

			n_rsp = d;
			n_fea = n_col - n_rsp;
		}
};
