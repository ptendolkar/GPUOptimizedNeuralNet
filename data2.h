#include <vector>
#include <string>
#include <sstream>
#include <fstream>

/* Reads data matrix (observations along rows and features along columns) and stores in row-major format */

template <typename T>
T convt_str(const std::string &s)
{
	std::istringstream iss(s);
	T convt_s;
	if (!(iss >> convt_s)) throw std::bad_alloc("Error in conversion."); 
	return convt_s;
}

template <typename T>
void read(std::string data_file, char delim, std::vector<T> &X, size_t &m, size_t &n)
{
	std::ifstream input(data_file);
	std::string   line;
	size_t i = 0;
	size_t j = 0;

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
			x = convt_str(item);
			X.push_back(x);
		}

		++i;
	}

	m = i;
	n = j;
};

template <typename T>
class Data
{
	private:
		size_t n_row;
		size_t n_col;
		size_t n_rsp;
		size_t n_fea;
		std::vector<T> X;

	public:
		Data() : X(), n_row(0), n_col(0), n_rsp(0), n_fea(0) {}

		Data (std::string data_file, char delim, size_t d)
		{
			n_row = 0;
			n_col = 0;
			n_fea = d;
			std::vector<T> X;
	
			read(data_file, delim, X, n_row, n_col);
			n_fea = n_col - n_rsp;
		}

		size_t nrsp() { return n_rsp; }
		size_t nfea() { return n_fea; }

		T* resp(size_t obs_id) { return &X[obs_id*n_col]; }
		T* feat(size_t obs_id) { return &X[obs_id*n_col + n_rsp]; }
};
