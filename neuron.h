//#include "matrix.h"
//#class Matrix;
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

class Matrix : public std::vector<double>
{
private:
	size_t n_row;
	size_t n_col;

public:
	Matrix() : std::vector<double>(), n_row(0), n_col(0) {}
	Matrix(size_t m, size_t n) : std::vector<double>(m*n), n_row(m), n_col(n) {}
	Matrix(size_t m, size_t n, double entries) : std::vector<double>(m*n, entries), n_row(m), n_col(n) {}

	void reserve(size_t m, size_t n)
		{std::vector<double>reserve(m*n);}
	void resize(size_t m, size_t n)
		{n_row=m; n_col=n; std::vector<double>resize(m*n);}
	void clear()
		{n_row=0; n_col=0; std::vector<double>clear();}

	size_t nrow() const { return n_row; }
	size_t ncol() const { return n_col; }

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

class Funct 
{
	private:
		double (*fun)(double);
		double (*grd)(double);

	public:
		Funct() : fun(NULL), grd(NULL) {}
		Funct(double (*Phi)(double)) : fun(Phi) {}
		Funct(double (*Phi)(double), double (*Psi)(double)) : fun(Phi), grd(Psi) {}

		~Funct()
		{
			fun = NULL;
			grd = NULL;
		};

		double (*get_fun())(double) { return fun; }
		double (*get_grd())(double) { return grd; }
};

class Layer : public Matrix
{
	private:
		size_t  iden;
		Layer   *prev_lay;
		Layer   *next_lay;
		std::vector<Funct *> Phi;

	public:
		Layer() : Matrix(), iden(0), prev_lay((Layer *)NULL), next_lay((Layer *)NULL), Phi() {}
		Layer(size_t i, size_t m, size_t n) : Matrix(m,n), iden(i), prev_lay((Layer *)NULL), next_lay((Layer *)NULL), Phi(m, (Funct *)NULL) {}
		Layer(size_t i, size_t m, size_t n, Layer *ipp) : Matrix(m,n), iden(i), prev_lay(ipp), next_lay((Layer *)NULL), Phi(1, (Funct *)NULL) {}
		Layer(size_t i, size_t m, size_t n, double w, Layer *ipp, Layer *inn) : Matrix(m,n,w), iden(i), prev_lay(ipp), next_lay(inn), Phi(m, (Funct *)NULL) {}
		Layer(size_t i, size_t m, size_t n, double w, Layer *ipp, Layer *inn, Funct *f) : Matrix(m,n,w), iden(i), prev_lay(ipp), next_lay(inn), Phi(1, f) {}

		Layer(size_t i, size_t m, size_t n, std::vector<double> &w, Layer *ipp, Layer *inn, std::vector<Funct *> &f)
		{
			iden = i;
			prev_lay = ipp;
			next_lay = inn;
			Matrix(m,n);
			std::vector<double>::swap(w);
			std::vector<Funct *> Phi(m);
			Phi.std::vector<Funct *>::swap(f);
		}

		size_t id()    const { return iden; }
		Layer *prev()  const { return prev_lay; }
		Layer *next()  const { return next_lay; }
		std::vector<Funct *> activ() const { return Phi; }

		Layer(const Layer &lay)
		{
			iden     = lay.id();
			prev_lay = lay.prev();
			next_lay = lay.next();
			Phi      = lay.activ();	
		};

		void id(size_t i)     { iden     = i; }
		void prev(Layer *lay) { prev_lay = lay; }
		void next(Layer *lay) { next_lay = lay; }
		void activ(std::vector<Funct *> Psi) { Phi = Psi; }

		void swap(Layer &lay)
		{
			Matrix::swap(lay);
			std::swap(iden, lay.iden);
			std::swap(prev_lay, lay.prev_lay);
			std::swap(next_lay, lay.next_lay);
			Phi.std::vector<Funct *>::swap(lay.Phi);
		};

		void clearMemory()
		{
			Layer empty;
			swap(empty);
		};
};

class Network
{
	private:
		size_t n_lay;
		Layer  *inp_lay;
		Layer  *out_lay;

	public:
		Network() : n_lay(0), inp_lay((Layer *)NULL), out_lay((Layer *)NULL) {}
		Network(size_t i) : n_lay(i), inp_lay((Layer *)NULL), out_lay((Layer *)NULL) {}
		Network(std::vector<size_t> &);

		size_t depth() const { return n_lay; }
		Layer *inp()   const { return inp_lay; }
		Layer *out()   const { return out_lay; }

		Network(const Network &net)
		{
			n_lay = net.depth();
			inp_lay = net.inp();
			out_lay = net.out();
		}

		~Network()
		{
			n_lay   = 0;
			inp_lay = (Layer *)NULL;
			out_lay = (Layer *)NULL;
		};

		Layer depth(size_t i) { n_lay = i; }
		Layer inp(Layer *lay) { inp_lay = lay; }
		Layer out(Layer *lay) { out_lay = lay; }

		int clear();
		int remove(size_t);
		int insert(size_t, size_t);
};

// Build network dynamically backwards (head to tail) from the output layer.  Single layer network (e.g. logistic regression) will have NULL input layer pointer,
// but all networks must have an output.  The first entry of the dimension array is the size of the covariate space, and the last entry is the size of the output space.
Network::Network(std::vector<size_t> &dim_lay)
{
	Layer *prev_ptr = (Layer *)NULL;
	Layer *curn_ptr = (Layer *)NULL;

	int head_id = dim_lay.size()-1;

	for (int i=head_id; i>0; i--)
	{
		curn_ptr = new Layer(i-1, dim_lay[i], dim_lay[i-1], prev_ptr);

		if (out_lay == (Layer *)NULL)
		{
			out_lay = curn_ptr;
		}
		else
		{
			(curn_ptr->prev())->next(curn_ptr);
			inp_lay = curn_ptr;
		}

		prev_ptr = curn_ptr;
	}
};

// Clear dynamically built network backwards.
int Network::clear()
{
	Layer *prev_ptr = NULL;
	Layer *next_ptr = NULL;

	prev_ptr = (this->out_lay)->prev();
	(this->out_lay)->Layer::clearMemory();

	for (int i=1; i<(this->n_lay); i++)
	{
		next_ptr = prev_ptr;
		prev_ptr = next_ptr->prev();
		next_ptr->Layer::clearMemory();
	}

	this->inp_lay = NULL;	
	this->n_lay   = 0;

	return 0;
};

// Delete layer from existing network.
/* int Network::remove(size_t id)
{
	if ((this->inp_lay != (Layer *)NULL) & (id <= (this->out())->ncol()))
	{
		Layer *curn_ptr = (this->inp_lay)->next();
		while (curn_ptr->id() != id)
		{
			curn_ptr = curn_ptr->next();
		}

		Layer *prev_ptr  = curn_ptr->prev();
		Layer *next_ptr  = curn_ptr->next();
		Layer *pprev_ptr = prev_ptr->prev();
		size_t d_out     = curn_ptr->nrow();

		curn_ptr->clearMemory();
		curn_ptr = new Layer(prev_ptr->id(), d_out, prev_ptr->ncol(), next_ptr, pprev_ptr); 
		prev_ptr->clearMemory();

		if (pprev_ptr)
		{
			pprev_ptr->next(curn_ptr);
		}
		else
		{
			this->inp(curn_ptr);
		}

		if (next_ptr)
		{
			next_ptr->prev(curn_ptr);
		}
		else
		{
			this->out(curn_ptr);
		}

		while (next_ptr)
		{
			next_ptr->id(next_ptr->id()-1);
			next_ptr = next_ptr->next();
		}
	}
	else
	{
		std::cout << "Illegal delete: id is outside of possible range." << std::endl;
	}

	return 0;
};

// Insert layer into existing network.
int Network::insert(size_t postn, size_t d_inp)
{
	if ((this->inp_lay != (Layer *)NULL) & !(postn > this->depth()))
	{
		Layer *curn_ptr = (this->inp_lay)->next();
		while (curn_ptr->id() != postn)
		{
			curn_ptr = curn_ptr->next();
		}

		Layer *prev_ptr = curn_ptr->prev();
		Layer *next_ptr = curn_ptr;
		Layer *pprev_ptr = prev_ptr->prev();
		Layer *nnext_ptr = next_ptr->next();

		curn_ptr = new Layer(postn, next_ptr->ncol(), d_inp);

		prev_ptr->Layer::clearMemory();
		prev_ptr = new Layer((postn-1), d_inp, pprev_ptr->nrow(), curn_ptr, pprev_ptr);

		next_ptr->Layer::clearMemory();
		next_ptr = new Layer((postn+1), nnext_ptr->ncol(), curn_ptr->nrow(), nnext_ptr, curn_ptr);

		if (pprev_ptr)
		{
			pprev_ptr->next(prev_ptr);
		}
		else
		{
			this->inp(prev_ptr);
		}

		if (nnext_ptr)
		{
			nnext_ptr->prev(next_ptr);
		}
		else
		{
			this->out(next_ptr);
		}

		next_ptr = nnext_ptr;
		while (next_ptr)
		{
			next_ptr->id(next_ptr->id()+1);
			next_ptr = next_ptr->next();
		}
	}
	else
	{
		std::cout << "Illegal insert: position is outside of possible range." << std::endl;
	}

	return 0;
}; */

class Data
{
	private:
		size_t n_data;
		size_t n_feat;
		Matrix X;
		Matrix y;

	public:
		Data (std::string, std::string, char);

		int read(std::string, char, Matrix &);
		int read(std::string, char, Matrix &, size_t &, size_t &);
};

Data::Data (std::string feat_file, std::string resp_file, char delim)
{
	Matrix X;
	Matrix y;
	n_data = 0;
	n_feat = 0;
	read(feat_file, delim, X, n_data, n_feat);
	read(resp_file, delim, y);
};

int Data::read(std::string data_file, char delim, Matrix &A)
{
	std::fstream input(data_file.c_str());
	std::string  line;
	int i = 0;
	int j = 0;

	std::vector<double> X;

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

	A.std::vector<double>::swap(X);

	return 0;
};

int Data::read(std::string data_file, char delim, Matrix &A, size_t &n_row, size_t &n_col)
{
	std::fstream input(data_file.c_str());
	std::string  line;
	int i = 0;
	int j = 0;

	std::vector<double> X;

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

	A.std::vector<double>::swap(X);
	n_row = i;
	n_col = j;

	return 0;
};
