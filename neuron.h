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
	Matrix() : n_row(0), n_col(0) {}
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
		Layer   *next_lay;
		Layer   *prev_lay;
		Funct   *Phi;

	public:
		Layer() : Matrix(), iden(0), next_lay(NULL), prev_lay(NULL), Phi(NULL) {}
		Layer(size_t i, size_t m, size_t n) : Matrix(m,n), iden(i), next_lay(NULL), prev_lay(NULL), Phi(NULL) {}
		Layer(size_t i, size_t m, size_t n, Layer *inn, Layer *ipp) : Matrix(m,n), iden(i), next_lay(inn), prev_lay(ipp), Phi(NULL) {}
		Layer(size_t i, size_t m, size_t n, double w, Layer *inn, Layer *ipp) : Matrix(m,n,w), iden(i), next_lay(inn), prev_lay(ipp), Phi(NULL) {}
		Layer(size_t i, size_t m, size_t n, double w, Layer *inn, Layer *ipp, Funct *f) : Matrix(m,n,w), iden(i), next_lay(inn), prev_lay(ipp), Phi(f) {}

		size_t id()    const { return iden; }
		Layer *next()  const { return next_lay; }
		Layer *prev()  const { return prev_lay; }
		Funct *activ() const { return Phi; }

		Layer(const Layer &lay)
		{
			iden     = lay.id();
			next_lay = lay.next();
			prev_lay = lay.prev();
			Phi      = lay.activ();	
		};

		Layer id(size_t i)     { iden     = i; }
		Layer next(Layer *lay) { next_lay = lay; }
		Layer prev(Layer *lay) { prev_lay = lay; }
		Layer set_activ(Funct *Psi) { Phi = Psi; }

		void swap(Layer &lay)
		{
			Matrix::swap(lay);
			std::swap(iden, lay.iden);
			std::swap(next_lay, lay.next_lay);
			std::swap(prev_lay, lay.prev_lay);
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
		Layer  *out_lay;
		Layer  *inp_lay;

	public:
		Network() : n_lay(0), out_lay(NULL), inp_lay(NULL) {}
		Network(size_t i) : n_lay(i), out_lay(NULL), inp_lay(NULL) {}

		size_t depth() const { return n_lay; }
		Layer *out()   const { return out_lay; }
		Layer *inp()   const { return inp_lay; }

		Network(const Network &net)
		{
			n_lay = net.depth();
			out_lay = net.out();
			inp_lay = net.inp();
		}

		~Network()
		{
			n_lay   = 0;
			out_lay = NULL;
			inp_lay = NULL;
		};

		Layer depth(size_t i) { n_lay = i; }
		Layer inp(Layer *lay) { inp_lay = lay; }
		Layer out(Layer *lay) { out_lay = lay; }

		int build(std::vector<size_t> &);
		int clear();
		int remove(size_t);
		int insert(size_t, size_t);
};

// Build network dynamically backwards (head to tail) from the output layer.  Single layer network (e.g. logistic regression) will have NULL input layer pointer,
// but all networks must have an output.  The first entry of the dimension array is the size of the covariate space.
int Network::build(std::vector<size_t> &dim_lay)
{
	Layer *prev_ptr = NULL;
	Layer *next_ptr = NULL;

	this->n_lay   = dim_lay.size();

	int i = (this->n_lay) - 1;
	this->out_lay = new Layer((size_t)i, 1, dim_lay[i]);
	next_ptr      = this->out_lay;

	for (int j=i; j>0; j--)
	{
		prev_ptr = new Layer((size_t)j, dim_lay[j], dim_lay[j-1]);
		prev_ptr->next(next_ptr);
		next_ptr->prev(prev_ptr);
		next_ptr = prev_ptr;
	}

	this->inp_lay = prev_ptr;

	return 0;
}

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
}

// Delete layer from existing network.
int Network::remove(size_t id)
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
}

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
}

class Data
{
	private
		// n_data is the number of data, n_feat is the size of the feature space
		size_t n_data;
		size_t n_feat;
		Matrix X(n_data, n_feat);
		Matrix y(n_data, 1);

	public
		read(char *, Matrix &);

		make(char *feat_file, char *resp_file)
		{
			Matrix X();
			Matrix y();
			read(feat_file, X);
			read(resp_file, y);
		}; 
}

Class::read(char *data_file, Matrix &A)
{
	std::fstream input(data_file);
	std::string  line;
	int i = 0;
	int j = 0;

	while (std::getline(input, line))
	{
		double a;
		std::stringstream ss(line);

		while (ss >> a)
		{
			if (i == 0)
			{
				++j;
			}
			A.push_back(a);
		}

		++i;
	}

	A.nrow(i);
	A.ncol(i);
}; 
