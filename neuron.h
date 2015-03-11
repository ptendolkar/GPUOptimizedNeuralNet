#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include "matrix.h"

#class Matrix;

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
		std::vector<Funct *> potn;
		std::vector<double>  bias;
		std::vector<double>  flux;
		std::vector<double>  actv;

	public:
		Layer() :													  Matrix(),    iden(0), prev_lay((Layer *)NULL), next_lay((Layer *)NULL), potn(), 				, bias(),  actv()  {}
		Layer(size_t i, size_t m, size_t n) :                         Matrix(m,n), iden(i), prev_lay((Layer *)NULL), next_lay((Layer *)NULL), potn(m, (Funct *)NULL), bias(n), actv(m) {}
		Layer(size_t i, size_t m, size_t n, Layer *ipp) :             Matrix(m,n), iden(i), prev_lay(ipp),           next_lay((Layer *)NULL), potn(m, (Funct *)NULL), bias(n), actv(m) {}
		Layer(size_t i, size_t m, size_t n, Layer *ipp, Layer *inn) : Matrix(m,n), iden(i), prev_lay(ipp),           next_lay(inn),           potn(m, (Funct *)NULL), bias(n), actv(m) {}

		Layer(size_t i, size_t m, size_t n, std::vector<double> &w, Layer *ipp, Layer *inn, std::vector<Funct *> &f, std::vector<double> &z)
		{
			iden = i;
			prev_lay = ipp;
			next_lay = inn;
			Matrix(m,n);
			std::vector<Funct *> potn(m);
			std::vector<double > bias(m);
			std::vector<double > flux(m);
			std::vector<double > actv(m);
			std::vector<double >::swap(w);
			poten.std::vector<Funct *>::swap(f);
			activ.std::vector<double >::swap(z);
		}

		Layer(const Layer &lay)
		{
			Matrix   = lay.Matrix();
			iden     = lay.id();
			prev_lay = lay.prev();
			next_lay = lay.next();
			potn     = lay.potn;
			bias     = lay.bias;
			flux     = lay.flux;
			actv     = lay.actv;
		};

		size_t id()    const { return iden; }
		Layer* prev()  const { return prev_lay; }
		Layer* next()  const { return next_lay; }
		Funct* f(size_t i) const { return potn[i]; }
		std::vector<double>  W() const { return (vector)Matrix; }
		std::vector<Funct *> f() const { return potn; }
		std::vector<double > b() const { return bias; }
		std::vector<double > z() const { return flux; }
		std::vector<double > a() const { return actv; }

		void W(std::vector<double> X) { Matrix = X; }
		void id(size_t i)     { iden     = i; }
		void prev(Layer *lay) { prev_lay = lay; }
		void next(Layer *lay) { next_lay = lay; }
		void f(size_t i, Funct *Phi) { potn[i] = Phi; }
		void b(size_t i, double x)   { bias[i] = x; }
		void z(size_t i, double x)   { flux[i] = x; }
		void a(size_t i, double x)   { actv[i] = x; }
		void f(std::vector<Funct *> Phi) { potn = Phi; }
		void b(std::vector<double > x)   { bias = x; }
		void z(std::vector<double > x)   { flux = x; }
		void a(std::vector<double > x)   { actv = x; }
		void f_swp(std::vector<Funct *> &Phi) { potn.std::vector<Funct *>::swap(Phi); }
		void b_swp(std::vector<double > &x)   { bias.std::vector<double >::swap(x); }
		void z_swp(std::vector<double > &x)   { flux.std::vector<double >::swap(x); }
		void a_swp(std::vector<double > &x)   { actv.std::vector<double >::swap(x); }

		void swap(Layer &lay)
		{
			Matrix::swap(lay);
			std::swap(iden, lay.iden);
			std::swap(prev_lay, lay.prev_lay);
			std::swap(next_lay, lay.next_lay);
			potn.std::vector<Funct *>::swap(lay.potn);
			bias.std::vector<double >::swap(lay.bias);
			actv.std::vector<double >::swap(lay.actv);
		};

		void clearMemory()
		{
			Layer empty;
			swap(empty);
		};
};

/* int Layer::push(std::vector<double> &z)
{
	std::vector<double> x(this->nrow());
	// multiply x = W*z;
	for (std::vector<int>::const_iterator i = poten.begin(); i != poten.end(); i++)
	{
		if (poten(*i) != (Funct *)NULL)
		{
			x(*i) = (double *(poten(*i)->fun))(x(*i));
		}
	}
	activ(x);
} */

class Network
{
	private:
		size_t n_lay;
		Layer  *inp_lay;
		Layer  *out_lay;

	public:
		Network() : 		n_lay(0), inp_lay((Layer *)NULL), out_lay((Layer *)NULL) {}
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
	if (dim_lay.size() < 2)
	{
		std::cout << "Insufficient parameters to create a network." << std::endl;
		return;
	}

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
	Layer *curn_ptr = out_lay;
	Layer *prev_ptr = curn_ptr->prev();
	curn_ptr->Layer::clearMemory();

	while (prev_ptr != (Layer *)NULL)
	{
		curn_ptr = prev_ptr;
		prev_ptr = curn_ptr->prev();
		curn_ptr->Layer::clearMemory();
	}

	inp_lay = prev_ptr;
	n_lay   = 0;

	return 0;
};

// Delete layer from existing network, by iterating backwards.
int Network::remove(size_t id)
{
	if ((id <= 0) | (id >= n_lay))
	{
		std::cout << "Illegal delete: id is outside of possible range." << std::endl;
		return 1;
	}

	Layer *next_ptr = out_lay;
	Layer *prev_ptr = next_ptr->prev();

	while (next_ptr->id() != id)
	{
		next_ptr->id(next_ptr->id() - 1);
		next_ptr = prev_ptr;
		prev_ptr = next_ptr->prev();
	}

	Layer *pprev_ptr = prev_ptr->prev();
	Layer *nnext_ptr = next_ptr->next();

	size_t n_inp = prev_ptr->ncol();
	size_t n_out = next_ptr->nrow();

	prev_ptr->clearMemory();
	next_ptr->clearMemory();

	Layer *curn_ptr = new Layer(id-1, n_out, n_inp, pprev_ptr, nnext_ptr);

	if (pprev_ptr != (Layer *)NULL)
	{
		pprev_ptr->next(curn_ptr);
	}
	else
	{
		inp_lay = curn_ptr;
	}

	if (nnext_ptr != (Layer *)NULL)
	{
		nnext_ptr->prev(curn_ptr);
	}
	else
	{
		out_lay = curn_ptr;
	}

	return 0;
};

// Insert layer into existing network.
int Network::insert(size_t postn, size_t n_new)
{
	if ((postn <= 0) | (postn >= n_lay )) 
	{
		std::cout << "Illegal insert: position is outside of possible range." << std::endl;
		return 1;
	}

	Layer *next_ptr = out_lay;
	Layer *prev_ptr = next_ptr->prev();

	while (next_ptr->id() != postn)
	{
		next_ptr->id(next_ptr->id() + 1);
		next_ptr = prev_ptr;
		prev_ptr = next_ptr->prev();
	}

	Layer *pprev_ptr = prev_ptr->prev();
	Layer *nnext_ptr = next_ptr;

	nnext_ptr->id(nnext_ptr->id() + 1);

	size_t n_inp = prev_ptr->nrow();
	size_t n_out = nnext_ptr->ncol();

	prev_ptr->clearMemory();

	prev_ptr = new Layer(postn-1, n_new, n_inp, pprev_ptr);
	next_ptr = new Layer(postn, n_out, n_new, prev_ptr, nnext_ptr);

	prev_ptr->next(next_ptr);
	nnext_ptr->prev(next_ptr);

	if (pprev_ptr != (Layer *)NULL)
	{
		pprev_ptr->next(prev_ptr);
	}
	else
	{
		inp_lay = prev_ptr;
	}

	return 0;
};

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
