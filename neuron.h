#include "matrix.h"
#include "data.h"

class Matrix;
class Data;

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
		Matrix  bias;
		Matrix  flux;
		Matrix  actv;
		std::vector<Funct *> potn;

	public:
		Layer() : Matrix(), iden(0), prev_lay((Layer *)NULL), next_lay((Layer *)NULL), bias(), flux(), actv(), potn() {}
		Layer(size_t i, size_t m, size_t n) : Matrix(m,n), iden(i), prev_lay((Layer *)NULL), next_lay((Layer *)NULL), bias(m,1), flux(m,1), actv(m,1), potn() {}
		Layer(size_t i, size_t m, size_t n, Layer *ipp, Layer *inn)	: Matrix(m,n), iden(i), prev_lay(ipp), next_lay(inn), bias(m,1), flux(m,1), actv(m,1), potn() {}
		Layer(size_t i, size_t m, size_t n, Layer *ipp, Layer *inn, Funct *f) : Matrix(m,n), iden(i), prev_lay(ipp), next_lay(inn), bias(m,1), flux(m,1), actv(m,1), potn(1,f) {}

		size_t id()   const { return iden; }
		Layer* prev() const { return prev_lay; }
		Layer* next() const { return next_lay; }

		Matrix* w() { return (Matrix*) this; }
		Matrix* b() { return &bias; }
		Matrix* z() { return &flux; }
		Matrix* a() { return &actv; }
		std::vector<Funct *>* f() { return &potn; }

		void id(size_t i)     { iden = i; }
		void prev(Layer *lay) { prev_lay = lay; }
		void next(Layer *lay) { next_lay = lay; }
		void f(size_t i, Funct *Phi) { potn[i] = Phi; }

		void w_swp(std::vector<double > &x)   { this->std::vector<double>::swap(x); }
		void b_swp(std::vector<double > &x)   { bias.std::vector<double >::swap(x); }
		void z_swp(std::vector<double > &x)   { flux.std::vector<double >::swap(x); }
		void a_swp(std::vector<double > &x)   { actv.std::vector<double >::swap(x); }
		void f_swp(std::vector<Funct *> &Phi) { potn.std::vector<Funct *>::swap(Phi); }

		void swap(Layer &lay)
		{
			this->Matrix::swap(lay);
			std::swap(iden, lay.iden);
			std::swap(prev_lay, lay.prev_lay);
			std::swap(next_lay, lay.next_lay);
			bias.std::vector<double >::swap(lay.bias);
			flux.std::vector<double >::swap(lay.flux);
			actv.std::vector<double >::swap(lay.actv);
			potn.std::vector<Funct *>::swap(lay.potn);
		};

		void clearMemory()
		{
			Layer empty;
			swap(empty);
		};

		void push(size_t, Data *);
		void push(Layer &);
};

void Layer::push(Layer &L)
{
	z_swp(bias);
	dgemm('N', 'N', 1.0, *w(), *(L.a()), 1.0, flux);

	if (potn.size() == 1)
	{
		if (potn[0] == (Funct *)NULL)
		{
			actv = flux;
		}
		else
		{
			for (int i = 0; i< actv.size(); i++)
			{	
				actv[i] = (*(potn[0]->get_fun()))(flux[i]);
			}
		}
	}
	else
	{
		for (int i = 0; i < actv.size(); i++)
		{
			if (potn[i] != (Funct *)NULL)
			{	
				actv[i] = (*(potn[i]->get_fun()))(flux[i]);
			}
			else
			{
				actv[i] = flux[i];
			}
		}
	}

	return;
}

void Layer::push(size_t obs_id, Data *dat_add)
{
	z_swp(bias);
	MASU_mult('N', 1.0, *w(), dat_add->X, obs_id, 1.0, flux);

	if (potn.size() == 1)
	{
		if (potn[0] == (Funct *)NULL)
		{
			actv = flux;
		}
		else
		{
			for (int i = 0; i< actv.size(); i++)
			{	
				actv[i] = (*(potn[0]->get_fun()))(flux[i]);
			}
		}
	}
	else
	{
		for (int i = 0; i < actv.size(); i++)
		{
			if (potn[i] != (Funct *)NULL)
			{	
				actv[i] = (*(potn[i]->get_fun()))(flux[i]);
			}
			else
			{
				actv[i] = flux[i];
			}
		}
	}

	return;
}

class Network
{
	private:
		size_t n_lay;
		Layer  *inp_lay;
		Layer  *out_lay;
		Data   *data;

	public:
		Network() :         n_lay(0), inp_lay((Layer *)NULL), out_lay((Layer *)NULL) {}
		Network(size_t i) : n_lay(i), inp_lay((Layer *)NULL), out_lay((Layer *)NULL) {}

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

		void build(std::vector<size_t> &);
		void clear();
		int  remove(size_t);
		int  insert(size_t, size_t);

		void feed_foward(size_t);
};

// Build network dynamically backwards (head to tail) from the output layer.  Single layer network (e.g. logistic regression) will have NULL input layer pointer,
// but all networks must have an output.  The first entry of the dimension array is the size of the covariate space, and the last entry is the size of the output space.
void Network::build(std::vector<size_t> &dim_lay, Funct *f)
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
		curn_ptr = new Layer(i-1, dim_lay[i], dim_lay[i-1], prev_ptr, (Layer *)NULL, f);

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
void Network::clear()
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
};

// Delete layer from existing network, by iterating backwards.
void Network::remove(size_t id)
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

	std::vector<Funct *> curn_potn = *(next_ptr->f());

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

	std::vector<Funct *> prev_potn = *(prev_ptr->f());

	prev_ptr->clearMemory();

	prev_ptr = new Layer(postn-1, n_new, n_inp, pprev_ptr, (Layer *)NULL);
	next_ptr = new Layer(postn,   n_out, n_new, prev_ptr,  nnext_ptr);

	prev_ptr->f_swp(prev_potn);

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

void Network::feed_foward(size_t obs_id)
{
	Layer *curn_lay;

	if (inp_lay != (Layer *)NULL)
	{
		curn_lay = inp_lay;
	}
	else
	{
		curn_lay = out_lay;
	}

	curn_lay->push(obs_id, data);
	curn_lay = curn_lay->next();


	while (curn_lay != (Layer *)NULL)
	{
		curn_lay->push(*(curn_lay->prev()));
		curn_lay = curn_lay->next();
	}
};
