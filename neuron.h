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
		Funct(double (*f)(double), double (*g)(double)) : fun(f), grd(g) {}

		~Funct()
		{
			fun = NULL;
			grd = NULL;
		};

		double (*get_fun())(double) const { return fun; }
		double (*get_grd())(double) const { return grd; }
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

		Matrix* w() /* const */ { return (Matrix*) this; }
		Matrix* b() /* const */ { return &bias; }
		Matrix* z() /* const */ { return &flux; }
		Matrix* a() /* const */ { return &actv; }
		std::vector<Funct *>* f() /* const */ { return &potn; }

		double eval_f(double x) { return (*((potn[0])->get_fun()))(x); }
		double eval_g(double x) { return (*((potn[0])->get_grd()))(x); }

		double eval_f(size_t i, double x) { return (*((potn[i])->get_fun()))(x); }
		double eval_g(size_t i, double x) { return (*((potn[i])->get_grd()))(x); }

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

void eval_pfun(const std::vector<Funct *> &f, const std::vector<double> &x, std::vector<double> &y)
{
	if (f.size() == 1)
	{
		if (f[0] == (Funct *)NULL)
		{
			std::copy(x.begin(), x.end(), y.begin());
		}
		else
		{
			for (int i = 0; i < x.size(); i++)
			{
				y[i] = (*(f[0]->get_fun()))(x[i]);
			}
		}
	}
	else
	{
		for (int i = 0; i < x.size(); i++)
		{
			if (f[i] != (Funct *)NULL)
			{
				y[i] = (*(f[i]->get_fun()))(x[i]);
			}
			else
			{
				y[i] = x[i];
			}
		}
	}
}

void eval_pgrd(const std::vector<Funct *> &f, const std::vector<double> &x, std::vector<double> &y)
{
	if (f.size() == 1)
	{
		if (f[0] == (Funct *)NULL)
		{
			std::fill(y.begin(), y.end(), 1.0);
		}
		else
		{
			for (int i = 0; i < x.size(); i++)
			{
				y[i] = (*(f[0]->get_grd()))(x[i]);
			}
		}
	}
	else
	{
		for (int i = 0; i < x.size(); i++)
		{
			if (f[i] != (Funct *)NULL)
			{
				y[i] = (*(f[i]->get_grd()))(x[i]);
			}
			else
			{
				y[i] = 1.0;
			}
		}
	}
}

void Layer::push(Layer &L)
{
	flux.copy(bias);
	dgemm('N', 'N', 1.0, *w(), *(L.a()), 1.0, flux);
	eval_pfun(*f(), flux, actv);
}

void Layer::push(size_t obs_id, Data *dat_add)
{
	flux.copy(bias);
	dgemv('N', 1.0, *w(), (std::vector<double>)(*dat_add->feat(obs_id)), 1, 1.0, flux, 1);
	eval_pfun(*f(), flux, actv);
}

class Network
{
	private:
		size_t n_lay;
		Layer  *inp_lay;
		Layer  *out_lay;
		Data   *data;
		Funct  *loss;

	public:
		Network() :         n_lay(0), inp_lay((Layer *)NULL), out_lay((Layer *)NULL) {}
		Network(size_t i) : n_lay(i), inp_lay((Layer *)NULL), out_lay((Layer *)NULL) {}

		size_t depth() const { return n_lay; }
		Layer *inp()   const { return inp_lay; }
		Layer *out()   const { return out_lay; }
		
		Funct *L() { return loss; }

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

		void build(std::vector<size_t> &, Funct *);
		void clear();
		void remove(size_t);
		void insert(size_t, size_t);

		void feed_foward(size_t);
		void backprop(double, size_t);
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
		return;
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
};

// Insert layer into existing network.
void Network::insert(size_t postn, size_t n_new)
{
	if ((postn <= 0) | (postn >= n_lay )) 
	{
		std::cout << "Illegal insert: position is outside of possible range." << std::endl;
		return;
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

void Network::backprop(double alpha, size_t obs_id)
{
	Layer *curn_lay = out_lay;
	Matrix del(curn_lay->nrow(), 1);
	//BP 1
	for (int i=0; i<curn_lay->nrow(); i++)
	{
		del[i] =(*(*(curn_lay->f()))[i]->get_grd())((*(curn_lay->z()))[i]); //del = sigma'(z)
		del[i] *= (*( L()->get_grd())) (*(data->resp(obs_id) + i) - (*(curn_lay->a()))[i]); // del = del*grad_a(C)
	}
//		//BP 3
	daxpy(-alpha, del, 1, *(curn_lay->b()), 1);
	
	//BP 4
	dger(-alpha, del, 1, *(curn_lay->a()), 1, *(curn_lay->w())); 
	
	curn_lay = curn_lay->prev();

	Matrix *p_odel = &del;
	
	while(curn_lay != (Layer *)NULL)
	{
		Matrix ndel(curn_lay->nrow(),1);
		Matrix dPhi(curn_lay->nrow(),1);

		//BP 2
		dgemv('T', 1.0, *(curn_lay->next()->w()), *p_odel, 1, 0.0, ndel, 1); 
		eval_pgrd(*(curn_lay->f()), *(curn_lay->z()), dPhi);
		dsbmv('U',1.0, dPhi,0, ndel, 1, 0.0, ndel, 1);
		
		//BP 3
		daxpy(-alpha, ndel, 1, *(curn_lay->b()), 1);
	
		//BP 4
		dger(-alpha, ndel, 1, *(curn_lay->a()), 1, *(curn_lay->w())); 		
		
		p_odel->clearMemory();
		p_odel = &ndel;
		
		curn_lay = curn_lay->prev();

	}
}
