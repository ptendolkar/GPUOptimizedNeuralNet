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
		size_t iden;
		Layer  *prev_lay_ptr;
		Layer  *next_lay_ptr;
		Matrix bias;
		Matrix flux;
		Matrix actv;
		std::vector<Funct *> potn;

	public:
		Layer() : Matrix(), iden(0), prev_lay_ptr((Layer *)NULL), next_lay_ptr((Layer *)NULL), bias(), flux(), actv(), potn() {}
		Layer(size_t i, size_t m, size_t n) : Matrix(m,n), iden(i), prev_lay_ptr((Layer *)NULL), next_lay_ptr((Layer *)NULL), bias(m,1), flux(m,1), actv(m,1), potn() {}
		Layer(size_t i, size_t m, size_t n, Layer *ipp, Layer *inn)	: Matrix(m,n), iden(i), prev_lay_ptr(ipp), next_lay_ptr(inn), bias(m,1), flux(m,1), actv(m,1), potn() {}
		Layer(size_t i, size_t m, size_t n, Layer *ipp, Layer *inn, Funct *f) : Matrix(m,n), iden(i), prev_lay_ptr(ipp), next_lay_ptr(inn), bias(m,1), flux(m,1), actv(m,1), potn(1,f) {}

		size_t id()   const { return iden; }
		Layer* prev() const { return prev_lay_ptr; }
		Layer* next() const { return next_lay_ptr; }

		Matrix* w() /* const */ { return (Matrix*) this; }
		Matrix* b() /* const */ { return &bias; }
		Matrix* z() /* const */ { return &flux; }
		Matrix* a() /* const */ { return &actv; }

		double eval_f(double x) { return (*((potn[0])->get_fun()))(x); }
		double eval_g(double x) { return (*((potn[0])->get_grd()))(x); }

		double eval_f(size_t i, double x) { return (*((potn[i])->get_fun()))(x); }
		double eval_g(size_t i, double x) { return (*((potn[i])->get_grd()))(x); }
		
		void eval_pfun(const std::vector<double> &x, std::vector<double> &y);
		void eval_pgrd(const std::vector<double> &x, std::vector<double> &y);

		void id(size_t i)     { iden = i; }
		void prev(Layer *lay) { prev_lay_ptr = lay; }
		void next(Layer *lay) { next_lay_ptr = lay; }
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
			std::swap(prev_lay_ptr, lay.prev_lay_ptr);
			std::swap(next_lay_ptr, lay.next_lay_ptr);
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
};

void Layer::eval_pfun(const std::vector<double> &x, std::vector<double> &y)
{
	if (potn.size() == 1)
	{
		if (potn[0] != (Funct *)NULL)
		{
			for (int i = 0; i < x.size(); i++)
			{
				y[i] = eval_f(x[i]);
			}
		}
		else
		{
			std::copy(x.begin(), x.end(), y.begin());	
		}
	}
	else
	{
		for (int i = 0; i < x.size(); i++)
		{
			if (potn[i] != (Funct *)NULL)
			{
				y[i] = eval_f(i, x[i]);
			}
			else
			{
				y[i] = x[i];
			}
		}
	}
}

void Layer::eval_pgrd(const std::vector<double> &x, std::vector<double> &y)
{
	if (potn.size() == 1)
	{
		if (potn[0] != (Funct *)NULL)
		{
			for (int i = 0; i < x.size(); i++)
			{
				y[i] = eval_g(x[i]);
			}
		}
		else
		{
			std::fill(y.begin(), y.end(), 1.0);	
		}
	}
	else
	{
		for (int i = 0; i < x.size(); i++)
		{
			if (potn[i] != (Funct *)NULL)
			{
				y[i] = eval_g(i, x[i]);
			}
			else
			{
				y[i] = 1;
			}
		}
	}
}

void Layer::push(size_t obs_id, Data *data_ptr)
{
	flux.copy(bias);
	if (prev() != (Layer *)NULL)
	{
		dgemv('N', 1.0, *w(), *(prev()->a()), 1, 1.0, flux, 1);
	}
	else
	{
		dgemv('N', 1.0, *w(), (std::vector<double>)(*data_ptr->feat(obs_id)), 1, 1.0, flux, 1);
	}
	eval_pfun(flux, actv);
}

class Network
{
	private:
		size_t n_lay;
		Layer  *head_lay_ptr;
		Layer  *tail_lay_ptr;
		Funct  *loss;
		Data   *data_ptr;

	public:
		Network() : n_lay(0), head_lay_ptr((Layer *)NULL), tail_lay_ptr((Layer *)NULL), data_ptr((Data *)NULL), loss((Funct *)NULL) {}

// Build network dynamically fowards (head to tail) from the output layer.  Single layer network (e.g. logistic regression) will have NULL input layer pointer,
// but all networks must have an output.  The first entry of the dimension array is the size of the covariate space, and the last entry is the size of the output space.

		Network(std::vector<size_t> &dim_lay, Funct *f, Funct *l, Data *train)
		{
			loss = l;
			data_ptr = train; 

			head_lay_ptr = (Layer *)NULL;
			tail_lay_ptr = (Layer *)NULL;

			if (dim_lay.size() < 2)
			{
				std::cout << "Insufficient parameters to create a network." << std::endl;
				return;
			}

			n_lay = dim_lay.size() - 1;

			Layer *curn_lay_ptr = new Layer(0, dim_lay[1], dim_lay[0], (Layer *)NULL, (Layer *)NULL, f);
			Layer *prev_lay_ptr = curn_lay_ptr;

			head_lay_ptr = curn_lay_ptr;

			for (int i=0; i<n_lay; i++)
			{
				curn_lay_ptr = new Layer(i, dim_lay[i+1], dim_lay[i], prev_lay_ptr, (Layer *)NULL, f);
				curn_lay_ptr->prev()->next(curn_lay_ptr);
				prev_lay_ptr = curn_lay_ptr;
			}

			tail_lay_ptr = curn_lay_ptr;
		};

		size_t depth() const { return n_lay; }
		Layer  *head() const { return head_lay_ptr; }
		Layer  *tail() const { return tail_lay_ptr; }
		Funct  *lfun() const { return loss; }
		Data   *data() const { return data_ptr; }

		~Network()
		{
			n_lay        = 0;
			head_lay_ptr = (Layer *)NULL;
			tail_lay_ptr = (Layer *)NULL;
			loss         = (Funct *)NULL;
			data_ptr     = (Data  *)NULL;
		};

		Layer depth(size_t i) { n_lay = i; }

		void build(std::vector<size_t> &, Funct *);
		void clear();

		void feed_foward(size_t);
		void backprop(double, size_t);
};

// Clear dynamically built network fowards.
void Network::clear()
{
	Layer *curn_lay_ptr = head_lay_ptr;
	Layer *next_lay_ptr = curn_lay_ptr->next();
	curn_lay_ptr->clearMemory();

	while (next_lay_ptr != (Layer *)NULL)
	{
		curn_lay_ptr = next_lay_ptr;
		next_lay_ptr = curn_lay_ptr->next();
		curn_lay_ptr->clearMemory();
	}

	head_lay_ptr = tail_lay_ptr = (Layer *)NULL;
	n_lay   = 0;
};

// Check a 'foward' iterator'
void Network::feed_foward(size_t obs_id)
{
	Layer *curn_lay_ptr = head_lay_ptr;
	curn_lay_ptr->push(obs_id, data_ptr);
	curn_lay_ptr = curn_lay_ptr->next();

	while (curn_lay_ptr != (Layer *)NULL)
	{
		curn_lay_ptr->push(obs_id, data_ptr);
		curn_lay_ptr = curn_lay_ptr->next();
	}
};

void Network::backprop(double alpha, size_t obs_id)
{
	Layer *curn_lay_ptr = tail_lay_ptr;
	Matrix *curn_del_ptr = new Matrix(curn_lay_ptr->nrow(), 1);

	curn_lay_ptr->eval_pgrd(*(curn_lay_ptr->z()), *(curn_del_ptr));

	Matrix diff(curn_lay_ptr->nrow(), 1);
	diff.copy(*(curn_lay_ptr->a()));
	daxpy(-1.0, (std::vector<double>)(*(data_ptr->feat(obs_id))), data_ptr->nrow(), diff, 1);
	curn_lay_ptr->eval_pgrd(diff, diff);

	Matrix tmp(curn_lay_ptr->nrow(), 1);
	dsbmv('L', 1.0, *curn_del_ptr, 0, diff, 1, 0.0, tmp, 1);
	curn_del_ptr->swap(tmp); 

	//BP 3
	daxpy(-alpha, *curn_del_ptr, 1, *(curn_lay_ptr->b()), 1);
	
	//BP 4
	dger (-alpha, *curn_del_ptr, 1, *(curn_lay_ptr->prev()->a()), 1, *(curn_lay_ptr->w())); 
	
	Matrix *past_del_ptr = curn_lay_ptr;
	curn_lay_ptr = curn_lay_ptr->prev();
	diff.clearMemory();

	while( curn_lay_ptr->prev() != (Layer *)NULL)
	{
		Matrix *curn_del_ptr = new Matrix(curn_lay_ptr->nrow(), 1);
		Matrix dPhi(curn_lay_ptr->nrow(), 1);

		//BP 2
		dgemv('T', 1.0, *(curn_lay_ptr->next()->w()), *past_del_ptr, 1, 0.0, *curn_del_ptr, 1); 
		curn_lay_ptr->eval_pgrd(*(curn_lay_ptr->z()), dPhi);

		Matrix tmp(curn_lay_ptr->nrow(),1);
		dsbmv('L', 1.0, dPhi, 0, *curn_del_ptr, 1, 0.0, tmp, 1);
		curn_del_ptr->swap(tmp);

		//BP 3
		daxpy(-alpha, *curn_del_ptr, 1, *(curn_lay_ptr->b()), 1);
	
		//BP 4
		
		if(curn_lay_ptr->prev() != head_lay_ptr)
		{
			dger(-alpha, *curn_del_ptr, 1, *(curn_lay_ptr->prev()->a()), 1, *(curn_lay_ptr->w()));
		}
		else
		{
			dger(-alpha, (std::vector<double>)(*(data_ptr->feat(obs_id))), 1, *curn_del_ptr, 1, *(curn_lay_ptr->w())); 
		}
	
		past_del_ptr->clearMemory();
		past_del_ptr = curn_lay_ptr;
		
		curn_lay_ptr = curn_lay_ptr->prev();
	}
}
