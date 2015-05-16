#include <Layer.h>

Layer::Layer() : Matrix(), iden(0), prev_lay_ptr((Layer *)NULL), next_lay_ptr((Layer *)NULL), bias(), flux(), actv(), potn() {}
Layer::Layer(size_t i, size_t m, size_t n) : Matrix(m,n), iden(i), prev_lay_ptr((Layer *)NULL), next_lay_ptr((Layer *)NULL), bias(m,1), flux(m,1), actv(m,1), potn() {}
Layer::Layer(size_t i, size_t m, size_t n, Layer *ipp, Layer *inn)	: Matrix(m,n), iden(i), prev_lay_ptr(ipp), next_lay_ptr(inn), bias(m,1), flux(m,1), actv(m,1), potn() {}
Layer::Layer(size_t i, size_t m, size_t n, Layer *ipp, Layer *inn, Funct *f) : Matrix(m,n), iden(i), prev_lay_ptr(ipp), next_lay_ptr(inn), bias(m,1), flux(m,1), actv(m,1), potn(1,f) {}

size_t Layer::id()   { return iden; }
Layer* Layer::prev() { return prev_lay_ptr; }
Layer* Layer::next() { return next_lay_ptr; }

Matrix* Layer::w() /* const */ { return (Matrix*) this; }
Matrix* Layer::b() /* const */ { return &bias; }
Matrix* Layer::z() /* const */ { return &flux; }
Matrix* Layer::a() /* const */ { return &actv; }

double Layer::eval_f(double x) { return (*((potn[0])->get_fun()))(x); }
double Layer::eval_g(double x) { return (*((potn[0])->get_grd()))(x); }

void Layer::id(size_t i)     { iden = i; }
void Layer::prev(Layer *lay) { prev_lay_ptr = lay; }
void Layer::next(Layer *lay) { next_lay_ptr = lay; }
void Layer::f(size_t i, Funct *Phi) { potn[i] = Phi; }

void Layer::w_swp(std::vector<double > &x)   { this->std::vector<double>::swap(x); }
void Layer::b_swp(std::vector<double > &x)   { bias.std::vector<double >::swap(x); }
void Layer::z_swp(std::vector<double > &x)   { flux.std::vector<double >::swap(x); }
void Layer::a_swp(std::vector<double > &x)   { actv.std::vector<double >::swap(x); }
void Layer::f_swp(std::vector<Funct *> &Phi) { potn.std::vector<Funct *>::swap(Phi); }

void Layer::swap(Layer &lay)
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

void Layer::clearMemory()
{
	Layer empty;
	swap(empty);
};

void Layer::push(size_t obs_id, Data *data_ptr)
{
	flux.copy(bias);
	if (prev() != (Layer *)NULL)
	{
		dgemv('N', 1.0, *w(), *(prev()->a()), 1, 1.0, flux, 1);
	}
	else
	{
		dgemv('N', 1.0, *w(),(*data_ptr->feat(obs_id)), 1, 1.0, flux, 1);
	}

	for (int i=0; i<flux.size(); i++)
	{
		actv[i] = eval_f(flux[i]);
	}
}

