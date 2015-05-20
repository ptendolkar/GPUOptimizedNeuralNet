#include "Layer.h"

__device__ Layer::Layer(size_t i, size_t m, size_t n) : DevMatrix(m,n), iden(i), prev_lay_ptr((Layer *)NULL), next_lay_ptr((Layer *)NULL), bias(m,1), flux(m,1), actv(m,1), potn(){};
__device__ Layer::Layer(size_t i, size_t m, size_t n, Layer *ipp, Layer *inn) : DevMatrix(m,n), iden(i), prev_lay_ptr(ipp), next_lay_ptr(inn), bias(m,1), flux(m,1), actv(m,1), potn() {}
__device__ Layer::Layer(size_t i, size_t m, size_t n, Layer *ipp, Layer *inn, Funct *f, cublasHandle_t *hdl) : DevMatrix(m,n), iden(i), prev_lay_ptr(ipp), next_lay_ptr(inn), bias(m,1), flux(m,1), actv(m,1)
{
	potn = new Funct *[1];
	potn[0] = f;
	handle = hdl;
}
__device__ size_t Layer::id()                    { return iden; }
__device__ Layer* Layer::prev()                  { return prev_lay_ptr; }
__device__ Layer* Layer::next()                  { return next_lay_ptr; }

__device__ float* Layer::w() /* const */         { return getM(); }
__device__ float* Layer::b() /* const */         { return bias.getM(); }
__device__ float* Layer::z() /* const */         { return flux.getM(); }
__device__ float* Layer::a() /* const */         { return actv.getM(); }

__device__ float  Layer::eval_f(float x)         { return (*((potn[0])->get_fun()))(x); }
__device__ float  Layer::eval_g(float x)         { return (*((potn[0])->get_grd()))(x); }

__device__ void   Layer::id(size_t i)            { iden = i; }
__device__ void   Layer::prev(Layer *lay)        { prev_lay_ptr = lay; }
__device__ void   Layer::next(Layer *lay)        { next_lay_ptr = lay; }
__device__ void   Layer::f(size_t i, Funct *Phi) { potn[i] = Phi; }

__device__ Layer::~Layer()
{
	delete potn;
	prev_lay_ptr = (Layer *) NULL;
	next_lay_ptr = (Layer *) NULL;

	bias.~DevMatrix();
	actv.~DevMatrix();
	flux.~DevMatrix();
}

__device__ void Layer::push(size_t obs_id, DevData *data_ptr)
{
        flux.copy(bias, handle);

        if (prev() != (Layer *)NULL)
        {
                sgemv(handle, CUBLAS_OP_N, 1.0, *this, prev()->actv, 1, 1.0, flux, 1);
        }
        else
        {
                sgemv(handle, CUBLAS_OP_N, 1.0, *this,(*data_ptr->feat(obs_id)), 1, 1.0, flux, 1);
        }

        for (int i=0; i<flux.size(); i++)
        {
                (actv.getM())[i] = eval_f((flux.getM())[i]);
        }
}
