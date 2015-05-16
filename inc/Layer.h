#ifndef LAYER_H
#define LAYER_H
#include <matrix.h>
#include <Funct.h>
#include <data.h>

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
		Layer();
		Layer(size_t, size_t, size_t);
		Layer(size_t, size_t, size_t, Layer *, Layer *);
		Layer(size_t , size_t, size_t , Layer *, Layer *, Funct *);

		size_t id();   
		Layer* prev();
		Layer* next();

		Matrix* w();
		Matrix* b();
		Matrix* z();
		Matrix* a();

		double eval_f(double);
		double eval_g(double);

		void id(size_t );
		void prev(Layer *);
		void next(Layer *);
		void f(size_t, Funct *);

		void w_swp(std::vector<double > &);
		void b_swp(std::vector<double > &);
		void z_swp(std::vector<double > &);
		void a_swp(std::vector<double > &);
		void f_swp(std::vector<Funct *> &);

		void swap(Layer &lay);

		void clearMemory();

		void push(size_t, Data *);
};

#endif
