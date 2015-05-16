#ifndef NETWORK_H
#define NETWORK_H
#include <Layer.h>

class Network
{
	private:
		size_t n_lay;
		Layer  *head_lay_ptr;
		Layer  *tail_lay_ptr;
		Funct  *loss;
		Data   *data_ptr;

	public:
		Network();

		Network(std::vector<size_t> &, Funct *, Funct *, Data *);

		Network(std::string filename, Funct *f , Funct *l, Data *train);

		size_t depth();
		Layer  *head();
		Layer  *tail();
		Funct  *lfun();
		Data   *data();

		~Network();

		Layer depth(size_t);

		void build(std::vector<size_t> &, Funct *);
		void clear();

		void feed_forward(size_t);
		void backprop(double, size_t);

		void train(double, std::vector<size_t>&, size_t);
		void writeModelToFile(std::string, size_t);
		void print();
		void initialize(long, double , double);

};

#endif
