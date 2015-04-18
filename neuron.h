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

			for (int i=1; i< n_lay; i++)
			{
				curn_lay_ptr = new Layer(i, dim_lay[i+1], dim_lay[i], prev_lay_ptr, (Layer *)NULL, f);
				curn_lay_ptr->prev()->next(curn_lay_ptr);
				prev_lay_ptr = curn_lay_ptr;
			}

			tail_lay_ptr = curn_lay_ptr;
		};

		Network(std::string filename, Funct *f , Funct *l, Data *train)
		{
			char delim = ' ';

			loss = l;
			data_ptr = train; 

			head_lay_ptr = (Layer *)NULL;
			tail_lay_ptr = (Layer *)NULL;

			std::fstream 		input(filename.c_str());
			std::string  		line;

						
			/* first, get the layer dimensions stored in first line */
			std::getline(input, line);
			std::stringstream 	ss(line);

			std::vector<size_t> dim_lay;	
			std::string item;
			while(std::getline(ss, item, delim))
			{
				size_t d = atoi(item.c_str());
				dim_lay.push_back(d);
			}
			
			if (dim_lay.size() < 2)
			{
				std::cout << "Insufficient parameters to create a network." << std::endl;
				return;
			}
			
			n_lay = dim_lay.size() - 1;
			
			Layer *curn_lay_ptr = new Layer(0, dim_lay[1], dim_lay[0], (Layer *)NULL, (Layer *)NULL, f);
			Layer *prev_lay_ptr = curn_lay_ptr;

			head_lay_ptr = curn_lay_ptr;
			
			std::getline(input, line);
			ss.str("");
			ss.clear(); // Clear state flags.
			ss.str(line);

			double *curn_elem_ptr;
			
			curn_elem_ptr = &curn_lay_ptr->front();
			while(std::getline(ss, item, delim))
			{
				*(curn_elem_ptr++) = atof(item.c_str());
			}

			std::getline(input, line);
			ss.str("");
			ss.clear(); // Clear state flags.
			ss.str(line);
			
			curn_elem_ptr = &curn_lay_ptr->b()->front();
			while(std::getline(ss, item, delim))
			{
				*(curn_elem_ptr++) = atof(item.c_str());
			}

			/* read two lines in per layer, the first line for weights, and the second line for biases*/
			for (int i = 1; i < n_lay; i++)
			{
				std::getline(input, line);
				ss.str("");
				ss.clear(); // Clear state flags.
				ss.str(line);

				std::string item;
			
				curn_lay_ptr = new Layer(i, dim_lay[i+1], dim_lay[i], prev_lay_ptr, (Layer *)NULL, f);
				curn_lay_ptr->prev()->next(curn_lay_ptr);
				prev_lay_ptr = curn_lay_ptr;
				

				curn_elem_ptr = &curn_lay_ptr->front();
				while(std::getline(ss, item, delim))
				{
					*(curn_elem_ptr++) = atof(item.c_str());
				}

				std::getline(input, line);
				ss.str("");
				ss.clear(); // Clear state flags.
				ss.str(line);
	
				
				curn_elem_ptr = &curn_lay_ptr->b()->front();
				while(std::getline(ss, item, delim))
				{
					*(curn_elem_ptr++) = atof(item.c_str());
				}
			} 
			
			tail_lay_ptr = curn_lay_ptr;

		}


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

		void feed_forward(size_t);
		void backprop(double, size_t);

		void train(double, std::vector<size_t>&, size_t);
		void writeModelToFile(std::string, size_t);
		void print();
		void initialize(double , double);

		Matrix predict(std::vector<double>&);
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
void Network::feed_forward(size_t obs_id)
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

	double curn_flx;
	double curn_act;
	double curn_obs;

	for (int i = 0; i < curn_lay_ptr->nrow(); i++)
	{
		curn_flx = (*(curn_lay_ptr->z()))[i];
		curn_act = (*(curn_lay_ptr->a()))[i];
		curn_obs = *(data_ptr->resp(obs_id) + i*data_ptr->nrow());
		(*curn_del_ptr)[i]  = curn_lay_ptr->eval_g(curn_flx);
		(*curn_del_ptr)[i] *= (*loss->get_grd())(curn_act - curn_obs);
	}

	//BP 3
	daxpy(-alpha, *curn_del_ptr, 1, *(curn_lay_ptr->b()), 1);
	
	//BP 4
	if(head_lay_ptr != tail_lay_ptr)
		dger (-alpha, *curn_del_ptr, 1, *(curn_lay_ptr->prev()->a()), 1, *(curn_lay_ptr->w())); 
	else
		dger(-alpha, *curn_del_ptr, 1, *(data_ptr->feat(obs_id)), 1, *(curn_lay_ptr->w()));

	Matrix *past_del_ptr = curn_del_ptr;
	curn_del_ptr = NULL;

	curn_lay_ptr = curn_lay_ptr->prev();

	while( curn_lay_ptr != (Layer *)NULL)
	{
		curn_del_ptr = new Matrix(curn_lay_ptr->nrow(), 1);

		//BP 2
		dgemv('T', 1.0, *(curn_lay_ptr->next()->w()), *past_del_ptr, 1, 0.0, *curn_del_ptr, 1); 

		for (int i=0; i<curn_lay_ptr->nrow(); i++)
		{
			curn_flx = (*(curn_lay_ptr->z()))[i];
			(*curn_del_ptr)[i] *= curn_lay_ptr->eval_g(curn_flx);
		}

		//BP 3
		daxpy(-alpha, *curn_del_ptr, 1, *(curn_lay_ptr->b()), 1);
	
		//BP 4
		
		if(curn_lay_ptr != head_lay_ptr)
		{
			dger(-alpha, *curn_del_ptr, 1, *(curn_lay_ptr->prev()->a()), 1, *(curn_lay_ptr->w()));
		}
		else
		{
			dger(-alpha,  *curn_del_ptr, 1, *(data_ptr->feat(obs_id)), 1,*(curn_lay_ptr->w())); 
		}

		delete past_del_ptr;	
		past_del_ptr = NULL;
		past_del_ptr = curn_del_ptr;
		curn_del_ptr = NULL;
		
		curn_lay_ptr = curn_lay_ptr->prev();
	}

	past_del_ptr = NULL;
	delete curn_del_ptr;
	curn_del_ptr = NULL;
	//delete past_del_ptr;
	//past_del_ptr = curn_del_ptr = (Matrix *)NULL;
}

void Network::train(double alpha, std::vector<size_t> &obs_id, size_t iterations )
{	
	for( size_t i = 0 ; i < iterations ; i++ ){
		for( size_t j=0 ; j < obs_id.size(); j++ ){
			feed_forward( obs_id[j] );
			backprop( alpha, obs_id[j] );
//			std::cout << "obs id " << j << std::endl;
		}
//		std::cout << "iteration " << i << std::endl;
//		this->print();
	}	
};


void Network::writeModelToFile(std::string filename, size_t prec=5)
{
	std::ofstream my_file(filename.c_str(), std::ios::trunc);
	
	Layer *curn_lay_ptr = head_lay_ptr;
	my_file << data_ptr->nfea() << " "; 
	while(curn_lay_ptr->next() != (Layer *)  NULL)
	{
		my_file << curn_lay_ptr->ncol() <<  " ";
		curn_lay_ptr = curn_lay_ptr->next();
	}
		my_file << tail_lay_ptr->nrow() << "\n";
	
	curn_lay_ptr = head_lay_ptr;
	while(curn_lay_ptr != (Layer*)NULL)
	{
		
		for(int i = 0; i < curn_lay_ptr->size() ; i++)
		{
			double val = (&curn_lay_ptr->front())[i];
			my_file << std::setprecision(prec) << val << " ";
		}
			my_file << "\n";
		for(int i = 0; i < curn_lay_ptr->b()->size(); i++)
		{
			double val = (&curn_lay_ptr->b()->front())[i];
			my_file << std::setprecision(prec) << val << " ";
		}	
			my_file << "\n";

		curn_lay_ptr = curn_lay_ptr->next();
	}
	my_file.close();
};

void Network::print()
{
	Layer *curn_lay_ptr = tail_lay_ptr;

	std::cout << "====== Layer " << curn_lay_ptr->id() << " ======" << std::endl;
	std::cout << "Weights" << std::endl;
	curn_lay_ptr->print();
	
	std::cout << "Biases" << std::endl;
	curn_lay_ptr->b()->print();

	curn_lay_ptr = curn_lay_ptr->prev();
	
	while(curn_lay_ptr != (Layer *)NULL)
	{
		std::cout << "====== Layer " << curn_lay_ptr->id() << " ======" << std::endl;
		std::cout << "Weights" << std::endl;
		curn_lay_ptr->print();
	
		std::cout << "Biases" << std::endl;
		curn_lay_ptr->b()->print();		

		curn_lay_ptr = curn_lay_ptr->prev();
	}
	std::cout << "======\n\n";
};

void Network::initialize(double mean = 0, double sigma = 1){
	Layer *curn_lay_ptr = tail_lay_ptr;
	while(curn_lay_ptr != (Layer *)NULL)
	{
		curn_lay_ptr->w()->initialize(mean, sigma);
		curn_lay_ptr = curn_lay_ptr->prev();
	}
};

Matrix Network::predict(std::vector<double> &inp)
{
	Data* dat = data_ptr;
	Data tmp_dat(inp);
	
	data_ptr = &tmp_dat;
	feed_forward(0);
	data_ptr = dat;
	
	return *(tail_lay_ptr->a());
};

