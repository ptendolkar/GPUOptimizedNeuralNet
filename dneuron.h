#include "dmatrix.h"
#include "ddata.h"

class DevMatrix;
class DevData;

class Funct 
{
	private:
		float (*fun)(float);
		float (*grd)(float);

	public:
		 __device__ Funct() : fun(NULL), grd(NULL) {}
		 __device__ Funct(float (*f)(float), float (*g)(float)) : fun(f), grd(g) {}

		 __device__ 
		~Funct()
		{
			fun = NULL;
			grd = NULL;
		};

		 __device__ float (*get_fun())(float) { return fun; }
		 __device__ float (*get_grd())(float) { return grd; }
};

class Layer : public DevMatrix
{
	private:
		size_t iden;
		Layer  *prev_lay_ptr;
		Layer  *next_lay_ptr;
	
		Funct **potn; //new Funct *[10]

	public:
		DevMatrix bias;
		DevMatrix flux;
		DevMatrix actv;

		 __device__ Layer() : DevMatrix(), iden(0), prev_lay_ptr((Layer *)NULL), next_lay_ptr((Layer *)NULL), bias(), flux(), actv(), potn() {}
		 __device__ Layer(size_t i, size_t m, size_t n) : DevMatrix(m,n), iden(i), prev_lay_ptr((Layer *)NULL), next_lay_ptr((Layer *)NULL), bias(m,1), flux(m,1), actv(m,1), potn() {}
		 __device__ Layer(size_t i, size_t m, size_t n, Layer *ipp, Layer *inn)	: DevMatrix(m,n), iden(i), prev_lay_ptr(ipp), next_lay_ptr(inn), bias(m,1), flux(m,1), actv(m,1), potn() {}
		 __device__ Layer(size_t i, size_t m, size_t n, Layer *ipp, Layer *inn, Funct *f) : DevMatrix(m,n), iden(i), prev_lay_ptr(ipp), next_lay_ptr(inn), bias(m,1), flux(m,1), actv(m,1) 
		{
			potn = new Funct *[1];
			potn[0] = f;
		}

		 __device__ size_t id()   const { return iden; }
		 __device__ Layer* prev() const { return prev_lay_ptr; }
		 __device__ Layer* next() const { return next_lay_ptr; }

		 __device__ float* w() /* const */ { return getM(); }
		 __device__ float* b() /* const */ { return bias.getM(); }
		 __device__ float* z() /* const */ { return flux.getM(); }
		 __device__ float* a() /* const */ { return actv.getM(); }

		 __device__ float eval_f(float x) { return (*((potn[0])->get_fun()))(x); }
		 __device__ float eval_g(float x) { return (*((potn[0])->get_grd()))(x); }

		 __device__ void eval_pfun(const std::vector<float> &x, std::vector<float> &y);
		 __device__ void eval_pgrd(const std::vector<float> &x, std::vector<float> &y);

		 __device__ void id(size_t i)     { iden = i; }
		 __device__ void prev(Layer *lay) { prev_lay_ptr = lay; }
		 __device__ void next(Layer *lay) { next_lay_ptr = lay; }
		 __device__ void f(size_t i, Funct *Phi) { potn[i] = Phi; }

		 __device__ void swap(Layer &lay)
		{
			printf("swap needs to be implemented\n");
		};

		__device__ ~Layer()
		{
			delete potn;
			prev_lay_ptr = (Layer *) NULL;
			next_lay_ptr = (Layer *) NULL;
			
			bias.~DevMatrix();
			actv.~DevMatrix();
			flux.~DevMatrix();
		};


		 __device__ void push(size_t, DevData *);
};


class Network
{
	private:
		size_t n_lay;
		Layer  *head_lay_ptr;
		Layer  *tail_lay_ptr;
		Funct  *loss;
		DevData   *data_ptr;

	public:

		 __device__ Network() : n_lay(0), head_lay_ptr((Layer *)NULL), tail_lay_ptr((Layer *)NULL), data_ptr((DevData *)NULL), loss((Funct *)NULL) {}

// Build network dynamically fowards (head to tail) from the output layer.  Single layer network (e.g. logistic regression) will have NULL input layer pointer,
// but all networks must have an output.  The first entry of the dimension array is the size of the covariate space, and the last entry is the size of the output space.

		 __device__ Network(int *dim_lay, int dim_size, Funct *f, Funct *l, DevData *train)
		{
			loss = l;
			data_ptr = train; 

			head_lay_ptr = (Layer *)NULL;
			tail_lay_ptr = (Layer *)NULL;

			if (dim_size < 2)
			{
				printf("Insufficient parameters to create a network.\n");
				return;
			}

			n_lay = dim_size - 1;

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

		 __device__ size_t depth() const { return n_lay; }
		 __device__ Layer  *head() const { return head_lay_ptr; }
		 __device__ Layer  *tail() const { return tail_lay_ptr; }
		 __device__ Funct  *lfun() const { return loss; }
		 __device__ DevData   *data() const { return data_ptr; }

		 __device__ ~Network()
		{
			n_lay        = 0;
			head_lay_ptr = (Layer *)NULL;
			tail_lay_ptr = (Layer *)NULL;
			loss         = (Funct *)NULL;
			data_ptr     = (DevData  *)NULL;
		};

		 __device__ void depth(size_t i) { n_lay = i; }

		 __device__ void build(std::vector<size_t> &, Funct *);
		 __device__ void clear();

		 __device__ void feed_forward(size_t);
		 __device__ void backprop(float, size_t);

	 	 __device__ void train(float, int *, int,  size_t);
		 __device__ void writeModelToFile(std::string, size_t);

		 __device__ void print();
		 __device__ void initialize(float, float);
		 __device__ DevMatrix predict(std::vector<float>&);
};

// Clear dynamically built network fowards.
 __device__ void Network::clear()
{
	Layer *curn_lay_ptr = head_lay_ptr;
	Layer *next_lay_ptr = curn_lay_ptr->next();
	delete curn_lay_ptr;

	while (next_lay_ptr != (Layer *)NULL)
	{
		curn_lay_ptr = next_lay_ptr;
		next_lay_ptr = curn_lay_ptr->next();
		delete curn_lay_ptr;
	}

	head_lay_ptr = tail_lay_ptr = (Layer *)NULL;
	n_lay   = 0;
};

// Check a 'foward' iterator'
 __device__ void Network::feed_forward(size_t obs_id)
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

 __device__ void Network::backprop(float alpha, size_t obs_id)
{
	Layer *curn_lay_ptr = tail_lay_ptr;
	DevMatrix *curn_del_ptr = new DevMatrix(curn_lay_ptr->nrow(), 1);

	float curn_flx;
	float curn_act;
	float curn_obs;

	for (int i = 0; i < curn_lay_ptr->nrow(); i++)
	{
		curn_flx = (curn_lay_ptr->z())[i];
		curn_act = (curn_lay_ptr->a())[i];
		curn_obs = *(data_ptr->resp(obs_id) + i*data_ptr->nrow());
		(curn_del_ptr->getM())[i]  = curn_lay_ptr->eval_g(curn_flx);
		(curn_del_ptr->getM())[i] *= (*loss->get_grd())(curn_act - curn_obs);
	}

	//BP 3
	saxpy(-alpha, *curn_del_ptr, 1, curn_lay_ptr->bias, 1);
	
	//BP 4
	if(head_lay_ptr != tail_lay_ptr)
		sger (-alpha, *curn_del_ptr, 1, curn_lay_ptr->prev()->actv, 1, *(curn_lay_ptr)); 
	else
		sger(-alpha, *curn_del_ptr, 1, *(data_ptr->feat(obs_id)), 1, *(curn_lay_ptr));

	DevMatrix *past_del_ptr = curn_del_ptr;
	curn_del_ptr = NULL;

	curn_lay_ptr = curn_lay_ptr->prev();

	while( curn_lay_ptr != (Layer *)NULL)
	{
		curn_del_ptr = new DevMatrix(curn_lay_ptr->nrow(), 1);

		//BP 2
		sgemv(CUBLAS_OP_T, 1.0, *(curn_lay_ptr->next()), *past_del_ptr, 1, 0.0, *curn_del_ptr, 1); 

		for (int i=0; i<curn_lay_ptr->nrow(); i++)
		{
			curn_flx = (curn_lay_ptr->z())[i];
			(curn_del_ptr->getM())[i] *= curn_lay_ptr->eval_g(curn_flx);
		}

		//BP 3
		saxpy(-alpha, *curn_del_ptr, 1, curn_lay_ptr->bias, 1);
	
		//BP 4
		
		if(curn_lay_ptr != head_lay_ptr)
		{
			sger(-alpha, *curn_del_ptr, 1, curn_lay_ptr->prev()->actv, 1, *(curn_lay_ptr));
		}
		else
		{
			sger(-alpha,  *curn_del_ptr, 1, *(data_ptr->feat(obs_id)), 1,*(curn_lay_ptr)); 
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
	//past_del_ptr = curn_del_ptr = (DevMatrix *)NULL;
}

__device__ void Network::print()
{
	Layer *curn_lay_ptr = tail_lay_ptr;

	printf("====== Layer %d ======\n", curn_lay_ptr->id());
	printf("Weights\n");
	curn_lay_ptr->print();
	
	printf("Biases\n");
	curn_lay_ptr->bias.print();

	curn_lay_ptr = curn_lay_ptr->prev();
	
	while(curn_lay_ptr != (Layer *)NULL)
	{
		printf("====== Layer %d ======\n", curn_lay_ptr->id());
		printf("Weights\n");
		curn_lay_ptr->print();
		
		printf("Biases\n");
		curn_lay_ptr->bias.print();

		curn_lay_ptr = curn_lay_ptr->prev();
	}
	printf("======\n\n");
};

 __device__ void Network::train(float alpha, int * obs_id, int n_obs, size_t iterations )
{	
	for( size_t i = 0 ; i < iterations ; i++ ){
		for( size_t j=0 ; j < n_obs; j++ ){
			feed_forward( obs_id[j] );
			backprop( alpha, obs_id[j] );
	//		printf("obs id %d\n", j);
		}
	//	printf("iteration %d\n", i); 
	//	this->print();
	}	
};

__device__ void Network::initialize(float mean = 0, float sigma = 1){
	Layer *curn_lay_ptr = tail_lay_ptr;
	while(curn_lay_ptr != (Layer *)NULL)
	{
		curn_lay_ptr->initialize(mean, sigma);
		curn_lay_ptr->bias.initialize(mean, sigma);
		curn_lay_ptr = curn_lay_ptr->prev();
	}
};

 __device__ void Layer::push(size_t obs_id, DevData *data_ptr)
{
	flux.copy(bias);
	
	if (prev() != (Layer *)NULL)
	{
		
		sgemv(CUBLAS_OP_N, 1.0, *this, prev()->actv, 1, 1.0, flux, 1);
	}
	else
	{
		sgemv(CUBLAS_OP_N, 1.0, *this,(*data_ptr->feat(obs_id)), 1, 1.0, flux, 1);
	}

	for (int i=0; i<flux.size(); i++)
	{
		(actv.getM())[i] = eval_f((flux.getM())[i]);
	}
}

