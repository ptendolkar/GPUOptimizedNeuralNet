#include "Network.h"

 __device__ Network::Network() : n_lay(0), head_lay_ptr((Layer *)NULL), tail_lay_ptr((Layer *)NULL), data_ptr((DevData *)NULL), loss((Funct *)NULL) {}

// Build network dynamically fowards (head to tail) from the output layer.  Single layer network (e.g. logistic regression) will have NULL input layer pointer,
// but all networks must have an output.  The first entry of the dimension array is the size of the covariate space, and the last entry is the size of the output space.

 __device__ Network::Network(int *dim_lay, int dim_size, Funct *f, Funct *l, DevData *train, cublasHandle_t *hdl)
{
	handle = hdl;
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

	Layer *curn_lay_ptr = new Layer(0, dim_lay[1], dim_lay[0], (Layer *)NULL, (Layer *)NULL, f, handle);
	Layer *prev_lay_ptr = curn_lay_ptr;

	head_lay_ptr = curn_lay_ptr;

	for (int i=1; i< n_lay; i++)
	{
		curn_lay_ptr = new Layer(i, dim_lay[i+1], dim_lay[i], prev_lay_ptr, (Layer *)NULL, f, handle);
		curn_lay_ptr->prev()->next(curn_lay_ptr);
		prev_lay_ptr = curn_lay_ptr;
	}

	tail_lay_ptr = curn_lay_ptr;
}

 __device__ size_t    Network::depth() { return n_lay; }
 __device__ Layer *   Network::head() { return head_lay_ptr; }
 __device__ Layer *   Network::tail() { return tail_lay_ptr; }
 __device__ Funct *   Network::lfun() { return loss; }
 __device__ DevData * Network::data() { return data_ptr; }

 __device__ Network::~Network()
{
	n_lay        = 0;
	head_lay_ptr = (Layer *)NULL;
	tail_lay_ptr = (Layer *)NULL;
	loss         = (Funct *)NULL;
	data_ptr     = (DevData  *)NULL;
}

 __device__ void Network::depth(size_t i) { n_lay = i; }

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
}

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
}

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
	saxpy(handle,-alpha, *curn_del_ptr, 1, curn_lay_ptr->bias, 1);
	
	//BP 4
	if(head_lay_ptr != tail_lay_ptr)
		sger (handle,-alpha, *curn_del_ptr, 1, curn_lay_ptr->prev()->actv, 1, *(curn_lay_ptr)); 
	else
		sger(handle, -alpha, *curn_del_ptr, 1, *(data_ptr->feat(obs_id)), 1, *(curn_lay_ptr));

	DevMatrix *past_del_ptr = curn_del_ptr;
	curn_del_ptr = NULL;

	curn_lay_ptr = curn_lay_ptr->prev();

	while( curn_lay_ptr != (Layer *)NULL)
	{
		curn_del_ptr = new DevMatrix(curn_lay_ptr->nrow(), 1);

		//BP 2
		sgemv(handle, CUBLAS_OP_T, 1.0, *(curn_lay_ptr->next()), *past_del_ptr, 1, 0.0, *curn_del_ptr, 1); 

		for (int i=0; i<curn_lay_ptr->nrow(); i++)
		{
			curn_flx = (curn_lay_ptr->z())[i];
			(curn_del_ptr->getM())[i] *= curn_lay_ptr->eval_g(curn_flx);
		}

		//BP 3
		saxpy(handle,-alpha, *curn_del_ptr, 1, curn_lay_ptr->bias, 1);
	
		//BP 4
		
		if(curn_lay_ptr != head_lay_ptr)
		{
			sger(handle,-alpha, *curn_del_ptr, 1, curn_lay_ptr->prev()->actv, 1, *(curn_lay_ptr));
		}
		else
		{
			sger(handle,-alpha,  *curn_del_ptr, 1, *(data_ptr->feat(obs_id)), 1,*(curn_lay_ptr)); 
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
}

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
}

__device__ void Network::initialize(unsigned long seed= 1234, float mean = 0, float sigma = 1){
	Layer *curn_lay_ptr = tail_lay_ptr;
	while(curn_lay_ptr != (Layer *)NULL)
	{
		curn_lay_ptr->initialize(seed, mean, sigma);
		curn_lay_ptr->bias.initialize(seed,mean, sigma);
		curn_lay_ptr = curn_lay_ptr->prev();
	}
};
