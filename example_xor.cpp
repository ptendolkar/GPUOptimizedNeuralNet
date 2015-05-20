#include <cuda_runtime.h>
#include <stdio.h>

#include "Data.h"
#include "DevData.h"
#include "Network.h"
#include <thrust/device_vector.h>
#include <time.h>

#include <iostream>

#define BILLION 1000000000L
int localpid(void) 
{ 
	static int a[9] = { 0 }; 
	return a[0]; 
}

__device__  float dtanh (float x) {
	return (1 - pow(tanhf(x), 2));
}
__device__ float  sqloss(float x) {
	return 0.5*powf(x, 2);
}
__device__ float dsqloss(float x) {
	return x;
}
__device__ float lact(float x)
{
	return x;
}
__device__ float lgrd(float x)
{
	return 1.0;
}

__global__ void train( Network *net, DevData *dd, cublasHandle_t *hdl, float *dX, int  n_row, int n_col, int n_rsp, int n_fea, float alpha, int iters)
{
	Funct L   ( &sqloss , &dsqloss);
	Funct Phi ( &lact   , &lgrd);
	Funct Psi ( &tanhf   , &dtanh);

	dd = new DevData(dX, n_row, n_col, n_rsp, n_fea);
	int num_lay = 3;
	int dim[3];
	    dim[0] = n_fea;
	    dim[1] = 2;
	    dim[2] = 1;

	int maxDimen = dim[0];
	for(int i = 1; i < num_lay; i++)
	{
		if(maxDimen < dim[i])
		{
			maxDimen = dim[i];
		}
	}	
	printf("max dimen %d\n", maxDimen);

	int *obs = new int[n_row];
	printf("Number of rows: %d\n", n_row);

	int nobs = 4; //train on the first nobs datapoints, for testing SHOULD BE CHANGED TO COMPARE TO SERIAL 
	for(int i=0; i < nobs; i++) {
		obs[i] = i;	
	}

	cublasCreate_v2(hdl);
	net = new Network(dim, 3, &Psi, &L, dd, hdl, maxDimen);
	net->initialize( 4891, 0, 1);
	net->print();
	net->train(alpha, obs, nobs, iters);
	cublasDestroy_v2(*hdl);
	cudaDeviceSynchronize();
	net->print();
}

__global__ void predict(Network *net, DevData *d_testdata, cublasHandle_t *hdl, float *dX,  float *labels, int n_row, int n_col, int n_rsp, int n_fea)
{
	d_testdata = new DevData(dX, n_row, n_col, n_rsp, n_fea);
	
	cublasCreate_v2(hdl);
	net->sethandle(hdl);
	net->predict(*d_testdata, labels);
	cublasDestroy_v2(*hdl);
}

int main(int argc, char* argv[])
{
	uint64_t diff;
	struct timespec start, end;

	//cudaDeviceSetLimit(cudaLimitMallocHeapSize, 512 * (1 << 20));

	clock_gettime(CLOCK_MONOTONIC, &start); /* mark start time */
	std::cout << "before allocation" << std::endl;

	Data d("data/train/training", ' ',  1);
	DevData *dd;
	Network *net;
	cublasHandle_t *hdl;
	
	cudaMalloc(&dd,  sizeof(DevData *));
	cudaMalloc(&net, sizeof(Network *));
	cudaMalloc(&hdl, sizeof(cublasHandle_t *));

	float alpha = atof(argv[1]);
	int iters = atoi(argv[2]);

	printf("Alpha: %f, Iterations: %d, Responses: %d, Columns %d\n", alpha, iters, d.nrsp(), d.ncol());
	train<<<1,1>>>(net, dd, hdl, thrust::raw_pointer_cast(&(d.X[0])), d.nrow(), d.ncol(), d.nrsp(), d.nfea(), alpha, iters);
	cudaDeviceSynchronize();
	
	clock_gettime(CLOCK_MONOTONIC, &end); /* mark the end time */	
	diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	
	printf("Seconds: %f\n", diff/1000000000.0);

//	Data                           testdata("testing", ',', 0);
//	DevData                        *d_testdata;
//	thrust::device_vector<float>   D(testdata.nrow());
//	cudaMalloc(&d_testdata, sizeof(DevData *)); 

//	predict<<<1,1>>>(net, d_testdata, hdl, thrust::raw_pointer_cast(&(testdata.X[0])), thrust::raw_pointer_cast(&(D[0])), testdata.nrow(), testdata.ncol(), testdata.nrsp(), testdata.nfea());
//	cudaDeviceSynchronize();

	std::cout << "Predicted Labels:" << std::endl << std::endl;
//	for(int i = 0; i < D.size(); i++)
 //       	std::cout << "D[" << i << "] = " << D[i] << std::endl;

	cudaSetDevice(0);
	cudaDeviceReset();
	
	return 0;

}
