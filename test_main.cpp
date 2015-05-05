// CUDA runtime
#include <cuda_runtime.h>
#include <stdio.h>

#include "Data.h"
#include "DevData.h"
#include "Network.h"
#include <thrust/device_vector.h>
#include <ctime>

__device__ float dtanh (float x) {
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
	printf("in train kernel\n");
	for (int i= 0; i< n_row*n_col; i++){
		printf("%f ", dX[i]);
	}
	printf("\n");

	Funct L   ( &sqloss , &dsqloss);
	Funct Phi ( &lact   , &lgrd);
	Funct Psi ( &tanhf   , &dtanh);

	dd = new DevData(dX, n_row, n_col, n_rsp, n_fea);
	int dim[3];
	    dim[0] = n_fea;
	    dim[1] = 15;
	    dim[2] = 1;

	int *obs = new int[n_row];
	for(int i=0; i < n_row; i++)
		obs[i] = i;	

	cublasCreate_v2(hdl);
	net = new Network(dim, 3, &Psi, &L, dd, hdl);
	net->initialize( 1234, 0, 1);
	net->print();
	net->train(alpha, obs, 4, iters);
	cublasDestroy_v2(*hdl);
	net->print();
}

int main(int argc, char* argv[])
{
	time_t start, end;
	time(&start);
	
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 512 * (1 << 20));
	
	std::cout << "before allocation" << std::endl;

	Data d("data.csv", ',', 1);
	DevData *dd;
	Network *net;
	cublasHandle_t *hdl;
	
	cudaMalloc(&dd,  sizeof(DevData *));
	cudaMalloc(&net, sizeof(Network *));
	cudaMalloc(&hdl, sizeof(cublasHandle_t *));

	float alpha = atof(argv[1]);
	int iters = atoi(argv[2]);
	train<<<1,1>>>(net, dd, hdl, thrust::raw_pointer_cast(&(d.X[0])), d.nrow(), d.ncol(), d.nrsp(), d.nfea(), alpha, iters);
	cudaDeviceSynchronize();
	time(&end);

	printf("Time: %f\n",difftime(end, start));
	//cudaDeviceReset();

	return 0;

}
