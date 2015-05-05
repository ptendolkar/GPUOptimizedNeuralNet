#pragma once
#include "Data.h"


class DevData
{
	private:
		int n_row;
		int n_col;
		int n_rsp;
		int n_fea;

		float *X;
	public:
		__device__ DevData();

		__device__ int nrsp();
		__device__ int nfea();
		__device__ int nrow();
		__device__ int ncol();

		__device__ float * resp(int);
		__device__ float * feat(int obs_id);

		__device__ DevData( float *, int , int, int, int);

};
