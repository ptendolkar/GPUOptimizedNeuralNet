#include "DevData.h"

__device__ DevData::DevData() : X(), n_row(0), n_col(0), n_rsp(0), n_fea(0) {}
__device__ int DevData::nrsp() { return n_rsp; }
__device__ int DevData::nfea() { return n_fea; }
__device__ int DevData::nrow() { return n_row; }
__device__ int DevData::ncol() { return n_col; }

__device__ float * DevData::resp(int obs_id) { return (&X[obs_id*n_col]); }
__device__ float * DevData::feat(int obs_id) { return (&X[obs_id*n_col + n_rsp]); }

__device__ DevData::DevData( float *dX, int row, int col, int rsp, int fea )
{
	X = dX;
	n_row = row;
	n_col = col;
	n_rsp = rsp;
	n_fea = fea;	
}

