/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *	Prepared for 15-681, Fall 1994.
 * Modified by Shuai Che
 ******************************************************************
 */


#include <stdio.h>
#include <stdlib.h>
#include "backprop.h"
#include <math.h>
#include <fcntl.h> 
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define ABS(x)          (((x) > 0.0) ? (x) : (-(x)))

#define TILE_SIZE 4

/*** Allocate 1d array of floats ***/

float *alloc_1d_dbl(int n)
{
  float *new_var;

  new_var = (float *) malloc ((unsigned) (n * sizeof (float)));
  if (new_var == NULL) {
    printf("ALLOC_1D_DBL: Couldn't allocate array of floats\n");
    return (NULL);
  }
  return (new_var);
}

/*** Allocate 2d array of floats ***/

float *alloc_2d_dbl(int m, int n)
{
  float *new_var;

  new_var = (float *) malloc ((unsigned) (m * n * sizeof (float)));
  if (new_var == NULL) {
    printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
    return (NULL);
  }

  return (new_var);
}


void bpnn_randomize_weights(float *w, int m, int n)
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i * (n+1) + j] = (float) rand()/RAND_MAX;
    }
  }

}

void bpnn_randomize_row(float *w, int m)
{
	int i;
	for (i = 0; i <= m; i++) {
    w[i] = 0.1;
  }
}


void bpnn_zero_weights(float *w, int m, int n)
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i * (n+1) + j] = 0.0;
    }
  }
}


void bpnn_initialize(int seed)
{
  printf("Random number generator seed: %d\n", seed);
  srand(seed);
}


BPNN *bpnn_internal_create(int n_in, int n_hidden, int n_out)
{
  BPNN *newnet;

  newnet = (BPNN *) malloc (sizeof (BPNN));
  if (newnet == NULL) {
    printf("BPNN_CREATE: Couldn't allocate neural network\n");
    return (NULL);
  }

  newnet->input_n = n_in;
  newnet->hidden_n = n_hidden;
  newnet->output_n = n_out;
  newnet->input_units = alloc_1d_dbl(n_in + 1);
  
  newnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
  newnet->output_units = alloc_1d_dbl(n_out + 1);

  newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
  newnet->output_delta = alloc_1d_dbl(n_out + 1);
  newnet->target = alloc_1d_dbl(n_out + 1);

  newnet->input_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  
  newnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

  newnet->input_prev_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newnet->hidden_prev_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

  return (newnet);
}


void bpnn_free(BPNN *net)
{
  free((char *) net->input_units);
  free((char *) net->hidden_units);
  free((char *) net->output_units);

  free((char *) net->hidden_delta);
  free((char *) net->output_delta);
  free((char *) net->target);

  free((char *) net->input_weights);
  free((char *) net->input_prev_weights);

  free((char *) net->hidden_weights);
  free((char *) net->hidden_prev_weights);

  free((char *) net);
}


/*** Creates a new fully-connected network from scratch,
     with the given numbers of input, hidden, and output units.
     Threshold units are automatically included.  All weights are
     randomly initialized.

     Space is also allocated for temporary storage (momentum weights,
     error computations, etc).
***/

BPNN *bpnn_create(int n_in, int n_hidden, int n_out)
{

  BPNN *newnet;

  newnet = bpnn_internal_create(n_in, n_hidden, n_out);

#ifdef INITZERO
  bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
#else
  bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);

#endif
  bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
  bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
  bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);
  bpnn_randomize_row(newnet->target, n_out);

  return (newnet);
}


__global__ void layerforward(float *l1, float *l2, float *conn, int n1, int n2) {

  int j = blockIdx.x * blockDim.x + threadIdx.x;

  float sum = 0;
  l1[0] = 1.0;

  if (j < n2 && j != 0) {
    for (int k = 0 ; k < n1 ; k++)   
      sum += conn[k * (n2) + j] * l1[k];

    l2[j] = (1.0 / (1.0 + exp(-sum)));
  }

}


void launch_layerforward(float *l1, float *l2, float *conn, int n1, int n2) {
  const unsigned int BLOCK_SIZE = TILE_SIZE;

  dim3 DimGrid((int) ((n2 - 1) / BLOCK_SIZE + 1), 1, 1);
  dim3 DimBlock((int) (BLOCK_SIZE), 1, 1);

  layerforward<<<DimGrid, DimBlock>>>(l1, l2, conn, n1, n2); 
}


__global__ void output_error(float *delta, float *target, float *output, int nj, float *err) {

  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if(j < nj && j != 0) { 
    float o = output[j];
    delta[j] = o * (1.0 - o) * (target[j] - o);
    atomicAdd(err, ABS(delta[j]));
  }

}

void launch_output_error(float *delta, float *target, float *output, int nj, float *err) {
  const unsigned int BLOCK_SIZE = TILE_SIZE;

  dim3 DimGrid((int) ((nj - 1)/BLOCK_SIZE + 1), 1, 1);
  dim3 DimBlock((int)(BLOCK_SIZE), 1, 1);

  // TODO this is all redundant if iteration is 1, since err is never used again
  
  *err = 0;
  
  float *err_cuda;

  cudaMalloc((void **) &(err_cuda), sizeof(float));
  cudaMemcpy(err_cuda, err, sizeof(float), cudaMemcpyHostToDevice);

  output_error<<<DimGrid, DimBlock>>>(delta, target, output, nj, err_cuda); 

  cudaMemcpy(err, err_cuda, sizeof(float), cudaMemcpyDeviceToHost);

}

__global__ void hidden_error(float *delta_h, int nh, float *delta_o, 
  int no, float *who, float *hidden, float *err) {

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  float h;

  float sum = 0;
  if (j < nh && j != 0) {

    h = hidden[j];

    for (int k = 1 ; k < no ; k++) { 
      sum += delta_o[k] * who[j * no + k];
    } 

    delta_h[j] = h * (1.0 - h) * sum;
    atomicAdd(err, ABS(delta_h[j]));
  }

}

void launch_hidden_error(float *delta_h, int nh, float *delta_o, 
  int no, float *who, float *hidden, float *err) {
  const unsigned int BLOCK_SIZE = TILE_SIZE;

  dim3 DimGrid((int) ((nh - 1)/BLOCK_SIZE + 1), 1, 1);
  dim3 DimBlock((int)(BLOCK_SIZE), 1, 1);

  // TODO this is all redundant if iteration is 1, since err is never used again
  
  *err = 0;
  
  float *err_cuda;

  cudaMalloc((void **) &(err_cuda), sizeof(float));
  cudaMemcpy(err_cuda, err, sizeof(float), cudaMemcpyHostToDevice);

  hidden_error<<<DimGrid, DimBlock>>>(delta_h, nh, delta_o, no, who, hidden, err_cuda); 

  cudaMemcpy(err, err_cuda, sizeof(float), cudaMemcpyDeviceToHost);

}


__global__ void adjust_weights(float *delta, int ndelta, float *ly, int nly,float *w, float *oldw) {

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  float new_dw;

  if (j < ndelta && j != 0) {
    for (int k = 0 ; k < nly ; k++) {
      new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k * (ndelta) + j])); 
      w[k * (ndelta) + j] += new_dw;
      oldw[k * (ndelta) + j] = new_dw;
    } 
  }

}

void launch_adjust_weights(float *delta, int ndelta, float *ly, int nly,float *w, float *oldw) {
  const unsigned int BLOCK_SIZE = TILE_SIZE;

  dim3 DimGrid((int) ((ndelta - 1) / BLOCK_SIZE + 1), 1, 1);
  dim3 DimBlock((int) (BLOCK_SIZE), 1, 1);

  adjust_weights<<<DimGrid, DimBlock>>>(delta, ndelta, ly, nly, w, oldw); 
}


BPNN *createNetDevice(int n_in, int n_hidden, int n_out)
{
  BPNN *newnet;

  newnet = (BPNN *) malloc (sizeof (BPNN));
  newnet->input_n = n_in;
  newnet->hidden_n = n_hidden;
  newnet->output_n = n_out;

  cudaMalloc((void **) &(newnet->input_units), sizeof(float) * (n_in + 1));
  
  cudaMalloc((void **) &(newnet->hidden_units), sizeof(float) * (n_hidden + 1));

  cudaMalloc((void **) &(newnet->output_units), sizeof(float) * (n_out + 1));

  cudaMalloc((void **) &(newnet->hidden_delta), sizeof(float) * (n_hidden + 1));

  cudaMalloc((void **) &(newnet->output_delta), sizeof(float) * (n_out + 1));

  cudaMalloc((void **) &(newnet->target), sizeof(float) * (n_out + 1));

  cudaMalloc((void **) &(newnet->input_weights), sizeof(float) * (n_in + 1) * (n_hidden + 1));
  
  cudaMalloc((void **) &(newnet->hidden_weights), sizeof(float) * (n_hidden + 1) * (n_out + 1));

  cudaMalloc((void **) &(newnet->input_prev_weights), sizeof(float) * (n_in + 1) * (n_hidden + 1));

  cudaMalloc((void **) &(newnet->hidden_prev_weights), sizeof(float) * (n_hidden + 1) * (n_out + 1));

  return (newnet);
}

void copyNetToDevice(BPNN *net, BPNN *cudanet, int n_in, int n_hidden, int n_out) {

  cudaMemcpy(cudanet->input_units, net->input_units, sizeof(float)*(n_in + 1), cudaMemcpyHostToDevice);
  
  cudaMemcpy(cudanet->hidden_units, net->hidden_units, sizeof(float)*(n_hidden + 1), cudaMemcpyHostToDevice);

  cudaMemcpy(cudanet->output_units, net->output_units, sizeof(float)*(n_out + 1), cudaMemcpyHostToDevice);

  cudaMemcpy(cudanet->hidden_delta, net->hidden_delta, sizeof(float)*(n_hidden + 1), cudaMemcpyHostToDevice);

  cudaMemcpy(cudanet->output_delta, net->output_delta, sizeof(float)*(n_out + 1), cudaMemcpyHostToDevice);

  cudaMemcpy(cudanet->target, net->target, sizeof(float)*(n_out + 1), cudaMemcpyHostToDevice);

  cudaMemcpy(cudanet->input_weights, net->input_weights, sizeof(float)*(n_in + 1) * (n_hidden + 1), cudaMemcpyHostToDevice);
  
  cudaMemcpy(cudanet->hidden_weights, net->hidden_weights, sizeof(float)*(n_hidden + 1) * (n_out + 1), cudaMemcpyHostToDevice);

  cudaMemcpy(cudanet->input_prev_weights, net->input_prev_weights, sizeof(float)*(n_in + 1) * (n_hidden + 1), cudaMemcpyHostToDevice);

  cudaMemcpy(cudanet->hidden_prev_weights, net->hidden_prev_weights, sizeof(float)*(n_hidden + 1) * (n_out + 1), cudaMemcpyHostToDevice);
}

void copyNetFromDevice(BPNN *net, BPNN *cudanet, int n_in, int n_hidden, int n_out) {

  cudaMemcpy(net->input_units, cudanet->input_units, sizeof(float)*(n_in + 1), cudaMemcpyDeviceToHost);
  
  cudaMemcpy(net->hidden_units, cudanet->hidden_units, sizeof(float)*(n_hidden + 1), cudaMemcpyDeviceToHost);

  cudaMemcpy(net->output_units, cudanet->output_units, sizeof(float)*(n_out + 1), cudaMemcpyDeviceToHost);

  cudaMemcpy(net->hidden_delta, cudanet->hidden_delta, sizeof(float)*(n_hidden + 1), cudaMemcpyDeviceToHost);

  cudaMemcpy(net->output_delta, cudanet->output_delta, sizeof(float)*(n_out + 1), cudaMemcpyDeviceToHost);

  cudaMemcpy(net->target, cudanet->target, sizeof(float)*(n_out + 1), cudaMemcpyDeviceToHost);

  cudaMemcpy(net->input_weights, cudanet->input_weights, sizeof(float)*(n_in + 1) * (n_hidden + 1), cudaMemcpyDeviceToHost);
  
  cudaMemcpy(net->hidden_weights, cudanet->hidden_weights, sizeof(float)*(n_hidden + 1) * (n_out + 1), cudaMemcpyDeviceToHost);

  cudaMemcpy(net->input_prev_weights, cudanet->input_prev_weights, sizeof(float)*(n_in + 1) * (n_hidden + 1), cudaMemcpyDeviceToHost);

  cudaMemcpy(net->hidden_prev_weights, cudanet->hidden_prev_weights, sizeof(float)*(n_hidden + 1) * (n_out + 1), cudaMemcpyDeviceToHost);
}

void freeDeviceNet(BPNN *net)
{
  cudaFree(net->input_units);
  cudaFree(net->hidden_units);
  cudaFree(net->output_units);

  cudaFree(net->hidden_delta);
  cudaFree(net->output_delta);
  cudaFree(net->target);

  cudaFree(net->input_weights);
  cudaFree(net->input_prev_weights);

  cudaFree(net->hidden_weights);
  cudaFree(net->hidden_prev_weights);

  free(net);
}
