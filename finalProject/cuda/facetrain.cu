#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
#include "backprop_kernel.cu"
#include "imagenet.cu"

extern char *strcpy();
extern void exit();

int layer_size = 0;



void bpnn_save_dbg(BPNN *net, const char *filename)
{
  int n1, n2, n3, i, j;
  float *w;

  FILE *pFile;
  pFile = fopen( filename, "w+" );

  n1 = net->input_n;  n2 = net->hidden_n;  n3 = net->output_n;
  fprintf(pFile, "Saving %dx%dx%d network\n", n1, n2, n3);

  w = net->hidden_weights;
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
      fprintf(pFile, "%d,%d,%f\n", i, j, w[i * (n3+1) + j]);
    }
  }

  fclose(pFile);
  return;
}


void backprop_face()
{
  BPNN *net;
  // int i;
  float out_err, hid_err;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  printf("Input layer size : %d\n", layer_size);
  load(net);
  //entering the training kernel, only one iteration

  BPNN *cudanet;

  cudanet = createNetDevice(layer_size, 16, 1);
  cudaDeviceSynchronize();

  copyNetToDevice(net, cudanet, layer_size, 16, 1);
  cudaDeviceSynchronize();

  printf("Starting training kernel\n");
  bpnn_train_kernel(cudanet, &out_err, &hid_err);
  cudaDeviceSynchronize();

  copyNetFromDevice(net, cudanet, layer_size, 16, 1);
  cudaDeviceSynchronize();

  bpnn_save_dbg(net, "out.txt");
  bpnn_free(net);

  freeDeviceNet(cudanet);

  printf("Training done\n");
}


int setup(int argc, char *argv[])
{
  if(argc!=2){
  fprintf(stderr, "usage: backprop <num of input elements>\n");
  exit(0);
  }

  layer_size = atoi(argv[1]);
  
  int seed;

  seed = 7;   
  bpnn_initialize(seed);
  backprop_face();

  exit(0);
}
