#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "backprop.h"

////////////////////////////////////////////////////////////////////////////////

extern void bpnn_layerforward(float *l1, float *l2, float *conn, int n1, int n2);

extern void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float *who, float *hidden, float *err);

extern void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float *w, float *oldw);


extern int setup(int argc, char** argv);

extern float *alloc_2d_dbl(int m, int n);

extern float squash(float x);

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
  setup(argc, argv);
}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

void bpnn_train_kernel(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;
  
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   

  Timer timer;
   
  printf("Performing CPU computation\n");

  startTime(&timer);

  launch_layerforward(net->input_units, net->hidden_units, net->input_weights, in + 1, hid + 1);

  launch_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid + 1, out + 1);
  
  launch_output_error(net->output_delta, net->target, net->output_units, out + 1, &out_err);

  launch_hidden_error(net->hidden_delta, hid + 1, net->output_delta, out + 1, net->hidden_weights, net->hidden_units, &hid_err);
  
  launch_adjust_weights(net->output_delta, out + 1, net->hidden_units, hid + 1, net->hidden_weights, net->hidden_prev_weights);

  launch_adjust_weights(net->hidden_delta, hid + 1, net->input_units, in + 1, net->input_weights, net->input_prev_weights);

  stopTime(&timer); printf("Total Time Taken: %f s\n", elapsedTime(timer));

}