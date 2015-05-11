/* Implicit offload of computation on Xeon-Phi 
 * Size chosen based on:
 * VSM size exceeds the limitation (17179869184) now!
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <new>

#include <cilk/cilk.h>

#include "CycleTimer.h"


#define MIN_VAL         (-1)
#define MAX_ITERATIONS  100

#define IN_CHANNELS     3
#define OUT_CHANNELS    10

#define IN_HEIGHT       256
#define IN_WIDTH        256

#define OUT_HEIGHT      80
#define OUT_WIDTH       80

#define KERNEL_HEIGHT   8
#define KERNEL_WIDTH    8

#define NUM             64


class
_Cilk_shared
Convolution {
  public:
    int conv_in_channels_;
    int conv_out_channels_;
    int conv_in_height_;
    int conv_in_width_;
    int height_out_;
    int width_out_;
    int kernel_h_;
    int kernel_w_;
    int num_;

    float *input;
    float *output;
    float *weight;
};



Convolution _Cilk_shared *cv;



void init()
{
  int i;
  float randVal;
  
  cv = new(_Offload_shared_malloc(sizeof(Convolution))) _Cilk_shared Convolution;
  cv->conv_in_channels_ = IN_CHANNELS;
  cv->conv_out_channels_ = OUT_CHANNELS;
  cv->conv_in_height_ = IN_HEIGHT;
  cv->conv_in_width_ = IN_WIDTH;
  cv->height_out_ = OUT_HEIGHT;
  cv->width_out_ = OUT_WIDTH;
  cv->kernel_h_ = KERNEL_HEIGHT;
  cv->kernel_w_ = KERNEL_WIDTH;
  cv->num_ = NUM;

  int inputSize = cv->conv_in_channels_ * cv->conv_in_height_ * cv->conv_in_width_;
  int weightSize = cv->conv_out_channels_ * cv->conv_in_channels_ *
                    cv->kernel_h_ * cv->kernel_w_;
  int outputSize = cv->conv_out_channels_ * cv->height_out_ * cv->width_out_;
  int totalInputSize = inputSize * cv->num_;
  int totalOutputSize = outputSize * cv->num_; 

  cv->input = new( _Offload_shared_malloc(totalInputSize * sizeof(float))) _Cilk_shared float;
  if(cv->input == NULL) {
    printf(" Canont alloc memory \n");
    exit(-1);
  }

  /* Read random float values */
  for(i = 0; i < totalInputSize; i++) {
    randVal = static_cast <float>(rand()) / static_cast <float>(RAND_MAX);
    cv->input[i] = randVal;
  }
#if 0
  printf("input: \n");
  for(i = 0; i < totalInputSize; i++) {
    printf(" %f ", cv->input[i]);
  }
  printf("\n\n");
#endif

  cv->weight = new(_Offload_shared_malloc(weightSize * sizeof(float))) _Cilk_shared float;
  if(cv->weight == NULL) {
    printf(" Canont alloc memory \n");
    exit(-1);
  }
  
  /* Read random float values */
  for(i = 0; i < weightSize; i++) {
    randVal = MIN_VAL + static_cast <float>(rand()) / static_cast <float>(RAND_MAX);
    cv->weight[i] = randVal;
  }
  
#if 0
  printf("weight: \n");
  for(i = 0; i < weightSize; i++) {
    printf(" %f ", cv->weight[i]);
  }
printf("\n\n");
#endif

  cv->output = new(_Offload_shared_malloc(totalOutputSize * sizeof(float))) _Cilk_shared float;
  if(cv->output == NULL) {
    printf(" Canont alloc memory \n");
    exit(-1);
  }
}


_Cilk_shared
int forward_convolution(Convolution _Cilk_shared *c, int inputOffset, int outputOffset)
{
#ifdef __MIC__  
  int inChan, outChan, h, w, i, j;
  int outputSize = c->conv_out_channels_ * c->height_out_ * c->width_out_;

  float *output = c->output + outputOffset;
  float *input = c->input + inputOffset;

  memset(c->output, 0, outputSize * sizeof(float));

  /* Loop for number of input channels */
  for(inChan = 0; inChan < c->conv_in_channels_; ++inChan) {
    /* Loop over all output channels */
    for(outChan = 0; outChan < c->conv_out_channels_; ++outChan) {
      /* Loop over output image height */
      for(h = 0; h < c->height_out_; ++h) {
        /* Loop over output image width */
        for(w = 0; w < c->width_out_; ++w) {
          /* Loop over kernel image height */
          for(i = 0; i < c->kernel_h_; ++i) {
            /* Loop over kernel image width */
            for(j = 0; j < c->kernel_w_; ++j) {

              output[outChan * c->height_out_ * c->width_out_ + h * c->width_out_ + w] +=
                  input[inChan * c->conv_in_height_ * c->conv_in_width_ +
                                   (h+i) * c->conv_in_width_ + (w+j)] *
                   c->weight[outChan * c->conv_in_channels_ * c->kernel_h_ * c->kernel_w_ +
                            inChan * c->kernel_h_ * c->kernel_w_ +
                          i * c->kernel_w_ + j];
            }
          }
        }
      }
    }
  }

  return 0;
#else

  return 1;
#endif  
}


_Cilk_shared
void Offload_work(_Cilk_shared Convolution *cv)
{
  int i, j;

  int inputSize = cv->conv_in_channels_ * cv->conv_in_height_ * cv->conv_in_width_;
  int outputSize = cv->conv_out_channels_ * cv->height_out_ * cv->width_out_;
  
  cilk_for(i = 0; i < cv->num_; i++) {
    forward_convolution(cv, i*inputSize, i*outputSize);
  }
  
}


int main()
{
  int j;
  int num_devices = 0;

  printf("Checking for Intel(R) Xeon Phi(TM) (Target CPU) devices...\n\n");

#ifdef __INTEL_OFFLOAD
  num_devices = _Offload_number_of_devices();
#endif
  printf("Number of Target devices installed: %d\n\n",num_devices);

  /* Init system */
  init();
  
  /* Start timer */ 
  double min_cpu = 1e30;
  double startTime = CycleTimer::currentSeconds();

  /* Do work */
  for(j = 0; j < MAX_ITERATIONS; j++) {
    _Cilk_offload Offload_work(cv);
  }

  /* Stop timer and print */
  double endTime = CycleTimer::currentSeconds();
  if((endTime - startTime) < min_cpu)
    min_cpu = endTime - startTime;

  std::cout<< "\nTime Time for implicit = " << min_cpu * 1000 << "ms\n\n\n";  

  return 0;
}

