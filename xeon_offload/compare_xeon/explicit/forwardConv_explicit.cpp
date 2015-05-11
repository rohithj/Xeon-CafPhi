/* Implicit offload of computation on Xeon-Phi 
 * Size chosen based on:
 * VSM size exceeds the limitation (17179869184) now!
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <new>
#include <unistd.h>

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


#pragma offload_attribute(push, target(mic))
float *input;
float *output;
float *weight;

int conv_in_channels_ = IN_CHANNELS;
int conv_out_channels_ = OUT_CHANNELS;
int conv_in_height_ = IN_HEIGHT;
int conv_in_width_ = IN_WIDTH;
int height_out_ = OUT_HEIGHT;
int width_out_ = OUT_WIDTH;
int kernel_h_ = KERNEL_HEIGHT;
int kernel_w_ = KERNEL_WIDTH;
int num_ = NUM;

int inputSize = conv_in_channels_ * conv_in_height_ * conv_in_width_;
int weightSize = conv_out_channels_ * conv_in_channels_ *
                    kernel_h_ * kernel_w_;
int outputSize = conv_out_channels_ * height_out_ * width_out_;
#pragma offload_attribute(pop)


void init()
{
  int i;
  float randVal;
  
  int totalInputSize = inputSize * num_;
  int totalOutputSize = outputSize * num_; 

  input = (float *)malloc(sizeof(float) * totalInputSize);
  if(input == NULL) {
    printf(" Canont alloc memory \n");
    exit(-1);
  }

  /* Read random float values */
  for(i = 0; i < totalInputSize; i++) {
    randVal = static_cast <float>(rand()) / static_cast <float>(RAND_MAX);
    input[i] = randVal;
  }
#if 0
  printf("input: \n");
  for(i = 0; i < totalInputSize; i++) {
    printf(" %f ", input[i]);
  }
  printf("\n\n");
#endif

  weight = (float *)malloc(sizeof(float) * weightSize);
  if(weight == NULL) {
    printf(" Canont alloc memory \n");
    exit(-1);
  }
  
  /* Read random float values */
  for(i = 0; i < weightSize; i++) {
    randVal = MIN_VAL + static_cast <float>(rand()) / static_cast <float>(RAND_MAX);
    weight[i] = randVal;
  }
  
#if 0
  printf("weight: \n");
  for(i = 0; i < weightSize; i++) {
    printf(" %f ", weight[i]);
  }
printf("\n\n");
#endif

  output = (float *)malloc(sizeof(float) * totalOutputSize);
  if(output == NULL) {
    printf(" Canont alloc memory \n");
    exit(-1);
  }
}


__attribute__(( target (mic))) int forward_convolution(int inputOffset,
                                int outputOffset)
{
  int inChan, outChan, h, w, i, j;
  int conv_in_channels_ = IN_CHANNELS;
  int conv_out_channels_ = OUT_CHANNELS;
  int conv_in_height_ = IN_HEIGHT;
  int conv_in_width_ = IN_WIDTH;
  int height_out_ = OUT_HEIGHT;
  int width_out_ = OUT_WIDTH;
  int kernel_h_ = KERNEL_HEIGHT;
  int kernel_w_ = KERNEL_WIDTH;
  int num_ = NUM;

  int outputSize = conv_out_channels_ * height_out_ * width_out_;
  
  float *out = output + outputOffset;
  float *in = input + inputOffset;

  memset(out, 0, outputSize * sizeof(float));

  /* Loop for number of input channels */
  for(inChan = 0; inChan < conv_in_channels_; ++inChan) {
    /* Loop over all output channels */
    for(outChan = 0; outChan < conv_out_channels_; ++outChan) {
      /* Loop over output image height */
      for(h = 0; h < height_out_; ++h) {
        /* Loop over output image width */
        for(w = 0; w < width_out_; ++w) {
          /* Loop over kernel image height */
          for(i = 0; i < kernel_h_; ++i) {
            /* Loop over kernel image width */
            for(j = 0; j < kernel_w_; ++j) {

              out[outChan * height_out_ * width_out_ + h * width_out_ + w] +=
                  in[inChan * conv_in_height_ * conv_in_width_ +
                                   (h+i) * conv_in_width_ + (w+j)] *
                   weight[outChan * conv_in_channels_ * kernel_h_ * kernel_w_ +
                            inChan * kernel_h_ * kernel_w_ +
                          i * kernel_w_ + j];
            }
          }
        }
      }
    }
  }

  return 0;
}


int main()
{
  int i, j;
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
  #pragma offload target(mic)                                     \
        in(input : length(num_ * inputSize))      \
        in(weight : length(weightSize))           \
        out(output : length(num_ * outputSize))
  for(j = 0; j < MAX_ITERATIONS; j++) {
    #pragma omp parallel for 
    for(i = 0; i < num_; i++) {
      forward_convolution(i * inputSize, i * outputSize);
    }
  }

  /* Stop timer and print */
  double endTime = CycleTimer::currentSeconds();
  if((endTime - startTime) < min_cpu)
    min_cpu = endTime - startTime;

  std::cout<< "\nTime Time for implicit = " << min_cpu * 1000 << "ms\n\n\n";  

  return 0;
}

