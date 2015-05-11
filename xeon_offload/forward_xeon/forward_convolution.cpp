#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <new>

#include <cilk/cilk.h>

#include "CycleTimer.h"

#define CONVOLUTION_1

#ifdef CONVOLUTION_1

#define GROUP_SIZE      1
#define IN_CHANNELS     1
#define OUT_CHANNELS    20
#define IN_HEIGHT       28
#define IN_WIDTH        28
#define OUT_HEIGHT      24
#define OUT_WIDTH       24
#define KERNEL_HEIGHT   5
#define KERNEL_WIDTH    5

#elif CONVOLUTION_2

#define GROUP_SIZE      1
#define IN_CHANNELS     1
#define OUT_CHANNELS    20
#define IN_HEIGHT       28
#define IN_WIDTH        28
#define OUT_HEIGHT      24
#define OUT_WIDTH       24
#define KERNEL_HEIGHT   5
#define KERNEL_WIDTH    5

#endif


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
    int group_;

    float *input;
    float *output;
    float *weight;
};



Convolution _Cilk_shared *cv;



void init()
{
  cv = new(_Offload_shared_malloc(sizeof(Convolution))) _Cilk_shared Convolution;
  cv->conv_in_channels_ = IN_CHANNELS;
  cv->conv_out_channels_ = OUT_CHANNELS;
  cv->conv_in_height_ = IN_HEIGHT;
  cv->conv_in_width_ = IN_WIDTH;
  cv->height_out_ = OUT_HEIGHT;
  cv->width_out_ = OUT_WIDTH;
  cv->kernel_h_ = KERNEL_HEIGHT;
  cv->kernel_w_ = KERNEL_WIDTH;
  cv->group_ = GROUP_SIZE;

  int num = 64;
  int inputSize = cv->conv_in_channels_ * cv->conv_in_height_ * cv->conv_in_width_;
  int weightSize = cv->conv_out_channels_ * cv->conv_in_channels_ / cv->group_ *
                    cv->kernel_h_ * cv->kernel_w_;
  int outputSize = cv->conv_out_channels_ * cv->height_out_ * cv->width_out_;
  int i;
  
  //cv->input = (float *)malloc(inputSize * sizeof(float));
  cv->input = new( _Offload_shared_malloc(64 * inputSize * sizeof(float))) _Cilk_shared float;
  if(cv->input == NULL) {
    printf(" Canont alloc memory \n");
    exit(-1);
  }

  FILE *fp = fopen("inputData.txt", "r");
  if(fp == NULL) {
    printf("File inputData.txt not found\n\n");
    exit(-1);
  }
  fread(cv->input, 64 * inputSize, sizeof(float), fp);
  fclose(fp);

#if 0
  printf("input: \n");
  for(i = 0; i < inputSize; i++) {
    printf(" %f ", cv->input[i]);
  }
  printf("\n\n");
#endif

  cv->weight = new(_Offload_shared_malloc(weightSize * sizeof(float))) _Cilk_shared float;
  if(cv->weight == NULL) {
    printf(" Canont alloc memory \n");
    exit(-1);
  }
  
  fp = fopen("weightData.txt", "r");
  if(fp == NULL) {
    printf("File weightData.txt not found\n\n");
    exit(-1);
  }
  fread(cv->weight, weightSize, sizeof(float), fp);
  fclose(fp);
  
#if 0
  printf("weight: \n");
  for(i = 0; i < weightSize; i++) {
    printf(" %f ", cv->weight[i]);
  }
printf("\n\n");
#endif

  cv->output = new(_Offload_shared_malloc(64 * outputSize * sizeof(float))) _Cilk_shared float;
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

  /* TODO: what about group? */
  //for (int g = 0; g < c->group_; ++g) {
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
  //}

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
  
    cilk_for(i = 0; i < 64; i++) {
      forward_convolution(cv, i*inputSize, i*outputSize);
    }
  
}


int main()
{
  printf("Checking for Intel(R) Xeon Phi(TM) (Target CPU) devices...\n\n");

  int num_devices = 0;
#ifdef __INTEL_OFFLOAD
  num_devices = _Offload_number_of_devices();
#endif
  printf("Number of Target devices installed: %d\n\n",num_devices);

  init();
   
  double min_cpu = 1e30;
  double startTime = CycleTimer::currentSeconds();

  //_Cilk_offload forward_convolution(cv);
  //forward_convolution();
  _Cilk_offload Offload_work(cv);


  double endTime = CycleTimer::currentSeconds();
  if((endTime - startTime) < min_cpu)
    min_cpu = endTime - startTime;

  std::cout<< "\nTime Time = " << min_cpu * 1000 << "ms\n\n\n";  

  return 0;
}

