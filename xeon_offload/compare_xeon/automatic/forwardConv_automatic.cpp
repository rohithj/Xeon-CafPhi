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
#include <mkl.h>

#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include "CycleTimer.h"


#define MIN_VAL         (-1)
#define MAX_ITERATIONS  100

#define IN_CHANNELS     5
#define OUT_CHANNELS    10

#define IN_HEIGHT       100
#define IN_WIDTH        60

#define OUT_HEIGHT      80
#define OUT_WIDTH       60

#define KERNEL_HEIGHT   8
#define KERNEL_WIDTH    8

#define NUM             64


float *input;
float *output;
float *weight;
float *col_buff;

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
int colBuffSize = conv_in_channels_ * kernel_w_ * kernel_h_ * conv_in_width_ * conv_in_height_;

void init()
{
  int i;
  float randVal;
  
  int totalInputSize = inputSize * num_;
  int totalOutputSize = outputSize * num_; 
  int totalColBuffSize = colBuffSize * num_;

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
  
  col_buff = (float *)malloc(sizeof(float) * totalColBuffSize);
  if(col_buff == NULL) {
    printf(" Canont alloc memory \n");
    exit(-1);
  }
  
  output = (float *)malloc(sizeof(float) * totalOutputSize);
  if(output == NULL) {
    printf(" Canont alloc memory \n");
    exit(-1);
  }
}

void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

int forward_convolution(int inputOffset, int outputOffset, int colOffset)
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

  float *data_in = input + inputOffset;
  float *data_col = col_buff + colOffset;

  int height_col = (conv_in_height_ - kernel_h_) + 1;
  int width_col = (conv_in_width_ - kernel_w_) + 1;
  int channels_col = conv_in_channels_ * kernel_h_ * kernel_w_;
  
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w_;
    int h_offset = (c / kernel_w_) % kernel_h_;
    int c_im = c / kernel_h_ / kernel_w_;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h + h_offset;
        int w_pad = w + w_offset;
        if (h_pad >= 0 && h_pad < conv_in_height_ && w_pad >= 0 && 
            w_pad < conv_in_width_)
          data_col[(c * height_col + h) * width_col + w] = 
            data_in[(c_im * conv_in_height_ + h_pad) * conv_in_width_ + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
  
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, conv_out_channels_,
          height_out_ * width_out_, kernel_h_ * kernel_w_,
          1., weight, data_col,
          0., output + outputOffset);
  
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
  for(j = 0; j < MAX_ITERATIONS; j++) {
    for(i = 0; i < num_; i++) {
      forward_convolution(i * inputSize, i * outputSize, i * colBuffSize);
    }
  }

  /* Stop timer and print */
  double endTime = CycleTimer::currentSeconds();
  if((endTime - startTime) < min_cpu)
    min_cpu = endTime - startTime;

  std::cout<< "\nTime Time for automatic = " << min_cpu * 1000 << "ms\n\n\n";  

  return 0;
}

