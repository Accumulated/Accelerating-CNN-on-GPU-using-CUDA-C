#ifndef KERNELSH_H_
#define KERNELSH_H_

#define THREAD_GRANULARITY_BLOCKS	2

extern const int BLOCK_SIZE;
extern int MBCONV1_0_flag;
__global__ void CastingDivision(float *A, int W1, float B);

__global__ void INPUT_UNROLLING(int stride, int Filter_Height,
                                float *Input, int H1, int W1, int D1,
                                float *X_unrolled, int H2, int W2, int D2,
                                int Output_Height, int Output_Width);
								
								
__global__ void DWConv2d_kernel(float *Input, int H1, int W1, int D1,
                                float *Filter, int H2, int W2, int D2,
                                float *Output, int H3, int W3, int D3,
                                int stride);
								
								
__global__ void MatrixMulKernel(float *M, int H1, int W1, int D1,
                                float *N, int H2, int W2, int D2,
                                float *P, int H3, int W3, int D3,
                                int num_blocks, int activation, 
                                int IS_BIASED, float *bias_mat);
								
__global__ void ConvChannelElementWiseMultiplication(float *A, int H1, int W1, int D1,
                                                     float *B);
													 
__global__ void Identity_Skip(float *A,  int H1, int W1, int D1,
                              float *B);
							  
__global__ void Complete_Padding_Process(float *Original_Padded, int H1, int W1, int D1, 
                                         float *Original,        int H2, int W2, int D2,
                                         int padding_value);
										 
__global__ void BN_Kernel_Mean_Reduction(float *input, int H1, int W1, int D1,
                                         float *Mean, int W2);
										 
										 
__global__ void ElementWiseSquaring(float *A, int H1, int W1, int D1);

__global__ void ElementWiseSubtraction(float *A, int H1, int W1, int D1,
                                       float *mean);
									   

__global__ void BN_Kernel_Final_Layer(float *A, int H1, int W1, int D1, 
                                      float *D_mean, float *D_variance,
                                      float *D_weight, float *D_bias,
                                      int activate);
									  
									  

										 
#endif			 
										 