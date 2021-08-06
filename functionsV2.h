#ifndef functions_H_
#define functions_H_

const int Tile_GEMM = 16;
const int TileDW = 16;

const int NO_ACTIVATION = 0;
const int SWISH_ACTIVATION = 1;
const int SIGMOID_ACTIVATION = 2;

const int BIASED = 1;

const int Conv2d_1_x_1 = 1;
const int DWConv_k_x_k = 2;
const int Regular_Conv = 3;

typedef struct {
    int width;
    int height;
    int depth;
    float* elements;
}
Matrix;

void show_me_enhanced(Matrix* ptr, char* NamePtr);

void BN_ALL_PRE_DEFINED(Matrix* D_input, int activate, 
			Matrix *mean, Matrix *variance, 
			Matrix *weights, Matrix *bias);

void set_allocate_copy_Matrix_Device(Matrix *child, Matrix *parent, char *notification);

void just_copy_DTH(Matrix *child, Matrix *parent, char *notification);

void just_copy_HTD(Matrix *child, Matrix *parent, char *notification);

void set_allocate_Host(Matrix *ptr, int height, int width, int depth);

void set_allocate_copy_array_Device(Matrix *child, float *parent, 
				    int height, int width, 
				    int depth, char *notification);

void set_allocate_copy_Matrix_Device_specific(Matrix *child, Matrix *parent, 
					      char *notification, int height,
					      int width, int depth);

void REDUCTION_SUM(Matrix* Output_Modified, Matrix *sum, Matrix *DMean);

// Padding function used in convolution
void Padding_Zeros_Function(Matrix* Original_Matrix_Before,
			    int padding_Value, Matrix* padded_Matrix);

// Input unrolling function that calls the unrolling kernel
void Input_Unroll_gpu(int st_stride, Matrix* Device_Input, 
		      Matrix* Device_Unrolled, int O_H, 
		      int O_W, int Filter_Size);

// The final matrix multiplication for output calculation of Convolution
void Conv_vidMultiplier(Matrix* out_11, Matrix* D_2, Matrix* D_1,
                        int ReconstructOutHieght, int ReconstructOutWidth, int ReconstructOutDepth,
                        int ConvType, int stride_DW, int activation_type,
			int BIASED_CHOISE, Matrix *biasMat);
						
// Check for errors for efficient code writing
void CheckCudaError(char* ptr, cudaError err);

// Set any Device matrix dimensions and allocations
void Set_DeviceMatrix(int height, int width, int depth, Matrix* ptr, char* NamePtr);

void Set_HostMatrix(int height, int width, int depth, Matrix* ptr);

// Convolution funcion
void Conv2d_Layer(Matrix* InputIMG, Matrix* FilterK, Matrix* ConvOut,
                  int stride, int padding,
                  int InputChannels, int OutputChannels, int FilterDensity,
                  int Conv_Type, int activation_type,
                  int BIASED_CHOISE, Matrix *biasMat);

// Squeeze function
void Squeeze_and_Excite(Matrix* InputIMG, Matrix* Result,
			Matrix* Filter1, Matrix* Filter2,
			int FilterDensity2, int FilterDensity1,
			int input_channels, int output_channels,
			float * First_bias, float *Second_bias);

// MBConv function
void MBConv_Layer(Matrix* Input, Matrix* MBConvOut,
		Matrix* F1, Matrix* F2, Matrix* F3, Matrix* F4, Matrix* F5,
		int FD1, int FD2, int FD3, int FD4, int FD5,
		int input_channels, int output_channels, int FilterSizeDW,
		int Stride, int padding, int skip,
		Matrix *bias1, Matrix *bias2,
		Matrix *MBConv_expansion_conv_BN_mean,     Matrix *MBConv_expansion_conv_BN_variance,
		Matrix *MBConv_expansion_conv_BN_weights,  Matrix *MBConv_expansion_conv_BN_bias,
		Matrix *MBConv_depthwise_conv_BN_mean,     Matrix *MBConv_depthwise_conv_BN_variance,
		Matrix *MBConv_depthwise_conv_BN_weights,  Matrix *MBConv_depthwise_conv_BN_bias,
		Matrix *MBConv_project_conv_BN_mean,       Matrix *MBConv_project_conv_BN_variance,
		Matrix *MBConv_project_conv_BN_weights,    Matrix *MBConv_project_conv_BN_bias);

void DEFINE_FILTERS_FOR_MBCONV(Matrix *f1, float *filter1, int h1, int w1, int dens1,
                               Matrix *f2, float *filter2, int h2, int w2, int dens2,
                               Matrix *f3, float *filter3, int h3, int w3, int dens3,
                               Matrix *f4, float *filter4, int h4, int w4, int dens4,
                               Matrix *f5, float *filter5, int h5, int w5, int dens5);


void DEFINE_FILTERS_FOR_MBCONV_BN(  Matrix *EXP_MEAN, 		  	float *filter1, 	int size_1,
                                    Matrix *EXP_VARIANCE, 		float *filter2, 	int size_2,
                                    Matrix *EXP_WEIGHTS, 	  	float *filter3, 	int size_3,
                                    Matrix *EXP_BIAS, 		  	float *filter4, 	int size_4,
                                  
                                    Matrix *DW_MEAN, 		    	float *filter5, 	int size_5,
                                    Matrix *DW_VARIANCE, 	  	float *filter6, 	int size_6,
                                    Matrix *DW_WEIGHTS, 		float *filter7, 	int size_7,
                                    Matrix *DW_BIAS, 		    	float *filter8, 	int size_8,
                                    
                                    Matrix *PRJ_MEAN, 		  	float *filter9, 	int size_9,
                                    Matrix *PRJ_VARIANCE, 		float *filter10, 	int size_10,
                                    Matrix *PRJ_WEIGHTS, 	  	float *filter11, 	int size_11,
                                    Matrix *PRJ_BIAS, 		  	float *filter12, 	int size_12);

                               
void MBConv_SKIP_IDENTITY(Matrix *parent, Matrix *child);

void STEM_LAYER(Matrix *DInput_Mat, Matrix *F_STEM,
		int image_height, int image_width, int image_depth,
		int filter_height, int filter_width, int filter_depth, int filter_density,
		int padding, int stride,
		Matrix *STEM_OUT);
				  
void HEAD_LAYER(Matrix *INPUT_MATRIX, Matrix *F_HEAD, Matrix *FC_WEIGHTS,
                int filter_height, int filter_width, int filter_depth, int filter_density,
                int padding, int stride,
                Matrix *HEAD_OUT);

void FREE_FILTERS_FOR_MBCONV(Matrix *D_f1, Matrix *D_f2, 
                             Matrix *D_f3, Matrix *D_f4,
                             Matrix *D_f5);

void show_me_enhanced_from_devince(Matrix *ptr, char *notification);;


void start();

void stop(char *notification, int pause_time);

void after_pause(char *notification);

void reset_time();


#endif
