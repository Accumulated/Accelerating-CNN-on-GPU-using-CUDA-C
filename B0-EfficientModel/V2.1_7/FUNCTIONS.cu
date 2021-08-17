
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <cusolverDn.h>
#include <cuda_runtime.h>


#include "/content/MBCONVS_float/functionsV2.h"
#include "/content/MBCONVS_float/KERNELSH.h"

static void HandleError( cudaError_t err,
                         char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

float time_defined = 0, tmp_time = 0, total_time_for_layer = 0;; 
cudaEvent_t start_timing, stop_timing;


int show_out = 0;

int total_constant_memory = 0;
                
// Device memory for filters
void DEFINE_FILTERS_FOR_MBCONV(Matrix *D_f1, float *filter1, int h1, int w1, int dens1,
                               Matrix *D_f2, float *filter2, int h2, int w2, int dens2,
                               Matrix *D_f3, float *filter3, int h3, int w3, int dens3,
                               Matrix *D_f4, float *filter4, int h4, int w4, int dens4,
                               Matrix *D_f5, float *filter5, int h5, int w5, int dens5)
{
    // Note: No allocations are done, just pointers point to matrices pre-defined

    // This condition is important as the float * is NULL
    if (MBCONV1_0_flag == 1);
    else
      set_allocate_copy_array_Device(D_f1, filter1,
                                    h1, w1, dens1,
                                    "1st filter allocated");
 
    set_allocate_copy_array_Device(D_f2, filter2,
                                    h2, w2, dens2,
                                    "2nd filter allocated");
 
    set_allocate_copy_array_Device(D_f3, filter3,
                                    h3, w3, dens3,
                                    "3rd filter allocated");

    set_allocate_copy_array_Device(D_f4, filter4,
                                    h4, w4, dens4,
                                    "4th filter allocated");

    set_allocate_copy_array_Device(D_f5, filter5,
                                    h5, w5, dens5,
                                    "5th filter allocated");                                                         
}

// Free the device filters
void FREE_FILTERS_FOR_MBCONV(Matrix *D_f1, Matrix *D_f2, 
                             Matrix *D_f3, Matrix *D_f4,
                             Matrix *D_f5)
{
  cudaFree(D_f1 -> elements);
  cudaFree(D_f2 -> elements);
  cudaFree(D_f3 -> elements);
  cudaFree(D_f4 -> elements);
  cudaFree(D_f5 -> elements);
}

void REDUCTION_SUM(Matrix* Output_Modified, Matrix *sum, Matrix *DMean)
{
    /*
      The mean will be a row vector of 1 x C;
      where C is number of original matrix channels
      All input matrices for this function are device matrices,
      except for sum, it's just a transition that later can be removed 
    */
    
    // Define number of blocks in different directions
    int nbx = 0;
    int nby = 0;
    int nbz = 1;

    size_t size;
    cudaError err;
 
    /*
      Load input Matrix inot device t calculate mean for it
      Unfortionately the code requires to copy the input matrix
    */
 
    Matrix DInputMat;
    // Allocate and set its dimensions as needed from the algorithm
    Set_DeviceMatrix(Output_Modified -> depth, Output_Modified -> height * Output_Modified -> width, 1,
                     &DInputMat, "Copyting Input of reduction mean function into Device memory");

    // Copy input from a device memory to device memory in order to process the elements
    size = DInputMat.height * DInputMat.width * DInputMat.depth * sizeof(float);
    err = cudaMemcpyAsync(DInputMat.elements, Output_Modified -> elements, size, cudaMemcpyDeviceToDevice, 0);
    CheckCudaError("Copying mean matrix elements from ", err);

    /* Starting reduction sum calculations */

    // Note: All blocks are 1D threads. We use Block.x to reduce 1 channel elements'
    // Diffferent block.y to address different number of channels
    nbx = (int)ceil((float)DInputMat.width / (2 * BLOCK_SIZE));
    nby = (int)ceil((float)DInputMat.height);

    if (nbx == 0) nbx = 1;
    if (nby == 0) nby = 1;

    // For loop is held to maintain huge number of summations needed
    for (int i = 0; DInputMat.width != 1; i++)
    {
        dim3 dim_Grid2(nbx, nby, nbz);
        dim3 dim_Block2(BLOCK_SIZE, 1, 1);

        // Make sure to synch between multiple runs
       //cudaDeviceSynchronize();
        
        BN_Kernel_Mean_Reduction <<< dim_Grid2, dim_Block2 >>> (DInputMat.elements,
                                                                DInputMat.height,
                                                                DInputMat.width,
                                                                DInputMat.depth,
                                                                DMean -> elements,
                                                                DMean -> width);

        
        // Save and copy mean values array into the filter array
        size = DMean -> height * DMean -> width * DMean -> depth * sizeof(float);
        err = cudaMemcpyAsync(DInputMat.elements, DMean -> elements, size, cudaMemcpyDeviceToDevice, 0);
        CheckCudaError("Copying mean matrix elements from ", err);

        // Modify filter width to fit into the new elements width
        DInputMat.width = nbx;
        DInputMat.height = nby;

        // Recalculate number of blocks in x direction
        nbx = (int)ceil((float)DInputMat.width / (2 * BLOCK_SIZE));
        nby = (int)ceil((float)DInputMat.height);

        if (nbx == 0) nbx = 1;
        if (nby == 0) nby = 1;

        // Set width of mean matrix to the new number of blocks
        DMean -> width = nbx;
        DMean -> height = nby;
    }
    
    // Set mean matrix to 1 X C X 1 to ease further calculations
    DMean -> height = 1; DMean -> width = Output_Modified -> depth; DMean -> depth = 1;
    
    nbx = (int)ceil((float)DMean -> width / 1024);
    
    dim3 dim_Grid2(nbx, 1, 1);
    dim3 dim_Block2(1024, 1, 1);
    CastingDivision <<<dim_Grid2, dim_Block2>>> (DMean -> elements, DMean -> width, 
                                                 Output_Modified->height * Output_Modified->width);
}


/*  Squeeze_and_Excite(&tmp2, &SE_OUT, F3, F4,
                      FD4, FD3, FD4, FD3);
                      */
// Note: input and output channels args represents the 2 Conv layers output channels respectively
void Squeeze_and_Excite(Matrix* InputIMG, Matrix* Result,
                        Matrix* Filter1, Matrix* Filter2,
                        int FilterDensity2, int FilterDensity1,
                        int input_channels, int output_channels,
                        Matrix * First_bias, Matrix *Second_bias)
{
    /*
       Steps in squeeze and excite layer:
        1. Get mean value for a tensor
        2. pass the mean to the covolution, swish, convolution, sigmoid
        3. the result will be a 1 x 1 x C, multiply elementwise.
          "each element in a channel is multiplied by the result's corresponding channel element"
        
        Filter Density means #filters used
 
      Note: All input matrices are device allocated matrices
    */

 
    /*
      Get mean values for all channels; Dims(1 x InputDepth x 1) 
      Note: Mean matrix is a host allocated memory in REDUCTION_SUM;
            It's used to get the final summation from device and
            then divide each element sequentially by total number 
            of elements. It's then later copied back to Result_Mean
            Matrix which is a device matrix.
            "This can be later changed"
    */
 
    Matrix MEAN, Result_Mean;

    Set_DeviceMatrix(InputIMG -> depth,
                      (int)ceil((double)InputIMG -> height * InputIMG -> width / (2 * BLOCK_SIZE)),
                      1, 
                      &Result_Mean, 
                      "Reesult Mean matrix allocated in device memory");

    REDUCTION_SUM(InputIMG, &MEAN, &Result_Mean);
 

    // Tmp1 is used as a transition between 2 convolution layers; Dims(1 x 1 x FilterDensity3)
    Matrix tmp1;
    Set_DeviceMatrix(1, 1, FilterDensity1, &tmp1, "Allocating tmp1 in device for transition");
 
    // tmp2 matrix is the result from sigmoid function: Dims(1 x 1 x FilterDensity4)
    Matrix tmp2;
    Set_DeviceMatrix( 1, 1, FilterDensity2, &tmp2, "Allocating tmp2 in device for final output");
 
    // Sequence: Conv1x1, swish, Conv1x1, sigmoid 
    // Warning: Remember to pre-process Result_Mean matrix to match 1 x 1 x C as it's the input in this case to Conv2d
    Set_HostMatrix(1, 1, InputIMG -> depth, &Result_Mean);
 
    Conv2d_Layer(&Result_Mean, Filter1, &tmp1, 1, 0, input_channels, output_channels, FilterDensity1,
                 Conv2d_1_x_1, SWISH_ACTIVATION,
                 BIASED, First_bias);
    
    Conv2d_Layer(&tmp1, Filter2, &tmp2, 1, 0, output_channels, input_channels, FilterDensity2,
                 Conv2d_1_x_1, SIGMOID_ACTIVATION,
                 BIASED, Second_bias);
 

    int nbx = (int)ceil((float)InputIMG -> width / DYNAMIC_TILE);
    int nby = (int)ceil((float)InputIMG -> height / DYNAMIC_TILE);
    int nbz = InputIMG -> depth;

    if (nbx == 0) nbx = 1;

    if (nby == 0) nby = 1;

    // This is the only kernel that runs 3d Grid; 
    // Each block in z dimension controls 1 channel  
    dim3 dim_Grid2(nbx, nby, nbz);
    dim3 dim_Block2(DYNAMIC_TILE, DYNAMIC_TILE, 1);

    // C then D, the final multiplication is in C matrix
    ConvChannelElementWiseMultiplication <<< dim_Grid2, dim_Block2 >>> (InputIMG -> elements,
                                                                        InputIMG -> height,
                                                                        InputIMG -> width,
                                                                        InputIMG -> depth,
                                                                        tmp2.elements);

   
    cudaFree(tmp1.elements);
    cudaFree(tmp2.elements);
}

// Warning: Fuction Input matrices are allocated in device memory directly
// InputIMG, FilterK and ConvOut are device memory allocations
void Conv2d_Layer(Matrix* InputIMG, Matrix* FilterK, Matrix* ConvOut,
                  int stride, int padding,
                  int InputChannels, int OutputChannels, int FilterDensity,
                  int Conv_Type, int activation_type,
                  int BIASED_CHOISE, Matrix *biasMat)
{
    //printf("The start of Conv2d layer\n\n");
    
    int OutputHeight = 0, OutputWidth = 0, OutputDepth = 0;

    // 1x1 Conv2d is a special case of Convolution
    if (Conv_Type == Conv2d_1_x_1)
    {
        // Conv2d 1x1 has stride = 1, no padding and K = 1

        /*
          Input Dimensions is the same as Output dimensions
          Only Depth of the output channels differ from input
        */
        OutputHeight = InputIMG -> height; OutputWidth = InputIMG -> width; OutputDepth = FilterDensity;

        /*
          Note: Set_HostMatrix function just changes the dimensions
                so it's okey to use on a device memory
        */
     
        // Modify Filter Matrix to have dimensions ((K^2 * M) x C x 1); K = 1
        Set_HostMatrix(1 * 1 * FilterDensity, InputIMG -> depth, 1, FilterK);

        // Modify Input matrix to have dimensions (C x (H * W) x 1)
        Set_HostMatrix(InputIMG -> depth, InputIMG -> height * InputIMG -> width, 1, InputIMG);

        // Modify Output Matrix preprocessing to have dimesions ((K^2 * M) x (H * W) x 1); K = 1
        Set_HostMatrix(1 * 1 * FilterDensity, OutputWidth * OutputHeight, 1, ConvOut);

        Conv_vidMultiplier(ConvOut, InputIMG, FilterK,
                            OutputHeight, OutputWidth, OutputDepth,
                            Conv2d_1_x_1, 1,
                            activation_type, 
                            BIASED_CHOISE, biasMat);
    }
    else if (Conv_Type == DWConv_k_x_k)
    {
        // Ptr is used to alternate between input image and padding if needed
        Matrix* ptr = InputIMG;

        // DWConv2d has stride = s, padding = p and kernel = k
        OutputHeight = ConvOut -> height; OutputWidth = ConvOut -> width; OutputDepth = ConvOut -> depth;

        Matrix padded_matr;
        if (padding != 0)
        {
            Padding_Zeros_Function(InputIMG, padding, &padded_matr);
            ptr = &padded_matr;
        }
       
        Conv_vidMultiplier(ConvOut, ptr, FilterK,
                            OutputHeight, OutputWidth, OutputDepth,
                            DWConv_k_x_k, stride,
                            activation_type, 
                            BIASED_CHOISE, biasMat);

        // Padded matrix is no longer needed as Convout has the final result
    }
    // Any other kernel size goes here
    else
    {        
        // Regular convolution: Filter and input unrolling
        Matrix* ptr = InputIMG;
        OutputHeight = (ptr -> height + 2 * padding - FilterK -> height) / stride + 1;
        OutputWidth = (ptr -> width + 2 * padding - FilterK -> width) / stride + 1;
        OutputDepth = FilterDensity;

        Matrix padded_matr;
        if (padding != 0)
        {
            Padding_Zeros_Function(InputIMG, padding, &padded_matr);
            ptr = &padded_matr;          
        }

        // 1st phase: Filter unrolling

        // Unrolled filter has dimesnios (M x (C * k * k) x 1)
        Set_HostMatrix(FilterDensity, FilterK -> depth / FilterDensity * FilterK -> height * FilterK -> width,
                      1, FilterK);

        // 2nd phase: Input unrolling

        // The unrolled Input matrix has dimensions((C * k * k) x (H_out * W_out) x 1)
        Matrix INPUT_MODIFIED;
        
        Set_DeviceMatrix(ptr -> depth * 3 * 3,
                        OutputHeight * OutputWidth, 1,
                        &INPUT_MODIFIED, 
                        "Input unrolled Matrix allocated in device memory");

        Input_Unroll_gpu(stride, ptr, &INPUT_MODIFIED, OutputHeight, OutputWidth, 3);

        // Convolution output has dimensions of (M x (H_out * W_out) x 1)
        Set_HostMatrix(FilterDensity, OutputWidth * OutputHeight, 1, ConvOut);

          
        // Perform Multiplication and re-edit the dimensions of output
        Conv_vidMultiplier(ConvOut, &INPUT_MODIFIED, FilterK,
                            OutputHeight, OutputWidth, OutputDepth,
                            Regular_Conv, stride,
                            activation_type,
                            BIASED_CHOISE, biasMat);

    }
 }

// 5 Filters needed to run the 4 layers sequentially
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
    Matrix *MBConv_project_conv_BN_weights,    Matrix *MBConv_project_conv_BN_bias)
{
    /*
      Note: MBConv1_0 doesn't have the expansion conv function;
            The input matrices to this function are device matrices;
            including all the filters, you don't need to allocate or
            copy any thing; just pass to the functions
    */

    /*
      ptr_mat is the pointer that gets past expansion conv;
      Meaning: in case of MBconv1_0 the pointer is same as input matrix;
                in case of any other MBConv6_! it's the output of Conv2d 
                and BN with swish
    */
    
    Matrix H_OUT;
    Matrix tmp1; Matrix *ptr_mat; 
 
    if (MBCONV1_0_flag == 1)
      ptr_mat = Input;
    else
    {     
      Set_DeviceMatrix(Input -> height, Input -> width , FD1, 
                       &tmp1,
                       "Output_1 is allocated in device memory"); 
               
      // 1st layer: 1x1 Conv2d, stride = 1, padding = 0, K = 1
      Conv2d_Layer(Input, F1, &tmp1, 1, 0,
                   input_channels, FD1, FD1,
                   Conv2d_1_x_1,
                   NO_ACTIVATION, 0, NULL);
  
      BN_ALL_PRE_DEFINED(&tmp1, SWISH_ACTIVATION, 
                          MBConv_expansion_conv_BN_mean,    MBConv_expansion_conv_BN_variance,
                          MBConv_expansion_conv_BN_weights, MBConv_expansion_conv_BN_bias);
      ptr_mat = &tmp1;
    }

   // 2nd Layer: KxK DWconv, stride = s, padding = p, K = k

    // Height and width changes, Only depth remains still
    int OutputHeight = (ptr_mat -> height + 2 * padding - FilterSizeDW)/Stride + 1;
    int OutputWidth = (ptr_mat -> width + 2 * padding - FilterSizeDW)/Stride + 1;
    int OutputDepth = ptr_mat -> depth;
 
    // Set and allocate tmp2 matrix; it's a transistion between expansion and squeeze
    Matrix tmp2;
    Set_DeviceMatrix(OutputHeight, OutputWidth, OutputDepth, &tmp2,
                    "Output_2 is allocated in device memory");    

    Conv2d_Layer(ptr_mat, F2, &tmp2,
                 Stride, padding, FD1, FD2, FD2, DWConv_k_x_k,
                 NO_ACTIVATION, 0, NULL);
  

    BN_ALL_PRE_DEFINED(&tmp2, SWISH_ACTIVATION, 
                       MBConv_depthwise_conv_BN_mean,     MBConv_depthwise_conv_BN_variance,
                       MBConv_depthwise_conv_BN_weights,  MBConv_depthwise_conv_BN_bias);
  
    // 3rd Layer: squeeze and excitation

    /*
      Squeeze excite layer doesn't change the final output dimensions;
      SE_OUT can be removed; Do so later
    */
 
    Matrix *SE_OUT;
    Squeeze_and_Excite(&tmp2, SE_OUT, F3, F4,
                        FD4, FD3, FD4, FD3,
                        bias1, bias2);

    // 4th Layer: 1x1 Conv2d
    // MBConv output pointer is set and finally updated after this layer execution
    Set_DeviceMatrix(tmp2.height, tmp2.width, FD5, MBConvOut,
                     "Matrix final output is allocated in device memory");
 

    // 1x1 Conv2d layer
    Conv2d_Layer(&tmp2, F5, MBConvOut, 1, 0, FD4, FD5, FD5, Conv2d_1_x_1,
                 NO_ACTIVATION, 0, NULL);


    // BatchNorm layer
    BN_ALL_PRE_DEFINED(MBConvOut, NO_ACTIVATION, 
                       MBConv_project_conv_BN_mean,     MBConv_project_conv_BN_variance,
                       MBConv_project_conv_BN_weights,  MBConv_project_conv_BN_bias);

    // Skip identity layer
    if(skip)
    {
      MBConv_SKIP_IDENTITY(MBConvOut, Input);
    }
}


void MBConv_SKIP_IDENTITY(Matrix *parent, Matrix *child)
{
    int nbx = (int)ceil((float)parent -> width / DYNAMIC_TILE);
    int nby = (int)ceil((float)parent -> height / DYNAMIC_TILE);
    int nbz = parent -> depth;

    if (nbx == 0) nbx = 1;

    if (nby == 0) nby = 1;

    // This is the only kernel that runs 3d Grid; 
    // Each block in z dimension controls 1 channel  
    dim3 dim_Grid2(nbx, nby, nbz);
    dim3 dim_Block2(DYNAMIC_TILE, DYNAMIC_TILE, 1);
     
    Identity_Skip <<<dim_Grid2, dim_Block2 >>> (parent -> elements,
                                                  parent -> height,
                                                  parent -> width,
                                                  parent -> depth, 
                                                  child -> elements);
}

void BN_ALL_PRE_DEFINED(Matrix* D_input, int activate, Matrix *mean, Matrix *variance, Matrix *weights, Matrix *bias)
{
    /* The ptr matrix is a device matrix */
     
    /*
      All weights, bias, running mean and running variance
      are pre-defined. Just call the function and use the
      matrices.
      
      All bias, weights, mean and bariance matrices are 1x1xC

      Output Matrix is modified by the equation
      (y = ((x - Mean) / (sqrt(variance) + epsilon)) * weights + bais)
    */

    int nbx = (int)ceil((float)D_input -> width / DYNAMIC_TILE);
    int nby = (int)ceil((float)D_input -> height / DYNAMIC_TILE);
    int nbz = D_input -> depth;

    if (nbx == 0) nbx = 1;
    if (nby == 0) nby = 1;

    // This is the only kernel that runs 3d Grid; 
    // Each block in z dimension controls 1 channel  
    dim3 dim_Grid3(nbx, nby, nbz);
    dim3 dim_Block3(DYNAMIC_TILE, DYNAMIC_TILE, 1);

    BN_Kernel_Final_Layer <<< dim_Grid3, dim_Block3 >>> (D_input -> elements,
                                                         D_input -> height,
                                                         D_input -> width,
                                                         D_input -> depth,
                                                         mean -> elements, variance -> elements,
                                                         weights -> elements, bias -> elements,
                                                         activate);
}

void Padding_Zeros_Function(Matrix* Original_Matrix_Before, int padding_Value, Matrix* padded_Matrix)
{
    /* 
      Note: Matrix coming is a device elemente matrix;
            Original Matrix is a Device input that needs padding
            padded_Matrix is the return of this function;

      Warning: Padded_Matrix has a different size than the Original 
                non padded matrix and it's not allocated in device yet.
                The allocateion is done inside this function.
    */    

    Set_DeviceMatrix(Original_Matrix_Before->height + 2 * padding_Value,
                      Original_Matrix_Before->width + 2 * padding_Value,
                      Original_Matrix_Before->depth,
                      padded_Matrix,
                      "Padded Matrix is allocated in device memory.");

    // 1st: Set padded Matrix with all zeros
    cudaMemset(padded_Matrix -> elements,
               0, padded_Matrix->height * padded_Matrix->width * padded_Matrix->depth * sizeof(float)); 

    int nbx = (int)ceil((float)padded_Matrix -> width / DYNAMIC_TILE);
    int nby = (int)ceil((float)padded_Matrix -> height / DYNAMIC_TILE);
    int nbz = padded_Matrix -> depth;

    if (nbx == 0) nbx = 1;

    if (nby == 0) nby = 1;

    dim3 dim_Grid2(nbx, nby, nbz);
    dim3 dim_Block2(DYNAMIC_TILE, DYNAMIC_TILE, 1);

    // Pass to the copying strided kernel to complete the padding process
 
    Complete_Padding_Process <<< dim_Grid2, dim_Block2 >>> (padded_Matrix -> elements,
                                                            padded_Matrix -> height,
                                                            padded_Matrix -> width,
                                                            padded_Matrix -> depth,
                                                            Original_Matrix_Before -> elements,
                                                            Original_Matrix_Before -> height,
                                                            Original_Matrix_Before -> width,
                                                            Original_Matrix_Before -> depth,
                                                            padding_Value);
}


// Call this function directly for 1x1 conv2d. Don't call for DWConv
void Conv_vidMultiplier(Matrix* out_11, Matrix* D_2, Matrix* D_1,
                        int ReconstructOutHieght, int ReconstructOutWidth, int ReconstructOutDepth,
                        int ConvType, int stride_DW, int activation_type, int BIASED_CHOISE, Matrix *biasMat)
{
    /* Note: Out_11, XXX_Trans and Host_Conv_Filter are device matrices */
 
    // The multiplication kernel is used for the 1x1 Conv2d and kxk Conv2d
    if (ConvType == Conv2d_1_x_1 || ConvType == Regular_Conv)
    {    
        // Get number of blocks
        int nbx = (int)ceil((float)out_11 -> width / (THREAD_GRANULARITY_BLOCKS * Tile_GEMM));
        int nby = (int)ceil((float)out_11 -> height / Tile_GEMM);
        int num_block_for_phases = (int)ceil((float)D_1 -> width / Tile_GEMM);

        // Check for zero blocks to make sure code runs correctly
        if (nbx == 0) nbx = 1;
        if (nby == 0) nby = 1;

        dim3 dim_Grid2(nbx, nby, 1);
        dim3 dim_Block2(Tile_GEMM, Tile_GEMM, 1);
     
        if (BIASED_CHOISE == BIASED)
        {
          Set_HostMatrix(out_11 -> height, 1, 1, biasMat);

          // Call shared memory tiled Multiplication  algorithm
          MatrixMulKernel <<< dim_Grid2, dim_Block2 >>> (D_1 -> elements, D_1 -> height, D_1 -> width, D_1 -> depth,
                                                         D_2 -> elements, D_2 -> height, D_2 -> width, D_2 -> depth,
                                                         out_11 -> elements, out_11 -> height, out_11 -> width, out_11 -> depth,
                                                         num_block_for_phases, activation_type,
                                                         BIASED_CHOISE, biasMat -> elements);         
        }
        else
        {
          MatrixMulKernel <<< dim_Grid2, dim_Block2 >>> (D_1 -> elements, D_1 -> height, D_1 -> width, D_1 -> depth,
                                                         D_2 -> elements, D_2 -> height, D_2 -> width, D_2 -> depth,
                                                         out_11 -> elements, out_11 -> height, out_11 -> width, out_11 -> depth,
                                                         num_block_for_phases, activation_type,
                                                         BIASED_CHOISE, NULL);        
         }    
    }

    // This case is for DWConv2d
    else
    {
        int nbx = (int)ceil((float)out_11 -> width / TileDW);
        int nby = (int)ceil((float)out_11 -> height / TileDW);
        int nbz = out_11 -> depth;
     
        if (nbx == 0) nbx = 1;
        if (nby == 0) nby = 1;

        // This is the only kernel that runs 3d Grid; 
        // Each block in z dimension controls 1 channel  
        dim3 dim_Grid2(nbx, nby, nbz);
        dim3 dim_Block2(TileDW, TileDW, 1);


        DWConv2d_kernel << < dim_Grid2, dim_Block2 >> > (D_2 -> elements, D_2 -> height, D_2 -> width, D_2 -> depth,
                                                         D_1 -> elements, D_1 -> height, D_1 -> width, D_1 -> depth,
                                                         out_11 -> elements, out_11 -> height, out_11 -> width, out_11 -> depth,
                                                         stride_DW);          
    }
 
    // Reset the output dimensions to continue in the network
    Set_HostMatrix(ReconstructOutHieght, ReconstructOutWidth, ReconstructOutDepth, out_11);
}

void Input_Unroll_gpu(int st_stride, Matrix* Device_Input, Matrix* Device_Unrolled, int O_H, int O_W, int Filter_Size)
{   
    /* Note: All the function input matrices are device matrices.
            Device_Input matrix is already allocated and ready.
            Device_Unrolled matrix is already allocated and ready. 
    */
    
    int nbx = (int)ceil((float)O_W / TileDW);
    int nby = (int)ceil((float)O_H / TileDW);
    int nbz = Device_Input -> depth;

    if (nbx == 0) nbx = 1;

    if (nby == 0) nby = 1;
 
    dim3 dim_Grid2(nbx, nby, nbz);
    dim3 dim_Block2(TileDW, TileDW, 1);

    // You need to use cudaDeviceSynchronize if the kernel isn't working

    INPUT_UNROLLING <<< dim_Grid2, dim_Block2 >>> (st_stride, Filter_Size,
                                                   
                                                   Device_Input -> elements,
                                                   Device_Input -> height,
                                                   Device_Input -> width,
                                                   Device_Input -> depth,

                                                   Device_Unrolled -> elements,
                                                   Device_Unrolled -> height,
                                                   Device_Unrolled -> width,
                                                   Device_Unrolled -> depth,

                                                   O_H, O_W);

    
    //cudaDeviceSynchronize(); 

    cudaError err = cudaGetLastError();

    if ( err != cudaSuccess )
    {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
      exit(-1);
    } 
}

void DEFINE_FILTERS_FOR_MBCONV_BN(  Matrix *EXP_MEAN, 		  float *filter1, int size_1,
                                    Matrix *EXP_VARIANCE, 	float *filter2, int size_2,
                                    Matrix *EXP_WEIGHTS, 	  float *filter3, int size_3,
                                    Matrix *EXP_BIAS, 		  float *filter4, int size_4,
                                  
                                    Matrix *DW_MEAN, 		    float *filter5, int size_5,
                                    Matrix *DW_VARIANCE, 	  float *filter6, int size_6,
                                    Matrix *DW_WEIGHTS, 		float *filter7, int size_7,
                                    Matrix *DW_BIAS, 		    float *filter8, int size_8,
                                    
                                    Matrix *PRJ_MEAN, 		  float *filter9,  int size_9,
                                    Matrix *PRJ_VARIANCE, 	float *filter10, int size_10,
                                    Matrix *PRJ_WEIGHTS, 	  float *filter11, int size_11,
                                    Matrix *PRJ_BIAS, 		  float *filter12, int size_12)
{
  if (MBCONV1_0_flag);
  else
  {
    set_allocate_copy_array_Device(EXP_MEAN, filter1,
                      size_1, 1, 1,
                      "expand mean"); 
    set_allocate_copy_array_Device(EXP_VARIANCE, filter2,
                      size_2, 1, 1,
                      "expand variance"); 
    set_allocate_copy_array_Device(EXP_WEIGHTS, filter3,
                      size_3, 1, 1,
                      "expand weights"); 
    set_allocate_copy_array_Device(EXP_BIAS, filter4,
                      size_4, 1, 1,
                      "expand bias");
  } 
									  
	set_allocate_copy_array_Device(DW_MEAN, filter5,
									  size_5, 1, 1,
									  "DW mean"); 
	set_allocate_copy_array_Device(DW_VARIANCE, filter6,
									  size_6, 1, 1,
									  "DW variance"); 
	set_allocate_copy_array_Device(DW_WEIGHTS, filter7,
									  size_7, 1, 1,
									  "DW weights"); 
	set_allocate_copy_array_Device(DW_BIAS, filter8,
									  size_8, 1, 1,
									  "expand bias"); 

	set_allocate_copy_array_Device(PRJ_MEAN, filter9,
									  size_9, 1, 1,
									  "DW mean"); 
	set_allocate_copy_array_Device(PRJ_VARIANCE, filter10,
									  size_10, 1, 1,
									  "DW variance"); 
	set_allocate_copy_array_Device(PRJ_WEIGHTS, filter11,
									  size_11, 1, 1,
									  "DW weights"); 
	set_allocate_copy_array_Device(PRJ_BIAS, filter12,
									  size_12, 1, 1,
									  "expand bias"); 
}

// 3 Sequential Operations: Same as "set_allocate_copy_Matrix_Device",
// However, it uses a pointer to float as a parent.
void set_allocate_copy_array_Device(Matrix *child, float *parent,
									int height, int width, int depth,
									char *notification)
{
	Set_DeviceMatrix(height, width, depth, child, notification);

	size_t size = height * width * depth * sizeof(float);
 
	cudaError err = cudaMemcpy(child -> elements, parent, size,
								cudaMemcpyHostToDevice);
  
	CheckCudaError(notification, err);
}

// 3 Sequential Operations: Set dimensions, allocate device memory and copy.
void set_allocate_copy_Matrix_Device(Matrix *child, Matrix *parent, char *notification)
{
	Set_DeviceMatrix(parent -> height, parent -> width, parent -> depth,
					          child, notification);

	size_t size = parent -> height * parent -> width * parent -> depth * sizeof(float);
	
	cudaError err = cudaMemcpy(child -> elements, parent -> elements,
								              size, cudaMemcpyHostToDevice);
	CheckCudaError(notification, err);
}

void set_allocate_copy_Matrix_Device_specific(Matrix *child, Matrix *parent, char *notification, int height, int width, int depth)
{
	Set_DeviceMatrix(height, width, depth, child, notification);

	size_t size = child -> height * child -> width * child -> depth * sizeof(float);
	

	cudaError err = cudaMemcpy(child -> elements, parent -> elements,
								            size, cudaMemcpyHostToDevice);

  
	CheckCudaError(notification, err);
}

void just_copy_HTD(Matrix *child, Matrix *parent, char *notification)
{
    // Read C from device memory
  size_t size = parent -> width * parent -> height * parent -> depth * sizeof(float);
    
	cudaError err = cudaMemcpy(child -> elements, parent -> elements, size, cudaMemcpyHostToDevice);

  
	CheckCudaError(notification, err);
}

void just_copy_DTH(Matrix *child, Matrix *parent, char *notification)
{
  // Read C from device memory
  size_t size = parent -> width * parent -> height * parent -> depth * sizeof(float);
  

	cudaError err = cudaMemcpy(child -> elements, parent -> elements, size, cudaMemcpyDeviceToHost);
  
	CheckCudaError(notification, err);
}

void set_allocate_Host(Matrix *ptr, int height, int width, int depth)
{
	// Note this function allocates memory, remember to free 
	Set_HostMatrix(height, width, depth, ptr);
	
	int Fsize = height * width * depth* sizeof(float);
 
	ptr -> elements = (float *) malloc(Fsize);
}

void FreeHost_Allocated(Matrix *ptr)
{
	free(ptr -> elements);
}

// Allocations for Device matrices
void Set_DeviceMatrix(int height, int width, int depth, Matrix* ptr, char* NamePtr)
{
    ptr->width = width;
    ptr->height = height;
    ptr->depth = depth;

    size_t size = width * height * depth * sizeof(float);
    cudaError err = cudaMalloc((void **)&(ptr->elements), size);
    CheckCudaError(NamePtr, err);
}

void Set_HostMatrix(int height, int width, int depth, Matrix* ptr)
{
    ptr->width = width;
    ptr->height = height;
    ptr->depth = depth;
}

void CheckCudaError(char* ptr, cudaError err)
{
    if (err == cudaSuccess);
    else
        printf("CUDA error in %s: %s\n", ptr, cudaGetErrorString(err));
}


void show_me_enhanced(Matrix* ptr, char* NamePtr)
{
    if(show_out == 1)
    {
      setvbuf(stdout, NULL, _IOLBF, 0);

          printf("%s,"
              "it has height = %d, "
              "width = %d, "
              "depth = %d \n",
              NamePtr, ptr->height, ptr->width, ptr->depth);

          printf("{\n");
          for (int i = 0; i < ptr -> height * ptr -> width * ptr -> depth; i++)
          {
              if (i % ptr->width == 0 && i >= ptr->width)
                  printf("\n");

              if (i % (ptr->width * ptr->height) == 0 && i >= (ptr->width * ptr->height));
                  //printf("\n");

              printf("%.8f", ptr->elements[i]);
              if (i + 1 == ptr->height * ptr->width * ptr->depth);
              else
                  printf(", ");
          }

          printf("} \n");
          printf("\n");

          setvbuf(stdout, NULL, _IOLBF, 0);        
    }
}


void start()
{
  HANDLE_ERROR(cudaEventCreate(&start_timing));
  HANDLE_ERROR(cudaEventCreate(&stop_timing));
  HANDLE_ERROR(cudaEventRecord(start_timing, 0));
}

void stop(char *notification, int pause_time)
{
  HANDLE_ERROR(cudaEventRecord(stop_timing, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop_timing));
  HANDLE_ERROR(cudaEventElapsedTime(&time_defined, start_timing, stop_timing));
 
  if(pause_time)
 {
    tmp_time += time_defined; 
 }   
 
  else
  {
    tmp_time = 0;
    printf("Time elapsed for %s:  %.8f ms\n", notification, time_defined);  
    total_time_for_layer += time_defined;
  }
}

void after_pause(char *notification)
{
  printf("Time elapsed for %s: %.8f ms\n", notification, tmp_time); 
  total_time_for_layer += tmp_time;
 
  tmp_time = 0;         
}

void reset_time()
{
  printf("Total time: %.8f ms\n", total_time_for_layer); 
  total_time_for_layer = 0;
}

void show_me_enhanced_from_devince(Matrix *ptr, char *notification)
{
    Matrix H_OUT;

    set_allocate_Host(&H_OUT, ptr -> height, ptr -> width, ptr -> depth);

    just_copy_DTH(&H_OUT, ptr, "show_device_elements");
  
    show_out = 1;
    show_me_enhanced(&H_OUT, notification);
    show_out = 0;  
}