
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <cusolverDn.h>
#include <cuda_runtime.h>


#include "/content/MBCONVS_float/Input_For_Stem_Layer.h"
#include "/content/MBCONVS_float/Stem/Stem_conv_parameters.h"
#include "/content/MBCONVS_float/functionsV2.h"
#include "/content/MBCONVS_float/CONFIG.h"
#include "/content/MBCONVS_float/Input_Matrix.h"
#include "/content/MBCONVS_float/KERNELSH.h"

#include "/content/MBCONVS_float/MBConv1_0/MBConv1_0_depthwise_conv_parameters.h"
#include "/content/MBCONVS_float/MBConv1_0/MBConv1_0_project_conv_parameters.h"
#include "/content/MBCONVS_float/MBConv1_0/MBConv1_0_squeeze_excitation_parameters.h"

#include "/content/MBCONVS_float/MbConv6_1/MBConv6_1_expansion_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_1/MBConv6_1_depthwise_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_1/MBConv6_1_squeeze_excitation_parameters.h"
#include "/content/MBCONVS_float/MbConv6_1/MBConv6_1_project_conv_parameters.h"

#include "/content/MBCONVS_float/MbConv6_2/MBConv6_2_expansion_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_2/MBConv6_2_depthwise_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_2/MBConv6_2_squeeze_excitation_parameters.h"
#include "/content/MBCONVS_float/MbConv6_2/MBConv6_2_project_conv_parameters.h"

#include "/content/MBCONVS_float/MbConv6_3/MBConv6_3_expansion_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_3/MBConv6_3_depthwise_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_3/MBConv6_3_squeeze_excitation_parameters.h"
#include "/content/MBCONVS_float/MbConv6_3/MBConv6_3_project_conv_parameters.h"

#include "/content/MBCONVS_float/MbConv6_4/MBConv6_4_expansion_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_4/MBConv6_4_depthwise_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_4/MBConv6_4_squeeze_excitation_parameters.h"
#include "/content/MBCONVS_float/MbConv6_4/MBConv6_4_project_conv_parameters.h"

#include "/content/MBCONVS_float/MbConv6_5/MBConv6_5_expansion_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_5/MBConv6_5_depthwise_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_5/MBConv6_5_squeeze_excitation_parameters.h"
#include "/content/MBCONVS_float/MbConv6_5/MBConv6_5_project_conv_parameters.h"

#include "/content/MBCONVS_float/MbConv6_6/MBConv6_6_expansion_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_6/MBConv6_6_depthwise_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_6/MBConv6_6_squeeze_excitation_parameters.h"
#include "/content/MBCONVS_float/MbConv6_6/MBConv6_6_project_conv_parameters.h"

#include "/content/MBCONVS_float/MbConv6_7/MBConv6_7_expansion_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_7/MBConv6_7_depthwise_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_7/MBConv6_7_squeeze_excitation_parameters.h"
#include "/content/MBCONVS_float/MbConv6_7/MBConv6_7_project_conv_parameters.h"

#include "/content/MBCONVS_float/MbConv6_8/MBConv6_8_expansion_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_8/MBConv6_8_depthwise_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_8/MBConv6_8_squeeze_excitation_parameters.h"
#include "/content/MBCONVS_float/MbConv6_8/MBConv6_8_project_conv_parameters.h"

#include "/content/MBCONVS_float/MbConv6_9/MBConv6_9_expansion_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_9/MBConv6_9_depthwise_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_9/MBConv6_9_squeeze_excitation_parameters.h"
#include "/content/MBCONVS_float/MbConv6_9/MBConv6_9_project_conv_parameters.h"

#include "/content/MBCONVS_float/MbConv6_10/MBConv6_10_expansion_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_10/MBConv6_10_depthwise_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_10/MBConv6_10_squeeze_excitation_parameters.h"
#include "/content/MBCONVS_float/MbConv6_10/MBConv6_10_project_conv_parameters.h"

#include "/content/MBCONVS_float/MbConv6_11/MBConv6_11_expansion_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_11/MBConv6_11_depthwise_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_11/MBConv6_11_squeeze_excitation_parameters.h"
#include "/content/MBCONVS_float/MbConv6_11/MBConv6_11_project_conv_parameters.h"

#include "/content/MBCONVS_float/MbConv6_12/MBConv6_12_expansion_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_12/MBConv6_12_depthwise_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_12/MBConv6_12_squeeze_excitation_parameters.h"
#include "/content/MBCONVS_float/MbConv6_12/MBConv6_12_project_conv_parameters.h"

#include "/content/MBCONVS_float/MbConv6_13/MBConv6_13_expansion_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_13/MBConv6_13_depthwise_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_13/MBConv6_13_squeeze_excitation_parameters.h"
#include "/content/MBCONVS_float/MbConv6_13/MBConv6_13_project_conv_parameters.h"

#include "/content/MBCONVS_float/MbConv6_14/MBConv6_14_expansion_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_14/MBConv6_14_depthwise_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_14/MBConv6_14_squeeze_excitation_parameters.h"
#include "/content/MBCONVS_float/MbConv6_14/MBConv6_14_project_conv_parameters.h"

#include "/content/MBCONVS_float/MbConv6_15/MBConv6_15_expansion_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_15/MBConv6_15_depthwise_conv_parameters.h"
#include "/content/MBCONVS_float/MbConv6_15/MBConv6_15_squeeze_excitation_parameters.h"
#include "/content/MBCONVS_float/MbConv6_15/MBConv6_15_project_conv_parameters.h"

#include "/content/MBCONVS_float/Head/Head_conv_parameters.h"


int MBCONV1_0_flag = 0;

int main()
{
  // 1. Define dimensions for input image.
  set_allocate_copy_array_Device(&DInput_Mat, Input_for_stem_conv,
                                 INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, 
                                 INPUT_IMAGE_DEPTH,
                                 "Input Image is allocated in device memory");  

  // 2. Get layers' filters ready
  set_allocate_copy_array_Device(&F_STEM, Stem_conv2d_weights,
                                 STEM_FILTER_HEIGHT, STEM_FILTER_WIDTH, 
                                 STEM_FILTER_DEPTH * STEM_FILTER_DENSITY,
                                 "Stem Filter  is allocated in device memory");
  
  DEFINE_FILTERS_FOR_MBCONV(&D_MBConv_1_0_EXPD_WEIGHTS, NULL, 
                            MBCONV_1_0_EXPD_F_HEIGHT,   MBCONV_1_0_EXPD_F_WIDTH, 
                            MBCONV_1_0_EXPD_F_DEPTH * MBCONV_1_0_EXPD_F_DENSITY,
                            &D_MBConv_1_0_DW_WEIGHTS, MBConv1_0_depthwise_conv_conv2d_weights, 
                            MBCONV_1_0_DW_F_HEIGHT, MBCONV_1_0_DW_F_WIDTH, 
                            MBCONV_1_0_DW_F_DEPTH * MBCONV_1_0_DW_F_DENSITY,
                            &D_MBConv_1_0_SQZ_1_WEIGHTS, MBConv1_0_squeeze_excitation1_conv2d_weights,
                            MBCONV_1_0_SQZ_1_F_HEIGHT, MBCONV_1_0_SQZ_1_F_WIDTH, 
                            MBCONV_1_0_SQZ_1_F_DEPTH * MBCONV_1_0_SQZ_1_F_DENSITY,
                            &D_MBConv_1_0_SQZ_2_WEIGHTS, MBConv1_0_squeeze_excitation2_conv2d_weights, 
                            MBCONV_1_0_SQZ_2_F_HEIGHT, MBCONV_1_0_SQZ_2_F_WIDTH, 
                            MBCONV_1_0_SQZ_2_F_DEPTH * MBCONV_1_0_SQZ_2_F_DENSITY,
                            &D_MBConv_1_0_PRJ_WEIGHTS, MBConv1_0_project_conv_conv2d_weights, 
                            MBCONV_1_0_PRJ_F_HEIGHT, MBCONV_1_0_PRJ_F_WIDTH, 
                            MBCONV_1_0_PRJ_F_DEPTH * MBCONV_1_0_PRJ_F_DENSITY); 

  DEFINE_FILTERS_FOR_MBCONV(&D_MBConv_6_1_EXPD_WEIGHTS, MBConv6_1_expansion_conv_conv2d_weights, 
                            MBCONV_6_1_EXPD_F_HEIGHT,   MBCONV_6_1_EXPD_F_WIDTH, 
                            MBCONV_6_1_EXPD_F_DEPTH * MBCONV_6_1_EXPD_F_DENSITY,
                            &D_MBConv_6_1_DW_WEIGHTS, MBConv6_1_depthwise_conv_conv2d_weights, 
                            MBCONV_6_1_DW_F_HEIGHT, MBCONV_6_1_DW_F_WIDTH, 
                            MBCONV_6_1_DW_F_DEPTH * MBCONV_6_1_DW_F_DENSITY,
                            &D_MBConv_6_1_SQZ_1_WEIGHTS, MBConv6_1_squeeze_excitation1_conv2d_weights,
                            MBCONV_6_1_SQZ_1_F_HEIGHT, MBCONV_6_1_SQZ_1_F_WIDTH, 
                            MBCONV_6_1_SQZ_1_F_DEPTH * MBCONV_6_1_SQZ_1_F_DENSITY,
                            &D_MBConv_6_1_SQZ_2_WEIGHTS, MBConv6_1_squeeze_excitation2_conv2d_weights, 
                            MBCONV_6_1_SQZ_2_F_HEIGHT, MBCONV_6_1_SQZ_2_F_WIDTH, 
                            MBCONV_6_1_SQZ_2_F_DEPTH * MBCONV_6_1_SQZ_2_F_DENSITY,
                            &D_MBConv_6_1_PRJ_WEIGHTS, MBConv6_1_project_conv_conv2d_weights, 
                            MBCONV_6_1_PRJ_F_HEIGHT, MBCONV_6_1_PRJ_F_WIDTH, 
                            MBCONV_6_1_PRJ_F_DEPTH * MBCONV_6_1_PRJ_F_DENSITY); 

  DEFINE_FILTERS_FOR_MBCONV(&D_MBConv_6_2_EXPD_WEIGHTS, MBConv6_2_expansion_conv_conv2d_weights, 
                            MBCONV_6_2_EXPD_F_HEIGHT,   MBCONV_6_2_EXPD_F_WIDTH, 
                            MBCONV_6_2_EXPD_F_DEPTH * MBCONV_6_2_EXPD_F_DENSITY,
                            &D_MBConv_6_2_DW_WEIGHTS, MBConv6_2_depthwise_conv_conv2d_weights, 
                            MBCONV_6_2_DW_F_HEIGHT, MBCONV_6_2_DW_F_WIDTH, 
                            MBCONV_6_2_DW_F_DEPTH * MBCONV_6_2_DW_F_DENSITY,
                            &D_MBConv_6_2_SQZ_1_WEIGHTS, MBConv6_2_squeeze_excitation1_conv2d_weights,
                            MBCONV_6_2_SQZ_1_F_HEIGHT, MBCONV_6_2_SQZ_1_F_WIDTH, 
                            MBCONV_6_2_SQZ_1_F_DEPTH * MBCONV_6_2_SQZ_1_F_DENSITY,
                            &D_MBConv_6_2_SQZ_2_WEIGHTS, MBConv6_2_squeeze_excitation2_conv2d_weights, 
                            MBCONV_6_2_SQZ_2_F_HEIGHT, MBCONV_6_2_SQZ_2_F_WIDTH, 
                            MBCONV_6_2_SQZ_2_F_DEPTH * MBCONV_6_2_SQZ_2_F_DENSITY,
                            &D_MBConv_6_2_PRJ_WEIGHTS, MBConv6_2_project_conv_conv2d_weights, 
                            MBCONV_6_2_PRJ_F_HEIGHT, MBCONV_6_2_PRJ_F_WIDTH, 
                            MBCONV_6_2_PRJ_F_DEPTH * MBCONV_6_2_PRJ_F_DENSITY);

  DEFINE_FILTERS_FOR_MBCONV(&D_MBConv_6_3_EXPD_WEIGHTS, MBConv6_3_expansion_conv_conv2d_weights, 
                            MBCONV_6_3_EXPD_F_HEIGHT,   MBCONV_6_3_EXPD_F_WIDTH, 
                            MBCONV_6_3_EXPD_F_DEPTH * MBCONV_6_3_EXPD_F_DENSITY,
                            &D_MBConv_6_3_DW_WEIGHTS, MBConv6_3_depthwise_conv_conv2d_weights, 
                            MBCONV_6_3_DW_F_HEIGHT, MBCONV_6_3_DW_F_WIDTH, 
                            MBCONV_6_3_DW_F_DEPTH * MBCONV_6_3_DW_F_DENSITY,
                            &D_MBConv_6_3_SQZ_1_WEIGHTS, MBConv6_3_squeeze_excitation1_conv2d_weights,
                            MBCONV_6_3_SQZ_1_F_HEIGHT, MBCONV_6_3_SQZ_1_F_WIDTH, 
                            MBCONV_6_3_SQZ_1_F_DEPTH * MBCONV_6_3_SQZ_1_F_DENSITY,
                            &D_MBConv_6_3_SQZ_2_WEIGHTS, MBConv6_3_squeeze_excitation2_conv2d_weights, 
                            MBCONV_6_3_SQZ_2_F_HEIGHT, MBCONV_6_3_SQZ_2_F_WIDTH, 
                            MBCONV_6_3_SQZ_2_F_DEPTH * MBCONV_6_3_SQZ_2_F_DENSITY,
                            &D_MBConv_6_3_PRJ_WEIGHTS, MBConv6_3_project_conv_conv2d_weights, 
                            MBCONV_6_3_PRJ_F_HEIGHT, MBCONV_6_3_PRJ_F_WIDTH, 
                            MBCONV_6_3_PRJ_F_DEPTH * MBCONV_6_3_PRJ_F_DENSITY);


  DEFINE_FILTERS_FOR_MBCONV(&D_MBConv_6_4_EXPD_WEIGHTS, MBConv6_4_expansion_conv_conv2d_weights, 
                            MBCONV_6_4_EXPD_F_HEIGHT,   MBCONV_6_4_EXPD_F_WIDTH, 
                            MBCONV_6_4_EXPD_F_DEPTH * MBCONV_6_4_EXPD_F_DENSITY,
                            &D_MBConv_6_4_DW_WEIGHTS, MBConv6_4_depthwise_conv_conv2d_weights, 
                            MBCONV_6_4_DW_F_HEIGHT, MBCONV_6_4_DW_F_WIDTH, 
                            MBCONV_6_4_DW_F_DEPTH * MBCONV_6_4_DW_F_DENSITY,
                            &D_MBConv_6_4_SQZ_1_WEIGHTS, MBConv6_4_squeeze_excitation1_conv2d_weights,
                            MBCONV_6_4_SQZ_1_F_HEIGHT, MBCONV_6_4_SQZ_1_F_WIDTH, 
                            MBCONV_6_4_SQZ_1_F_DEPTH * MBCONV_6_4_SQZ_1_F_DENSITY,
                            &D_MBConv_6_4_SQZ_2_WEIGHTS, MBConv6_4_squeeze_excitation2_conv2d_weights, 
                            MBCONV_6_4_SQZ_2_F_HEIGHT, MBCONV_6_4_SQZ_2_F_WIDTH, 
                            MBCONV_6_4_SQZ_2_F_DEPTH * MBCONV_6_4_SQZ_2_F_DENSITY,
                            &D_MBConv_6_4_PRJ_WEIGHTS, MBConv6_4_project_conv_conv2d_weights, 
                            MBCONV_6_4_PRJ_F_HEIGHT, MBCONV_6_4_PRJ_F_WIDTH, 
                            MBCONV_6_4_PRJ_F_DEPTH * MBCONV_6_4_PRJ_F_DENSITY);

  DEFINE_FILTERS_FOR_MBCONV(&D_MBConv_6_5_EXPD_WEIGHTS, MBConv6_5_expansion_conv_conv2d_weights, 
                            MBCONV_6_5_EXPD_F_HEIGHT,   MBCONV_6_5_EXPD_F_WIDTH, 
                            MBCONV_6_5_EXPD_F_DEPTH * MBCONV_6_5_EXPD_F_DENSITY,
                            &D_MBConv_6_5_DW_WEIGHTS, MBConv6_5_depthwise_conv_conv2d_weights, 
                            MBCONV_6_5_DW_F_HEIGHT, MBCONV_6_5_DW_F_WIDTH, 
                            MBCONV_6_5_DW_F_DEPTH * MBCONV_6_5_DW_F_DENSITY,
                            &D_MBConv_6_5_SQZ_1_WEIGHTS, MBConv6_5_squeeze_excitation1_conv2d_weights,
                            MBCONV_6_5_SQZ_1_F_HEIGHT, MBCONV_6_5_SQZ_1_F_WIDTH, 
                            MBCONV_6_5_SQZ_1_F_DEPTH * MBCONV_6_5_SQZ_1_F_DENSITY,
                            &D_MBConv_6_5_SQZ_2_WEIGHTS, MBConv6_5_squeeze_excitation2_conv2d_weights, 
                            MBCONV_6_5_SQZ_2_F_HEIGHT, MBCONV_6_5_SQZ_2_F_WIDTH, 
                            MBCONV_6_5_SQZ_2_F_DEPTH * MBCONV_6_5_SQZ_2_F_DENSITY,
                            &D_MBConv_6_5_PRJ_WEIGHTS, MBConv6_5_project_conv_conv2d_weights, 
                            MBCONV_6_5_PRJ_F_HEIGHT, MBCONV_6_5_PRJ_F_WIDTH, 
                            MBCONV_6_5_PRJ_F_DEPTH * MBCONV_6_5_PRJ_F_DENSITY);
     
  DEFINE_FILTERS_FOR_MBCONV(&D_MBConv_6_6_EXPD_WEIGHTS, MBConv6_6_expansion_conv_conv2d_weights, 
                            MBCONV_6_6_EXPD_F_HEIGHT,   MBCONV_6_6_EXPD_F_WIDTH, 
                            MBCONV_6_6_EXPD_F_DEPTH * MBCONV_6_6_EXPD_F_DENSITY,
                            &D_MBConv_6_6_DW_WEIGHTS, MBConv6_6_depthwise_conv_conv2d_weights, 
                            MBCONV_6_6_DW_F_HEIGHT, MBCONV_6_6_DW_F_WIDTH, 
                            MBCONV_6_6_DW_F_DEPTH * MBCONV_6_6_DW_F_DENSITY,
                            &D_MBConv_6_6_SQZ_1_WEIGHTS, MBConv6_6_squeeze_excitation1_conv2d_weights,
                            MBCONV_6_6_SQZ_1_F_HEIGHT, MBCONV_6_6_SQZ_1_F_WIDTH, 
                            MBCONV_6_6_SQZ_1_F_DEPTH * MBCONV_6_6_SQZ_1_F_DENSITY,
                            &D_MBConv_6_6_SQZ_2_WEIGHTS, MBConv6_6_squeeze_excitation2_conv2d_weights, 
                            MBCONV_6_6_SQZ_2_F_HEIGHT, MBCONV_6_6_SQZ_2_F_WIDTH, 
                            MBCONV_6_6_SQZ_2_F_DEPTH * MBCONV_6_6_SQZ_2_F_DENSITY,
                            &D_MBConv_6_6_PRJ_WEIGHTS, MBConv6_6_project_conv_conv2d_weights, 
                            MBCONV_6_6_PRJ_F_HEIGHT, MBCONV_6_6_PRJ_F_WIDTH, 
                            MBCONV_6_6_PRJ_F_DEPTH * MBCONV_6_6_PRJ_F_DENSITY);
     
  DEFINE_FILTERS_FOR_MBCONV(&D_MBConv_6_7_EXPD_WEIGHTS, MBConv6_7_expansion_conv_conv2d_weights, 
                            MBCONV_6_7_EXPD_F_HEIGHT,   MBCONV_6_7_EXPD_F_WIDTH, 
                            MBCONV_6_7_EXPD_F_DEPTH * MBCONV_6_7_EXPD_F_DENSITY,
                            &D_MBConv_6_7_DW_WEIGHTS, MBConv6_7_depthwise_conv_conv2d_weights, 
                            MBCONV_6_7_DW_F_HEIGHT, MBCONV_6_7_DW_F_WIDTH, 
                            MBCONV_6_7_DW_F_DEPTH * MBCONV_6_7_DW_F_DENSITY,
                            &D_MBConv_6_7_SQZ_1_WEIGHTS, MBConv6_7_squeeze_excitation1_conv2d_weights,
                            MBCONV_6_7_SQZ_1_F_HEIGHT, MBCONV_6_7_SQZ_1_F_WIDTH, 
                            MBCONV_6_7_SQZ_1_F_DEPTH * MBCONV_6_7_SQZ_1_F_DENSITY,
                            &D_MBConv_6_7_SQZ_2_WEIGHTS, MBConv6_7_squeeze_excitation2_conv2d_weights, 
                            MBCONV_6_7_SQZ_2_F_HEIGHT, MBCONV_6_7_SQZ_2_F_WIDTH, 
                            MBCONV_6_7_SQZ_2_F_DEPTH * MBCONV_6_7_SQZ_2_F_DENSITY,
                            &D_MBConv_6_7_PRJ_WEIGHTS, MBConv6_7_project_conv_conv2d_weights, 
                            MBCONV_6_7_PRJ_F_HEIGHT, MBCONV_6_7_PRJ_F_WIDTH, 
                            MBCONV_6_7_PRJ_F_DEPTH * MBCONV_6_7_PRJ_F_DENSITY);


  DEFINE_FILTERS_FOR_MBCONV(&D_MBConv_6_8_EXPD_WEIGHTS, MBConv6_8_expansion_conv_conv2d_weights, 
                            MBCONV_6_8_EXPD_F_HEIGHT,   MBCONV_6_8_EXPD_F_WIDTH, 
                            MBCONV_6_8_EXPD_F_DEPTH * MBCONV_6_8_EXPD_F_DENSITY,
                            &D_MBConv_6_8_DW_WEIGHTS, MBConv6_8_depthwise_conv_conv2d_weights, 
                            MBCONV_6_8_DW_F_HEIGHT, MBCONV_6_8_DW_F_WIDTH, 
                            MBCONV_6_8_DW_F_DEPTH * MBCONV_6_8_DW_F_DENSITY,
                            &D_MBConv_6_8_SQZ_1_WEIGHTS, MBConv6_8_squeeze_excitation1_conv2d_weights,
                            MBCONV_6_8_SQZ_1_F_HEIGHT, MBCONV_6_8_SQZ_1_F_WIDTH, 
                            MBCONV_6_8_SQZ_1_F_DEPTH * MBCONV_6_8_SQZ_1_F_DENSITY,
                            &D_MBConv_6_8_SQZ_2_WEIGHTS, MBConv6_8_squeeze_excitation2_conv2d_weights, 
                            MBCONV_6_8_SQZ_2_F_HEIGHT, MBCONV_6_8_SQZ_2_F_WIDTH, 
                            MBCONV_6_8_SQZ_2_F_DEPTH * MBCONV_6_8_SQZ_2_F_DENSITY,
                            &D_MBConv_6_8_PRJ_WEIGHTS, MBConv6_8_project_conv_conv2d_weights, 
                            MBCONV_6_8_PRJ_F_HEIGHT, MBCONV_6_8_PRJ_F_WIDTH, 
                            MBCONV_6_8_PRJ_F_DEPTH * MBCONV_6_8_PRJ_F_DENSITY);

  DEFINE_FILTERS_FOR_MBCONV(&D_MBConv_6_9_EXPD_WEIGHTS, MBConv6_9_expansion_conv_conv2d_weights, 
                            MBCONV_6_9_EXPD_F_HEIGHT,   MBCONV_6_9_EXPD_F_WIDTH, 
                            MBCONV_6_9_EXPD_F_DEPTH * MBCONV_6_9_EXPD_F_DENSITY,
                            &D_MBConv_6_9_DW_WEIGHTS, MBConv6_9_depthwise_conv_conv2d_weights, 
                            MBCONV_6_9_DW_F_HEIGHT, MBCONV_6_9_DW_F_WIDTH, 
                            MBCONV_6_9_DW_F_DEPTH * MBCONV_6_9_DW_F_DENSITY,
                            &D_MBConv_6_9_SQZ_1_WEIGHTS, MBConv6_9_squeeze_excitation1_conv2d_weights,
                            MBCONV_6_9_SQZ_1_F_HEIGHT, MBCONV_6_9_SQZ_1_F_WIDTH, 
                            MBCONV_6_9_SQZ_1_F_DEPTH * MBCONV_6_9_SQZ_1_F_DENSITY,
                            &D_MBConv_6_9_SQZ_2_WEIGHTS, MBConv6_9_squeeze_excitation2_conv2d_weights, 
                            MBCONV_6_9_SQZ_2_F_HEIGHT, MBCONV_6_9_SQZ_2_F_WIDTH, 
                            MBCONV_6_9_SQZ_2_F_DEPTH * MBCONV_6_9_SQZ_2_F_DENSITY,
                            &D_MBConv_6_9_PRJ_WEIGHTS, MBConv6_9_project_conv_conv2d_weights, 
                            MBCONV_6_9_PRJ_F_HEIGHT, MBCONV_6_9_PRJ_F_WIDTH, 
                            MBCONV_6_9_PRJ_F_DEPTH * MBCONV_6_9_PRJ_F_DENSITY);


  DEFINE_FILTERS_FOR_MBCONV(&D_MBConv_6_10_EXPD_WEIGHTS, MBConv6_10_expansion_conv_conv2d_weights, 
                            MBCONV_6_10_EXPD_F_HEIGHT,   MBCONV_6_10_EXPD_F_WIDTH, 
                            MBCONV_6_10_EXPD_F_DEPTH * MBCONV_6_10_EXPD_F_DENSITY,
                            &D_MBConv_6_10_DW_WEIGHTS, MBConv6_10_depthwise_conv_conv2d_weights, 
                            MBCONV_6_10_DW_F_HEIGHT, MBCONV_6_10_DW_F_WIDTH, 
                            MBCONV_6_10_DW_F_DEPTH * MBCONV_6_10_DW_F_DENSITY,
                            &D_MBConv_6_10_SQZ_1_WEIGHTS, MBConv6_10_squeeze_excitation1_conv2d_weights,
                            MBCONV_6_10_SQZ_1_F_HEIGHT, MBCONV_6_10_SQZ_1_F_WIDTH, 
                            MBCONV_6_10_SQZ_1_F_DEPTH * MBCONV_6_10_SQZ_1_F_DENSITY,
                            &D_MBConv_6_10_SQZ_2_WEIGHTS, MBConv6_10_squeeze_excitation2_conv2d_weights, 
                            MBCONV_6_10_SQZ_2_F_HEIGHT, MBCONV_6_10_SQZ_2_F_WIDTH, 
                            MBCONV_6_10_SQZ_2_F_DEPTH * MBCONV_6_10_SQZ_2_F_DENSITY,
                            &D_MBConv_6_10_PRJ_WEIGHTS, MBConv6_10_project_conv_conv2d_weights, 
                            MBCONV_6_10_PRJ_F_HEIGHT, MBCONV_6_10_PRJ_F_WIDTH, 
                            MBCONV_6_10_PRJ_F_DEPTH * MBCONV_6_10_PRJ_F_DENSITY);


  DEFINE_FILTERS_FOR_MBCONV(&D_MBConv_6_11_EXPD_WEIGHTS, MBConv6_11_expansion_conv_conv2d_weights, 
                            MBCONV_6_11_EXPD_F_HEIGHT,   MBCONV_6_11_EXPD_F_WIDTH, 
                            MBCONV_6_11_EXPD_F_DEPTH * MBCONV_6_11_EXPD_F_DENSITY,
                            &D_MBConv_6_11_DW_WEIGHTS, MBConv6_11_depthwise_conv_conv2d_weights, 
                            MBCONV_6_11_DW_F_HEIGHT, MBCONV_6_11_DW_F_WIDTH, 
                            MBCONV_6_11_DW_F_DEPTH * MBCONV_6_11_DW_F_DENSITY,
                            &D_MBConv_6_11_SQZ_1_WEIGHTS, MBConv6_11_squeeze_excitation1_conv2d_weights,
                            MBCONV_6_11_SQZ_1_F_HEIGHT, MBCONV_6_11_SQZ_1_F_WIDTH, 
                            MBCONV_6_11_SQZ_1_F_DEPTH * MBCONV_6_11_SQZ_1_F_DENSITY,
                            &D_MBConv_6_11_SQZ_2_WEIGHTS, MBConv6_11_squeeze_excitation2_conv2d_weights, 
                            MBCONV_6_11_SQZ_2_F_HEIGHT, MBCONV_6_11_SQZ_2_F_WIDTH, 
                            MBCONV_6_11_SQZ_2_F_DEPTH * MBCONV_6_11_SQZ_2_F_DENSITY,
                            &D_MBConv_6_11_PRJ_WEIGHTS, MBConv6_11_project_conv_conv2d_weights, 
                            MBCONV_6_11_PRJ_F_HEIGHT, MBCONV_6_11_PRJ_F_WIDTH, 
                            MBCONV_6_11_PRJ_F_DEPTH * MBCONV_6_11_PRJ_F_DENSITY);

 
  DEFINE_FILTERS_FOR_MBCONV(&D_MBConv_6_12_EXPD_WEIGHTS, MBConv6_12_expansion_conv_conv2d_weights, 
                            MBCONV_6_12_EXPD_F_HEIGHT,   MBCONV_6_12_EXPD_F_WIDTH, 
                            MBCONV_6_12_EXPD_F_DEPTH * MBCONV_6_12_EXPD_F_DENSITY,
                            &D_MBConv_6_12_DW_WEIGHTS, MBConv6_12_depthwise_conv_conv2d_weights, 
                            MBCONV_6_12_DW_F_HEIGHT, MBCONV_6_12_DW_F_WIDTH, 
                            MBCONV_6_12_DW_F_DEPTH * MBCONV_6_12_DW_F_DENSITY,
                            &D_MBConv_6_12_SQZ_1_WEIGHTS, MBConv6_12_squeeze_excitation1_conv2d_weights,
                            MBCONV_6_12_SQZ_1_F_HEIGHT, MBCONV_6_12_SQZ_1_F_WIDTH, 
                            MBCONV_6_12_SQZ_1_F_DEPTH * MBCONV_6_12_SQZ_1_F_DENSITY,
                            &D_MBConv_6_12_SQZ_2_WEIGHTS, MBConv6_12_squeeze_excitation2_conv2d_weights, 
                            MBCONV_6_12_SQZ_2_F_HEIGHT, MBCONV_6_12_SQZ_2_F_WIDTH, 
                            MBCONV_6_12_SQZ_2_F_DEPTH * MBCONV_6_12_SQZ_2_F_DENSITY,
                            &D_MBConv_6_12_PRJ_WEIGHTS, MBConv6_12_project_conv_conv2d_weights, 
                            MBCONV_6_12_PRJ_F_HEIGHT, MBCONV_6_12_PRJ_F_WIDTH, 
                            MBCONV_6_12_PRJ_F_DEPTH * MBCONV_6_12_PRJ_F_DENSITY);


  DEFINE_FILTERS_FOR_MBCONV(&D_MBConv_6_13_EXPD_WEIGHTS, MBConv6_13_expansion_conv_conv2d_weights, 
                            MBCONV_6_13_EXPD_F_HEIGHT,   MBCONV_6_13_EXPD_F_WIDTH, 
                            MBCONV_6_13_EXPD_F_DEPTH * MBCONV_6_13_EXPD_F_DENSITY,
                            &D_MBConv_6_13_DW_WEIGHTS, MBConv6_13_depthwise_conv_conv2d_weights, 
                            MBCONV_6_13_DW_F_HEIGHT, MBCONV_6_13_DW_F_WIDTH, 
                            MBCONV_6_13_DW_F_DEPTH * MBCONV_6_13_DW_F_DENSITY,
                            &D_MBConv_6_13_SQZ_1_WEIGHTS, MBConv6_13_squeeze_excitation1_conv2d_weights,
                            MBCONV_6_13_SQZ_1_F_HEIGHT, MBCONV_6_13_SQZ_1_F_WIDTH, 
                            MBCONV_6_13_SQZ_1_F_DEPTH * MBCONV_6_13_SQZ_1_F_DENSITY,
                            &D_MBConv_6_13_SQZ_2_WEIGHTS, MBConv6_13_squeeze_excitation2_conv2d_weights, 
                            MBCONV_6_13_SQZ_2_F_HEIGHT, MBCONV_6_13_SQZ_2_F_WIDTH, 
                            MBCONV_6_13_SQZ_2_F_DEPTH * MBCONV_6_13_SQZ_2_F_DENSITY,
                            &D_MBConv_6_13_PRJ_WEIGHTS, MBConv6_13_project_conv_conv2d_weights, 
                            MBCONV_6_13_PRJ_F_HEIGHT, MBCONV_6_13_PRJ_F_WIDTH, 
                            MBCONV_6_13_PRJ_F_DEPTH * MBCONV_6_13_PRJ_F_DENSITY);


  DEFINE_FILTERS_FOR_MBCONV(&D_MBConv_6_14_EXPD_WEIGHTS, MBConv6_14_expansion_conv_conv2d_weights, 
                            MBCONV_6_14_EXPD_F_HEIGHT,   MBCONV_6_14_EXPD_F_WIDTH, 
                            MBCONV_6_14_EXPD_F_DEPTH * MBCONV_6_14_EXPD_F_DENSITY,
                            &D_MBConv_6_14_DW_WEIGHTS, MBConv6_14_depthwise_conv_conv2d_weights, 
                            MBCONV_6_14_DW_F_HEIGHT, MBCONV_6_14_DW_F_WIDTH, 
                            MBCONV_6_14_DW_F_DEPTH * MBCONV_6_14_DW_F_DENSITY,
                            &D_MBConv_6_14_SQZ_1_WEIGHTS, MBConv6_14_squeeze_excitation1_conv2d_weights,
                            MBCONV_6_14_SQZ_1_F_HEIGHT, MBCONV_6_14_SQZ_1_F_WIDTH, 
                            MBCONV_6_14_SQZ_1_F_DEPTH * MBCONV_6_14_SQZ_1_F_DENSITY,
                            &D_MBConv_6_14_SQZ_2_WEIGHTS, MBConv6_14_squeeze_excitation2_conv2d_weights, 
                            MBCONV_6_14_SQZ_2_F_HEIGHT, MBCONV_6_14_SQZ_2_F_WIDTH, 
                            MBCONV_6_14_SQZ_2_F_DEPTH * MBCONV_6_14_SQZ_2_F_DENSITY,
                            &D_MBConv_6_14_PRJ_WEIGHTS, MBConv6_14_project_conv_conv2d_weights, 
                            MBCONV_6_14_PRJ_F_HEIGHT, MBCONV_6_14_PRJ_F_WIDTH, 
                            MBCONV_6_14_PRJ_F_DEPTH * MBCONV_6_14_PRJ_F_DENSITY);
     
  DEFINE_FILTERS_FOR_MBCONV(&D_MBConv_6_15_EXPD_WEIGHTS, MBConv6_15_expansion_conv_conv2d_weights, 
                            MBCONV_6_15_EXPD_F_HEIGHT,   MBCONV_6_15_EXPD_F_WIDTH, 
                            MBCONV_6_15_EXPD_F_DEPTH * MBCONV_6_15_EXPD_F_DENSITY,
                            &D_MBConv_6_15_DW_WEIGHTS, MBConv6_15_depthwise_conv_conv2d_weights, 
                            MBCONV_6_15_DW_F_HEIGHT, MBCONV_6_15_DW_F_WIDTH, 
                            MBCONV_6_15_DW_F_DEPTH * MBCONV_6_15_DW_F_DENSITY,
                            &D_MBConv_6_15_SQZ_1_WEIGHTS, MBConv6_15_squeeze_excitation1_conv2d_weights,
                            MBCONV_6_15_SQZ_1_F_HEIGHT, MBCONV_6_15_SQZ_1_F_WIDTH, 
                            MBCONV_6_15_SQZ_1_F_DEPTH * MBCONV_6_15_SQZ_1_F_DENSITY,
                            &D_MBConv_6_15_SQZ_2_WEIGHTS, MBConv6_15_squeeze_excitation2_conv2d_weights, 
                            MBCONV_6_15_SQZ_2_F_HEIGHT, MBCONV_6_15_SQZ_2_F_WIDTH, 
                            MBCONV_6_15_SQZ_2_F_DEPTH * MBCONV_6_15_SQZ_2_F_DENSITY,
                            &D_MBConv_6_15_PRJ_WEIGHTS, MBConv6_15_project_conv_conv2d_weights, 
                            MBCONV_6_15_PRJ_F_HEIGHT, MBCONV_6_15_PRJ_F_WIDTH, 
                            MBCONV_6_15_PRJ_F_DEPTH * MBCONV_6_15_PRJ_F_DENSITY);


  set_allocate_copy_array_Device(&HEAD_CONV_WEIGHTS, Head_conv2d_weights,
                                  HEAD_CONV_F_HEIGHT, HEAD_CONV_F_WIDTH, HEAD_CONV_F_DEPTH * HEAD_CONV_F_DENSITY,
                                "Head Filter  is allocated in device memory");   
 
  set_allocate_copy_array_Device(&HEAD_FC_WEIGHTS, Head_linear_weights,
                                HEAD_FC_F_HEIGHT, HEAD_FC_F_WIDTH, 1,
                                "Fully Connected weights matrix is allocated in device memory");  
  
  // Define bias matrices for all squeeze layers
  set_allocate_copy_array_Device(&MBConv6_15_SQZ_1_bias, MBConv6_15_squeeze_excitation1_conv2d_bias,
                                  sizeof(MBConv6_15_squeeze_excitation1_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 1 layer #15");  
  set_allocate_copy_array_Device(&MBConv6_14_SQZ_1_bias, MBConv6_14_squeeze_excitation1_conv2d_bias,
                                  sizeof(MBConv6_14_squeeze_excitation1_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 1 layer #14");
  set_allocate_copy_array_Device(&MBConv6_13_SQZ_1_bias, MBConv6_13_squeeze_excitation1_conv2d_bias,
                                  sizeof(MBConv6_13_squeeze_excitation1_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 1 layer #13");
  set_allocate_copy_array_Device(&MBConv6_12_SQZ_1_bias, MBConv6_12_squeeze_excitation1_conv2d_bias,
                                  sizeof(MBConv6_12_squeeze_excitation1_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 1 layer #12");
  set_allocate_copy_array_Device(&MBConv6_11_SQZ_1_bias, MBConv6_11_squeeze_excitation1_conv2d_bias,
                                  sizeof(MBConv6_11_squeeze_excitation1_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 1 layer #11");
  set_allocate_copy_array_Device(&MBConv6_10_SQZ_1_bias, MBConv6_10_squeeze_excitation1_conv2d_bias,
                                  sizeof(MBConv6_10_squeeze_excitation1_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 1 layer #10");  
  set_allocate_copy_array_Device(&MBConv6_9_SQZ_1_bias, MBConv6_9_squeeze_excitation1_conv2d_bias,
                                  sizeof(MBConv6_9_squeeze_excitation1_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 1 layer #9");
  set_allocate_copy_array_Device(&MBConv6_8_SQZ_1_bias, MBConv6_8_squeeze_excitation1_conv2d_bias,
                                  sizeof(MBConv6_8_squeeze_excitation1_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 1 layer #8");
  set_allocate_copy_array_Device(&MBConv6_7_SQZ_1_bias, MBConv6_7_squeeze_excitation1_conv2d_bias,
                                  sizeof(MBConv6_7_squeeze_excitation1_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 1 layer #7");
  set_allocate_copy_array_Device(&MBConv6_6_SQZ_1_bias, MBConv6_6_squeeze_excitation1_conv2d_bias,
                                  sizeof(MBConv6_6_squeeze_excitation1_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 1 layer #6");
  set_allocate_copy_array_Device(&MBConv6_5_SQZ_1_bias, MBConv6_5_squeeze_excitation1_conv2d_bias,
                                  sizeof(MBConv6_5_squeeze_excitation1_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 1 layer #5");  
  set_allocate_copy_array_Device(&MBConv6_4_SQZ_1_bias, MBConv6_4_squeeze_excitation1_conv2d_bias,
                                  sizeof(MBConv6_4_squeeze_excitation1_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 1 layer #4");
  set_allocate_copy_array_Device(&MBConv6_3_SQZ_1_bias, MBConv6_3_squeeze_excitation1_conv2d_bias,
                                  sizeof(MBConv6_3_squeeze_excitation1_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 1 layer #3");
  set_allocate_copy_array_Device(&MBConv6_2_SQZ_1_bias, MBConv6_2_squeeze_excitation1_conv2d_bias,
                                  sizeof(MBConv6_2_squeeze_excitation1_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 1 layer #2");
  set_allocate_copy_array_Device(&MBConv6_1_SQZ_1_bias, MBConv6_1_squeeze_excitation1_conv2d_bias,
                                  sizeof(MBConv6_1_squeeze_excitation1_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 1 layer #1");
  set_allocate_copy_array_Device(&MBConv1_0_SQZ_1_bias, MBConv1_0_squeeze_excitation1_conv2d_bias,
                                  sizeof(MBConv1_0_squeeze_excitation1_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 1 layer #0");   
  set_allocate_copy_array_Device(&MBConv6_15_SQZ_2_bias, MBConv6_15_squeeze_excitation2_conv2d_bias,
                                  sizeof(MBConv6_15_squeeze_excitation2_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 2 layer #15");  
  set_allocate_copy_array_Device(&MBConv6_14_SQZ_2_bias, MBConv6_14_squeeze_excitation2_conv2d_bias,
                                  sizeof(MBConv6_14_squeeze_excitation2_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 2 layer #14");
  set_allocate_copy_array_Device(&MBConv6_13_SQZ_2_bias, MBConv6_13_squeeze_excitation2_conv2d_bias,
                                  sizeof(MBConv6_13_squeeze_excitation2_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 2 layer #13");
  set_allocate_copy_array_Device(&MBConv6_12_SQZ_2_bias, MBConv6_12_squeeze_excitation2_conv2d_bias,
                                  sizeof(MBConv6_12_squeeze_excitation2_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 2 layer #12");
  set_allocate_copy_array_Device(&MBConv6_11_SQZ_2_bias, MBConv6_11_squeeze_excitation2_conv2d_bias,
                                  sizeof(MBConv6_11_squeeze_excitation2_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 2 layer #11");
  set_allocate_copy_array_Device(&MBConv6_10_SQZ_2_bias, MBConv6_10_squeeze_excitation2_conv2d_bias,
                                  sizeof(MBConv6_10_squeeze_excitation2_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 2 layer #10");  
  set_allocate_copy_array_Device(&MBConv6_9_SQZ_2_bias, MBConv6_9_squeeze_excitation2_conv2d_bias,
                                  sizeof(MBConv6_9_squeeze_excitation2_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 2 layer #9");
  set_allocate_copy_array_Device(&MBConv6_8_SQZ_2_bias, MBConv6_8_squeeze_excitation2_conv2d_bias,
                                  sizeof(MBConv6_8_squeeze_excitation2_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 2 layer #8");
  set_allocate_copy_array_Device(&MBConv6_7_SQZ_2_bias, MBConv6_7_squeeze_excitation2_conv2d_bias,
                                  sizeof(MBConv6_7_squeeze_excitation2_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 2 layer #7");
  set_allocate_copy_array_Device(&MBConv6_6_SQZ_2_bias, MBConv6_6_squeeze_excitation2_conv2d_bias,
                                  sizeof(MBConv6_6_squeeze_excitation2_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 2 layer #6");
  set_allocate_copy_array_Device(&MBConv6_5_SQZ_2_bias, MBConv6_5_squeeze_excitation2_conv2d_bias,
                                  sizeof(MBConv6_5_squeeze_excitation2_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 2 layer #5");  
  set_allocate_copy_array_Device(&MBConv6_4_SQZ_2_bias, MBConv6_4_squeeze_excitation2_conv2d_bias,
                                  sizeof(MBConv6_4_squeeze_excitation2_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 2 layer #4");
  set_allocate_copy_array_Device(&MBConv6_3_SQZ_2_bias, MBConv6_3_squeeze_excitation2_conv2d_bias,
                                  sizeof(MBConv6_3_squeeze_excitation2_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 2 layer #3");
  set_allocate_copy_array_Device(&MBConv6_2_SQZ_2_bias, MBConv6_2_squeeze_excitation2_conv2d_bias,
                                  sizeof(MBConv6_2_squeeze_excitation2_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 2 layer #2");
  set_allocate_copy_array_Device(&MBConv6_1_SQZ_2_bias, MBConv6_1_squeeze_excitation2_conv2d_bias,
                                  sizeof(MBConv6_1_squeeze_excitation2_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 2 layer #1");
  set_allocate_copy_array_Device(&MBConv1_0_SQZ_2_bias, MBConv1_0_squeeze_excitation2_conv2d_bias,
                                  sizeof(MBConv1_0_squeeze_excitation2_conv2d_bias)/sizeof(float), 1, 1,
                                  "Bias for squeeze 2 layer #0");    

// 3. Define BN mean,variance, weights and bias
MBCONV1_0_flag = 1;

DEFINE_FILTERS_FOR_MBCONV_BN(	
  &MBConv1_0_EXPD_BN_MEAN,      NULL, 0,
  &MBConv1_0_EXPD_BN_VARIANCE,	NULL, 0,
  &MBConv1_0_EXPD_BN_WEIGHTS,		NULL, 0,
  &MBConv1_0_EXPD_BN_BIAS,			NULL, 0,

  &MBConv1_0_DW_BN_MEAN,        MBConv1_0_depthwise_conv_BN_mean,		  sizeof(MBConv1_0_depthwise_conv_BN_mean) / sizeof(float), 		
  &MBConv1_0_DW_BN_VARIANCE,		MBConv1_0_depthwise_conv_BN_variance,	sizeof(MBConv1_0_depthwise_conv_BN_variance) / sizeof(float),
  &MBConv1_0_DW_BN_WEIGHTS,     MBConv1_0_depthwise_conv_BN_weights,	sizeof(MBConv1_0_depthwise_conv_BN_weights) / sizeof(float),
  &MBConv1_0_DW_BN_BIAS,				MBConv1_0_depthwise_conv_BN_bias,		  sizeof(MBConv1_0_depthwise_conv_BN_bias) / sizeof(float),

  &MBConv1_0_PRJ_BN_MEAN,       MBConv1_0_project_conv_BN_mean,			  sizeof(MBConv1_0_project_conv_BN_mean) / sizeof(float),
  &MBConv1_0_PRJ_BN_VARIANCE,		MBConv1_0_project_conv_BN_variance,		sizeof(MBConv1_0_project_conv_BN_variance) / sizeof(float),
  &MBConv1_0_PRJ_BN_WEIGHTS,    MBConv1_0_project_conv_BN_weights,		sizeof(MBConv1_0_project_conv_BN_weights) / sizeof(float),
  &MBConv1_0_PRJ_BN_BIAS,				MBConv1_0_project_conv_BN_bias, 		  sizeof(MBConv1_0_project_conv_BN_bias) / sizeof(float));

MBCONV1_0_flag = 0;

DEFINE_FILTERS_FOR_MBCONV_BN(	
  &MBConv6_1_EXPD_BN_MEAN,      MBConv6_1_expansion_conv_BN_mean,		  sizeof(MBConv6_1_expansion_conv_BN_mean) / sizeof(float),
  &MBConv6_1_EXPD_BN_VARIANCE,	MBConv6_1_expansion_conv_BN_variance,	sizeof(MBConv6_1_expansion_conv_BN_variance) / sizeof(float),
  &MBConv6_1_EXPD_BN_WEIGHTS,   MBConv6_1_expansion_conv_BN_weights,	sizeof(MBConv6_1_expansion_conv_BN_weights) / sizeof(float),
  &MBConv6_1_EXPD_BN_BIAS,			MBConv6_1_expansion_conv_BN_bias,		  sizeof(MBConv6_1_expansion_conv_BN_bias) / sizeof(float),

  &MBConv6_1_DW_BN_MEAN,        MBConv6_1_depthwise_conv_BN_mean,		  sizeof(MBConv6_1_depthwise_conv_BN_mean) / sizeof(float),
  &MBConv6_1_DW_BN_VARIANCE,		MBConv6_1_depthwise_conv_BN_variance,	sizeof(MBConv6_1_depthwise_conv_BN_variance) / sizeof(float),	
  &MBConv6_1_DW_BN_WEIGHTS,     MBConv6_1_depthwise_conv_BN_weights,	sizeof(MBConv6_1_depthwise_conv_BN_weights) / sizeof(float),
  &MBConv6_1_DW_BN_BIAS,				MBConv6_1_depthwise_conv_BN_bias,		  sizeof(MBConv6_1_depthwise_conv_BN_bias) / sizeof(float),

  &MBConv6_1_PRJ_BN_MEAN,       MBConv6_1_project_conv_BN_mean,			  sizeof(MBConv6_1_project_conv_BN_mean) / sizeof(float),
  &MBConv6_1_PRJ_BN_VARIANCE,		MBConv6_1_project_conv_BN_variance,		sizeof(MBConv6_1_project_conv_BN_variance) / sizeof(float),
  &MBConv6_1_PRJ_BN_WEIGHTS,    MBConv6_1_project_conv_BN_weights,		sizeof(MBConv6_1_project_conv_BN_weights) / sizeof(float),
  &MBConv6_1_PRJ_BN_BIAS,				MBConv6_1_project_conv_BN_bias, 		  sizeof(MBConv6_1_project_conv_BN_bias) / sizeof(float));

DEFINE_FILTERS_FOR_MBCONV_BN(	
  &MBConv6_2_EXPD_BN_MEAN,      MBConv6_2_expansion_conv_BN_mean,		  sizeof(MBConv6_2_expansion_conv_BN_mean) / sizeof(float),
  &MBConv6_2_EXPD_BN_VARIANCE,	MBConv6_2_expansion_conv_BN_variance,	sizeof(MBConv6_2_expansion_conv_BN_variance) / sizeof(float),
  &MBConv6_2_EXPD_BN_WEIGHTS,   MBConv6_2_expansion_conv_BN_weights,	sizeof(MBConv6_2_expansion_conv_BN_weights) / sizeof(float),
  &MBConv6_2_EXPD_BN_BIAS,			MBConv6_2_expansion_conv_BN_bias,		  sizeof(MBConv6_2_expansion_conv_BN_bias) / sizeof(float),
  &MBConv6_2_DW_BN_MEAN,        MBConv6_2_depthwise_conv_BN_mean,		  sizeof(MBConv6_2_depthwise_conv_BN_mean) / sizeof(float),
  &MBConv6_2_DW_BN_VARIANCE,		MBConv6_2_depthwise_conv_BN_variance,	sizeof(MBConv6_2_depthwise_conv_BN_variance) / sizeof(float),	
  &MBConv6_2_DW_BN_WEIGHTS,     MBConv6_2_depthwise_conv_BN_weights,	sizeof(MBConv6_2_depthwise_conv_BN_weights) / sizeof(float),
  &MBConv6_2_DW_BN_BIAS,			  MBConv6_2_depthwise_conv_BN_bias,		  sizeof(MBConv6_2_depthwise_conv_BN_bias) / sizeof(float),
  &MBConv6_2_PRJ_BN_MEAN,       MBConv6_2_project_conv_BN_mean,			  sizeof(MBConv6_2_project_conv_BN_mean) / sizeof(float),
  &MBConv6_2_PRJ_BN_VARIANCE,		MBConv6_2_project_conv_BN_variance,		sizeof(MBConv6_2_project_conv_BN_variance) / sizeof(float),
  &MBConv6_2_PRJ_BN_WEIGHTS,    MBConv6_2_project_conv_BN_weights,		sizeof(MBConv6_2_project_conv_BN_weights) / sizeof(float),
  &MBConv6_2_PRJ_BN_BIAS,				MBConv6_2_project_conv_BN_bias, 		  sizeof(MBConv6_2_project_conv_BN_bias) / sizeof(float));

DEFINE_FILTERS_FOR_MBCONV_BN(	
  &MBConv6_3_EXPD_BN_MEAN,      MBConv6_3_expansion_conv_BN_mean, 		sizeof(MBConv6_3_expansion_conv_BN_mean) / sizeof(float),
  &MBConv6_3_EXPD_BN_VARIANCE,	MBConv6_3_expansion_conv_BN_variance,	sizeof(MBConv6_3_expansion_conv_BN_variance) / sizeof(float),
  &MBConv6_3_EXPD_BN_WEIGHTS,   MBConv6_3_expansion_conv_BN_weights,	sizeof(MBConv6_3_expansion_conv_BN_weights) / sizeof(float),
  &MBConv6_3_EXPD_BN_BIAS,			MBConv6_3_expansion_conv_BN_bias,		  sizeof(MBConv6_3_expansion_conv_BN_bias) / sizeof(float),
  &MBConv6_3_DW_BN_MEAN,        MBConv6_3_depthwise_conv_BN_mean,		  sizeof(MBConv6_3_depthwise_conv_BN_mean) / sizeof(float),
  &MBConv6_3_DW_BN_VARIANCE,		MBConv6_3_depthwise_conv_BN_variance,	sizeof(MBConv6_3_depthwise_conv_BN_variance) / sizeof(float),	
  &MBConv6_3_DW_BN_WEIGHTS,     MBConv6_3_depthwise_conv_BN_weights,	sizeof(MBConv6_3_depthwise_conv_BN_weights) / sizeof(float),
  &MBConv6_3_DW_BN_BIAS,				MBConv6_3_depthwise_conv_BN_bias,		  sizeof(MBConv6_3_depthwise_conv_BN_bias) / sizeof(float),
  &MBConv6_3_PRJ_BN_MEAN,       MBConv6_3_project_conv_BN_mean,			  sizeof(MBConv6_3_project_conv_BN_mean) / sizeof(float),
  &MBConv6_3_PRJ_BN_VARIANCE,		MBConv6_3_project_conv_BN_variance,		sizeof(MBConv6_3_project_conv_BN_variance) / sizeof(float),
  &MBConv6_3_PRJ_BN_WEIGHTS,    MBConv6_3_project_conv_BN_weights,		sizeof(MBConv6_3_project_conv_BN_weights) / sizeof(float),
  &MBConv6_3_PRJ_BN_BIAS,				MBConv6_3_project_conv_BN_bias, 		  sizeof(MBConv6_3_project_conv_BN_bias) / sizeof(float));

DEFINE_FILTERS_FOR_MBCONV_BN(	
  &MBConv6_4_EXPD_BN_MEAN,      MBConv6_4_expansion_conv_BN_mean, 		sizeof(MBConv6_4_expansion_conv_BN_mean) / sizeof(float),
  &MBConv6_4_EXPD_BN_VARIANCE,	MBConv6_4_expansion_conv_BN_variance,	sizeof(MBConv6_4_expansion_conv_BN_variance) / sizeof(float),
  &MBConv6_4_EXPD_BN_WEIGHTS,   MBConv6_4_expansion_conv_BN_weights,	sizeof(MBConv6_4_expansion_conv_BN_weights) / sizeof(float),
  &MBConv6_4_EXPD_BN_BIAS,			MBConv6_4_expansion_conv_BN_bias,		  sizeof(MBConv6_4_expansion_conv_BN_bias) / sizeof(float),
  &MBConv6_4_DW_BN_MEAN,        MBConv6_4_depthwise_conv_BN_mean,		  sizeof(MBConv6_4_depthwise_conv_BN_mean) / sizeof(float),
  &MBConv6_4_DW_BN_VARIANCE,		MBConv6_4_depthwise_conv_BN_variance,	sizeof(MBConv6_4_depthwise_conv_BN_variance) / sizeof(float),	
  &MBConv6_4_DW_BN_WEIGHTS,     MBConv6_4_depthwise_conv_BN_weights,	sizeof(MBConv6_4_depthwise_conv_BN_weights) / sizeof(float),
  &MBConv6_4_DW_BN_BIAS,				MBConv6_4_depthwise_conv_BN_bias,		  sizeof(MBConv6_4_depthwise_conv_BN_bias) / sizeof(float),
  &MBConv6_4_PRJ_BN_MEAN,       MBConv6_4_project_conv_BN_mean,			  sizeof(MBConv6_4_project_conv_BN_mean) / sizeof(float),
  &MBConv6_4_PRJ_BN_VARIANCE,		MBConv6_4_project_conv_BN_variance,		sizeof(MBConv6_4_project_conv_BN_variance) / sizeof(float),
  &MBConv6_4_PRJ_BN_WEIGHTS,    MBConv6_4_project_conv_BN_weights,		sizeof(MBConv6_4_project_conv_BN_weights) / sizeof(float),
  &MBConv6_4_PRJ_BN_BIAS,				MBConv6_4_project_conv_BN_bias, 		  sizeof(MBConv6_4_project_conv_BN_bias) / sizeof(float));

DEFINE_FILTERS_FOR_MBCONV_BN(	
  &MBConv6_5_EXPD_BN_MEAN,      MBConv6_5_expansion_conv_BN_mean,		  sizeof(MBConv6_5_expansion_conv_BN_mean) / sizeof(float),
  &MBConv6_5_EXPD_BN_VARIANCE,	MBConv6_5_expansion_conv_BN_variance,	sizeof(MBConv6_5_expansion_conv_BN_variance) / sizeof(float),
  &MBConv6_5_EXPD_BN_WEIGHTS,   MBConv6_5_expansion_conv_BN_weights,	sizeof(MBConv6_5_expansion_conv_BN_weights) / sizeof(float),
  &MBConv6_5_EXPD_BN_BIAS,			MBConv6_5_expansion_conv_BN_bias,		  sizeof(MBConv6_5_expansion_conv_BN_bias) / sizeof(float),
  &MBConv6_5_DW_BN_MEAN,        MBConv6_5_depthwise_conv_BN_mean,		  sizeof(MBConv6_5_depthwise_conv_BN_mean) / sizeof(float),
  &MBConv6_5_DW_BN_VARIANCE,		MBConv6_5_depthwise_conv_BN_variance,	sizeof(MBConv6_5_depthwise_conv_BN_variance) / sizeof(float),	
  &MBConv6_5_DW_BN_WEIGHTS,     MBConv6_5_depthwise_conv_BN_weights,	sizeof(MBConv6_5_depthwise_conv_BN_weights) / sizeof(float),
  &MBConv6_5_DW_BN_BIAS,				MBConv6_5_depthwise_conv_BN_bias,		  sizeof(MBConv6_5_depthwise_conv_BN_bias) / sizeof(float),
  &MBConv6_5_PRJ_BN_MEAN,       MBConv6_5_project_conv_BN_mean,			  sizeof(MBConv6_5_project_conv_BN_mean) / sizeof(float),
  &MBConv6_5_PRJ_BN_VARIANCE,		MBConv6_5_project_conv_BN_variance,		sizeof(MBConv6_5_project_conv_BN_variance) / sizeof(float),
  &MBConv6_5_PRJ_BN_WEIGHTS,    MBConv6_5_project_conv_BN_weights,		sizeof(MBConv6_5_project_conv_BN_weights) / sizeof(float),
  &MBConv6_5_PRJ_BN_BIAS,				MBConv6_5_project_conv_BN_bias, 		  sizeof(MBConv6_5_project_conv_BN_bias) / sizeof(float));

DEFINE_FILTERS_FOR_MBCONV_BN(	
  &MBConv6_6_EXPD_BN_MEAN,      MBConv6_6_expansion_conv_BN_mean,		  sizeof(MBConv6_6_expansion_conv_BN_mean) / sizeof(float),
  &MBConv6_6_EXPD_BN_VARIANCE,	MBConv6_6_expansion_conv_BN_variance,	sizeof(MBConv6_6_expansion_conv_BN_variance) / sizeof(float),
  &MBConv6_6_EXPD_BN_WEIGHTS,   MBConv6_6_expansion_conv_BN_weights,	sizeof(MBConv6_6_expansion_conv_BN_weights) / sizeof(float),
  &MBConv6_6_EXPD_BN_BIAS,			MBConv6_6_expansion_conv_BN_bias,		  sizeof(MBConv6_6_expansion_conv_BN_bias) / sizeof(float),
  &MBConv6_6_DW_BN_MEAN,        MBConv6_6_depthwise_conv_BN_mean,		  sizeof(MBConv6_6_depthwise_conv_BN_mean) / sizeof(float),
  &MBConv6_6_DW_BN_VARIANCE,		MBConv6_6_depthwise_conv_BN_variance,	sizeof(MBConv6_6_depthwise_conv_BN_variance) / sizeof(float),	
  &MBConv6_6_DW_BN_WEIGHTS,     MBConv6_6_depthwise_conv_BN_weights,	sizeof(MBConv6_6_depthwise_conv_BN_weights) / sizeof(float),
  &MBConv6_6_DW_BN_BIAS,				MBConv6_6_depthwise_conv_BN_bias,		  sizeof(MBConv6_6_depthwise_conv_BN_bias) / sizeof(float),
  &MBConv6_6_PRJ_BN_MEAN,       MBConv6_6_project_conv_BN_mean,			  sizeof(MBConv6_6_project_conv_BN_mean) / sizeof(float),
  &MBConv6_6_PRJ_BN_VARIANCE,		MBConv6_6_project_conv_BN_variance,		sizeof(MBConv6_6_project_conv_BN_variance) / sizeof(float),
  &MBConv6_6_PRJ_BN_WEIGHTS,    MBConv6_6_project_conv_BN_weights,		sizeof(MBConv6_6_project_conv_BN_weights) / sizeof(float),
  &MBConv6_6_PRJ_BN_BIAS,				MBConv6_6_project_conv_BN_bias, 		  sizeof(MBConv6_6_project_conv_BN_bias) / sizeof(float));

DEFINE_FILTERS_FOR_MBCONV_BN(	
  &MBConv6_7_EXPD_BN_MEAN,      MBConv6_7_expansion_conv_BN_mean,		  sizeof(MBConv6_7_expansion_conv_BN_mean) / sizeof(float),
  &MBConv6_7_EXPD_BN_VARIANCE,	MBConv6_7_expansion_conv_BN_variance,	sizeof(MBConv6_7_expansion_conv_BN_variance) / sizeof(float),
  &MBConv6_7_EXPD_BN_WEIGHTS,   MBConv6_7_expansion_conv_BN_weights,	sizeof(MBConv6_7_expansion_conv_BN_weights) / sizeof(float),
  &MBConv6_7_EXPD_BN_BIAS,			MBConv6_7_expansion_conv_BN_bias,		  sizeof(MBConv6_7_expansion_conv_BN_bias) / sizeof(float),
  &MBConv6_7_DW_BN_MEAN,        MBConv6_7_depthwise_conv_BN_mean,		  sizeof(MBConv6_7_depthwise_conv_BN_mean) / sizeof(float),
  &MBConv6_7_DW_BN_VARIANCE,		MBConv6_7_depthwise_conv_BN_variance,	sizeof(MBConv6_7_depthwise_conv_BN_variance) / sizeof(float),	
  &MBConv6_7_DW_BN_WEIGHTS,     MBConv6_7_depthwise_conv_BN_weights,	sizeof(MBConv6_7_depthwise_conv_BN_weights) / sizeof(float),
  &MBConv6_7_DW_BN_BIAS,				MBConv6_7_depthwise_conv_BN_bias,		  sizeof(MBConv6_7_depthwise_conv_BN_bias) / sizeof(float),
  &MBConv6_7_PRJ_BN_MEAN,       MBConv6_7_project_conv_BN_mean,			  sizeof(MBConv6_7_project_conv_BN_mean) / sizeof(float),
  &MBConv6_7_PRJ_BN_VARIANCE,		MBConv6_7_project_conv_BN_variance,		sizeof(MBConv6_7_project_conv_BN_variance) / sizeof(float),
  &MBConv6_7_PRJ_BN_WEIGHTS,    MBConv6_7_project_conv_BN_weights,		sizeof(MBConv6_7_project_conv_BN_weights) / sizeof(float),
  &MBConv6_7_PRJ_BN_BIAS,				MBConv6_7_project_conv_BN_bias, 		  sizeof(MBConv6_7_project_conv_BN_bias) / sizeof(float));

DEFINE_FILTERS_FOR_MBCONV_BN(	
  &MBConv6_8_EXPD_BN_MEAN,      MBConv6_8_expansion_conv_BN_mean,		  sizeof(MBConv6_8_expansion_conv_BN_mean) / sizeof(float),
  &MBConv6_8_EXPD_BN_VARIANCE,	MBConv6_8_expansion_conv_BN_variance,	sizeof(MBConv6_8_expansion_conv_BN_variance) / sizeof(float),
  &MBConv6_8_EXPD_BN_WEIGHTS,   MBConv6_8_expansion_conv_BN_weights,	sizeof(MBConv6_8_expansion_conv_BN_weights) / sizeof(float),
  &MBConv6_8_EXPD_BN_BIAS,			MBConv6_8_expansion_conv_BN_bias,		  sizeof(MBConv6_8_expansion_conv_BN_bias) / sizeof(float),
  &MBConv6_8_DW_BN_MEAN,        MBConv6_8_depthwise_conv_BN_mean,		  sizeof(MBConv6_8_depthwise_conv_BN_mean) / sizeof(float),
  &MBConv6_8_DW_BN_VARIANCE,		MBConv6_8_depthwise_conv_BN_variance,	sizeof(MBConv6_8_depthwise_conv_BN_variance) / sizeof(float),	
  &MBConv6_8_DW_BN_WEIGHTS,     MBConv6_8_depthwise_conv_BN_weights,	sizeof(MBConv6_8_depthwise_conv_BN_weights) / sizeof(float),
  &MBConv6_8_DW_BN_BIAS,				MBConv6_8_depthwise_conv_BN_bias,		  sizeof(MBConv6_8_depthwise_conv_BN_bias) / sizeof(float),
  &MBConv6_8_PRJ_BN_MEAN,       MBConv6_8_project_conv_BN_mean,			  sizeof(MBConv6_8_project_conv_BN_mean) / sizeof(float),
  &MBConv6_8_PRJ_BN_VARIANCE,		MBConv6_8_project_conv_BN_variance,		sizeof(MBConv6_8_project_conv_BN_variance) / sizeof(float),
  &MBConv6_8_PRJ_BN_WEIGHTS,    MBConv6_8_project_conv_BN_weights,		sizeof(MBConv6_8_project_conv_BN_weights) / sizeof(float),
  &MBConv6_8_PRJ_BN_BIAS,				MBConv6_8_project_conv_BN_bias, 		  sizeof(MBConv6_8_project_conv_BN_bias) / sizeof(float));

DEFINE_FILTERS_FOR_MBCONV_BN(	
  &MBConv6_9_EXPD_BN_MEAN,      MBConv6_9_expansion_conv_BN_mean,		  sizeof(MBConv6_9_expansion_conv_BN_mean) / sizeof(float),
  &MBConv6_9_EXPD_BN_VARIANCE,	MBConv6_9_expansion_conv_BN_variance,	sizeof(MBConv6_9_expansion_conv_BN_variance) / sizeof(float),
  &MBConv6_9_EXPD_BN_WEIGHTS,   MBConv6_9_expansion_conv_BN_weights,	sizeof(MBConv6_9_expansion_conv_BN_weights) / sizeof(float),
  &MBConv6_9_EXPD_BN_BIAS,			MBConv6_9_expansion_conv_BN_bias,		  sizeof(MBConv6_9_expansion_conv_BN_bias) / sizeof(float),
  &MBConv6_9_DW_BN_MEAN,        MBConv6_9_depthwise_conv_BN_mean,		  sizeof(MBConv6_9_depthwise_conv_BN_mean) / sizeof(float),
  &MBConv6_9_DW_BN_VARIANCE,		MBConv6_9_depthwise_conv_BN_variance,	sizeof(MBConv6_9_depthwise_conv_BN_variance) / sizeof(float),	
  &MBConv6_9_DW_BN_WEIGHTS,     MBConv6_9_depthwise_conv_BN_weights,	sizeof(MBConv6_9_depthwise_conv_BN_weights) / sizeof(float),
  &MBConv6_9_DW_BN_BIAS,				MBConv6_9_depthwise_conv_BN_bias,		  sizeof(MBConv6_9_depthwise_conv_BN_bias) / sizeof(float),
  &MBConv6_9_PRJ_BN_MEAN,       MBConv6_9_project_conv_BN_mean,			  sizeof(MBConv6_9_project_conv_BN_mean) / sizeof(float),
  &MBConv6_9_PRJ_BN_VARIANCE,		MBConv6_9_project_conv_BN_variance,		sizeof(MBConv6_9_project_conv_BN_variance) / sizeof(float),
  &MBConv6_9_PRJ_BN_WEIGHTS,    MBConv6_9_project_conv_BN_weights,		sizeof(MBConv6_9_project_conv_BN_weights) / sizeof(float),
  &MBConv6_9_PRJ_BN_BIAS,				MBConv6_9_project_conv_BN_bias, 		  sizeof(MBConv6_9_project_conv_BN_bias) / sizeof(float));

DEFINE_FILTERS_FOR_MBCONV_BN(	
  &MBConv6_10_EXPD_BN_MEAN,     MBConv6_10_expansion_conv_BN_mean,    sizeof(MBConv6_10_expansion_conv_BN_mean) / sizeof(float),
  &MBConv6_10_EXPD_BN_VARIANCE,	MBConv6_10_expansion_conv_BN_variance,sizeof(MBConv6_10_expansion_conv_BN_variance) / sizeof(float),
  &MBConv6_10_EXPD_BN_WEIGHTS,  MBConv6_10_expansion_conv_BN_weights,	sizeof(MBConv6_10_expansion_conv_BN_weights) / sizeof(float),
  &MBConv6_10_EXPD_BN_BIAS,			MBConv6_10_expansion_conv_BN_bias,		sizeof(MBConv6_10_expansion_conv_BN_bias) / sizeof(float),
  &MBConv6_10_DW_BN_MEAN,       MBConv6_10_depthwise_conv_BN_mean,		sizeof(MBConv6_10_depthwise_conv_BN_mean) / sizeof(float),
  &MBConv6_10_DW_BN_VARIANCE,		MBConv6_10_depthwise_conv_BN_variance,sizeof(MBConv6_10_depthwise_conv_BN_variance) / sizeof(float),	
  &MBConv6_10_DW_BN_WEIGHTS,    MBConv6_10_depthwise_conv_BN_weights,	sizeof(MBConv6_10_depthwise_conv_BN_weights) / sizeof(float),
  &MBConv6_10_DW_BN_BIAS,				MBConv6_10_depthwise_conv_BN_bias,		sizeof(MBConv6_10_depthwise_conv_BN_bias) / sizeof(float),
  &MBConv6_10_PRJ_BN_MEAN,      MBConv6_10_project_conv_BN_mean,		  sizeof(MBConv6_10_project_conv_BN_mean) / sizeof(float),
  &MBConv6_10_PRJ_BN_VARIANCE,	MBConv6_10_project_conv_BN_variance,	sizeof(MBConv6_10_project_conv_BN_variance) / sizeof(float),
  &MBConv6_10_PRJ_BN_WEIGHTS,   MBConv6_10_project_conv_BN_weights,		sizeof(MBConv6_10_project_conv_BN_weights) / sizeof(float),
  &MBConv6_10_PRJ_BN_BIAS,			MBConv6_10_project_conv_BN_bias, 		  sizeof(MBConv6_10_project_conv_BN_bias) / sizeof(float));

DEFINE_FILTERS_FOR_MBCONV_BN(	
  &MBConv6_11_EXPD_BN_MEAN,     MBConv6_11_expansion_conv_BN_mean,		sizeof(MBConv6_11_expansion_conv_BN_mean) / sizeof(float),
  &MBConv6_11_EXPD_BN_VARIANCE,	MBConv6_11_expansion_conv_BN_variance,sizeof(MBConv6_11_expansion_conv_BN_variance) / sizeof(float),
  &MBConv6_11_EXPD_BN_WEIGHTS,  MBConv6_11_expansion_conv_BN_weights,	sizeof(MBConv6_11_expansion_conv_BN_weights) / sizeof(float),
  &MBConv6_11_EXPD_BN_BIAS,			MBConv6_11_expansion_conv_BN_bias,		sizeof(MBConv6_11_expansion_conv_BN_bias) / sizeof(float),
  &MBConv6_11_DW_BN_MEAN,       MBConv6_11_depthwise_conv_BN_mean,		sizeof(MBConv6_11_depthwise_conv_BN_mean) / sizeof(float),
  &MBConv6_11_DW_BN_VARIANCE,		MBConv6_11_depthwise_conv_BN_variance,sizeof(MBConv6_11_depthwise_conv_BN_variance) / sizeof(float),	
  &MBConv6_11_DW_BN_WEIGHTS,    MBConv6_11_depthwise_conv_BN_weights,	sizeof(MBConv6_11_depthwise_conv_BN_weights) / sizeof(float),
  &MBConv6_11_DW_BN_BIAS,				MBConv6_11_depthwise_conv_BN_bias,		sizeof(MBConv6_11_depthwise_conv_BN_bias) / sizeof(float),
  &MBConv6_11_PRJ_BN_MEAN,      MBConv6_11_project_conv_BN_mean,		  sizeof(MBConv6_11_project_conv_BN_mean) / sizeof(float),
  &MBConv6_11_PRJ_BN_VARIANCE,	MBConv6_11_project_conv_BN_variance,	sizeof(MBConv6_11_project_conv_BN_variance) / sizeof(float),
  &MBConv6_11_PRJ_BN_WEIGHTS,   MBConv6_11_project_conv_BN_weights,		sizeof(MBConv6_11_project_conv_BN_weights) / sizeof(float),
  &MBConv6_11_PRJ_BN_BIAS,			MBConv6_11_project_conv_BN_bias, 		  sizeof(MBConv6_11_project_conv_BN_bias) / sizeof(float));

  DEFINE_FILTERS_FOR_MBCONV_BN(	
  &MBConv6_12_EXPD_BN_MEAN,     MBConv6_12_expansion_conv_BN_mean,		sizeof(MBConv6_12_expansion_conv_BN_mean) / sizeof(float),
  &MBConv6_12_EXPD_BN_VARIANCE,	MBConv6_12_expansion_conv_BN_variance,sizeof(MBConv6_12_expansion_conv_BN_variance) / sizeof(float),
  &MBConv6_12_EXPD_BN_WEIGHTS,  MBConv6_12_expansion_conv_BN_weights,	sizeof(MBConv6_12_expansion_conv_BN_weights) / sizeof(float),
  &MBConv6_12_EXPD_BN_BIAS,			MBConv6_12_expansion_conv_BN_bias,		sizeof(MBConv6_12_expansion_conv_BN_bias) / sizeof(float),
  &MBConv6_12_DW_BN_MEAN,       MBConv6_12_depthwise_conv_BN_mean,		sizeof(MBConv6_12_depthwise_conv_BN_mean) / sizeof(float),
  &MBConv6_12_DW_BN_VARIANCE,		MBConv6_12_depthwise_conv_BN_variance,sizeof(MBConv6_12_depthwise_conv_BN_variance) / sizeof(float),	
  &MBConv6_12_DW_BN_WEIGHTS,    MBConv6_12_depthwise_conv_BN_weights,	sizeof(MBConv6_12_depthwise_conv_BN_weights) / sizeof(float),
  &MBConv6_12_DW_BN_BIAS,				MBConv6_12_depthwise_conv_BN_bias,		sizeof(MBConv6_12_depthwise_conv_BN_bias) / sizeof(float),
  &MBConv6_12_PRJ_BN_MEAN,      MBConv6_12_project_conv_BN_mean,		  sizeof(MBConv6_12_project_conv_BN_mean) / sizeof(float),
  &MBConv6_12_PRJ_BN_VARIANCE,	MBConv6_12_project_conv_BN_variance,	sizeof(MBConv6_12_project_conv_BN_variance) / sizeof(float),
  &MBConv6_12_PRJ_BN_WEIGHTS,   MBConv6_12_project_conv_BN_weights,		sizeof(MBConv6_12_project_conv_BN_weights) / sizeof(float),
  &MBConv6_12_PRJ_BN_BIAS,			MBConv6_12_project_conv_BN_bias, 		  sizeof(MBConv6_12_project_conv_BN_bias) / sizeof(float));

DEFINE_FILTERS_FOR_MBCONV_BN(	
  &MBConv6_13_EXPD_BN_MEAN,     MBConv6_13_expansion_conv_BN_mean,		sizeof(MBConv6_13_expansion_conv_BN_mean) / sizeof(float),
  &MBConv6_13_EXPD_BN_VARIANCE,	MBConv6_13_expansion_conv_BN_variance,sizeof(MBConv6_13_expansion_conv_BN_variance) / sizeof(float),
  &MBConv6_13_EXPD_BN_WEIGHTS,  MBConv6_13_expansion_conv_BN_weights,	sizeof(MBConv6_13_expansion_conv_BN_weights) / sizeof(float),
  &MBConv6_13_EXPD_BN_BIAS,			MBConv6_13_expansion_conv_BN_bias,		sizeof(MBConv6_13_expansion_conv_BN_bias) / sizeof(float),
  &MBConv6_13_DW_BN_MEAN,       MBConv6_13_depthwise_conv_BN_mean,		sizeof(MBConv6_13_depthwise_conv_BN_mean) / sizeof(float),
  &MBConv6_13_DW_BN_VARIANCE,		MBConv6_13_depthwise_conv_BN_variance,sizeof(MBConv6_13_depthwise_conv_BN_variance) / sizeof(float),	
  &MBConv6_13_DW_BN_WEIGHTS,    MBConv6_13_depthwise_conv_BN_weights,	sizeof(MBConv6_13_depthwise_conv_BN_weights) / sizeof(float),
  &MBConv6_13_DW_BN_BIAS,				MBConv6_13_depthwise_conv_BN_bias,		sizeof(MBConv6_13_depthwise_conv_BN_bias) / sizeof(float),
  &MBConv6_13_PRJ_BN_MEAN,      MBConv6_13_project_conv_BN_mean,		  sizeof(MBConv6_13_project_conv_BN_mean) / sizeof(float),
  &MBConv6_13_PRJ_BN_VARIANCE,	MBConv6_13_project_conv_BN_variance,	sizeof(MBConv6_13_project_conv_BN_variance) / sizeof(float),
  &MBConv6_13_PRJ_BN_WEIGHTS,   MBConv6_13_project_conv_BN_weights,		sizeof(MBConv6_13_project_conv_BN_weights) / sizeof(float),
  &MBConv6_13_PRJ_BN_BIAS,			MBConv6_13_project_conv_BN_bias, 		  sizeof(MBConv6_13_project_conv_BN_bias) / sizeof(float));

DEFINE_FILTERS_FOR_MBCONV_BN(	
  &MBConv6_14_EXPD_BN_MEAN,     MBConv6_14_expansion_conv_BN_mean,		sizeof(MBConv6_14_expansion_conv_BN_mean) / sizeof(float),
  &MBConv6_14_EXPD_BN_VARIANCE,	MBConv6_14_expansion_conv_BN_variance,sizeof(MBConv6_14_expansion_conv_BN_variance) / sizeof(float),
  &MBConv6_14_EXPD_BN_WEIGHTS,  MBConv6_14_expansion_conv_BN_weights,	sizeof(MBConv6_14_expansion_conv_BN_weights) / sizeof(float),
  &MBConv6_14_EXPD_BN_BIAS,			MBConv6_14_expansion_conv_BN_bias,		sizeof(MBConv6_14_expansion_conv_BN_bias) / sizeof(float),
  &MBConv6_14_DW_BN_MEAN,       MBConv6_14_depthwise_conv_BN_mean,		sizeof(MBConv6_14_depthwise_conv_BN_mean) / sizeof(float),
  &MBConv6_14_DW_BN_VARIANCE,		MBConv6_14_depthwise_conv_BN_variance,sizeof(MBConv6_14_depthwise_conv_BN_variance) / sizeof(float),	
  &MBConv6_14_DW_BN_WEIGHTS,    MBConv6_14_depthwise_conv_BN_weights,	sizeof(MBConv6_14_depthwise_conv_BN_weights) / sizeof(float),
  &MBConv6_14_DW_BN_BIAS,				MBConv6_14_depthwise_conv_BN_bias,		sizeof(MBConv6_14_depthwise_conv_BN_bias) / sizeof(float),
  &MBConv6_14_PRJ_BN_MEAN,      MBConv6_14_project_conv_BN_mean,		  sizeof(MBConv6_14_project_conv_BN_mean) / sizeof(float),
  &MBConv6_14_PRJ_BN_VARIANCE,	MBConv6_14_project_conv_BN_variance,	sizeof(MBConv6_14_project_conv_BN_variance) / sizeof(float),
  &MBConv6_14_PRJ_BN_WEIGHTS,   MBConv6_14_project_conv_BN_weights,		sizeof(MBConv6_14_project_conv_BN_weights) / sizeof(float),
  &MBConv6_14_PRJ_BN_BIAS,			MBConv6_14_project_conv_BN_bias, 		  sizeof(MBConv6_14_project_conv_BN_bias) / sizeof(float));

DEFINE_FILTERS_FOR_MBCONV_BN(	
  &MBConv6_15_EXPD_BN_MEAN,     MBConv6_15_expansion_conv_BN_mean,		sizeof(MBConv6_15_expansion_conv_BN_mean) / sizeof(float),
  &MBConv6_15_EXPD_BN_VARIANCE,	MBConv6_15_expansion_conv_BN_variance,sizeof(MBConv6_15_expansion_conv_BN_variance) / sizeof(float),
  &MBConv6_15_EXPD_BN_WEIGHTS,  MBConv6_15_expansion_conv_BN_weights,	sizeof(MBConv6_15_expansion_conv_BN_weights) / sizeof(float),
  &MBConv6_15_EXPD_BN_BIAS,			MBConv6_15_expansion_conv_BN_bias,		sizeof(MBConv6_15_expansion_conv_BN_bias) / sizeof(float),
  &MBConv6_15_DW_BN_MEAN,       MBConv6_15_depthwise_conv_BN_mean,		sizeof(MBConv6_15_depthwise_conv_BN_mean) / sizeof(float),
  &MBConv6_15_DW_BN_VARIANCE,		MBConv6_15_depthwise_conv_BN_variance,sizeof(MBConv6_15_depthwise_conv_BN_variance) / sizeof(float),	
  &MBConv6_15_DW_BN_WEIGHTS,    MBConv6_15_depthwise_conv_BN_weights,	sizeof(MBConv6_15_depthwise_conv_BN_weights) / sizeof(float),
  &MBConv6_15_DW_BN_BIAS,				MBConv6_15_depthwise_conv_BN_bias,		sizeof(MBConv6_15_depthwise_conv_BN_bias) / sizeof(float),
  &MBConv6_15_PRJ_BN_MEAN,      MBConv6_15_project_conv_BN_mean,		  sizeof(MBConv6_15_project_conv_BN_mean) / sizeof(float),
  &MBConv6_15_PRJ_BN_VARIANCE,	MBConv6_15_project_conv_BN_variance,	sizeof(MBConv6_15_project_conv_BN_variance) / sizeof(float),
  &MBConv6_15_PRJ_BN_WEIGHTS,   MBConv6_15_project_conv_BN_weights,		sizeof(MBConv6_15_project_conv_BN_weights) / sizeof(float),
  &MBConv6_15_PRJ_BN_BIAS,			MBConv6_15_project_conv_BN_bias, 		  sizeof(MBConv6_15_project_conv_BN_bias) / sizeof(float));


set_allocate_copy_array_Device(&D_STEM_BN_MEAN, Stem_BN_mean,
                sizeof(Stem_BN_mean)/sizeof(float), 1, 1,
                "STEM MEAN"); 
set_allocate_copy_array_Device(&D_STEM_BN_VARIANCE, Stem_BN_variance,
                sizeof(Stem_BN_variance)/sizeof(float), 1, 1,
                "STEAM VARIANCE"); 
set_allocate_copy_array_Device(&D_STEM_BN_WEIGHTS, Stem_BN_weights,
                sizeof(Stem_BN_weights)/sizeof(float), 1, 1,
                "STEM WEIGHTS"); 
set_allocate_copy_array_Device(&D_STEM_BN_BIAS, Stem_BN_bias,
                sizeof(Stem_BN_bias)/sizeof(float), 1, 1,
                "STEM BIAS"); 
                
set_allocate_copy_array_Device(&D_HEAD_BN_MEAN, Head_BN_mean,
                sizeof(Head_BN_mean)/sizeof(float), 1, 1,
                "HEAD MEAN"); 
set_allocate_copy_array_Device(&D_HEAD_BN_VARIANCE, Head_BN_variance,
                sizeof(Head_BN_variance)/sizeof(float), 1, 1,
                "HEAD VARIANCE"); 
set_allocate_copy_array_Device(&D_HEAD_BN_WEIGHTS, Head_BN_weights,
                sizeof(Head_BN_weights)/sizeof(float), 1, 1,
                "HEAD WEIGHTS"); 
set_allocate_copy_array_Device(&D_HEAD_BN_BIAS, Head_BN_bias,
                sizeof(Head_BN_bias)/sizeof(float), 1, 1,
                "HEAD BIAS"); 
start();
  // 3. Move through all layers starting from stem layer till head layer
  Matrix ConvOutStem;
  STEM_LAYER(&DInput_Mat, &F_STEM,
              INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, INPUT_IMAGE_DEPTH,
              STEM_FILTER_HEIGHT, STEM_FILTER_WIDTH, STEM_FILTER_DEPTH, 
              STEM_FILTER_DENSITY,STEM_PADDING,      STEM_STRIDE,
              &ConvOutStem);


  Matrix ConvOut1_0;
  MBCONV1_0_flag = 1;  

  MBConv_Layer(&ConvOutStem, &ConvOut1_0,
                &D_MBConv_1_0_EXPD_WEIGHTS, &D_MBConv_1_0_DW_WEIGHTS,
                &D_MBConv_1_0_SQZ_1_WEIGHTS,&D_MBConv_1_0_SQZ_2_WEIGHTS,
                &D_MBConv_1_0_PRJ_WEIGHTS,
                MBCONV_1_0_EXPD_F_DENSITY,  MBCONV_1_0_DW_F_DENSITY, 
                MBCONV_1_0_SQZ_1_F_DENSITY, MBCONV_1_0_SQZ_2_F_DENSITY, 
                MBCONV_1_0_PRJ_F_DENSITY,
                ConvOutStem.depth,          MBCONV_1_0_PRJ_F_DENSITY, MBCONV_1_0_DW_F_HEIGHT,
                MBCONV_1_0_STRIDE,          MBCONV_1_0_PADDING, MBCONV_1_0_SKIP,
                &MBConv1_0_SQZ_1_bias, 	    &MBConv1_0_SQZ_2_bias,
                NULL,                       NULL,
                NULL,                       NULL,
                &MBConv1_0_DW_BN_MEAN,      &MBConv1_0_DW_BN_VARIANCE,
                &MBConv1_0_DW_BN_WEIGHTS,   &MBConv1_0_DW_BN_BIAS,
                &MBConv1_0_PRJ_BN_MEAN,     &MBConv1_0_PRJ_BN_VARIANCE,
                &MBConv1_0_PRJ_BN_WEIGHTS,  &MBConv1_0_PRJ_BN_BIAS);
  MBCONV1_0_flag = 0;
  

  Matrix ConvOut;
  MBConv_Layer(&ConvOut1_0, &ConvOut,
                &D_MBConv_6_1_EXPD_WEIGHTS, &D_MBConv_6_1_DW_WEIGHTS,
                &D_MBConv_6_1_SQZ_1_WEIGHTS,&D_MBConv_6_1_SQZ_2_WEIGHTS,
                &D_MBConv_6_1_PRJ_WEIGHTS,
                MBCONV_6_1_EXPD_F_DENSITY,  MBCONV_6_1_DW_F_DENSITY, 
                MBCONV_6_1_SQZ_1_F_DENSITY, MBCONV_6_1_SQZ_2_F_DENSITY, 
                MBCONV_6_1_PRJ_F_DENSITY,
                ConvOut1_0.depth,           MBCONV_6_1_PRJ_F_DENSITY, MBCONV_6_1_DW_F_HEIGHT,
                MBCONV_6_1_STRIDE,          MBCONV_6_1_PADDING, MBCONV_6_1_SKIP,
                &MBConv6_1_SQZ_1_bias, 	    &MBConv6_1_SQZ_2_bias,
                &MBConv6_1_EXPD_BN_MEAN,    &MBConv6_1_EXPD_BN_VARIANCE,
                &MBConv6_1_EXPD_BN_WEIGHTS, &MBConv6_1_EXPD_BN_BIAS,
                &MBConv6_1_DW_BN_MEAN,      &MBConv6_1_DW_BN_VARIANCE,
                &MBConv6_1_DW_BN_WEIGHTS,   &MBConv6_1_DW_BN_BIAS,
                &MBConv6_1_PRJ_BN_MEAN,     &MBConv6_1_PRJ_BN_VARIANCE,
                &MBConv6_1_PRJ_BN_WEIGHTS,  &MBConv6_1_PRJ_BN_BIAS);


  Matrix ConvOut2;
  MBConv_Layer(&ConvOut, &ConvOut2,
                &D_MBConv_6_2_EXPD_WEIGHTS, &D_MBConv_6_2_DW_WEIGHTS,
                &D_MBConv_6_2_SQZ_1_WEIGHTS,&D_MBConv_6_2_SQZ_2_WEIGHTS,
                &D_MBConv_6_2_PRJ_WEIGHTS,
                MBCONV_6_2_EXPD_F_DENSITY,  MBCONV_6_2_DW_F_DENSITY, 
                MBCONV_6_2_SQZ_1_F_DENSITY, MBCONV_6_2_SQZ_2_F_DENSITY, 
                MBCONV_6_2_PRJ_F_DENSITY,
                ConvOut.depth,              MBCONV_6_2_PRJ_F_DENSITY, MBCONV_6_2_DW_F_HEIGHT,
                MBCONV_6_2_STRIDE,          MBCONV_6_2_PADDING, MBCONV_6_2_SKIP,
                &MBConv6_2_SQZ_1_bias, 	    &MBConv6_2_SQZ_2_bias,
                &MBConv6_2_EXPD_BN_MEAN,    &MBConv6_2_EXPD_BN_VARIANCE,
                &MBConv6_2_EXPD_BN_WEIGHTS, &MBConv6_2_EXPD_BN_BIAS,
                &MBConv6_2_DW_BN_MEAN,      &MBConv6_2_DW_BN_VARIANCE,
                &MBConv6_2_DW_BN_WEIGHTS,   &MBConv6_2_DW_BN_BIAS,
                &MBConv6_2_PRJ_BN_MEAN,     &MBConv6_2_PRJ_BN_VARIANCE,
                &MBConv6_2_PRJ_BN_WEIGHTS,  &MBConv6_2_PRJ_BN_BIAS); 


  Matrix ConvOut3;
	MBConv_Layer(&ConvOut2, &ConvOut3,
                &D_MBConv_6_3_EXPD_WEIGHTS, &D_MBConv_6_3_DW_WEIGHTS,
                &D_MBConv_6_3_SQZ_1_WEIGHTS,&D_MBConv_6_3_SQZ_2_WEIGHTS,
                &D_MBConv_6_3_PRJ_WEIGHTS,
                MBCONV_6_3_EXPD_F_DENSITY,  MBCONV_6_3_DW_F_DENSITY, 
                MBCONV_6_3_SQZ_1_F_DENSITY, MBCONV_6_3_SQZ_2_F_DENSITY, 
                MBCONV_6_3_PRJ_F_DENSITY,
                ConvOut2.depth,             MBCONV_6_3_PRJ_F_DENSITY, MBCONV_6_3_DW_F_HEIGHT,
                MBCONV_6_3_STRIDE,          MBCONV_6_3_PADDING, MBCONV_6_3_SKIP,
                &MBConv6_3_SQZ_1_bias,  	  &MBConv6_3_SQZ_2_bias,
                &MBConv6_3_EXPD_BN_MEAN,    &MBConv6_3_EXPD_BN_VARIANCE,
                &MBConv6_3_EXPD_BN_WEIGHTS, &MBConv6_3_EXPD_BN_BIAS,
                &MBConv6_3_DW_BN_MEAN,      &MBConv6_3_DW_BN_VARIANCE,
                &MBConv6_3_DW_BN_WEIGHTS,   &MBConv6_3_DW_BN_BIAS,
                &MBConv6_3_PRJ_BN_MEAN,     &MBConv6_3_PRJ_BN_VARIANCE,
                &MBConv6_3_PRJ_BN_WEIGHTS,  &MBConv6_3_PRJ_BN_BIAS);  
 

  // MBConv6_4 layer implementation

  Matrix ConvOut4;
  MBConv_Layer(&ConvOut3, &ConvOut4,
                &D_MBConv_6_4_EXPD_WEIGHTS, &D_MBConv_6_4_DW_WEIGHTS,
                &D_MBConv_6_4_SQZ_1_WEIGHTS,&D_MBConv_6_4_SQZ_2_WEIGHTS,
                &D_MBConv_6_4_PRJ_WEIGHTS,
                MBCONV_6_4_EXPD_F_DENSITY,  MBCONV_6_4_DW_F_DENSITY, 
                MBCONV_6_4_SQZ_1_F_DENSITY, MBCONV_6_4_SQZ_2_F_DENSITY, 
                MBCONV_6_4_PRJ_F_DENSITY,
                ConvOut3.depth,             MBCONV_6_4_PRJ_F_DENSITY, MBCONV_6_4_DW_F_HEIGHT,
                MBCONV_6_4_STRIDE,          MBCONV_6_4_PADDING, MBCONV_6_4_SKIP,
                &MBConv6_4_SQZ_1_bias,  	  &MBConv6_4_SQZ_2_bias,
                &MBConv6_4_EXPD_BN_MEAN,    &MBConv6_4_EXPD_BN_VARIANCE,
                &MBConv6_4_EXPD_BN_WEIGHTS, &MBConv6_4_EXPD_BN_BIAS,
                &MBConv6_4_DW_BN_MEAN,      &MBConv6_4_DW_BN_VARIANCE,
                &MBConv6_4_DW_BN_WEIGHTS,   &MBConv6_4_DW_BN_BIAS,
                &MBConv6_4_PRJ_BN_MEAN,     &MBConv6_4_PRJ_BN_VARIANCE,
                &MBConv6_4_PRJ_BN_WEIGHTS,  &MBConv6_4_PRJ_BN_BIAS);   
  

  Matrix ConvOut5;
  MBConv_Layer(&ConvOut4, &ConvOut5,
                &D_MBConv_6_5_EXPD_WEIGHTS, &D_MBConv_6_5_DW_WEIGHTS,
                &D_MBConv_6_5_SQZ_1_WEIGHTS,&D_MBConv_6_5_SQZ_2_WEIGHTS,
                &D_MBConv_6_5_PRJ_WEIGHTS,
                MBCONV_6_5_EXPD_F_DENSITY,  MBCONV_6_5_DW_F_DENSITY, 
                MBCONV_6_5_SQZ_1_F_DENSITY, MBCONV_6_5_SQZ_2_F_DENSITY, 
                MBCONV_6_5_PRJ_F_DENSITY,
                ConvOut4.depth,             MBCONV_6_5_PRJ_F_DENSITY, MBCONV_6_5_DW_F_HEIGHT,
                MBCONV_6_5_STRIDE,          MBCONV_6_5_PADDING, MBCONV_6_5_SKIP,
                &MBConv6_5_SQZ_1_bias,  	  &MBConv6_5_SQZ_2_bias,
                &MBConv6_5_EXPD_BN_MEAN,    &MBConv6_5_EXPD_BN_VARIANCE,
                &MBConv6_5_EXPD_BN_WEIGHTS, &MBConv6_5_EXPD_BN_BIAS,
                &MBConv6_5_DW_BN_MEAN,      &MBConv6_5_DW_BN_VARIANCE,
                &MBConv6_5_DW_BN_WEIGHTS,   &MBConv6_5_DW_BN_BIAS,
                &MBConv6_5_PRJ_BN_MEAN,     &MBConv6_5_PRJ_BN_VARIANCE,
                &MBConv6_5_PRJ_BN_WEIGHTS,  &MBConv6_5_PRJ_BN_BIAS); 
            


  // MBConv6_6 layer implementation


  Matrix ConvOut6;
  MBConv_Layer(&ConvOut5, &ConvOut6,
                &D_MBConv_6_6_EXPD_WEIGHTS, &D_MBConv_6_6_DW_WEIGHTS,
                &D_MBConv_6_6_SQZ_1_WEIGHTS,&D_MBConv_6_6_SQZ_2_WEIGHTS,
                &D_MBConv_6_6_PRJ_WEIGHTS,
                MBCONV_6_6_EXPD_F_DENSITY,  MBCONV_6_6_DW_F_DENSITY, 
                MBCONV_6_6_SQZ_1_F_DENSITY, MBCONV_6_6_SQZ_2_F_DENSITY, 
                MBCONV_6_6_PRJ_F_DENSITY,
                ConvOut5.depth,             MBCONV_6_6_PRJ_F_DENSITY, MBCONV_6_6_DW_F_HEIGHT,
                MBCONV_6_6_STRIDE,          MBCONV_6_6_PADDING, MBCONV_6_6_SKIP,
                &MBConv6_6_SQZ_1_bias, 	    &MBConv6_6_SQZ_2_bias,
                &MBConv6_6_EXPD_BN_MEAN,    &MBConv6_6_EXPD_BN_VARIANCE,
                &MBConv6_6_EXPD_BN_WEIGHTS, &MBConv6_6_EXPD_BN_BIAS,
                &MBConv6_6_DW_BN_MEAN,      &MBConv6_6_DW_BN_VARIANCE,
                &MBConv6_6_DW_BN_WEIGHTS,   &MBConv6_6_DW_BN_BIAS,
                &MBConv6_6_PRJ_BN_MEAN,     &MBConv6_6_PRJ_BN_VARIANCE,
                &MBConv6_6_PRJ_BN_WEIGHTS,  &MBConv6_6_PRJ_BN_BIAS);  
            


  // MBConv6_7 layer implementation


  Matrix ConvOut7;
  MBConv_Layer(&ConvOut6, &ConvOut7,
                &D_MBConv_6_7_EXPD_WEIGHTS, &D_MBConv_6_7_DW_WEIGHTS,
                &D_MBConv_6_7_SQZ_1_WEIGHTS,&D_MBConv_6_7_SQZ_2_WEIGHTS,
                &D_MBConv_6_7_PRJ_WEIGHTS,
                MBCONV_6_7_EXPD_F_DENSITY,  MBCONV_6_7_DW_F_DENSITY, 
                MBCONV_6_7_SQZ_1_F_DENSITY, MBCONV_6_7_SQZ_2_F_DENSITY, 
                MBCONV_6_7_PRJ_F_DENSITY,
                ConvOut6.depth,             MBCONV_6_7_PRJ_F_DENSITY, MBCONV_6_7_DW_F_HEIGHT,                   
                MBCONV_6_7_STRIDE,          MBCONV_6_7_PADDING, MBCONV_6_7_SKIP,
                &MBConv6_7_SQZ_1_bias,  	  &MBConv6_7_SQZ_2_bias,
                &MBConv6_7_EXPD_BN_MEAN,    &MBConv6_7_EXPD_BN_VARIANCE,
                &MBConv6_7_EXPD_BN_WEIGHTS, &MBConv6_7_EXPD_BN_BIAS,
                &MBConv6_7_DW_BN_MEAN,      &MBConv6_7_DW_BN_VARIANCE,
                &MBConv6_7_DW_BN_WEIGHTS,   &MBConv6_7_DW_BN_BIAS,
                &MBConv6_7_PRJ_BN_MEAN,     &MBConv6_7_PRJ_BN_VARIANCE,
                &MBConv6_7_PRJ_BN_WEIGHTS,  &MBConv6_7_PRJ_BN_BIAS);  
          


  // MBConv6_8 layer implementation
  Matrix ConvOut8;
  MBConv_Layer(&ConvOut7, &ConvOut8,
                &D_MBConv_6_8_EXPD_WEIGHTS, &D_MBConv_6_8_DW_WEIGHTS,
                &D_MBConv_6_8_SQZ_1_WEIGHTS,&D_MBConv_6_8_SQZ_2_WEIGHTS,
                &D_MBConv_6_8_PRJ_WEIGHTS,
                MBCONV_6_8_EXPD_F_DENSITY,  MBCONV_6_8_DW_F_DENSITY, 
                MBCONV_6_8_SQZ_1_F_DENSITY, MBCONV_6_8_SQZ_2_F_DENSITY, 
                MBCONV_6_8_PRJ_F_DENSITY,
                ConvOut7.depth,             MBCONV_6_8_PRJ_F_DENSITY, MBCONV_6_8_DW_F_HEIGHT,    
                MBCONV_6_8_STRIDE,          MBCONV_6_8_PADDING, MBCONV_6_8_SKIP,
                &MBConv6_8_SQZ_1_bias,      &MBConv6_8_SQZ_2_bias,
                &MBConv6_8_EXPD_BN_MEAN,    &MBConv6_8_EXPD_BN_VARIANCE,
                &MBConv6_8_EXPD_BN_WEIGHTS, &MBConv6_8_EXPD_BN_BIAS,
                &MBConv6_8_DW_BN_MEAN,      &MBConv6_8_DW_BN_VARIANCE,
                &MBConv6_8_DW_BN_WEIGHTS,   &MBConv6_8_DW_BN_BIAS,
                &MBConv6_8_PRJ_BN_MEAN,     &MBConv6_8_PRJ_BN_VARIANCE,
                &MBConv6_8_PRJ_BN_WEIGHTS,  &MBConv6_8_PRJ_BN_BIAS); 
        


  // MBConv6_9 layer implementation
  Matrix ConvOut9;
  MBConv_Layer(&ConvOut8, &ConvOut9,
                &D_MBConv_6_9_EXPD_WEIGHTS, &D_MBConv_6_9_DW_WEIGHTS,
                &D_MBConv_6_9_SQZ_1_WEIGHTS,&D_MBConv_6_9_SQZ_2_WEIGHTS,
                &D_MBConv_6_9_PRJ_WEIGHTS,
                MBCONV_6_9_EXPD_F_DENSITY,  MBCONV_6_9_DW_F_DENSITY, 
                MBCONV_6_9_SQZ_1_F_DENSITY, MBCONV_6_9_SQZ_2_F_DENSITY, 
                MBCONV_6_9_PRJ_F_DENSITY,
                ConvOut8.depth,             MBCONV_6_9_PRJ_F_DENSITY, MBCONV_6_9_DW_F_HEIGHT,
                MBCONV_6_9_STRIDE,          MBCONV_6_9_PADDING, MBCONV_6_9_SKIP,
                &MBConv6_9_SQZ_1_bias,  	  &MBConv6_9_SQZ_2_bias,
                &MBConv6_9_EXPD_BN_MEAN,    &MBConv6_9_EXPD_BN_VARIANCE,
                &MBConv6_9_EXPD_BN_WEIGHTS, &MBConv6_9_EXPD_BN_BIAS,
                &MBConv6_9_DW_BN_MEAN,      &MBConv6_9_DW_BN_VARIANCE,
                &MBConv6_9_DW_BN_WEIGHTS,   &MBConv6_9_DW_BN_BIAS,
                &MBConv6_9_PRJ_BN_MEAN,     &MBConv6_9_PRJ_BN_VARIANCE,
                &MBConv6_9_PRJ_BN_WEIGHTS,  &MBConv6_9_PRJ_BN_BIAS);  				  



  // MBConv6_10 layer implementation
  Matrix ConvOut10;
	MBConv_Layer(&ConvOut9, &ConvOut10,
                &D_MBConv_6_10_EXPD_WEIGHTS,  &D_MBConv_6_10_DW_WEIGHTS,
                &D_MBConv_6_10_SQZ_1_WEIGHTS, &D_MBConv_6_10_SQZ_2_WEIGHTS,
                &D_MBConv_6_10_PRJ_WEIGHTS,
                MBCONV_6_10_EXPD_F_DENSITY,   MBCONV_6_10_DW_F_DENSITY, 
                MBCONV_6_10_SQZ_1_F_DENSITY,  MBCONV_6_10_SQZ_2_F_DENSITY, 
                MBCONV_6_10_PRJ_F_DENSITY,
                ConvOut9.depth,               MBCONV_6_10_PRJ_F_DENSITY, MBCONV_6_10_DW_F_HEIGHT,
                MBCONV_6_10_STRIDE,           MBCONV_6_10_PADDING, MBCONV_6_10_SKIP,
                &MBConv6_10_SQZ_1_bias, 	    &MBConv6_10_SQZ_2_bias,
                &MBConv6_10_EXPD_BN_MEAN,     &MBConv6_10_EXPD_BN_VARIANCE,
                &MBConv6_10_EXPD_BN_WEIGHTS,  &MBConv6_10_EXPD_BN_BIAS,
                &MBConv6_10_DW_BN_MEAN,       &MBConv6_10_DW_BN_VARIANCE,
                &MBConv6_10_DW_BN_WEIGHTS,    &MBConv6_10_DW_BN_BIAS,
                &MBConv6_10_PRJ_BN_MEAN,      &MBConv6_10_PRJ_BN_VARIANCE,
                &MBConv6_10_PRJ_BN_WEIGHTS,   &MBConv6_10_PRJ_BN_BIAS);   
  


  // MBConv6_11 layer implementation


  Matrix ConvOut11;
  MBConv_Layer(&ConvOut10, &ConvOut11,
                &D_MBConv_6_11_EXPD_WEIGHTS,  &D_MBConv_6_11_DW_WEIGHTS,
                &D_MBConv_6_11_SQZ_1_WEIGHTS, &D_MBConv_6_11_SQZ_2_WEIGHTS,
                &D_MBConv_6_11_PRJ_WEIGHTS,
                MBCONV_6_11_EXPD_F_DENSITY,   MBCONV_6_11_DW_F_DENSITY, 
                MBCONV_6_11_SQZ_1_F_DENSITY,  MBCONV_6_11_SQZ_2_F_DENSITY, 
                MBCONV_6_11_PRJ_F_DENSITY,  
                ConvOut10.depth,              MBCONV_6_11_PRJ_F_DENSITY, MBCONV_6_11_DW_F_HEIGHT,
                MBCONV_6_11_STRIDE,           MBCONV_6_11_PADDING, MBCONV_6_11_SKIP,
                &MBConv6_11_SQZ_1_bias,       &MBConv6_11_SQZ_2_bias,
                &MBConv6_11_EXPD_BN_MEAN,     &MBConv6_11_EXPD_BN_VARIANCE,
                &MBConv6_11_EXPD_BN_WEIGHTS,  &MBConv6_11_EXPD_BN_BIAS,
                &MBConv6_11_DW_BN_MEAN,       &MBConv6_11_DW_BN_VARIANCE,
                &MBConv6_11_DW_BN_WEIGHTS,    &MBConv6_11_DW_BN_BIAS,
                &MBConv6_11_PRJ_BN_MEAN,      &MBConv6_11_PRJ_BN_VARIANCE,
                &MBConv6_11_PRJ_BN_WEIGHTS,   &MBConv6_11_PRJ_BN_BIAS);  
  


  // MBConv6_12 layer implementation


  Matrix ConvOut12;
  MBConv_Layer(&ConvOut11, &ConvOut12,
                &D_MBConv_6_12_EXPD_WEIGHTS,  &D_MBConv_6_12_DW_WEIGHTS,
                &D_MBConv_6_12_SQZ_1_WEIGHTS, &D_MBConv_6_12_SQZ_2_WEIGHTS,
                &D_MBConv_6_12_PRJ_WEIGHTS,
                MBCONV_6_12_EXPD_F_DENSITY,   MBCONV_6_12_DW_F_DENSITY, 
                MBCONV_6_12_SQZ_1_F_DENSITY,  MBCONV_6_12_SQZ_2_F_DENSITY, 
                MBCONV_6_12_PRJ_F_DENSITY,
                ConvOut11.depth,              MBCONV_6_12_PRJ_F_DENSITY, MBCONV_6_12_DW_F_HEIGHT,
                MBCONV_6_12_STRIDE,           MBCONV_6_12_PADDING, MBCONV_6_12_SKIP,
                &MBConv6_12_SQZ_1_bias,       &MBConv6_12_SQZ_2_bias,
                &MBConv6_12_EXPD_BN_MEAN,     &MBConv6_12_EXPD_BN_VARIANCE,
                &MBConv6_12_EXPD_BN_WEIGHTS,  &MBConv6_12_EXPD_BN_BIAS,
                &MBConv6_12_DW_BN_MEAN,       &MBConv6_12_DW_BN_VARIANCE,
                &MBConv6_12_DW_BN_WEIGHTS,    &MBConv6_12_DW_BN_BIAS,
                &MBConv6_12_PRJ_BN_MEAN,      &MBConv6_12_PRJ_BN_VARIANCE,
                &MBConv6_12_PRJ_BN_WEIGHTS,   &MBConv6_12_PRJ_BN_BIAS);   
  


  // MBConv6_13 layer implementation

  Matrix ConvOut13;
  MBConv_Layer(&ConvOut12, &ConvOut13,
                &D_MBConv_6_13_EXPD_WEIGHTS,  &D_MBConv_6_13_DW_WEIGHTS,
                &D_MBConv_6_13_SQZ_1_WEIGHTS, &D_MBConv_6_13_SQZ_2_WEIGHTS,
                &D_MBConv_6_13_PRJ_WEIGHTS,
                MBCONV_6_13_EXPD_F_DENSITY,   MBCONV_6_13_DW_F_DENSITY, 
                MBCONV_6_13_SQZ_1_F_DENSITY,  MBCONV_6_13_SQZ_2_F_DENSITY, 
                MBCONV_6_13_PRJ_F_DENSITY,
                ConvOut12.depth,              MBCONV_6_13_PRJ_F_DENSITY, MBCONV_6_13_DW_F_HEIGHT,
                MBCONV_6_13_STRIDE,           MBCONV_6_13_PADDING, MBCONV_6_13_SKIP,
                &MBConv6_13_SQZ_1_bias,       &MBConv6_13_SQZ_2_bias,
                &MBConv6_13_EXPD_BN_MEAN,     &MBConv6_13_EXPD_BN_VARIANCE,
                &MBConv6_13_EXPD_BN_WEIGHTS,  &MBConv6_13_EXPD_BN_BIAS,
                &MBConv6_13_DW_BN_MEAN,       &MBConv6_13_DW_BN_VARIANCE,
                &MBConv6_13_DW_BN_WEIGHTS,    &MBConv6_13_DW_BN_BIAS,
                &MBConv6_13_PRJ_BN_MEAN,      &MBConv6_13_PRJ_BN_VARIANCE,
                &MBConv6_13_PRJ_BN_WEIGHTS,   &MBConv6_13_PRJ_BN_BIAS);


  Matrix ConvOut14;
  MBConv_Layer(&ConvOut13, &ConvOut14,
                &D_MBConv_6_14_EXPD_WEIGHTS,  &D_MBConv_6_14_DW_WEIGHTS,
                &D_MBConv_6_14_SQZ_1_WEIGHTS, &D_MBConv_6_14_SQZ_2_WEIGHTS,
                &D_MBConv_6_14_PRJ_WEIGHTS,
                MBCONV_6_14_EXPD_F_DENSITY,   MBCONV_6_14_DW_F_DENSITY, 
                MBCONV_6_14_SQZ_1_F_DENSITY,  MBCONV_6_14_SQZ_2_F_DENSITY, 
                MBCONV_6_14_PRJ_F_DENSITY,
                ConvOut13.depth,              MBCONV_6_14_PRJ_F_DENSITY, MBCONV_6_14_DW_F_HEIGHT,
                MBCONV_6_14_STRIDE,           MBCONV_6_14_PADDING, MBCONV_6_14_SKIP,
                &MBConv6_14_SQZ_1_bias, 	    &MBConv6_14_SQZ_2_bias,
                &MBConv6_14_EXPD_BN_MEAN,     &MBConv6_14_EXPD_BN_VARIANCE,
                &MBConv6_14_EXPD_BN_WEIGHTS,  &MBConv6_14_EXPD_BN_BIAS,
                &MBConv6_14_DW_BN_MEAN,       &MBConv6_14_DW_BN_VARIANCE,
                &MBConv6_14_DW_BN_WEIGHTS,    &MBConv6_14_DW_BN_BIAS,
                &MBConv6_14_PRJ_BN_MEAN,      &MBConv6_14_PRJ_BN_VARIANCE,
                &MBConv6_14_PRJ_BN_WEIGHTS,   &MBConv6_14_PRJ_BN_BIAS);  


  Matrix ConvOut15;
  MBConv_Layer(&ConvOut14, &ConvOut15,
                &D_MBConv_6_15_EXPD_WEIGHTS,  &D_MBConv_6_15_DW_WEIGHTS,
                &D_MBConv_6_15_SQZ_1_WEIGHTS, &D_MBConv_6_15_SQZ_2_WEIGHTS,
                &D_MBConv_6_15_PRJ_WEIGHTS,
                MBCONV_6_15_EXPD_F_DENSITY,   MBCONV_6_15_DW_F_DENSITY, 
                MBCONV_6_15_SQZ_1_F_DENSITY,  MBCONV_6_15_SQZ_2_F_DENSITY, 
                MBCONV_6_15_PRJ_F_DENSITY,
                ConvOut14.depth,              MBCONV_6_15_PRJ_F_DENSITY, MBCONV_6_15_DW_F_HEIGHT,
                MBCONV_6_15_STRIDE,           MBCONV_6_15_PADDING, MBCONV_6_15_SKIP,
                &MBConv6_15_SQZ_1_bias,       &MBConv6_15_SQZ_2_bias,
                &MBConv6_15_EXPD_BN_MEAN,     &MBConv6_15_EXPD_BN_VARIANCE,
                &MBConv6_15_EXPD_BN_WEIGHTS,  &MBConv6_15_EXPD_BN_BIAS,
                &MBConv6_15_DW_BN_MEAN,       &MBConv6_15_DW_BN_VARIANCE,
                &MBConv6_15_DW_BN_WEIGHTS,    &MBConv6_15_DW_BN_BIAS,
                &MBConv6_15_PRJ_BN_MEAN,      &MBConv6_15_PRJ_BN_VARIANCE,
                &MBConv6_15_PRJ_BN_WEIGHTS,   &MBConv6_15_PRJ_BN_BIAS);   

  // Head layer
  Matrix HEAD_OUT;
  HEAD_LAYER(&ConvOut15, &HEAD_CONV_WEIGHTS, &HEAD_FC_WEIGHTS,
              HEAD_CONV_F_HEIGHT, HEAD_CONV_F_WIDTH, HEAD_CONV_F_DEPTH, HEAD_CONV_F_DENSITY,
              0, 1,
              &HEAD_OUT);
  
}



// The last layer in efficient net
void HEAD_LAYER(Matrix *INPUT_MATRIX, Matrix *F_HEAD, Matrix *FC_WEIGHTS,
                int filter_height, int filter_width, int filter_depth, int filter_density,
                int padding, int stride,
                Matrix *HEAD_OUT)
{                
  // Calculate output dimensions       
  int out_height = (INPUT_MATRIX -> height + 2 * padding - filter_height) / stride + 1;
  int out_width = (INPUT_MATRIX -> width + 2 * padding - filter_width) / stride + 1;
  int out_depth = filter_density;

  Set_DeviceMatrix(out_height, out_width, out_depth, HEAD_OUT,
                   "Output is allocated in device memory"); 

  // 1st 3 layers: Conv2d 1x1: BN: Swish()
  Conv2d_Layer(INPUT_MATRIX,  F_HEAD, HEAD_OUT,
              stride, padding,
              INPUT_MATRIX -> depth, out_depth, filter_density,
              Conv2d_1_x_1, NO_ACTIVATION,
              0, NULL);
 
  BN_ALL_PRE_DEFINED(HEAD_OUT, SWISH_ACTIVATION, 
                      &D_HEAD_BN_MEAN,	&D_HEAD_BN_VARIANCE ,
                      &D_HEAD_BN_WEIGHTS, &D_HEAD_BN_BIAS);


  // 4th layer: Average pooling layer which is just a reduction sum layer
  // Get mean values for all channels; Dims(1 x 1 x InputDepth)
  
  Matrix MEAN, Result_Mean;

  Set_DeviceMatrix(HEAD_OUT -> depth,
                    (int)ceil((double)HEAD_OUT -> height * HEAD_OUT -> width / (2 * BLOCK_SIZE)),
                    1, 
                    &Result_Mean, 
                    "Reesult Mean matrix allocated in device memory");

  REDUCTION_SUM(HEAD_OUT, &MEAN, &Result_Mean);


  // 5th layer: Fully connected layer::

  // Set Output matrix details
  Matrix Out1;
  Set_DeviceMatrix(1, 1000, 1, &Out1, "Setting Final Model Output matrix in device memory");
     
  Conv_vidMultiplier(&Out1, FC_WEIGHTS, &Result_Mean,
                      1, 1000, 1,
                      Conv2d_1_x_1, 1,
                      NO_ACTIVATION, 
                      0, NULL);
  
  stop("Model: ", 0);
  
  Matrix tmp_out_host;
  set_allocate_Host(&tmp_out_host, 1, 1000, 1);
  just_copy_DTH(&tmp_out_host, &Out1, "Copying to add bias");
 
  for (int i = 0; i < 1000; i++)
  {
    tmp_out_host.elements[i] += Head_linear_bias[i];
  }

  just_copy_HTD(&Out1, &tmp_out_host, "Copying to add bias");
  show_me_enhanced_from_devince(&Out1, "Model final output::");
}

// The first layer in efficient net: 
// It reutnrs a pointer to matrix, its elements are allocated in device memory 
void STEM_LAYER(Matrix *DInput_Mat, Matrix *F_STEM,
                  int image_height, int image_width, int image_depth,
                  int filter_height, int filter_width, int filter_depth, int filter_density,
                  int padding, int stride,
                  Matrix *STEM_OUT)
{

  // Calculate output dimensions       
  int out_height = (image_height + 2 * padding - filter_height) / stride + 1;
  int out_width = (image_width + 2 * padding - filter_width) / stride + 1;
  int out_depth = filter_density;
 

  // Allow the output from this layer to go accross the next layer       
  Set_DeviceMatrix(out_height, out_width, out_depth, STEM_OUT,
                   "Output is allocated in device memory"); 
 
  Conv2d_Layer(DInput_Mat,  F_STEM, STEM_OUT,
              stride, padding,
              image_depth, out_depth, filter_density,
              Regular_Conv, NO_ACTIVATION,
              0, NULL);
 

  BN_ALL_PRE_DEFINED(STEM_OUT, SWISH_ACTIVATION, 
                      &D_STEM_BN_MEAN, &D_STEM_BN_VARIANCE ,
                      &D_STEM_BN_WEIGHTS, &D_STEM_BN_BIAS);  
}