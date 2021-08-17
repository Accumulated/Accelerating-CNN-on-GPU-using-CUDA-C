EfficientNet file is the final output from the nvcc command. 
So in google colab use the 2 following command in order to generate the file again and run it.
!nvcc -o /content/src/EfficientNet /content/src/APP.cu /content/src/KERNELS.cu /content/src/FUNCTIONS.cu --use_fast_math 
!/content/src/EfficientNet
