These are the .cu files for that were used to implement the model. However due to lack of a hardware GPU to practice on, the code right now works only on Google Colab. \par
Feel free to use these .cu files in a visual studio CUDA template. \par
The "EfficientNet" is the script you need to run; i.e. in Colab, \par
!nvcc -o /content/src/EfficientNet /content/src/APP.cu /content/src/KERNELS.cu /content/src/FUNCTIONS.cu --use_fast_math
!/content/src/EfficientNet

using
