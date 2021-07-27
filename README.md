# CNN-on-GPU-using-CUDA-C

The current CUDA model is implementing EfficientNet-B0 and uses pretrained weights. CUDA code implemented for the model has 10 msec for the whole model detection timing, which is triple the speed of the python code implemented on GPU. The results are shown for the whole model and each layer inside the EfficientNet model