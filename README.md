# CNN-on-GPU-using-CUDA-C

The current CUDA model is implementing EfficientNet-B0 and uses pretrained weights. CUDA code implemented for the model has 10 msec for the whole model detection timing, which is triple the speed of the python code implemented on GPU. The results are shown for the whole model and each layer inside the EfficientNet model.


![image](https://user-images.githubusercontent.com/20490432/127180264-6928bbb2-de7b-477f-9cda-b5e273da9c81.png)


The code implements kxk Conv, DW Conv, BatchNorm and Squeeze-Excitation using parallel algorithms like GEMM and reduction algorithms. All algorithms in this code are implemented from scratch.

The main building block for EfficientNet-B0 is MBConv layer, which contains the following sequence of operations:

	MBConv(
		(expansion_conv): 
		Sequential(
			(0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
			(1): BatchNorm2d(144, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
			(2): Swish()
					)
					
		(depthwise_conv): 
		Sequential(
			(0): Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144, bias=False)
			(1): BatchNorm2d(144, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
			(2): Swish()
					)
     
		(squeeze_excitation): SqueezeExcitation((reduce_expand): 
		Sequential(
			(0): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
			(1): Swish()
			(2): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
			(3): Sigmoid()
					)
						)
						
		(project_conv): 
		Sequential(
			(0): Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
			(1): BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
					)
					
			)
   
   
The results for the latest code edition: 


![image](https://drive.google.com/uc?export=view&id=1iHwm-wSoOsgkrVpXsEOD6N-hwQCUJnja)

Note: Any Code modifications or refactoring done and resulted in enhancement in the execution time, the above table will be updated and a new version will be available.
