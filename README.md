# CNN-on-GPU-using-CUDA-C

The current CUDA model is implementing EfficientNet-B0 and uses pretrained weights which you can find and use in this link:
https://drive.google.com/drive/folders/1v2o_xorzFvwhRLoJSwsNfcuU9ytGE_aJ?usp=sharing

CUDA code implemented for the model has 8 msec for the whole model detection timing, which is 4 times the speed of the python code implemented on GPU. The results are shown for the whole model and each layer inside the EfficientNet model.


![image](https://user-images.githubusercontent.com/20490432/127180264-6928bbb2-de7b-477f-9cda-b5e273da9c81.png)


The code implements kxk Conv, DW Conv, BatchNorm and Squeeze-Excitation using parallel algorithms like GEMM and reduction algorithms. All algorithms in this code are implemented from scratch. Concerning the optimization took into consideration: Shared memory traffic, Tiling, thread granuality. 

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

![image](https://drive.google.com/uc?export=view&id=1Nymz532z4F1H511pbAH4StuK_O5snHJd)

Note:      
1. Any Code modifications or refactoring done and resulted in enhancement in the execution time, the above table will be updated and a new version will be available.
2. Code runs on Colab mainly, scaling to real GPU is in progress

GPU details used on colab for testing the model:
Device Number: 0
  Device name: Tesla T4                          
  Memory Clock Rate (KHz): 5001000                          
  Memory Bus Width (bits): 256                          
  Number of totalGlobalMem 15843721216                          
  Number of sharedMemPerBlock 49152                          
                                                    
  Number of warpSize 32                          
  Number of maxThreadsPerBlock 1024                          
  Number of maxBlocksPerMultiProcessor 16                          
  Number of multiProcessorCount 40                          
  Number of maxThreadsPerMultiProcessor 1024                          
                          
  Number of maxThreadsDim 1024                          
  Number of maxGridSize 2147483647                          
  Number of totalConstMem 65536                          
  Number of sharedMemPerMultiprocessor 65536                          
  Peak Memory Bandwidth (GB/s): 320.064000                          
  asyncEngineCount: 3                          
                                                                                                          
Python model used to compare results:                
https://www.kaggle.com/hmendonca/efficientnet-cifar-10-ignite

References:

[1] Tan, Mingxing, and Quoc V. Le. “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.” ArXiv:1905.11946 [Cs, Stat], Sept. 2020. arXiv.org.      
[2] Sandler, Mark, et al. “MobileNetV2: Inverted Residuals and Linear Bottlenecks.” ArXiv:1801.04381 [Cs], Mar. 2019. arXiv.org.        
[3] Hu, Jie, et al. “Squeeze-and-Excitation Networks.” ArXiv:1709.01507 [Cs], May 2019. arXiv.org.        
[4] CUDA C++ Programming Guide. http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html.              
[5] Cook, S., 2012. CUDA programming: a developer's guide to parallel computing with GPUs. Newnes.                       
[6] Udacity introduction to parallel programming https://youtube.com/playlist?list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2                         
[7] Anderson, A., Vasudevan, A., Keane, C. and Gregg, D., 2017. Low-memory gemm-based convolution algorithms for deep neural networks. arXiv preprint arXiv:1709.03395.        
[8] Chellapilla, K., Puri,S., & Simard,P.(2006).High performance convolutional neural networks for document processing.
	< https://hal.archives-ouvertes.fr/inria-00112631/document >.                                            
[9] Warden, Pete. "Why GEMM Is at the Heart of Deep Learning". Pete Warden's Blog, April 20, 2015, https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/.       
[10] Kirk, D.B. and Wen-Mei, W.H., 2016. Programming massively parallel processors: a hands-on approach. Morgan kaufmann.             
