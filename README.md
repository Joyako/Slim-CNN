## Slim-CNN in PyTorch  
A [PyTorch](https://pytorch.org/) implementation of [Slim-CNN: A Light-Weight CNN for Face Attribute Prediction](https://arxiv.org/pdf/1907.02157.pdf)

### Micro-Architecture in CNN
SSE Block && Slim Module:  
![Slim Module](https://github.com/Joyako/Slim-CNN/blob/master/data/image.jpg)

#### Notes
In this Paper, it is not clear in the skip connection that the number of channels int the input layer is not equal to the number of channels output after the first use of SSE block.
Thus, I add a point-wise convolution layer to change the number of channels. 