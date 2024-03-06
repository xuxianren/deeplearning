
+ LeNet(1995)
  - 2卷积 + 池化
  - 2全连接
+ AlexNet(2012)
  - 更大更深(认知)
  - ReLu, Dropout, 数据增强
+ VGG
  - 块结构
  - 更大更深的AlexNet (重复的vgg块)


### 卷积和池化对size的影响

#### 简易计算
卷积和池化计算size的公式相同，不考虑空洞卷积的简易公式如下
$$
 W_{out} = \lfloor \frac {W_{in} + 2P - F} {S} + 1\rfloor 
$$
其中，W为输入的size，P为padding的size，F为核的size，S为stride的size。


#### 2d卷积的理解
把每个通道看成线性层的节点, 相比线性层多了卷积的步骤
每个输入通道和输出通道之间，都有1个卷积核
参数个数 c_in * c_out * (k_w * k_z + 1 + 1) 两个1分布是权重和偏置

#### torch.nn.Conv2d

**参数**
+ in_channels (int) – Number of channels in the input image
+ out_channels (int) – Number of channels produced by the convolution
+ kernel_size (int or tuple) – Size of the convolving kernel
+ stride (int or tuple, optional) – Stride of the convolution. Default: 1
+ padding (int, tuple or str, optional) – Padding added to all four sides of the input. Default: 0
+ padding_mode (str, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
+ dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
+ groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
+ bias (bool, optional) – If True, adds a learnable bias to the output. Default: True

$$
H_{out} = \lfloor \frac{H_{in} + 2 * padding[0] - dilation[0]*(kernel\_size[0]-1)-1}{stride[0]} + 1 \rfloor \\

W_{out} = \lfloor \frac{W_{in} + 2 * padding[1] - dilation[1]*(kernel\_size[1]-1)-1}{stride[1]} + 1 \rfloor \\
$$