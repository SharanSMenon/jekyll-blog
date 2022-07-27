---
title: "Implement EfficientNet in PyTorch"
date: 2022-02-05
toc: true
toc_label: "Contents"
toc_sticky: True
published: true
excerpt: "Implement EfficientNet in PyTorch from Scratch. Based on the [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf) paper. "
categories:
  - Programming
  - Machine Learning
tags:
  - Python
  - PyTorch
  - Computer Vision
  - Implementations
---

In this article, we will be implementing EfficientNet in PyTorch from scratch.

> This tutorial focuses on the implementation of EfficientNet in PyTorch. There is no training code presented here.

## Libraries

The only library necessary is PyTorch, not even `torchvision`.

If you want to view a keras-like summary of the model, you can install the `torchinfo` library with `pip install torchinfo`.

```python
import torch
from torch import nn, optim
import math
import os
from torchinfo import summary # this is an optional import
```

## Blocks

We need to define the different blocks of the model. We can start by making a basic `conv-bn-silu` block. 

```python
def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1, bias=False, bn=True, act=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, 
                  padding=padding, groups=groups, bias=bias),
        nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
        nn.SiLU() if act else nn.Identity()
    ]
    return nn.Sequential(*layers)
```

This is a highly flexible convolutional block. It will make implementing the more complex blocks easier. 

### Squeeze and Exicitation

EfficientNet uses a Squeeze and Exicitation block. It is simple to implement in PyTorch

```python
class SEBlock(nn.Module):
    def __init__(self, c, r=24):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveMaxPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(c, c // r, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(c // r, c, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        s = self.squeeze(x)
        e = self.excitation(s)
        return x * e
```

### MB Conv

The paper also says to use a Inverted Residual block (this block is also used in MobileNet v2).  It can also be implemented in PyTorch

```python
class MBConv(nn.Module):
    def __init__(self, n_in, n_out, expansion, kernel_size=3, stride=1, r=24, dropout=0.1):
        super(MBConv, self).__init__()
        self.skip_connection = (n_in == n_out) and (stride == 1)
        padding = (kernel_size-1)//2
        expanded = expansion*n_in
        
        self.expand_pw = nn.Identity() if expansion == 1 else conv_block(n_in, expanded, kernel_size=1)
        self.depthwise = conv_block(expanded, expanded, kernel_size=kernel_size, 
                                    stride=stride, padding=padding, groups=expanded)
        self.se = SEBlock(expanded, r=r)
        self.reduce_pw = conv_block(expanded, n_out, kernel_size=1, act=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.expand_pw(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.reduce_pw(x)
        if self.skip_connection:
            x = self.dropout(x)
            x = x + residual
        return x
```

The above is a generic MBConv block with any expansion factor.

The paper uses two types of MBConv blocks, one with an `expansion_factor` of 1 and a second with an `expansion_factor` of 6, MBConv1 and MBConv6 respectively.

```python
def mbconv1(n_in, n_out, kernel_size=3, stride=1, r=24, dropout=0.1):
    return MBConv(n_in, n_out, 1, kernel_size=kernel_size, stride=stride, r=r, dropout=dropout)
def mbconv6(n_in, n_out, kernel_size=3, stride=1, r=24, dropout=0.1):
    return MBConv(n_in, n_out, 6, kernel_size=kernel_size, stride=stride, r=r, dropout=dropout)
```

## EfficientNet

The following table shows the structure for the base EfficientNet, or `efficientnet-b0`.

| Stage (i) | Block     | Resolution | Channels | Layers |
|-----------|-----------|------------|----------|--------|
| 1         | `mbconv1` | 224 x 224  | 32       | 1      |
| 2         | `mbconv6` | 112 x 112  | 16       | 1      |
| 3         | `mbconv6` | 112 x 112  | 24       | 2      |
| 4         | `mbconv6` | 56 x 56    | 40       | 2      |
| 5         | `mbconv6` | 28 x 28    | 80       | 3      |
| 6         | `mbconv6` | 14 x 14    | 112      | 3      |
| 7         | `mbconv6` | 14 x 14    | 192      | 4      |
| 8         | `mbconv6` | 7 x 7      | 320      | 1      |
| 9         | `mbconv6` | 7 x 7      | 1080     | 1      |

We can define these in python.

```python
widths = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
depths = [1, 2, 2, 3, 3, 4, 1]
kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
strides = [1, 2, 2, 2, 1, 2, 1]
ps = [0, 0.029, 0.057, 0.086, 0.114, 0.143, 0.171]
```

The strides and kernel sizes were obtained from the paper, not listed in the table above.

### Model Scaling 

The other models `b1-b7` are created by scaling up this architecture. We can define the scaling functions as the following


```python
def scale_width(w, w_factor):
    w *= w_factor
    new_w = (int(w+4) // 8) * 8
    new_w = max(8, new_w)
    if new_w < 0.9*w:
        new_w += 8
    return int(new_w)

def efficientnet_scaler(w_factor=1, d_factor=1):
    scaled_widths = [scale_width(w, w_factor) for w in widths]
    scaled_depths = [math.ceil(d_factor*d) for d in depths]
    return scaled_widths, scaled_depths
```

`efficientnet_scaler` generates the widths and depths for the other models. The models will be constructed with the output of `efficientnet_scaler`.

Each stage in the Efficient Net model has a certain number of `mbconv` blocks, we can abstract the creation of each stage into a function.

```python
def create_stage(n_in, n_out, num_layers, layer=mbconv6, 
                 kernel_size=3, stride=1, r=24, ps=0):
    layers = [layer(n_in, n_out, kernel_size=kernel_size,
                       stride=stride, r=r, dropout=ps)]
    layers += [layer(n_out, n_out, kernel_size=kernel_size,
                        r=r, dropout=ps) for _ in range(num_layers-1)]
    return nn.Sequential(*layers)
```

Now, we can define a generic EfficientNet class. This class will take a scaling factor for width and depth, and generate a model based on the result from `efficientnet_scaler`.

```python
class EfficientNet(nn.Module):
    def __init__(self, w_factor=1, d_factor=1, n_classes=1000):
        super(EfficientNet, self).__init__()
        scaled_widths, scaled_depths = efficientnet_scaler(w_factor=w_factor, d_factor=d_factor)
        
        self.stem = conv_block(3, scaled_widths[0], stride=2, padding=1)
        stages = [
            create_stage(scaled_widths[i], scaled_widths[i+1], scaled_depths[i], layer= mbconv1 if i==0 else mbconv6, 
                         kernel_size=kernel_sizes[i], stride=strides[i], r= 4 if i==0 else 24, ps=ps[i]) for i in range(7)
        ]
        self.stages = nn.Sequential(*stages)
        self.pre = conv_block(scaled_widths[-2], scaled_widths[-1], kernel_size=1)
        self.pool_flatten = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(
            nn.Linear(scaled_widths[-1], n_classes)
        )
            
    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pre(x)
        x = self.pool_flatten(x)
        x = self.head(x)
        return x
```

> If you wanted, you could reduce this whole class to a single nn.Sequential model, since the forward method is basically a sequential model.

With the generic EfficientNet implemented, we can now create the models specified in the paper.

### Creating B1 - B7

```python
def efficientnet_b0(n_classes=1000):
    return EfficientNet(n_classes=n_classes)
def efficientnet_b1(n_classes=1000):
    return EfficientNet(1, 1.1, n_classes=n_classes)
def efficientnet_b2(n_classes=1000):
    return EfficientNet(1.1, 1.2, n_classes=n_classes)
def efficientnet_b3(n_classes=1000):
    return EfficientNet(1.2, 1.4, n_classes=n_classes)
def efficientnet_b4(n_classes=1000):
    return EfficientNet(1.4, 1.8, n_classes=n_classes)
def efficientnet_b5(n_classes=1000):
    return EfficientNet(1.6, 2.2, n_classes=n_classes)
def efficientnet_b6(n_classes=1000):
    return EfficientNet(1.8, 2.6, n_classes=n_classes)
def efficientnet_b7(n_classes=1000):
    return EfficientNet(2, 3.1, n_classes=n_classes)
```

To initialize a model, just use the following code

```python
b0 = efficientnet_b0()
```

It is even possible to customize the number of classes. The default number of classes is 1000.

```python
b0 = efficientnet_b0(n_classes=542) # for a dataset that has 542 classes
```

This is a simple implementation of EfficientNet, and that's it for this article.

## Bonus (EfficientNet Sequential)

As a bonus, here is the sequential version of the `EfficientNet` class.

```python
def EfficientNetSequential( w_factor=1, d_factor=1, n_classes=1000):
    scaled_widths, scaled_depths = efficientnet_scaler(w_factor=w_factor, d_factor=d_factor)
    layers = [
        conv_block(3, scaled_widths[0], stride=2, padding=1)
    ]
    stages = [
            create_stage(scaled_widths[i], scaled_widths[i+1], scaled_depths[i], layer= mbconv1 if i==0 else mbconv6, 
                         kernel_size=kernel_sizes[i], stride=strides[i], r= 4 if i==0 else 24, ps=ps[i]) for i in range(7)
    ]
    layers = layers + stages
    layers.append(conv_block(scaled_widths[-2], scaled_widths[-1], kernel_size=1))
    layers.append(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()))
    layers.append(nn.Sequential(nn.Linear(scaled_widths[-1], n_classes)))
    return nn.Sequential(*layers)
```

> The paper can be found [here](https://arxiv.org/abs/1905.11946).