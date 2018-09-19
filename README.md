# Optimal Ternary Weights Approximation

Caffe implementation of Optimal-Ternary-Weights-Approximation in "Two-Step Quantization for Low-bit Neural Networks" (CVPR2018).

### Objective Function
![equation](http://latex.codecogs.com/gif.latex?\min_{\alpha,\hat{w}}||w-\alpha\hat{w}||_2^2)

where ![equation](http://latex.codecogs.com/gif.latex?\alpha>0) and ![equation](http://latex.codecogs.com/gif.latex?\hat{w}\in[-1,0,+1\]^m).

### Weight Blob
We use a temporary memory block to store ![equation](http://latex.codecogs.com/gif.latex?\alpha\hat{w}) and keep ![equation](http://latex.codecogs.com/gif.latex?w) in the **this->blobs_[0]**. 
During the backwardpropagation, ![equation](http://latex.codecogs.com/gif.latex?w) was used in the gradient accumulation and ![equation](http://latex.codecogs.com/gif.latex?\alpha\hat{w}) was used in the calculation of bottom gradients.

### How to use ?
change **type: "Convolution"** into **type: "TernaryConvolution"**, e.g.

```prototxt
layer {
    bottom: "pool1"
    top: "res2a_branch1"
    name: "res2a_branch1"
    type: "TernaryConvolution"
    convolution_param {
        num_output: 64
        kernel_size: 1
        pad: 0
        stride: 1
        weight_filler {
            type: "msra"
        }
        bias_term: false
    }
}
```
So far, GPU only.

### 2-bit Activation Quantization
Please refer to [wps712](https://github.com/wps712/Two-Step-Quantization-AlexNet).
