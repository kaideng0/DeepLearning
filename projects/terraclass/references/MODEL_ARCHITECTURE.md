# TerraClass model architecture and parameter counting

This reference explains the models defined in `src/terraclass/models.py`, with an emphasis on the custom `SimpleCNN`. It shows how data shapes change, which layers learn values, how to calculate those values by hand, and what PyTorch must estimate during training.

## Three meanings of “parameter”

The word parameter is used in several related ways:

1. A function argument, such as `num_classes=10`.
2. A hyperparameter chosen by the developer, such as `kernel_size=3`, 32 output channels, or a learning rate of `3e-4`.
3. A learnable model parameter, such as a convolution weight or Linear bias. These are the tensors for which training calculates gradients and updates values.

This document uses **parameter count** to mean the third kind unless stated otherwise.

## Core vocabulary

| Concept | Definition |
| --- | --- |
| Sample | One input example, such as one satellite image and its class label. |
| Batch | A group of samples processed together before one optimizer update. |
| Tensor | A multidimensional array with a shape, data type, and device. |
| Input value | Data supplied to the model, such as an RGB pixel value. It is not learned. |
| Channel | One plane of values. An RGB image has three input channels; convolution layers create feature channels. |
| Feature map | The spatial output produced by one convolution filter across an image. |
| Activation | An intermediate value calculated during the forward pass. Activations are not model parameters. |
| Weight | A learned multiplier used by a Conv2d or Linear layer. |
| Bias | A learned offset added after a weighted sum. |
| Kernel | The small spatial grid of weights used by a convolution or the window size used by pooling. A pooling kernel contains no learned weights. |
| Filter | In Conv2d, the complete set of kernels spanning all input channels that produces one output channel. “Kernel” and “filter” are often used informally as synonyms. |
| Hyperparameter | A setting selected before training, such as kernel size, channel count, batch size, or learning rate. |
| Logit | A model's raw, unnormalized score for one class. |
| Loss | A scalar measurement of prediction error that training tries to minimize. |
| Gradient | The derivative of the loss with respect to a trainable parameter. It indicates how a small parameter change affects loss. |
| Optimizer | An algorithm such as AdamW that uses gradients to update trainable parameters. |
| Epoch | One complete pass through the training dataset. |
| Training step | One forward pass, backward pass, and parameter update for one batch. |

Input values, activations, parameters, and hyperparameters play different roles. For a `64 x 64` RGB image:

```text
input values = 3 x 64 x 64 = 12,288 values per image
```

Those values change from image to image. Model parameters are shared learned values that remain part of the model and are applied to every image.

## SimpleCNN overview

The custom model receives a batch of RGB images:

```text
[batch_size, 3, 64, 64]
```

The dimensions represent:

```text
[number of images, color channels, height, width]
```

The high-level flow is:

```text
RGB image
  -> convolution block: 3 to 32 channels
  -> convolution block: 32 to 64 channels
  -> convolution block: 64 to 128 channels
  -> global average pooling
  -> dropout
  -> Linear classifier
  -> 10 class logits
```

For EuroSAT, `num_classes=10`, so the final output has shape:

```text
[batch_size, 10]
```

Each of the 10 output values is a logit, or unnormalized class score. The model does not apply Softmax because `CrossEntropyLoss` expects raw logits.

## PyTorch model-building concepts

### `nn.Module`

`nn.Module` is the base class for PyTorch models and layers. Subclassing it allows PyTorch to discover nested layers, move them between devices, switch training/evaluation behavior, and save their state.

### `super().__init__()`

Calling the parent `nn.Module` initializer sets up this tracking before layers are assigned to the model. Without it, PyTorch cannot reliably register the model's parameters and child modules.

### `nn.Sequential`

`nn.Sequential` sends a tensor through its child layers in declaration order:

```text
input -> layer 1 -> layer 2 -> layer 3 -> output
```

TerraClass uses one Sequential container for feature extraction and another for classification.

### `forward()`

`forward()` defines the computation performed for one model call. Calling `model(images)` invokes `forward()` through PyTorch's module machinery.

### `train()` and `eval()`

`model.train()` enables training behavior for layers such as BatchNorm and Dropout. `model.eval()` enables inference behavior. These methods do not enable or disable gradient calculation by themselves.

### `requires_grad`

A parameter with `requires_grad=True` participates in autograd and receives a gradient after `loss.backward()`. Frozen parameters have `requires_grad=False` and are not updated by the optimizer.

### `state_dict()`

A state dictionary maps names to parameter and buffer tensors. TerraClass checkpoints save it so learned weights and BatchNorm running statistics can be restored.

## Layer-by-layer shapes and parameter counts

The table assumes an input image size of `64 x 64` and uses `B` for batch size.

| Layer | Configuration | Output shape | Trainable parameters |
| --- | --- | --- | ---: |
| Input | RGB images | `[B, 3, 64, 64]` | 0 |
| Conv2d | `3 -> 32`, kernel `3 x 3`, padding 1 | `[B, 32, 64, 64]` | 896 |
| BatchNorm2d | 32 channels | `[B, 32, 64, 64]` | 64 |
| ReLU | Elementwise activation | `[B, 32, 64, 64]` | 0 |
| MaxPool2d | kernel 2, stride 2 | `[B, 32, 32, 32]` | 0 |
| Conv2d | `32 -> 64`, kernel `3 x 3`, padding 1 | `[B, 64, 32, 32]` | 18,496 |
| BatchNorm2d | 64 channels | `[B, 64, 32, 32]` | 128 |
| ReLU | Elementwise activation | `[B, 64, 32, 32]` | 0 |
| MaxPool2d | kernel 2, stride 2 | `[B, 64, 16, 16]` | 0 |
| Conv2d | `64 -> 128`, kernel `3 x 3`, padding 1 | `[B, 128, 16, 16]` | 73,856 |
| BatchNorm2d | 128 channels | `[B, 128, 16, 16]` | 256 |
| ReLU | Elementwise activation | `[B, 128, 16, 16]` | 0 |
| MaxPool2d | kernel 2, stride 2 | `[B, 128, 8, 8]` | 0 |
| AdaptiveAvgPool2d | output `1 x 1` | `[B, 128, 1, 1]` | 0 |
| Flatten | Preserve batch dimension | `[B, 128]` | 0 |
| Dropout | `p=0.3` | `[B, 128]` | 0 |
| Linear | `128 -> 10` | `[B, 10]` | 1,290 |
| **Total** | | | **94,986** |

## Convolution parameter formula

For a standard Conv2d layer with bias enabled:

```text
weight count = output_channels x input_channels x kernel_height x kernel_width
bias count   = output_channels

total = weight count + bias count
```

PyTorch stores the weight tensor with shape:

```text
[output_channels, input_channels, kernel_height, kernel_width]
```

### What a `3 x 3` kernel means

A spatial `3 x 3` kernel examines a neighborhood of nine positions. Because an RGB convolution must examine red, green, and blue together, one filter in the first layer contains three `3 x 3` kernel slices:

```text
red slice:    3 x 3 = 9 weights
green slice:  3 x 3 = 9 weights
blue slice:   3 x 3 = 9 weights
                       ----------
one filter:            27 weights
```

At one image location, the layer multiplies 27 input values by these 27 weights, adds the results, and adds one bias. That produces one value in one output feature map.

### Weight sharing

The same filter slides across every spatial location. It does not learn a separate set of weights for every pixel:

```text
patch at position (0, 0) x the filter's 27 weights
patch at position (0, 1) x the same 27 weights
patch at position (0, 2) x the same 27 weights
...
```

With padding, one filter produces `64 x 64 = 4,096` activations from a `64 x 64` image, but it still owns only 27 weights and one bias. The spatial positions increase computation and activation memory, not parameter count.

Weight sharing allows one learned pattern detector—for example, an edge detector—to recognize that pattern anywhere in the image. Without sharing, the first convolution would require millions of location-specific parameters instead of 896.

### First convolution: 3 to 32 channels

```python
nn.Conv2d(3, 32, kernel_size=3, padding=1)
```

The weight tensor has shape `[32, 3, 3, 3]`:

```text
weights = 32 x 3 x 3 x 3 = 864
biases  = 32
total   = 864 + 32 = 896
```

The three input channels are red, green, and blue. The layer learns 32 separate filters, and every filter examines all three input channels.

### Second convolution: 32 to 64 channels

```python
nn.Conv2d(32, 64, kernel_size=3, padding=1)
```

```text
weights = 64 x 32 x 3 x 3 = 18,432
biases  = 64
total   = 18,432 + 64 = 18,496
```

### Third convolution: 64 to 128 channels

```python
nn.Conv2d(64, 128, kernel_size=3, padding=1)
```

```text
weights = 128 x 64 x 3 x 3 = 73,728
biases  = 128
total   = 73,728 + 128 = 73,856
```

The parameter count grows quickly when channel counts increase because every output channel has a kernel for every input channel.

### Why padding preserves image size

For one spatial dimension, the convolution output size is:

```text
output = floor((input + 2 x padding - kernel_size) / stride) + 1
```

For these convolutions:

```text
output = ((64 + 2 x 1 - 3) / 1) + 1 = 64
```

Therefore, `kernel_size=3`, `padding=1`, and the default `stride=1` preserve height and width. MaxPool then halves them.

Padding is a fixed operation, not a learned parameter. `padding=1` places a one-value-wide zero border around the feature map. Without padding, a `3 x 3` convolution would shrink `64 x 64` to `62 x 62`.

## MaxPool2d: downsampling strong responses

The model uses:

```python
nn.MaxPool2d(2)
```

This is equivalent to:

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

The `kernel_size=2` setting means that pooling examines a `2 x 2` window. `stride=2` means the window moves two positions after each result, so adjacent windows do not overlap.

For example:

```text
input                 2 x 2 window maxima

1  3  2  1
4  2  0  5       ->       4  5
2  1  3  2                6  4
0  6  1  4
```

MaxPool operates independently on every channel. It preserves batch size and channel count while halving height and width:

```text
[B, 32, 64, 64] -> [B, 32, 32, 32]
[B, 64, 32, 32] -> [B, 64, 16, 16]
[B, 128, 16, 16] -> [B, 128, 8, 8]
```

It has no trainable parameters. Its fixed rule is to keep the maximum value in each window. During backpropagation, the gradient for a window flows through the position that supplied the maximum.

Pooling reduces computation and activation memory, increases the effective receptive field of later layers, and makes exact feature position somewhat less important. Its tradeoff is that it discards spatial detail.

## BatchNorm parameter formula

`BatchNorm2d(C)` normalizes each of `C` feature channels independently while preserving tensor shape. During training, it calculates a channel's mean and variance across batch and spatial dimensions:

```text
statistics for each channel are calculated over [B, height, width]
```

It first standardizes values:

```text
normalized = (x - batch_mean) / sqrt(batch_variance + epsilon)
```

It then applies a learned affine transformation:

```text
output = gamma * normalized + beta
```

`epsilon` is a small fixed value that prevents division by zero.

### Why gamma and beta are learned

Normalization provides a stable scale but would otherwise force every channel to remain approximately zero-centered with unit variance. That could restrict the representations needed by later layers.

- Gamma controls the strength and direction of a channel.
- Beta controls the channel's offset.
- Before ReLU, beta also affects how many values fall above or below ReLU's zero threshold.

Gamma starts at one and beta starts at zero, so BatchNorm initially performs ordinary normalization. Backpropagation changes them only when another scale or offset reduces the final classification loss.

For one channel:

```text
dLoss/dGamma = sum(dLoss/dOutput * normalized)
dLoss/dBeta  = sum(dLoss/dOutput)
```

The sums cover the batch and spatial positions for that channel. In the training loop, `loss.backward()` calculates these gradients and `optimizer.step()` updates gamma and beta just like convolution weights and biases.

In PyTorch:

```python
batch_norm.weight  # gamma, an nn.Parameter
batch_norm.bias    # beta, an nn.Parameter
```

Gamma and beta can be disabled with `BatchNorm2d(C, affine=False)`, but the network then loses the ability to choose a useful post-normalization scale and offset.

For `BatchNorm2d(C)` with its default affine transformation, every channel learns:

- one scale value, gamma;
- one shift value, beta.

The trainable parameter count is therefore:

```text
trainable parameters = 2 x C
```

For the three BatchNorm layers:

```text
BatchNorm2d(32):  2 x 32  = 64
BatchNorm2d(64):  2 x 64  = 128
BatchNorm2d(128): 2 x 128 = 256

total BatchNorm trainable parameters = 448
```

BatchNorm also keeps a running mean, running variance, and batch counter. These are **buffers**, not trainable parameters:

```text
buffer values per BatchNorm layer = C running means + C running variances + 1 counter
```

The model contains:

```text
(32 + 32 + 1) + (64 + 64 + 1) + (128 + 128 + 1) = 451 buffer values
```

Buffers are saved in `state_dict()`, but `loss.backward()` does not calculate gradients for them and the optimizer does not update them. BatchNorm running statistics are updated by the layer during training.

During `model.train()`, BatchNorm uses current-batch statistics and updates its running mean and variance. During `model.eval()`, it uses the stored running statistics so predictions do not depend on the composition of the current batch.

## AdaptiveAvgPool2d: global spatial summary

The classifier begins with:

```python
nn.AdaptiveAvgPool2d(1)
```

The `1` specifies a target spatial size of `1 x 1`; it is not a kernel size. PyTorch automatically selects averaging regions that produce that target:

```text
[B, 128, 8, 8] -> [B, 128, 1, 1]
```

For the current `8 x 8` input, it averages all 64 values in each channel. It performs this independently for all 128 channels:

```text
one output value = sum of one channel's 8 x 8 values / 64
```

The operation is adaptive because it always produces the requested output size even if the input size changes:

```text
8 x 8   -> 1 x 1
16 x 16 -> 1 x 1
32 x 32 -> 1 x 1
```

It has no trainable parameters. It summarizes whether a learned feature is present across the image while discarding its exact spatial location.

Without this pooling, flattening `[B, 128, 8, 8]` would give 8,192 classifier inputs and require `8,192 x 10 + 10 = 81,930` Linear parameters. Adaptive pooling reduces the classifier input to 128 values, so the Linear layer needs only 1,290 parameters.

## Linear parameter formula

For a Linear layer:

```text
weight count = input_features x output_features
bias count   = output_features

total = weight count + bias count
```

The final classifier is:

```python
nn.Linear(128, 10)
```

Therefore:

```text
weights = 128 x 10 = 1,280
biases  = 10
total   = 1,280 + 10 = 1,290
```

Changing `num_classes` changes only this layer. In general:

```text
classifier parameters = 128 x num_classes + num_classes
                      = 129 x num_classes
```

## Layers with no learnable parameters

These layers change values or shapes but do not contain weights or biases:

- `ReLU` replaces negative values with zero.
- `MaxPool2d` selects a maximum from each local window.
- `AdaptiveAvgPool2d` averages each feature map to a requested output size.
- `Flatten` changes tensor shape.
- `Dropout` randomly masks activations during training.

### ReLU

ReLU applies a fixed nonlinear function element by element:

```text
ReLU(x) = max(0, x)
```

Negative values become zero and positive values remain. Nonlinearity allows stacked layers to represent relationships that cannot be reduced to one linear transformation.

### Flatten

Flatten preserves the batch dimension while combining the remaining dimensions:

```text
[B, 128, 1, 1] -> [B, 128]
```

It changes only the tensor view used by the Linear layer; it does not learn values.

### Dropout

`Dropout(p=0.3)` randomly masks 30% of its input activations during training. This discourages the network from depending too heavily on particular features and can reduce overfitting. PyTorch scales the surviving activations during training so their expected magnitude stays consistent.

Dropout is random during `model.train()` and disabled during `model.eval()`. Its probability `p` is a hyperparameter, not a trainable parameter.

Although these layers have zero parameters, they still affect model behavior, compute cost, and activation memory.

## Adding the SimpleCNN total by hand

```text
first Conv2d       896
first BatchNorm     64
second Conv2d   18,496
second BatchNorm   128
third Conv2d    73,856
third BatchNorm    256
Linear            1,290
-----------------------
total            94,986
```

All 94,986 parameters have `requires_grad=True`, so all of them are estimated during ordinary SimpleCNN training.

## What “estimated during training” means

The model begins with initial parameter values. For each training batch:

```python
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = criterion(logits, targets)
loss.backward()
optimizer.step()
```

The steps have distinct responsibilities:

1. The forward pass uses the current 94,986 parameter values to produce logits.
2. The loss measures prediction error.
3. `loss.backward()` calculates a gradient for every trainable parameter.
4. `optimizer.step()` uses those gradients to estimate improved parameter values.

The number of parameters does not depend on:

- the number of training examples;
- batch size;
- the number of epochs;
- input height and width, for this architecture's adaptive pooling design.

Those choices change how often the parameters are used or updated, not how many parameters the model contains.

## Parameter count versus training memory

Parameter count is not the same as GPU memory usage. In float32, each value occupies four bytes.

The SimpleCNN parameter tensor storage is approximately:

```text
94,986 x 4 bytes = 379,944 bytes
                     about 0.38 MB
```

Training also stores:

- gradients, approximately one value per trainable parameter;
- AdamW's first-moment estimate, one value per parameter;
- AdamW's second-moment estimate, one value per parameter;
- intermediate activations needed for backpropagation;
- BatchNorm buffers and temporary operation workspaces.

Ignoring activations and temporary workspaces, float32 parameters, gradients, and two AdamW state tensors require roughly four copies:

```text
94,986 x 4 bytes x 4 copies = 1,519,776 bytes
                                about 1.52 MB
```

For this CNN, activation memory can be much larger than parameter memory. For example, the first convolution output for batch size 64 contains:

```text
64 x 32 x 64 x 64 = 8,388,608 float32 values
8,388,608 x 4 bytes = 33,554,432 bytes
                       about 32 MiB for that tensor alone
```

Autograd may retain multiple layer outputs for the backward pass. This is why reducing batch size usually reduces GPU memory even though it does not change parameter count.

## Verify counts with PyTorch

Count every model parameter:

```python
total_parameters = sum(parameter.numel() for parameter in model.parameters())
print(total_parameters)
```

Count only parameters that receive gradients:

```python
trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)
print(trainable_parameters)
```

Inspect every named parameter:

```python
for name, parameter in model.named_parameters():
    print(
        name,
        tuple(parameter.shape),
        parameter.numel(),
        parameter.requires_grad,
    )
```

For `SimpleCNN(num_classes=10)`, this produces:

| Parameter tensor | Shape | Values |
| --- | --- | ---: |
| `features.0.weight` | `[32, 3, 3, 3]` | 864 |
| `features.0.bias` | `[32]` | 32 |
| `features.1.weight` | `[32]` | 32 |
| `features.1.bias` | `[32]` | 32 |
| `features.4.weight` | `[64, 32, 3, 3]` | 18,432 |
| `features.4.bias` | `[64]` | 64 |
| `features.5.weight` | `[64]` | 64 |
| `features.5.bias` | `[64]` | 64 |
| `features.8.weight` | `[128, 64, 3, 3]` | 73,728 |
| `features.8.bias` | `[128]` | 128 |
| `features.9.weight` | `[128]` | 128 |
| `features.9.bias` | `[128]` | 128 |
| `classifier.3.weight` | `[10, 128]` | 1,280 |
| `classifier.3.bias` | `[10]` | 10 |
| **Total** | | **94,986** |

## SimpleCNN versus ResNet18

### Convolutional neural network

A convolutional neural network, or CNN, uses shared spatial filters to learn local patterns such as edges, textures, shapes, and increasingly complex feature combinations. `SimpleCNN` and ResNet18 are both CNN architectures.

### SimpleCNN

`SimpleCNN` is TerraClass's compact teaching model. It has three convolution blocks, begins with random parameter values, uses native `64 x 64` EuroSAT inputs, and contains 94,986 trainable parameters for 10 classes.

### ResNet18 and residual connections

ResNet18 is a deeper CNN built from residual blocks. A residual block learns a transformation `F(x)` and adds the original input through a shortcut:

```text
output = F(x) + x
```

The shortcut gives gradients a direct path through deep stacks of layers, making deeper networks easier to optimize. The “18” refers to the conventional count of weighted layers in this architecture.

### Transfer learning

Transfer learning starts from parameters learned on one task and adapts them to another. TerraClass loads ResNet18 parameters trained on ImageNet, replaces the original 1,000-class classifier with a 10-class EuroSAT classifier, and then trains on satellite images.

Pretraining changes parameter values, not architecture or parameter count. Early and middle features such as edges and textures can often be reused, so transfer learning usually needs less target data and training time than learning the same large model from random initialization.

ResNet18 is also a convolutional neural network, but it is deeper and uses residual connections. TerraClass replaces its original ImageNet classifier with:

```python
nn.Linear(512, 10)
```

The replacement classifier contains:

```text
512 x 10 + 10 = 5,130 parameters
```

Counts from the actual TerraClass model are:

| Model/training mode | Total parameters | Trainable parameters | Buffers |
| --- | ---: | ---: | ---: |
| SimpleCNN | 94,986 | 94,986 | 451 |
| ResNet18, full fine-tuning | 11,181,642 | 11,181,642 | 9,620 |
| ResNet18, frozen backbone | 11,181,642 | 5,130 | 9,620 |

ResNet18 has about 118 times as many total parameters as SimpleCNN. Loading pretrained weights changes the initial values, not the number of parameters.

### Full transfer-learning fine-tuning

This is the default behavior:

```powershell
terraclass train --model resnet18
```

ResNet18 loads ImageNet weights, replaces the classifier, and allows gradients for the entire network. All 11,181,642 parameters are trainable.

### Frozen feature extractor

This command freezes existing ResNet parameters before creating the new classifier:

```powershell
terraclass train --model resnet18 --freeze-backbone
```

The model still contains 11,181,642 parameters, but only the new classifier's 5,130 parameters have `requires_grad=True` and are passed to AdamW.

One nuance in the current implementation is that frozen BatchNorm weights do not receive gradients, but their running-statistic buffers can still update while the overall model is in training mode. Freezing parameters and switching a module to evaluation behavior are separate operations.

### ResNet18 without transfer learning

```powershell
terraclass train --model resnet18 --no-pretrained
```

The architecture and parameter count remain identical, but all weights start randomly. This is no longer transfer learning.

## Summary

- `SimpleCNN` contains 94,986 learned values for 10-class EuroSAT classification.
- Conv2d counts depend on input channels, output channels, and kernel area.
- BatchNorm learns two values per channel.
- Linear counts depend on input and output feature counts.
- Pooling, activation, flattening, and dropout layers have no learned values.
- Only parameters with `requires_grad=True` receive gradients and optimizer updates.
- Freezing a model reduces trainable parameter count without reducing total parameter count.
- Parameter storage is only one part of training memory; activations often dominate and scale with batch size and image size.
