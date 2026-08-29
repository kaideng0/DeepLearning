# TerraClass learning guide

This guide explains the project in the order data flows through it. Keep the source files open and run small experiments as you read.

If PyTorch is new to you, begin with `notebooks/00_pytorch_basics.ipynb`. It introduces tensors, CUDA devices, autograd, neural-network modules, data loaders, and a complete small training loop before this guide applies those concepts to TerraClass.

## 1. Start with the data

Open `src/terraclass/data.py`.

A PyTorch `Dataset` answers two questions: how many examples exist, and what image/label pair should be returned for an index? TorchVision's `EuroSAT` already provides that behavior. `TransformSubset` adds two ideas:

1. it exposes only the indices assigned to a particular split;
2. it applies augmentation only to training examples.

This separation matters. Validation and test images must use deterministic preprocessing, otherwise their scores change with random crops and flips.

`DataLoader` groups samples into batches. Shuffling the training loader changes the order seen by the optimizer, while validation and test loaders preserve a stable order.

Try changing `--batch-size`. A larger batch can improve accelerator utilization but consumes more memory and performs fewer optimizer updates per epoch.

## 2. Understand an input tensor

An image batch has shape:

```text
[batch, channels, height, width]
```

For the custom CNN, a batch of 64 RGB images has shape `[64, 3, 64, 64]`. Pixel values are converted to floating point and normalized using ImageNet channel statistics. Normalization is essential for the pretrained ResNet and also gives the CNN well-scaled inputs.

Use `notebooks/01_explore_eurosat.ipynb` to inspect the raw images, labels, tensor shapes, and batches interactively.

## 3. Read the model from top to bottom

Open `src/terraclass/models.py` and find `SimpleCNN`.

Each convolution learns filters that respond to useful local patterns. The progression is:

```text
RGB image -> 32 feature maps -> 64 feature maps -> 128 feature maps -> 10 logits
```

- `Conv2d` learns spatial filters.
- `BatchNorm2d` stabilizes feature scales during training.
- `ReLU` introduces nonlinearity.
- `MaxPool2d` halves spatial resolution.
- `AdaptiveAvgPool2d` summarizes every feature map with one number.
- `Linear` maps the 128 learned features to 10 class scores.

The model returns raw scores called logits. Do not add softmax before `CrossEntropyLoss`; that loss applies the appropriate log-softmax operation internally.

Set a breakpoint in `SimpleCNN.forward` and run a one-epoch experiment to inspect the feature tensor's shape.

## 4. Follow one optimization step

Open `src/terraclass/engine.py` and read `run_epoch`.

The five central training operations are:

```python
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = criterion(logits, targets)
loss.backward()
optimizer.step()
```

Their roles are:

1. Clear gradients left by the previous batch.
2. Run the forward pass.
3. Measure how wrong the predictions are.
4. Use autograd to calculate each parameter's contribution to the loss.
5. Let AdamW update the parameters.

During validation, there is no optimizer. `torch.set_grad_enabled(False)` prevents PyTorch from building the backward graph, which reduces memory and computation.

## 5. Separate model selection from final evaluation

Training metrics describe the examples used to update weights. Validation metrics select the best epoch and trigger early stopping. Test metrics are calculated once with the selected model to estimate performance on unseen data.

TerraClass chooses `best.pt` using validation macro F1. Macro F1 gives every category equal weight, so performance on a smaller class cannot disappear inside a high overall accuracy.

Open `src/terraclass/metrics.py` and trace one row of the confusion matrix. Rows are true labels and columns are predicted labels. Large off-diagonal values reveal specific pairs of classes the model confuses.

## 6. Compare learning from scratch with transfer learning

The custom CNN starts with random weights. ResNet18 starts with filters learned from ImageNet, replaces its 1,000-class output layer with a 10-class layer, and fine-tunes the network on EuroSAT.

Run these two controlled experiments:

```powershell
terraclass train --model cnn --epochs 15 --output-dir outputs/cnn
terraclass train --model resnet18 --epochs 10 --batch-size 32 --output-dir outputs/resnet18
```

Compare test macro F1, training time, learning curves, and confusion matrices. A good project discussion explains not just which number is higher, but why pretraining, input resolution, and model capacity affect the outcome.

Then try freezing ResNet's backbone:

```powershell
terraclass train --model resnet18 --freeze-backbone --epochs 5 --output-dir outputs/frozen
```

Only the final linear layer updates in that run. It is faster, but it has less freedom to adapt pretrained features to satellite imagery.

## 7. Inspect errors instead of stopping at accuracy

Use the confusion matrix to choose two frequently confused classes. Save Grad-CAM overlays for correct and incorrect examples from both classes. Ask:

- Is the model looking at relevant land patterns or at image edges?
- Does the prediction depend on color, texture, road geometry, or building density?
- Are some labels visually ambiguous at 64-by-64 resolution?
- Does ResNet focus differently from the custom CNN?

This qualitative analysis is often the most interesting part of a portfolio write-up.

## Exercises

Complete these in order rather than changing many variables at once:

1. Run a one-epoch smoke test and explain every logged value.
2. Train the CNN with and without augmentation and compare validation F1.
3. Change the CNN's channel counts and predict the memory/performance effect before running it.
4. Compare frozen and fully fine-tuned ResNet18.
5. Find the two most-confused classes programmatically from `metrics.json`.
6. Add a precision/recall bar chart for each class.
7. Replace ResNet18 with another TorchVision model while preserving the shared pipeline.

For each experiment, record the hypothesis, change, result, and conclusion. That habit is more valuable than collecting isolated accuracy scores.
