# SupportRouter learning guide

Work through the project in this order. The goal is not just to run a pretrained model; it is to
understand the tensor transformations, optimization loop, evaluation design, and deployment
tradeoffs behind a Transformer classifier.

## 1. Learn the attention operation

Start with `notebooks/00_transformer_basics.ipynb`. It constructs token embeddings, positional
embeddings, query/key/value tensors, scaled dot-product attention, multi-head attention, and a
small `TransformerEncoder`.

At every cell, inspect the shape. NLP batches in this project use:

```text
[batch size, sequence length, embedding dimension]
```

Read `references/ATTENTION_EXPLAINED.md` when a formula in the notebook is unfamiliar.

## 2. Inspect Banking77 before training

Open `notebooks/01_explore_banking77.ipynb`. Look at class counts, text lengths, and messages
from several similar intents. Notice that labels such as a declined cash withdrawal and an
unrecognized cash withdrawal share vocabulary but require different semantic decisions.

The official training set is split deterministically in `data.py`. Ten percent from every class
becomes validation data. The official test set stays untouched until final evaluation.

## 3. Follow tokenization and batching

Open `tokenization.py`.

The scratch tokenizer lowercases text, separates words and punctuation, maps frequent tokens to
integer IDs, and maps unseen tokens to `<unk>`. It is fitted only on training text to avoid data
leakage. `ScratchCollator` pads each batch only to its longest sequence and creates an attention
mask where real tokens are 1 and padding is 0.

DistilBERT uses its pretrained subword tokenizer. Keeping that tokenizer is essential because
the pretrained embedding rows correspond to its exact vocabulary IDs.

## 4. Read the scratch model from top to bottom

Open `models.py` and find `ScratchTransformerClassifier`:

```text
token IDs
  -> token embedding + learned position embedding
  -> Transformer encoder layer x 2
  -> final LayerNorm
  -> masked mean pooling
  -> dropout
  -> 77-class Linear layer
  -> logits
```

The padding mask prevents attention from using `<pad>` positions. Masked mean pooling averages
only real token representations. The output is raw logits; `CrossEntropyLoss` applies the needed
log-softmax internally.

Use `references/MODEL_ARCHITECTURE.md` to calculate every trainable-parameter group.

## 5. Trace one optimizer update

Open `engine.py`. During training, every batch performs:

```python
optimizer.zero_grad(set_to_none=True)
output = model(input_ids=input_ids, attention_mask=attention_mask)
loss = criterion(logits, targets)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

Backpropagation calculates gradients for embeddings, attention projections, feed-forward
networks, normalization parameters, and the classifier. AdamW then updates all trainable
parameters. Gradient clipping limits the global gradient norm to reduce unstable updates.

During validation and test evaluation, there is no optimizer and PyTorch does not build a
backward graph.

## 6. Understand model selection

Training chooses the checkpoint with the highest validation macro F1. Macro F1 gives each of
the 77 intents equal weight, which helps expose weak classes that overall accuracy can hide.
Early stopping ends training after several epochs without improvement.

Only after model selection does the CLI evaluate the official test set. Repeatedly adjusting the
model based on test results would turn the test set into another validation set.

## 7. Compare scratch learning with transfer learning

Train both models:

```powershell
support-router train --model scratch --output-dir outputs/scratch
support-router train --model distilbert --output-dir outputs/distilbert
```

The scratch model must learn language representations and intent boundaries from Banking77
alone. DistilBERT begins with representations learned from a large text corpus and fine-tunes
them for the 77 labels. Compare:

- test accuracy and macro F1;
- training time and GPU memory;
- total/trainable parameter counts printed by the CLI;
- per-class failures in `test_report.md`;
- confidence calibration error;
- predictions for ambiguous or unrelated messages.

## 8. Examine uncertainty rather than only accuracy

Softmax turns logits into probabilities, and the largest probability is used as confidence.
`predict` routes a message to a human when that confidence is below a threshold. Try messages
that contain no banking context and vary `--confidence-threshold`.

Softmax confidence is not guaranteed to be a true probability of correctness. Expected
calibration error compares confidence with empirical accuracy in bins; it is a diagnostic, not a
guarantee. A stronger production project could add temperature scaling using validation data.

## 9. Suggested extensions

After reproducing the baseline, choose one extension and document the comparison:

1. Add temperature scaling and plot a reliability diagram.
2. Compare mean pooling with a learned classification token.
3. Add focal loss or class-weighted loss and study weak intents.
4. Export a model to ONNX and benchmark inference latency.
5. Serve predictions through FastAPI with structured logs and monitoring fields.
6. Add an out-of-distribution dataset and measure human-routing coverage versus accuracy.
