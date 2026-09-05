# SupportRouter model architecture and parameter counts

## From-scratch Transformer

The default scratch configuration is:

| Setting | Value |
| --- | ---: |
| Maximum sequence length `L` | 64 |
| Model dimension `D` | 128 |
| Attention heads `H` | 4 |
| Dimension per head `D / H` | 32 |
| Encoder layers `N` | 2 |
| Feed-forward dimension `F` | 256 |
| Output intents `C` | 77 |
| Maximum vocabulary `V` | 20,000 |

The actual vocabulary can be smaller than 20,000 because it is built from the training set.
The CLI records the exact value in `experiment.json` and prints the parameter count.

### Tensor shapes

| Stage | Operation | Output shape |
| --- | --- | --- |
| Token IDs | Dynamically padded batch | `[B, L]` |
| Token embedding | Lookup 128 values per token | `[B, L, 128]` |
| Position embedding | Lookup and add by position | `[B, L, 128]` |
| Encoder layer 1 | Self-attention + feed-forward | `[B, L, 128]` |
| Encoder layer 2 | Self-attention + feed-forward | `[B, L, 128]` |
| Final LayerNorm | Normalize 128 features | `[B, L, 128]` |
| Masked mean | Average real token positions | `[B, 128]` |
| Classifier | Linear `128 -> 77` | `[B, 77]` |

Batch size and sequence length affect activation memory and computation, but not the number of
trainable parameters. The same weights are reused for every example and token position.

## How to count the parameters

### Embeddings

The token table has one `D`-value row per vocabulary token:

```text
token embedding = V x D
position embedding = L x D
```

At `V=20,000`, these contain `2,560,000` and `8,192` parameters respectively.

### Multi-head self-attention in one layer

PyTorch stores the query, key, and value projections together, but the count is equivalent to
three `D -> D` linear layers plus one output `D -> D` projection:

```text
Q, K, V weights = 3 x D x D
Q, K, V biases  = 3 x D
output weight   = D x D
output bias     = D
total           = 4D^2 + 4D
```

For `D=128`, that is `4(128^2) + 4(128) = 66,048` parameters. Splitting the values across four
heads changes how the computation is organized, not this total.

### Feed-forward network in one layer

Two linear layers expand from `D` to `F` and contract back to `D`:

```text
first linear  = D x F + F
second linear = F x D + D
total         = 2DF + F + D
```

For `D=128` and `F=256`, this is `65,920` parameters.

### LayerNorm in one layer

Each LayerNorm learns one `gamma` and one `beta` for every one of the `D` features. Each encoder
layer contains two LayerNorm modules:

```text
2 LayerNorms x (D gamma + D beta) = 4D = 512
```

Therefore one full encoder layer has:

```text
(4D^2 + 4D) + (2DF + F + D) + 4D
= 4D^2 + 2DF + F + 9D
= 132,480 parameters with D=128 and F=256
```

Two layers contain `264,960` parameters.

### Final normalization and classifier

```text
final LayerNorm = D + D = 256
classifier      = D x C weights + C biases
                = 128 x 77 + 77
                = 9,933
```

### Default total

The exact formula is:

```text
V x 128                     token embedding
+ 64 x 128                  position embedding
+ 2 x 132,480               encoder layers
+ 256                       final LayerNorm
+ 9,933                     classifier
= 128V + 283,341
```

With the maximum vocabulary `V=20,000`, the model has `2,843,341` trainable parameters. If the
fitted vocabulary has 8,000 entries, it has `1,307,341`. Vocabulary size changes embedding rows
but not attention, feed-forward, or classifier parameters.

You can verify any configuration directly:

```python
sum(parameter.numel() for parameter in model.parameters())
```

## DistilBERT transfer learning

The second model loads `distilbert/distilbert-base-uncased` and replaces/configures its final
classification head for 77 labels. Its pretrained subword embeddings and six Transformer layers
already encode broad English-language patterns. Fine-tuning updates the full network at a small
learning rate so those representations specialize to banking intents.

| Aspect | Scratch Transformer | DistilBERT |
| --- | --- | --- |
| Tokenizer | Learned word vocabulary | Pretrained WordPiece vocabulary |
| Encoder layers | 2 | 6 |
| Hidden dimension | 128 | 768 |
| Starting weights | Random | Pretrained language weights |
| Default learning rate | `1e-3` | `2e-5` |
| Typical strength | Transparent and fast to study | Better language representations |
| Typical cost | Lower memory | Higher memory and download size |

The CLI prints the exact model parameter count for the installed Transformers version rather
than relying on an approximate number in documentation.

## Confidence-aware routing

Softmax converts the 77 logits into values that sum to one. SupportRouter takes the largest value
as confidence:

```text
predicted intent = argmax(softmax(logits))
route to human   = confidence < threshold
```

The default threshold is 0.65. It is a policy setting, not a trained parameter. A higher
threshold routes more messages to humans and usually increases accuracy among automatically
routed messages; a lower threshold automates more messages but accepts more uncertain cases.
