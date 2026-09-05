# Attention and Transformer concepts

## Tokens and embeddings

A tokenizer converts text into discrete token IDs. The IDs themselves do not express meaning;
they are row numbers in a learned embedding matrix. For vocabulary size `V` and model dimension
`D`, the token embedding has shape `[V, D]` and contains `V x D` trainable values.

A batch of IDs shaped `[B, L]` becomes embeddings shaped `[B, L, D]`, where `B` is batch size
and `L` is sequence length.

## Why position information is needed

Self-attention alone treats its inputs like an unordered set. A position embedding gives each
location a learned vector, which is added to its token embedding:

```text
input representation = token embedding + position embedding
```

SupportRouter learns a `[max_length, D]` position table from scratch. DistilBERT starts with a
position table learned during pretraining.

## Queries, keys, and values

Each token representation `X` is multiplied by three learned matrices:

```text
Q = X W_Q       queries: what this token is looking for
K = X W_K       keys: what this token advertises
V = X W_V       values: information this token can contribute
```

For one attention head, all three tensors have shape `[B, L, d_head]`. A query is compared with
every key using a dot product. Larger scores mean the model should draw more information from
that key's value.

## Scaled dot-product attention

The operation is:

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_head)) V
```

`Q K^T` creates an `[L, L]` score matrix for each example and head. Dividing by
`sqrt(d_head)` prevents dot products from growing so large that softmax becomes nearly one-hot
and produces poorly behaved gradients. Softmax normalizes each query's scores to weights that
sum to one. Multiplying those weights by `V` produces a context-aware representation.

## Multi-head attention

Rather than use one large attention calculation, multi-head attention splits dimension `D` into
`H` heads, each of size `d_head = D / H`. Different heads can learn different relationships—for
example, one may focus on transaction types while another connects negation with an action.
Head outputs are concatenated and passed through a learned output projection.

`D` must be divisible by `H`. SupportRouter defaults to `D=128`, `H=4`, so each head uses 32
features. Four heads do not multiply the input/output dimension: their 32-feature outputs join
back into 128 features.

## Padding and attention masks

Messages have different lengths, so a batch adds `<pad>` tokens to shorter messages. The
attention mask is 1 at real tokens and 0 at padding. PyTorch's key-padding mask excludes the
padding columns from attention, and masked mean pooling excludes padding from the final average.

The mask contains no learned parameters. It only tells the computation which positions are
valid.

## Residual connections and LayerNorm

Each attention or feed-forward sublayer is wrapped in a residual connection:

```text
output = input + sublayer(input)
```

This creates a direct path for information and gradients through deep networks. LayerNorm then
keeps each token's feature scale controlled. Its learned `gamma` scale and `beta` shift allow
the network to preserve or reshape normalized features when that reduces loss. They are learned
through the same backpropagation and AdamW updates as every other parameter.

SupportRouter uses pre-normalization (`norm_first=True`): normalization occurs before each
sublayer, a layout that is often stable for training a Transformer from scratch.

## The feed-forward network

Attention mixes information between token positions. The feed-forward network transforms each
position independently with the same two linear layers:

```text
D -> feedforward_size -> D
```

A GELU nonlinearity between the layers lets the model learn nonlinear features. SupportRouter
uses `128 -> 256 -> 128`.

## Encoder versus decoder Transformers

An encoder lets every real input token attend to every other real input token, which suits
classification because the entire message is already known. A causal decoder masks future
positions and predicts text one token at a time, which suits generation.

SupportRouter and DistilBERT are encoder-only classifiers. They do not generate support replies;
they map a complete message to one intent label.

## Pooling and classification

The scratch model averages the final representations of all non-padding tokens. A linear layer
maps the resulting `[B, D]` tensor to `[B, 77]` logits. The largest logit is the predicted intent.
During training, cross-entropy compares all 77 logits with the correct label and backpropagates
the error through the entire model.
