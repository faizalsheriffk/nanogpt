# Zero to One

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.7-brightgreen.svg)]

A minimal, educational implementation of a GPT-style language model in a single file, inspired by Karpathy's NanoGPT which is inturn inspired by OpenAI's GPT-2 and HuggingFace's Transformers.  This repository is designed for going from novice understanding of LLMs to next iteration with the emphasis on basics, clarity, experimentation, and small-scale CPU based training.


## Table of Contents

- [Overview](#overview)  
- [Installation](#installation)  
- [Quickstart](#usage)  
- [Training](#training)  
- [Generation](#generation)  
- [High-Level Architecture](#architecture)  
- [Class-by-Class Deep Dive](#Class-by-Class Deep Dive)  
- [Acknowledgements](#acknowledgements)  


## Overview

The simplest, fastest repository for training/finetuning medium-sized GPTs. It is an inspiration of [nanoGPT](https://github.com/karpathy/nanoGPT) by @karpathy where I prioritize going over the basics and obvious concepts for a beginner. As karpathy mentions the file `train.py` reproduces GPT-2 (124M) on OpenWebText. If you run this on a single 8XA100 40GB node it will take around 4 days of training. The code itself is plain and readable: `train.py` is a ~300-line boilerplate training loop and `model.py` a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. It is as powerful as it sounds -- my goal in this repo is to mellow it down further for beginners so it is so simple, it is very easy to hack to your needs, train new models from scratch, most importantly run on CPU as a default option, or finetune pretrained checkpoints

## Installation

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## Quick start

The fastest way to get started is to train a character-level GPT on the works of Shakespeare. First, we download it as a single (1MB) file and turn it from raw text into one large stream of integers:

```sh
python3 data/shakespeare_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Now it is time to train your GPT. The size of it very much depends on the computational resources of your system:

## Training
I have just a CPU. Great, don't worry about it. We can still quickly train a baby GPT with the settings provided in the [config/train_shakespeare_char.py](config/train_shakespeare_char.py) config file:

```sh
python3 train.py config/train_shakespeare_char.py
```

Peeking inside the config file, you’ll find settings tuned for CPU:

- Evaluation every 20 iterations
- Context window size of 64 characters
- Batch size of only 12 examples
- Transformer with 4 layers, 4 attention heads, -128-dimensional embeddings
- 2000 total iterations
- Dropout set to 0.0 to ease regularization on such a small model:

## Generation

```sh
python3 sample.py --out_dir=out-shakespeare-char
```

Despite the modest setup, training completes in around ~3 minutes, achieving a loss of approximately 1.88.
Although the model generates poorer-quality samples compared to large GPTs, it still captures the basic character structure and is fun to explore.

Model checkpoints are saved under the out-shakespeare-char/ directory.
After training, you can sample from the best model: This generates a few samples, for example:

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear

```

## High-Level Architecture
At its core, this single-file GPT implementation mirrors the standard GPT-2 design:

1) Embeddings
    - Token embeddings (wte) turn vocabulary indices into vectors.
    - Position embeddings (wpe) inject information about each token’s position in the sequence.

2) Transformer Blocks (Block) stacked n_layer times, each consisting of:
    - LayerNorm → Causal Self-Attention → residual add
    - LayerNorm → Feed-Forward (MLP) → residual add

3) Final LayerNorm and a Linear “head” (lm_head) that projects back to vocabulary logits.

4) Weight tying: the token embedding matrix and the final linear projection share weights.

5) Utilities for initializing, counting parameters, adjusting context length, loading pretrained weights, configuring optimizers, measuring FLOP utilization, and autoregressive generation.


## Class-by-Class Deep Dive

### LayerNorm (`nn.Module`)

Implements layer normalization with optional bias.

- `__init__(ndim, bias)`  
  - `self.weight = Parameter(torch.ones(ndim))` → learnable scale γ, initialized to 1.  
  - `self.bias = Parameter(torch.zeros(ndim))` if `bias=True`, else `None` → learnable shift β.

- `forward(input)`  
  Applies layer normalization:
  
```
y = (x - μ) / sqrt(σ² + ε) * γ + β

```

with `eps=1e-5` to avoid division by zero.


### CausalSelfAttention (`nn.Module`)

Multi-head self-attention with a causal mask to prevent peeking into the future.

- `__init__(config)`
- Confirms `n_embd % n_head == 0`.
- `c_attn`: Projects to Query, Key, Value (3 × embedding).
- `c_proj`: Projects back from embedding.
- Dropouts: `attn_dropout` (attention weights) and `resid_dropout` (output).
- Detects FlashAttention (`scaled_dot_product_attention`), else registers manual causal mask.

- `forward(x)`
- Projects: `q, k, v = self.c_attn(x).split(n_embd, dim=2)`.
- Reshapes for heads: `(batch, n_head, seq_len, head_dim)`.
- Attention paths:
  - **Flash path**: Use `scaled_dot_product_attention(q, k, v, is_causal=True)`.
  - **Manual path**:
    1. Scaled dot-product:
       ```
       scores = (q @ k^T) / sqrt(head_dim)
       ```
    2. Mask out future positions.
    3. Softmax → dropout → multiply with v.
- Reassemble heads and project: `y = resid_dropout(c_proj(y))`.


### MLP (`nn.Module`)

Position-wise feed-forward network.

- `__init__(config)`
- `c_fc`: Expand from `n_embd → 4 * n_embd`.
- GELU activation.
- `c_proj`: Project back `4 * n_embd → n_embd`.
- `dropout` for regularization.

- `forward(x)`

```
x = c_fc(x)
x = gelu(x)
x = c_proj(x)
x = dropout(x)
```



### Block (`nn.Module`)

One Transformer block combining attention + MLP with residual connections.

- `__init__(config)`
- Instantiates `ln_1`, `attn`, `ln_2`, `mlp`.

- `forward(x)`

```
x = x + attn(ln_1(x))
x = x + mlp(ln_2(x))
return x
```


### GPTConfig (`dataclass`)

Holds model hyperparameters:

```python
block_size: int    # max context length
vocab_size: int    # size of token vocabulary
n_layer: int       # number of Transformer blocks
n_head: int        # number of attention heads
n_embd: int        # embedding dimension
dropout: float     # dropout probability
bias: bool         # whether to include biases
```
## Acknowledgements

nanoGPT @https://github.com/karpathy/minGPT
3blue1brown https://www.youtube.com/@3blue1brown
