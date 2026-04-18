# 🧠 LLM Architectures

> Dead-simple PyTorch implementations of large language model architectures — built and trained from scratch with the simplest code possible.

## What is this?

This repo is a collection of LLM architectures implemented in minimal, readable PyTorch. No bloated abstractions, no magic frameworks — just clean code that shows exactly how these models work under the hood.

The goal is to make it easy to:
- Understand how LLM architectures are structured
- Train models from scratch on custom data
- Experiment with different configurations

---

## Quickstart

### 1. Define and create a model

```python
from types import SimpleNamespace
from model.model import GPT
import torch

gpt1 = GPT(SimpleNamespace(
    vocab_size=100,
    n_layers=2,
    dropout=0.1,
    n_embd=200,
    n_head=8,
    attn_pdrop=0.1,
    resid_pdrop=0.1,
    block_size=100,
    flash=True,
    dtype=torch.bfloat16
))
```

### 2. Train the model

```python
from model.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = SimpleNamespace(epochs=10, batch_size=4, lr=1e-3, shuffle=True, device=device)

trainer = Trainer(config, gpt1, dataset)
trainer.train(use_tqdm=True)
```

### 3. Inspect the model

```python
gpt1.get_num_params()    # Total Parameters: 1,234,567
gpt1.get_model_size()    # model size: 2.34 MB
gpt1.get_model_dtype()   # model dtype: torch.bfloat16
```

---

## Model Configuration

| Parameter | Description |
|-----------|-------------|
| `vocab_size` | Size of the token vocabulary |
| `n_layers` | Number of transformer blocks |
| `n_embd` | Embedding dimension |
| `n_head` | Number of attention heads |
| `block_size` | Maximum sequence length |
| `dropout` | Dropout rate |
| `attn_pdrop` | Attention dropout rate |
| `resid_pdrop` | Residual dropout rate |
| `flash` | Use Flash Attention (`True`/`False`) |
| `dtype` | Model dtype (`torch.float32`, `torch.bfloat16`) |

## Trainer Configuration

| Parameter | Description |
|-----------|-------------|
| `epochs` | Number of training epochs |
| `batch_size` | Batch size |
| `lr` | Learning rate |
| `shuffle` | Shuffle dataset each epoch |
| `device` | Training device (`cpu` / `cuda`) |

---

## Notes on dtype

- **`torch.float32`** — safest, works everywhere
- **`torch.bfloat16`** — recommended for faster training, stable on both CPU and GPU
- **`torch.float16`** — avoid for training, especially on CPU (unstable gradients)

---

## Architectures

- [x] GPT (decoder-only transformer)
- [ ] More coming soon...