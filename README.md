# Mini Language Model (MiniLM) — From Scratch Transformer in PyTorch

A lightweight implementation of a decoder-style Transformer language model built from scratch using PyTorch. This project focuses on understanding and experimenting with core components of modern language models, including tokenization, self-attention, and autoregressive training.

---

## Overview

This repository implements a small-scale language model inspired by GPT-style architectures. It is designed for educational and experimental purposes, enabling exploration of:

* Transformer decoder architecture
* Self-attention mechanisms
* Autoregressive next-token prediction
* Training dynamics on character/subword-level datasets
* Attention visualization

---

## Features

* Custom Transformer-based language model (`MiniLM`)
* Tokenization pipeline (`MTokenizer`)
* Windowed dataset loading for sequence modeling
* Training loop with:

  * Cross-entropy loss
  * AdamW optimizer
  * Cosine annealing learning rate scheduler
  * Gradient clipping
* Model checkpoint saving and loading
* Attention heatmap visualization
* Sampling-based text generation (sanity check function)

---

## Model Architecture

The model is a simplified decoder-only Transformer consisting of:

* Token embedding layer
* Positional encoding (learned or implicit in implementation)
* Multiple Transformer decoder layers
* Multi-head self-attention
* Feed-forward networks
* Output projection to vocabulary logits

The model is trained using next-token prediction:

Each input sequence ( x = (x_1, x_2, ..., x_n) ) predicts the next tokens ( y = (x_2, x_3, ..., x_{n+1}) ).

---

## Training Objective

The model minimizes cross-entropy loss:

The training objective is the standard next-token prediction cross-entropy loss:

```text
L = - Σ log P(x_t | x_<t)
```

Where:

* `L` = total loss
* `T` = sequence length
* `x_t` = token at position `t`
* `x_<t` = all tokens before position `t`
* `P(x_t | x_<t)` = predicted probability of token `x_t` given the previous context

---

## Sanity Check (Generation)

The project includes a sampling utility for inspecting model behavior during training.

It supports:

* Greedy decoding
* Stochastic sampling
* Attention visualization

Example:

```python
sanity_check(model, inp="KING:", limit=32, do_sample=True)
```

---

## Training Details

* Optimizer: AdamW
* Scheduler: Cosine Annealing
* Loss: CrossEntropyLoss
* Gradient clipping: 1.0
* Device: CPU (configurable)

---

## Usage

### 1. Train Model

```bash
python train.py
```

### 2. Load Checkpoint

```python
model, optimizer, scheduler, _, _ = load_model(
    "checkpoints/your_checkpoint.pt",
    MiniLM,
    AdamW,
    CosineAnnealingLR,
    device,
    epochs,
)
```

### 3. Run Sanity Check

```python
sanity_check(model, "To be, or not to be", limit=30)
```

---

## Notes

* This implementation is intended for learning purposes, not production-scale training.

---
This was as an experimental project to deeply understand transformer-based language models from first principles.