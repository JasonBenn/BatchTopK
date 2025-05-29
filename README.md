# BatchTopK Sparse Autoencoders

BatchTopK is a novel approach to Sparse Autoencoders (SAEs) that offers an alternative to TopK SAEs as introduced by OpenAI. This repository contains the implementation and experiments for BatchTopK SAEs, as described in our preliminary findings.

## What is the BatchTopK activation function?

BatchTopK modifies the TopK activation in SAEs in the following way:

1. Instead of applying TopK to each sample independently, it flattens all feature activations across the batch.
2. It then takes the top (K * batch_size) activations.
3. Finally, it reshapes the result back to the original batch shape.

## Installation

### From Source
```bash
git clone https://github.com/jasoncbenn/BatchTopK.git
cd BatchTopK
pip install -e .
```

### As a Package
```bash
pip install batchtopk
```

## Quick Start

### Basic Usage
```python
from batchtopk import BatchTopKSAE, get_default_cfg

# Create configuration
cfg = get_default_cfg()
cfg["act_size"] = 768  # GPT-2 small hidden size
cfg["dict_size"] = 768 * 16  # 16x expansion
cfg["top_k"] = 32  # Number of active features
cfg["device"] = "cuda"

# Initialize SAE
sae = BatchTopKSAE(cfg)

# Use with activations
activations = torch.randn(batch_size, 768)  # Your model activations
output = sae(activations)
print(f"Loss: {output['loss']:.4f}, L0: {output['l0_norm']:.1f}")
```

### Using BatchTopKCrossCoder for Model Diffing
```python
from batchtopk import BatchTopKCrossCoder

# Configuration for cross-coder
cfg = {
    "act_size": 768,      # Activation dimension
    "dict_size": 12288,   # Dictionary size  
    "top_k": 200,         # Sparsity per example
    "device": "cuda",
    "dtype": torch.float32,
    "seed": 42,
    "dec_init_norm": 0.08,
    "n_batches_to_dead": 250
}

# Initialize cross-coder
crosscoder = BatchTopKCrossCoder(cfg)

# Compare activations from two models
base_acts = torch.randn(batch_size, 768)     # Base model
ft_acts = torch.randn(batch_size, 768)       # Fine-tuned model
combined = torch.stack([base_acts, ft_acts], dim=1)  # Shape: (batch, 2, 768)

# Encode and decode
sparse_features = crosscoder.encode(combined)  # Shape: (batch, dict_size)
reconstructed = crosscoder.decode(sparse_features)  # Shape: (batch, 2, 768)
```

## API Reference

### Core Classes

#### `BatchTopKSAE`
Sparse autoencoder with batch-wise TopK activation.
- **Methods**: `forward(x)`, `encode(x)`, `decode(acts)`

#### `BatchTopKCrossCoder`
Cross-coder for comparing activations from two models using BatchTopK.
- **Methods**: `forward(x_B2D)`, `encode(x_B2D)`, `decode(acts_BH)`

#### `VanillaSAE`, `TopKSAE`, `JumpReLUSAE`
Alternative SAE architectures for comparison.

### Training Functions

#### `train_sae(sae, activation_store, model, cfg)`
Train a single SAE on model activations.

#### `train_sae_group(sae_group, activation_store, model, cfg)` 
Train multiple SAEs in parallel.

### Configuration

#### `get_default_cfg()`
Returns default configuration dictionary.

#### `post_init_cfg(cfg)`
Post-processes configuration after initialization.

## Acknowledgments
The training code is heavily inspired and basically a stripped-down version of [SAELens](https://github.com/jbloomAus/SAELens). Thanks to the SAELens team for their foundational work in this area!
