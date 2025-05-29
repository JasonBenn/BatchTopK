# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BatchTopK is a novel approach to Sparse Autoencoders (SAEs) that modifies the TopK activation by:
1. Flattening all feature activations across the batch
2. Taking the top (K * batch_size) activations
3. Reshaping back to the original batch shape

The implementation is based on transformer_lens and is inspired by SAELens.

## Key Commands

### Running experiments
```bash
python main.py
```

### Generating visualizations from experiment results
```bash
python results.py
```

### Installing dependencies
```bash
pip install transformer_lens
```

## Architecture

### Core Components

1. **SAE Types** (sae.py):
   - `VanillaSAE`: Standard SAE with L1 regularization
   - `TopKSAE`: OpenAI's TopK sparse autoencoder
   - `BatchTopKSAE`: Novel batch-wise TopK implementation
   - `JumpReLUSAE`: JumpReLU activation-based SAE

2. **Training Pipeline**:
   - `activation_store.py`: Manages loading and batching activations from transformer models
   - `training.py`: Contains training loops for single SAE (`train_sae`) and multiple SAEs (`train_sae_group`)
   - `config.py`: Configuration management with `get_default_cfg()` and `post_init_cfg()`

3. **Experiment Configuration**:
   - Models are trained on GPT-2 small, layer 8, residual stream
   - Default dictionary size: 768 * 16 = 12,288
   - Default batch size: 4,096
   - Uses OpenWebText dataset
   - Weights & Biases logging enabled

### Key Configuration Parameters
- `sae_type`: "vanilla", "topk", "batchtopk", or "jumprelu"
- `top_k`: Number of active features (default: 32)
- `dict_size`: SAE dictionary size
- `l1_coeff`: L1 regularization coefficient (0 for TopK variants)
- `aux_penalty`: Auxiliary loss penalty for TopK variants
- `bandwidth`: JumpReLU-specific parameter

## Development Notes

- All SAEs inherit from `BaseAutoencoder` which handles weight initialization and normalization
- Decoder weights are kept unit norm during training via `make_decoder_weights_and_grad_unit_norm()`
- Dead feature tracking is implemented to monitor inactive features
- Input normalization is supported via `input_unit_norm` config flag
- Experiments in main.py include sweeps over L1 coefficients, top_k values, and dictionary sizes