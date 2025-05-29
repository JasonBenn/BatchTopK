"""BatchTopK: Novel Sparse Autoencoders with batch-wise TopK activation.

This package provides implementations of various Sparse Autoencoder (SAE) architectures,
with a focus on the novel BatchTopK approach that modifies TopK activation by:
1. Flattening all feature activations across the batch
2. Taking the top (K * batch_size) activations
3. Reshaping back to the original batch shape
"""

from .sae import (
    BaseAutoencoder,
    VanillaSAE,
    TopKSAE,
    BatchTopKSAE,
    JumpReLUSAE,
    BatchTopKCrossCoder,
)
from .config import get_default_cfg, post_init_cfg

# Optional imports that require transformer_lens
try:
    from .training import train_sae, train_sae_group
    from .activation_store import ActivationStore
except ImportError:
    # These will be None if transformer_lens is not installed
    train_sae = None
    train_sae_group = None
    ActivationStore = None

__version__ = "0.1.0"

__all__ = [
    # SAE Classes
    "BaseAutoencoder",
    "VanillaSAE",
    "TopKSAE",
    "BatchTopKSAE",
    "JumpReLUSAE",
    "BatchTopKCrossCoder",
    # Configuration
    "get_default_cfg",
    "post_init_cfg",
    # Training
    "train_sae",
    "train_sae_group",
    # Data
    "ActivationStore",
]