#!/usr/bin/env python3
"""
Minimal test for CrossCoder implementation.
Run with: python test_crosscoder.py
"""

import sys
import os

# Try to import required packages
try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    from pathlib import Path
    import tempfile
    import json
    import shutil
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from train_crosscoder import CrossCoder, CrossCoderConfig, ActivationsSource, Trainer
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("\nPlease install required packages:")
    print("pip install torch numpy pydantic wandb tqdm einops")
    sys.exit(1)


def create_minimal_test_data(base_dir, ft_dir, n_samples=8, d_model=4):
    """Create minimal test activation shards."""
    base_dir = Path(base_dir)
    ft_dir = Path(ft_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    ft_dir.mkdir(parents=True, exist_ok=True)
    
    # Create small random activations
    base_acts = np.random.randn(n_samples, d_model).astype(np.float32)
    ft_acts = base_acts + 0.1 * np.random.randn(n_samples, d_model).astype(np.float32)
    
    # Save as memory-mapped files
    base_mm_path = base_dir / "0_activations.mm"
    ft_mm_path = ft_dir / "0_activations.mm"
    
    base_mm = np.memmap(base_mm_path, dtype=np.float32, mode='w+', shape=(n_samples, d_model))
    base_mm[:] = base_acts
    base_mm.flush()
    
    ft_mm = np.memmap(ft_mm_path, dtype=np.float32, mode='w+', shape=(n_samples, d_model))
    ft_mm[:] = ft_acts
    ft_mm.flush()
    
    # Create metadata
    metadata = {
        "activations": {
            "shape": [n_samples, d_model],
            "dtype": "float32"
        }
    }
    
    with open(base_dir / "0_metadata.json", 'w') as f:
        json.dump(metadata, f)
    
    with open(ft_dir / "0_metadata.json", 'w') as f:
        json.dump(metadata, f)
    
    # Create dataset.json
    dataset_meta = {"shards": ["0"]}
    with open(base_dir / "dataset.json", 'w') as f:
        json.dump(dataset_meta, f)
    with open(ft_dir / "dataset.json", 'w') as f:
        json.dump(dataset_meta, f)


def test_minimal_crosscoder():
    """Test the CrossCoder with minimal configuration."""
    print("Starting minimal CrossCoder test...")
    
    # Create temporary directories for test data
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir) / "base"
        ft_dir = Path(tmpdir) / "ft"
        
        # Create minimal test data
        d_model = 4
        n_samples = 8
        batch_size = 4
        create_minimal_test_data(base_dir, ft_dir, n_samples=n_samples, d_model=d_model)
        
        # Create minimal config
        cfg = CrossCoderConfig(
            D=d_model,
            H=8,  # Small hidden dimension
            k=2,  # Small top-k
            batch_size=batch_size,
            num_tokens=n_samples,  # Just one batch worth
            lr=1e-3,
            log_every=1,
            save_every=10,
            wandb_project="test",
            wandb_entity="test",
            tiny_mode=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print(f"Config: D={cfg.D}, H={cfg.H}, k={cfg.k}, batch_size={cfg.batch_size}")
        print(f"Device: {cfg.device}")
        
        # Test 1: CrossCoder initialization
        print("\n1. Testing CrossCoder initialization...")
        crosscoder = CrossCoder(cfg)
        assert crosscoder.W_dec_H2D.shape == (cfg.H, 2, cfg.D)
        assert crosscoder.W_enc_2DH.shape == (2, cfg.D, cfg.H)
        assert crosscoder.b_enc_H.shape == (cfg.H,)
        assert crosscoder.b_dec_2D.shape == (2, cfg.D)
        print("✓ CrossCoder initialized correctly")
        
        # Test 2: ActivationsSource
        print("\n2. Testing ActivationsSource...")
        activation_source = ActivationsSource(base_dir, ft_dir, cfg)
        batch = activation_source.next()
        assert batch.shape == (batch_size, 2, d_model)
        print(f"✓ ActivationsSource returned batch shape: {batch.shape}")
        
        # Test 3: Forward pass
        print("\n3. Testing forward pass...")
        x_B2D = torch.randn(batch_size, 2, d_model, device=cfg.device, dtype=crosscoder.dtype)
        
        # Test encode
        acts_BH = crosscoder.encode(x_B2D)
        assert acts_BH.shape == (batch_size, cfg.H)
        print(f"✓ Encode output shape: {acts_BH.shape}")
        
        # Check sparsity (should have at most k*batch_size non-zero activations)
        num_nonzero = (acts_BH > 0).sum().item()
        max_expected = cfg.k * batch_size
        assert num_nonzero <= max_expected, f"Too many non-zero activations: {num_nonzero} > {max_expected}"
        print(f"✓ Sparsity check: {num_nonzero} non-zero activations (max {max_expected})")
        
        # Test decode
        x_reconstruct_B2D = crosscoder.decode(acts_BH)
        assert x_reconstruct_B2D.shape == (batch_size, 2, d_model)
        print(f"✓ Decode output shape: {x_reconstruct_B2D.shape}")
        
        # Test full forward
        x_reconstruct_B2D_full = crosscoder.forward(x_B2D)
        assert x_reconstruct_B2D_full.shape == (batch_size, 2, d_model)
        print(f"✓ Full forward pass shape: {x_reconstruct_B2D_full.shape}")
        
        # Test 4: Loss computation
        print("\n4. Testing loss computation...")
        losses = crosscoder.get_losses(x_B2D)
        assert hasattr(losses, 'mse')
        assert hasattr(losses, 'aux_loss')
        assert hasattr(losses, 'total_loss')
        assert hasattr(losses, 'explained_variance')
        assert hasattr(losses, 'num_dead_features')
        print(f"✓ Losses computed: MSE={losses.mse:.4f}, Aux={losses.aux_loss:.4f}, Total={losses.total_loss:.4f}")
        
        # Test 5: Single training step
        print("\n5. Testing single training step...")
        optimizer = torch.optim.Adam(crosscoder.parameters(), lr=cfg.lr)
        
        # Store initial weights
        initial_W_enc = crosscoder.W_enc_2DH.clone()
        
        # Training step
        acts = activation_source.next()
        losses = crosscoder.get_losses(acts)
        losses.total_loss.backward()
        
        # Check gradients exist
        assert crosscoder.W_enc_2DH.grad is not None
        assert crosscoder.W_dec_H2D.grad is not None
        print("✓ Gradients computed")
        
        # Normalize decoder weights and project gradients
        crosscoder.make_decoder_weights_and_grad_unit_norm()
        
        # Check decoder weights are unit norm
        dec_norms = crosscoder.W_dec_H2D.norm(dim=-1)
        assert torch.allclose(dec_norms, torch.ones_like(dec_norms), atol=1e-5)
        print("✓ Decoder weights normalized to unit norm")
        
        optimizer.step()
        optimizer.zero_grad()
        
        # Check weights changed
        assert not torch.allclose(initial_W_enc, crosscoder.W_enc_2DH)
        print("✓ Weights updated after optimization step")
        
        # Test 6: Trainer integration (without W&B)
        print("\n6. Testing Trainer (mock W&B)...")
        import unittest.mock
        with unittest.mock.patch('train_crosscoder.wandb'):
            trainer = Trainer(cfg, activation_source)
            loss_dict = trainer.step()
            assert 'total_loss' in loss_dict
            assert 'mse' in loss_dict
            assert 'aux_loss' in loss_dict
            assert 'explained_variance' in loss_dict
            print(f"✓ Trainer step completed: {loss_dict}")
        
        print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_minimal_crosscoder()