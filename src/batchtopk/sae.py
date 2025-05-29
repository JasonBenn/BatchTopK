import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import einops
import numpy as np
from typing import Optional, Tuple


class BaseAutoencoder(nn.Module):
    """Base class for autoencoder models."""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        torch.manual_seed(self.cfg["seed"])

        self.b_dec = nn.Parameter(torch.zeros(self.cfg["act_size"]))
        self.b_enc = nn.Parameter(torch.zeros(self.cfg["dict_size"]))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["act_size"], self.cfg["dict_size"])
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["dict_size"], self.cfg["act_size"])
            )
        )
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num_batches_not_active = torch.zeros((self.cfg["dict_size"],)).to(
            cfg["device"]
        )

        self.to(cfg["dtype"]).to(cfg["device"])

    def preprocess_input(self, x):
        if self.cfg.get("input_unit_norm", False):
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        else:
            return x, None, None

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        if self.cfg.get("input_unit_norm", False):
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def update_inactive_features(self, acts):
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0


class BatchTopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts.flatten(), self.cfg["top_k"] * x.shape[0], dim=-1)
        acts_topk = (
            torch.zeros_like(acts.flatten())
            .scatter(-1, acts_topk.indices, acts_topk.values)
            .reshape(acts.shape)
        )
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + l1_loss + aux_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)


class TopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)
        acts_topk = torch.zeros_like(acts).scatter(
            -1, acts_topk.indices, acts_topk.values
        )
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + l1_loss + aux_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)


class VanillaSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        self.update_inactive_features(acts)
        output = self.get_loss_dict(x, x_reconstruct, acts, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts > 0).float().sum(-1).mean()
        loss = l2_loss + l1_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
        }
        return output


import torch
import torch.nn as nn
import torch.autograd as autograd

class RectangleFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input

class JumpReLUFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth

class JumpReLU(nn.Module):
    def __init__(self, feature_size, bandwidth, device='cpu'):
        super(JumpReLU, self).__init__()
        self.log_threshold = nn.Parameter(torch.zeros(feature_size, device=device))
        self.bandwidth = bandwidth

    def forward(self, x):
        return JumpReLUFunction.apply(x, self.log_threshold, self.bandwidth)

class StepFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth

class JumpReLUSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.jumprelu = JumpReLU(feature_size=cfg["dict_size"], bandwidth=cfg["bandwidth"], device=cfg["device"])

    def forward(self, x, use_pre_enc_bias=False):
        x, x_mean, x_std = self.preprocess_input(x)

        if use_pre_enc_bias:
            x = x - self.b_dec

        pre_activations = torch.relu(x @ self.W_enc + self.b_enc)
        feature_magnitudes = self.jumprelu(pre_activations)

        x_reconstructed = feature_magnitudes @ self.W_dec + self.b_dec

        return self.get_loss_dict(x, x_reconstructed, feature_magnitudes, x_mean, x_std)

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()

        l0 = StepFunction.apply(acts, self.jumprelu.log_threshold, self.cfg["bandwidth"]).sum(dim=-1).mean()
        l0_loss = self.cfg["l1_coeff"] * l0
        l1_loss = l0_loss

        loss = l2_loss + l1_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0,
            "l1_norm": l0,
        }
        return output


class BatchTopKCrossCoder(nn.Module):
    """BatchTopK CrossCoder for model diffing between two models.
    
    This is a sparse autoencoder that learns to represent activations from two different
    models (e.g., base and fine-tuned) using a shared dictionary with batch-wise TopK
    sparsity. It uses the BatchTopK activation function where:
    1. All feature activations across the batch are flattened
    2. The top (K * batch_size) activations are selected
    3. Activations are reshaped back to the original batch shape
    
    Args:
        cfg: Configuration dict with the following keys:
            - act_size (int): Size of input activations (D)
            - dict_size (int): Size of the sparse dictionary (H)
            - top_k (int): Number of active features per example
            - device (str): Device to run on
            - dtype (torch.dtype): Data type to use
            - seed (int): Random seed
            - dec_init_norm (float): Initial norm for decoder weights
            - aux_penalty (float): Auxiliary loss penalty
            - top_k_aux (int): Top-k for auxiliary loss
            - n_batches_to_dead (int): Batches before feature considered dead
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.H = cfg["dict_size"]
        self.D = cfg["act_size"]
        self.k = cfg["top_k"]
        
        torch.manual_seed(cfg["seed"])
        
        # Encoder and decoder weights for 2 models
        self.W_dec_H2D = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.H, 2, self.D, dtype=cfg["dtype"])
            )
        )
        # Initialize decoder norms
        dec_init_norm = cfg.get("dec_init_norm", 0.08)
        self.W_dec_H2D.data = (
            self.W_dec_H2D.data
            / self.W_dec_H2D.data.norm(dim=-1, keepdim=True)
            * dec_init_norm
        )
        
        # Initialize encoder as transpose of decoder
        self.W_enc_2DH = nn.Parameter(torch.empty(2, self.D, self.H, dtype=cfg["dtype"]))
        self.W_enc_2DH.data[:] = self.W_dec_H2D.permute(1, 2, 0).data
        
        self.b_enc_H = nn.Parameter(torch.zeros((self.H,), dtype=cfg["dtype"]))
        self.b_dec_2D = nn.Parameter(torch.zeros((2, self.D), dtype=cfg["dtype"]))
        
        # Dead feature tracking
        self.num_batches_not_active = torch.zeros(
            (self.H,), dtype=torch.float32, device=cfg["device"]
        )
        
        # For inference mode threshold estimation
        self.inference_threshold = None
        self.threshold_samples = []
        
        self.to(cfg["device"])
    
    def get_batch_topk_mask(self, x_enc_BH: torch.Tensor) -> torch.Tensor:
        """Apply BatchTopK: select top k*B activations across entire batch."""
        B = x_enc_BH.size(0)
        F_total_batch = x_enc_BH.numel()
        k_total_batch = min(self.k * B, F_total_batch)
        
        x_flat = x_enc_BH.reshape(-1)
        _, flat_indices = torch.topk(x_flat, k_total_batch, sorted=False)
        mask_flat = torch.zeros_like(x_flat, dtype=torch.bool)
        mask_flat[flat_indices] = True
        return mask_flat.view_as(x_enc_BH)
    
    def encode(self, x_B2D: torch.Tensor, inference: bool = False) -> torch.Tensor:
        """Encode activations from both models into sparse features.
        
        Args:
            x_B2D: Activations of shape (batch, 2, act_size)
            inference: If True, use threshold-based activation (for deployment)
        
        Returns:
            Sparse features of shape (batch, dict_size)
        """
        # Combine both models' activations
        x_enc_BH = einops.einsum(
            x_B2D,
            self.W_enc_2DH,
            "batch n_models d_model, n_models d_model d_hidden -> batch d_hidden",
        )
        # Apply ReLU before topk selection
        x_enc_BH = F.relu(x_enc_BH + self.b_enc_H)
        
        if inference and self.inference_threshold is not None:
            # Use threshold for inference (JumpReLU-style)
            acts = x_enc_BH * (x_enc_BH > self.inference_threshold).float()
        else:
            # Training: use BatchTopK
            mask_BH = self.get_batch_topk_mask(x_enc_BH)
            acts = x_enc_BH * mask_BH
            
            # Collect threshold samples for later estimation
            if self.training and len(self.threshold_samples) < 100:
                min_positive = acts[acts > 0].min().item() if (acts > 0).any() else 0
                if min_positive > 0:
                    self.threshold_samples.append(min_positive)
        
        return acts
    
    def decode(self, acts_BH: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to both models' activation spaces.
        
        Args:
            acts_BH: Sparse features of shape (batch, dict_size)
            
        Returns:
            Reconstructed activations of shape (batch, 2, act_size)
        """
        acts_dec_B2D = einops.einsum(
            acts_BH,
            self.W_dec_H2D,
            "batch d_hidden, d_hidden n_models d_model -> batch n_models d_model",
        )
        return acts_dec_B2D + self.b_dec_2D
    
    def forward(self, x_B2D: torch.Tensor, inference: bool = False) -> torch.Tensor:
        """Forward pass: encode then decode.
        
        Args:
            x_B2D: Activations of shape (batch, 2, act_size) 
            inference: If True, use inference mode
            
        Returns:
            Reconstructed activations of shape (batch, 2, act_size)
        """
        acts_BH = self.encode(x_B2D, inference=inference)
        return self.decode(acts_BH)
    
    @torch.no_grad()
    def estimate_inference_threshold(self):
        """Estimate threshold for inference mode based on training samples."""
        if len(self.threshold_samples) > 0:
            self.inference_threshold = torch.tensor(
                np.mean(self.threshold_samples), 
                device=self.cfg["device"], 
                dtype=self.cfg["dtype"]
            )
    
    @torch.no_grad()
    def update_inactive_features(self, acts_BH: torch.Tensor):
        """Track which features haven't fired recently."""
        inactive_H = acts_BH.sum(0) == 0
        self.num_batches_not_active += inactive_H.float()
        self.num_batches_not_active[~inactive_H] = 0
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        """Normalize decoder weights and project gradients to maintain unit norm."""
        W_dec_normed = self.W_dec_H2D / self.W_dec_H2D.norm(dim=-1, keepdim=True)
        
        if self.W_dec_H2D.grad is not None:
            W_dec_grad_proj = (self.W_dec_H2D.grad * W_dec_normed).sum(
                -1, keepdim=True
            ) * W_dec_normed
            self.W_dec_H2D.grad -= W_dec_grad_proj
        
        self.W_dec_H2D.data = W_dec_normed
