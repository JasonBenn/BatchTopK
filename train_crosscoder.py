# %%
import numpy as np
import ipdb
import json
from torch import nn
import time
import torch.nn.functional as F
import einops
from typing import Optional, Union
import torch
from pathlib import Path
import argparse
import tqdm
import torch
from pydantic import BaseModel
import wandb
from torch.nn.utils import clip_grad_norm_

# %%
ROOT_DIR = Path(__file__).parent
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
DEVICE = "cuda:0"


class CrossCoderConfig(BaseModel):
    seed: int = 49
    batch_size: int = 4096
    lr: float = 5e-5
    num_tokens: int = 18_106_967
    beta1: float = 0.9
    beta2: float = 0.999
    enc_dtype: str = "fp32"
    aux_loss_coeff: float = 1 / 32.0
    device: str = "cuda:0"
    log_every: int = 25
    save_every: int = 30000
    dec_init_norm: float = 0.08
    wandb_project: str = "model_diffing"
    wandb_entity: str = "goodfire"
    wandb_name: str = None
    wandb_note: str = None
    tiny_mode: bool = False
    D: int = 5120  # d_model
    H: int = 5120 * 8  # better loss at 32
    k: int = 200  # batch top K sparsity - try 50-200, do a hyperparam sweep
    dataset_size: int = 42066

    # --- New for dead neuron tracking ---
    n_batches_to_dead: int = (
        250  # rule of thumb: should fire every 1M tokens. 250 batches * 4k tokens = 1M tokens.
    )
    aux_penalty: float = 1.0
    top_k_aux: int = 20

    def get_changed_params(self) -> dict:
        """Returns a dict of parameters that differ from default values."""
        default_config = CrossCoderConfig()
        changed = {}
        for field in self.model_fields:
            if field in ["wandb_project", "wandb_entity", "device"]:
                continue
            if getattr(self, field) != getattr(default_config, field):
                changed[field] = getattr(self, field)
        return changed

    def get_wandb_name(self) -> str:
        """Generates a W&B run name with an auto-incrementing numeric prefix.

        The counter is persisted in a text file (``wandb_run_id.txt``) inside
        ``CHECKPOINTS_DIR`` so that consecutive script invocations will keep
        increasing the ID.  If the file is missing or unreadable, the counter
        starts from 0.
        """
        if self.wandb_name is not None:
            return self.wandb_name

        # ── Obtain auto-incrementing run id ────────────────────────────────
        id_file = CHECKPOINTS_DIR / "wandb_run_id.txt"
        # Make sure the checkpoints directory exists so we can write the id file.
        id_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            run_id = int(id_file.read_text().strip()) + 1
        except Exception:
            run_id = 0  # First run or unreadable file

        # Persist the updated id for the next run.
        try:
            id_file.write_text(str(run_id))
        except Exception:
            # If writing fails, we still proceed – the id just won't persist.
            pass

        prefix = f"{run_id:05d}"  # Zero-padded for nicer sorting (e.g. 00001_…)

        # ── Build suffix describing changed hyper-parameters ──────────────
        changed = self.get_changed_params()
        if not changed:
            return f"{prefix}_default"

        parts = [f"{key}={value}" for key, value in changed.items()]
        return f"{prefix}_" + "_".join(parts)


# %%
class ActivationsSource:
    """
    Generator for pre-computed activation shards stored as *.mm files with
    accompanying *_metadata.json and an optional top-level dataset.json.
    Loads one shard at a time to keep the memory footprint low and streams
    batches to the trainer on demand.
    """

    def __init__(self, base_act_dir, ft_act_dir, cfg: CrossCoderConfig):
        self.cfg = cfg
        self.base_dir = Path(base_act_dir)
        self.ft_dir = Path(ft_act_dir)

        # Discover shard IDs (e.g. 0, 1, 2 ...).
        self.shard_ids = self._get_shard_ids(self.base_dir)
        if not self.shard_ids:
            raise RuntimeError(f"No shard metadata found under {self.base_dir}")

        # Book-keeping
        self.batch_size = cfg.batch_size
        self.current_shard_idx = -1  # will be advanced to 0 below
        self.pointer = 0  # row pointer inside the current shard

        # ── Estimate global μ/σ from a single shard ───────────────────────
        self._compute_dataset_norm()

        # ── Load first shard ──────────────────────────────────────────────
        self._load_next_acts_shard()

        print(
            f"Loaded shard {self.shard_ids[self.current_shard_idx]} and "
            f"computed normalisation stats at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    # ------------------------------------------------------------------ #
    # Helper utilities                                                   #
    # ------------------------------------------------------------------ #
    def _get_shard_ids(self, directory: Path):
        """Return a sorted list of shard IDs available under *directory*."""
        dataset_meta = directory / "dataset.json"
        try:
            with open(dataset_meta, "r") as f:
                data = json.load(f)
            if data.get("shards") is not None:
                return sorted(int(s) for s in data["shards"])
        except Exception:
            pass  # Fall through to regex discovery

        # Fallback: infer IDs from filenames like "17_metadata.json"
        return sorted(
            int(p.stem.split("_")[0]) for p in directory.glob("*_metadata.json")
        )

    @staticmethod
    def _load_shard(directory: Path, shard_id: int) -> torch.Tensor:
        """Load a single activation shard and return it as a CPU tensor."""
        meta_path = directory / f"{shard_id}_metadata.json"
        act_path = directory / f"{shard_id}_activations.mm"

        with open(meta_path, "r") as f:
            meta = json.load(f)

        shape = tuple(meta["activations"]["shape"])
        dtype_str = meta["activations"]["dtype"]
        np_dtype = {"float16": np.float16, "float32": np.float32}.get(
            dtype_str, np.float16
        )

        mmap_array = np.memmap(act_path, dtype=np_dtype, mode="c", shape=shape)
        xs_BD = torch.from_numpy(mmap_array)  # stays on CPU
        return xs_BD

    # ------------------------------------------------------------------ #
    # Dataset statistics                                                 #
    # ------------------------------------------------------------------ #
    def _compute_dataset_norm(self):
        """Estimate μ/σ from the first shard to avoid loading everything."""
        eps = 1e-6
        shard_id = self.shard_ids[0]

        base_tmp = ActivationsSource._load_shard(self.base_dir, shard_id).float()
        ft_tmp = ActivationsSource._load_shard(self.ft_dir, shard_id).float()

        self.base_mean = base_tmp.mean()
        self.ft_mean = ft_tmp.mean()
        self.base_std = base_tmp.std().clamp(min=eps)
        self.ft_std = ft_tmp.std().clamp(min=eps)

        # Release temporary tensors
        del base_tmp, ft_tmp

    # ------------------------------------------------------------------ #
    # Shard management                                                   #
    # ------------------------------------------------------------------ #
    def _load_next_acts_shard(self):
        """Advance to the next shard and load it into memory."""
        self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shard_ids)
        shard_id = self.shard_ids[self.current_shard_idx]

        base_raw = ActivationsSource._load_shard(self.base_dir, shard_id)
        ft_raw = ActivationsSource._load_shard(self.ft_dir, shard_id)

        # Apply global normalisation
        self.base_acts_ND = ((base_raw.float() - self.base_mean) / self.base_std).to(
            base_raw.dtype
        )
        self.ft_acts_ND = ((ft_raw.float() - self.ft_mean) / self.ft_std).to(
            ft_raw.dtype
        )

        self.current_shard_size = self.base_acts_ND.shape[0]
        self.pointer = 0

        print(
            f"Loaded shard {shard_id} containing {self.current_shard_size:,} activations"
        )

    # ------------------------------------------------------------------ #
    # Public iterator                                                    #
    # ------------------------------------------------------------------ #
    def next(self):
        """
        Return the next batch in shape (B, 2, D). Automatically loads the next
        shard when the current one is exhausted.
        """
        if self.pointer + self.batch_size > self.current_shard_size:
            self._load_next_acts_shard()

        start = self.pointer
        end = self.pointer + self.batch_size
        self.pointer = end

        base_batch = self.base_acts_ND[start:end]
        ft_batch = self.ft_acts_ND[start:end]

        out_B2D = torch.stack([base_batch, ft_batch], dim=0).swapdims(0, 1)
        return out_B2D.to(self.cfg.device, non_blocking=True)


class LossOutput(BaseModel):
    mse: torch.Tensor
    aux_loss: torch.Tensor
    total_loss: torch.Tensor
    explained_variance: torch.Tensor
    explained_variance_A: torch.Tensor
    explained_variance_B: torch.Tensor
    num_dead_features: torch.Tensor

    class Config:
        arbitrary_types_allowed = True


class CrossCoder(nn.Module):
    def __init__(self, cfg: CrossCoderConfig):
        super().__init__()
        self.cfg = cfg
        self.H = self.cfg.H
        self.D = self.cfg.D

        self.dtype = DTYPES[self.cfg.enc_dtype]
        torch.manual_seed(self.cfg.seed)
        # hardcoding n_models to 2
        self.W_dec_H2D = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.H, 2, self.D, dtype=self.dtype)
            )
        )
        # Make norm of W_dec 0.1 for each column, separate per layer
        self.W_dec_H2D.data = (
            self.W_dec_H2D.data
            / self.W_dec_H2D.data.norm(dim=-1, keepdim=True)
            * self.cfg.dec_init_norm
        )
        # Initialise W_enc to be the transpose of W_dec
        self.W_enc_2DH = nn.Parameter(torch.empty(2, self.D, self.H, dtype=self.dtype))
        self.W_enc_2DH.data[:] = self.W_dec_H2D.permute(1, 2, 0).data

        self.b_enc_H = nn.Parameter(torch.zeros((self.H,), dtype=self.dtype))
        self.b_dec_2D = nn.Parameter(torch.zeros((2, self.D), dtype=self.dtype))

        # Dead-feature tracking tensor
        self.num_batches_not_active = torch.zeros(
            (self.H,), dtype=torch.float32, device=self.cfg.device
        )

        self.to(self.cfg.device)
        self.save_dir = None
        self.save_version = 0

    def get_batch_topk_mask(self, x_enc_BH: torch.Tensor) -> torch.Tensor:
        B = x_enc_BH.size(0)
        F_total_batch = x_enc_BH.numel()

        # "Top K" is actually top K*B across the entire flattened batch
        k_total_batch = min(self.cfg.k * B, F_total_batch)

        x_flat = x_enc_BH.reshape(-1)
        _, flat_indices = torch.topk(x_flat, k_total_batch, sorted=False)
        mask_flat = torch.zeros_like(x_flat, dtype=torch.bool)
        mask_flat[flat_indices] = True
        return mask_flat.view_as(x_enc_BH)

    def encode(self, x_B2D):
        x_enc_BH = einops.einsum(
            x_B2D,
            self.W_enc_2DH,
            "batch n_models d_model, n_models d_model d_hidden -> batch d_hidden",
        )
        mask_BH = self.get_batch_topk_mask(x_enc_BH)
        return x_enc_BH * mask_BH + self.b_enc_H

    def get_features(self, x_BD: torch.Tensor, model_idx: int = 1) -> torch.Tensor:
        with torch.no_grad():
            x_enc_BH = einops.einsum(
                x_BD,
                self.W_enc_2DH[model_idx],
                "batch d_model, d_model d_hidden -> batch d_hidden",
            )
            features_BH = x_enc_BH + self.b_enc_H
            return features_BH

    def decode(self, acts_BH):
        acts_dec_B2D = einops.einsum(
            acts_BH,
            self.W_dec_H2D,
            "batch d_hidden, d_hidden n_models d_model -> batch n_models d_model",
        )
        return acts_dec_B2D + self.b_dec_2D

    def forward(self, x_B2D):
        acts_BH = self.encode(x_B2D)
        return self.decode(acts_BH)

    def get_losses(self, x_B2D):
        x_B2D = x_B2D.to(self.dtype)
        acts_BH = self.encode(x_B2D)
        x_reconstruct_B2D = self.decode(acts_BH)

        # Dead-neuron bookkeeping
        self.update_inactive_features(acts_BH)

        diff = x_reconstruct_B2D.float() - x_B2D.float()
        squared_diff = diff.pow(2)
        l2_B = einops.reduce(squared_diff, "batch n_models d_model -> batch", "sum")
        mse_loss = squared_diff.mean()

        total_variance = einops.reduce(
            (x_B2D - x_B2D.mean(0)).pow(2), "batch n_models d_model -> batch", "sum"
        )
        explained_variance = 1 - l2_B / (total_variance + 1e-8)

        per_token_l2_loss_A = (
            (x_reconstruct_B2D[:, 0, :] - x_B2D[:, 0, :]).pow(2).sum(dim=-1).squeeze()
        )
        total_variance_A = (
            (x_B2D[:, 0, :] - x_B2D[:, 0, :].mean(0)).pow(2).sum(-1).squeeze()
        )
        explained_variance_A = 1 - per_token_l2_loss_A / (total_variance_A + 1e-8)

        per_token_l2_loss_B = (
            (x_reconstruct_B2D[:, 1, :] - x_B2D[:, 1, :]).pow(2).sum(dim=-1).squeeze()
        )
        total_variance_B = (
            (x_B2D[:, 1, :] - x_B2D[:, 1, :].mean(0)).pow(2).sum(-1).squeeze()
        )
        explained_variance_B = 1 - per_token_l2_loss_B / (total_variance_B + 1e-8)

        aux_loss = self.get_auxiliary_loss(x_B2D, x_reconstruct_B2D, acts_BH)
        total_loss = mse_loss + self.cfg.aux_loss_coeff * aux_loss

        num_dead = (self.num_batches_not_active > self.cfg.n_batches_to_dead).sum()

        return LossOutput(
            mse=mse_loss,
            aux_loss=aux_loss,
            total_loss=total_loss,
            explained_variance=explained_variance,
            explained_variance_A=explained_variance_A,
            explained_variance_B=explained_variance_B,
            num_dead_features=num_dead,
        )

    @torch.no_grad()
    def update_inactive_features(self, acts_BH):
        """
        Increment counter for units (H) that were completely inactive in this batch (B);
        reset counter for any unit that fired at least once.
        acts_BH: (batch, d_hidden)
        """
        inactive_H = acts_BH.sum(0) == 0  # (H,)
        self.num_batches_not_active += inactive_H.float()
        self.num_batches_not_active[~inactive_H] = 0

    def get_auxiliary_loss(self, x_B2D, x_reconstruct_B2D, acts_BH):
        """
        Encourage currently-dead units to explain the residual.
        x_B2D, x_reconstruct_B2D: (B, 2, D)
        acts_BH: (B, H)
        """
        dead_mask_H = self.num_batches_not_active >= self.cfg.n_batches_to_dead  # (H,)
        n_dead = dead_mask_H.sum().item()
        if n_dead == 0:
            return torch.zeros((), device=x_B2D.device, dtype=x_B2D.dtype)

        # Residual still left over after normal reconstruction
        residual_B2D = x_B2D.float() - x_reconstruct_B2D.float()  # (B, 2, D)

        # Select top-k activations for *dead* units only
        acts_dead_BH = acts_BH[:, dead_mask_H]  # (B, H_dead)
        if acts_dead_BH.shape[1] == 0:
            return torch.zeros((), device=x_B2D.device, dtype=x_B2D.dtype)
        k_aux = min(self.cfg.top_k_aux, acts_dead_BH.shape[1])
        values_BK, idx_BK = torch.topk(acts_dead_BH, k_aux, dim=-1)  # (B, k_aux)
        acts_aux_BH = torch.zeros_like(acts_dead_BH)  # (B, H_dead)
        acts_aux_BH.scatter_(-1, idx_BK, values_BK)

        # Decode with decoder rows that correspond to dead units
        # W_dec_H2D: (H, 2, D) -> W_dec_dead_H2D: (H_dead, 2, D)
        W_dec_dead_H2D = self.W_dec_H2D[dead_mask_H]  # (H_dead, 2, D)
        # acts_aux_BH: (B, H_dead), W_dec_dead_H2D: (H_dead, 2, D)
        # Output: (B, 2, D)
        x_reconstruct_aux_B2D = einops.einsum(
            acts_aux_BH,
            W_dec_dead_H2D,
            "batch h_dead, h_dead n_models d_model -> batch n_models d_model",
        )

        aux_loss = self.cfg.aux_penalty * (
            (x_reconstruct_aux_B2D - residual_B2D).pow(2).mean()
        )
        return aux_loss

    def create_save_dir(self):
        if not CHECKPOINTS_DIR.exists():
            CHECKPOINTS_DIR.mkdir(parents=True)

        version_list = [
            int(file.name.split("_")[1])
            for file in list(CHECKPOINTS_DIR.iterdir())
            if "version" in str(file)
        ]
        if len(version_list):
            version = 1 + max(version_list)
        else:
            version = 0
        self.save_dir = CHECKPOINTS_DIR / f"version_{version}"
        self.save_dir.mkdir(parents=True)

    def save(self):
        if self.save_dir is None:
            self.create_save_dir()
        weight_path = self.save_dir / f"{self.save_version}.pt"
        cfg_path = self.save_dir / f"{self.save_version}_cfg.json"

        torch.save(self.state_dict(), weight_path)
        with open(cfg_path, "w") as f:
            json.dump(self.cfg.model_dump(), f)

        print(f"Saved as checkpoint {self.save_version} in {self.save_dir}")
        self.save_version += 1

    @classmethod
    def load(cls, version: int, checkpoint_version: int, device: str = "cuda:0"):
        save_dir = CHECKPOINTS_DIR / f"version_{version}"
        if not save_dir.exists():
            raise ValueError(f"Save directory {save_dir} does not exist")

        cfg_path = save_dir / f"{str(checkpoint_version)}_cfg.json"
        weight_path = save_dir / f"{str(checkpoint_version)}.pt"

        print(f"Loading {cfg_path}")
        cfg_dict = json.load(open(cfg_path, "r"))
        # pprint.pprint(cfg)
        self = cls(
            cfg=CrossCoderConfig(**{**cfg_dict, **{"device": device}}),
        )
        self.load_state_dict(torch.load(weight_path, map_location=device))
        return self


class Trainer:
    def __init__(
        self,
        cfg: CrossCoderConfig,
        activation_source: ActivationsSource,
        *,
        crosscoder: "CrossCoder" = None,
        step_offset: int = 0,
    ):
        """Trainer wrapper.

        Args:
            cfg: Hyper-parameter configuration (may be modified when resuming).
            activation_source: Iterator over activation batches.
            crosscoder: Optional pre-initialised model (used when resuming training).
            step_offset: Starting global step (for W&B continuity).
        """
        self.cfg = cfg
        # If a pre-trained model was passed in, use it; otherwise instantiate fresh
        self.crosscoder = crosscoder if crosscoder is not None else CrossCoder(cfg)
        self.activation_source = activation_source
        self.total_steps = cfg.num_tokens // cfg.batch_size
        if cfg.tiny_mode:
            self.total_steps = 20
            self.cfg.log_every = 2
            print(
                f"TEST RUN: total steps set to {self.total_steps} and log_every set to {self.cfg.log_every}"
            )

        self.optimizer = torch.optim.Adam(
            self.crosscoder.parameters(),
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self.lr_lambda
        )
        # Start counting from the desired offset so that logged steps are continuous
        self.step_counter = int(step_offset)

        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            notes=cfg.wandb_note,
            config=cfg.model_dump(),
            # Append a suffix when resuming so the run name is distinguishable yet sortable
            name=(
                f"{cfg.get_wandb_name()}_cont"
                if crosscoder is not None
                else cfg.get_wandb_name()
            ),
            # Allow automatic resume if an existing run with the same ID is found
            resume="allow",
        )

    def lr_lambda(self, step):
        if step < 0.8 * self.total_steps:
            return 1.0
        else:
            return 1.0 - (step - 0.8 * self.total_steps) / (0.2 * self.total_steps)

    def step(self):
        acts = self.activation_source.next()
        losses = self.crosscoder.get_losses(acts)

        losses.total_loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        loss_dict = {
            "total_loss": losses.total_loss.item(),
            "mse": losses.mse.item(),
            "aux_loss": losses.aux_loss.item(),
            "lr": self.scheduler.get_last_lr()[0],
            "explained_variance": losses.explained_variance.mean().item(),
            "explained_variance_A": losses.explained_variance_A.mean().item(),
            "explained_variance_B": losses.explained_variance_B.mean().item(),
            "num_dead_features": losses.num_dead_features.item(),
        }
        self.step_counter += 1
        return loss_dict

    def log(self, loss_dict):
        wandb.log(loss_dict, step=self.step_counter)
        print(loss_dict)

    def save(self):
        self.crosscoder.save()

    def train(self):
        try:
            for i in tqdm.trange(self.total_steps):
                loss_dict = self.step()
                if i % self.cfg.log_every == 0:
                    self.log(loss_dict)
                if (i + 1) % self.cfg.save_every == 0:
                    self.save()
        finally:
            self.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny_mode", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--aux_loss_coeff", type=float, default=1 / 32.0)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_note", type=str, default=None)
    parser.add_argument(
        "--model_checkpoint",
        "--model-checkpoint",
        dest="model_checkpoint",
        type=int,
        default=None,
        help="Checkpoint version directory to load (e.g. 25 for checkpoints/version_25). If omitted, starts a fresh run.",
    )
    parser.add_argument(
        "--num_additional_examples",
        "--num-additional-examples",
        dest="num_additional_examples",
        type=int,
        default=None,
        help="Number of extra training examples (rows) to train on after loading the checkpoint. Ignored if --model_checkpoint is not supplied.",
    )
    args = parser.parse_args()

    # ──────────────────────────────────────────────────────────────────────
    # Handle checkpoint loading (if requested) ─────────────────────────────
    # ----------------------------------------------------------------------
    loaded_crosscoder = None
    step_offset = 0

    if args.model_checkpoint is not None:
        # Load the model weights & cfg from the checkpoint directory
        loaded_crosscoder = CrossCoder.load(
            version=args.model_checkpoint, checkpoint_version=0
        )
        cfg = (
            loaded_crosscoder.cfg
        )  # Use the same hyper-params that were saved with the checkpoint

        # Calculate how many steps have already been completed so we can continue
        prev_tokens = cfg.num_tokens
        step_offset = prev_tokens // cfg.batch_size

        # If the user specified extra training examples, overwrite `num_tokens` so
        # that Trainer.total_steps refers only to the *new* examples.
        if args.num_additional_examples is not None:
            cfg.num_tokens = args.num_additional_examples

    else:
        # Fresh run (no checkpoint)
        cfg = CrossCoderConfig()

    if args.wandb_name is not None:
        cfg.wandb_name = args.wandb_name
    if args.wandb_note is not None:
        cfg.wandb_note = args.wandb_note

    # ----------------------------------------------------------------------
    # Apply any CLI-specified overrides on top of whichever cfg we are using
    # ----------------------------------------------------------------------
    cfg.tiny_mode = args.tiny_mode
    if args.k is not None:
        cfg.k = args.k
    if args.H is not None:
        cfg.H = args.H
    if args.aux_loss_coeff is not None:
        cfg.aux_loss_coeff = args.aux_loss_coeff

    activations_dir = Path("/mnt/polished-lake/data/model-diffing/")
    # suffix = 'tiny' if cfg.tiny_mode else 'full'
    suffix = "full"
    base_act_path = activations_dir / f"activations_base_{suffix}.pt" / "partition_0"
    ft_act_path = activations_dir / f"activations_ft_{suffix}.pt" / "partition_0"
    # 489377
    # last one, 488775
    # /mnt/polished-lake/data/model-diffing/activations_base_full.pt/partition_0/36_metadata.json

    activation_source = ActivationsSource(base_act_path, ft_act_path, cfg)
    trainer = Trainer(
        cfg, activation_source, crosscoder=loaded_crosscoder, step_offset=step_offset
    )
    trainer.train()
