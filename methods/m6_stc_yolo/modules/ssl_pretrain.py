"""STC-YOLO C1 — Cross-Frame Token Consistency SSL pretrain (MS2 D5).

Implements DenseCL-style dense contrastive pretraining for the YOLOv12s
backbone on intra-session (t, t+k) frame pairs (provided by FramePairDataset).

Why this is *NOT* paper3's failed m1 TempoSpread (input-level 3D conv):
  - m1 stacked frames at the INPUT layer via Conv3d with c_in = 3T, creating
    an information bottleneck at c_out = 3 and an ImageNet distribution
    mismatch that couldn't be recovered (see archive/m1_temposspread_paused).
  - C1 operates at the LOSS level: backbone stays 2D, processes single frames
    independently, and a contrastive loss aligns features between the two
    frames of each (t, t+k) pair. No input-level 3D conv. No information
    bottleneck. The frame pair only enters via the loss term.

Algorithm:
  For each (img_t, img_t+k) pair in batch:
      f_t  = backbone(img_t)                            # (B, C, h, w)
      f_tk = backbone(img_t+k)                          # (B, C, h, w)
      z_t  = projection_head(f_t)                       # (B, D, h, w)
      z_tk = projection_head(f_tk)                      # (B, D, h, w)
      For each spatial position p in z_t:
          positive = argmax cosine-sim(z_t[p], z_tk[*])  # auto-aligned
          negatives = all other z_tk[*] in same example
          L += -log( exp(sim_pos / tau) / sum_q exp(sim_q / tau) )

  Argmax-positive selection avoids the spatial alignment problem caused by
  augmentation (h-flip would otherwise mis-align). DenseCL (CVPR 2021) uses
  this exact trick.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class CrossFrameTokenConsistencyLoss(nn.Module):
    """Dense InfoNCE between (z_t, z_t+k) feature maps.

    Each spatial position in z_t finds its positive via cosine-similarity
    argmax over all positions in z_t+k of the same example. Negatives are
    the remaining z_t+k positions in the same example.

    Token subsampling (num_tokens): the dense (B, HW, HW) similarity matrix is
    O(HW^2) and at imgsz=640 the P3 grid is 80x80=6400 tokens -> a 5.2 GB
    matrix per batch that OOMs a 16 GB GPU. DenseCL (CVPR 2021) handles this by
    matching over a *sampled* subset of grid positions rather than the full
    grid. When num_tokens is set and HW > num_tokens, we randomly sample
    num_tokens query positions (from z_t) and num_tokens key positions (from
    z_t+k) per forward call, bounding the matrix to (B, num_tokens, num_tokens).
    Fresh positions each step => full grid coverage over epochs. num_tokens=None
    keeps the exact full-grid behaviour (used by unit tests / small grids).

    Args:
        temperature: InfoNCE temperature scaling (default 0.2 per DenseCL).
        num_tokens: spatial positions to sample per image for the loss; None or
            >= HW means use the full grid.
    """

    def __init__(self, temperature: float = 0.2,
                 num_tokens: int | None = None) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.num_tokens = int(num_tokens) if num_tokens else None

    def forward(self, z_t: torch.Tensor, z_tk: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t:  (B, C, H, W) projection-head output for img_t
            z_tk: (B, C, H, W) projection-head output for img_t+k
        Returns:
            scalar loss
        """
        if z_t.dim() != 4 or z_tk.dim() != 4:
            raise ValueError(
                f"expected (B,C,H,W) features; got {tuple(z_t.shape)} and "
                f"{tuple(z_tk.shape)}"
            )
        if z_t.shape != z_tk.shape:
            raise ValueError(
                f"z_t and z_tk must match shapes; got {tuple(z_t.shape)} vs "
                f"{tuple(z_tk.shape)}"
            )
        B, C, H, W = z_t.shape
        HW = H * W
        # Reshape + L2 normalise along channel dim: (B, HW, C)
        z_t_flat = F.normalize(z_t.view(B, C, HW).transpose(1, 2), dim=2)
        z_tk_flat = F.normalize(z_tk.view(B, C, HW).transpose(1, 2), dim=2)
        # Optionally subsample query (z_t) and key (z_tk) positions to bound the
        # O(HW^2) similarity matrix. Fresh indices each call -> grid coverage
        # accumulates over training steps.
        if self.num_tokens is not None and HW > self.num_tokens:
            q_idx = torch.randperm(HW, device=z_t.device)[: self.num_tokens]
            k_idx = torch.randperm(HW, device=z_t.device)[: self.num_tokens]
            z_t_flat = z_t_flat[:, q_idx, :]   # (B, Nq, C)
            z_tk_flat = z_tk_flat[:, k_idx, :]  # (B, Nk, C)
        Nq = z_t_flat.shape[1]
        Nk = z_tk_flat.shape[1]
        # Per-batch cosine sim matrix: (B, Nq, Nk)
        sim = torch.bmm(z_t_flat, z_tk_flat.transpose(1, 2))
        # Positive = argmax over z_tk keys for each query position in z_t.
        positive_idx = sim.argmax(dim=2)  # (B, Nq)
        # InfoNCE: logits = sim / tau, target = positive_idx
        logits = sim / self.temperature
        loss = F.cross_entropy(
            logits.view(B * Nq, Nk),
            positive_idx.view(B * Nq),
        )
        return loss


class SSLProjectionHead(nn.Module):
    """1x1 Conv MLP projection head used by both arms of the SSL pair.

    Following SimCLR / DenseCL: 2-layer conv with BN + ReLU in between,
    no normalisation at the output (LayerNorm is applied via F.normalize
    inside the loss).
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 256,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=True),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(feat)


class BackboneFeatureExtractor:
    """Wraps a YOLO model with a forward hook that captures the output of a
    specified backbone layer.

    Args:
        yolo_model: an ultralytics DetectionModel (`YOLO(...).model`).
        layer_idx: index of the layer whose output to capture. For YOLOv12s,
            P3 (stride 8) typically lives around index 4-5; verify via the
            yaml. Use the helper `probe_layer_channels` to confirm.
    """

    def __init__(self, yolo_model, layer_idx: int = 4) -> None:
        self.yolo_model = yolo_model
        self.layer_idx = int(layer_idx)
        self._captured: torch.Tensor | None = None
        self._hook = yolo_model.model[self.layer_idx].register_forward_hook(
            self._capture
        )

    def _capture(self, module, input, output) -> None:
        self._captured = output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Full forward; we only need the captured intermediate. If the head
        # forward errors out after capture (e.g. due to mode quirks), the
        # captured feature is still usable.
        self._captured = None
        try:
            _ = self.yolo_model(x)
        except Exception:
            if self._captured is None:
                raise
        if self._captured is None:
            raise RuntimeError(
                f"BackboneFeatureExtractor failed to capture output of layer "
                f"{self.layer_idx}; check layer index for this model"
            )
        feat = self._captured
        self._captured = None
        return feat

    def remove(self) -> None:
        self._hook.remove()


class SSLPretrainer:
    """C1 self-supervised pretraining orchestrator.

    Drives the dense contrastive pretraining of a YOLOv12s backbone on a
    FramePairDataset. Saves the full YOLO model state_dict (so D7 can
    selectively load backbone weights) and a training-history JSON.

    Args:
        yolo_model: an ultralytics DetectionModel (typically YOLO('yolo12s.pt').model).
        projection_head: SSLProjectionHead instance.
        dataset: FramePairDataset (or any (t, tk, k) yielding Dataset).
        cfg: dict with epochs / batch / lr0 / weight_decay / workers /
             temperature / feature_layer_idx / output_dir / device.
    """

    def __init__(
        self,
        yolo_model,
        projection_head: SSLProjectionHead,
        dataset,
        cfg: dict,
    ) -> None:
        self.yolo_model = yolo_model
        self.projection_head = projection_head
        self.dataset = dataset
        self.cfg = cfg

    def _save_resume_ckpt(self, ckpt_path: Path, opt, history: list[dict],
                          epoch: int, layer_idx: int, epochs: int) -> None:
        """Atomically write the per-epoch resume checkpoint.

        Written to a `.tmp` sibling then os.replace'd so a crash mid-save can
        never corrupt the existing `last.pt` (os.replace is atomic on the same
        filesystem on both POSIX and Windows).
        """
        tmp = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
        torch.save({
            "epoch": epoch,                       # last COMPLETED epoch (0-based)
            "model_state": self.yolo_model.state_dict(),
            "proj_state": self.projection_head.state_dict(),
            "opt_state": opt.state_dict(),
            "history": history,
            "feature_layer_idx": layer_idx,
            "epochs": epochs,
        }, tmp)
        os.replace(tmp, ckpt_path)

    def fit(self) -> str:
        """Run SSL pretrain end-to-end. Returns path to saved state_dict.

        Crash-resilient: a `last.pt` resume checkpoint is written after every
        epoch (atomically). If `last.pt` is found in out_dir at startup, training
        resumes from the next epoch — a crash costs at most one epoch, not the
        whole multi-hour run.
        """
        device = str(self.cfg.get("device", "cuda:0"
                                    if torch.cuda.is_available() else "cpu"))
        layer_idx = int(self.cfg.get("feature_layer_idx", 4))
        epochs = int(self.cfg.get("epochs", 80))
        batch = int(self.cfg.get("batch", 32))
        lr0 = float(self.cfg.get("lr0", 1e-4))
        wd = float(self.cfg.get("weight_decay", 1e-4))
        workers = int(self.cfg.get("workers", 4))
        temperature = float(self.cfg.get("temperature", 0.2))
        num_tokens = self.cfg.get("num_tokens", None)
        num_tokens = int(num_tokens) if num_tokens else None
        out_dir = Path(self.cfg.get(
            "output_dir",
            Path(__file__).resolve().parents[3]
            / "results_v2" / "methods" / "stc_yolo" / "_ms2",
        ))
        out_dir.mkdir(parents=True, exist_ok=True)
        resume_path = out_dir / "last.pt"

        self.yolo_model.to(device).train()
        self.projection_head.to(device).train()

        extractor = BackboneFeatureExtractor(self.yolo_model, layer_idx=layer_idx)
        loss_fn = CrossFrameTokenConsistencyLoss(
            temperature=temperature, num_tokens=num_tokens)

        params = list(self.yolo_model.parameters()) + list(
            self.projection_head.parameters()
        )
        opt = torch.optim.AdamW(params, lr=lr0, weight_decay=wd)

        # ---- resume from last.pt if present --------------------------------
        history: list[dict] = []
        start_epoch = 0
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            ck_layer = int(ckpt.get("feature_layer_idx", layer_idx))
            if ck_layer != layer_idx:
                raise RuntimeError(
                    f"resume checkpoint {resume_path} was trained with "
                    f"feature_layer_idx={ck_layer} but config requests "
                    f"{layer_idx}; refusing to resume into an incompatible run. "
                    f"Delete last.pt to start fresh."
                )
            self.yolo_model.load_state_dict(ckpt["model_state"])
            self.projection_head.load_state_dict(ckpt["proj_state"])
            opt.load_state_dict(ckpt["opt_state"])
            history = list(ckpt.get("history", []))
            start_epoch = int(ckpt["epoch"]) + 1
            print(f"[ssl] RESUME from {resume_path}: completed {start_epoch} "
                  f"epoch(s), continuing to {epochs}", flush=True)
            if start_epoch >= epochs:
                print(f"[ssl] checkpoint already at/over target {epochs} epochs; "
                      f"skipping training, finalising outputs", flush=True)

        loader = DataLoader(
            self.dataset,
            batch_size=batch,
            shuffle=True,
            num_workers=workers,
            drop_last=True,
            pin_memory=(device.startswith("cuda")),
        )

        for epoch in range(start_epoch, epochs):
            epoch_loss = 0.0
            n_batches = 0
            for img_t, img_tk, k in loader:
                img_t = img_t.to(device, non_blocking=True)
                img_tk = img_tk.to(device, non_blocking=True)

                feat_t = extractor(img_t)
                feat_tk = extractor(img_tk)
                z_t = self.projection_head(feat_t)
                z_tk = self.projection_head(feat_tk)
                loss = loss_fn(z_t, z_tk)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                epoch_loss += float(loss.item())
                n_batches += 1
            avg = epoch_loss / max(n_batches, 1)
            history.append({"epoch": epoch, "loss": avg, "n_batches": n_batches})
            print(f"[ssl] epoch {epoch+1}/{epochs}  loss={avg:.4f}  "
                  f"n_batches={n_batches}", flush=True)
            # Per-epoch resume checkpoint (atomic). Crash here costs 1 epoch max.
            self._save_resume_ckpt(resume_path, opt, history, epoch,
                                   layer_idx, epochs)

        # Save full YOLO state_dict (D7 will selectively load backbone keys).
        bb_path = out_dir / "pretrained_backbone_stc.pt"
        torch.save({
            "state_dict": self.yolo_model.state_dict(),
            "feature_layer_idx": layer_idx,
            "epochs": epochs,
            "final_loss": history[-1]["loss"] if history else None,
        }, bb_path)

        summary = {
            "epochs": epochs,
            "batch": batch,
            "lr0": lr0,
            "weight_decay": wd,
            "temperature": temperature,
            "num_tokens": num_tokens,
            "feature_layer_idx": layer_idx,
            "n_pairs": len(self.dataset),
            "final_loss": history[-1]["loss"] if history else None,
            "history": history,
            "backbone_path": str(bb_path),
        }
        summary_path = out_dir / "ssl_pretrain_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[ssl] saved backbone -> {bb_path}", flush=True)
        print(f"[ssl] saved summary  -> {summary_path}", flush=True)

        # Final outputs are durable now; drop the resume checkpoint so a future
        # fresh rerun (after orchestrator reset) does not silently resume.
        if resume_path.exists():
            resume_path.unlink()

        extractor.remove()
        return str(bb_path)


def probe_layer_channels(yolo_model, layer_idx: int, imgsz: int = 640) -> int:
    """Return the channel dimension of the named backbone layer's output."""
    extractor = BackboneFeatureExtractor(yolo_model, layer_idx=layer_idx)
    yolo_model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=next(yolo_model.parameters()).device)
        feat = extractor(dummy)
    c = int(feat.shape[1])
    extractor.remove()
    return c
