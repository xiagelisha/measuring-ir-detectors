"""STC-YOLO C2 — Temporal Mean-Teacher semi-supervised fine-tune (MS3).

EMA teacher generates pseudo-boxes on intra-session unlabeled neighbor frames
(gap in {1, 2, 3} from a labeled f_t); student consumes both labeled f_t loss
and consistency loss between its prediction on f_{t+δ} and the teacher's
pseudo-boxes.

Why this is *NOT* paper3's failed approaches:
  - Not focal loss (N1): we don't reweight Oil class in the supervised term.
  - Not L2-SP (N3): no backbone weight anchor.
  - Not WBF TTA (N7): not a multi-scale fusion of one model's outputs; we use
    a *different* model (teacher EMA) on a *different* frame (t+δ).

Critical constraints:
  - Unlabeled neighbors MUST come from train sessions only. val/test session
    frames forbidden to avoid leakage. Enforced by
    data.unlabeled_neighbor_sampler.
  - Teacher silent for first 5 epochs to avoid cold-start noise.
  - Consistency loss ramped up 0 -> max over `mt_consistency_rampup_epochs`.
  - Confidence-aware weighting: pseudo-box loss weight = teacher's box conf.

Contract:
    Constructed alongside the supervised trainer; the trainer's training step
    is augmented to:
        L_total = L_sup(f_t) + lambda_t * L_consistency(f_{t+δ})
        teacher_params <- ema(teacher_params, student_params, momentum)

Implementation status: MS3 — stub.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn


def letterbox_load(paths, imgsz: int, device) -> torch.Tensor:
    """Load + square-letterbox images to (len, 3, imgsz, imgsz) float[0,1] RGB.

    Teacher and student both consume this exact tensor, so the pseudo-boxes are
    self-consistent regardless of how closely the letterbox matches the training
    dataloader (only internal coherence matters for the consistency term).
    Missing files load as gray canvases (-> teacher finds nothing -> no signal).
    """
    import cv2
    out = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            im = np.full((imgsz, imgsz, 3), 114, np.uint8)
        h0, w0 = im.shape[:2]
        r = min(imgsz / h0, imgsz / w0)
        nh, nw = int(round(h0 * r)), int(round(w0 * r))
        canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
        top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(im, (nw, nh))
        out.append(canvas[:, :, ::-1])   # BGR -> RGB
    arr = np.stack(out).transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray(arr)).float().to(device) / 255.0


def pseudo_to_targets(pseudo_list, imgsz: int, device) -> dict:
    """Convert teacher pseudo-boxes (list of (N,6) xyxy-abs+conf+cls) into an
    ultralytics target dict (batch_idx, cls, bboxes normalized xywh) so the
    student's neighbor forward can be scored by the standard v8DetectionLoss."""
    bidx, clss, boxes = [], [], []
    for i, p in enumerate(pseudo_list):
        if p.shape[0] == 0:
            continue
        xyxy = p[:, :4]
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2 / imgsz
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2 / imgsz
        w = (xyxy[:, 2] - xyxy[:, 0]) / imgsz
        h = (xyxy[:, 3] - xyxy[:, 1]) / imgsz
        boxes.append(torch.stack([cx, cy, w, h], dim=1))
        clss.append(p[:, 5:6])
        bidx.append(torch.full((p.shape[0],), float(i)))
    if boxes:
        return {"batch_idx": torch.cat(bidx).to(device),
                "cls": torch.cat(clss).to(device),
                "bboxes": torch.cat(boxes).to(device)}
    return {"batch_idx": torch.zeros(0, device=device),
            "cls": torch.zeros(0, 1, device=device),
            "bboxes": torch.zeros(0, 4, device=device)}


class EMATeacher:
    """EMA copy of the student that produces pseudo-boxes for unlabeled
    neighbor frames.

    The teacher is a deep copy of the student with grads disabled, updated only
    by exponential moving average of the student weights (never by backprop).
    Buffers (BN running stats) are copied from the student each update — the
    standard Mean-Teacher choice, since EMA-ing BN stats can drift.

    Args:
        student: the student DetectionModel to mirror.
        momentum: EMA decay; teacher = m*teacher + (1-m)*student.
    """

    def __init__(self, student: nn.Module, momentum: float = 0.999) -> None:
        self.momentum = float(momentum)
        self.model = copy.deepcopy(student)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        """EMA-update teacher params from the student; copy buffers."""
        m = self.momentum
        for tp, sp in zip(self.model.parameters(), student.parameters()):
            tp.mul_(m).add_(sp.detach(), alpha=1.0 - m)
        for tb, sb in zip(self.model.buffers(), student.buffers()):
            tb.copy_(sb)

    @torch.no_grad()
    def predict_pseudo(
        self, frames: torch.Tensor, conf_threshold: float = 0.5,
        iou_threshold: float = 0.6,
    ) -> list[torch.Tensor]:
        """Run the teacher on `frames` and return NMS'd pseudo-boxes.

        Returns a list (len = batch) of (N_i, 6) tensors
        [x1, y1, x2, y2, conf, cls]. Frames with no detection yield (0, 6).
        """
        try:
            from ultralytics.utils.nms import non_max_suppression
        except ImportError:  # older ultralytics layout
            from ultralytics.utils.ops import non_max_suppression

        self.model.eval()
        out = self.model(frames)
        preds = out[0] if isinstance(out, (tuple, list)) else out
        dets = non_max_suppression(
            preds, conf_thres=float(conf_threshold), iou_thres=float(iou_threshold))
        return dets


def _pairwise_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between every box in a (M,4 xyxy) and b (N,4 xyxy) -> (M,N).
    Differentiable w.r.t. a (used as student boxes)."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


class TemporalConsistencyLoss(nn.Module):
    """Consistency between student detections on f_{t+δ} and the teacher's
    pseudo-boxes on the same frame.

    For each teacher pseudo-box (treated as the target), greedily match the
    highest-IoU student box and accumulate:
        L = mean_n[ conf_n * ((1 - IoU) + alpha_cls * cls_mismatch_n) ]
    weighting each term by the teacher's confidence (confidence-aware MT).
    Differentiable w.r.t. the student boxes via IoU. Returns 0 when student
    and teacher detections coincide, and 0 if either side is empty (no signal).

    NOTE: D10d uses the pseudo-label integration path (teacher boxes fed through
    the supervised v8DetectionLoss); this module is the standalone consistency
    formulation kept per the C2 contract and used for unit-level checks.
    """

    def __init__(self, alpha_cls: float = 1.0) -> None:
        super().__init__()
        self.alpha_cls = float(alpha_cls)

    def forward(
        self,
        student_dets: list[torch.Tensor],
        teacher_pseudo: list[torch.Tensor],
    ) -> torch.Tensor:
        """student_dets / teacher_pseudo: per-image lists of (k, 6) tensors
        [x1,y1,x2,y2,conf,cls]. Returns a scalar consistency loss."""
        if len(student_dets) != len(teacher_pseudo):
            raise ValueError(
                f"batch mismatch: {len(student_dets)} student vs "
                f"{len(teacher_pseudo)} teacher frames")
        terms: list[torch.Tensor] = []
        device = None
        for s, t in zip(student_dets, teacher_pseudo):
            if s.numel():
                device = s.device
            if s.shape[0] == 0 or t.shape[0] == 0:
                continue
            ious = _pairwise_iou(s[:, :4], t[:, :4])      # (M, N)
            best_iou, best_s = ious.max(dim=0)             # per teacher box
            conf_t = t[:, 4]
            cls_mismatch = (s[best_s, 5] != t[:, 5]).float()
            per_box = conf_t * ((1.0 - best_iou) + self.alpha_cls * cls_mismatch)
            terms.append(per_box.mean())
        if not terms:
            return torch.zeros((), device=device)
        return torch.stack(terms).mean()
