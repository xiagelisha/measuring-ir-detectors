"""STC-YOLO model factory + trainer subclass for C4 (colormap inversion).

Architecture (MS1 D3b):
  RGB (B, 3, H, W) -> Conv1chFromRGB (model.model[0])
                       |-> internally:
                       |     ColormapInverter (empirical LUT, B,3,H,W -> B,1,H,W)
                       |     parent ultralytics Conv (1 -> 32, k=3, s=2)
                       |
                       -> rest of YOLOv12s backbone / neck / head (UNCHANGED)

CRITICAL design lesson inherited from tg_yolo (paper3 N4 root cause):
    ultralytics' yolo.train() internally calls
        trainer.get_model(weights=self.model, cfg=self.model.yaml)
    which REBUILDS a vanilla DetectionModel from the yaml + intersect_dicts
    loading. Any custom swap we did to model.model[0] in build_stc_yolo is
    SILENTLY DROPPED at this rebuild step (the saved best.pt comes out as a
    vanilla YOLOv12s with no C4 wiring).

Fix: we pass a custom DetectionTrainer subclass (make_stc_trainer) whose
get_model() re-applies the swap immediately after the rebuild. EMA then
constructs from the corrected model and best.pt actually persists C4.

As MS2/MS3 deliver:
    MS2:    optionally load SSL-pretrained backbone weights via
            cfg.stc.ssl_pretrained_backbone.
    MS3:    optionally inject Session-FiLM into the detect head when
            cfg.stc.enable_session_film is true.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel

from .modules.conv1ch_from_rgb import Conv1chFromRGB


class _STCDetectionModel(DetectionModel):
    """DetectionModel whose training loss adds a stashed Mean-Teacher
    consistency term (`_mt_aux_loss`, set per-batch by the trainer).

    Module-level subclass (pickle-safe; promoted in-place via `__class__=`).
    The aux term is consumed (cleared) on each forward so gradient accumulation
    can't double-count it, and is None on every non-training path (eval/predict)
    so the model reduces exactly to a vanilla DetectionModel there. The neighbor
    consistency forward calls `DetectionModel.loss` directly (not this override)
    to avoid recursion."""

    def loss(self, batch, preds=None):
        base_loss, base_items = DetectionModel.loss(self, batch, preds)
        aux = getattr(self, "_mt_aux_loss", None)
        if aux is not None:
            self._mt_aux_loss = None      # consume once
            return base_loss + aux, base_items
        return base_loss, base_items


class SessionFiLMDetect(Detect):
    """Detect head whose cls branch input is FiLM-modulated per image.

    Module-level subclass (NOT an instance monkey-patch) so best.pt pickles and
    reloads cleanly — same lesson as Conv1chFromRGB for C4. A plain Detect is
    promoted to this class in-place via `detect.__class__ = SessionFiLMDetect`
    by `_inject_session_film`, which also attaches the `film_gen` submodule and
    `_film_levels`. Per-image metadata is stashed on `self._film_metadata` by
    the train/val/eval threading hooks; when absent, FiLM is the identity.
    """

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return dict()
        from .modules.session_film import apply_film
        bs = x[0].shape[0]
        boxes = torch.cat(
            [box_head[i](x[i]).view(bs, 4 * self.reg_max, -1)
             for i in range(self.nl)], dim=-1)
        meta = getattr(self, "_film_metadata", None)
        gen = getattr(self, "film_gen", None)
        if meta is not None and gen is not None and meta.shape[0] == bs:
            params = gen(meta)
            x_cls = [apply_film(x[i], *params[self._film_levels[i]])
                     for i in range(self.nl)]
        else:
            x_cls = x   # no metadata -> identity (graceful)
        scores = torch.cat(
            [cls_head[i](x_cls[i]).view(bs, self.nc, -1)
             for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)


# --- C3 Session-FiLM metadata cache (shared, lazy) -------------------------
_FILM_META_CACHE: dict | None = None


def _film_metadata_for_imfiles(im_files, device) -> "torch.Tensor":
    """Map a batch of image paths -> (B, METADATA_DIM) encoded metadata tensor.

    Looks up each image by stem in metadata.csv (cached). Files without
    metadata get a zero vector (FiLM then behaves near-identity for that row).
    Used by the train/val/eval callbacks to stash metadata on the Detect head
    so FiLM is applied consistently on every forward path.
    """
    global _FILM_META_CACHE
    from .data.session_metadata import (
        load_metadata, encode_metadata, METADATA_DIM)
    if _FILM_META_CACHE is None:
        metas = load_metadata()
        _FILM_META_CACHE = {name: encode_metadata(m) for name, m in metas.items()}
    vecs = []
    for f in im_files:
        stem = Path(str(f)).stem
        vec = _FILM_META_CACHE.get(stem)
        vecs.append(vec if vec is not None else torch.zeros(8))
    return torch.stack(vecs).to(device)


def _inject_session_film(model, metadata_dim: int = 8) -> None:
    """Attach a SessionFiLMGenerator to the Detect head and wrap its
    forward_head so the cls branch input is FiLM-modulated per image.

    The generator is registered as a submodule of Detect (`detect.film_gen`)
    so it trains, moves with .to(device), and is saved in best.pt. Per-level
    channel counts are read from the box head's first conv. Idempotent.
    """
    from .modules.session_film import SessionFiLMGenerator

    detect = model.model[-1]
    if getattr(detect, "_film_injected", False):
        return
    if type(detect) is not Detect:
        raise TypeError(
            f"Session-FiLM injection expects a plain ultralytics Detect head; "
            f"got {type(detect).__name__}. Add a module-level subclass for it.")
    nl = int(detect.nl)
    level_names = ["P3", "P4", "P5", "P6", "P7"][:nl]
    ch = {level_names[i]: int(detect.cv2[i][0].conv.in_channels) for i in range(nl)}

    detect.film_gen = SessionFiLMGenerator(
        metadata_dim=metadata_dim, feature_dim=ch,
        inject_levels=tuple(level_names))
    detect._film_levels = level_names
    detect._film_metadata = None          # set per-batch by the threading hook
    # Promote to the FiLM subclass (pickle-safe; survives best.pt round-trip).
    detect.__class__ = SessionFiLMDetect
    detect._film_injected = True


def _find_film_detect(model):
    """Return the FiLM-injected Detect module inside a (possibly wrapped)
    DetectionModel, or None."""
    inner = getattr(model, "model", model)
    # inner may be the layer ModuleList (DetectionModel.model) or the model.
    seq = getattr(inner, "model", inner)
    try:
        last = seq[-1]
    except (TypeError, KeyError, IndexError):
        last = None
    if last is not None and getattr(last, "_film_injected", False):
        return last
    for m in model.modules():
        if getattr(m, "_film_injected", False):
            return m
    return None


def set_film_metadata(model, im_files) -> bool:
    """Stash per-image metadata on the model's FiLM Detect for the next
    forward. Returns True if a FiLM head was found and set."""
    detect = _find_film_detect(model)
    if detect is None:
        return False
    try:
        device = next(detect.film_gen.parameters()).device
    except StopIteration:
        device = "cpu"
    detect._film_metadata = _film_metadata_for_imfiles(im_files, device)
    return True


def make_film_validator():
    """DetectionValidator subclass that stashes Session-FiLM metadata from each
    batch's im_file onto the model before inference. Used for both in-training
    validation and the final test eval so FiLM is applied on every path."""
    from ultralytics.models.yolo.detect import DetectionValidator

    class _FiLMDetectionValidator(DetectionValidator):
        def preprocess(self, batch):
            batch = super().preprocess(batch)
            model = getattr(self, "model", None)
            if model is not None and "im_file" in batch:
                set_film_metadata(model, batch["im_file"])
            return batch

    return _FiLMDetectionValidator


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRETRAINED = _PROJECT_ROOT / "yolo12s.pt"


def _swap_first_conv_to_c4(
    model,
    colormap_inverter_kwargs: dict | None = None,
) -> None:
    """In-place swap `model.model[0]` to a Conv1chFromRGB.

    Idempotent: if model.model[0] is already Conv1chFromRGB, no-op.

    The swap preserves ultralytics' parse_model attributes (f / i / type)
    that DetectionModel._predict_once reads on every forward.
    """
    orig = model.model[0]
    if isinstance(orig, Conv1chFromRGB):
        return
    if not isinstance(orig, Conv):
        raise TypeError(
            f"expected model.model[0] to be an ultralytics Conv, "
            f"got {type(orig).__name__}"
        )
    new = Conv1chFromRGB.build_from_pretrained_conv(
        orig, colormap_inverter_kwargs=colormap_inverter_kwargs
    )
    # Preserve ultralytics' parse_model annotations -- without these, the
    # DetectionModel._predict_once forward triggers AttributeError.
    for attr in ("f", "i", "type"):
        if hasattr(orig, attr):
            setattr(new, attr, getattr(orig, attr))
    model.model[0] = new


def _load_ssl_backbone(model, ssl_pretrained_backbone: str | Path) -> tuple[int, int]:
    """Load SSL-pretrained weights (MS2 D6) into a vanilla DetectionModel.

    The D6 checkpoint stores the FULL yolo12s state_dict with a **3-channel**
    stem (SSL ran on plain yolo12s, no C4 swap). This MUST be applied BEFORE
    `_swap_first_conv_to_c4` so the SSL-adapted 3-ch stem is what the ch-sum
    migration folds into the 1-ch Conv1chFromRGB — applying it after the swap
    would silently skip the (shape-mismatched) stem and lose the SSL adaptation
    of the very first conv.

    Weights are intersected by name+shape (mirrors ultralytics' intersect_dicts)
    so an accidental post-swap call skips the 1-ch stem instead of crashing.

    Returns:
        (n_matched, n_target) for logging.

    Raises:
        FileNotFoundError: checkpoint path missing.
        RuntimeError: implausibly small overlap (guards a wrong-arch file).
    """
    path = Path(ssl_pretrained_backbone)
    if not path.is_file():
        raise FileNotFoundError(
            f"ssl_pretrained_backbone not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    tgt = model.state_dict()
    matched = {k: v for k, v in sd.items()
               if k in tgt and tgt[k].shape == v.shape}
    if len(matched) < 0.5 * len(tgt):
        raise RuntimeError(
            f"SSL backbone {path} matched only {len(matched)}/{len(tgt)} "
            f"params by name+shape; wrong/corrupt checkpoint?")
    model.load_state_dict(matched, strict=False)
    return len(matched), len(tgt)


def _wrap_model_args(model) -> None:
    """Wrap model.args dict into SimpleNamespace so v8DetectionLoss can use
    attribute access (model.args.box / .cls / .dfl)."""
    if isinstance(getattr(model, "args", None), dict):
        d = model.args
        d.setdefault("box", 7.5)
        d.setdefault("cls", 0.5)
        d.setdefault("dfl", 1.5)
        model.args = SimpleNamespace(**d)
    elif not hasattr(model, "args") or model.args is None:
        model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)


def build_stc_yolo(
    pretrained: str | Path = DEFAULT_PRETRAINED,
    enable_colormap_inversion: bool = False,
    ssl_pretrained_backbone: str | Path | None = None,
    enable_session_film: bool = False,
    enable_mean_teacher: bool = False,
    colormap_inverter_kwargs: dict | None = None,
) -> "YOLO":
    """Build a STC-YOLO model.

    Args:
        pretrained: path to yolo12s.pt (or a saved best.pt for resume; the
            swap is idempotent so reload paths just work).
        enable_colormap_inversion: D3b -- swap model.model[0] to
            Conv1chFromRGB. Also must be propagated to make_stc_trainer().
        ssl_pretrained_backbone: MS2 -- path to SSL-pretrained backbone.
        enable_session_film: MS3 -- inject Session-FiLM into detect head.
        enable_mean_teacher: MS3 -- consumed by trainer, not build.
        colormap_inverter_kwargs: forwarded to Conv1chFromRGB's
            ColormapInverter. Default {"mode": "empirical"}.

    Returns:
        ultralytics YOLO wrapper. `.train()`, `.val()`, `.predict()` work
        standard. WITH enable_colormap_inversion=True, you MUST also pass
        the trainer from make_stc_trainer(...) to yolo.train(), otherwise
        the swap is silently dropped during ultralytics' internal rebuild.
    """
    yolo = YOLO(str(pretrained))

    # MS2 (D7): load SSL-pretrained backbone BEFORE the C4 swap. The SSL ckpt
    # has a 3-ch stem; loading first lets _swap_first_conv_to_c4 migrate the
    # SSL-adapted stem (ch-sum) into the 1-ch Conv1chFromRGB. Order matters.
    if ssl_pretrained_backbone is not None:
        _load_ssl_backbone(yolo.model, ssl_pretrained_backbone)

    if enable_colormap_inversion:
        _swap_first_conv_to_c4(yolo.model,
                                colormap_inverter_kwargs=colormap_inverter_kwargs)

    if enable_session_film:
        _inject_session_film(yolo.model)

    # YOLO('xxx.pt') ships with all params frozen (inference default).
    # Unfreeze so yolo.train() will actually update weights. ColormapInverter
    # buffers (lut_rgb / t) are buffers, not params, so they're unaffected.
    for p in yolo.model.parameters():
        p.requires_grad = True

    _wrap_model_args(yolo.model)
    return yolo


def make_stc_trainer(
    enable_colormap_inversion: bool = False,
    ssl_pretrained_backbone: str | Path | None = None,
    enable_session_film: bool = False,
    enable_mean_teacher: bool = False,
    mt_cfg: dict | None = None,
    colormap_inverter_kwargs: dict | None = None,
):
    """Factory returning a DetectionTrainer subclass that survives ultralytics'
    internal rebuild.

    PASS the returned class as ``trainer=`` to ``yolo.train(...)``. Without
    this, ultralytics rebuilds model.model[0] from the yaml and drops our
    Conv1chFromRGB swap silently (paper3 N4 root cause; tg_yolo's exact bug).

    The subclass overrides get_model so that immediately after the rebuild
    (which still has model.model[0] = vanilla Conv with pretrained weights
    loaded via intersect_dicts), we re-swap to Conv1chFromRGB. The new
    Conv1chFromRGB's ch-mean migration draws from the *intersect_dicts-loaded*
    pretrained weights, so we never lose the ImageNet stem.

    C2 (D10d): when enable_mean_teacher=True, the subclass promotes the model to
    `_STCDetectionModel`, maintains an EMATeacher (EMA-updated after each
    optimizer step), and per batch samples one unlabeled train neighbor per
    labeled frame, has the teacher pseudo-label it, and adds the student's
    detection loss on those pseudo-labels (rampup-weighted) as a temporal
    consistency term. Teacher stays silent for `teacher_warmup_epochs`.
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    mtc = dict(mt_cfg or {})

    class _STCDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            # MS2 (D7): re-load SSL backbone after the rebuild (ultralytics'
            # intersect_dicts would otherwise drop it, esp. the C4 1-ch stem
            # mismatch). Load BEFORE the swap so the SSL 3-ch stem is migrated.
            if ssl_pretrained_backbone is not None:
                _load_ssl_backbone(model, ssl_pretrained_backbone)
            if enable_colormap_inversion:
                _swap_first_conv_to_c4(
                    model, colormap_inverter_kwargs=colormap_inverter_kwargs
                )
            # C3 (D11): re-inject Session-FiLM after the rebuild (the wrapped
            # forward_head + film_gen submodule are dropped by the rebuild).
            if enable_session_film:
                _inject_session_film(model)
            if enable_mean_teacher and type(model) is DetectionModel:
                # C2: promote so model.loss adds the stashed consistency term.
                model.__class__ = _STCDetectionModel
                model._mt_aux_loss = None
            for p in model.parameters():
                p.requires_grad = True
            _wrap_model_args(model)
            return model

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            # C3: stash FiLM metadata for the training forward.
            if enable_session_film and "im_file" in batch:
                set_film_metadata(self.model, batch["im_file"])
            # C2: compute + stash the Mean-Teacher consistency term so the
            # model's loss override adds it to the supervised loss this step.
            if enable_mean_teacher:
                self.model._mt_aux_loss = self._mt_compute_aux(batch)
            return batch

        def get_validator(self):
            v = super().get_validator()
            if enable_session_film:
                # Re-instantiate as the FiLM-aware validator (same ctor args).
                FiLMVal = make_film_validator()
                v = FiLMVal(self.test_loader, save_dir=self.save_dir,
                            args=v.args, _callbacks=self.callbacks)
            return v

        def _maybe_build_teacher(self):
            if not enable_mean_teacher:
                return
            if getattr(self, "_ema_teacher", None) is None:
                from .modules.mean_teacher import EMATeacher
                self._ema_teacher = EMATeacher(
                    self.model, momentum=float(mtc.get("mt_ema_momentum", 0.999)))

        def _mt_lazy_init(self):
            """Build the neighbor sampler + stem->path map from the train set."""
            if getattr(self, "_mt_sampler", None) is not None:
                return
            from .data.unlabeled_neighbor_sampler import UnlabeledNeighborSampler
            im_files = list(getattr(self.train_loader.dataset, "im_files", []))
            self._mt_path_map = {Path(p).stem: p for p in im_files}
            train_stems = set(self._mt_path_map.keys())
            self._mt_sampler = UnlabeledNeighborSampler(
                train_stems, max_gap=int(mtc.get("mt_neighbor_gap_max", 3)))

        def _mt_compute_aux(self, batch):
            """Sample one train neighbor per labeled frame, teacher-pseudo-label
            it, and return the rampup-weighted student detection loss on those
            pseudo-labels (the temporal consistency term). None during warmup."""
            from .modules.mean_teacher import letterbox_load, pseudo_to_targets
            epoch = int(getattr(self, "epoch", 0))
            warmup = int(mtc.get("teacher_warmup_epochs", 5))
            if epoch < warmup or getattr(self, "_ema_teacher", None) is None:
                return None
            im_files = batch.get("im_file")
            if not im_files:
                return None
            self._mt_lazy_init()
            paths = []
            for f in im_files:
                try:
                    nb, _ = self._mt_sampler.sample(Path(str(f)).stem)
                    paths.append(self._mt_path_map.get(nb, ""))
                except ValueError:
                    paths.append("")     # no neighbor -> gray canvas, no signal
            imgsz = int(batch["img"].shape[-1])
            device = batch["img"].device
            neighbor_imgs = letterbox_load(paths, imgsz, device)
            conf = float(mtc.get("mt_pseudo_conf_threshold", 0.5))
            pseudo = self._ema_teacher.predict_pseudo(neighbor_imgs, conf)
            targets = pseudo_to_targets(pseudo, imgsz, device)
            nb_batch = {"img": neighbor_imgs, **targets}
            # Parent loss (NOT the override) to avoid recursion; student grad flows.
            aux_loss, _ = DetectionModel.loss(self.model, nb_batch)
            rampup = max(1, int(mtc.get("mt_consistency_rampup_epochs", 10)))
            wmax = float(mtc.get("mt_consistency_weight_max", 1.0))
            w = min(1.0, (epoch - warmup + 1) / rampup) * wmax
            return w * aux_loss

        def optimizer_step(self):
            super().optimizer_step()
            # C2: EMA-update the teacher from the just-stepped student.
            if enable_mean_teacher:
                self._maybe_build_teacher()
                self._ema_teacher.update(self.model)

    return _STCDetectionTrainer


# Backwards-compat alias.
build = build_stc_yolo
