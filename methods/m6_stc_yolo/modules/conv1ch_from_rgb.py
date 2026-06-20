"""STC-YOLO C4 sub-module 2 — Conv1chFromRGB (MS1 D3).

Replaces YOLOv12s's first Conv(3 -> c_out) with a module that:
    (i) inverts the RGB pseudo-color colormap to a 1-channel T_hat in [0, 1]
        via ColormapInverter (D2),
    (ii) applies a Conv(1 -> c_out) whose weight is migrated from the original
        3-channel Conv via the ImageNet-standard ch-mean collapse:
            new_weight[oc, 0, kh, kw] = mean_c (old_weight[oc, c, kh, kw])
        BN running stats / weight / bias / eps / momentum are copied verbatim
        from the original (ultralytics customises BN eps=1e-3 momentum=0.03
        away from PyTorch defaults; failing to copy these silently shifts
        eval-mode outputs).

Why this is *NOT* paper3 N5 (TG-Adapter, which failed):
  - TG-Adapter kept first Conv as 3 -> c_out and *prepended* a 6 -> 3 mixer
    with frozen Sobel features. Mechanism: handcrafted gradient prior
    injected pre-stem.
  - Conv1chFromRGB *replaces* first Conv with 1 -> c_out and removes the
    3-channel pseudo-color redundancy entirely. Mechanism: input-space
    physical reparametrization via LUT^-1, no handcrafted features,
    no extra mixer layer.

Inheritance choice (matches tg_yolo's TGAugmentedConv lesson):
  Conv1chFromRGB extends ultralytics' Conv so that:
    - isinstance(., ultralytics.nn.modules.conv.Conv) checks inside
      DetectionModel still pass,
    - deepcopy survives intact (EMA + save_model rebuild via deepcopy and
      forward hooks would be silently dropped -- paper3 N4 root cause).

Note: D1b finding sets num_colormaps = 0 (single LUT). We keep the parameter
for forward-compat with a hypothetical multi-LUT dataset but emit
NotImplementedError if it's set non-zero.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

from .colormap_inverter import ColormapInverter


class Conv1chFromRGB(Conv):
    """Drop-in replacement for YOLOv12s's first Conv.

    Forward contract:
        x: (B, 3, H, W) RGB in [0, 1]
        -> ColormapInverter -> T_hat (B, 1, H, W) in [0, 1]
        -> parent Conv (1 -> c_out, k, s) -> (B, c_out, H/s, W/s)

    Args:
        c_out: output channel count (32 for YOLOv12s).
        kernel_size: 3 (matches stock YOLOv12s stem).
        stride: 2 (matches stock YOLOv12s stem).
        bn_eps / bn_momentum: copied from the pretrained Conv's BN.
            (ultralytics uses eps=1e-3 momentum=0.03; PyTorch defaults
            would silently shift activations.)
        colormap_inverter_kwargs: forwarded to ColormapInverter.__init__.
            Default is {"mode": "empirical"} (D1b-recommended path).
        num_colormaps: aux one-hot channels for multi-LUT datasets.
            D1b confirmed single LUT; must be 0.
    """

    def __init__(
        self,
        c_out: int = 32,
        kernel_size: int = 3,
        stride: int = 2,
        bn_eps: float = 1e-3,
        bn_momentum: float = 0.03,
        colormap_inverter_kwargs: dict | None = None,
        num_colormaps: int = 0,
    ) -> None:
        if num_colormaps != 0:
            raise NotImplementedError(
                f"num_colormaps={num_colormaps}; only 0 is supported in v1 "
                "(D1b confirmed single LUT across all 22 sessions). Multi-LUT "
                "support is reserved for future datasets."
            )
        # Initialise the parent Conv as a 1-channel-input conv (no bias kept by
        # ultralytics' Conv default; BN follows). Parent inits conv + bn + act.
        super().__init__(1, int(c_out), k=int(kernel_size), s=int(stride))
        # Customise BN to match ultralytics' (eps=1e-3, momentum=0.03).
        self.bn.eps = float(bn_eps)
        self.bn.momentum = float(bn_momentum)
        # Colormap inverter (D2). Owned as submodule so it travels with
        # deepcopy / .to(device) and shows up under model.model[0].colormap_inv.
        self.colormap_inv = ColormapInverter(
            **(colormap_inverter_kwargs if colormap_inverter_kwargs is not None
               else {"mode": "empirical"})
        )

    # ----------------------------------------------------------------------
    # Construction helper: migrate weights from a pretrained 3-ch first Conv.
    # ----------------------------------------------------------------------
    @classmethod
    def build_from_pretrained_conv(
        cls,
        pretrained_conv: Conv,
        num_colormaps: int = 0,
        colormap_inverter_kwargs: dict | None = None,
    ) -> "Conv1chFromRGB":
        """Build a Conv1chFromRGB initialised from a stock 3-ch first Conv.

        Args:
            pretrained_conv: an ultralytics Conv with .conv.weight of shape
                (c_out, 3, k, k) and a BN.
            num_colormaps: must be 0 (D1b finding).
            colormap_inverter_kwargs: forwarded to ColormapInverter.

        Returns:
            A Conv1chFromRGB whose:
              - conv.weight is the ch-mean of the input weight (RGB -> 1 ch)
              - bn.{weight, bias, running_mean, running_var, num_batches_tracked,
                eps, momentum} all copied from input
        """
        if pretrained_conv is None:
            raise ValueError("pretrained_conv must not be None")
        c_out = int(pretrained_conv.conv.out_channels)
        kernel_size = int(pretrained_conv.conv.kernel_size[0])
        stride = int(pretrained_conv.conv.stride[0])
        bn_eps = float(pretrained_conv.bn.eps)
        bn_momentum = float(pretrained_conv.bn.momentum)

        new = cls(
            c_out=c_out,
            kernel_size=kernel_size,
            stride=stride,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
            colormap_inverter_kwargs=colormap_inverter_kwargs,
            num_colormaps=num_colormaps,
        )

        with torch.no_grad():
            # Channel-SUM collapse (NOT mean): (c_out, 3, k, k) -> (c_out, 1, k, k).
            # Rationale: for uniform input R=G=B=x feeding both the original
            # 3-ch Conv and our 1-ch Conv with T_hat=x, sum-migration makes
            # the outputs IDENTICAL. Mean-migration would attenuate by 3x and
            # destroy the pretrained BN's running statistics on first forward,
            # which is the root cause of the catastrophic mAP drop observed
            # in the first D4b run (0.255 vs vanilla 0.610). Sum preserves
            # the integral over channels, which matches BN's expected scale.
            rgb_w = pretrained_conv.conv.weight.data
            if rgb_w.dim() != 4 or rgb_w.shape[1] != 3:
                raise ValueError(
                    f"pretrained_conv weight has shape {tuple(rgb_w.shape)}; "
                    "expected (c_out, 3, k, k)"
                )
            new.conv.weight.data.copy_(rgb_w.sum(dim=1, keepdim=True))
            # Bias (ultralytics' stock Conv has bias=False, but be defensive).
            if (pretrained_conv.conv.bias is not None
                    and new.conv.bias is not None):
                new.conv.bias.data.copy_(pretrained_conv.conv.bias.data)
            # BN running stats + parameters.
            new.bn.weight.data.copy_(pretrained_conv.bn.weight.data)
            new.bn.bias.data.copy_(pretrained_conv.bn.bias.data)
            new.bn.running_mean.data.copy_(pretrained_conv.bn.running_mean.data)
            new.bn.running_var.data.copy_(pretrained_conv.bn.running_var.data)
            new.bn.num_batches_tracked.data.copy_(
                pretrained_conv.bn.num_batches_tracked.data
            )
        return new

    # ----------------------------------------------------------------------
    # Forward (two code paths -- ultralytics rewires .forward to .forward_fuse
    # after fuse_convs() merges BN into conv; we MUST override both.)
    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RGB (B, 3, H, W) -> (B, c_out, H/stride, W/stride). Unfused path."""
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(
                f"expected RGB tensor (B, 3, H, W), got {tuple(x.shape)}"
            )
        t_hat = self.colormap_inv(x)            # (B, 1, H, W) in [0, 1]
        return super().forward(t_hat)            # parent Conv: 1 -> c_out

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Same chain but for the BN-fused inference path.

        ultralytics' Conv.fuse_convs() merges BN into self.conv and assigns
        self.forward = self.forward_fuse at the INSTANCE level. The parent's
        forward_fuse skips our ColormapInverter and feeds raw 3-ch RGB into
        a 1-ch Conv -- root cause of the first D4b run's crash inside
        validator.warmup() -> AutoBackend.forward(). We override here so the
        fused path also goes through ColormapInverter.

        Note: after fuse_convs(), self.bn is deleted; only self.conv (with
        BN folded) and self.act remain.
        """
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(
                f"expected RGB tensor (B, 3, H, W), got {tuple(x.shape)}"
            )
        t_hat = self.colormap_inv(x)
        return self.act(self.conv(t_hat))
