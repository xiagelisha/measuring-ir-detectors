"""STC-YOLO MS1 skeleton sanity tests.

Verifies:
  1. All modules import successfully (no syntax / circular issues).
  2. Stub modules raise NotImplementedError when called (proves contract is
     in place but body deferred).
  3. session_metadata.py functions on the real metadata.csv (2593 entries,
     22 sessions). This module is the only functional part of MS1 skeleton.

Run with (stdlib unittest, oilleak env has no pytest):
    conda activate oilleak
    python -m unittest methods.m6_stc_yolo.tests.test_skeleton -v
    # or:
    python methods/m6_stc_yolo/tests/test_skeleton.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class TestImports(unittest.TestCase):
    """Section 1 — Import sanity (no syntax / circular issues)."""

    def test_import_modules(self):
        from methods.m6_stc_yolo.modules import (
            colormap_inverter,
            conv1ch_from_rgb,
            ssl_pretrain,
            mean_teacher,
            session_film,
        )
        self.assertTrue(hasattr(colormap_inverter, "ColormapInverter"))
        self.assertTrue(hasattr(conv1ch_from_rgb, "Conv1chFromRGB"))
        self.assertTrue(hasattr(ssl_pretrain, "SSLPretrainer"))
        self.assertTrue(hasattr(mean_teacher, "EMATeacher"))
        self.assertTrue(hasattr(session_film, "SessionFiLMGenerator"))

    def test_import_data_helpers(self):
        from methods.m6_stc_yolo.data import (
            session_metadata,
            frame_pair_dataset,
            unlabeled_neighbor_sampler,
        )
        self.assertTrue(hasattr(session_metadata, "load_metadata"))
        self.assertTrue(hasattr(session_metadata, "encode_metadata"))
        self.assertTrue(hasattr(session_metadata, "build_session_segments"))
        self.assertTrue(hasattr(frame_pair_dataset, "FramePairDataset"))
        self.assertTrue(hasattr(unlabeled_neighbor_sampler, "UnlabeledNeighborSampler"))

    def test_import_top_level(self):
        from methods.m6_stc_yolo import model as stc_model
        self.assertTrue(hasattr(stc_model, "build_stc_yolo"))


class TestStubContracts(unittest.TestCase):
    """Section 2 — Stub modules raise NotImplementedError on call."""

    # D5 implemented 2026-05-29 -- ssl_pretrain stub test superseded by
    # TestSSLPretrain. D11 implemented 2026-05-30 -- session_film stub test
    # superseded by TestSessionFiLM. No remaining stub-raises contracts.
    pass


class TestSessionMetadata(unittest.TestCase):
    """Section 3 — session_metadata.py is the one functional MS1 module."""

    def test_load_metadata_real_csv(self):
        from methods.m6_stc_yolo.data.session_metadata import (
            load_metadata,
            DEFAULT_METADATA_CSV,
            METADATA_DIM,
        )
        metas = load_metadata(DEFAULT_METADATA_CSV)
        self.assertEqual(
            len(metas), 2593, f"expected 2593 entries, got {len(metas)}"
        )
        sample = next(iter(metas.values()))
        self.assertTrue(sample.session_id)
        self.assertGreaterEqual(sample.frame_idx, 0)
        self.assertIsNotNone(sample.temperature)
        self.assertEqual(METADATA_DIM, 8)
        # day_night is intentionally a raw string (not int) — see module docstring.
        self.assertIsInstance(sample.day_night_raw, str)

    def test_encode_metadata_shape_and_range(self):
        import torch
        from methods.m6_stc_yolo.data.session_metadata import (
            load_metadata,
            encode_metadata,
            METADATA_DIM,
        )
        metas = load_metadata()
        sample = next(iter(metas.values()))
        vec = encode_metadata(sample)
        self.assertEqual(vec.shape, (METADATA_DIM,))
        self.assertEqual(vec.dtype, torch.float32)
        self.assertGreaterEqual(vec[0].item(), 0.0)
        self.assertLessEqual(vec[0].item(), 1.0)
        # Layout: [T_norm, loc_oh(4), cond_oh(3)]
        for s, e in [(1, 5), (5, 8)]:
            block_sum = vec[s:e].sum().item()
            self.assertGreaterEqual(block_sum, 0.0)
            self.assertLessEqual(block_sum, 1.0 + 1e-6)

    def test_session_segments_match_paper3_facts(self):
        """Per paper3 facts: 22 sessions, ~144 continuous segments at gap=1
        cover 95.3% of images. Allow tolerance for boundary definition."""
        from methods.m6_stc_yolo.data.session_metadata import (
            load_metadata,
            build_session_segments,
        )
        metas = load_metadata()
        segs = build_session_segments(metas, max_gap=1)
        self.assertEqual(len(segs), 22, f"expected 22 sessions, got {len(segs)}")
        total_segments = sum(len(s) for s in segs.values())
        self.assertGreaterEqual(total_segments, 100)
        self.assertLessEqual(total_segments, 250)
        coverage = sum(len(seg) for s in segs.values() for seg in s) / len(metas)
        self.assertGreaterEqual(coverage, 0.99)


class TestTrainOneHelpers(unittest.TestCase):
    """Section 4 — train_one.py CLI helpers (no GPU needed)."""

    def test_coerce_value_basics(self):
        from methods.m6_stc_yolo.train_one import _coerce_value
        self.assertIs(_coerce_value("true"), True)
        self.assertIs(_coerce_value("False"), False)
        self.assertIsNone(_coerce_value("null"))
        self.assertIsNone(_coerce_value("none"))
        self.assertEqual(_coerce_value("42"), 42)
        self.assertEqual(_coerce_value("1e-3"), 0.001)
        self.assertEqual(_coerce_value("hello"), "hello")
        self.assertEqual(_coerce_value("[1,2,3]"), [1, 2, 3])

    def test_set_dotted_creates_path(self):
        from methods.m6_stc_yolo.train_one import _set_dotted
        cfg = {"stc": {"enable_x": False}}
        _set_dotted(cfg, "stc.enable_x", True)
        self.assertIs(cfg["stc"]["enable_x"], True)
        _set_dotted(cfg, "stc.new.flag", 42)
        self.assertEqual(cfg["stc"]["new"]["flag"], 42)
        _set_dotted(cfg, "top_level", "hello")
        self.assertEqual(cfg["top_level"], "hello")

    def test_apply_overrides_typed(self):
        from methods.m6_stc_yolo.train_one import _apply_overrides
        cfg = {"epochs": 30, "stc": {"enable_colormap_inversion": False}}
        _apply_overrides(cfg, [
            "epochs=10",
            "stc.enable_colormap_inversion=true",
            "stc.ssl_pretrained_backbone=path/to/ckpt.pt",
        ])
        self.assertEqual(cfg["epochs"], 10)
        self.assertIs(cfg["stc"]["enable_colormap_inversion"], True)
        self.assertEqual(cfg["stc"]["ssl_pretrained_backbone"], "path/to/ckpt.pt")

    def test_apply_overrides_rejects_no_equals(self):
        from methods.m6_stc_yolo.train_one import _apply_overrides
        with self.assertRaises(ValueError):
            _apply_overrides({}, ["badkey"])

    def test_state_inheritance_translates_keys(self):
        """When orchestrator state.json sets c4_in_main=true,
        train_one should translate to stc.enable_colormap_inversion=true."""
        import json
        import tempfile
        from methods.m6_stc_yolo.train_one import _apply_state_inheritance
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "state.json"
            state_path.write_text(json.dumps({
                "config_updates": {
                    "c4_in_main": True,
                    "venue_fallback": "tier3",          # not in cfg map
                    "some_random_unknown_key": 123,     # not in cfg map
                },
                "step_results": {
                    "ms2_d6_run_ssl_pretrain": {
                        "backbone_path": "results_v2/.../backbone.pt",
                    },
                },
            }))
            cfg: dict = {"stc": {"enable_colormap_inversion": False}}
            applied = _apply_state_inheritance(cfg, state_path)
            self.assertIs(cfg["stc"]["enable_colormap_inversion"], True)
            # venue_fallback / unknown keys must NOT pollute cfg
            self.assertNotIn("venue_fallback", cfg.get("stc", {}))
            # D6 backbone present and C1 not dropped -> SSL IS wired (this is
            # how D8, the SSL experiment, obtains the backbone before its gate
            # has produced c1_in_main).
            self.assertEqual(cfg["stc"]["ssl_pretrained_backbone"],
                             "results_v2/.../backbone.pt")
            applied_keys = {k for k, _ in applied}
            self.assertIn("stc.enable_colormap_inversion", applied_keys)
            self.assertIn("stc.ssl_pretrained_backbone", applied_keys)

    def test_state_inheritance_skips_ssl_when_c1_dropped(self):
        """After the D8 gate FAILs it sets c1_dropped; downstream steps must
        NOT wire the SSL backbone even though D6 produced one."""
        import json
        import tempfile
        from methods.m6_stc_yolo.train_one import _apply_state_inheritance
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "state.json"
            state_path.write_text(json.dumps({
                "config_updates": {"c4_in_main": True, "c1_dropped": True},
                "step_results": {
                    "ms2_d6_run_ssl_pretrain": {
                        "backbone_path": "results_v2/m2/backbone.pt",
                    },
                },
            }))
            cfg: dict = {"stc": {"ssl_pretrained_backbone": "hardcoded/in/yaml.pt"}}
            _apply_state_inheritance(cfg, state_path)
            # c1_dropped must FORCE ssl off, overriding any hardcoded yaml path.
            self.assertIsNone(cfg["stc"]["ssl_pretrained_backbone"])

    def test_state_inheritance_wires_ssl_when_c1_pass(self):
        import json
        import tempfile
        from methods.m6_stc_yolo.train_one import _apply_state_inheritance
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "state.json"
            state_path.write_text(json.dumps({
                "config_updates": {"c4_in_main": True, "c1_in_main": True},
                "step_results": {
                    "ms2_d6_run_ssl_pretrain": {
                        "backbone_path": "results_v2/m2/backbone.pt",
                    },
                },
            }))
            cfg: dict = {}
            _apply_state_inheritance(cfg, state_path)
            self.assertEqual(cfg["stc"]["enable_colormap_inversion"], True)
            self.assertEqual(cfg["stc"]["ssl_pretrained_backbone"],
                             "results_v2/m2/backbone.pt")

    def test_state_inheritance_missing_file_is_noop(self):
        from methods.m6_stc_yolo.train_one import _apply_state_inheritance
        cfg: dict = {"stc": {"enable_colormap_inversion": False}}
        applied = _apply_state_inheritance(cfg, Path("/nonexistent/state.json"))
        self.assertEqual(applied, [])
        self.assertIs(cfg["stc"]["enable_colormap_inversion"], False)


class TestColormapInverter(unittest.TestCase):
    """Section 6 — D2 ColormapInverter (implemented 2026-05-29)."""

    # ---- init / mode validation ----

    def test_init_empirical_loads_buffers(self):
        import torch
        from methods.m6_stc_yolo.modules.colormap_inverter import ColormapInverter
        inv = ColormapInverter(mode="empirical")
        # Buffers exist and have correct shapes
        self.assertTrue(hasattr(inv, "lut_rgb"))
        self.assertTrue(hasattr(inv, "t"))
        self.assertEqual(inv.lut_rgb.shape, (256, 3))
        self.assertEqual(inv.t.shape, (256,))
        # Non-persistent: NOT in state_dict
        sd = inv.state_dict()
        self.assertNotIn("lut_rgb", sd)
        self.assertNotIn("t", sd)

    def test_init_mlp_raises_not_implemented(self):
        from methods.m6_stc_yolo.modules.colormap_inverter import ColormapInverter
        # D1b finding: empirical LUT MSE 0.000568 makes MLP unnecessary
        with self.assertRaises(NotImplementedError):
            ColormapInverter(mode="mlp")

    def test_init_invalid_mode(self):
        from methods.m6_stc_yolo.modules.colormap_inverter import ColormapInverter
        with self.assertRaises(ValueError):
            ColormapInverter(mode="not_a_real_mode")

    def test_init_missing_empirical_path(self):
        from methods.m6_stc_yolo.modules.colormap_inverter import ColormapInverter
        with self.assertRaises(FileNotFoundError):
            ColormapInverter(mode="empirical",
                              empirical_lut_path="/does/not/exist.npz")

    # ---- forward: shape + range contracts ----

    def test_forward_shape_contract(self):
        import torch
        from methods.m6_stc_yolo.modules.colormap_inverter import ColormapInverter
        inv = ColormapInverter(mode="empirical")
        x = torch.rand(2, 3, 64, 96)
        y = inv(x)
        self.assertEqual(y.shape, (2, 1, 64, 96))

    def test_forward_output_range_in_unit_interval(self):
        import torch
        from methods.m6_stc_yolo.modules.colormap_inverter import ColormapInverter
        inv = ColormapInverter(mode="empirical")
        x = torch.rand(1, 3, 64, 64)
        y = inv(x)
        self.assertGreaterEqual(float(y.min()), 0.0)
        self.assertLessEqual(float(y.max()), 1.0)

    def test_forward_rejects_wrong_input_shape(self):
        import torch
        from methods.m6_stc_yolo.modules.colormap_inverter import ColormapInverter
        inv = ColormapInverter(mode="empirical")
        with self.assertRaises(ValueError):
            inv(torch.zeros(3, 64, 64))   # missing batch
        with self.assertRaises(ValueError):
            inv(torch.zeros(1, 4, 8, 8))  # 4-ch input

    # ---- empirical: feeding LUT entries back should recover t ----

    def test_empirical_lut_self_consistency(self):
        """If we feed in lut_rgb[k] as a 1-pixel image, NN lookup should
        return that same index k -> t[k]. Tests for k spanning the LUT."""
        import torch
        from methods.m6_stc_yolo.modules.colormap_inverter import ColormapInverter
        inv = ColormapInverter(mode="empirical")
        # Build a 1x3x1xK image where pixel k along the W axis is lut_rgb[k]
        K = inv.lut_rgb.shape[0]
        # Use a stride to test K representative indices
        ks = list(range(0, K, K // 16))   # 16 sample points
        rgb_strip = inv.lut_rgb[ks].T.unsqueeze(0).unsqueeze(2)  # (1, 3, 1, len(ks))
        t_hat = inv(rgb_strip)  # (1, 1, 1, len(ks))
        out = t_hat[0, 0, 0]    # (len(ks),)
        expected = inv.t[ks]
        # Empirical LUT may have near-duplicate adjacent entries -> use tolerance
        diff = (out - expected).abs().max().item()
        self.assertLess(diff, 0.02, f"max LUT identity error {diff} > 0.02")

    # ---- analytic_matplotlib mode ----

    def test_analytic_matplotlib_mode_inferno(self):
        import torch
        from methods.m6_stc_yolo.modules.colormap_inverter import ColormapInverter
        inv = ColormapInverter(mode="analytic_matplotlib",
                                matplotlib_colormap="inferno")
        self.assertEqual(inv.lut_rgb.shape, (256, 3))
        x = torch.rand(1, 3, 32, 32)
        y = inv(x)
        self.assertEqual(y.shape, (1, 1, 32, 32))
        self.assertGreaterEqual(float(y.min()), 0.0)
        self.assertLessEqual(float(y.max()), 1.0)

    # ---- grayscale_collapse mode (no LUT) ----

    def test_grayscale_collapse_exact_formula(self):
        """Verify Y = 0.299R + 0.587G + 0.114B exactly."""
        import torch
        from methods.m6_stc_yolo.modules.colormap_inverter import ColormapInverter
        inv = ColormapInverter(mode="grayscale_collapse")
        # Construct a known input: R=0.5, G=0.25, B=1.0
        x = torch.tensor([[[[0.5]], [[0.25]], [[1.0]]]], dtype=torch.float32)
        y = inv(x)
        expected = 0.299 * 0.5 + 0.587 * 0.25 + 0.114 * 1.0
        self.assertAlmostEqual(float(y[0, 0, 0, 0]), expected, places=5)


class TestConv1chFromRGB(unittest.TestCase):
    """Section 7 — D3 Conv1chFromRGB (implemented 2026-05-29)."""

    @staticmethod
    def _make_pretrained_conv():
        """Build a synthetic 'pretrained' ultralytics Conv(3 -> 32) for tests."""
        from ultralytics.nn.modules.conv import Conv
        orig = Conv(3, 32, k=3, s=2)
        # Match ultralytics' customised BN behavior to expose copy correctness
        orig.bn.eps = 1e-3
        orig.bn.momentum = 0.03
        # Push some non-trivial values into BN running stats
        import torch
        with torch.no_grad():
            orig.bn.running_mean.fill_(0.1)
            orig.bn.running_var.fill_(0.5)
            orig.bn.num_batches_tracked.fill_(123)
        return orig

    def test_build_returns_subclass_of_ultralytics_conv(self):
        from ultralytics.nn.modules.conv import Conv
        from methods.m6_stc_yolo.modules.conv1ch_from_rgb import Conv1chFromRGB
        orig = self._make_pretrained_conv()
        new = Conv1chFromRGB.build_from_pretrained_conv(orig)
        # isinstance check matters for ultralytics DetectionModel internals.
        self.assertIsInstance(new, Conv)
        self.assertIsInstance(new, Conv1chFromRGB)

    def test_build_weight_is_chsum_of_pretrained(self):
        """Weight migration uses ch-SUM (not mean) so that uniform R=G=B=x
        feeding the original 3-ch Conv vs our 1-ch Conv with T_hat=x produces
        IDENTICAL outputs (preserves activation magnitude & BN compatibility)."""
        import torch
        from methods.m6_stc_yolo.modules.conv1ch_from_rgb import Conv1chFromRGB
        orig = self._make_pretrained_conv()
        new = Conv1chFromRGB.build_from_pretrained_conv(orig)
        # Shape: (c_out, 1, k, k)
        self.assertEqual(new.conv.weight.shape, (32, 1, 3, 3))
        expected = orig.conv.weight.data.sum(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(new.conv.weight.data, expected,
                                        atol=1e-6))

    def test_uniform_input_output_invariance(self):
        """Mathematical guarantee: uniform R=G=B=x input through original
        3-ch Conv equals x through migrated 1-ch Conv. This is the property
        that makes sum migration preserve the pretrained activation
        distribution (and downstream BN statistics)."""
        import torch
        from methods.m6_stc_yolo.modules.conv1ch_from_rgb import Conv1chFromRGB
        orig = self._make_pretrained_conv()
        orig.eval()
        new = Conv1chFromRGB.build_from_pretrained_conv(orig)
        new.eval()
        # Use a constant-color input (R=G=B=0.5) so colormap inversion is
        # well-defined and behaves as identity-ish on the channel dimension.
        # Compare just the Conv(weight) output, bypassing ColormapInverter
        # and BN entirely -- this isolates the weight migration math.
        x = torch.ones(1, 3, 8, 8) * 0.5
        # Original: 3-ch conv on (1, 3, 8, 8)
        orig_conv_out = torch.nn.functional.conv2d(
            x, orig.conv.weight, bias=orig.conv.bias,
            stride=orig.conv.stride, padding=orig.conv.padding)
        # New: 1-ch conv on (1, 1, 8, 8) of x
        x_1ch = x[:, :1]
        new_conv_out = torch.nn.functional.conv2d(
            x_1ch, new.conv.weight, bias=new.conv.bias,
            stride=new.conv.stride, padding=new.conv.padding)
        self.assertTrue(torch.allclose(orig_conv_out, new_conv_out, atol=1e-5),
                        f"max diff {(orig_conv_out - new_conv_out).abs().max()}")

    def test_build_copies_bn_state_and_params(self):
        import torch
        from methods.m6_stc_yolo.modules.conv1ch_from_rgb import Conv1chFromRGB
        orig = self._make_pretrained_conv()
        new = Conv1chFromRGB.build_from_pretrained_conv(orig)
        self.assertTrue(torch.allclose(new.bn.weight.data, orig.bn.weight.data))
        self.assertTrue(torch.allclose(new.bn.bias.data, orig.bn.bias.data))
        self.assertTrue(torch.allclose(new.bn.running_mean.data,
                                        orig.bn.running_mean.data))
        self.assertTrue(torch.allclose(new.bn.running_var.data,
                                        orig.bn.running_var.data))
        self.assertEqual(int(new.bn.num_batches_tracked.item()),
                         int(orig.bn.num_batches_tracked.item()))

    def test_build_preserves_ultralytics_bn_eps_momentum(self):
        """BN eps/momentum are ultralytics-specific (1e-3 / 0.03 vs PyTorch
        defaults 1e-5 / 0.1). Missing copy silently shifts eval outputs."""
        from methods.m6_stc_yolo.modules.conv1ch_from_rgb import Conv1chFromRGB
        orig = self._make_pretrained_conv()
        new = Conv1chFromRGB.build_from_pretrained_conv(orig)
        self.assertAlmostEqual(new.bn.eps, 1e-3, places=8)
        self.assertAlmostEqual(new.bn.momentum, 0.03, places=8)

    def test_forward_shape_with_stride_2(self):
        import torch
        from methods.m6_stc_yolo.modules.conv1ch_from_rgb import Conv1chFromRGB
        orig = self._make_pretrained_conv()
        new = Conv1chFromRGB.build_from_pretrained_conv(orig)
        new.eval()
        x = torch.rand(2, 3, 64, 96)
        y = new(x)
        # parent Conv stride=2, kernel=3, padding=1 (ultralytics autopad)
        self.assertEqual(y.shape, (2, 32, 32, 48))

    def test_forward_rejects_non_rgb_input(self):
        import torch
        from methods.m6_stc_yolo.modules.conv1ch_from_rgb import Conv1chFromRGB
        orig = self._make_pretrained_conv()
        new = Conv1chFromRGB.build_from_pretrained_conv(orig)
        with self.assertRaises(ValueError):
            new(torch.zeros(1, 1, 16, 16))   # 1-ch input

    def test_num_colormaps_nonzero_raises(self):
        from methods.m6_stc_yolo.modules.conv1ch_from_rgb import Conv1chFromRGB
        # D1b confirmed single LUT, only num_colormaps=0 implemented.
        with self.assertRaises(NotImplementedError):
            Conv1chFromRGB(num_colormaps=4)

    def test_build_rejects_non_rgb_pretrained_weight(self):
        from methods.m6_stc_yolo.modules.conv1ch_from_rgb import Conv1chFromRGB
        # Build a stub with 4-ch weight to verify the migration shape check.
        from ultralytics.nn.modules.conv import Conv
        orig = Conv(4, 32, k=3, s=2)
        with self.assertRaises(ValueError):
            Conv1chFromRGB.build_from_pretrained_conv(orig)

    def test_build_from_none_raises(self):
        from methods.m6_stc_yolo.modules.conv1ch_from_rgb import Conv1chFromRGB
        with self.assertRaises(ValueError):
            Conv1chFromRGB.build_from_pretrained_conv(pretrained_conv=None)

    def test_forward_fuse_path_goes_through_colormap_inv(self):
        """Regression test for the first D4b crash.

        ultralytics' BaseModel.fuse() (tasks.py L240-242) directly does:
            m.conv = fuse_conv_and_bn(m.conv, m.bn)
            m.forward = m.forward_fuse
        on each Conv. The instance-level rewire means our class-level
        `forward` override is shadowed; only `forward_fuse` (resolved via MRO)
        runs. The parent Conv's forward_fuse is `self.act(self.conv(x))` --
        skipping our ColormapInverter and feeding raw 3-ch RGB into our
        1-ch Conv -> RuntimeError('expected 1 channels but got 3') inside
        validator.warmup() -> AutoBackend.forward.

        After the fix, forward_fuse must also chain through ColormapInverter,
        so a fused module accepts 3-ch RGB just like the unfused module."""
        import torch
        from ultralytics.utils.torch_utils import fuse_conv_and_bn
        from methods.m6_stc_yolo.modules.conv1ch_from_rgb import Conv1chFromRGB
        orig = self._make_pretrained_conv()
        new = Conv1chFromRGB.build_from_pretrained_conv(orig)
        # Replicate ultralytics BaseModel.fuse() exactly (tasks.py L240-242).
        new.conv = fuse_conv_and_bn(new.conv, new.bn)
        delattr(new, "bn")
        new.forward = new.forward_fuse
        new.eval()
        x = torch.rand(1, 3, 64, 96)
        # Before fix: this would raise RuntimeError(1 vs 3 channels).
        y = new(x)
        self.assertEqual(y.shape, (1, 32, 32, 48))


class TestC4Integration(unittest.TestCase):
    """Section 8 — D3b: build_stc_yolo + make_stc_trainer wire C4 end-to-end."""

    def test_build_vanilla_unchanged(self):
        """Without enable_colormap_inversion, model.model[0] stays vanilla."""
        from ultralytics.nn.modules.conv import Conv
        from methods.m6_stc_yolo.model import build_stc_yolo
        from methods.m6_stc_yolo.modules.conv1ch_from_rgb import Conv1chFromRGB
        yolo = build_stc_yolo(enable_colormap_inversion=False)
        self.assertIsInstance(yolo.model.model[0], Conv)
        self.assertNotIsInstance(yolo.model.model[0], Conv1chFromRGB)

    def test_build_with_c4_swaps_first_conv(self):
        from methods.m6_stc_yolo.model import build_stc_yolo
        from methods.m6_stc_yolo.modules.conv1ch_from_rgb import Conv1chFromRGB
        yolo = build_stc_yolo(enable_colormap_inversion=True)
        first = yolo.model.model[0]
        self.assertIsInstance(first, Conv1chFromRGB)
        # ultralytics parse_model attrs preserved
        for attr in ("f", "i", "type"):
            self.assertTrue(hasattr(first, attr),
                            f"missing parse_model attribute {attr}")
        # ColormapInverter present
        self.assertTrue(hasattr(first, "colormap_inv"))

    def test_swap_is_idempotent(self):
        """Re-building (or rebuilding from a best.pt that already has C4)
        should not double-swap."""
        from methods.m6_stc_yolo.model import build_stc_yolo, _swap_first_conv_to_c4
        from methods.m6_stc_yolo.modules.conv1ch_from_rgb import Conv1chFromRGB
        yolo = build_stc_yolo(enable_colormap_inversion=True)
        first_before = yolo.model.model[0]
        _swap_first_conv_to_c4(yolo.model)
        first_after = yolo.model.model[0]
        # Identity preserved (same object, no rebuild)
        self.assertIs(first_before, first_after)
        self.assertIsInstance(first_after, Conv1chFromRGB)

    def test_c4_forward_smoke(self):
        """Forward 1 small batch through the C4-enabled model; check output
        shape matches a YOLO detect-head expectation (3-scale list)."""
        import torch
        from methods.m6_stc_yolo.model import build_stc_yolo
        yolo = build_stc_yolo(enable_colormap_inversion=True)
        yolo.model.eval()
        with torch.no_grad():
            # ultralytics' forward expects (B, 3, H, W). H, W must be multiples
            # of 32 (max stride). Use 32x32 to keep memory small.
            x = torch.rand(1, 3, 64, 64)
            out = yolo.model(x)
        # Output structure varies by mode (train vs eval) but at minimum
        # must not raise -- the swap must be functional end-to-end.
        self.assertIsNotNone(out)

    def test_make_stc_trainer_returns_subclass(self):
        from ultralytics.models.yolo.detect import DetectionTrainer
        from methods.m6_stc_yolo.model import make_stc_trainer
        TrainerCls = make_stc_trainer(enable_colormap_inversion=True)
        self.assertTrue(issubclass(TrainerCls, DetectionTrainer))
        # Has overridden get_model
        self.assertIn("get_model", TrainerCls.__dict__)

    def test_make_stc_trainer_mean_teacher_subclass(self):
        """D10d completion check: make_stc_trainer(enable_mean_teacher=True)
        returns a DetectionTrainer subclass with the MT hooks."""
        from ultralytics.models.yolo.detect import DetectionTrainer
        from methods.m6_stc_yolo.model import make_stc_trainer
        TrainerCls = make_stc_trainer(
            enable_mean_teacher=True, mt_cfg={"mt_ema_momentum": 0.99})
        self.assertTrue(issubclass(TrainerCls, DetectionTrainer))
        self.assertIn("optimizer_step", TrainerCls.__dict__)
        self.assertIn("_maybe_build_teacher", TrainerCls.__dict__)


class TestSSLBackboneLoad(unittest.TestCase):
    """Section 8b — D7: load SSL backbone into build_stc_yolo (+ C4 ordering)."""

    @staticmethod
    def _make_fake_ssl_ckpt(tmp_path):
        """A D6-format checkpoint built from a vanilla model with two backbone
        weights perturbed, so a successful load is detectable."""
        import torch
        from methods.m6_stc_yolo.model import build_stc_yolo
        vanilla = build_stc_yolo(enable_colormap_inversion=False)
        sd = {k: v.clone() for k, v in vanilla.model.state_dict().items()}
        sd["model.0.conv.weight"] += 1.0   # 3-ch stem marker
        sd["model.1.conv.weight"] += 2.0
        p = tmp_path / "fake_ssl.pt"
        torch.save({"state_dict": sd, "feature_layer_idx": 4,
                    "epochs": 1, "final_loss": 0.0}, p)
        return p, sd

    def test_load_ssl_into_vanilla(self):
        import tempfile
        import torch
        from pathlib import Path as _P
        from methods.m6_stc_yolo.model import build_stc_yolo
        with tempfile.TemporaryDirectory() as td:
            ckpt_p, sd = self._make_fake_ssl_ckpt(_P(td))
            yolo = build_stc_yolo(ssl_pretrained_backbone=str(ckpt_p))
            loaded = yolo.model.state_dict()["model.1.conv.weight"]
            self.assertTrue(torch.allclose(loaded, sd["model.1.conv.weight"]))

    def test_load_ssl_then_c4_migrates_ssl_stem(self):
        """With C4+SSL, model.0 is Conv1chFromRGB whose 1-ch weight is the
        ch-SUM of the *SSL-trained* 3-ch stem (proves SSL-before-swap order)."""
        import tempfile
        import torch
        from pathlib import Path as _P
        from methods.m6_stc_yolo.model import build_stc_yolo
        from methods.m6_stc_yolo.modules.conv1ch_from_rgb import Conv1chFromRGB
        with tempfile.TemporaryDirectory() as td:
            ckpt_p, sd = self._make_fake_ssl_ckpt(_P(td))
            yolo = build_stc_yolo(enable_colormap_inversion=True,
                                  ssl_pretrained_backbone=str(ckpt_p))
            first = yolo.model.model[0]
            self.assertIsInstance(first, Conv1chFromRGB)
            expected = sd["model.0.conv.weight"].sum(dim=1, keepdim=True)
            self.assertTrue(torch.allclose(first.conv.weight.data, expected,
                                            atol=1e-5))

    def test_load_ssl_missing_file_raises(self):
        from methods.m6_stc_yolo.model import _load_ssl_backbone, build_stc_yolo
        yolo = build_stc_yolo()
        with self.assertRaises(FileNotFoundError):
            _load_ssl_backbone(yolo.model, "/does/not/exist.pt")

    def test_load_ssl_wrong_arch_raises(self):
        import tempfile
        import torch
        from pathlib import Path as _P
        from methods.m6_stc_yolo.model import _load_ssl_backbone, build_stc_yolo
        with tempfile.TemporaryDirectory() as td:
            p = _P(td) / "wrong.pt"
            torch.save({"state_dict": {"foo.bar": torch.zeros(3)}}, p)
            yolo = build_stc_yolo()
            with self.assertRaises(RuntimeError):
                _load_ssl_backbone(yolo.model, str(p))


class TestFramePairDataset(unittest.TestCase):
    """Section 9 — D5 FramePairDataset (implemented 2026-05-29)."""

    def test_dataset_has_pairs(self):
        """Should produce > 1000 pairs from train+val with gap in {1, 2}."""
        from methods.m6_stc_yolo.data.frame_pair_dataset import FramePairDataset
        ds = FramePairDataset()
        self.assertGreater(len(ds), 1000,
                            f"expected > 1000 pairs, got {len(ds)}")

    def test_no_test_split_leakage(self):
        """No pair endpoint may be from the test split (strict)."""
        from methods.m6_stc_yolo.data.frame_pair_dataset import (
            FramePairDataset,
            _read_split_names,
            DEFAULT_SPLIT_DIR,
        )
        ds = FramePairDataset()
        test_names = set(_read_split_names(DEFAULT_SPLIT_DIR / "test.txt"))
        self.assertGreater(len(test_names), 0)
        # Spot-check first 200 pairs (full sweep would be slow on bigger sets)
        for n_t, n_tk, k in ds.pairs[:200]:
            self.assertNotIn(n_t, test_names)
            self.assertNotIn(n_tk, test_names)

    def test_pair_gaps_within_configured_range(self):
        from methods.m6_stc_yolo.data.frame_pair_dataset import FramePairDataset
        ds = FramePairDataset(pair_gap_min=1, pair_gap_max=2)
        for _, _, k in ds.pairs[:500]:
            self.assertGreaterEqual(k, 1)
            self.assertLessEqual(k, 2)

    def test_getitem_returns_tensor_pair(self):
        import torch
        from methods.m6_stc_yolo.data.frame_pair_dataset import FramePairDataset
        ds = FramePairDataset(image_size=320, augment=False)
        img_t, img_tk, k = ds[0]
        self.assertIsInstance(img_t, torch.Tensor)
        self.assertEqual(img_t.shape, (3, 320, 320))
        self.assertEqual(img_tk.shape, (3, 320, 320))
        self.assertGreaterEqual(float(img_t.min()), 0.0)
        self.assertLessEqual(float(img_t.max()), 1.0)
        self.assertIsInstance(k, int)
        self.assertIn(k, (1, 2))


class TestSSLPretrain(unittest.TestCase):
    """Section 10 — D5 SSL loss / projection / extractor unit tests."""

    def test_projection_head_shape_preserved(self):
        import torch
        from methods.m6_stc_yolo.modules.ssl_pretrain import SSLProjectionHead
        head = SSLProjectionHead(in_dim=64, hidden_dim=128, out_dim=32)
        x = torch.randn(2, 64, 20, 20)
        y = head(x)
        self.assertEqual(y.shape, (2, 32, 20, 20))

    def test_loss_returns_scalar_and_finite(self):
        import torch
        from methods.m6_stc_yolo.modules.ssl_pretrain import (
            CrossFrameTokenConsistencyLoss,
        )
        loss_fn = CrossFrameTokenConsistencyLoss(temperature=0.2)
        z_t = torch.randn(2, 32, 8, 8)
        z_tk = torch.randn(2, 32, 8, 8)
        loss = loss_fn(z_t, z_tk)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_loss_identical_inputs_lower_than_random(self):
        """When z_t = z_tk the argmax-positive lookup returns the identity,
        which is the maximal-similarity column. Loss should be measurably
        lower than for unrelated random tensors."""
        import torch
        from methods.m6_stc_yolo.modules.ssl_pretrain import (
            CrossFrameTokenConsistencyLoss,
        )
        torch.manual_seed(0)
        loss_fn = CrossFrameTokenConsistencyLoss(temperature=0.05)
        z = torch.randn(1, 16, 4, 4)
        loss_same = loss_fn(z, z).item()
        loss_diff = loss_fn(
            torch.randn(1, 16, 4, 4),
            torch.randn(1, 16, 4, 4),
        ).item()
        self.assertLess(loss_same, loss_diff)

    def test_loss_token_subsampling_bounds_and_runs(self):
        """With num_tokens < HW, the loss must sample positions so the sim
        matrix is bounded, still return a finite scalar, and backprop. This is
        the fix for the imgsz=640 P3-grid (6400 tokens) OOM."""
        import torch
        from methods.m6_stc_yolo.modules.ssl_pretrain import (
            CrossFrameTokenConsistencyLoss,
        )
        torch.manual_seed(0)
        # 40x40 grid = 1600 tokens; sample only 64.
        z_t = torch.randn(2, 16, 40, 40, requires_grad=True)
        z_tk = torch.randn(2, 16, 40, 40)
        loss_fn = CrossFrameTokenConsistencyLoss(temperature=0.2, num_tokens=64)
        loss = loss_fn(z_t, z_tk)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(z_t.grad)

    def test_loss_num_tokens_none_uses_full_grid(self):
        """num_tokens=None preserves exact full-grid behaviour (regression)."""
        import torch
        from methods.m6_stc_yolo.modules.ssl_pretrain import (
            CrossFrameTokenConsistencyLoss,
        )
        torch.manual_seed(0)
        z_t = torch.randn(1, 8, 6, 6)
        z_tk = torch.randn(1, 8, 6, 6)
        full = CrossFrameTokenConsistencyLoss(temperature=0.2, num_tokens=None)
        # num_tokens >= HW also falls through to full grid (no sampling).
        big = CrossFrameTokenConsistencyLoss(temperature=0.2, num_tokens=10_000)
        self.assertTrue(torch.allclose(full(z_t, z_tk), big(z_t, z_tk)))

    def test_loss_rejects_mismatched_shapes(self):
        import torch
        from methods.m6_stc_yolo.modules.ssl_pretrain import (
            CrossFrameTokenConsistencyLoss,
        )
        loss_fn = CrossFrameTokenConsistencyLoss()
        with self.assertRaises(ValueError):
            loss_fn(torch.randn(1, 8, 4, 4), torch.randn(1, 8, 5, 5))

    def test_feature_extractor_captures_layer_output(self):
        """Hook should fire and capture a (B, C, H/s, W/s) tensor on a real
        YOLOv12s.pt model."""
        import torch
        from ultralytics import YOLO
        from methods.m6_stc_yolo.modules.ssl_pretrain import (
            BackboneFeatureExtractor,
        )
        yolo = YOLO(str(_PROJECT_ROOT / "yolo12s.pt"))
        yolo.model.eval()
        ext = BackboneFeatureExtractor(yolo.model, layer_idx=4)
        with torch.no_grad():
            x = torch.zeros(1, 3, 320, 320)
            feat = ext(x)
        self.assertEqual(feat.dim(), 4)
        self.assertEqual(feat.shape[0], 1)
        self.assertGreater(feat.shape[1], 0)
        ext.remove()


class TestSSLResume(unittest.TestCase):
    """Section 10b — D6 crash-resilient checkpoint/resume (added after the
    2026-05-29 crash that lost a full SSL run because fit() only saved at the
    very end). Uses a tiny CPU backbone so no GPU / real YOLO is required."""

    @staticmethod
    def _tiny_setup(tmp_dir, epochs):
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset
        from methods.m6_stc_yolo.modules.ssl_pretrain import (
            SSLPretrainer, SSLProjectionHead,
        )

        class TinyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.ModuleList([
                    nn.Conv2d(3, 8, 3, stride=2, padding=1),   # idx 0
                    nn.ReLU(),                                  # idx 1
                    nn.Conv2d(8, 8, 3, stride=2, padding=1),   # idx 2 (capture)
                ])

            def forward(self, x):
                for m in self.model:
                    x = m(x)
                return x

        class TinyPairDS(Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, i):
                g = torch.Generator().manual_seed(i)
                return (torch.rand(3, 32, 32, generator=g),
                        torch.rand(3, 32, 32, generator=g), 1)

        model = TinyBackbone()
        head = SSLProjectionHead(in_dim=8, hidden_dim=16, out_dim=8)
        cfg = {
            "epochs": epochs, "batch": 2, "lr0": 1e-3, "weight_decay": 0.0,
            "workers": 0, "temperature": 0.2, "feature_layer_idx": 2,
            "device": "cpu", "output_dir": str(tmp_dir),
        }
        return SSLPretrainer(model, head, TinyPairDS(), cfg)

    def test_complete_run_produces_backbone_and_clears_last(self):
        import tempfile
        from pathlib import Path as _P
        with tempfile.TemporaryDirectory() as td:
            tmp = _P(td)
            self._tiny_setup(tmp, epochs=2).fit()
            self.assertTrue((tmp / "pretrained_backbone_stc.pt").exists())
            self.assertTrue((tmp / "ssl_pretrain_summary.json").exists())
            self.assertFalse((tmp / "last.pt").exists(),
                             "last.pt should be cleaned after a complete run")

    def test_resume_continues_from_partial_checkpoint(self):
        import tempfile
        import torch
        from pathlib import Path as _P
        with tempfile.TemporaryDirectory() as td:
            tmp = _P(td)
            # Build a partial checkpoint: hand-save last.pt at epoch index 1
            # (i.e. 2 epochs completed) for a target of 4 epochs.
            p = self._tiny_setup(tmp, epochs=4)
            p.yolo_model.to("cpu")
            p.projection_head.to("cpu")
            opt = torch.optim.AdamW(
                list(p.yolo_model.parameters())
                + list(p.projection_head.parameters()), lr=1e-3)
            hist = [{"epoch": 0, "loss": 1.0, "n_batches": 2},
                    {"epoch": 1, "loss": 0.9, "n_batches": 2}]
            p._save_resume_ckpt(tmp / "last.pt", opt, hist, epoch=1,
                                layer_idx=2, epochs=4)
            self.assertTrue((tmp / "last.pt").exists())
            # Fresh pretrainer (simulates process restart) resumes to epoch 4.
            p2 = self._tiny_setup(tmp, epochs=4)
            p2.fit()
            import json
            summary = json.loads(
                (tmp / "ssl_pretrain_summary.json").read_text(encoding="utf-8"))
            # History must contain all 4 epochs (2 restored + 2 new), in order.
            epochs_seen = [h["epoch"] for h in summary["history"]]
            self.assertEqual(epochs_seen, [0, 1, 2, 3])
            self.assertFalse((tmp / "last.pt").exists())

    def test_resume_rejects_incompatible_layer_idx(self):
        import tempfile
        import torch
        from pathlib import Path as _P
        with tempfile.TemporaryDirectory() as td:
            tmp = _P(td)
            p = self._tiny_setup(tmp, epochs=4)
            opt = torch.optim.AdamW(list(p.yolo_model.parameters()), lr=1e-3)
            # Save a checkpoint claiming a DIFFERENT feature layer.
            p._save_resume_ckpt(tmp / "last.pt", opt, [], epoch=0,
                                layer_idx=99, epochs=4)
            p2 = self._tiny_setup(tmp, epochs=4)  # configured for layer_idx=2
            with self.assertRaises(RuntimeError):
                p2.fit()


class TestUnlabeledNeighborSampler(unittest.TestCase):
    """Section 11 — D10 C2 unlabeled neighbor sampler (train-only, no leakage)."""

    @staticmethod
    def _train_names():
        from methods.m6_stc_yolo.data.frame_pair_dataset import (
            _read_split_names, DEFAULT_SPLIT_DIR)
        return set(_read_split_names(DEFAULT_SPLIT_DIR / "train.txt"))

    def test_sample_is_train_only_and_within_gap(self):
        from methods.m6_stc_yolo.data.unlabeled_neighbor_sampler import (
            UnlabeledNeighborSampler)
        train = self._train_names()
        samp = UnlabeledNeighborSampler(train, max_gap=3, seed=42)
        tested = 0
        for name in list(train):
            if not samp.candidates(name):
                continue
            nb, gap = samp.sample(name)
            self.assertIn(nb, train, "neighbor leaked outside train split")
            self.assertNotEqual(nb, name)
            self.assertGreaterEqual(gap, 1)
            self.assertLessEqual(gap, 3)
            tested += 1
            if tested >= 50:
                break
        self.assertGreater(tested, 0, "no sampleable frames found")

    def test_no_test_split_leakage(self):
        from methods.m6_stc_yolo.data.unlabeled_neighbor_sampler import (
            UnlabeledNeighborSampler)
        from methods.m6_stc_yolo.data.frame_pair_dataset import (
            _read_split_names, DEFAULT_SPLIT_DIR)
        train = self._train_names()
        test = set(_read_split_names(DEFAULT_SPLIT_DIR / "test.txt"))
        samp = UnlabeledNeighborSampler(train, max_gap=3, seed=1)
        for name in list(train)[:300]:
            for nb, _ in samp.candidates(name):
                self.assertNotIn(nb, test)

    def test_sample_raises_when_no_neighbor(self):
        from methods.m6_stc_yolo.data.unlabeled_neighbor_sampler import (
            UnlabeledNeighborSampler)
        samp = UnlabeledNeighborSampler({"nonexistent_frame_xyz"}, max_gap=3)
        with self.assertRaises(ValueError):
            samp.sample("nonexistent_frame_xyz")


class TestMeanTeacher(unittest.TestCase):
    """Section 12 — D10 C2 EMATeacher + TemporalConsistencyLoss."""

    def test_ema_teacher_frozen_and_is_deepcopy(self):
        import torch
        import torch.nn as nn
        from methods.m6_stc_yolo.modules.mean_teacher import EMATeacher
        student = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4))
        t = EMATeacher(student, momentum=0.9)
        self.assertFalse(any(p.requires_grad for p in t.model.parameters()))
        # independent storage
        self.assertIsNot(t.model[0].weight, student[0].weight)

    def test_ema_update_math(self):
        import torch
        import torch.nn as nn
        from methods.m6_stc_yolo.modules.mean_teacher import EMATeacher
        student = nn.Conv2d(3, 4, 3, bias=False)
        t = EMATeacher(student, momentum=0.9)
        before = t.model.weight.detach().clone()
        with torch.no_grad():
            student.weight.add_(1.0)            # student now = before + 1
        t.update(student)
        # teacher = 0.9*before + 0.1*(before+1) = before + 0.1
        expected = before + 0.1
        self.assertTrue(torch.allclose(t.model.weight, expected, atol=1e-6))

    def test_predict_pseudo_returns_list_of_tensors(self):
        """D10 completion check: predict_pseudo(rand_frames, 0.5) -> List[Tensor]."""
        import torch
        from methods.m6_stc_yolo.model import build_stc_yolo
        from methods.m6_stc_yolo.modules.mean_teacher import EMATeacher
        yolo = build_stc_yolo()
        t = EMATeacher(yolo.model, momentum=0.999)
        out = t.predict_pseudo(torch.rand(2, 3, 64, 64), conf_threshold=0.5)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        for d in out:
            self.assertIsInstance(d, torch.Tensor)
            self.assertEqual(d.shape[-1], 6)  # x1y1x2y2 conf cls

    def test_consistency_loss_zero_on_identical(self):
        import torch
        from methods.m6_stc_yolo.modules.mean_teacher import TemporalConsistencyLoss
        boxes = torch.tensor([[10., 10., 50., 50., 0.9, 0.],
                              [20., 30., 60., 80., 0.8, 1.]])
        loss_fn = TemporalConsistencyLoss(alpha_cls=1.0)
        loss = loss_fn([boxes.clone()], [boxes.clone()])
        self.assertLess(float(loss), 1e-5)

    def test_consistency_loss_positive_and_differentiable(self):
        import torch
        from methods.m6_stc_yolo.modules.mean_teacher import TemporalConsistencyLoss
        s = torch.tensor([[10., 10., 50., 50., 0.9, 0.]], requires_grad=True)
        t = torch.tensor([[30., 30., 90., 90., 0.9, 1.]])  # shifted + wrong cls
        loss_fn = TemporalConsistencyLoss(alpha_cls=1.0)
        loss = loss_fn([s], [t])
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(s.grad)

    def test_pseudo_to_targets_format(self):
        import torch
        from methods.m6_stc_yolo.modules.mean_teacher import pseudo_to_targets
        # two frames: frame0 has 2 boxes, frame1 empty
        p0 = torch.tensor([[10., 10., 50., 50., 0.9, 0.],
                           [0., 0., 100., 100., 0.8, 1.]])
        p1 = torch.zeros(0, 6)
        t = pseudo_to_targets([p0, p1], imgsz=100, device="cpu")
        self.assertEqual(t["batch_idx"].tolist(), [0., 0.])
        self.assertEqual(t["cls"].shape, (2, 1))
        self.assertEqual(t["bboxes"].shape, (2, 4))
        # box0 center (30,30)/100=0.3, wh (40,40)/100=0.4
        self.assertTrue(torch.allclose(t["bboxes"][0],
                                       torch.tensor([0.3, 0.3, 0.4, 0.4])))

    def test_mt_detection_model_loss_adds_and_consumes_aux(self):
        """_STCDetectionModel.loss adds the stashed aux once then clears it."""
        import torch
        from methods.m6_stc_yolo.model import _STCDetectionModel
        from ultralytics.nn.tasks import DetectionModel
        # Stub: bypass the real forward by monkeypatching the parent loss on an
        # instance whose __class__ is _STCDetectionModel.
        obj = _STCDetectionModel.__new__(_STCDetectionModel)
        obj._mt_aux_loss = torch.tensor(2.0)
        orig = DetectionModel.loss
        try:
            DetectionModel.loss = lambda self, batch, preds=None: (
                torch.tensor(5.0), torch.tensor([1., 2., 3.]))
            loss1, _ = obj.loss({})
            self.assertAlmostEqual(float(loss1), 7.0)        # 5 + 2 aux
            self.assertIsNone(obj._mt_aux_loss)              # consumed
            loss2, _ = obj.loss({})
            self.assertAlmostEqual(float(loss2), 5.0)        # no aux now
        finally:
            DetectionModel.loss = orig

    def test_consistency_loss_empty_is_zero(self):
        import torch
        from methods.m6_stc_yolo.modules.mean_teacher import TemporalConsistencyLoss
        loss_fn = TemporalConsistencyLoss()
        empty = torch.zeros(0, 6)
        boxes = torch.tensor([[10., 10., 50., 50., 0.9, 0.]])
        self.assertEqual(float(loss_fn([empty], [boxes])), 0.0)
        self.assertEqual(float(loss_fn([boxes], [empty])), 0.0)


class TestSessionFiLM(unittest.TestCase):
    """Section 13 — D11 C3 SessionFiLMGenerator + apply_film."""

    def test_generator_returns_per_level_gamma_beta(self):
        """D11 completion check: SessionFiLMGenerator(8)(rand(2,8)) ->
        dict P3/P4/P5 -> (gamma, beta)."""
        import torch
        from methods.m6_stc_yolo.modules.session_film import SessionFiLMGenerator
        gen = SessionFiLMGenerator(8)
        out = gen(torch.rand(2, 8))
        self.assertEqual(set(out.keys()), {"P3", "P4", "P5"})
        for lv, (gamma, beta) in out.items():
            self.assertEqual(gamma.shape, (2, 128))
            self.assertEqual(beta.shape, (2, 128))

    def test_identity_init_gamma1_beta0(self):
        """Zero-init head => FiLM starts as identity (graceful baseline)."""
        import torch
        from methods.m6_stc_yolo.modules.session_film import SessionFiLMGenerator
        gen = SessionFiLMGenerator(8)
        out = gen(torch.rand(4, 8))
        for _, (gamma, beta) in out.items():
            self.assertTrue(torch.allclose(gamma, torch.ones_like(gamma)))
            self.assertTrue(torch.allclose(beta, torch.zeros_like(beta)))

    def test_per_level_dims_dict(self):
        import torch
        from methods.m6_stc_yolo.modules.session_film import SessionFiLMGenerator
        gen = SessionFiLMGenerator(8, feature_dim={"P3": 64, "P4": 128, "P5": 256})
        out = gen(torch.rand(1, 8))
        self.assertEqual(out["P3"][0].shape, (1, 64))
        self.assertEqual(out["P5"][1].shape, (1, 256))

    def test_apply_film_identity_and_affine(self):
        import torch
        from methods.m6_stc_yolo.modules.session_film import apply_film
        feat = torch.randn(2, 5, 8, 8)
        ident = apply_film(feat, torch.ones(2, 5), torch.zeros(2, 5))
        self.assertTrue(torch.allclose(ident, feat))
        out = apply_film(feat, torch.full((2, 5), 2.0), torch.full((2, 5), 1.0))
        self.assertTrue(torch.allclose(out, 2.0 * feat + 1.0))

    def test_generator_rejects_wrong_metadata_dim(self):
        import torch
        from methods.m6_stc_yolo.modules.session_film import SessionFiLMGenerator
        gen = SessionFiLMGenerator(8)
        with self.assertRaises(ValueError):
            gen(torch.rand(2, 10))


class TestSessionFiLMIntegration(unittest.TestCase):
    """Section 14 — D11 C3 FiLM injected into the real Detect head."""

    def test_build_injects_film_gen_submodule(self):
        from methods.m6_stc_yolo.model import build_stc_yolo
        from methods.m6_stc_yolo.modules.session_film import SessionFiLMGenerator
        yolo = build_stc_yolo(enable_session_film=True)
        detect = yolo.model.model[-1]
        self.assertTrue(getattr(detect, "_film_injected", False))
        self.assertIsInstance(detect.film_gen, SessionFiLMGenerator)
        # registered as submodule -> appears in state_dict
        self.assertTrue(any(k.startswith("model.")
                            and "film_gen" in k for k in yolo.model.state_dict()))

    def test_film_identity_at_init_changes_nothing(self):
        """Zero-init FiLM => stashing metadata must NOT change predictions."""
        import torch
        from methods.m6_stc_yolo.model import build_stc_yolo
        yolo = build_stc_yolo(enable_session_film=True)
        m = yolo.model.eval()
        detect = m.model[-1]
        x = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            detect._film_metadata = None
            y0 = m(x)[0]
            detect._film_metadata = torch.rand(2, 8)
            y1 = m(x)[0]
        self.assertTrue(torch.allclose(y0, y1, atol=1e-5),
                        "identity-init FiLM should not alter outputs")

    def test_film_nonidentity_changes_predictions(self):
        """After perturbing FiLM params off identity, stashed metadata must
        change the cls predictions (proves FiLM is actually in the forward)."""
        import torch
        from methods.m6_stc_yolo.model import build_stc_yolo
        yolo = build_stc_yolo(enable_session_film=True)
        m = yolo.model.eval()
        detect = m.model[-1]
        with torch.no_grad():
            for lv in detect._film_levels:
                detect.film_gen.heads[lv].bias.fill_(0.4)  # gamma=1.4, beta=0.4
        x = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            detect._film_metadata = None
            y0 = m(x)[0]
            detect._film_metadata = torch.rand(2, 8)
            y1 = m(x)[0]
        self.assertFalse(torch.allclose(y0, y1, atol=1e-4),
                         "non-identity FiLM must change predictions")

    def test_film_model_pickle_roundtrips(self):
        """Regression for the smoke crash: best.pt is a pickle; an instance
        method monkey-patch broke unpickling. The module-level subclass must
        save+load cleanly and still apply FiLM."""
        import io
        import torch
        from methods.m6_stc_yolo.model import build_stc_yolo, SessionFiLMDetect
        yolo = build_stc_yolo(enable_session_film=True)
        detect = yolo.model.model[-1]
        self.assertIsInstance(detect, SessionFiLMDetect)
        buf = io.BytesIO()
        torch.save(yolo.model, buf)          # mimics best.pt full-model pickle
        buf.seek(0)
        reloaded = torch.load(buf, weights_only=False)
        rdetect = reloaded.model[-1]
        self.assertIsInstance(rdetect, SessionFiLMDetect)
        reloaded.eval()
        with torch.no_grad():
            _ = reloaded(torch.rand(1, 3, 64, 64))  # must not raise

    def test_set_film_metadata_stashes_on_detect(self):
        """set_film_metadata (used by trainer/validator threading) must find
        the FiLM Detect and stash a (B, 8) tensor for the next forward."""
        from methods.m6_stc_yolo.model import (
            build_stc_yolo, set_film_metadata, _find_film_detect)
        from methods.m6_stc_yolo.data.session_metadata import load_metadata
        yolo = build_stc_yolo(enable_session_film=True)
        names = list(load_metadata().keys())[:3]
        ok = set_film_metadata(yolo.model, [f"{n}.jpg" for n in names])
        self.assertTrue(ok)
        detect = _find_film_detect(yolo.model)
        self.assertIsNotNone(detect)
        self.assertEqual(tuple(detect._film_metadata.shape), (3, 8))

    def test_metadata_helper_lookup(self):
        import torch
        from methods.m6_stc_yolo.model import _film_metadata_for_imfiles
        from methods.m6_stc_yolo.data.session_metadata import load_metadata
        name = next(iter(load_metadata().keys()))
        meta = _film_metadata_for_imfiles([f"/some/path/{name}.jpg"], "cpu")
        self.assertEqual(meta.shape, (1, 8))
        # unknown file -> zero vector, still right shape
        meta2 = _film_metadata_for_imfiles(["/x/nonexistent_zzz.jpg"], "cpu")
        self.assertEqual(meta2.shape, (1, 8))
        self.assertTrue(torch.allclose(meta2, torch.zeros(1, 8)))


class TestDataYamlIntegration(unittest.TestCase):
    """Section 5 — data.yaml resolver (idempotent build, real splits)."""

    def test_resolve_data_yaml_auto_build(self):
        """First call builds the yaml; second call is a no-op (idempotent)."""
        import argparse as _ap
        from methods.m6_stc_yolo.train_one import _resolve_data_yaml
        args = _ap.Namespace(data_yaml=None, split="main_holdout")
        # First call may build (slow if data not yet copied); allow up to 90s
        # by checking the actual file existence after.
        yaml_path = _resolve_data_yaml(args, _PROJECT_ROOT, logger=None)
        self.assertTrue(yaml_path.is_file(), f"yaml not created: {yaml_path}")
        # Idempotent: second call returns same path without rebuilding
        yaml_path2 = _resolve_data_yaml(args, _PROJECT_ROOT, logger=None)
        self.assertEqual(yaml_path, yaml_path2)
        # Content sanity
        import yaml as _y
        info = _y.safe_load(yaml_path.read_text(encoding="utf-8"))
        self.assertIn("path", info)
        self.assertIn("train", info)
        self.assertIn("val", info)
        self.assertIn("test", info)
        self.assertEqual(info["names"], {0: "Water", 1: "Oil"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
