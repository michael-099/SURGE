import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surge import metrics as M  # noqa: E402
from surge.data import inference_windows, window_indices  # noqa: E402
from surge.lora import LoRALinear, inject_lora  # noqa: E402
from surge.losses import saliency_loss  # noqa: E402
from surge.model import SCM, CenterPrior, Decoder, central  # noqa: E402


def _blob(h=36, w=64, cy=18, cx=32, s=5.0):
    y, x = np.mgrid[:h, :w]
    return np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * s * s))


def test_metric_identities():
    g = _blob()
    fix = np.zeros_like(g, dtype=np.uint8)
    fix[18, 32] = 255
    fix[17, 30] = 255
    assert M.cc(g, g) == pytest.approx(1.0)
    assert M.sim(g, g) == pytest.approx(1.0)
    assert M.kld(g, g) == pytest.approx(0.0, abs=1e-9)
    assert M.nss(g, fix) > 3
    assert M.auc_judd(g, fix) > 0.95
    rng = np.random.default_rng(0)
    many = np.zeros((90, 160), np.uint8)
    many[rng.integers(0, 90, 300), rng.integers(0, 160, 300)] = 255
    assert M.auc_judd(rng.random((90, 160)), many) == pytest.approx(0.5, abs=0.05)
    assert M.cc(g, _blob(cx=10)) < M.cc(g, _blob(cx=30))


def test_shuffled_auc_center_bias():
    g = _blob()
    fix = np.zeros_like(g, dtype=np.uint8)
    fix[18, 32] = 255
    other = np.zeros_like(fix)
    other[18, 31] = 255  # other videos also fixate the centre -> a centre map gets no credit
    other[5, 5] = 255
    a = M.auc_shuffled(g, fix, other)
    assert 0.4 <= a <= 1.0
    # a map peaked away from the other videos' fixations scores higher than a centre map
    other2 = np.zeros_like(fix)
    other2[5:10, 5:10] = 255
    assert M.auc_shuffled(g, fix, other2) > 0.9


def test_auc_ranks_ties():
    assert M._auc_from_scores(np.array([1.0, 1.0]), np.array([1.0, 0.0])) == pytest.approx(0.75)


@pytest.mark.parametrize("n", [1, 5, 31, 32, 33, 450])
@pytest.mark.parametrize("stride", [1, 2, 4])
def test_inference_windows_cover_each_frame_once(n, stride):
    T = 16
    seen = []
    for idx, assign in inference_windows(n, T, stride):
        assert len(idx) == T and all(0 <= j < n for j in idx)
        for f, q in assign:
            assert T // 4 <= q < 3 * T // 4  # within the central half
            seen.append(f)
    assert sorted(seen) == list(range(n))


def test_window_clamp():
    assert window_indices(10, -4, 4, 2) == [0, 0, 0, 2]


def _fake_taps(B=2, D=32, t=8, g=4):
    return [torch.randn(B, D, t, g, g) for _ in range(4)]


def test_decoder_shapes_and_scm_noop():
    torch.manual_seed(0)
    dec = Decoder(d=32, c=32, use_scm=True).eval()
    taps = _fake_taps()
    s1 = torch.randn(2, 2, 8, 4, 4)
    s2 = torch.randn(2, 2, 8, 4, 4)
    o1 = dec(taps, s1, T_out=16)["logits"]
    o2 = dec(taps, s2, T_out=16)["logits"]
    assert o1.shape == (2, 16, 16, 16)
    assert torch.allclose(o1, o2)  # zero-initialized SCM ignores surprise at init
    dec.train()
    o = dec(taps, s1, T_out=16)
    assert set(o) == {"logits", "aux16", "aux8"} and o["aux8"].shape == o1.shape


def test_prior_is_normalized():
    p = CenterPrior()
    lp = p(16, 16, "cpu") / p.strength
    assert torch.logsumexp(lp.flatten(), 0).item() == pytest.approx(0.0, abs=1e-5)
    assert lp[8, 8] > lp[0, 0]


def test_lora_noop_at_init():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(8, 8))
    x = torch.randn(3, 8)
    y0 = m(x)
    names = inject_lora(m, r"0", r=4)
    assert names == ["0"] and isinstance(m[0], LoRALinear)
    assert torch.allclose(m(x), y0)


def test_loss_prefers_correct_map():
    B, T, h, w = 1, 2, 8, 8
    y = torch.zeros(B, T, h, w)
    y[..., 3, 3] = 1.0
    fx = y.clone()
    good = torch.full((B, T, h, w), -5.0)
    good[..., 3, 3] = 5.0
    bad = torch.full((B, T, h, w), -5.0)
    bad[..., 6, 6] = 5.0
    lg, _ = saliency_loss(good, y, fx)
    lb, _ = saliency_loss(bad, y, fx)
    assert lg < lb


def test_central():
    x = torch.arange(16).view(1, 16)
    assert central(x, 16).tolist() == [list(range(4, 12))]


def test_stavis_layout_and_mat_fixations():
    import os
    import pytest
    if not os.path.isdir("D:/Datasets/AVAD/extracted"):
        pytest.skip("STAViS data not on this machine")
    from surge.data import load_gray, parse_stavis, stavis_split_videos
    assert parse_stavis("avad_f2") == ("avad", 2) and parse_stavis("dhf1k") is None
    root = os.path.join(os.path.dirname(__file__), "..", "configs", "splits")
    test = stavis_split_videos("avad_f1", "val", root)
    train = stavis_split_videos("avad_f1", "train", root)
    hold = stavis_split_videos("avad_f1", "holdout", root)
    assert len(test) == 15 and len(train) + len(hold) == 30 and not {v.name for v in hold} & {v.name for v in train}
    assert not {v.name for v in test} & {v.name for v in train + hold}
    v = test[0]
    assert v.skip_empty and len(v.frames) == len(v.maps) == len(v.fixations) == len(v)
    for a, b, c in zip(v.frames[:3], v.maps[:3], v.fixations[:3]):  # same frame number in all three files
        assert a[-9:-4] == b[-9:-4] == c[-9:-4]
    f = load_gray(v.fixations[10])
    assert f.dtype == np.uint8 and set(np.unique(f)) <= {0, 255} and f.shape == load_gray(v.maps[10]).shape
