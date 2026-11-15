"""Tests for the code of research/log.md section 15 (decoder ablations, linear readouts, static clips, probes)."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from surge.data import ClipDataset, Video  # noqa: E402
from surge.infer import predict_video, static_windows  # noqa: E402
from surge.model import Decoder, LinearReadout  # noqa: E402


def _old_model_module(tmp_path):
    """surge/model.py as committed before section 15 (git HEAD~ of the file is not needed: the first commit of
    section 15's code changes it; we take the version at the commit of the section 15 rules, 5e6fb30)."""
    src = subprocess.run(["git", "show", "5e6fb30:surge/model.py"], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout
    pkg = tmp_path / "oldsurge"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "model.py").write_text(src.replace("from .backbones import Backbone", "Backbone = object"))
    spec = importlib.util.spec_from_file_location("oldsurge.model", pkg / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["oldsurge.model"] = mod
    spec.loader.exec_module(mod)
    return mod


def _taps(B=2, d=32, t=4, h=4, w=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, d, t, h, w, generator=g) for _ in range(4)]


def test_default_decoder_unchanged(tmp_path):
    old = _old_model_module(tmp_path)
    torch.manual_seed(0)
    a = old.Decoder(32, 16)
    torch.manual_seed(0)
    b = Decoder(32, 16)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)  # same modules, same order, same initialisation
    taps = _taps()
    for mode in ("train", "eval"):
        getattr(a, mode)()
        getattr(b, mode)()
        torch.manual_seed(1)
        oa = a(taps, T_out=8)
        torch.manual_seed(1)
        ob = b(taps, T_out=8)
        assert set(oa) == set(ob)
        for k in oa:
            assert torch.equal(oa[k], ob[k]), k


@pytest.mark.parametrize("kw", [{"tap_mode": "last"}, {"temporal": False}, {"topdown": False}, {"prior": False, "aux": False}])
def test_decoder_variants_shapes(kw):
    m = Decoder(32, 16, **kw)
    taps = _taps()
    m.train()
    out = m(taps, T_out=8)
    assert out["logits"].shape == (2, 8, 16, 16)
    aux = {k for k in out if k.startswith("aux")}
    if kw.get("aux") is False:
        assert aux == set()
    elif kw.get("topdown") is False:
        assert aux == {"aux16"}
    else:
        assert aux == {"aux16", "aux8"}
    for k in aux:
        assert out[k].shape == out["logits"].shape
    m.eval()
    assert set(m(taps, T_out=8)) == {"logits"}


def test_temporal_off_has_no_temporal_mixing():
    m = Decoder(32, 16, temporal=False, prior=False, aux=False).eval()
    taps = _taps()
    base = m(taps)["logits"]
    taps2 = [t.clone() for t in taps]
    for t in taps2:
        t[:, :, 2] += 5.0  # change one time step only
    out = m(taps2)["logits"]
    changed = (out - base).abs().amax(dim=(0, 2, 3)) > 1e-6
    assert changed.tolist() == [False, False, True, False]
    m3 = Decoder(32, 16, prior=False, aux=False).eval()  # the default decoder does mix time
    d3 = (m3(taps2)["logits"] - m3(taps)["logits"]).abs().amax(dim=(0, 2, 3)) > 1e-6
    assert d3.sum() > 1


def test_last_tap_ignores_other_taps():
    m = Decoder(32, 16, tap_mode="last").eval()
    taps = _taps()
    o1 = m(taps)["logits"]
    taps[0] = taps[0] + 3.0
    assert torch.equal(m(taps)["logits"], o1)


def test_linear_readout_and_prior_only():
    taps = _taps(d=8)
    lin = LinearReadout(32)
    out = lin(taps, T_out=8)["logits"]
    assert out.shape == (2, 8, 16, 16)
    probe = LinearReadout(8)
    assert probe([taps[1]], T_out=8)["logits"].shape == (2, 8, 16, 16)
    p0 = LinearReadout(8, zero=True)
    a = p0([taps[0]], T_out=8)["logits"]
    b = p0([taps[0] * 10], T_out=8)["logits"]
    assert torch.equal(a, b)  # content-free
    assert sum(p.numel() for p in p0.parameters()) == sum(p.numel() for p in p0.prior.parameters())


def _fake_video(tmp_path, n=40, name="v"):
    d = tmp_path / name
    for sub in ("frames", "maps", "fix"):
        (d / sub).mkdir(parents=True)
    rng = np.random.default_rng(0)
    fr, mp, fx = [], [], []
    for i in range(n):
        im = np.full((36, 64, 3), i * 5 % 255, np.uint8)
        cv2.imwrite(str(d / "frames" / f"{i:04d}.jpg"), im)
        m = (rng.random((36, 64)) * 255).astype(np.uint8)
        cv2.imwrite(str(d / "maps" / f"{i:04d}.png"), m)
        f = np.zeros((36, 64), np.uint8)
        f[rng.integers(0, 36), rng.integers(0, 64)] = 255
        cv2.imwrite(str(d / "fix" / f"{i:04d}.png"), f)
        fr.append(str(d / "frames" / f"{i:04d}.jpg"))
        mp.append(str(d / "maps" / f"{i:04d}.png"))
        fx.append(str(d / "fix" / f"{i:04d}.png"))
    return Video(name, fr, mp, fx)


def test_static_clips_and_default_unchanged(tmp_path):
    v = _fake_video(tmp_path)
    kw = dict(T=8, stride=2, size=(32, 32), out_size=(8, 8), clips_per_video=3, seed=0)
    a = ClipDataset([v], **kw)
    b = ClipDataset([v], static=False, **kw)
    s = ClipDataset([v], static=True, **kw)
    for i in range(3):
        ia, ib, is_ = a[i], b[i], s[i]
        assert torch.equal(ia["frames"], ib["frames"]) and torch.equal(ia["density"], ib["density"])
        fr = is_["frames"]
        assert all(torch.equal(fr[0], fr[k]) for k in range(8))  # one frame repeated
        de = is_["density"]
        assert all(torch.equal(de[0], de[k]) for k in range(4))
        # the static frame is one of the default clip's target frames
        assert any(torch.equal(de[0], ia["density"][k]) for k in range(4))


class _Echo(torch.nn.Module):
    """Fake model: logits at position k = mean pixel value of input frame k (constant map)."""
    uses_audio = False

    def forward(self, x, surprise=None, audio=None):
        B, T = x.shape[:2]
        v = x.mean(dim=(2, 3, 4))  # B,T
        return {"logits": v[:, :, None, None].expand(B, T, 4, 4).clone()}


def test_static_windows_predict(tmp_path):
    v = _fake_video(tmp_path, n=12)
    assert [w for w in static_windows(12, 8, [0, 5, 11, 12])] == [([0] * 8, [(0, -1.0)]), ([5] * 8, [(5, -1.0)]),
                                                                   ([11] * 8, [(11, -1.0)])]
    out = dict(predict_video(_Echo(), v, 8, 2, (16, 16), torch.device("cpu"), frames=[0, 5], static=True))
    assert set(out) == {0, 5}
    assert out[0].shape == (4, 4)


# ---- section 15M: dataset-bias transform (scripts/inv_bias.py)
def test_bias_transform_identity_equals_to_native():
    import numpy as np
    from scripts.inv_bias import transform
    from surge.infer import to_native
    rng = np.random.default_rng(0)
    small = rng.random((64, 64)).astype(np.float32)
    for sigma in (0.0, 8.0):
        a = transform(small, (360, 640), sigma, 1.0, 0.0)
        b = to_native(small, (360, 640), sigma)
        assert a.dtype == b.dtype and np.array_equal(a, b)


def test_bias_fit_recovers_known_transform():
    import cv2
    import numpy as np
    from scripts.inv_bias import FIT_WIDTH, centre_r2, fit
    rng = np.random.default_rng(1)
    W, H, sigma0 = 640, 360, 8.0
    h = int(round(H * FIT_WIDTH / W))
    items, vid = [], []
    for i in range(24):
        small = np.zeros((64, 64), np.float32)
        for _ in range(3):
            small[rng.integers(4, 60), rng.integers(4, 60)] = rng.random() + 0.5
        small = cv2.GaussianBlur(small, (0, 0), 2.0) + 1e-4
        b = cv2.resize(small, (FIT_WIDTH, h), interpolation=cv2.INTER_LINEAR)
        g = cv2.GaussianBlur(b, (0, 0), 2.0 * sigma0 * FIT_WIDTH / W).astype(np.float64)
        g = g / g.sum()
        g = np.power(g, 1.5) * np.exp(-4.0 * centre_r2(h, FIT_WIDTH))
        items.append((b, g.astype(np.float32), W))
        vid.append(i // 4)
    (m, gm, k), info = fit(items, np.array(vid), sigma0)
    assert (m, gm, k) == (2.0, 1.5, 4.0)
    assert info["fit_cc_best"] > 0.999 and info["fit_cc_identity"] < info["fit_cc_best"]


def test_prep_mvva_fixation_scaling():
    """Per-axis scaling reproduces the first layout (video pixels / 2) for 1280x720 videos and maps 1920x1080."""
    from scripts.prep_mvva import fixation_map
    p = np.array([[0.0, 1279.0, 640.4, np.nan, -3.0, 2000.0], [0.0, 719.0, 360.9, 5.0, 10.0, 100.0]])
    old = np.zeros((360, 640), np.uint8)
    x, y = np.floor(p[0] / 2.0), np.floor(p[1] / 2.0)
    keep = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < 640) & (y >= 0) & (y < 360)
    old[y[keep].astype(int), x[keep].astype(int)] = 255
    assert np.array_equal(fixation_map(p, 640 / 1280, 360 / 720), old)
    m = fixation_map(np.array([[1919.0, 960.0], [1079.0, 540.0]]), 640 / 1920, 360 / 1080)
    assert m[359, 639] == 255 and m[180, 320] == 255 and m.sum() == 2 * 255


def test_prep_avimos_grid_equals_maxpool():
    """Fixations mapped to the map grid equal 2x2 max-pooling for landscape videos at twice the map size."""
    from scripts.prep_avimos_fix import to_grid
    rng = np.random.default_rng(0)
    f = (rng.random((108, 192)) > 0.995).astype(np.uint8) * 255
    pooled = (f > 0).reshape(54, 2, 96, 2).any(axis=(1, 3)).astype(np.uint8) * 255
    assert np.array_equal(to_grid(f, 54, 96), pooled)
    g = np.zeros((192, 108), np.uint8)  # portrait fixations onto a landscape grid: per-axis scaling
    g[191, 107] = g[0, 0] = 255
    t = to_grid(g, 54, 96)
    assert t[53, 95] == 255 and t[0, 0] == 255 and t.sum() == 2 * 255


def test_mvva_face_boxes_scale_per_axis():
    from scripts.inv_mvva_faces import masks
    t = np.array([[[100.0, 100.0, 300.0, 300.0], [900.0, 500.0, 1100.0, 700.0]]])  # (frames, faces, 4)
    k = np.array([[1], [0]])  # face 0 talks in frame 0
    mt, mn = masks(t, k, 0)  # default: a 1280x720 video, boxes halved
    assert mt[50:150, 50:150].all() and not mt[160, 160] and mn[250:350, 450:550].all()
    mt2, mn2 = masks(t * 1.5, k, 0, (1920, 1080))  # the same boxes in a 1920x1080 video
    assert np.array_equal(mt, mt2) and np.array_equal(mn, mn2)
