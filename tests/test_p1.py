"""Tests for the code of research/log.md section 16 (P1: a stream buffer in the decoder's temporal convolutions)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from surge.data import ClipDataset, density_small, load_gray  # noqa: E402
from surge.model import Decoder  # noqa: E402
from tests.test_inv import _fake_video, _taps  # noqa: E402


def _stream(prev=None):
    return {"in": {} if prev is None else prev, "out": {}}


def test_empty_stream_equals_no_stream():
    """With no buffer from a previous window, the stream path is the default zero padding."""
    for kw in ({}, {"norm_per_frame": True}, {"topdown": False}, {"temporal": False}):
        torch.manual_seed(0)
        m = Decoder(32, 16, **kw).eval()
        taps = _taps(t=8)
        a = m(taps, T_out=16)["logits"]
        st = _stream()
        b = m(taps, T_out=16, stream=st)["logits"]
        assert torch.allclose(a, b, atol=1e-6, rtol=0), kw
        n_conv = 2 if kw.get("topdown") is False else 6
        assert len(st["out"]) == (0 if kw.get("temporal") is False else n_conv), kw


def test_buffer_holds_last_step_and_reaches_six_steps():
    """Per-frame norms: the buffer of window A changes window B's first six time steps (six 3x1x1 convolutions) and
    leaves the last two unchanged, so output positions 13-15 of 16 are identical with and without the buffer."""
    torch.manual_seed(0)
    m = Decoder(32, 16, norm_per_frame=True, prior=False).eval()
    ta, tb = _taps(t=8, seed=1), _taps(t=8, seed=2)
    st = _stream()
    m(ta, T_out=16, stream=st)
    assert set(st["out"]) == {"r16.0", "r16.1", "r8.0", "r8.1", "r4.0", "r4.1"}
    assert all(b.shape[2] == 1 for b in st["out"].values())
    with_buf = m(tb, T_out=16, stream=_stream(st["out"]))["logits"]
    without = m(tb, T_out=16)["logits"]
    diff = (with_buf - without).abs().amax(dim=(0, 2, 3))
    assert (diff[:13] > 1e-6).all()
    assert torch.equal(diff[13:], torch.zeros(3))


def test_buffer_is_last_step_of_conv_input():
    torch.manual_seed(0)
    m = Decoder(32, 16, norm_per_frame=True).eval()
    blk = m.r16[0]
    x = torch.randn(2, 16, 8, 4, 4)
    st = _stream()
    blk(x, st, "k")
    y = torch.nn.functional.gelu(blk.n1(blk.s(x)))
    assert torch.equal(st["out"]["k"], y[:, :, -1:])


def test_pair_forward_streams_a_into_b():
    from scripts.train_multi import pair_forward
    torch.manual_seed(0)
    dec = Decoder(32, 16, norm_per_frame=True).eval()
    v = SimpleNamespace(stream=True, module=dec, sel=lambda t: t)
    ta, tb = _taps(B=2, t=8, seed=1), _taps(B=2, t=8, seed=2)
    taps = [torch.cat([a, b]) for a, b in zip(ta, tb)]
    out = pair_forward(v, taps, 16, 2)["logits"]
    st = _stream()
    oa = dec(ta, T_out=16, stream=st)["logits"]
    ob = dec(tb, T_out=16, stream=_stream(st["out"]))["logits"]
    assert torch.equal(out, torch.cat([oa, ob]))
    v.stream = False
    assert torch.equal(pair_forward(v, taps, 16, 2)["logits"], dec(taps, T_out=16)["logits"])


def test_pair_items(tmp_path):
    v = _fake_video(tmp_path, n=40)
    kw = dict(T=8, stride=2, size=(32, 32), out_size=(8, 8), clips_per_video=3, seed=0)
    ds = ClipDataset([v], pair=True, hflip=False, **kw)
    starts = set()
    for i in range(3):
        it = ds[i]
        assert it["frames"].shape == (16, 3, 32, 32) and it["density"].shape == (16, 8, 8)
        assert it["fixations"].shape == (16, 8, 8)
        # the frames are constant images with value 5*i: recover their indices
        ref = [density_small(load_gray(v.maps[j]), (8, 8)) for j in range(40)]
        idx = [next(j for j in range(40) if np.array_equal(ref[j], it["density"][k].numpy())) for k in range(16)]
        assert idx == list(range(idx[0], idx[0] + 32, 2))  # 2T sampled frames, all inside the video
        assert 0 <= idx[0] <= 40 - 1 - 30
        starts.add(idx[0])
    flipped = ClipDataset([v], pair=True, hflip=True, **kw)
    for i in range(3):
        a, b = ds[i], flipped[i]
        same = torch.equal(a["density"], b["density"])
        assert same or torch.equal(a["density"].flip(-1), b["density"])  # one flip for the whole pair
    plain = ClipDataset([v], **kw)
    assert plain[0]["frames"].shape == (8, 3, 32, 32)  # the default item is unchanged in shape
