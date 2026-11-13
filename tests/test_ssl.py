"""CPU tests of the localisation-map condition (research/log.md section 14F)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surge.data import Video, ssl_condition, ssl_maps  # noqa: E402
from surge.model import Decoder  # noqa: E402


def _fake_taps(B=2, d=32, t=8, g=8):
    torch.manual_seed(1)
    return [torch.randn(B, d, t, g, g) for _ in range(4)]


def test_ssl_condition_values_and_flip():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(20, 7, 7)).astype(np.float32)
    idx = list(range(2, 18))
    c = ssl_condition(M, idx, (16, 16))
    assert c.shape == (2, 8, 16, 16) and c.dtype == np.float32
    assert np.all(c[1] == 1.0)
    # tubelet 0 = mean of the two standardised, resized frames 2 and 3
    z = [(M[k] - M[k].mean()) / (M[k].std() + 1e-6) for k in (2, 3)]
    ref = np.mean([cv2.resize(x, (16, 16), interpolation=cv2.INTER_LINEAR) for x in z], 0)
    assert np.allclose(c[0, 0], ref, atol=1e-5)
    assert np.allclose(ssl_condition(M, idx, (16, 16), flip=True)[0], c[0][..., ::-1])  # flips with the frames
    assert not ssl_condition(None, idx, (16, 16)).any()  # no audio: both channels zero


def test_ssl_maps_padding(tmp_path):
    a = np.arange(5 * 49, dtype=np.float16).reshape(5, 7, 7)
    np.save(tmp_path / "v.npy", a)
    v = Video("v", [f"{i}.jpg" for i in range(7)], ssl=str(tmp_path / "v.npy"))
    m = ssl_maps(v)
    assert m.shape == (7, 7, 7) and m.dtype == np.float32 and np.array_equal(m[6], m[4])
    assert ssl_maps(Video("w", ["0.jpg"])) is None


def test_ssl_decoder_starts_as_visual_model():
    torch.manual_seed(0)
    vis = Decoder(d=32, c=32).eval()
    cond = Decoder(d=32, c=32, use_scm=True).eval()
    r = cond.load_state_dict(vis.state_dict(), strict=False)  # visual checkpoint -> new zero-init modules
    assert not r.unexpected_keys and r.missing_keys and all(k.startswith(("phi.", "scm.")) for k in r.missing_keys)
    taps = _fake_taps()
    o0 = vis(taps, T_out=16)["logits"]
    for s in (torch.randn(2, 2, 8, 16, 16), torch.zeros(2, 2, 8, 16, 16)):
        assert torch.allclose(cond(taps, s, T_out=16)["logits"], o0, atol=1e-6)


def test_predict_video_feeds_the_condition(tmp_path):
    from surge.infer import predict_video
    n = 20
    frames = []
    for i in range(n):
        p = tmp_path / f"img_{i + 1:05d}.jpg"
        cv2.imwrite(str(p), np.full((24, 32, 3), 10 * i % 255, np.uint8))
        frames.append(str(p))
    M = np.random.default_rng(1).normal(size=(n, 7, 7)).astype(np.float16)
    np.save(tmp_path / "m.npy", M)
    seen = []

    class Fake(torch.nn.Module):
        ssl = True

        def forward(self, x, surprise=None):
            seen.append(surprise)
            B, T = x.shape[:2]
            return {"logits": torch.zeros(B, T, 8, 8)}

    v = Video("v", frames, ssl=str(tmp_path / "m.npy"))
    out = dict(predict_video(Fake(), v, 16, 2, (32, 32), torch.device("cpu"), batch=2))
    assert sorted(out) == list(range(n))
    assert all(s is not None and s.shape[1:] == (2, 8, 2, 2) and torch.all(s[:, 1] == 1) for s in seen)


def test_ezvsl_reimplementation_matches_its_forward():
    from scripts.ssl_maps import EZVSL
    torch.manual_seed(0)
    m = EZVSL().eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 224, 224)
        ref = F.normalize(m.img_proj(m.imgnet(x).unflatten(1, (512, 7, 7))), dim=1)  # the original forward
        assert torch.allclose(m.image(x), ref, atol=1e-6)
        s = torch.randn(2, 1, 257, 300)
        assert torch.allclose(m.audio(s), F.normalize(m.aud_proj(m.audnet(s)), dim=1), atol=1e-6)
