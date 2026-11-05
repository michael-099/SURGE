"""Independent replication of the SURGE reality-check diagnostics on real V-JEPA 2 ViT-L weights.

For each clip: 16 frames (8 context + 8 target = 4 + 4 tubelets), squashed to 256x256.
Computes, per clip and frame rate:
  err_true      mean L1(LN(pred), LN(target)) with the clip's own (leak-free) past
  err_wrong     same, with another clip's past
  err_zero      predicting zeros (LN space mean)
  err_copy      copying last context tubelet's target features forward
  horizon       per-target-tubelet mean error (h=1..4)
  r_atyp        spatial corr of error map with target atypicality |z - mean_frame(z)|
  r_wrongmap    spatial corr of true-past map with wrong-past map
  r_motion      spatial corr with pixel frame-difference energy
  r_leaky       spatial corr of leak-free map with HF-default (leaky) map
  r_center      spatial corr with a centered Gaussian (center-bias check)
"""
import sys, json, glob, os
import numpy as np, torch, torch.nn.functional as F, av
from transformers import AutoModel

torch.set_grad_enabled(False)
VID_DIR = sys.argv[1]
OUT = sys.argv[2]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_frames(path, fps, end_t, n=16, size=256):
    c = av.open(path)
    s = c.streams.video[0]
    frames, times = [], []
    for fr in c.decode(s):
        frames.append(fr.to_ndarray(format="rgb24"))
        times.append(float(fr.pts * s.time_base))
    c.close()
    times = np.array(times)
    want = end_t - (np.arange(n)[::-1]) / fps
    idx = [int(np.argmin(np.abs(times - w))) for w in want]
    x = torch.from_numpy(np.stack([frames[i] for i in idx])).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return ((x - MEAN) / STD), x  # (16,3,256,256) normalized, raw


def ln(t):
    return F.layer_norm(t.float(), t.shape[-1:])


def corr(a, b):
    a = a.flatten() - a.mean(); b = b.flatten() - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def main():
    model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256").eval()
    vids = sorted(glob.glob(os.path.join(VID_DIR, "*.mp4")))
    hs = model(pixel_values_videos=torch.randn(1, 2, 3, 256, 256), skip_predictor=True, output_hidden_states=True)
    print("n_hidden_states", len(hs.hidden_states), "last==LN?",
          float((model.encoder.layernorm(hs.hidden_states[-1]) - hs.last_hidden_state).abs().max()), flush=True)
    results = []
    g = torch.arange(16).float() - 7.5
    center = torch.exp(-(g[:, None] ** 2 + g[None, :] ** 2) / (2 * 4.0 ** 2))
    for fps, end_t in [(4, 8.0), (15, 6.0)]:
        cache = []
        for v in vids:
            et = end_t if "sintel" not in v else 30.0
            x, raw = load_frames(v, fps, et)
            X = x.unsqueeze(0)  # (1,16,3,H,W)
            full = model(pixel_values_videos=X, skip_predictor=True)
            z = ln(full.last_hidden_state)[0]  # (2048,1024)
            ctx = model(pixel_values_videos=X[:, :8], skip_predictor=True).last_hidden_state  # (1,1024,1024)
            # pixel motion energy per target tubelet on 16x16 grid
            d = (raw[1:] - raw[:-1]).abs().mean(1, keepdim=True)  # (15,1,H,W)
            d = F.adaptive_avg_pool2d(d, 16)[:, 0]
            mot = torch.stack([d[7 + 2 * k: 9 + 2 * k].mean(0) for k in range(4)])  # frames 8..15
            cache.append(dict(name=os.path.basename(v), X=X, z=z, ctx=ctx, mot=mot,
                              leaky_ctx=full.last_hidden_state[:, :1024]))
            print("encoded", fps, os.path.basename(v), flush=True)
        ctx_ids = [torch.arange(1024).unsqueeze(0)]
        tgt_ids = [torch.arange(1024, 2048).unsqueeze(0)]

        def predict(c):
            out = model.predictor(encoder_hidden_states=c, context_mask=ctx_ids, target_mask=tgt_ids)
            return ln(out.last_hidden_state)[0]  # (1024,1024)

        for i, c in enumerate(cache):
            zt = c["z"][1024:].view(4, 256, 1024)
            zc_last = c["z"][768:1024]
            p_true = predict(c["ctx"]).view(4, 256, 1024)
            p_wrong = predict(cache[(i + 1) % len(cache)]["ctx"]).view(4, 256, 1024)
            p_leaky = predict(c["leaky_ctx"]).view(4, 256, 1024)
            e_true = (p_true - zt).abs().mean(-1)  # (4,256)
            e_wrong = (p_wrong - zt).abs().mean(-1)
            e_leaky = (p_leaky - zt).abs().mean(-1)
            e_zero = zt.abs().mean(-1)
            e_copy = (zc_last.unsqueeze(0) - zt).abs().mean(-1)
            atyp = (zt - zt.mean(1, keepdim=True)).abs().mean(-1)
            mot = c["mot"].view(4, 256)
            per = lambda a, b: float(np.mean([corr(a[k], b[k]) for k in range(4)]))
            r = dict(fps=fps, clip=c["name"],
                     err_true=float(e_true.mean()), err_wrong=float(e_wrong.mean()),
                     err_zero=float(e_zero.mean()), err_copy=float(e_copy.mean()),
                     err_leaky=float(e_leaky.mean()),
                     horizon=[float(e_true[k].mean()) for k in range(4)],
                     r_atyp=per(e_true, atyp), r_wrongmap=per(e_true, e_wrong),
                     r_motion=per(e_true, mot), r_leaky=per(e_true, e_leaky),
                     r_center=per(e_true, center.flatten().expand(4, 256)),
                     r_ctxdiff_atyp=per(e_true - e_wrong, atyp), r_ctxdiff_motion=per(e_true - e_wrong, mot))
            results.append(r)
            print(json.dumps(r), flush=True)
    json.dump(results, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
