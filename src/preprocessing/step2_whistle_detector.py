"""
STEP 2 - Neural-network whistle detector (U-Net, per-pixel segmentation)
Input : normalized spectrogram [F, T] from step 1
Output: whistle probability map [F, T]  (saved as <stem>_prob.npy, float16)

Training data options (both supported, mixed with --synth_prob):
  (a) REAL labels : for each <stem>.npy put a binary mask <stem>_mask.npy (same shape, uint8)
                    -> rasterize the DCLDE 2022 whistle annotations (time, freq) with
                       rasterize_contour() below (adapt the annotation parser to your files).
  (b) SYNTHETIC   : FM whistles are painted onto real noise crops (idea of Li et al., ref [30]).
                    Good for pre-training / when labels are missing. Real crops may contain
                    unlabeled whistles, so use a LOW synth noise-only share and fine-tune on real labels.

Usage:
  python step2_whistle_detector.py train --spec_dir /data/specs --out detector.pt --epochs 40
  python step2_whistle_detector.py infer --spec_dir /data/specs --ckpt detector.pt --out_dir /data/probs
"""
import argparse, json, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Small U-Net (~1.9M params). Input H,W must be divisible by 2**depth (=16)."""
    def __init__(self, base=16, depth=4):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]           # 16,32,64,128,256
        self.enc = nn.ModuleList([block(1 if i == 0 else ch[i - 1], ch[i]) for i in range(depth + 1)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in range(depth)])
        self.dec = nn.ModuleList([block(ch[i] * 2, ch[i]) for i in range(depth)])
        self.head = nn.Conv2d(ch[0], 1, 1)
        self.depth = depth

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.enc[i](x); skips.append(x); x = Fn.max_pool2d(x, 2)
        x = self.enc[self.depth](x)
        for i in reversed(range(self.depth)):
            x = self.up[i](x)
            x = self.dec[i](torch.cat([x, skips[i]], 1))
        return self.head(x)                                       # logits


# ----------------------------------------------------------------------------
# Labels / synthetic data
# ----------------------------------------------------------------------------
def rasterize_contour(mask, times_s, freqs_hz, meta, width=1):
    """Paint one annotated contour (arrays of time [s] and freq [Hz]) into mask [F,T]."""
    f0 = meta["freqs"][0]
    t_idx = np.round(np.asarray(times_s) / meta["dt"]).astype(int)
    f_idx = np.round((np.asarray(freqs_hz) - f0) / meta["df"]).astype(int)
    # linear interpolation between annotation points so the line is continuous
    tt = np.arange(t_idx.min(), t_idx.max() + 1)
    ff = np.round(np.interp(tt, t_idx, f_idx)).astype(int)
    F, T = mask.shape
    for t, f in zip(tt, ff):
        if 0 <= t < T:
            mask[max(f - width, 0):min(f + width + 1, F), t] = 1


def add_synthetic_whistles(img, rng, max_n=3):
    """Paint random FM whistles into img [F,T] (in place); return binary mask."""
    F, T = img.shape
    mask = np.zeros((F, T), np.uint8)
    for _ in range(rng.integers(0, max_n + 1)):
        D = int(rng.integers(20, min(200, T)))                    # 0.1 - 1.0 s at 5.33 ms/frame
        t0 = int(rng.integers(0, max(T - D, 1)))
        tt = np.arange(D)
        f0 = rng.uniform(0.15 * F, 0.85 * F)
        slope = rng.uniform(-0.25, 0.25)
        amp = rng.uniform(0, 0.25 * F)
        cyc = rng.uniform(0.3, 2.5)
        ph = rng.uniform(0, 2 * np.pi)
        f = f0 + slope * tt + amp * np.sin(2 * np.pi * cyc * tt / D + ph)
        strength = rng.uniform(0.35, 1.0)
        env = np.clip(np.minimum(tt, D - 1 - tt) / 4.0, 0.3, 1.0)  # soft on/off
        for k, (t, fv) in enumerate(zip(tt + t0, np.round(f).astype(int))):
            if t >= T or fv < 1 or fv >= F - 1:
                continue
            img[fv - 1:fv + 2, t] = np.maximum(img[fv - 1:fv + 2, t], strength * env[k] * np.array([0.6, 1.0, 0.6]))
            mask[fv - 1:fv + 2, t] = 1
    return mask


class SpecDataset(Dataset):
    def __init__(self, files, crop_f=128, crop_t=256, synth_prob=0.5, length=20000, seed=0):
        self.files = files; self.cf = crop_f; self.ct = crop_t
        self.synth_prob = synth_prob; self.length = length
        self.arrs = {}            # lazily memory-mapped
        self.seed = seed

    def __len__(self): return self.length

    def _load(self, p):
        if p not in self.arrs:
            self.arrs[p] = np.load(p, mmap_mode="r")
        return self.arrs[p]

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + idx + random.randint(0, 10**6))
        p = self.files[int(rng.integers(len(self.files)))]
        a = self._load(p)
        F, T = a.shape
        f0 = int(rng.integers(0, max(F - self.cf, 1))); t0 = int(rng.integers(0, max(T - self.ct, 1)))
        x = np.zeros((self.cf, self.ct), np.float32)
        seg = np.asarray(a[f0:f0 + self.cf, t0:t0 + self.ct], np.float32)
        x[:seg.shape[0], :seg.shape[1]] = seg
        mpath = Path(str(p).replace(".npy", "_mask.npy"))
        if mpath.exists() and rng.random() >= self.synth_prob:     # real label
            m = np.zeros_like(x, np.uint8)
            ms = np.load(mpath, mmap_mode="r")[f0:f0 + self.cf, t0:t0 + self.ct]
            m[:ms.shape[0], :ms.shape[1]] = ms
        else:                                                      # synthetic label
            m = add_synthetic_whistles(x, rng)
        if rng.random() < 0.5:                                     # time flip augmentation
            x = x[:, ::-1]; m = m[:, ::-1]
        x = x + rng.normal(0, 0.02, x.shape).astype(np.float32)    # light noise jitter
        return (torch.from_numpy(np.ascontiguousarray(x))[None].clamp(0, 1),
                torch.from_numpy(np.ascontiguousarray(m)).float()[None])


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def dice_loss(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3)); den = p.sum((1, 2, 3)) + y.sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (den + eps)).mean()


def train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(Path(a.spec_dir).glob("*_ch*.npy"))
    files = [f for f in files if not f.name.endswith(("_mask.npy", "_prob.npy"))]
    random.Random(0).shuffle(files)
    nval = max(1, int(0.1 * len(files)))
    val_f, tr_f = files[:nval], files[nval:]
    print(f"{len(tr_f)} train files, {len(val_f)} val files, device={dev}")

    tr = DataLoader(SpecDataset(tr_f, synth_prob=a.synth_prob, length=a.iters_per_epoch * a.bs),
                    batch_size=a.bs, num_workers=a.workers, persistent_workers=a.workers > 0)
    va = DataLoader(SpecDataset(val_f, synth_prob=a.synth_prob, length=40 * a.bs, seed=999),
                    batch_size=a.bs, num_workers=a.workers)

    model = UNet().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * a.iters_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0, device=dev))   # whistles are ~1% of pixels
    best = 0
    for ep in range(a.epochs):
        model.train(); run = 0
        for x, y in tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev, enabled=dev == "cuda"):
                lg = model(x)
                loss = bce(lg.float(), y) + dice_loss(lg.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
        model.eval(); d = []
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(dev), y.to(dev)
                d.append(1 - dice_loss(model(x).float(), y).item())
        dsc = float(np.mean(d))
        print(f"ep {ep+1}/{a.epochs} loss {run/a.iters_per_epoch:.4f}  val dice {dsc:.4f}")
        if dsc > best:
            best = dsc; torch.save(model.state_dict(), a.out)
    print("best val dice", best)


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict_spec(model, spec, dev, chunk=512, overlap=64, bs=8):
    F, T = spec.shape
    Fp = (F + 15) // 16 * 16
    step = chunk - overlap
    starts = list(range(0, max(T - overlap, 1), step))
    out = np.zeros((F, T), np.float32); cnt = np.zeros(T, np.float32)
    for i in range(0, len(starts), bs):
        ss = starts[i:i + bs]
        xb = np.zeros((len(ss), 1, Fp, chunk), np.float32)
        for j, s in enumerate(ss):
            seg = spec[:, s:s + chunk]; xb[j, 0, :F, :seg.shape[1]] = seg
        with torch.autocast(dev, enabled=dev == "cuda"):
            p = torch.sigmoid(model(torch.from_numpy(xb).to(dev))).float().cpu().numpy()
        for j, s in enumerate(ss):
            w = min(chunk, T - s)
            out[:, s:s + w] += p[j, 0, :F, :w]; cnt[s:s + w] += 1
    return out / np.maximum(cnt, 1)[None]


def infer(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(dev); model.load_state_dict(torch.load(a.ckpt, map_location=dev)); model.eval()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files = [f for f in sorted(Path(a.spec_dir).glob("*_ch*.npy"))
             if not f.name.endswith(("_mask.npy", "_prob.npy"))]
    for k, f in enumerate(files, 1):
        prob = predict_spec(model, np.load(f).astype(np.float32), dev)
        np.save(out_dir / (f.stem + "_prob.npy"), prob.astype(np.float16))
        print(f"[{k}/{len(files)}] {f.name}  max prob {prob.max():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--spec_dir", required=True); t.add_argument("--out", default="detector.pt")
    t.add_argument("--epochs", type=int, default=20); t.add_argument("--bs", type=int, default=32)
    t.add_argument("--lr", type=float, default=2e-3); t.add_argument("--iters_per_epoch", type=int, default=300)
    t.add_argument("--synth_prob", type=float, default=0.5); t.add_argument("--workers", type=int, default=0)
    i = sp.add_parser("infer")
    i.add_argument("--spec_dir", required=True); i.add_argument("--ckpt", required=True)
    i.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else infer(a)
