"""
STEP 2b - Classical (training-free) whistle detector: oriented line-filter bank
Same input/output as step2_whistle_detector.py infer:  spec [F,T] in [0,1]  ->  <stem>_prob.npy [F,T]

Why it exists
  * works immediately on real data (no labels, no GPU) - useful as a baseline for the U-Net
  * its output can be used as pseudo-labels to train the U-Net on your own recordings
  * a whistle is a thin ridge that is locally a straight line, so we correlate the spectrogram with
    zero-mean line kernels at many slopes (bins/frame) and keep the best response per pixel.

Usage:
  python step2b_classical_detector.py --spec_dir SPECS --out_dir PROBS
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

SLOPES = [-3.0, -2.0, -1.25, -0.75, -0.4, -0.15, 0.0, 0.15, 0.4, 0.75, 1.25, 2.0, 3.0]   # freq bins per frame
HALF_T = 4            # kernel spans 2*HALF_T+1 = 9 frames (~48 ms)
FLANK = 3             # side lobes at +-3 bins


def _line_kernel(slope, half_t=HALF_T, flank=FLANK):
    rows = int(np.ceil(abs(slope) * half_t)) + flank + 2
    k = np.zeros((2 * rows + 1, 2 * half_t + 1), np.float32)
    for j, t in enumerate(range(-half_t, half_t + 1)):
        for off, w in ((0.0, 1.0), (-flank, -0.5), (flank, -0.5)):
            y = rows + slope * t + off
            y0 = int(np.floor(y)); fr = y - y0                    # bilinear splat
            k[y0, j] += w * (1 - fr); k[y0 + 1, j] += w * fr
    k[k > 0] /= k[k > 0].sum(); k[k < 0] /= -k[k < 0].sum()      # +1 / -1 balanced -> zero mean
    return k


_KERNELS = None


def detect(spec, z0=5.0, width=1.2):
    """spec [F,T] float32 in [0,1] -> probability map [F,T] float32."""
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = [_line_kernel(s) for s in SLOPES]
    x = ndi.gaussian_filter(spec.astype(np.float32), 0.6)
    R = np.full_like(x, -np.inf)
    for k in _KERNELS:
        np.maximum(R, ndi.correlate(x, k, mode="nearest"), out=R)
    med = np.median(R)
    sig = 1.4826 * np.median(np.abs(R - med)) + 1e-6              # robust noise sigma
    z = (R - med) / sig
    return (1.0 / (1.0 + np.exp(-(z - z0) / width))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_dir", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--z0", type=float, default=5.0, help="detection level in noise sigmas")
    ap.add_argument("--mask_thr", type=float, default=None,
                    help="if set (e.g. 0.8), also write <stem>_mask.npy pseudo-labels next to the spectrograms "
                         "so step2_whistle_detector.py train can learn from real whistle shapes")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    files = [f for f in sorted(Path(a.spec_dir).glob("*_ch*.npy")) if not f.name.endswith(("_mask.npy", "_prob.npy"))]
    for i, f in enumerate(files, 1):
        p = detect(np.load(f).astype(np.float32), z0=a.z0)
        np.save(out / (f.stem + "_prob.npy"), p.astype(np.float16))
        if a.mask_thr is not None:
            np.save(f.with_name(f.stem + "_mask.npy"), (p > a.mask_thr).astype(np.uint8))
        print(f"[{i}/{len(files)}] {f.name}  pixels>0.5: {(p > 0.5).mean() * 100:.3f}%")


if __name__ == "__main__":
    main()
