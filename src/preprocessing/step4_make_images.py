"""
STEP 4 - Whistle time-frequency vectors -> 128x128 grayscale whistle images (paper Sec. II-B/C, Fig. 2-3)

For each whistle contour f(t) [Hz, one value per 5.33 ms frame]:
  1. remove the center frequency  fc = (fmin+fmax)/2         -> baseband contour
  2. synthesize a constant-amplitude complex NFM baseband signal
        x[n] = exp(j*2*pi*cumsum(f_inst[n])/fs_bb),  fs_bb = 12 kHz
  3. STFT with 128-pt Hamming window, hop 64  ->  93.75 Hz x 5.33 ms resolution
        (identical to the step-1 spectrogram resolution: 12000/128 = 93.75 Hz, 64/12000 = 5.33 ms)
  4. fftshift -> 128 freq bins spanning -6..+6 kHz (0 Hz at row 64), 128 time frames,
     whistle centered in time. Magnitude normalized to [0,1].
Image layout: img[row=frequency bin (row 0 = -6 kHz, increasing upward), col=time frame].

Whistles that do not fit (> 128 frames = 0.68 s, or baseband excursion > +-5.9 kHz) are skipped and counted.

Outputs in --out_dir:
  images.npy    uint8  [N,128,128]  (value/255 = intensity in [0,1]); memory-mappable
  features.npy  float32[N,7]        [length(frames), fmin, fmax, bandwidth, fstart, fend, fc] (all in image bins, fc = 64)
  index.csv     source file, start frame, length, score, center freq in Hz (needed later to re-assign fc)

Usage:
  python step4_make_images.py --tracks_dir /data/tracks --out_dir /data/dataset --workers 8 [--preview]
"""
import argparse, csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

IMG = 128
FS_BB = 12_000.0
WIN, HOP = 128, 64
DF = FS_BB / WIN                       # 93.75 Hz per bin
N_SAMP = (IMG - 1) * HOP + WIN         # 8256 samples -> exactly 128 STFT frames
MAX_REL_HZ = 5_900.0
_WINDOW = np.hamming(WIN).astype(np.float32)
_TAPER = np.hanning(17); _TAPER /= _TAPER.sum()


def contour_to_image(f_hz):
    """f_hz: 1-D contour (Hz). Returns (img float32 [128,128] in [0,1], fc_hz) or None if it doesn't fit."""
    L = len(f_hz)
    if L > IMG or L < 3:
        return None
    fc = 0.5 * (f_hz.min() + f_hz.max())
    rel = f_hz.astype(np.float64) - fc
    if np.abs(rel).max() > MAX_REL_HZ:
        return None
    off = (IMG - L) // 2
    centers = (off + np.arange(L)) * HOP + WIN // 2                    # sample index of each frame center
    n = np.arange(N_SAMP)
    inst = np.interp(n, centers, rel, left=np.nan, right=np.nan)
    gate = np.isfinite(inst).astype(np.float64)
    inst = np.nan_to_num(inst)
    gate = np.convolve(gate, _TAPER, mode="same")                      # soft on/off (avoids splatter)
    x = gate * np.exp(1j * 2 * np.pi * np.cumsum(inst) / FS_BB)
    frames = sliding_window_view(x, WIN)[::HOP]                        # [128 frames, 128 samples]
    S = np.abs(np.fft.fftshift(np.fft.fft(frames * _WINDOW, axis=1), axes=1))   # [time, freq]
    img = S.T                                                          # [freq, time]
    m = img.max()
    if m <= 0:
        return None
    return (img / m).astype(np.float32), float(fc)


def features(f_hz, fc):
    """7 whistle features in image bins (paper Sec. III-B)."""
    rel = (f_hz - fc) / DF + IMG // 2
    return np.array([len(f_hz), rel.min(), rel.max(), rel.max() - rel.min(), rel[0], rel[-1], IMG // 2], np.float32)


def _work(chunk):
    imgs, feats, keep = [], [], []
    for k, f in chunk:
        r = contour_to_image(f)
        if r is None:
            keep.append(False); continue
        img, fc = r
        imgs.append(np.round(img * 255).astype(np.uint8)); feats.append(features(f, fc)); keep.append(True)
    return (np.stack(imgs) if imgs else np.zeros((0, IMG, IMG), np.uint8),
            np.stack(feats) if feats else np.zeros((0, 7), np.float32), keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks_dir", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ---- gather all contours
    items, rows = [], []
    for fp in sorted(Path(a.tracks_dir).glob("*_tracks.npz")):
        z = np.load(fp)
        for i in range(len(z["start"])):
            f = z["freq"][z["offset"][i]:z["offset"][i + 1]]
            items.append(f); rows.append((fp.stem.replace("_tracks", ""), int(z["start"][i]), len(f), float(z["score"][i])))
    print(f"{len(items)} whistle vectors loaded")

    chunks = [list(enumerate(items[i:i + a.chunk])) for i in range(0, len(items), a.chunk)]
    # first pass over chunks writes into a growing list of memmap-able parts, then we stitch (keeps RAM low)
    parts_img, parts_feat, keep_all = [], [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for ci, (im, ft, kp) in enumerate(ex.map(_work, chunks), 1):
            np.save(out / f"_part_img_{ci}.npy", im); parts_img.append(out / f"_part_img_{ci}.npy")
            parts_feat.append(ft); keep_all.extend(kp)
            if ci % 20 == 0 or ci == len(chunks):
                print(f"  chunk {ci}/{len(chunks)}")
    keep = np.array(keep_all, bool); N = int(keep.sum())
    print(f"kept {N}/{len(items)}  (skipped {len(items) - N}: too long / too wide)")

    images = np.lib.format.open_memmap(out / "images.npy", mode="w+", dtype=np.uint8, shape=(N, IMG, IMG))
    pos = 0
    for p in parts_img:
        im = np.load(p); images[pos:pos + len(im)] = im; pos += len(im); p.unlink()
    images.flush()
    np.save(out / "features.npy", np.concatenate(parts_feat) if parts_feat else np.zeros((0, 7), np.float32))

    # index with the center frequency (Hz) of every kept whistle
    with open(out / "index.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["file", "start_frame", "length", "score", "center_hz"])
        for (r, f, k) in zip(rows, items, keep):
            if k:
                w.writerow([*r, f"{0.5 * (f.min() + f.max()):.1f}"])

    if a.preview and N:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            sel = np.random.default_rng(0).choice(N, min(40, N), replace=False)
            fig, ax = plt.subplots(5, 8, figsize=(16, 10))
            for axi, j in zip(ax.ravel(), sel):
                axi.imshow(images[j], origin="lower", cmap="gray_r", vmin=0, vmax=255); axi.axis("off")
            fig.tight_layout(); fig.savefig(out / "preview.png", dpi=80)
            print("saved preview.png")
        except ImportError:
            print("matplotlib not installed - preview skipped")
    print("done ->", out)


if __name__ == "__main__":
    main()
