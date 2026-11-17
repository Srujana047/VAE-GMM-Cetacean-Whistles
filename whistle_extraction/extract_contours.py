"""
Classical whistle-contour extractor (REPLACES the paper's neural-network extractor, so expect lower quality).

Per file:  one channel -> 192 kHz -> high-pass 2 kHz -> spectrogram (2048 Hamming, 50% overlap, as paper)
  1. subtract per-frequency median over time           (removes steady noise / tonal hum)
  2. flag frames where much of 2-50 kHz is elevated     (= clicks) and blank them
  3. subtract per-frame median inside the whistle band  (removes leftover broadband energy)
  4. threshold robust z-score, close 1-frame gaps, label connected blobs
  5. keep blobs that are long enough and thin (whistle-like), take the ridge (max per frame) -> contour in Hz
Then whistle_preprocess.contour_to_image() -> 128x128 images.

  python extract_contours.py --selftest
  python extract_contours.py --audio audio/ --out whistles.npz --channel 0
"""
import argparse, glob, os
import numpy as np
from scipy import signal, ndimage
from math import gcd
from whistle_preprocess import FS, NFFT, HOP, HP_HZ, contour_to_image, image_to_features, synth_contour, centre_freq

BAND = (2500, 20000)


def contours_from_spec(f, D, z_thr=4.0, min_len=12, max_len=128, max_thick=7, click_frac=0.25, min_bw=3, max_jump=8):
    """f: frequency of each row (Hz, must cover 2-50 kHz); D: dB spectrogram. Returns (contours, D_normalised, click_frames)."""
    D = np.array(D, dtype=np.float64)
    D -= np.median(D, axis=1, keepdims=True)                           # 1. per-frequency background
    wide = (f >= HP_HZ) & (f <= 50000)
    click = (D[wide] > 6).mean(axis=0) > click_frac                    # 2. click frames
    click = ndimage.binary_dilation(click, iterations=1)
    b = (f >= BAND[0]) & (f <= BAND[1])
    fb, Db = f[b], D[b].copy()
    Db -= np.median(Db, axis=0, keepdims=True)                         # 3. per-frame median
    Db[:, click] = 0
    sigma = 1.4826 * np.median(np.abs(Db - np.median(Db)))
    Z = Db / (sigma + 1e-9)
    mask = ndimage.binary_closing(Z > z_thr, structure=np.ones((1, 3)))  # 4. bridge 1-frame gaps
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        r, c = sl
        width = c.stop - c.start
        if width < min_len or width > max_len:
            continue
        sub = lab[sl] == i
        cols = sub.any(axis=0)
        if cols.mean() < 0.6 or np.median(sub.sum(axis=0)[cols]) > max_thick:   # 5. thin + mostly continuous
            continue
        ridge = np.where(sub, Z[sl], -np.inf).argmax(axis=0).astype(float)
        idx = np.arange(width)
        ridge = np.interp(idx, idx[cols], ridge[cols])                 # fill gaps
        ridge = ndimage.median_filter(ridge, size=3, mode="nearest")   # edge-safe (medfilt zero-pads)
        if ridge.max() - ridge.min() < min_bw or np.abs(np.diff(ridge)).max() > max_jump:
            continue                                                   # flat stubs / jumps (merged blobs)
        out.append((fb[r.start + np.round(ridge).astype(int)], int(c.start)))   # Hz per frame, start frame
    return out, D, click


def contours_from_signal(x, fs=FS, **kw):
    f, t, S = signal.spectrogram(x, fs=fs, window="hamming", nperseg=NFFT, noverlap=NFFT - HOP, mode="magnitude")
    out, D, click = contours_from_spec(f, 20 * np.log10(S + 1e-12), **kw)
    return out, (f, t, D, click)


def load_channel(path, channel=0, fs_target=FS):
    import soundfile as sf
    x, fs = sf.read(path, dtype="float32", always_2d=True)             # ~0.7 GB RAM for a 60 s, 6-ch file
    x = x[:, channel].astype(np.float64)
    if fs != fs_target:
        g = gcd(fs_target, fs); x = signal.resample_poly(x, fs_target // g, fs // g)
    sos = signal.butter(4, HP_HZ, "highpass", fs=fs_target, output="sos")
    return signal.sosfiltfilt(sos, x)


def selftest():
    rng = np.random.default_rng(1)
    n = 12 * FS; x = rng.normal(0, 0.01, n)
    truth = 0
    for t0 in (1.0, 3.0, 4.5, 6.0, 8.0, 10.0):                        # 6 whistles
        c = synth_contour(rng); c = np.clip(c, 3000, 15000)
        fi = np.interp(np.arange(len(c) * HOP), np.arange(len(c)) * HOP, c)
        a = int(t0 * FS); seg = 0.05 * np.sin(2 * np.pi * np.cumsum(fi) / FS)
        x[a:a + len(seg)] += seg[: n - a]; truth += 1
    for tc in rng.uniform(0.5, 11.5, 30):                              # 30 clicks (broadband)
        a = int(tc * FS); x[a:a + 40] += rng.normal(0, 1.0, 40)
    sos = signal.butter(4, HP_HZ, "highpass", fs=FS, output="sos"); x = signal.sosfiltfilt(sos, x)
    cs, _ = contours_from_signal(x)
    print(f"selftest: planted {truth} whistles + 30 clicks -> extracted {len(cs)} contours "
          f"(lengths {[len(c) for c, _ in cs]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="audio")
    ap.add_argument("--out", default="whistles.npz")
    ap.add_argument("--channel", type=int, default=0)       # one channel per file -> avoids 6x duplicates
    ap.add_argument("--min_len", type=int, default=20)      # frames (20 = 107 ms); paper examples are ~40-110
    ap.add_argument("--min_bw", type=int, default=3)         # min frequency span in bins (flat fragments dropped)
    ap.add_argument("--z", type=float, default=4.0)         # lower = more (noisier) contours
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); raise SystemExit

    files = sorted(glob.glob(os.path.join(a.audio, "*.flac")))
    imgs, feats, groups, fcs, bad = [], [], [], [], []; best = 0
    for gi, p in enumerate(files):
        try:
            x = load_channel(p, a.channel)
        except Exception as e:                               # truncated / corrupt download
            print(f"[{gi+1}/{len(files)}] {os.path.basename(p)}: SKIPPED ({e})"); bad.append(os.path.basename(p)); continue
        cs, (f, t, D, click) = contours_from_signal(x, z_thr=a.z, min_len=a.min_len, min_bw=a.min_bw)
        kept = 0
        for c, _ in cs:
            im = contour_to_image(c)
            ft = None if im is None else image_to_features(im)
            if ft is None: continue
            imgs.append(im); feats.append(list(ft.values())); groups.append(gi); fcs.append(centre_freq(c)); kept += 1
        print(f"[{gi+1}/{len(files)}] {os.path.basename(p)}: {len(cs)} contours, {kept} images, {click.mean():.0%} click frames")
        if len(cs) > best:                                   # QC plot of the file with the most contours
            best = len(cs)
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            plt.figure(figsize=(16, 5)); m = (f >= 0) & (f <= 20000)
            plt.imshow(D[m], origin="lower", aspect="auto", vmin=0, vmax=30, cmap="magma",
                       extent=[0, t[-1], 0, 20])
            for c, s in cs: plt.plot((s + np.arange(len(c))) * HOP / FS, c / 1000, "c", lw=1)
            plt.xlabel("s"); plt.ylabel("kHz"); plt.savefig("qc_first_file.png", dpi=80)
        del x
    if bad:
        open("bad_files.txt", "w").write("\n".join(bad)); print("corrupt files (re-download):", bad)
    X = np.stack(imgs)[:, None].astype(np.float32)
    np.savez_compressed(a.out, images=X, feats=np.array(feats, np.int32), groups=np.array(groups), fc=np.array(fcs, np.float32))
    print(f"saved {a.out}: {X.shape}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt   # contour overview grid
    n = min(25, len(X)); sel = np.random.default_rng(0).choice(len(X), n, replace=False)
    fig, ax = plt.subplots(5, 5, figsize=(15, 15), constrained_layout=True)
    for k, axk in enumerate(ax.flat):
        axk.axis("off")
        if k < n:
            i = sel[k]; axk.imshow(X[i, 0], origin="lower", cmap="magma")
            axk.set_title(f"#{i} file{groups[i]} len{feats[i][0]} bw{feats[i][3]}", fontsize=8)
    fig.suptitle(f"Random {n} of {len(X)} extracted whistles"); plt.savefig("contour_overview.png", dpi=80)
    print("saved contour_overview.png")
