"""
Preprocessing for the VAE-GMM cetacean whistle paper (Park et al., IEEE Access 2026).

Paper pipeline (what is stated):
  raw audio (500 kHz, ~60 s, 6-ch towed array)
    -> resample to 192 kHz, high-pass > 2 kHz
    -> spectrogram: 2048-pt Hamming, 50% overlap  (dt ~5.33 ms, df = 93.75 Hz)
    -> whistle contour extraction (their own NN + CFAR/tracking)  -> time-frequency vector per whistle
    -> contour -> constant-amplitude NFM baseband signal (centre frequency removed)
    -> spectrogram -> 128x128 grayscale image in [0,1]

Steps marked ASSUMPTION are NOT specified in the paper; I chose them so the pipeline runs.

Usage:
  python whistle_preprocess.py --synthetic 3000 --out whistles.npz      # no real data needed
  (real data) call audio_to_spectrogram() to look at files, then feed your extracted contours
  (list of 1-D arrays, frequency in Hz per spectrogram frame) to build_dataset().
"""
import argparse
from math import gcd

import numpy as np
from scipy import signal

FS = 192_000          # paper: resample to 192 kHz
NFFT = 2048           # paper: 2048-point Hamming window
HOP = NFFT // 2       # paper: 50% overlap -> 5.33 ms
IMG = 128             # paper: 128 x 128 image
HP_HZ = 2_000         # paper: high-pass above 2 kHz
DF = FS / NFFT        # 93.75 Hz per frequency bin


def centre_freq(freq_hz, mode="mean"):
    """Centre frequency removed before synthesis. Paper Fig. 2(c) is asymmetric about 0 -> 'mean' (default); 'mid' = (max+min)/2."""
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    return float(freq_hz.mean()) if mode == "mean" else float(0.5 * (freq_hz.max() + freq_hz.min()))


# ----------------------------------------------------------------------------
# 1) Raw audio -> spectrogram (use this to LOOK at the data / feed a contour extractor)
# ----------------------------------------------------------------------------
def audio_to_spectrogram(path, channel=0, fs_target=FS):
    import soundfile as sf
    x, fs = sf.read(path, always_2d=True)          # shape (samples, channels)
    x = x[:, channel]                              # ASSUMPTION: process one channel at a time
    if fs != fs_target:                            # 500 kHz -> 192 kHz  == up 48 / down 125
        g = gcd(fs_target, fs)
        x = signal.resample_poly(x, fs_target // g, fs // g)
    sos = signal.butter(4, HP_HZ, "highpass", fs=fs_target, output="sos")
    x = signal.sosfiltfilt(sos, x)                 # ASSUMPTION: 4th-order zero-phase Butterworth
    f, t, S = signal.spectrogram(x, fs=fs_target, window="hamming", nperseg=NFFT,
                                 noverlap=NFFT - HOP, mode="magnitude")
    return f, t, 20 * np.log10(S + 1e-10)          # dB


# ----------------------------------------------------------------------------
# 2) Contour (Hz per frame) -> NFM signal -> 128x128 image
# ----------------------------------------------------------------------------
def contour_to_image(freq_hz, center="mean"):
    """freq_hz: 1-D array, instantaneous frequency in Hz, one value per spectrogram frame."""
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    n_frames = len(freq_hz)
    if n_frames < 3 or n_frames > IMG:
        return None                                # ASSUMPTION: whistles longer than 128 frames are dropped

    fc = centre_freq(freq_hz, center)              # centre frequency removed (exact definition = ASSUMPTION)
    f_bb = freq_hz - fc
    if np.abs(f_bb).max() >= (IMG // 2 - 1) * DF:  # would fall outside the +-6 kHz image
        return None

    n_samp = (n_frames - 1) * HOP + NFFT
    frame_centres = np.arange(n_frames) * HOP + NFFT / 2
    f_inst = np.interp(np.arange(n_samp), frame_centres, f_bb)   # per-sample frequency
    phase = 2 * np.pi * np.cumsum(f_inst) / FS
    z = np.exp(1j * phase)                         # constant-amplitude NFM, complex baseband

    _, _, Z = signal.stft(z, fs=FS, window="hamming", nperseg=NFFT, noverlap=NFFT - HOP,
                          boundary=None, padded=False, return_onesided=False)
    Z = np.fft.fftshift(np.abs(Z), axes=0)         # (2048, frames), 0 Hz in the middle
    mid = NFFT // 2
    Z = Z[mid - IMG // 2: mid + IMG // 2, :]       # +-6 kHz around 0 Hz -> 128 bins (93.75 Hz each)
    img = np.zeros((IMG, IMG), dtype=np.float32)   # CENTRED in time, as in paper Figs. 2-3 (0 s = middle column)
    w = min(Z.shape[1], IMG); s0 = (IMG - w) // 2
    img[:, s0:s0 + w] = Z[:, :w]
    img /= img.max() + 1e-12                       # normalise to [0, 1]
    return img


# ----------------------------------------------------------------------------
# 3) Image -> simple whistle features (paper: threshold -> time-frequency vector)
# ----------------------------------------------------------------------------
def image_to_features(img, thr=0.05):
    """Returns dict of the 6 conditioning features in image units (frames / frequency bins)."""
    img = np.asarray(img).squeeze()
    active = np.where(img.max(axis=0) > thr)[0]    # time columns that contain whistle energy
    if len(active) == 0:
        return None
    rows = img[:, active].argmax(axis=0)           # ridge = strongest bin per column
    return dict(length=len(active), fmin=rows.min(), fmax=rows.max(), bw=rows.max() - rows.min(),
                fstart=rows[0], fend=rows[-1])


# ----------------------------------------------------------------------------
# 4) Synthetic whistles (so you can run everything WITHOUT the NOAA data)
# ----------------------------------------------------------------------------
def synth_contour(rng):
    n = int(rng.integers(25, 110))
    t = np.linspace(0, 1, n)
    kind = rng.integers(0, 5)
    base = 9_000 + rng.uniform(-1500, 1500)
    span = rng.uniform(800, 4500)
    if kind == 0:   shape = t                                       # up-sweep
    elif kind == 1: shape = 1 - t                                   # down-sweep
    elif kind == 2: shape = np.sin(np.pi * t)                       # arch
    elif kind == 3: shape = 1 - np.sin(np.pi * t)                   # inverted arch
    else:           shape = 0.5 + 0.5 * np.sin(2 * np.pi * rng.integers(1, 4) * t)  # wavy
    return base + span * shape + rng.normal(0, 30, n)


# ----------------------------------------------------------------------------
def build_dataset(contours):
    imgs, feats = [], []
    for c in contours:
        im = contour_to_image(c)
        if im is None:
            continue
        ft = image_to_features(im)
        if ft is None:
            continue
        imgs.append(im); feats.append([ft[k] for k in ("length", "fmin", "fmax", "bw", "fstart", "fend")])
    return np.stack(imgs)[:, None].astype(np.float32), np.array(feats, dtype=np.int32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=2000)
    ap.add_argument("--out", default="whistles.npz")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    X, F = build_dataset([synth_contour(rng) for _ in range(a.synthetic)])
    np.savez_compressed(a.out, images=X, feats=F)
    print(f"saved {a.out}: images {X.shape}, range [{X.min():.2f},{X.max():.2f}], "
          f"mean length {F[:,0].mean():.1f} frames, sparsity {(X<0.05).mean():.1%} pixels ~0")
