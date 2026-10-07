"""
STEP 1 - Raw NOAA/DCLDE-2022 audio -> normalized spectrograms
Follows Park et al. (IEEE Access 2026), Sec. II-C:
  500 kHz -> resample 192 kHz -> high-pass 2 kHz -> 2048-pt Hamming, 50% overlap
  => 5.33 ms time step, 93.75 Hz frequency step.
Extra (my additions, standard for whistle detection): click suppression and
background whitening, then scaling to [0, 1] so the NN sees a stable input.

Usage:
  python step1_spectrogram.py --in_dir /data/dclde2022 --out_dir /data/specs --channels 0 --workers 8
Output per file/channel:  <stem>_ch<k>.npy   (float16, shape [freq_bins, time_frames])
Plus one meta.json with the frequency axis and time step.
"""
import argparse, json, math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import binary_dilation


@dataclass
class Cfg:
    target_fs: int = 192_000
    hp_cutoff: float = 2_000.0      # Hz (paper: passband above 2 kHz)
    hp_order: int = 4
    nperseg: int = 2048             # Hamming window (paper)
    noverlap: int = 1024            # 50% overlap (paper)
    fmin: float = 2_000.0           # keep band [fmin, fmax]
    fmax: float = 30_000.0          # false killer whale whistles mostly < ~25 kHz
    dyn_range_db: float = 25.0      # dB above background mapped to 1.0
    click_db: float = 3.0           # frame whose median over freq is > this many dB = click frame


def wav_to_spec(x: np.ndarray, fs: int, cfg: Cfg):
    """x: 1-D float32 audio. Returns (spec[F,T] float32 in [0,1], freqs[F])."""
    if fs != cfg.target_fs:
        g = math.gcd(int(fs), cfg.target_fs)
        x = signal.resample_poly(x, cfg.target_fs // g, int(fs) // g).astype(np.float32)

    sos = signal.butter(cfg.hp_order, cfg.hp_cutoff, btype="highpass",
                        fs=cfg.target_fs, output="sos")
    x = signal.sosfiltfilt(sos, x).astype(np.float32)

    f, _, S = signal.spectrogram(
        x, fs=cfg.target_fs, window="hamming", nperseg=cfg.nperseg,
        noverlap=cfg.noverlap, detrend=False, scaling="spectrum", mode="psd")

    band = (f >= cfg.fmin) & (f <= cfg.fmax)
    f, S = f[band], S[band]
    S = 10.0 * np.log10(S + 1e-20)                      # dB

    # --- whitening -------------------------------------------------------
    S -= np.median(S, axis=1, keepdims=True)            # per-frequency background (ship/ocean noise)

    # --- click removal ------------------------------------------------------
    # Clicks are broadband, so the median over frequency jumps in those frames (measured on real DCLDE
    # data: ~1.4% of frames exceed 3 dB). Subtracting the column median everywhere (my first version)
    # left dark stripes that cut whistles in two. Instead: flag click frames (+1 neighbour because of the
    # 50% window overlap) and re-fill them from the nearest clean frames, so whistles stay continuous.
    bad = binary_dilation(np.median(S, axis=0) > cfg.click_db, iterations=1)
    T = S.shape[1]
    if bad.any() and not bad.all():
        idx = np.arange(T)
        prev = np.maximum.accumulate(np.where(~bad, idx, -1))
        nxt = np.minimum.accumulate(np.where(~bad, idx, T)[::-1])[::-1]
        p = np.where(prev < 0, nxt, prev)
        n = np.where(nxt >= T, prev, nxt)
        S[:, bad] = 0.5 * (S[:, p[bad]] + S[:, n[bad]])

    S = np.clip(S, 0.0, cfg.dyn_range_db) / cfg.dyn_range_db
    return S.astype(np.float32), f.astype(np.float32)


def read_channels(path, channels, block=2_000_000):
    """Read only the requested channels, block by block. A 60 s, 6-channel, 500 kHz file is 720 MB as
    float32 if read whole; this keeps ~120 MB per channel per worker. Works for .wav and .flac."""
    with sf.SoundFile(str(path)) as f:
        fs = f.samplerate
        chans = [c for c in channels if c < f.channels]
        out = {c: np.empty(f.frames, np.float32) for c in chans}
        pos = 0
        for blk in f.blocks(blocksize=block, dtype="float32", always_2d=True):
            for c in chans:
                out[c][pos:pos + len(blk)] = blk[:, c]
            pos += len(blk)
    return {c: v[:pos] for c, v in out.items()}, fs


def process_file(args):
    path, out_dir, channels, cfg = args
    try:
        chans, fs = read_channels(path, channels)
        results = []
        for c, x in chans.items():
            spec, f = wav_to_spec(x, fs, cfg)
            out = Path(out_dir) / f"{Path(path).stem}_ch{c}.npy"
            np.save(out, spec.astype(np.float16))
            results.append((str(out), spec.shape))
        return str(path), results, None
    except Exception as e:  # keep the batch running
        return str(path), [], repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="dataset",
                    help="folder containing audio files (searched recursively)")
    ap.add_argument("--out_dir", default="spectrograms")
    ap.add_argument("--channels", type=int, nargs="+", default=[0],
                    help="hydrophone channels to process (files have up to 6)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ext", nargs="+", default=[".wav", ".flac"])
    a = ap.parse_args()

    cfg = Cfg()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for e in a.ext for p in Path(a.in_dir).rglob(f"*{e}"))
    print(f"{len(files)} files found")

    # meta (identical for every file)
    f_axis = np.fft.rfftfreq(cfg.nperseg, 1 / cfg.target_fs)
    f_axis = f_axis[(f_axis >= cfg.fmin) & (f_axis <= cfg.fmax)]
    meta = dict(cfg=asdict(cfg), freqs=f_axis.tolist(),
                dt=(cfg.nperseg - cfg.noverlap) / cfg.target_fs,
                df=cfg.target_fs / cfg.nperseg)
    (out_dir / "meta.json").write_text(json.dumps(meta))

    jobs = [(p, out_dir, a.channels, cfg) for p in files]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (p, res, err) in enumerate(ex.map(process_file, jobs), 1):
            print(f"[{i}/{len(jobs)}] {Path(p).name}", "ERR " + err if err else res[0][1] if res else "")


if __name__ == "__main__":
    main()
