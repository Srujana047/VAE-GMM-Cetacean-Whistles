"""
STEP 3 - Post-processing / tracking:  probability map  ->  individual whistle contours
Input : <stem>_prob.npy from step 2 (float16 [F,T]) + meta.json from step 1
Output: <stem>_tracks.npz  with (one entry per whistle)
          start   int32  [n]   first spectrogram frame of the whistle
          length  int32  [n]   number of frames (5.33 ms each)
          offset  int64  [n+1] slice boundaries into `freq`
          freq    float32[sum] contour frequency in Hz, one value per frame
          score   float32[n]   mean detection probability along the contour

Algorithm
 1. Hysteresis threshold + removal of small blobs and overly thick (noise) blobs
 2. Per-frame peak extraction (sub-bin centroid of every connected run)
 3. Multi-target tracker: Hungarian assignment frame-to-frame, velocity prediction,
    gating, gap bridging (max_gap frames)
 4. Gap interpolation + Savitzky-Golay smoothing, duration / confidence filtering
 5. Optional removal of harmonic tracks (~2x / 3x of a stronger fundamental)

Usage:
  python step3_tracking.py --prob_dir /data/probs --meta /data/specs/meta.json --out_dir /data/tracks --workers 8
"""
import argparse, json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter


@dataclass
class TrackCfg:
    thr_low: float = 0.35        # hysteresis low threshold
    thr_high: float = 0.65       # a blob must reach this somewhere
    min_pixels: int = 12         # drop tiny blobs
    max_width_bins: int = 10     # drop frame-runs thicker than this (noise / clicks)
    gate_bins: float = 4.0       # max |df| per frame (4 bins = 375 Hz / 5.3 ms ~ 70 kHz/s)
    max_gap: int = 3             # frames a track may coast without a detection
    min_len: int = 15            # frames (~80 ms) minimum whistle duration
    min_det_frac: float = 0.7    # fraction of frames that must be real detections
    min_score: float = 0.5       # mean probability along track
    smooth_win: int = 7          # Savitzky-Golay window (odd)
    drop_harmonics: bool = True


# ------------------------------------------------------------------ peaks
def extract_peaks(prob, c: TrackCfg):
    low = prob >= c.thr_low
    lab, n = ndi.label(low, structure=np.ones((3, 3), bool))
    if n == 0:
        return {}
    idx = np.arange(1, n + 1)
    mx = ndi.maximum(prob, lab, idx)
    sz = ndi.sum(low, lab, idx)
    keep = np.zeros(n + 1, bool)
    keep[1:] = (mx >= c.thr_high) & (sz >= c.min_pixels)
    P = np.where(keep[lab], prob, 0.0).astype(np.float32)

    peaks = {}
    for t in np.flatnonzero((P > 0).any(0)):
        col = P[:, t]
        nz = np.flatnonzero(col)
        runs = np.split(nz, np.flatnonzero(np.diff(nz) > 1) + 1)
        for r in runs:
            if len(r) > c.max_width_bins:
                continue
            w = col[r]
            peaks.setdefault(int(t), []).append((float((r * w).sum() / w.sum()), float(w.max())))
    return peaks


# ------------------------------------------------------------------ tracker
class _Trk:
    __slots__ = ("t", "f", "p", "vel", "miss")

    def __init__(self, t, f, p):
        self.t, self.f, self.p, self.vel, self.miss = [t], [f], [p], 0.0, 0


def track_peaks(peaks, c: TrackCfg):
    if not peaks:
        return []
    active, done = [], []
    for t in range(min(peaks), max(peaks) + 1):
        pk = peaks.get(t, [])
        # --- association
        if active and pk:
            pred = np.array([a.f[-1] + np.clip(a.vel, -c.gate_bins, c.gate_bins) * (t - a.t[-1]) for a in active])
            pf = np.array([q[0] for q in pk])
            cost = np.abs(pred[:, None] - pf[None, :])
            gates = np.array([c.gate_bins * min(t - a.t[-1], 3) for a in active])[:, None]
            cost = np.where(cost <= gates, cost, 1e6)
            ri, ci = linear_sum_assignment(cost)
            matched_t, matched_p = set(), set()
            for r, k in zip(ri, ci):
                if cost[r, k] >= 1e6:
                    continue
                a = active[r]; dt = t - a.t[-1]
                a.vel = 0.6 * a.vel + 0.4 * (pk[k][0] - a.f[-1]) / dt
                a.t.append(t); a.f.append(pk[k][0]); a.p.append(pk[k][1]); a.miss = 0
                matched_t.add(r); matched_p.add(k)
        else:
            matched_t, matched_p = set(), set()
        # --- new tracks / miss bookkeeping
        for k, q in enumerate(pk):
            if k not in matched_p:
                active.append(_Trk(t, q[0], q[1]))
        still = []
        for i, a in enumerate(active):
            if i in matched_t or a.t[-1] == t:
                still.append(a)
            else:
                a.miss += 1
                (still if a.miss <= c.max_gap else done).append(a)
        active = still
    return done + active


def finalize(trk: _Trk, c: TrackCfg):
    t = np.asarray(trk.t); f = np.asarray(trk.f); p = np.asarray(trk.p)
    span = t[-1] - t[0] + 1
    if span < c.min_len or len(t) < c.min_det_frac * span or p.mean() < c.min_score:
        return None
    tt = np.arange(t[0], t[-1] + 1)
    ff = np.interp(tt, t, f)                                  # bridge gaps
    w = min(c.smooth_win, span if span % 2 else span - 1)
    if w >= 5:
        ff = savgol_filter(ff, w, 2, mode="interp")
    return int(t[0]), ff.astype(np.float32), float(p.mean())


def drop_harmonic_tracks(tracks):
    """tracks: list of (start, f_bins, score). Remove tracks lying at ~2x/3x of a stronger one."""
    order = sorted(range(len(tracks)), key=lambda i: -tracks[i][2] * len(tracks[i][1]))
    kill = set()
    for ii, i in enumerate(order):
        if i in kill:
            continue
        s1, f1, _ = tracks[i]
        for j in order[ii + 1:]:
            if j in kill:
                continue
            s2, f2, _ = tracks[j]
            a, b = max(s1, s2), min(s1 + len(f1), s2 + len(f2))
            if b - a < 0.5 * len(f2):
                continue
            ratio = np.median((f2[a - s2:b - s2]) / np.maximum(f1[a - s1:b - s1], 1e-3))
            if any(abs(ratio - h) < 0.08 * h for h in (2.0, 3.0)):
                kill.add(j)
    return [tr for k, tr in enumerate(tracks) if k not in kill]


# ------------------------------------------------------------------ driver
def process(args):
    path, out_dir, meta, c = args
    try:
        prob = np.load(path).astype(np.float32)
        peaks = extract_peaks(prob, c)
        tracks = [r for r in (finalize(t, c) for t in track_peaks(peaks, c)) if r is not None]
        # frequency in absolute "bin" units -> compare harmonics in Hz-proportional space
        f0, df = meta["freqs"][0], meta["df"]
        tracks = [(s, f0 + f * df, sc) for s, f, sc in tracks]          # bins -> Hz
        if c.drop_harmonics and len(tracks) > 1:
            tracks = drop_harmonic_tracks(tracks)
        tracks.sort(key=lambda x: x[0])
        lens = np.array([len(f) for _, f, _ in tracks], np.int32)
        stem = Path(path).stem.replace("_prob", "")
        np.savez(Path(out_dir) / f"{stem}_tracks.npz",
                 start=np.array([s for s, _, _ in tracks], np.int32), length=lens,
                 offset=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
                 freq=np.concatenate([f for _, f, _ in tracks]).astype(np.float32) if tracks else np.zeros(0, np.float32),
                 score=np.array([sc for _, _, sc in tracks], np.float32))
        return str(path), len(tracks), None
    except Exception as e:
        return str(path), 0, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob_dir", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--out_dir", required=True); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    meta = json.loads(Path(a.meta).read_text()); c = TrackCfg()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.prob_dir).glob("*_prob.npy"))
    total = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (p, n, err) in enumerate(ex.map(process, [(f, out, meta, c) for f in files]), 1):
            total += n
            print(f"[{i}/{len(files)}] {Path(p).name}: {n} whistles", "ERR " + err if err else "")
    print("total whistles:", total)


if __name__ == "__main__":
    main()
