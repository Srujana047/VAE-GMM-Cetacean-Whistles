# Whistle extraction (false killer whale, DCLDE 2022)

Turns raw FLAC recordings into 128x128 whistle images for the VAE-GMM model
(Park et al., IEEE Access 2026, A Generative Model for Cetacean Whistles Using VAE and Mixture of Gaussians).

## Files

`extract_contours.py` finds whistles in the audio.
1. Reads one channel of each FLAC, resamples to 192 kHz, high-pass at 2 kHz.
2. Spectrogram 2048-point Hamming window, 50% overlap (5.33 ms, 93.75 Hz per bin).
3. Removes background noise and blanks click frames (broadband vertical lines).
4. Thresholds the 2.5-20 kHz band, keeps long, thin, continuous blobs, and tracks the ridge to get one frequency contour per whistle.
5. Calls `whistle_preprocess.contour_to_image()` and saves `whistles.npz`.

`whistle_preprocess.py` turns one contour into an image (paper Fig. 23).
- Removes the centre frequency (mean), synthesises a constant-amplitude frequency-modulated baseband signal, takes its spectrogram and crops 128 bins around 0 Hz (about ±6 kHz).
- Centres the whistle in time and scales the image to [0, 1].
- Whistles longer than 128 frames (~683 ms) are dropped.
- Also has `image_to_features` (length, fmin, fmax, bandwidth, start and end frequency) and a synthetic whistle generator for testing.

## Usage
```
pip install numpy scipy soundfile matplotlib
python extract_contours.py --selftest
python extract_contours.py --audio folder with .flac --out whistles.npz --channel 0 --z 3.5
```
Options `--z` (detection threshold, lower = more contours), `--min_len` (min frames, default 20), `--min_bw` (min frequency span in bins, default 3).

## Output
- `whistles.npz` `images` (N,1,128,128), `feats` (N,6), `groups` (source file index), `fc` (centre frequency in Hz of each whistle)
- `qc_first_file.png` spectrogram with the extracted contours overlaid (check it by eye)
- `contour_overview.png` 25 random extracted whistle images
- `bad_files.txt` corrupt or truncated FLAC files that were skipped

## Limitations
- The paper uses a neural-network contour extractor that is not available to us, so this is a classical substitute. Recall is partial and some whistles are split into fragments.
- The 4th-order Butterworth filter, the centre-frequency definition, and leftcentre alignment are assumptions, not stated in the paper.
- Only one channel per file is used, to avoid duplicate whistles across the 6 hydrophones.