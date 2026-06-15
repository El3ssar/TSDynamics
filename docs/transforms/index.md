---
description: The signal layer — power spectra, detrend/normalize/filter preprocessing, and feature extraction that turn a Trajectory into analysable features.
---

<span class="ts-kicker">Transforms</span>

# Transforms

The **signal layer**: estimators that turn a raw recording — a
[`Trajectory`](../reference/base.md) or a bare NumPy array — into spectra,
clean preprocessed signals, and flat feature vectors. Everything lives in
`tsdynamics.transforms` (it is **not** in the top-level `tsdynamics`
namespace), so reach for it explicitly:

```python
from tsdynamics import transforms
# or pull individual functions:
from tsdynamics.transforms import power_spectral_density, dominant_frequency
```

Every entry point is **shape-preserving** where it makes sense and dual-source:
pass a `Trajectory` and the sampling interval `dt` / rate `fs` is read from its
metadata (one result *per component*); pass a bare array and give `fs` (or `dt`)
explicitly (one scalar result). Preprocessors (`detrend`, `normalize`, the
Butterworth filters) return a signal of the same shape as their input.

| Group | Functions | Does |
|---|---|---|
| Spectral | `power_spectral_density`, `spectral_entropy`, `spectral_centroid`, `dominant_frequency` | frequency-domain summaries of a signal |
| Preprocessing | `detrend`, `normalize`, `butter_filter`, `lowpass`, `highpass`, `bandpass`, `bandstop` | clean / condition a signal in place |
| Features | `extract_features`, `feature_names`, `hjorth_parameters`, `zero_crossing_rate` | flat feature vectors for ML / comparison |

## Spectral estimators

`power_spectral_density` returns `(freqs, psd)` via Welch's averaged
periodogram (Welch 1967) — the signal is split into overlapping windowed
segments whose periodograms are averaged, trading frequency resolution for
variance. The three summary functions read that PSD: `spectral_entropy` is the
Shannon entropy of the normalised spectrum (0 = a single tone, ≈1 = white
noise), `spectral_centroid` its power-weighted mean frequency, and
`dominant_frequency` the location of its tallest peak (DC excluded by default).

=== "Array + fs"

    ```python
    import numpy as np
    from tsdynamics import transforms as T

    fs = 1000
    t = np.arange(0, 2.0, 1 / fs)
    sine = np.sin(2 * np.pi * 5 * t)          # a clean 5 Hz tone

    f, psd = T.power_spectral_density(sine, fs=fs)   # → (freqs, psd) arrays
    T.dominant_frequency(sine, fs=fs, nperseg=2000)  # ≈ 5.0  (Hz)
    T.spectral_entropy(sine, fs=fs, nperseg=2000)    # ≈ 0.126 (sharply peaked)
    T.spectral_centroid(sine, fs=fs)                 # ≈ 4.47 (Hz)
    ```

=== "Trajectory"

    ```python
    # dt / fs is read from traj.meta; one result per component
    T.dominant_frequency(traj)        # → array, e.g. [5.0]  (one entry per channel)
    T.spectral_entropy(traj)          # → array, one per channel
    ```

!!! note "Frequency resolution"
    Welch's default segmenting gives coarse bins — on the 2 s tone above the
    raw default lands the peak at ≈3.9 Hz. Widen `nperseg` (here `2000`, a
    single full segment) or pass `method="periodogram"` for the full-record
    estimate when you need the peak *exactly* on 5 Hz.

White noise spreads its power across all bins, so its spectral entropy climbs
toward the ceiling:

| Signal (`fs = 1000`, 2 s) | `dominant_frequency` | `spectral_entropy` |
|---|---|---|
| pure 5 Hz sine (`nperseg=2000`) | `5.0` Hz | `0.126` |
| Gaussian white noise | — (flat) | `0.939` |

(Values run on the fast synthetic signals above — a seeded
`np.random.default_rng(0)` for the noise row, `method="periodogram"`.)

## Preprocessing

`detrend` removes a `"linear"` (default) or `"constant"` trend; `normalize`
rescales by `"zscore"` (default, → zero mean / unit variance), `"minmax"`, or
similar. The Butterworth family wraps `butter_filter(x, cutoff, *, btype=...)`
in named conveniences — `lowpass` / `highpass` take a scalar cutoff,
`bandpass` / `bandstop` take a `(low, high)` pair:

```python
clean = T.detrend(sine)                    # drop a linear trend
z     = T.normalize(sine)                  # zero mean, unit variance
lp    = T.lowpass(sine, 10.0, fs=fs)       # 4th-order Butterworth, < 10 Hz
bp    = T.bandpass(sine, 3.0, 7.0, fs=fs)  # keep the 3–7 Hz band
```

Each returns a signal the same shape as its input, so they chain and compose
with the analysis layer (filter, then embed or run RQA).

## Feature extraction

`extract_features` evaluates a selection from the `FEATURE_FUNCTIONS` registry
into a flat `dict` — pass `features=[...]` to pick a subset, or omit it for the
full bank. `feature_names()` lists what is available:

```python
T.feature_names()
# ['mean', 'std', 'var', 'rms', 'ptp', 'skewness', 'kurtosis',
#  'zero_crossing_rate', 'hjorth_activity', 'hjorth_mobility',
#  'hjorth_complexity', 'dominant_frequency', 'spectral_centroid',
#  'spectral_entropy']

T.extract_features(sine, fs=fs,
                   features=["dominant_frequency", "spectral_entropy"])
# {'dominant_frequency': 3.906..., 'spectral_entropy': 0.195...}
```

Two estimators are also exposed directly. `hjorth_parameters` returns the three
**Hjorth descriptors** — activity (signal variance), mobility (mean frequency
proxy) and complexity (bandwidth proxy) — the classic time-domain EEG
descriptors (Hjorth 1970):

```python
T.hjorth_parameters(sine)
# {'activity': 0.5, 'mobility': 0.0314, 'complexity': 1.001}
T.zero_crossing_rate(sine)        # ≈ 0.0095  (fraction of samples that cross 0)
```

!!! note "Trajectory features are per component"
    Given a `Trajectory`, `extract_features` returns one value per feature *per
    channel*; given a 1-D array it returns plain scalars. `fs` / `dt` are read
    from `traj.meta` when present, so spectral features need no extra argument.

## See also

- [Recurrence & RQA](../analysis/recurrence.md) — quantify a (filtered) signal's
  recurrence structure
- [Entropy](../analysis/entropy.md) — complexity measures that complement the
  spectral ones here
- [Lyapunov spectra](../analysis/lyapunov.md) — chaos quantifiers for the same
  signals

## References

- Welch (1967), *IEEE Trans. Audio Electroacoust.* **15**, 70.
- Hjorth (1970), *Electroencephalogr. Clin. Neurophysiol.* **29**, 306.
