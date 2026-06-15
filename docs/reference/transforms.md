---
description: API reference for tsdynamics.transforms — power spectra, detrend/normalize/filter preprocessing, and feature extraction.
---

<span class="ts-kicker">Reference</span>

# Transforms

The signal layer that turns a raw [`Trajectory`](base.md) (or a bare array)
into analysable features: spectral estimators, preprocessing, and a feature
bank. Prose-first treatment lives in the
[Transforms](../transforms/index.md) section.

Every entry point is shape-preserving where it makes sense — pass a
`Trajectory` in and the sampling interval is read from its metadata — and is
re-exported from `tsdynamics.transforms`.

## Spectral

::: tsdynamics.transforms.spectral.power_spectral_density

::: tsdynamics.transforms.spectral.spectral_entropy

::: tsdynamics.transforms.spectral.spectral_centroid

::: tsdynamics.transforms.spectral.dominant_frequency

## Preprocessing

::: tsdynamics.transforms.preprocessing.detrend

::: tsdynamics.transforms.preprocessing.normalize

::: tsdynamics.transforms.preprocessing.butter_filter

::: tsdynamics.transforms.preprocessing.lowpass

::: tsdynamics.transforms.preprocessing.highpass

::: tsdynamics.transforms.preprocessing.bandpass

::: tsdynamics.transforms.preprocessing.bandstop

## Feature extraction

The `FEATURE_FUNCTIONS` registry maps a feature name to its estimator;
[`extract_features`](#tsdynamics.transforms.features.extract_features)
evaluates a selection of them into a flat `dict`, and
[`feature_names`](#tsdynamics.transforms.features.feature_names) lists what is
available.

::: tsdynamics.transforms.features.extract_features

::: tsdynamics.transforms.features.feature_names

::: tsdynamics.transforms.features.hjorth_parameters

::: tsdynamics.transforms.features.zero_crossing_rate
