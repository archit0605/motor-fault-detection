"""
Turn raw recordings into a tidy feature table for classification.

Each 2s recording is split into overlapping 1s windows (so we get several
labelled examples per recording, which also gives GroupShuffleSplit in
classify.py real groups to split on), and each window is reduced to a
handful of time-domain, frequency-domain, and envelope-domain numbers that
a real vibration analyst would look at.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from spectral import compute_psd, envelope_spectrum, spectral_centroid, spectral_bandwidth, spectral_entropy, band_energy
from signal_model import BPFO_RATIO

WINDOW_SEC = 1.0
OVERLAP = 0.5
HARMONIC_SEARCH_HZ = 2.0   # +/- Hz around k*f_shaft when looking for a harmonic peak
BPFO_SEARCH_HZ = 3.0
ENVELOPE_BAND = (2000, 4000)  # band around the resonance used for amplitude demodulation


def _window_signal(x, fs, window_sec=WINDOW_SEC, overlap=OVERLAP):
    win_len = int(window_sec * fs)
    hop = int(win_len * (1 - overlap))
    windows = []
    start = 0
    while start + win_len <= len(x):
        windows.append(x[start:start + win_len])
        start += hop
    return windows


def _time_domain_features(w):
    rms = np.sqrt(np.mean(w ** 2))
    peak = np.max(np.abs(w))
    mean_abs = np.mean(np.abs(w))
    return {
        "rms": rms,
        "variance": np.var(w),
        "peak": peak,
        "peak_to_peak": np.max(w) - np.min(w),
        "crest_factor": peak / rms if rms > 0 else 0.0,
        "kurtosis": kurtosis(w),          # heavy tails -> impulsive signal (bearing faults are classically high-kurtosis)
        "skewness": skew(w),
        "shape_factor": rms / mean_abs if mean_abs > 0 else 0.0,
        "zero_crossing_rate": np.sum(np.diff(np.sign(w)) != 0) / len(w),
    }


def _harmonic_amplitude(freqs, psd, f_target, search_hz=HARMONIC_SEARCH_HZ):
    mask = (freqs >= f_target - search_hz) & (freqs <= f_target + search_hz)
    return np.max(psd[mask]) if np.any(mask) else 0.0


def _freq_domain_features(w, fs, f_shaft):
    freqs, psd = compute_psd(w, fs)
    total_power = np.trapezoid(psd, freqs)
    dominant_freq = freqs[np.argmax(psd)]

    a1 = _harmonic_amplitude(freqs, psd, 1 * f_shaft)
    a2 = _harmonic_amplitude(freqs, psd, 2 * f_shaft)
    a3 = _harmonic_amplitude(freqs, psd, 3 * f_shaft)

    be_low = band_energy(freqs, psd, 0, 500)
    be_mid = band_energy(freqs, psd, 500, 2000)
    be_high = band_energy(freqs, psd, 2000, 6000)
    total_be = be_low + be_mid + be_high

    return {
        "total_power": total_power,
        "dominant_freq": dominant_freq,
        "amp_1x": a1,
        "amp_2x": a2,
        "amp_3x": a3,
        "harmonic_ratio_2x_1x": a2 / a1 if a1 > 0 else 0.0,
        "harmonic_ratio_3x_1x": a3 / a1 if a1 > 0 else 0.0,
        "band_ratio_low": be_low / total_be if total_be > 0 else 0.0,
        "band_ratio_mid": be_mid / total_be if total_be > 0 else 0.0,
        "band_ratio_high": be_high / total_be if total_be > 0 else 0.0,
        "spectral_centroid": spectral_centroid(freqs, psd),
        "spectral_bandwidth": spectral_bandwidth(freqs, psd),
        "spectral_entropy": spectral_entropy(psd),
    }


def _envelope_features(w, fs, bpfo_expected):
    freqs, mag = envelope_spectrum(w, fs, band=ENVELOPE_BAND)
    mask = (freqs >= bpfo_expected - BPFO_SEARCH_HZ) & (freqs <= bpfo_expected + BPFO_SEARCH_HZ)
    peak_env = np.max(mag[mask]) if np.any(mask) else 0.0
    mean_env = np.mean(mag[1:])  # skip DC bin
    return {
        "envelope_bpfo_peak": peak_env,
        "envelope_bpfo_ratio": peak_env / mean_env if mean_env > 0 else 0.0,
    }


def extract_features_for_recording(x, fs, f_shaft, label, recording_id):
    """Window one recording and return a list of per-window feature dicts."""
    bpfo_expected = BPFO_RATIO * f_shaft
    rows = []
    for i, w in enumerate(_window_signal(x, fs)):
        feats = {}
        feats.update(_time_domain_features(w))
        feats.update(_freq_domain_features(w, fs, f_shaft))
        feats.update(_envelope_features(w, fs, bpfo_expected))
        feats["label"] = label
        feats["recording_id"] = recording_id
        feats["window_id"] = f"{recording_id}_w{i}"
        rows.append(feats)
    return rows


def build_feature_table(recordings):
    """recordings: list of (signal, label, metadata) as produced by
    signal_model.generate_dataset(). Returns a tidy pandas DataFrame."""
    rows = []
    for x, label, meta in recordings:
        rows.extend(extract_features_for_recording(x, meta["fs"], meta["f_shaft"], label, meta["recording_id"]))
    return pd.DataFrame(rows)
