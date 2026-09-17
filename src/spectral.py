"""
Frequency-domain analysis: FFT, PSD, STFT, and envelope (amplitude
demodulation) analysis, plus a Parseval energy-conservation check.
"""

import numpy as np
from scipy.signal import welch, stft, hilbert, butter, filtfilt


def compute_fft(x, fs):
    """Single-sided magnitude spectrum, correctly scaled so the amplitude
    at each frequency matches the amplitude of that sinusoidal component in
    the time-domain signal (not just a relative shape).
    """
    n = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # rfft only keeps the non-negative half of the spectrum, so we must put
    # back the amplitude that "belongs" to the discarded negative-frequency
    # half by doubling every bin except DC, and except Nyquist when n is
    # even (Nyquist has no distinct negative-frequency partner in that case)
    mag = np.abs(X) / n
    if n % 2 == 0:
        mag[1:-1] *= 2
    else:
        mag[1:] *= 2
    return freqs, mag


def compute_psd(x, fs, nperseg=2048):
    """Welch's method: average the periodogram over overlapping segments to
    get a lower-variance estimate of power spectral density (power per Hz)."""
    freqs, psd = welch(x, fs=fs, nperseg=min(nperseg, len(x)))
    return freqs, psd


def compute_stft(x, fs, nperseg=1024, noverlap=768):
    """Short-Time Fourier Transform -- slides a window along the signal and
    takes an FFT of each chunk, trading some frequency resolution for time
    resolution so we can see how the spectral content evolves."""
    f, tt, Zxx = stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, tt, np.abs(Zxx)


def hilbert_envelope(x, fs, band=None):
    """Amplitude (envelope) demodulation via the analytic signal.

    If `band` is given as (lo, hi), we first band-pass around a resonance
    so the Hilbert transform tracks the envelope of *that* resonance rather
    than of the whole broadband signal -- this is exactly how real bearing
    diagnostics isolate a high-frequency structural resonance before
    demodulating it.
    """
    if band is not None:
        lo, hi = band
        b, a = butter(4, [lo, hi], btype="bandpass", fs=fs)
        x = filtfilt(b, a, x)
    analytic = hilbert(x)
    return np.abs(analytic)


def envelope_spectrum(x, fs, band=(2000, 4000)):
    """FFT of the envelope. A fault impulse train hidden inside a
    high-frequency resonance shows up here as a clean low-frequency line at
    the impact repetition rate (e.g. BPFO), because amplitude-demodulating
    the resonance strips away the ~3 kHz carrier and leaves the ~100 Hz
    modulation that carries the fault information -- the same idea as AM
    radio demodulation.
    """
    env = hilbert_envelope(x, fs, band=band)
    env = env - np.mean(env)
    freqs, mag = compute_fft(env, fs)
    return freqs, mag


def verify_parseval(x, fs):
    """Parseval's theorem: signal energy computed in the time domain must
    equal energy computed in the frequency domain. This is a correctness
    check on our FFT scaling, not just a theory footnote.
    """
    energy_time = np.sum(x ** 2)

    n = len(x)
    X = np.fft.rfft(x)
    # rfft drops the negative-frequency half, so double all bins except
    # DC/Nyquist to recover the full-spectrum energy sum
    power = np.abs(X) ** 2
    power[1:-1] *= 2
    energy_freq = np.sum(power) / n

    pct_error = 100 * abs(energy_time - energy_freq) / energy_time
    return energy_time, energy_freq, pct_error


def spectral_centroid(freqs, psd):
    """'Center of mass' of the spectrum -- a single number summarizing
    where most of the signal's power sits in frequency."""
    return np.sum(freqs * psd) / np.sum(psd)


def spectral_bandwidth(freqs, psd):
    """Power-weighted spread of the spectrum around the centroid (like a
    frequency-domain standard deviation)."""
    centroid = spectral_centroid(freqs, psd)
    return np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / np.sum(psd))


def spectral_entropy(psd):
    """Shannon entropy of the normalized PSD: low for a spectrum dominated
    by a few tones (e.g. misalignment), high for noise-like broadband
    content (e.g. a less-faulted healthy machine)."""
    p = psd / np.sum(psd)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def band_energy(freqs, psd, lo, hi):
    """Integrate PSD over [lo, hi) Hz to get the power carried in that
    band."""
    mask = (freqs >= lo) & (freqs < hi)
    return np.trapezoid(psd[mask], freqs[mask]) if np.any(mask) else 0.0
