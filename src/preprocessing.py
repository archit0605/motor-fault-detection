"""
Basic time-domain signal conditioning: detrending, bandpass filtering, an
explicit convolution-based smoother, and autocorrelation.
"""

import numpy as np
from scipy.signal import detrend, butter, filtfilt, freqz


def remove_dc_and_detrend(x):
    """Remove any DC offset and linear drift. A DC bias would show up as a
    spurious spike at 0 Hz and a linear trend smears energy across many low
    frequency bins, both of which corrupt the FFT/PSD we compute later."""
    x = x - np.mean(x)
    return detrend(x)


def butter_bandpass(lowcut=10, highcut=5000, fs=12000, order=4):
    """Design a Butterworth bandpass filter. Butterworth is chosen because
    it has a maximally flat passband (no ripple to distort harmonic
    amplitudes we later measure). Returns both the (b, a) transfer-function
    coefficients and the numerically computed frequency response so the
    filter's Bode-style magnitude/phase plot can be drawn.
    """
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass")
    w, h = freqz(b, a, worN=2048, fs=fs)
    return b, a, w, h


def apply_bandpass(x, b, a):
    """Zero-phase filtering via filtfilt: it runs the filter forward then
    backward, which cancels the phase distortion a single-pass IIR filter
    would introduce. We care about this because the impulse timing of the
    bearing-fault signature must be preserved for the envelope/autocorrelation
    analysis later -- a phase-shifted impulse train would misrepresent BPFO.
    """
    return filtfilt(b, a, x)


def moving_average_convolution(x, M):
    """Moving-average smoother implemented as an explicit convolution with a
    length-M rectangular window (all values 1/M). This *is* discrete
    convolution: y[n] = sum_k x[k] * h[n-k]. A rectangular window in the time
    domain has a sinc-shaped frequency response, so this is really a
    (fairly poor, sidelobe-y) FIR low-pass filter -- useful for showing why
    windowed/Butterworth filters are usually preferred in practice.
    """
    kernel = np.ones(M) / M
    return np.convolve(x, kernel, mode="same")


def normalized_autocorrelation(x):
    """Normalized autocorrelation via FFT (much faster than np.correlate for
    long signals: correlation is multiplication in the frequency domain).
    Used to reveal periodicity -- e.g. the regular impulse spacing of a
    bearing fault shows up as repeated side peaks at multiples of the
    impulse period, even when that periodicity is buried in noise.
    """
    x = x - np.mean(x)
    n = len(x)
    # zero-pad to avoid circular wraparound (linear, not circular, autocorrelation)
    nfft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    X = np.fft.fft(x, nfft)
    acf = np.fft.ifft(X * np.conj(X)).real
    acf = acf[:n]
    acf /= acf[0]  # normalize so zero-lag correlation is 1
    return acf
