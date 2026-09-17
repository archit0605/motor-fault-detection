# Walkthrough: Motor Fault Detection Pipeline

This is the script for the video walkthrough. Each section explains one pipeline stage --
what it does, why it's there, and which Signals & Systems topic it's really demonstrating.

## 1. Synthetic signal generation (`src/signal_model.py`)

**What it does.** Generates 2-second vibration recordings at 12 kHz for four motor
conditions: healthy, imbalance, misalignment, and outer-race bearing fault. Every signal is
built as a sum of sinusoids at the shaft rotation frequency and its harmonics (1x, 2x, 3x),
plus additive white Gaussian noise.

**Why it's built this way.** A rotating machine's vibration is periodic at the shaft
rotation rate, so its steady-state signature is naturally described by a **Fourier series**:
a fundamental plus harmonics, each with its own amplitude and phase. Different faults
change *which* harmonic dominates -- that's the whole basis for diagnosing faults from
vibration spectra, and it's exactly the Fourier series idea from the course, just applied
to a real engineering signal instead of a square wave.

**The bearing fault is the interesting one.** Instead of adding energy at a shaft harmonic,
a bearing defect on the outer race gets struck by a rolling element once per revolution at
a rate called the ball-pass frequency (BPFO), here set to 3.5x the shaft speed. Each strike
excites a **resonance** in the bearing housing -- a natural structural frequency around 3
kHz that "rings" briefly after being hit and decays. That ringing is modeled as
`exp(-zeta*wn*t) * sin(wd*t)`, which is literally the **impulse response of an underdamped
2nd-order LTI system** (a mass-spring-damper), the same math used for RLC circuits and
mechanical resonators. The impulse train is built by convolving a train of (slightly
jittered, slightly randomized-amplitude) delta functions with that impulse response --
convolution because the total signal really is the sum of many shifted, scaled copies of
one system's response to repeated strikes. Crucially, the fault energy is *small* at the
shaft harmonics and *large* around 3 kHz -- it's hidden from a casual low-frequency look,
which sets up why envelope analysis (stage 3) is necessary at all.

## 2. Preprocessing (`src/preprocessing.py`)

**What it does.** Removes DC offset and drift, applies a Butterworth bandpass filter, and
demonstrates an explicit convolution-based moving-average smoother and autocorrelation.

**Why it's built this way.** Detrending is signal operations 101: a nonzero mean creates a
spurious spike at 0 Hz in the FFT, and any linear drift smears energy across low-frequency
bins, both of which corrupt every spectral estimate downstream.

The bandpass filter is designed with `scipy.signal.butter` -- a **Butterworth filter**,
chosen specifically because it has a maximally flat passband, so it doesn't ripple and
distort the harmonic amplitudes we're about to measure. We apply it with `filtfilt`
(forward-backward filtering) rather than a single causal pass, because that cancels out
phase distortion entirely -- **zero-phase filtering**. That matters here because the
bearing-fault impulse timing has to be preserved exactly for the envelope and
autocorrelation analysis; a phase-shifted impulse train would misrepresent the BPFO.

The moving-average smoother is implemented with an explicit `np.convolve` against a
rectangular window, on purpose, instead of calling a library smoother -- because this *is*
discrete **convolution**: `y[n] = sum_k x[k] h[n-k]`. And because a rectangular window's
Fourier transform is a sinc function, this "simple" smoother is really a fairly crude FIR
low-pass filter with sinc-shaped sidelobes, which is a nice bridge to why designed filters
like the Butterworth above are usually preferred.

`normalized_autocorrelation` computes **correlation** (via FFT, since correlation is
multiplication in the frequency domain) to detect hidden periodicity -- useful later for
spotting the bearing impulse spacing even when it's buried in noise.

## 3. Spectral analysis (`src/spectral.py`)

**What it does.** FFT magnitude spectra, Welch PSD, STFT spectrograms, and Hilbert
envelope / envelope-spectrum analysis, plus a Parseval energy check.

**Why it's built this way.** `compute_fft` is a direct application of the **discrete
Fourier transform**, carefully scaled (doubling all bins except DC/Nyquist) so the reported
amplitude actually matches the amplitude of each sinusoidal component in the time domain --
not just a relative shape. `verify_parseval` checks that time-domain energy
(`sum(x**2)`) equals frequency-domain energy (`sum(|X|**2)/N`), which is **Parseval's
theorem** -- energy is conserved under the Fourier transform, and if our FFT scaling were
wrong, this check would fail. It agrees to well under 1% here (essentially floating-point
precision), which is the sanity check that the rest of the frequency-domain math is
trustworthy.

`compute_psd` uses Welch's method -- averaging periodograms over overlapping segments --
to get a lower-variance estimate of **power spectral density**, power per unit frequency.
From the PSD we compute `spectral_centroid`, `spectral_bandwidth`, and `spectral_entropy`:
the "center of mass," the power-weighted spread around it, and how concentrated vs.
noise-like the spectrum is. These are the **energy/power spectral density and bandwidth**
concepts from the syllabus, turned into single numbers a classifier can use.

`compute_stft` trades some frequency resolution for time resolution by windowing the signal
and taking an FFT of each chunk -- the classic time-frequency tradeoff, letting us see *when*
spectral content appears, not just whether it's present anywhere in the 2-second window.

The envelope analysis is the payoff of the whole DSP chapter. We band-pass around the ~3
kHz resonance, take the magnitude of the analytic signal (`scipy.signal.hilbert`) to get the
envelope, then FFT *that* envelope. This is **amplitude demodulation** -- exactly the AM
radio idea, where a low-frequency message (the BPFO impulse repetition rate) rides on a
high-frequency carrier (the 3 kHz resonance) and has to be demodulated to be seen. The BPFO
line is essentially invisible in the raw spectrum (figure 5, left panel) but jumps out
cleanly in the envelope spectrum (figure 5, right panel) -- that contrast is the whole
argument for why vibration analysts use envelope analysis on bearing faults instead of just
looking at the raw FFT.

## 4. Feature extraction (`src/features.py`)

**What it does.** Splits every 2-second recording into 1-second windows with 50% overlap,
then computes ~24 time-domain, frequency-domain, and envelope-domain numbers per window.

**Why it's built this way.** Windowing turns each recording into several training examples
and mimics how a real monitoring system would process a continuous stream in chunks.
Time-domain features (RMS, kurtosis, crest factor, etc.) summarize the waveform shape --
kurtosis in particular is the classic "impulsiveness" detector, since sharp isolated spikes
(like bearing impulses) drive up the fourth moment far more than smooth periodic content.
Frequency-domain features search narrow bins around the shaft harmonics (this is just
reading amplitudes off the Fourier series we built in stage 1) and compute band-energy
ratios and the spectral shape statistics from stage 3. Envelope-domain features measure how
strong the BPFO peak is in the envelope spectrum relative to the noise floor around it --
directly operationalizing the envelope-analysis argument above into a number a classifier
can threshold on.

## 5. Classification (`src/classify.py`)

**What it does.** Trains Logistic Regression, an RBF-kernel SVM, and a Random Forest on the
feature table, using a `StandardScaler` inside a `Pipeline`, and compares them.

**The one methodological point worth saying out loud on camera:** the train/test split is
done with `GroupShuffleSplit`, grouped by `recording_id`, **not** by individual window.
Adjacent windows from the same recording overlap by 50% of their samples. If we split by
window instead, a window from a given recording could land in training while another window
from the *same* recording lands in test -- the model would then be partly recognizing that
specific recording rather than generalizing to an unseen machine, which leaks information
across the split and inflates test accuracy. Grouping by `recording_id` keeps every window
from a given recording entirely on one side of the split, and the same reasoning is why
cross-validation uses `GroupKFold` instead of plain `KFold`.

## 6. Results

Best model: **Logistic Regression**, test accuracy **0.938**, 5-fold `GroupKFold`
cross-validation **0.946 ± 0.030**. The confusion matrix shows the classifier's only real
mistakes are between `imbalance` and `healthy` -- which makes physical sense, since a
*mild* imbalance is, by construction, a small perturbation on top of a healthy signature,
and the two classes are designed to overlap at the low-severity end rather than being
trivially separable. Random Forest feature importances confirm the DSP story: the top
features are `dominant_freq`, `band_ratio_high` (how much energy sits in the 2-6 kHz band,
which is where the bearing resonance lives), `amp_1x`, and the envelope-domain BPFO
features -- exactly the quantities the walkthrough above argued should matter.
