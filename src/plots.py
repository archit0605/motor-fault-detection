"""
All figure generation for the pipeline. Figures are numbered in the order
they're narrated in the walkthrough video.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from signal_model import generate_one, FS, CLASSES

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

CLASS_COLORS = {
    "healthy": "tab:green",
    "imbalance": "tab:orange",
    "misalignment": "tab:red",
    "bearing_fault": "tab:purple",
}
CLASS_TITLES = {
    "healthy": "Healthy",
    "imbalance": "Imbalance",
    "misalignment": "Misalignment",
    "bearing_fault": "Bearing Fault (Outer Race)",
}

ENVELOPE_BAND = (2000, 4000)

# Bearing-fault severity is randomized per recording (see signal_model.py) so
# the classifier sees a realistic range of fault strengths. For the
# explanatory figures below we want a clearly diagnosable example, so we use
# a fixed seed known to produce a strong-but-still-physically-valid fault
# rather than whatever severity the default demo seed happens to draw.
DEMO_BEARING_SEED = 21


def _ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name):
    _ensure_fig_dir()
    out_path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_time_domain(seed=42):
    """Figure 1: ~0.2s of raw waveform per class, 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, label in zip(axes.flat, CLASSES):
        x, _, meta = generate_one(label, seed)
        t = np.arange(len(x)) / FS
        mask = t <= 0.2
        ax.plot(t[mask], x[mask], color=CLASS_COLORS[label], linewidth=0.9)
        ax.set_title(CLASS_TITLES[label])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Raw Vibration Waveforms by Fault Class", fontsize=14)
    return _save(fig, "01_time_domain.png")


def plot_filtering(seed=42):
    """Figure 2: noisy vs filtered waveform, plus the Butterworth bandpass's
    magnitude and phase response (Bode-style, log-frequency axis)."""
    from preprocessing import butter_bandpass, apply_bandpass

    x, label, meta = generate_one("bearing_fault", seed)
    t = np.arange(len(x)) / FS
    b, a, w, h = butter_bandpass(lowcut=10, highcut=5000, fs=FS, order=4)
    filtered = apply_bandpass(x, b, a)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    mask = t <= 0.05
    axes[0].plot(t[mask], x[mask], color="gray", alpha=0.7, linewidth=0.9, label="Noisy")
    axes[0].plot(t[mask], filtered[mask], color="tab:blue", linewidth=1.2, label="Filtered (10-5000 Hz)")
    axes[0].set_title("Time Domain: Noisy vs Filtered")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    mag_db = 20 * np.log10(np.abs(h) + 1e-12)
    axes[1].semilogx(w, mag_db, color="tab:blue")
    axes[1].set_title("Butterworth Magnitude Response")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].grid(True, which="both", alpha=0.3)

    phase_deg = np.unwrap(np.angle(h)) * 180 / np.pi
    axes[2].semilogx(w, phase_deg, color="tab:blue")
    axes[2].set_title("Butterworth Phase Response")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Phase (degrees)")
    axes[2].grid(True, which="both", alpha=0.3)

    fig.suptitle("Bandpass Filtering (order-4 Butterworth, zero-phase via filtfilt)", fontsize=13)
    return _save(fig, "02_filtering.png")


def plot_fft_spectra(seed=42):
    """Figure 3: 2x2 grid of FFT magnitude spectra, 0-500 Hz zoom, with the
    1x/2x/3x shaft harmonics annotated so the class signatures described in
    signal_model.py are visually verifiable."""
    from spectral import compute_fft

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, label in zip(axes.flat, CLASSES):
        x, _, meta = generate_one(label, seed)
        freqs, mag = compute_fft(x, FS)
        f_shaft = meta["f_shaft"]

        mask = freqs <= 500
        ax.plot(freqs[mask], mag[mask], color=CLASS_COLORS[label], linewidth=1)
        ax.set_title(CLASS_TITLES[label])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

        for k, style in zip([1, 2, 3], ["--", "-.", ":"]):
            fk = k * f_shaft
            if fk <= 500:
                ax.axvline(fk, color="black", linestyle=style, linewidth=1, alpha=0.6)
                ymax = ax.get_ylim()[1]
                ax.text(fk, ymax * 0.92, f"{k}x", rotation=0, fontsize=9,
                         ha="center", va="top")

    fig.suptitle("FFT Magnitude Spectra by Fault Class (0-500 Hz)", fontsize=14)
    return _save(fig, "03_fft_spectra.png")


def plot_spectrogram(seed=42):
    """Figure 4: STFT spectrograms of healthy vs bearing-fault signals,
    side by side -- the bearing fault shows repeated broadband bursts
    around the ~3 kHz resonance that the healthy signal doesn't have."""
    from spectral import compute_stft

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    seeds = {"healthy": seed, "bearing_fault": DEMO_BEARING_SEED}
    for ax, label in zip(axes, ["healthy", "bearing_fault"]):
        x, _, meta = generate_one(label, seeds[label])
        f, tt, mag = compute_stft(x, FS)
        mag_db = 20 * np.log10(mag + 1e-12)
        pcm = ax.pcolormesh(tt, f, mag_db, shading="gouraud", cmap="viridis")
        ax.set_ylim(0, 6000)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(CLASS_TITLES[label])
        fig.colorbar(pcm, ax=ax, label="Power (dB)")

    fig.suptitle("STFT Spectrogram: Healthy vs Bearing Fault", fontsize=14)
    return _save(fig, "04_spectrogram.png")


def plot_envelope_analysis(seed=DEMO_BEARING_SEED):
    """Figure 5: raw spectrum (BPFO invisible), Hilbert envelope over time,
    and the envelope spectrum with the BPFO peak annotated -- the payoff
    figure showing why envelope analysis is needed at all."""
    from spectral import compute_fft, hilbert_envelope, envelope_spectrum

    x, label, meta = generate_one("bearing_fault", seed)
    bpfo = meta["bpfo"]
    t = np.arange(len(x)) / FS

    freqs, mag = compute_fft(x, FS)
    env = hilbert_envelope(x, FS, band=ENVELOPE_BAND)
    env_freqs, env_mag = envelope_spectrum(x, FS, band=ENVELOPE_BAND)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    mask = freqs <= 500
    axes[0].plot(freqs[mask], mag[mask], color="tab:purple", linewidth=1)
    axes[0].axvline(bpfo, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axes[0].text(bpfo, axes[0].get_ylim()[1] * 0.9, "BPFO\n(not visible)", fontsize=8, ha="center")
    axes[0].set_title("Raw Spectrum")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    mask_t = t <= 0.5
    axes[1].plot(t[mask_t], env[mask_t], color="tab:purple", linewidth=0.9)
    axes[1].set_title(f"Hilbert Envelope ({ENVELOPE_BAND[0]}-{ENVELOPE_BAND[1]} Hz band)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Envelope Amplitude")
    axes[1].grid(True, alpha=0.3)

    mask_e = env_freqs <= 500
    axes[2].plot(env_freqs[mask_e], env_mag[mask_e], color="tab:purple", linewidth=1)
    bpfo_bin = (env_freqs >= bpfo - 3) & (env_freqs <= bpfo + 3)
    peak_val = np.max(env_mag[bpfo_bin])
    ymax = axes[2].get_ylim()[1]
    axes[2].annotate(
        f"BPFO = {bpfo:.1f} Hz",
        xy=(bpfo, peak_val), xytext=(bpfo + 90, ymax * 0.85),
        fontsize=9, ha="left",
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1),
    )
    axes[2].set_title("Envelope Spectrum (BPFO revealed)")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Envelope (Amplitude Demodulation) Analysis -- Bearing Fault", fontsize=13)
    return _save(fig, "05_envelope_analysis.png")


def plot_autocorrelation(seed=42):
    """Figure 6: autocorrelation of healthy vs bearing-fault signals, both
    computed on the same 2-4 kHz envelope used in Figure 5 (not the raw
    signal). Raw-signal autocorrelation is dominated by the much stronger
    low-frequency shaft harmonics, which would bury the impulse-train
    periodicity we're actually trying to show -- same motivation as the
    envelope spectrum. On the envelope, the bearing signal shows repeated
    peaks spaced at the impulse period; the healthy signal does not.
    """
    from preprocessing import normalized_autocorrelation
    from spectral import hilbert_envelope

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    seeds = {"healthy": seed, "bearing_fault": DEMO_BEARING_SEED}
    for ax, label in zip(axes, ["healthy", "bearing_fault"]):
        x, _, meta = generate_one(label, seeds[label])
        env = hilbert_envelope(x, FS, band=ENVELOPE_BAND)
        acf = normalized_autocorrelation(env)
        lags = np.arange(len(acf)) / FS
        mask = lags <= 0.1
        ax.plot(lags[mask], acf[mask], color=CLASS_COLORS[label], linewidth=1)
        if label == "bearing_fault":
            period = 1.0 / meta["bpfo"]
            for k in range(1, 4):
                ax.axvline(k * period, color="black", linestyle=":", alpha=0.6)
        ax.set_title(CLASS_TITLES[label])
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel("Normalized Autocorrelation\n(of 2-4 kHz envelope)")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Autocorrelation: Periodicity Detection", fontsize=14)
    return _save(fig, "06_autocorrelation.png")


def plot_parseval(seed=42):
    """Figure 7: time-domain vs frequency-domain energy per class, with the
    % error annotated -- a correctness check on our FFT scaling."""
    from spectral import verify_parseval

    energies_t, energies_f, errs = [], [], []
    for label in CLASSES:
        x, _, meta = generate_one(label, seed)
        et, ef, err = verify_parseval(x, FS)
        energies_t.append(et)
        energies_f.append(ef)
        errs.append(err)

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(CLASSES))
    width = 0.35
    ax.bar(x_pos - width / 2, energies_t, width, label="Time-domain energy", color="tab:blue")
    ax.bar(x_pos + width / 2, energies_f, width, label="Frequency-domain energy", color="tab:orange")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([CLASS_TITLES[l] for l in CLASSES], rotation=12)
    ax.set_ylabel("Energy")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for i, err in enumerate(errs):
        ymax = max(energies_t[i], energies_f[i])
        ax.text(i, ymax * 1.02, f"err={err:.1e}%", ha="center", fontsize=8)

    ax.set_title("Parseval's Theorem: Time-Domain vs Frequency-Domain Energy")
    return _save(fig, "07_parseval.png")


def plot_feature_distributions(df, features=None):
    """Figure 8: boxplots of 4 discriminative features grouped by class."""
    if features is None:
        features = ["dominant_freq", "band_ratio_high", "amp_1x", "envelope_bpfo_ratio"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, feat in zip(axes.flat, features):
        data = [df[df["label"] == label][feat].values for label in CLASSES]
        bp = ax.boxplot(data, tick_labels=[CLASS_TITLES[l] for l in CLASSES], patch_artist=True)
        for patch, label in zip(bp["boxes"], CLASSES):
            patch.set_facecolor(CLASS_COLORS[label])
            patch.set_alpha(0.6)
        ax.set_title(feat)
        ax.set_ylabel(feat)
        ax.tick_params(axis="x", rotation=12)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Feature Distributions by Class", fontsize=14)
    return _save(fig, "08_feature_distributions.png")


def plot_model_comparison(results, best_name, cv_mean, cv_std):
    """Figure 9: bar chart of test accuracy / macro-F1 per model. The best
    model's accuracy bar also gets an error bar from its 5-fold
    GroupKFold cross-validation (mean +/- std)."""
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    f1s = [results[n]["f1_macro"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, accs, width, label="Test Accuracy", color="tab:blue")
    ax.bar(x + width / 2, f1s, width, label="Macro F1", color="tab:orange")

    best_idx = names.index(best_name)
    ax.errorbar(x[best_idx] - width / 2, cv_mean, yerr=cv_std, fmt="none",
                ecolor="black", capsize=6, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title(f"Model Comparison (error bar = 5-fold GroupKFold CV on {best_name})")
    return _save(fig, "09_model_comparison.png")


def plot_confusion_matrix(cm, labels, model_name):
    """Figure 10: annotated confusion matrix of the best model."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([CLASS_TITLES[l] for l in labels], rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([CLASS_TITLES[l] for l in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix -- {model_name}")

    thresh = cm.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, label="Count")
    return _save(fig, "10_confusion_matrix.png")


def plot_feature_importance(fi_list):
    """Figure 11: horizontal bar chart of the top-12 Random Forest feature
    importances."""
    names = [n for n, _ in fi_list][::-1]
    vals = [v for _, v in fi_list][::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, vals, color="tab:green")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance (Top 12)")
    ax.grid(True, alpha=0.3, axis="x")
    return _save(fig, "11_feature_importance.png")


if __name__ == "__main__":
    plot_time_domain()
    plot_filtering()
    plot_fft_spectra()
    plot_spectrogram()
    plot_envelope_analysis()
    plot_autocorrelation()
    plot_parseval()
