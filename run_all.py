"""
Runs the full motor-fault-detection pipeline end to end: generate synthetic
vibration data, extract features, train/compare classifiers, save all 11
figures, and run a single-signal demo. Prints a narrated log as it goes so
this script can be screen-recorded directly.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

TOTAL_STEPS = 7


def _header(step, text):
    print(f"\n[{step}/{TOTAL_STEPS}] {text}")


def main():
    t_start = time.time()

    print("=" * 70)
    print("MOTOR FAULT DETECTION -- Time-Frequency Signal Processing + ML")
    print("=" * 70)

    # ---- 1. Generate synthetic dataset -----------------------------------
    _header(1, "Generating synthetic vibration dataset...")
    from signal_model import generate_dataset, SEED, FS, N_PER_CLASS, CLASSES
    recordings = generate_dataset(seed=SEED)
    print(f"    Generated {len(recordings)} recordings "
          f"({N_PER_CLASS} per class x {len(CLASSES)} classes), "
          f"fs={FS} Hz, seed={SEED}")

    # ---- 2. Preprocessing demo (convolution) ------------------------------
    _header(2, "Preprocessing demo: detrend, Butterworth bandpass, convolution smoother...")
    from preprocessing import remove_dc_and_detrend, butter_bandpass, apply_bandpass, moving_average_convolution
    demo_x, demo_label, demo_meta = recordings[0]
    detrended = remove_dc_and_detrend(demo_x)
    b, a, w, h = butter_bandpass(lowcut=10, highcut=5000, fs=FS, order=4)
    filtered = apply_bandpass(detrended, b, a)
    smoothed = moving_average_convolution(filtered, M=15)
    print(f"    Detrended + bandpass-filtered + convolution-smoothed a {demo_label} recording")
    print(f"    Raw RMS={np.sqrt(np.mean(demo_x**2)):.4f}  Filtered RMS={np.sqrt(np.mean(filtered**2)):.4f}  "
          f"Smoothed RMS={np.sqrt(np.mean(smoothed**2)):.4f}")

    # ---- 3. Feature extraction ---------------------------------------------
    _header(3, "Extracting features from 1.0s windows (50% overlap)...")
    from features import build_feature_table
    from spectral import verify_parseval
    df = build_feature_table(recordings)
    n_feature_cols = df.shape[1] - 3  # minus label, recording_id, window_id
    print(f"    Extracted {n_feature_cols} features from {df.shape[0]} windows")

    et, ef, err = verify_parseval(demo_x, FS)
    print(f"    Parseval check: time-domain energy={et:.4f}, freq-domain energy={ef:.4f}, error={err:.2e}%")

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/features.csv", index=False)
    print("    Saved data/processed/features.csv")

    # ---- 4. Optional real-data hook ----------------------------------------
    _header(4, "Checking for real CWRU bearing data...")
    from dataset import load_cwru_if_available
    load_cwru_if_available()

    # ---- 5. Train and compare classifiers ----------------------------------
    _header(5, "Training and comparing classifiers (split by recording_id, not window)...")
    from classify import train_and_compare, cross_validate_best, random_forest_feature_importance
    results, feature_cols, train_df, test_df = train_and_compare(df)
    print(f"    Train: {train_df['recording_id'].nunique()} recordings ({len(train_df)} windows)  "
          f"Test: {test_df['recording_id'].nunique()} recordings ({len(test_df)} windows)")
    print(f"    {'Model':<22s}{'Accuracy':>10s}{'Precision':>12s}{'Recall':>10s}{'F1 (macro)':>12s}")
    for name, r in results.items():
        print(f"    {name:<22s}{r['accuracy']:>10.3f}{r['precision_macro']:>12.3f}"
              f"{r['recall_macro']:>10.3f}{r['f1_macro']:>12.3f}")

    best_name = max(results, key=lambda n: results[n]["accuracy"])
    print(f"\n    Best model: {best_name}")
    print(results[best_name]["report"])

    cv_mean, cv_std, cv_scores = cross_validate_best(df, best_name, feature_cols)
    print(f"    5-fold GroupKFold CV accuracy for {best_name}: {cv_mean:.3f} +/- {cv_std:.3f}")

    fi_list = random_forest_feature_importance(results, feature_cols)

    # ---- 6. Generate all figures --------------------------------------------
    _header(6, "Generating figures...")
    from plots import (
        plot_time_domain, plot_filtering, plot_fft_spectra, plot_spectrogram,
        plot_envelope_analysis, plot_autocorrelation, plot_parseval,
        plot_feature_distributions, plot_model_comparison, plot_confusion_matrix,
        plot_feature_importance,
    )
    plot_time_domain()
    plot_filtering()
    plot_fft_spectra()
    plot_spectrogram()
    plot_envelope_analysis()
    plot_autocorrelation()
    plot_parseval()
    plot_feature_distributions(df)
    plot_model_comparison(results, best_name, cv_mean, cv_std)
    plot_confusion_matrix(results[best_name]["confusion_matrix"], results[best_name]["labels"], best_name)
    plot_feature_importance(fi_list)
    print("    All 11 figures saved to figures/")

    # ---- 7. Single-signal demo ------------------------------------------------
    _header(7, "Single-signal demo: predicting an unseen bearing-fault recording...")
    from signal_model import generate_one
    from classify import predict_condition
    demo_signal, true_label, demo_meta2 = generate_one("bearing_fault", seed=99999)
    pred_label, class_probs = predict_condition(demo_signal, FS, results[best_name]["pipeline"], model_metadata=demo_meta2)
    print(f"    True label:      {true_label}")
    print(f"    Predicted label: {pred_label}")
    print("    Class probabilities:")
    for cls, p in sorted(class_probs.items(), key=lambda kv: -kv[1]):
        print(f"      {cls:<16s} {p:.3f}")

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"Pipeline complete in {elapsed:.1f}s. Figures saved in figures/, "
          f"features in data/processed/features.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
