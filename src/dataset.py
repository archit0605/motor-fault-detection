"""
Build the labelled feature dataset used for training, and (optionally) run
real CWRU bearing-fault .mat files through the identical feature pipeline
if the user has dropped any into data/raw/.
"""

import os
import glob
import numpy as np
import pandas as pd

from signal_model import generate_dataset, FS, SEED
from features import build_feature_table

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_PATH = os.path.join(DATA_DIR, "processed", "features.csv")
RAW_DIR = os.path.join(DATA_DIR, "raw")


def build_feature_dataset(save=True):
    """Generate the synthetic dataset, extract features from every window,
    and save the resulting table to data/processed/features.csv."""
    recordings = generate_dataset(seed=SEED)
    df = build_feature_table(recordings)
    if save:
        os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False)
    return df


def load_cwru_if_available(path=RAW_DIR):
    """Optional real-data hook: if CWRU-style .mat files are present in
    data/raw/, load each one's vibration channel with scipy.io.loadmat and
    push it through the same feature pipeline used for the synthetic data,
    so results are directly comparable. If the folder is empty, this is a
    no-op -- we never download anything or fail the pipeline over it.
    """
    mat_files = glob.glob(os.path.join(path, "*.mat"))
    if not mat_files:
        print("Real-data validation (CWRU) skipped: no .mat files found in data/raw/.")
        return None

    from scipy.io import loadmat
    from features import extract_features_for_recording

    rows = []
    for i, fpath in enumerate(mat_files):
        mat = loadmat(fpath)
        signal_key = next((k for k in mat if k.endswith("_DE_time")), None)
        if signal_key is None:
            print(f"Skipping {fpath}: no *_DE_time channel found.")
            continue
        x = mat[signal_key].squeeze().astype(float)
        label = os.path.splitext(os.path.basename(fpath))[0]
        recording_id = f"cwru_{i:03d}"
        # CWRU recordings are sampled at 12kHz same as our synthetic fs by default;
        # shaft speed isn't known exactly per-file here, so we fall back to the nominal value
        rows.extend(extract_features_for_recording(x, FS, 30.0, label, recording_id))

    if not rows:
        print("Real-data validation (CWRU) skipped: no usable channels found.")
        return None
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_feature_dataset()
    print(f"Built feature table: {df.shape[0]} windows x {df.shape[1]} columns")
    print(f"Saved to {PROCESSED_PATH}")
    load_cwru_if_available()
