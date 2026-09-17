"""
Train and compare classifiers on the windowed feature table.

The key methodological point (worth saying out loud in the video): we split
by recording_id, not by individual window. Adjacent windows from the same
2s recording overlap by 50% and share most of their samples, so if a window
from a recording ends up in training and another window from the *same*
recording ends up in test, the model can partly "recognize" that recording
rather than generalize to unseen machines -- this leaks information across
the split and inflates test accuracy. Grouping by recording_id keeps every
window from a given recording on the same side of the split.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from signal_model import SEED, FS

FEATURE_COLS_EXCLUDE = {"label", "recording_id", "window_id"}


def get_feature_columns(df):
    return [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]


def split_by_recording(df, test_size=0.3, seed=SEED):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["recording_id"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def _make_models(seed=SEED):
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            # CalibratedClassifierCV wraps the SVM to give us predict_proba
            # without the deprecated SVC(probability=True) path
            ("clf", CalibratedClassifierCV(SVC(kernel="rbf", random_state=seed), ensemble=False)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),  # not strictly needed for trees, but keeps the pipeline uniform
            ("clf", RandomForestClassifier(n_estimators=200, random_state=seed)),
        ]),
    }


def train_and_compare(df, seed=SEED):
    """Train all three models on a recording-level train/test split. Returns
    a dict of per-model results (accuracy, precision/recall/f1, reports,
    confusion matrices) plus the fitted pipelines."""
    feature_cols = get_feature_columns(df)
    train_df, test_df = split_by_recording(df, seed=seed)

    X_train, y_train = train_df[feature_cols].values, train_df["label"].values
    X_test, y_test = test_df[feature_cols].values, test_df["label"].values

    models = _make_models(seed=seed)
    results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
        report = classification_report(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=sorted(df["label"].unique()))

        results[name] = {
            "pipeline": pipe,
            "accuracy": acc,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "report": report,
            "confusion_matrix": cm,
            "labels": sorted(df["label"].unique()),
        }
    return results, feature_cols, train_df, test_df


def cross_validate_best(df, best_model_name, feature_cols, seed=SEED, n_splits=5):
    """5-fold GroupKFold cross-validation for the best model, grouped by
    recording_id for the same leakage reason described above."""
    models = _make_models(seed=seed)
    pipe = models[best_model_name]

    X = df[feature_cols].values
    y = df["label"].values
    groups = df["recording_id"].values

    gkf = GroupKFold(n_splits=n_splits)
    scores = cross_val_score(pipe, X, y, groups=groups, cv=gkf, scoring="accuracy")
    return scores.mean(), scores.std(), scores


def random_forest_feature_importance(results, feature_cols, top_n=12):
    rf_pipe = results["Random Forest"]["pipeline"]
    importances = rf_pipe.named_steps["clf"].feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    return [(feature_cols[i], importances[i]) for i in order]


def predict_condition(signal, fs, model_pipe, model_metadata=None):
    """Single-signal inference: window the signal, extract the same
    features used in training, average the per-window class probabilities,
    and return the predicted label + probability dict."""
    from features import extract_features_for_recording

    f_shaft = (model_metadata or {}).get("f_shaft", 30.0)
    rows = extract_features_for_recording(signal, fs, f_shaft, label="unknown", recording_id="demo")
    feat_df = pd.DataFrame(rows)
    feature_cols = [c for c in feat_df.columns if c not in FEATURE_COLS_EXCLUDE]

    X = feat_df[feature_cols].values
    proba = model_pipe.predict_proba(X).mean(axis=0)
    classes = model_pipe.named_steps["clf"].classes_
    class_probs = dict(zip(classes, proba))
    predicted = max(class_probs, key=class_probs.get)
    return predicted, class_probs
