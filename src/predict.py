"""
Prediction pipeline used by the web app.

Takes raw GPX text -> DataFrame -> (elevation if missing) -> features ->
trained model -> difficulty label + probabilities.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from features import load_trail_from_text, compute_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "rf_model.pkl"
META_PATH = ROOT / "models" / "model_meta.json"

TOPO_API = "https://api.opentopodata.org/v1/srtm30m"

_model = None
_meta = None
_explainer = None


def _load_model():
    """Load the model and metadata once, then reuse (cached in module globals)."""
    global _model, _meta
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _meta = json.load(open(META_PATH))
    return _model, _meta


def _get_explainer():
    """Build the SHAP TreeExplainer once, then reuse (cached in module globals)."""
    global _explainer
    if _explainer is None:
        import shap

        model, _ = _load_model()
        _explainer = shap.TreeExplainer(model)
    return _explainer


def explain_prediction(features, label):
    """
    Return how much each terrain feature pushed this trail toward the predicted
    `label`, as a dict {feature_name: signed_contribution}.

    Positive = pushed the prediction toward this difficulty; negative = pushed away.
    Uses SHAP values for the single predicted class.
    """
    model, meta = _load_model()
    explainer = _get_explainer()

    X = pd.DataFrame([features])[meta["features"]]
    sv = np.array(explainer.shap_values(X))

    n_feat = len(meta["features"])
    n_class = len(model.classes_)
    class_int = meta["order"].index(label)
    class_pos = list(model.classes_).index(class_int)

    # SHAP's output shape varies by version: a per-class list -> (n_class, 1, n_feat),
    # or a newer 3-D array -> (1, n_feat, n_class). Handle both.
    if sv.shape == (n_class, 1, n_feat):
        contrib = sv[class_pos, 0, :]
    elif sv.shape == (1, n_feat, n_class):
        contrib = sv[0, :, class_pos]
    else:  # binary/regression fallback
        contrib = sv.reshape(-1)[:n_feat]

    return {f: float(c) for f, c in zip(meta["features"], contrib)}


def ensure_elevation(df):
    """If a GPX has no usable elevation, fetch it from OpenTopoData (SRTM 30m)."""
    if df["elevation_m"].notna().all() and (df["elevation_m"] != 0).any():
        return df

    df = df.copy()
    coords = list(zip(df["lat"], df["lon"]))
    elevations = []
    for i in range(0, len(coords), 100):  # API allows 100 points/request
        batch = coords[i : i + 100]
        locations = "|".join(f"{lat},{lon}" for lat, lon in batch)
        resp = requests.get(TOPO_API, params={"locations": locations}, timeout=30)
        resp.raise_for_status()
        elevations.extend(r["elevation"] for r in resp.json()["results"])
        time.sleep(1)  # be polite to the free API
    df["elevation_m"] = elevations
    return df


def predict_from_gpx_text(text):
    """Full pipeline: GPX text -> dict with label, probabilities, features, and points."""
    model, meta = _load_model()

    df = load_trail_from_text(text)
    df = ensure_elevation(df)
    features = compute_features(df)

    X = pd.DataFrame([features])[meta["features"]]
    proba = model.predict_proba(X)[0]
    order = meta["order"]

    # model.classes_ are integer indices into `order`
    proba_by_label = {order[int(c)]: float(p) for c, p in zip(model.classes_, proba)}
    label = max(proba_by_label, key=proba_by_label.get)

    return {
        "label": label,
        "probabilities": proba_by_label,
        "features": features,
        "points": df,
    }
