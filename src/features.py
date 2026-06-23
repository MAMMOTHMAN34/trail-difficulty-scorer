"""
Reusable trail feature engineering.

These three functions are the single source of truth for turning a GPX file
into difficulty features. They're imported by the notebooks, the model training
code, and the web app — so the features are computed identically everywhere.
"""

import gpxpy
import numpy as np
import pandas as pd
from defusedxml.ElementTree import fromstring as _safe_xml_parse
from defusedxml.common import DefusedXmlException

# Upper bound on track points we'll process from an uploaded file (DoS guard).
MAX_POINTS = 3000


def _gpx_to_df(gpx):
    """Collect every track (or route) point from a parsed GPX into a DataFrame."""
    rows = []
    for track in gpx.tracks:
        for segment in track.segments:
            for p in segment.points:
                rows.append({"lat": p.latitude, "lon": p.longitude, "elevation_m": p.elevation})
    if not rows:  # some files store points as routes instead of tracks
        for route in gpx.routes:
            for p in route.points:
                rows.append({"lat": p.latitude, "lon": p.longitude, "elevation_m": p.elevation})
    if not rows:
        raise ValueError("No track or route points found in this GPX file.")
    return pd.DataFrame(rows)


def load_trail(path):
    """Read a GPX file from disk and return a DataFrame of its track points."""
    with open(path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    return _gpx_to_df(gpx)


def load_trail_from_text(text):
    """
    Parse GPX content (a string from an uploaded file) into a DataFrame.

    Hardened for untrusted input: rejects malicious XML (entity-expansion bombs,
    external entities) before parsing, and caps the number of points processed.
    """
    # Safety gate: screen the XML before gpxpy touches it
    try:
        _safe_xml_parse(text)
    except DefusedXmlException:
        raise ValueError("This file was rejected as potentially malicious XML.")
    except Exception:
        raise ValueError("This doesn't look like a valid GPX file.")

    df = _gpx_to_df(gpxpy.parse(text))

    # Cap points so an oversized (but valid) track can't exhaust memory
    if len(df) > MAX_POINTS:
        idx = np.linspace(0, len(df) - 1, MAX_POINTS).astype(int)
        df = df.iloc[idx].reset_index(drop=True)

    return df


def haversine(lat1, lon1, lat2, lon2):
    """Ground distance in metres between two lat/lon points."""
    R = 6_371_000  # Earth's radius in metres
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_features(df, smooth_window=3):
    """Turn a trail DataFrame (lat, lon, elevation_m) into difficulty features."""
    d = df.copy()

    # Smooth elevation to remove GPS jitter, then measure changes on the smoothed line
    d["elev_smooth"] = d["elevation_m"].rolling(smooth_window, center=True, min_periods=1).mean()
    d["elev_change"] = d["elev_smooth"].diff().fillna(0)

    # Horizontal distance between consecutive points
    d["step_m"] = haversine(d["lat"].shift(), d["lon"].shift(), d["lat"], d["lon"]).fillna(0)

    # Slope of each segment in degrees (avoid divide-by-zero on stationary points)
    horiz = d["step_m"].replace(0, np.nan)
    slope = np.degrees(np.arctan(d["elev_change"] / horiz)).fillna(0).abs()

    dist_km = d["step_m"].sum() / 1000
    gain = d.loc[d["elev_change"] > 0, "elev_change"].sum()
    loss = -d.loc[d["elev_change"] < 0, "elev_change"].sum()

    elev = d["elevation_m"]
    elev_range = elev.max() - elev.min()
    exposure = (elev >= elev.min() + 0.8 * elev_range).mean()

    return {
        "distance_km": round(dist_km, 2),
        "elevation_gain_m": round(gain),
        "elevation_loss_m": round(loss),
        "gain_rate_m_per_km": round(gain / dist_km, 1),
        "max_elevation_m": round(elev.max()),
        "min_elevation_m": round(elev.min()),
        "elevation_range_m": round(elev_range),
        "mean_slope_deg": round(slope.mean(), 1),
        "max_slope_deg": round(slope.max(), 1),
        "slope_variability": round(slope.std(), 1),
        "pct_steep_segments": round((slope > 20).mean() * 100, 1),
        "exposure_index": round(exposure, 2),
    }
