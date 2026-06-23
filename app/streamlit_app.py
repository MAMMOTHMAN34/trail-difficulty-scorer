"""
Trail Difficulty Scorer — Streamlit web app.

Run locally:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Let the app import our code in src/
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import folium
from streamlit_folium import st_folium

from features import haversine
from predict import predict_from_gpx_text, explain_prediction

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Trail Difficulty Scorer", layout="wide")

# Forest ramp: light sage (easy) → near-black green (very difficult).
COLOURS = {
    "easy": "#8ea57f",
    "moderate": "#577550",
    "difficult": "#2f4a31",
    "very difficult": "#192c20",
}
ORDER = ["easy", "moderate", "difficult", "very difficult"]
GREY = "#3a463d"  # muted tone for "pushed away" SHAP bars

# Plotly inherits Streamlit's dark theme; keep chart backgrounds transparent.
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e8efe4"),
)

st.title("Trail Difficulty Scorer")
st.caption(
    "Upload a GPX track and get an ML-predicted difficulty rating, with the terrain "
    "features that drive it. Trained on 116 hand-labelled trails worldwide."
)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
        1. **Upload** a `.gpx` file (from Wikiloc, Strava, AllTrails, etc.)
        2. The app extracts the route + elevation profile
        3. It computes terrain features (climb, slope, distance, exposure…)
        4. A **Random Forest** model predicts the difficulty

        The model leans most on **total climb and distance**, the endurance
        factors, with steepness in a supporting role.
        """
    )
    st.info("Tip: no elevation in your file? The app fetches it from OpenTopoData automatically.")

# ── About the model (always visible, collapsed) ────────────────────────────────

with st.expander("About the model"):
    st.markdown(
        """
        This scorer was trained on **116 hand-labelled trails** worldwide, balanced
        29 per level (easy / moderate / difficult / very difficult), using difficulty
        ratings from **Wikiloc** as ground truth.

        A **Random Forest** reaches **~41% exact accuracy** on the 4-level scale and
        **~72% within one level** (5-fold cross-validated). Its mistakes are almost
        always near-misses between adjacent difficulties.

        A key finding: *difficult* and *very difficult* trails are nearly identical in
        **steepness**. What separates them is **distance and total climb**. So, at the
        hard end, difficulty is about *endurance*. That's why a model that
        combines features beats any single-threshold rule, and why SHAP is used to show,
        per trail, which features drove the rating.
        """
    )

# ── Inputs ─────────────────────────────────────────────────────────────────────

trail_name = st.text_input("Trail name (optional)", placeholder="e.g. Anello della Cima dell'Albero")
uploaded = st.file_uploader("Upload a .gpx file", type=["gpx"])

if uploaded is None:
    st.info("Upload a GPX file to get a difficulty prediction.")
    st.markdown(
        """
        **What you'll get back:**

        - A predicted difficulty rating with the model's confidence across all four levels
        - A **SHAP explanation** showing which terrain features pushed the rating up or down
        - An interactive **elevation profile** and **route map**
        - A full **terrain breakdown** (11 features) you can download as a CSV
        """
    )
    st.stop()

with st.spinner("Analysing your trail…"):
    text = uploaded.read().decode("utf-8", errors="replace")
    try:
        result = predict_from_gpx_text(text)
    except ValueError as e:
        st.error(f"Could not read that GPX file: {e}")
        st.stop()

label = result["label"]
proba = result["probabilities"]
feats = result["features"]
pts = result["points"]
colour = COLOURS.get(label, "#888")

# ── Headline prediction ───────────────────────────────────────────────────────

heading = trail_name.strip() or "Predicted difficulty"

# The model's strength is "within one level": difficult vs very-difficult trails
# have near-identical terrain, so when the top two ratings are close we surface
# the runner-up instead of showing false precision.
ranked = sorted(proba.items(), key=lambda kv: kv[1], reverse=True)
runner_up, runner_p = ranked[1]
borderline = (proba[label] - runner_p) < 0.15
subtitle = (
    f"{proba[label]:.0%} confidence&nbsp;·&nbsp;borderline {runner_up}"
    if borderline
    else f"{proba[label]:.0%} confidence"
)

st.markdown(
    f"""
    <div style="background:{colour};padding:18px 24px;border-radius:12px;margin:8px 0 4px;
                border:1px solid #8ea57f;">
      <span style="color:#e8efe4;font-size:15px;">{heading}</span><br>
      <span style="color:#ffffff;font-size:34px;font-weight:700;">{label.upper()}</span>
      <span style="color:#e8efe4;font-size:18px;">&nbsp;·&nbsp;{subtitle}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
if borderline:
    st.caption(
        f"This trail sits close to the **{runner_up}** boundary. The model is right "
        "within one difficulty level ~72% of the time, so treat adjacent ratings as plausible."
    )

# Confidence across all classes
conf = go.Figure(
    go.Bar(
        x=[proba[c] for c in ORDER],
        y=[c.title() for c in ORDER],
        orientation="h",
        marker_color=[COLOURS[c] for c in ORDER],
        marker_line=dict(color="#8ea57f", width=1),
        text=[f"{proba[c]:.0%}" for c in ORDER],
        textposition="auto",
    )
)
conf.update_layout(
    height=210, margin=dict(t=10, b=10, l=10, r=10),
    xaxis=dict(range=[0, 1], tickformat=".0%", title=None),
    yaxis=dict(autorange="reversed"),
    **CHART_LAYOUT,
)
st.plotly_chart(conf, use_container_width=True)

# ── Why this rating? (SHAP) ────────────────────────────────────────────────────

st.subheader("Why this rating?")
st.caption(
    f"How each terrain feature pushed the prediction toward **{label}**. "
    "Green pushes toward this rating; grey pushes away."
)

try:
    contrib = explain_prediction(feats, label)
    items = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
    items.reverse()  # biggest at the top of a horizontal bar
    names = [n.replace("_", " ") for n, _ in items]
    vals = [v for _, v in items]
    shap_fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=[colour if v >= 0 else GREY for v in vals],
            marker_line=dict(color="#8ea57f", width=1),
        )
    )
    shap_fig.update_layout(
        height=320, margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title="← pushes away      contribution      pushes toward →",
        **CHART_LAYOUT,
    )
    st.plotly_chart(shap_fig, use_container_width=True)
except Exception as e:
    st.info(f"Explanation unavailable for this trail ({e}).")

# ── Two columns: elevation profile + map ──────────────────────────────────────

left, right = st.columns(2)

# Cumulative distance for the x-axis
step = haversine(pts["lat"].shift(), pts["lon"].shift(), pts["lat"], pts["lon"]).fillna(0)
dist_km = step.cumsum() / 1000

with left:
    st.subheader("Elevation profile")
    elev = go.Figure()
    elev.add_trace(
        go.Scatter(
            x=dist_km, y=pts["elevation_m"],
            fill="tozeroy", mode="lines", line=dict(color="#8ea57f", width=2),
            fillcolor="rgba(142,165,127,0.25)",
        )
    )
    elev.update_layout(
        height=360, margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title="Distance (km)", yaxis_title="Elevation (m)",
        **CHART_LAYOUT,
    )
    st.plotly_chart(elev, use_container_width=True)

with right:
    st.subheader("Route map")
    coords = list(zip(pts["lat"], pts["lon"]))
    m = folium.Map(location=[pts["lat"].mean(), pts["lon"].mean()], zoom_start=12)
    folium.PolyLine(coords, color="#8ea57f", weight=4, opacity=0.9).add_to(m)
    folium.Marker(coords[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(coords[-1], tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    st_folium(m, height=360, use_container_width=True)

# ── Feature breakdown ─────────────────────────────────────────────────────────

st.subheader("Terrain features")

display = {
    "Distance": f"{feats['distance_km']:.1f} km",
    "Total climb": f"{feats['elevation_gain_m']:.0f} m",
    "Total descent": f"{feats['elevation_loss_m']:.0f} m",
    "Climb rate": f"{feats['gain_rate_m_per_km']:.0f} m/km",
    "Max elevation": f"{feats['max_elevation_m']:.0f} m",
    "Elevation range": f"{feats['elevation_range_m']:.0f} m",
    "Mean slope": f"{feats['mean_slope_deg']:.1f}°",
    "Max slope": f"{feats['max_slope_deg']:.1f}°",
    "Slope variability": f"{feats['slope_variability']:.1f}",
    "% steep (>20°)": f"{feats['pct_steep_segments']:.0f}%",
    "Exposure index": f"{feats['exposure_index']:.2f}",
}

cols = st.columns(4)
for i, (name, value) in enumerate(display.items()):
    with cols[i % 4].container(border=True):
        st.metric(name, value)

# ── Download summary ───────────────────────────────────────────────────────────

summary = pd.DataFrame(
    [{"trail_name": trail_name.strip() or "unnamed", "predicted_difficulty": label,
      "confidence": round(proba[label], 3), **feats}]
)
st.download_button(
    "Download summary (CSV)",
    data=summary.to_csv(index=False).encode("utf-8"),
    file_name=f"{(trail_name.strip() or 'trail').replace(' ', '_').lower()}_difficulty.csv",
    mime="text/csv",
)

st.caption(
    "Difficulty is predicted by a Random Forest trained on Wikiloc-labelled trails. "
    "Ratings are inherently subjective; treat this as a helpful estimate."
)
