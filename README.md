# Trail Difficulty Scorer

A geospatial machine learning project that predicts hiking trail
difficulty from elevation and route geometry, then predicts through a
live web app where users upload a `.gpx` file and get an instant difficulty score
with a terrain breakdown.

**Live app: [trail-difficulty-scorer.onrender.com](https://trail-difficulty-scorer.onrender.com)**

---

## Why this project

Trail difficulty ratings (Easy / Moderate / Hard) are everywhere on hiking apps,
but they're subjective and inconsistent across regions. This project aims to explore if I can predict difficulty objectively from terrain alone, and explain which terrain features drive the rating.

It combines geospatial data engineering and machine learning, built around a personal love of hiking.

## What it does

- Pulls route geometry from **OpenStreetMap** (via OSMnx) and elevation from the
  **OpenTopoData API** for 100+ trails across different regions
- Engineers terrain features: elevation gain rate, cumulative ascent/descent,
  slope variance, exposure index (ridge vs valley), and more
- Trains and compares **XGBoost vs Random Forest** classifiers
- Uses **SHAP** values to explain which terrain features matter most
- Deploys a **Streamlit** web app where users upload a `.gpx` file and get a
  difficulty prediction + feature breakdown

## Early findings (116-trail dataset)

Across 116 hand-labelled trails, a balanced 29 per difficulty level, terrain
features rise cleanly with difficulty:

| Difficulty       | Avg distance | Avg climb | Mean slope | % steep (>20°) |
|------------------|--------------|-----------|------------|----------------|
| Easy             | 9.6 km       | 416 m     | 7.1°       | 7.2%           |
| Moderate         | 15.1 km      | 838 m     | 8.9°       | 11.1%          |
| Difficult        | 12.5 km      | 911 m     | 10.3°      | 14.5%          |
| Very difficult   | 23.3 km      | 1,362 m   | 10.7°      | 14.9%          |

Steepness, total climb, and climb-rate all increase step-by-step from easy to very
difficult. But notice the top two rows: *difficult* and *very difficult* are almost
identical in steepness (10.3° vs 10.7°); what sets them apart is **distance
(12.5 → 23.3 km) and total climb (911 → 1,362 m)**. It's *endurance*,
not steepness, that tips a trail into "very difficult."

**Takeaway:** difficulty is multi-dimensional. Some trails are hard because they're
steep, others because they're long, so no single feature captures it. This is exactly why the project uses a *model* that combines features rather than a simple rule, and why SHAP is used to reveal, per trail, whether distance or slope drives the rating.

## Model results

A **Random Forest** (trained on the 116 trails, 5-fold cross-validated) reaches
**~41% exact accuracy** on the 4-level scale and **~72% within one level**. Its
mistakes are almost always near-misses between adjacent difficulties, not wild
errors. (XGBoost underperforms on a dataset this small.)

Crucially, **SHAP** shows the model relies on sensible terrain features, led by
total climb (`elevation_gain_m`), elevation loss, climb-rate, and distance, i.e.
the *endurance* factors, with steepness in a supporting role. The labels are
subjective single-source ratings, so explainable performance matters more
here than a headline accuracy number.

## Limitations

The model predicts difficulty from **route geometry alone**: distance, elevation
gain/loss, slope, and exposure. That captures a lot, but real-world difficulty
ratings also depend on things a GPX track simply doesn't contain:

- **Technical terrain**: rock scrambles, ladders, roots, loose footing, stream
  crossings. A trail can be physically gentle but technically demanding.
- **Trail surface, navigation, and exposure** to weather or drop-offs.

This shows up clearly on **long but gentle technical trails**. For example,
Luxembourg's Mullerthal "Little Switzerland" (24 km) and the M³ Moselle Trail
(33 km) are both rated *difficult* on Wikiloc, but their mean slope is only ~4°
with under 3% steep segments, *flatter than the average easy trail in the dataset*.
The model sees gentle terrain and predicts **moderate**, because the rocky, scrambly
character that makes them hard is invisible to elevation data.

There's also a **training-distribution gap**: the 116 trails contain few long-but-flat
routes, so the model hasn't strongly learned that distance alone (without steepness)
can mean difficult. Closing this would need both more such trails in the training set
and features describing trail surface/technicality that GPX doesn't provide.

**Takeaway:** terrain geometry explains much of trail difficulty, but not all of it.

### Next steps

Given more time, there are the most promising directions, roughly in order of expected payoff:

- **Close the data gap.** Add more long-but-gentle and technical trails so the model
  learns that distance alone (without steepness) can mean difficult. This is the single
  biggest lever, since it directly addresses the under-rating shown above.
- **Add technicality features.** Enrich each route with OpenStreetMap trail tags
  (`sac_scale`, `trail_visibility`, `surface`) to capture the rocky/scrambly character
  that elevation data can't see.
- **Reduce label noise.** Average difficulty across multiple sources (Wikiloc, AllTrails,
  Komoot) instead of a single subjective rating, giving the model a cleaner target.
- **Model difficulty as ordinal.** Use ordinal regression so the model is penalised more
  for predicting *easy* on a *very difficult* trail than for an adjacent near-miss.

## Stack

| Layer            | Tools                                          |
|------------------|------------------------------------------------|
| Data sourcing    | OSMnx, OpenTopoData API, gpxpy                 |
| Feature eng.     | pandas, numpy, geopandas, shapely             |
| Modelling        | scikit-learn (Random Forest), XGBoost, SHAP   |
| App & serving    | Streamlit                                     |
| Deployment       | Render (free tier)                            |

## Project structure

```
trail-difficulty-scorer/
├── notebooks/      # exploration & explanation, step by step
├── src/            # reusable pipeline code
├── app/            # Streamlit web app
├── data/           # (gitignored — regenerated by the pipeline)
├── requirements.txt
└── README.md
```
