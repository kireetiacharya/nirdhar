---
title: Nirdhar
emoji: 🏠
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Nirdhar — AI-powered digital property valuation (prototype)

Hero Housing Fintech Challenge · Problem Statement 1 · Pune (primary market) → Hyderabad → India.

Nirdhar values a RERA-registered apartment the way an expert valuer does: from real comparables,
corrected for what listings hide, with a calibrated 90% range whose width decides the lending route.

## Run it

**Pune is the primary market**, because its Index-II registered prices are reachable through IGR Maharashtra e-Search. Hyderabad is the expansion market.

```powershell
pip install -r requirements.txt
# 0. REAL Pune layers (OpenStreetMap western-zone .gpkg from Geofabrik + Makaan rent CSV), ~5 s
python -m nirdhar.build_real_pune --gpkg ..\western-zone.gpkg --rents ..\Indian_housing_Pune_data.csv
python -m nirdhar.generate      # 1. raw data (synthetic transactions on real Pune geography, ~5 s)
python -m nirdhar.train         # 2. clean + merge, train, validate, package (~50 s)
python -m uvicorn app:app --port 8000   # 3. open http://localhost:8000
```

`run.ps1` does all three. The trained engine is already in `artifacts/`, so step 3 alone is enough for a demo.

## Demo script (3 minutes)

1. **Happy path**: Kondapur 2 BHK in a reliable builder's project. Grade **A** range, agreement value in range → **Desktop valuation**.
2. **Hard case**: delayed project in a Nagole flood cell. Range widens to about ±16% → grade **C** → **field visit**. The requested loan exceeds the LTV limit computed on the 5th-percentile value.
3. **Fraud catch**: agreement value about 27% above estimate → red flag **refer before disbursal**.
4. **Model & validation** tab: the ablation ladder from the deck, scored on held-out registered deeds.
5. **Open valuation report** gives a one-page PDF. **Replay** re-runs it and proves an exact match (audit).

## What is real and what is synthetic

| | |
|---|---|
| **Real** | Micro-market, metro-station and IT-hub locations (approximate); the full pipeline, models, calibration, explanations, routing, report and audit. |
| **Synthetic** | All transactions, listings, deeds, RERA records, builders (fictional names) and raster features. The generator (`nirdhar/generate.py`) reproduces the *defects* of the real sources: lakh/crore strings, carpet vs super built-up area, cross-portal duplicates, typos, misspelt project names, understated deeds, sparse Hyderabad deeds. The models see only these raw files. Hidden truth is used only to score the pipeline. |

Every screen and report carries this disclosure. Confidence ranges are **genuinely calibrated** on
held-out (synthetic) registered deeds, never hard-coded. Real-data accuracy comes from the pilot.

## Architecture (maps to deck slides 6–12)

```
data/raw/*.csv ──► pipeline.py ──► sales table (deeds + deed-calibrated listings, carpet basis, as-of log ₹/sqft)
   1 normalise units  2 resolve RERA project (fuzzy+geo)  3 geocode to cell  4 dedupe  5 robust outliers
   6 hedonic time index  7 listing→deed calibration (learned on Pune, transferred to Hyderabad)  8 spatial join  9 comps pool

models.py
   M1  LightGBM, monotonic constraints, TreeSHAP-capable
   M2  comparable-sales attention: softmax over best 30 comps; distance/size/age/floor/BHK/same-project/recency
       scales tuned on a validation split; each comp hedonically adjusted to the subject
   M3  hierarchical hedonic: city → micro-market → project random effects (ridge-shrunk)
   Stacker: convex weights per local-data-density bucket
   Uncertainty: heteroscedastic scale model × (1 + OOD) → spatially weighted split-conformal 90% range
   Explanations: exact grouped Shapley (8 groups, 256 coalitions) of the full ensemble vs a typical flat, in ₹

service.py: three-price reconciliation, project intelligence, risk flags, routing + risk-adjusted LTV, audit/replay
app.py:     FastAPI  ·  POST /api/valuations · GET /api/valuations/{id} · POST /api/valuations/{id}/replay · GET /api/model · /docs
web/:       single-page UI + printable report
```

Validation protocol: time split (train ≤ Jun 2025, val Jul–Dec 2025, calibration Jan–Mar 2026,
test Apr–Sep 2026 registered deeds). Metrics: MdAPE, PE10/PE20, COD, PRD, interval coverage, and
per-grade error. See `artifacts/model_card.json`.

## Environment notes

* Python 3.15, numpy, pandas, LightGBM, FastAPI. On this machine an application-control policy blocks
  scikit-learn's and h3's native DLLs, so the prototype uses numpy implementations, and a fixed
  ~0.38 km grid stands in for H3 resolution 9. Swapping in `h3` is a one-function change in `geo.cell_id`.
* The map and fonts load from unpkg / Google Fonts. Offline, the map shows a placeholder.

## Going to real data

Replace the generator with connectors that write the same raw CSVs:
IGR Maharashtra Index-II (deeds), IGRS Telangana guidance rates + EC samples, TS-RERA / MahaRERA,
portal listings, Open Buildings / VIIRS / Sentinel-2 / OSM rasters per cell. Then run `nirdhar.train`.
In a pilot, Hero Housing's past valuer reports become deed-grade labels.
