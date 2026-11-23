"""Synthetic raw data for the Nirdhar prototype.

Real Index-II / IGRS / RERA / portal data cannot be bulk-downloaded for a hackathon, so
this module produces *raw* records with the same shape and the same defects as the real
sources (lakh/crore price strings, carpet vs super built-up area, cross-posted duplicates,
typos, misspelt project names, understated deeds, sparse Hyderabad deeds). The cleaning
pipeline and models only see these raw files, never the hidden generating parameters,
exactly as they would with scraped data. Every number the app shows is therefore labelled
as coming from a synthetic prototype dataset.

Builder and project names are fictional.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import geo

SEED = 20261122
N_MONTHS = 45  # Jan 2023 .. Sep 2026
MONTHS = pd.period_range("2023-01", periods=N_MONTHS, freq="M")
RERA_START = pd.Period("2017-05", "M")

BUILDER_FIRST = ["Nivora", "Tarang", "Akshara", "Meghana", "Pranava", "Shriyan", "Vedant", "Kaveri",
                 "Ishaan", "Anvaya", "Sukrut", "Devaki", "Raghav", "Tanvika", "Ojas", "Prithvi",
                 "Saket", "Varenya", "Miraya", "Hemadri", "Lavanya", "Charvi", "Aadhya", "Kesar",
                 "Rudransh", "Vihan", "Samhita", "Nakshatra", "Advait", "Ritam"]
BUILDER_SUFFIX = ["Developers", "Infra", "Constructions", "Homes", "Estates", "Realty", "Builders"]
PROJECT_SUFFIX = ["Heights", "Greens", "Meadows", "Residency", "Towers", "Enclave", "Vista", "Park",
                  "Crest", "Habitat", "Courtyard", "Skyline", "Gardens", "Signature", "Elite"]
ABBREV = {"Heights": "Hts", "Residency": "Resi", "Towers": "Twrs", "Enclave": "Encl", "Gardens": "Gdns",
          "Apartments": "Apts", "Signature": "Sig"}

GROWTH = {"Kokapet": 0.32, "Tellapur": 0.30, "Narsingi": 0.26, "Kondapur": 0.18, "Gachibowli": 0.16,
          "Nallagandla": 0.24, "Puppalaguda": 0.22, "Bachupally": 0.22, "Kompally": 0.18,
          "Shamshabad": 0.20, "Wagholi": 0.28, "Hinjewadi": 0.24, "Wakad": 0.20, "Kharadi": 0.22,
          "Undri": 0.18}


def city_index(city: str, m: np.ndarray) -> np.ndarray:
    """True (hidden) city log price index, 0 at the as-of month."""
    m = np.asarray(m, float)
    if city == "Hyderabad":
        # 8%/yr to Dec 2025, slowing to 3%/yr in 2026 (registrations fell in 2026)
        k = np.minimum(m, 36)
        val = 0.08 * k / 12 + 0.03 * np.maximum(m - 36, 0) / 12
        end = 0.08 * 36 / 12 + 0.03 * 8 / 12
    else:
        val = 0.055 * m / 12
        end = 0.055 * 44 / 12
    return val - end + 0.006 * np.sin(2 * np.pi * m / 12)


def _smooth_field(rng, lat, lon, n_bumps, scale_deg, centre_lat, centre_lon, spread):
    f = np.zeros_like(lat)
    for _ in range(n_bumps):
        bl = centre_lat + rng.normal(0, spread)
        bo = centre_lon + rng.normal(0, spread)
        amp = rng.normal(0, 1)
        f += amp * np.exp(-((lat - bl) ** 2 + (lon - bo) ** 2) / (2 * scale_deg ** 2))
    return f / max(np.std(f), 1e-6)


def build_cells(rng) -> pd.DataFrame:
    """Precomputed location-feature grid (the 'h3_cell' table)."""
    rows = []
    size = 0.0035
    for city in ("Hyderabad", "Pune"):
        locs = {k: v for k, v in geo.LOCALITIES.items() if v[0] == city}
        llat = np.array([v[1] for v in locs.values()])
        llon = np.array([v[2] for v in locs.values()])
        names = list(locs)
        la = np.arange(llat.min() - 0.04, llat.max() + 0.04, size)
        lo = np.arange(llon.min() - 0.04, llon.max() + 0.04, size)
        LA, LO = np.meshgrid(la + size / 2, lo + size / 2, indexing="ij")
        LA, LO = LA.ravel(), LO.ravel()
        d = geo.haversine_m(LA[:, None], LO[:, None], llat[None], llon[None])
        nearest = d.argmin(1)
        keep = d.min(1) < 4200
        LA, LO, nearest, dmin = LA[keep], LO[keep], nearest[keep], d.min(1)[keep]
        cl, co = llat.mean(), llon.mean()
        f_ndvi = _smooth_field(rng, LA, LO, 40, 0.02, cl, co, 0.08)
        f_poi = _smooth_field(rng, LA, LO, 40, 0.02, cl, co, 0.08)
        f_flood = _smooth_field(rng, LA, LO, 60, 0.012, cl, co, 0.09)
        f_light = _smooth_field(rng, LA, LO, 30, 0.03, cl, co, 0.08)
        lf = geo.location_features(city, LA, LO)
        for i in range(len(LA)):
            loc = names[nearest[i]]
            _, _, _, rate, gratio, fbase = geo.LOCALITIES[loc]
            centrality = np.exp(-dmin[i] / 2500)
            flood_score = fbase + 0.18 * f_flood[i]
            flood = 2 if flood_score > 0.45 else (1 if flood_score > 0.22 else 0)
            gr = GROWTH.get(loc, 0.08)
            rows.append({
                "cell_id": geo.cell_id(LA[i], LO[i]), "city": city, "micro_market": loc,
                "lat": round(LA[i], 5), "lon": round(LO[i], 5),
                "govt_rate_inr_sqft": float(round(rate * gratio * (1 + 0.04 * rng.normal()) / 50) * 50),
                "poi_density_1km": float(np.clip(40 + 25 * centrality + 10 * f_poi[i] + (15 if rate > 9000 else 0), 3, None)),
                "ndvi": float(np.clip(0.24 + 0.06 * f_ndvi[i] - 0.05 * centrality, 0.05, 0.6)),
                "night_light_level": float(np.clip(30 + 25 * centrality + 6 * f_light[i], 2, None)),
                "night_light_trend": float(np.clip(0.03 + gr * 0.25 + 0.01 * rng.normal(), -0.02, None)),
                "built_volume_growth": float(np.clip(gr + 0.05 * rng.normal(), 0.0, None)),
                "flood_hazard": int(flood),
                "dist_metro_m": float(lf["dist_metro_m"][i]),
                "dist_upcoming_metro_m": float(lf["dist_upcoming_metro_m"][i]),
                "metro_stage": int(lf["metro_stage"][i]),
                "dist_it_hub_m": float(lf["dist_it_hub_m"][i]),
            })
    cells = pd.DataFrame(rows).drop_duplicates("cell_id")
    cells["rent_psf_month"] = np.nan
    cells["real_features"] = False
    real = geo.REAL_DIR / "pune_osm_cells.csv"
    if geo.REAL_PUNE and real.exists():
        # Pune: replace generated location features with REAL OpenStreetMap / Makaan layers
        R = pd.read_csv(real).set_index("cell_id")
        m = (cells["city"] == "Pune").values
        ids = cells.loc[m, "cell_id"]
        hit = ids.isin(R.index).values
        idx = ids[hit]
        sub = cells.index[m][hit]
        cells.loc[cells.index[m], "poi_density_1km"] = 0.0        # no OSM activity recorded in the cell
        cells.loc[cells.index[m], "flood_hazard"] = 0
        cells.loc[sub, "poi_density_1km"] = R.loc[idx, "poi_all_1km"].values
        cells.loc[sub, "flood_hazard"] = R.loc[idx, "flood_proxy"].values.astype(int)
        cells.loc[sub, "rent_psf_month"] = R.loc[idx, "rent_psf_month_1500m"].values
        cells.loc[cells.index[m], "real_features"] = True
    return cells


def _fmt_price(inr: float, rng) -> str:
    style = rng.random()
    if inr >= 1e7 and style < 0.7:
        return f"₹{inr / 1e7:.2f} Cr"
    if style < 0.85:
        return f"₹{inr / 1e5:.1f} L" if rng.random() < 0.5 else f"{inr / 1e5:.2f} Lac"
    return f"Rs. {int(round(inr, -3)):,}"


def _noisy_name(name: str, builder_first: str, rng) -> str:
    r = rng.random()
    if r < 0.45:
        return name
    if r < 0.60:
        return name.lower()
    if r < 0.72:
        return name.replace(builder_first + " ", "")
    if r < 0.84:
        out = name
        for k, v in ABBREV.items():
            out = out.replace(k, v)
        return out
    if r < 0.94:  # single typo
        s = list(name)
        i = rng.integers(1, len(s) - 2)
        s[i], s[i + 1] = s[i + 1], s[i]
        return "".join(s)
    return ""  # missing project name


def generate(out_dir: Path) -> dict:
    rng = np.random.default_rng(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = build_cells(rng)
    cell_index = cells.set_index("cell_id")

    # ---------------- builders ----------------
    builders = []
    demo_b = {"Meghana": 0.93, "Kaveri": 0.30, "Akshara": 0.80, "Vedant": 0.92, "Ojas": 0.30, "Prithvi": 0.82}
    demo_city = {"Vedant": "Pune", "Ojas": "Pune", "Prithvi": "Pune"}
    others = [f for f in BUILDER_FIRST if f not in demo_b]
    rng.shuffle(others)
    firsts = list(demo_b) + others  # 30 builders: first 17 Hyderabad, 13 Pune
    for b, first in enumerate(firsts):
        nm = f"{first} {rng.choice(BUILDER_SUFFIX)}"
        rel = float(np.clip(rng.beta(5, 2), 0.15, 0.98))
        builders.append({"builder_id": b + 1, "name": nm, "_reliability": rel,
                         "_city": "Hyderabad" if b < 17 else "Pune"})
    # demo builders (fixed reliability)
    for bd in builders:
        f = bd["name"].split()[0]
        if f in demo_b:
            bd["_reliability"] = demo_b[f]
            bd["_city"] = demo_city.get(f, "Hyderabad")
    for f, rel in demo_b.items():
        if not any(bd["name"].startswith(f) for bd in builders):
            builders.append({"builder_id": len(builders) + 1, "name": f"{f} {'Developers' if f=='Meghana' else ('Infra' if f=='Kaveri' else 'Homes')}",
                             "_reliability": rel, "_city": "Hyderabad"})
    bdf = pd.DataFrame(builders)

    # ---------------- projects ----------------
    projects = []
    pid = 0
    for city, n_proj in (("Pune", 240), ("Hyderabad", 150)):   # Pune is the primary market
        locs = [k for k, v in geo.LOCALITIES.items() if v[0] == city]
        cb = bdf[bdf["_city"] == city]
        for _ in range(n_proj):
            pid += 1
            loc = locs[rng.integers(len(locs))]
            _, lat0, lon0, rate, _, _ = geo.LOCALITIES[loc]
            b = cb.iloc[rng.integers(len(cb))]
            projects.append(_make_project(rng, pid, city, loc, lat0 + rng.normal(0, 0.007),
                                          lon0 + rng.normal(0, 0.007), b, rate))
    # Demo projects with fixed, documented characteristics
    def bid(first):
        return bdf[bdf["name"].str.startswith(first)].iloc[0]
    demo_specs = [
        dict(name="Meghana Kondapur Heights", loc="Kondapur", lat=17.4555, lon=78.3765, b=bid("Meghana"),
             launch="2016-03", promised="2020-06", revised="2020-08", floors=14, units=420, oc=True,
             complaints=0, loading=1.32),
        dict(name="Kaveri Riverside Enclave", loc="Nagole", lat=17.3702, lon=78.5652, b=bid("Kaveri"),
             launch="2019-02", promised="2023-06", revised="2027-06", floors=12, units=560, oc=False,
             complaints=17, loading=1.38, flood=2),
        dict(name="Akshara Gachibowli Vista", loc="Gachibowli", lat=17.4385, lon=78.3505, b=bid("Akshara"),
             launch="2018-01", promised="2022-03", revised="2022-07", floors=20, units=680, oc=True,
             complaints=1, loading=1.30),
        # ---- Pune (primary market); coordinates are real OSM cells, see data/real/pune_osm_cells.csv
        dict(name="Vedant Baner Heights", loc="Baner", lat=18.55575, lon=73.78475, b=bid("Vedant"), city="Pune",
             launch="2017-04", promised="2021-06", revised="2021-09", floors=14, units=380, oc=True,
             complaints=0, loading=1.30),
        dict(name="Ojas Riverside Residency", loc="Kharadi", lat=18.54525, lon=73.94225, b=bid("Ojas"), city="Pune",
             launch="2019-05", promised="2023-03", revised="2027-03", floors=16, units=620, oc=False,
             complaints=15, loading=1.38),
        dict(name="Prithvi Wakad Vista", loc="Wakad", lat=18.59775, lon=73.76725, b=bid("Prithvi"), city="Pune",
             launch="2018-02", promised="2022-01", revised="2022-05", floors=18, units=540, oc=True,
             complaints=1, loading=1.31),
    ]
    for spec in demo_specs:
        pid += 1
        city = spec.get("city", "Hyderabad")
        p = _make_project(rng, pid, city, spec["loc"], spec["lat"], spec["lon"], spec["b"],
                          geo.LOCALITIES[spec["loc"]][3])
        p.update(project_name=spec["name"], launch_date=spec["launch"], promised_possession=spec["promised"],
                 revised_possession=spec["revised"], approved_floors=spec["floors"], total_units=spec["units"],
                 oc_received=spec["oc"], complaints=spec["complaints"], loading_factor=spec["loading"],
                 _effect=0.0, _demo=True)
        p["rera_id"] = f"{'P0240000' if city == 'Hyderabad' else 'P5210000'}{9000 + pid}"
        p["rera_registered"] = True
        p["approvals"] = "Building permission;Environmental clearance;Fire NOC" + (";Occupancy certificate" if spec["oc"] else "")
        if "flood" in spec:
            cid = geo.cell_id(spec["lat"], spec["lon"])
            if cid in cell_index.index:
                cell_index.loc[cid, "flood_hazard"] = spec["flood"]
        projects.append(p)
    cells = cell_index.reset_index()
    pdf = pd.DataFrame(projects)
    pdf["cell_id"] = [geo.cell_id(a, b) for a, b in zip(pdf.lat, pdf.lon)]
    pdf = pdf[pdf["cell_id"].isin(cells["cell_id"])].reset_index(drop=True)

    # ---------------- transactions (hidden truth) ----------------
    tx = _make_transactions(rng, pdf, bdf, cells.set_index("cell_id"))

    # ---------------- deeds ----------------
    is_pune = tx["city"] == "Pune"
    deed_p = np.where(is_pune, 0.75, 0.30)
    has_deed = rng.random(len(tx)) < deed_p
    d = tx[has_deed].copy()
    under = np.maximum(0, rng.normal(0.03, 0.03, len(d)))
    guidance_value = d["govt_rate"] * d["carpet_area"]
    consid = np.maximum(d["market_value"] * (1 - under), guidance_value * rng.uniform(1.0, 1.02, len(d)))
    deeds = pd.DataFrame({
        "deed_id": [f"{'IDX2' if c == 'Pune' else 'ECS'}-{i:06d}" for i, c in zip(range(len(d)), d["city"])],
        "state": d["city"].map(geo.CITY_STATE),
        "city": d["city"], "project_id": d["project_id"], "registered_on": d["month"].astype(str),
        "consideration_inr": np.round(consid, -3).astype("int64"),
        "carpet_area_sqft": d["carpet_area"].round(0),
        "bhk": d["bhk"], "floor": d["floor"],
        "lat": d["lat"], "lon": d["lon"],
        "party_names": "[redacted at ingestion]",
        "_unit_id": d["unit_id"], "_market_value": d["market_value"].round(0),
    })

    # ---------------- listings ----------------
    listings = _make_listings(rng, tx, pdf, bdf, cells.set_index("cell_id"))

    # builders public table: observed stats only
    bpub = bdf[["builder_id", "name"]].copy()

    pub_projects = pdf.drop(columns=[c for c in pdf.columns if c.startswith("_")])
    cells.to_csv(out_dir / "cells.csv", index=False)
    bpub.to_csv(out_dir / "builders.csv", index=False)
    pub_projects.to_csv(out_dir / "projects_rera.csv", index=False)
    deeds.to_csv(out_dir / "deeds.csv", index=False)
    listings.to_csv(out_dir / "listings.csv", index=False)
    # hidden truth kept separately for audit of the generator only (never read by the models)
    tx.to_csv(out_dir / "_hidden_transactions.csv", index=False)
    stats = {"cells": len(cells), "builders": len(bpub), "projects": len(pub_projects),
             "deeds": len(deeds), "listings": len(listings), "transactions": len(tx), "seed": SEED}
    (out_dir / "generation_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def _make_project(rng, pid, city, loc, lat, lon, b, rate):
    launch_year = int(rng.integers(2008, 2026))
    launch = pd.Period(f"{launch_year}-{rng.integers(1, 13):02d}", "M")
    promised = launch + int(rng.integers(36, 54))
    rel = b["_reliability"]
    delay = int(max(0, rng.normal((1 - rel) * 40, 6)))
    revised = promised + delay
    floors = int(np.clip(rng.normal(16 if rate > 8000 else 10, 5), 4, 40))
    first = b["name"].split()[0]
    name = f"{first} {loc.split()[0]} {rng.choice(PROJECT_SUFFIX)}"
    completed_by_asof = revised <= pd.Period(geo.AS_OF, "M")
    registered = not (revised < RERA_START)
    oc = bool(completed_by_asof and rng.random() < 0.55 + 0.45 * rel)
    approvals = ["Building permission"]
    if registered:
        if rng.random() < 0.9:
            approvals.append("Environmental clearance")
        if floors > 6 and rng.random() < 0.92:
            approvals.append("Fire NOC")
    if oc:
        approvals.append("Occupancy certificate")
    complaints = int(rng.poisson(max(delay, 0) / 4 + 0.3)) if registered else None
    return {
        "project_id": f"{'HYD' if city == 'Hyderabad' else 'PUN'}-{pid:04d}",
        "rera_id": (f"P0240000{1000 + pid}" if city == "Hyderabad" else f"P5210000{1000 + pid}") if registered else "",
        "rera_registered": registered,
        "builder_id": int(b["builder_id"]), "project_name": name, "city": city, "micro_market": loc,
        "lat": round(lat, 5), "lon": round(lon, 5),
        "total_units": int(rng.integers(80, 1400)), "approved_floors": floors,
        "launch_date": str(launch), "promised_possession": str(promised), "revised_possession": str(revised),
        "oc_received": oc, "complaints": complaints,
        "loading_factor": round(float(rng.uniform(1.24, 1.45)), 3),
        "approvals": ";".join(approvals) if registered else "",
        "amenities_score": round(float(np.clip(rng.normal(0.6, 0.15), 0.1, 1.0)), 2),
        "_effect": float(rng.normal(0, 0.045)), "_demo": False,
    }


AREA_BY_BHK = {1: (480, 620), 2: (880, 1220), 3: (1280, 1850), 4: (2000, 3000)}


def _make_transactions(rng, pdf, bdf, cells):
    rel_map = bdf.set_index("builder_id")["_reliability"]
    rows = []
    uid = 0
    for _, p in pdf.iterrows():
        n = int(rng.poisson(48 * (1.6 if p["_demo"] else 1.0)))
        launch = pd.Period(p["launch_date"], "M")
        done = pd.Period(p["revised_possession"], "M")
        c = cells.loc[p["cell_id"]]
        rel = rel_map[p["builder_id"]]
        rate = geo.LOCALITIES[p["micro_market"]][3]
        for _ in range(n):
            mi = int(rng.integers(0, N_MONTHS))
            month = MONTHS[mi]
            if month < launch:
                continue
            uid += 1
            bhk = int(rng.choice([1, 2, 3, 4], p=[0.12, 0.48, 0.33, 0.07]))
            area = float(rng.uniform(*AREA_BY_BHK[bhk]))
            floor = int(rng.integers(1, p["approved_floors"] + 1))
            parking = bool(rng.random() < (0.85 if bhk >= 2 else 0.45))
            east = bool(rng.random() < 0.35)
            uc = month < done
            age = 0.0 if uc else (month - done).n / 12
            delay_m = max(0, (pd.Period(p["revised_possession"], "M") - pd.Period(p["promised_possession"], "M")).n)
            lp = (np.log(rate) + city_index(p["city"], mi) + p["_effect"]
                  + 0.10 * np.exp(-c["dist_metro_m"] / 700) - 0.03 * np.clip((c["dist_metro_m"] - 1500) / 3000, 0, 1)
                  + {0: 0, 1: 0.3, 2: 0.6, 3: 1.0}[int(c["metro_stage"])] * 0.06 * np.exp(-c["dist_upcoming_metro_m"] / 1000)
                  + 0.06 * np.exp(-c["dist_it_hub_m"] / 3000)
                  - 0.011 * age + 0.004 * min(floor, 25) - 0.05 * np.log(area / 1000)
                  + 0.035 * parking + 0.015 * east + 0.15 * (rel - 0.7)
                  + (-0.05 if (not uc and not p["oc_received"]) else 0.0)
                  + ((-0.05 - 0.003 * delay_m) if uc else 0.0)
                  - 0.03 * c["flood_hazard"] + 0.08 * (c["ndvi"] - 0.25)
                  + 0.03 * (np.log1p(c["poi_density_1km"]) - 4.1)
                  + (0.08 * np.log(c["rent_psf_month"] / 21) if c["rent_psf_month"] == c["rent_psf_month"] else 0.0) + 0.10 * (c["built_volume_growth"] - 0.1)
                  + 0.04 * (p["amenities_score"] - 0.6)
                  + rng.normal(0, 0.032 + 0.035 * uc + 0.03 * (c["flood_hazard"] == 2)
                               + 0.025 * (rate > 12000) + 0.02 * (not p["rera_id"]) + 0.015 * (bhk == 4)))
            ppsf = float(np.exp(lp))
            rows.append({
                "unit_id": uid, "project_id": p["project_id"], "city": p["city"], "micro_market": p["micro_market"],
                "month": str(month), "month_idx": mi, "bhk": bhk, "carpet_area": round(area, 0), "floor": floor,
                "total_floors": int(p["approved_floors"]), "parking": parking, "east_facing": east,
                "sale_type": "new" if (uc or age < 1) else "resale", "age_years": round(age, 2),
                "lat": round(p["lat"] + rng.normal(0, 0.0004), 6), "lon": round(p["lon"] + rng.normal(0, 0.0004), 6),
                "govt_rate": float(c["govt_rate_inr_sqft"]), "true_ppsf": ppsf,
                "market_value": ppsf * area,
            })
    return pd.DataFrame(rows)


def _make_listings(rng, tx, pdf, bdf, cells):
    pmap = pdf.set_index("project_id")
    bname = bdf.set_index("builder_id")["name"]
    sources = ["portal_a", "portal_b", "portal_c"]
    rows = []
    lid = 0

    def one(t, month_idx, price):
        nonlocal lid
        lid += 1
        p = pmap.loc[t["project_id"]]
        first = bname[p["builder_id"]].split()[0]
        at = rng.choice(["carpet", "builtup", "superbuiltup"], p=[0.35, 0.30, 0.35])
        factor = {"carpet": 1.0, "builtup": 1.12, "superbuiltup": p["loading_factor"]}[at]
        area_rep = round(t["carpet_area"] * factor, 0)
        price_txt = _fmt_price(price, rng)
        lat, lon = t["lat"] + rng.normal(0, 0.0012), t["lon"] + rng.normal(0, 0.0012)
        if rng.random() < 0.08:  # geocoded to locality centroid only
            lat, lon = geo.LOCALITIES[t["micro_market"]][1:3]
        parking_txt = "covered car parking" if t["parking"] else "no dedicated parking"
        facing = "East" if t["east_facing"] else rng.choice(["North", "West", "South"])
        desc = (f"{t['bhk']} BHK {t['sale_type']} flat, {int(t['floor'])} of {int(t['total_floors'])} floor, "
                f"{facing.lower()} facing, {parking_txt}" + (", OC received" if p["oc_received"] else ""))
        age = t["age_years"] if rng.random() > 0.1 else np.nan
        return {
            "listing_id": f"L{lid:07d}", "source": sources[rng.integers(3)],
            "listed_on": str(MONTHS[month_idx]), "city": t["city"], "locality_text": t["micro_market"],
            "project_name_raw": _noisy_name(p["project_name"], first, rng),
            "lat": round(lat, 6), "lon": round(lon, 6), "price_text": price_txt,
            "area_value": area_rep, "area_type": at, "bhk": t["bhk"], "bathrooms": int(t["bhk"] + (rng.random() < 0.5)),
            "floor": int(t["floor"]), "total_floors": int(t["total_floors"]), "facing": facing,
            "sale_type": t["sale_type"], "age_years": age, "description": desc,
            "_unit_id": t["unit_id"],
        }

    for _, t in tx.iterrows():
        if rng.random() > 0.85:
            continue
        lag = int(rng.integers(0, 5))
        mi = max(0, t["month_idx"] - lag)
        hidx = geo_index_shift(t, mi)
        head = rng.normal(0.085, 0.035) + (0.03 if t["sale_type"] == "new" else 0.0)
        price = t["market_value"] * np.exp(hidx + head)
        r = one(t, mi, price)
        # typos
        u = rng.random()
        if u < 0.008:
            r["price_text"] = _fmt_price(price * 10, rng)
        elif u < 0.012:
            r["price_text"] = _fmt_price(price / 10, rng)
        elif u < 0.016:
            r["area_value"] = r["area_value"] * 10
        rows.append(r)
        if rng.random() < 0.15:  # cross-posted duplicate
            dup = dict(r)
            lid += 1
            dup["listing_id"] = f"L{lid:07d}"
            dup["source"] = [s for s in sources if s != r["source"]][rng.integers(2)]
            dup["listed_on"] = str(MONTHS[min(N_MONTHS - 1, mi + int(rng.integers(0, 3)))])
            dup["price_text"] = _fmt_price(price * (1 + rng.normal(0, 0.015)), rng)
            rows.append(dup)
    # unsold listings (no deed ever) - re-use unit templates from the same projects
    extra = tx.sample(frac=0.2, random_state=SEED)
    for _, t in extra.iterrows():
        mi = int(rng.integers(0, N_MONTHS))
        price = t["market_value"] * np.exp(geo_index_shift(t, mi) + rng.normal(0.12, 0.05))
        rows.append(one(t, mi, price) | {"_unit_id": -1})
    return pd.DataFrame(rows)


def geo_index_shift(t, mi):
    return city_index(t["city"], mi) - city_index(t["city"], t["month_idx"])


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(generate(root / "data" / "raw"))
