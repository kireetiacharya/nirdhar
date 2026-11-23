"""Clean and merge: raw records -> one trusted row per property sale (deck slide 7).

Steps: 1 normalise units, 2 resolve project, 3 geocode to cell, 4 deduplicate listings,
5 remove outliers, 6 time-adjust, 7 calibrate listings to deeds, 8 spatial join, 9 comps pool.
Columns starting with "_" are hidden generator truth; they are used ONLY to score the
pipeline in the data-quality report, never as model inputs.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import geo

ASOF = pd.Period(geo.AS_OF, "M")
START = pd.Period("2023-01", "M")
ABBREV_BACK = {"hts": "heights", "resi": "residency", "twrs": "towers", "encl": "enclave", "gdns": "gardens",
               "apts": "apartments", "sig": "signature"}


def month_idx(s) -> np.ndarray:
    return np.array([(pd.Period(x, "M") - START).n for x in s])


# ------------------------------------------------------------------ 1 normalise units
def parse_price(txt: str) -> float:
    t = str(txt).replace(",", "").strip()
    num = r"(\d+(?:\.\d+)?)"
    m = re.search(num + r"\s*(cr|crore)", t, re.I)
    if m:
        return float(m.group(1)) * 1e7
    m = re.search(num + r"\s*(l|lac|lakh|lakhs)\b", t, re.I)
    if m:
        return float(m.group(1)) * 1e5
    m = re.search(num, t)
    return float(m.group(1)) if m else np.nan


def parse_parking(desc: str) -> float:
    d = str(desc).lower()
    if "no dedicated parking" in d or "no parking" in d:
        return 0.0
    if "parking" in d:
        return 1.0
    return np.nan


# ------------------------------------------------------------------ 2 resolve project
def _norm_name(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    return " ".join(ABBREV_BACK.get(w, w) for w in s.split())


def _trigrams(s: str) -> set:
    s = f"  {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def resolve_projects(lst: pd.DataFrame, proj: pd.DataFrame) -> pd.DataFrame:
    """Fuzzy name + geo match of each listing to a RERA/project record."""
    out_pid = np.full(len(lst), "", dtype=object)
    out_score = np.zeros(len(lst))
    ptri = {r.project_id: _trigrams(_norm_name(r.project_name)) for r in proj.itertuples()}
    for city in lst["city"].unique():
        li = np.where(lst["city"].values == city)[0]
        pc = proj[proj["city"] == city].reset_index(drop=True)
        d = geo.haversine_m(lst["lat"].values[li, None], lst["lon"].values[li, None],
                            pc["lat"].values[None], pc["lon"].values[None])
        names = lst["project_name_raw"].fillna("").values[li]
        cache = {}
        for k, i in enumerate(li):
            nm = names[k]
            if nm not in cache:
                t = _trigrams(_norm_name(nm)) if nm else set()
                cache[nm] = np.array([len(t & ptri[p]) / max(1, len(t | ptri[p])) if t else 0.0
                                      for p in pc["project_id"]])
            sim = cache[nm]
            geo_s = np.exp(-d[k] / 800)
            floors_ok = (pc["approved_floors"].values == lst["total_floors"].values[i]).astype(float)
            if nm:
                score = 0.65 * sim + 0.25 * geo_s + 0.10 * floors_ok
                thr = 0.42
            else:  # no name: geo + building height only
                score = 0.6 * geo_s + 0.4 * floors_ok
                thr = 0.80
            j = int(score.argmax())
            if score[j] >= thr and d[k, j] < 3000:
                out_pid[i] = pc["project_id"].iloc[j]
                out_score[i] = score[j]
    lst = lst.copy()
    lst["project_id"] = out_pid
    lst["match_score"] = out_score
    return lst


# ------------------------------------------------------------------ builder / project tables
def build_project_tables(proj: pd.DataFrame, builders: pd.DataFrame):
    p = proj.copy()
    prom = p["promised_possession"].map(lambda x: pd.Period(x, "M"))
    rev = p["revised_possession"].map(lambda x: pd.Period(x, "M"))
    p["delay_months"] = [(r - q).n for r, q in zip(rev, prom)]
    p["completed"] = [r <= ASOF for r in rev]
    p["possession_idx"] = [(r - START).n for r in rev]
    agg = p.groupby("builder_id").agg(
        projects_count=("project_id", "count"),
        avg_delay_months=("delay_months", "mean"),
        oc_rate=("oc_received", lambda s: float(np.mean(s[p.loc[s.index, "completed"]])) if p.loc[s.index, "completed"].any() else np.nan),
        complaints_per_project=("complaints", "mean"),
    ).reset_index()
    agg["reliability_score"] = (0.7 * np.clip(1 - agg["avg_delay_months"] / 40, 0, 1)
                                + 0.3 * agg["oc_rate"].fillna(0.6)).round(3)
    b = builders.merge(agg, on="builder_id", how="left")
    p = p.merge(b[["builder_id", "name", "reliability_score", "avg_delay_months"]].rename(
        columns={"name": "builder_name", "reliability_score": "builder_reliability",
                 "avg_delay_months": "builder_avg_delay"}), on="builder_id", how="left")
    return p, b


# ------------------------------------------------------------------ 6 time index
def time_index(lst: pd.DataFrame) -> dict:
    """Hedonic repeat-project index per city from listings (two-way fixed effects)."""
    out = {}
    for city, g in lst.groupby("city"):
        y = np.log(g["ask_ppsf"].values)
        proj = g["project_id"].values
        m = g["month_idx"].values
        tm = np.zeros(45)
        resid = y.copy()
        for _ in range(15):
            pe = pd.Series(y - tm[m]).groupby(proj).transform("mean").values
            r = y - pe
            tm = pd.Series(r).groupby(m).mean().reindex(range(45)).interpolate().bfill().ffill().values
        tm = pd.Series(tm).rolling(3, center=True, min_periods=1).mean().values
        out[city] = (tm - tm[44]).tolist()
    return out


# ------------------------------------------------------------------ 7 listing -> deed calibration
CAL_FEATS = ["is_new", "age_f", "band", "ask_to_govt"]


def _cal_X(df):
    return np.column_stack([np.ones(len(df))] + [df[c].values for c in CAL_FEATS])


def fit_deed_calibration(pairs: pd.DataFrame, lam=1.0):
    X = _cal_X(pairs)
    y = pairs["r"].values
    A = X.T @ X + lam * np.diag([0] + [1] * (X.shape[1] - 1))
    return np.linalg.solve(A, X.T @ y)


def run(raw_dir: Path) -> dict:
    report = {}
    cells = pd.read_csv(raw_dir / "cells.csv")
    builders = pd.read_csv(raw_dir / "builders.csv")
    proj_raw = pd.read_csv(raw_dir / "projects_rera.csv", keep_default_na=False, na_values=[""])
    proj_raw["rera_id"] = proj_raw["rera_id"].fillna("")
    proj_raw["approvals"] = proj_raw["approvals"].fillna("")
    lst = pd.read_csv(raw_dir / "listings.csv")
    deeds = pd.read_csv(raw_dir / "deeds.csv")
    report["input"] = {"listings": len(lst), "deeds": len(deeds), "projects": len(proj_raw), "cells": len(cells)}

    proj, bld = build_project_tables(proj_raw, builders)
    pidx = proj.set_index("project_id")

    # 1 normalise
    lst["ask_inr"] = lst["price_text"].map(parse_price)
    lst["parking"] = lst["description"].map(parse_parking)
    lst["east_facing"] = (lst["facing"] == "East").astype(float)
    lst["month_idx"] = month_idx(lst["listed_on"])

    # 2 resolve project
    lst = resolve_projects(lst, proj)
    hidden = pd.read_csv(raw_dir / "_hidden_transactions.csv", usecols=["unit_id", "project_id"])
    truth = lst["_unit_id"].map(hidden.set_index("unit_id")["project_id"])
    ok = truth.notna()
    report["project_resolution"] = {
        "matched_share": round(float((lst["project_id"] != "").mean()), 4),
        "precision_vs_hidden_truth": round(float((lst.loc[ok & (lst.project_id != ""), "project_id"] == truth[ok & (lst.project_id != "")]).mean()), 4),
    }
    lst = lst[lst["project_id"] != ""].copy()

    # carpet normalisation using each project's RERA loading factor
    lf = lst["project_id"].map(pidx["loading_factor"])
    lst["carpet_area"] = np.where(lst["area_type"] == "carpet", lst["area_value"],
                                  np.where(lst["area_type"] == "builtup", lst["area_value"] / 1.12,
                                           lst["area_value"] / lf)).round(0)
    lst["ask_ppsf"] = lst["ask_inr"] / lst["carpet_area"]
    lst["ask_ppsf_reported"] = lst["ask_inr"] / lst["area_value"]

    # 3 geocode: resolved project coordinates are more reliable than portal pins
    lst["lat"] = lst["project_id"].map(pidx["lat"])
    lst["lon"] = lst["project_id"].map(pidx["lon"])

    # 4 deduplicate: same project/bhk/floor/area within 60 days across portals
    lst = lst.sort_values(["project_id", "bhk", "floor", "month_idx"])
    lst["area_key"] = (lst["carpet_area"] / 30).round()
    grp = lst.groupby(["project_id", "bhk", "floor", "area_key"])
    lst["gap"] = grp["month_idx"].diff()
    lst["new_cluster"] = lst["gap"].isna() | (lst["gap"] > 2)
    lst["dup_cluster_id"] = lst["new_cluster"].cumsum()
    before = len(lst)
    is_dup = lst.duplicated("dup_cluster_id")
    true_dup = lst["_unit_id"].gt(0) & lst.duplicated(["_unit_id"])
    report["dedup"] = {"removed": int(is_dup.sum()), "true_duplicates": int(true_dup.sum()),
                       "recall": round(float((is_dup & true_dup).sum() / max(1, true_dup.sum())), 4)}
    lst = lst[~is_dup].copy()

    # 5 robust outliers per micro-market
    lst["micro_market"] = lst["project_id"].map(pidx["micro_market"])
    lp = np.log(lst["ask_ppsf"])
    med = lp.groupby(lst["micro_market"]).transform("median")
    mad = (lp - med).abs().groupby(lst["micro_market"]).transform("median") * 1.4826
    z = (lp - med) / mad
    out = z.abs() > 3.5
    report["outliers"] = {"removed": int(out.sum()), "share": round(float(out.mean()), 4)}
    lst = lst[~out].copy()

    # 6 time index
    tidx = time_index(lst)
    report["time_index_yoy"] = {c: round(float(np.exp(v[44] - v[32]) - 1), 4) for c, v in tidx.items()}

    # project-derived fields for both sources
    def add_project(df, mcol):
        df["total_floors"] = df["project_id"].map(pidx["approved_floors"])
        pos = df["project_id"].map(pidx["possession_idx"]).values
        df["under_construction"] = (df[mcol].values < pos).astype(float)
        derived_age = np.maximum(0, (df[mcol].values - pos) / 12)
        if "age_years" in df:
            df["age_years"] = df["age_years"].fillna(pd.Series(derived_age, index=df.index))
        else:
            df["age_years"] = derived_age
        df["sale_new"] = ((df["under_construction"] == 1) | (df["age_years"] < 1)).astype(float)
        return df

    lst = add_project(lst, "month_idx")
    deeds["month_idx"] = month_idx(deeds["registered_on"])
    deeds["micro_market"] = deeds["project_id"].map(pidx["micro_market"])
    deeds = add_project(deeds, "month_idx")
    deeds = deeds.rename(columns={"carpet_area_sqft": "carpet_area"})
    deeds["lat"] = deeds["project_id"].map(pidx["lat"])
    deeds["lon"] = deeds["project_id"].map(pidx["lon"])

    # 8 spatial join (cell features)
    cidx = cells.set_index("cell_id")
    cell_cols = ["govt_rate_inr_sqft", "poi_density_1km", "ndvi", "night_light_level", "night_light_trend",
                 "built_volume_growth", "flood_hazard", "dist_metro_m", "dist_upcoming_metro_m", "metro_stage",
                 "dist_it_hub_m", "rent_psf_month"]
    for df in (lst, deeds):
        cid = [geo.cell_id(a, b) for a, b in zip(df["lat"], df["lon"])]
        df["cell_id"] = cid
        for c in cell_cols:
            df[c] = cidx.loc[cid, c].values

    # 7 listing -> deed calibration: match listings to later deeds of the same unit
    lst["govt_value"] = lst["govt_rate_inr_sqft"] * lst["carpet_area"]
    lmed = lst.groupby("micro_market")["ask_ppsf"].transform("median")
    lst["band"] = np.log(lst["ask_ppsf"] / lmed)
    lst["ask_to_govt"] = np.log(lst["ask_ppsf"] / lst["govt_rate_inr_sqft"])
    lst["is_new"] = lst["sale_new"]
    lst["age_f"] = np.minimum(lst["age_years"], 20) / 10
    m = lst.reset_index().merge(deeds.reset_index(), on=["project_id", "bhk", "floor"], suffixes=("_l", "_d"))
    m = m[(m["month_idx_d"] >= m["month_idx_l"]) & (m["month_idx_d"] <= m["month_idx_l"] + 6)
          & ((m["carpet_area_d"] / m["carpet_area_l"] - 1).abs() < 0.03)]
    m = m.sort_values("month_idx_d").drop_duplicates("index_l").drop_duplicates("index_d")
    tix = {c: np.array(v) for c, v in tidx.items()}
    shift = np.array([tix[c][a] - tix[c][b] for c, a, b in zip(m["city_l"], m["month_idx_d"], m["month_idx_l"])])
    m["r"] = np.log(m["consideration_inr"] / m["ask_inr"]) - shift
    pairs = m.rename(columns={c + "_l": c for c in CAL_FEATS})
    pune = pairs[pairs["city_l"] == "Pune"]
    rng = np.random.default_rng(0)
    msk = rng.random(len(pune)) < 0.8
    beta = fit_deed_calibration(pune[msk])
    pred_hold = _cal_X(pune[~msk]) @ beta
    hyd = pairs[pairs["city_l"] == "Hyderabad"]
    pred_hyd = _cal_X(hyd) @ beta
    report["deed_calibration"] = {
        "pune_pairs": int(len(pune)), "coef": dict(zip(["intercept"] + CAL_FEATS, np.round(beta, 4).tolist())),
        "mean_listing_to_deed_gap": round(float(np.exp(pune["r"].mean()) - 1), 4),
        "pune_holdout_mae_pct": round(float(np.mean(np.abs(np.exp(pune[~msk]["r"] - pred_hold) - 1)) * 100), 2),
        "hyderabad_transfer_mae_pct": round(float(np.mean(np.abs(np.exp(hyd["r"] - pred_hyd) - 1)) * 100), 2),
        "hyderabad_pairs_checked": int(len(hyd)),
    }
    beta = fit_deed_calibration(pune)  # final fit on all Pune pairs
    lst["deed_discount"] = _cal_X(lst) @ beta
    lst["label_value"] = lst["ask_inr"] * np.exp(lst["deed_discount"])

    # 9 modelling table: deeds (true labels) + calibrated listings (minus those already matched)
    matched_l = set(m["index_l"])
    L = lst[~lst.index.isin(matched_l)].copy()
    L["label_source"] = "listing_calibrated"
    L["row_ref"] = L["listing_id"]
    D = deeds.copy()
    D["label_source"] = "deed"
    D["label_value"] = D["consideration_inr"]
    D["row_ref"] = D["deed_id"]
    D["parking"] = np.nan
    D["east_facing"] = np.nan
    D["ask_inr"] = np.nan
    D["ask_ppsf_reported"] = np.nan
    D["area_value"] = D["carpet_area"]
    keep = ["row_ref", "label_source", "city", "micro_market", "project_id", "month_idx", "lat", "lon", "cell_id",
            "bhk", "carpet_area", "area_value", "floor", "total_floors", "age_years", "under_construction",
            "sale_new", "parking", "east_facing", "label_value", "ask_inr", "ask_ppsf_reported"] + cell_cols
    sales = pd.concat([L[keep], D[keep]], ignore_index=True)
    sales["label_ppsf"] = sales["label_value"] / sales["carpet_area"]
    tshift = np.array([tix[c][mm] for c, mm in zip(sales["city"], sales["month_idx"])])
    sales["y"] = np.log(sales["label_ppsf"]) - tshift          # log ppsf, time-adjusted to as-of
    sales["month"] = [str(START + int(k)) for k in sales["month_idx"]]
    report["output"] = {"sales_rows": len(sales), "deeds": int((sales.label_source == "deed").sum()),
                        "calibrated_listings": int((sales.label_source != "deed").sum()),
                        "hyderabad_deeds": int(((sales.label_source == "deed") & (sales.city == "Hyderabad")).sum())}

    # raw (uncleaned-label) listings kept for the portal-style baseline in the ablation
    raw_listings = lst[["city", "micro_market", "project_id", "month_idx", "lat", "lon", "bhk", "area_value",
                        "carpet_area", "floor", "total_floors", "age_years", "ask_inr", "ask_ppsf_reported",
                        "sale_new", "under_construction", "parking", "east_facing"] + cell_cols].copy()
    return {"sales": sales, "projects": proj, "builders": bld, "cells": cells, "time_index": tidx,
            "raw_listings": raw_listings, "listings_clean": lst, "report": report}
