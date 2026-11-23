"""Train, validate and package the Nirdhar engine.

Time-based protocol (deck slide 10):
  train  <= Jun 2025 | val Jul-Dec 2025 (stacker, comps tuning, sigma model)
  cal    Jan-Mar 2026 registered deeds (conformal calibration)
  test   Apr-Sep 2026 registered deeds (final score; Pune = headline, primary market)
Then production models are refit on every row except the calibration split, and conformal
scores are recomputed with the production models so the interval stays valid.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import geo, models as M, pipeline

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ART = ROOT / "artifacts"
MODEL_VERSION = "nirdhar-0.1.0-proto"
SPLITS = {"train": (0, 29), "val": (30, 35), "cal": (36, 38), "test": (39, 44)}
GRADE_CUTS = (0.10, 0.15, 0.20)  # 90% range within +-10% (the PE10 standard) = A, +-15% = B, +-20% = C, else D


def grade_of(hw):
    a, b, c = GRADE_CUTS
    return np.where(hw <= a, "A", np.where(hw <= b, "B", np.where(hw <= c, "C", "D")))


def split(df, name):
    lo, hi = SPLITS[name]
    return df[(df["month_idx"] >= lo) & (df["month_idx"] <= hi)]


def snapshot_hash():
    h = hashlib.sha256()
    for f in sorted(RAW.glob("*.csv")):
        if f.name.startswith("_"):
            continue
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def ood_stats(train):
    st = {}
    for loc, g in train.groupby("micro_market"):
        st[loc] = {c: (float(g[c].median()), float((g[c] - g[c].median()).abs().median() * 1.4826 + 1e-6))
                   for c in ("log_area", "floor_c", "age_years", "total_floors")}
    return st


def ood_score(df, st, nearest_m):
    zmax = np.zeros(len(df))
    for c in ("log_area", "floor_c", "age_years", "total_floors"):
        med = df["micro_market"].map(lambda l: st.get(l, {}).get(c, (np.nan, 1))[0]).values
        mad = df["micro_market"].map(lambda l: st.get(l, {}).get(c, (np.nan, 1))[1]).values
        z = np.abs(df[c].values - med) / np.maximum(mad, {"log_area": 0.12, "floor_c": 3, "age_years": 2,
                                                          "total_floors": 3}[c])
        zmax = np.maximum(zmax, np.nan_to_num(z, nan=6.0))
    near = np.nan_to_num(np.asarray(nearest_m, float), nan=1e5, posinf=1e5) / 1000
    return np.maximum(0, zmax - 3) / 2 + np.maximum(0, near - 1.5), zmax


class Engine:
    """Everything needed to value one property; pickled as the deployable artifact."""

    def predict_frame(self, df, before=False, cache=None, with_detail=False):
        m1 = self.m1.predict(df)
        m3 = self.m3.predict(df)
        c = self.comps.predict(df, before=before, cache=cache)
        P = np.column_stack([m1, c["pred"].values, m3])
        bucket = M.density_bucket(c["n1km"].values)
        yhat = M.stack(P, bucket, self.W)
        ood, zmax = ood_score(df, self.ood, c["nearest"].values)
        F = M.sigma_features(P, c["disp"].values, c["n1km"].values, df)
        sigma = M.sigma_of(F, self.sig, ood)
        return dict(P=P, yhat=yhat, sigma=sigma, bucket=bucket, ood=ood, zmax=zmax, comps=c)


def predict_P(m1, m3, comps, df, before=True):
    c = comps.predict(df, before=before)
    return np.column_stack([m1.predict(df), c["pred"].values, m3.predict(df)]), c


def main():
    t0 = time.time()
    ART.mkdir(exist_ok=True)
    data = pipeline.run(RAW)
    proj = data["projects"]
    sales = M.enrich(data["sales"], proj)
    w_all = np.where(sales["label_source"] == "deed", 1.5, 1.0)
    sales["w"] = w_all
    tr, va = split(sales, "train"), split(sales, "val")
    cal = split(sales, "cal").query("label_source == 'deed'")
    te = split(sales, "test").query("label_source == 'deed'")
    log = {"rows": {k: int(len(v)) for k, v in dict(train=tr, val=va, cal=cal, test=te).items()}}
    print("rows", log["rows"])

    # ---------------- honest stage: fit on train ----------------
    m3 = M.M3().fit(tr, tr["y"].values, tr["w"].values)
    comps = M.Comps(sales, m3.beta_raw)
    vq = va.sample(min(700, len(va)), random_state=1)
    l = comps.tune(vq, n_iter=40)
    print("comps tuned, val median abs log err", round(l, 4), {k: round(v, 3) for k, v in comps.params.items()})
    feats_all = M.F_BASE + M.F_LOC + M.F_PROJ
    m1 = M.M1(feats_all).fit(tr, tr["y"].values, tr["w"].values)

    Pv, cv = predict_P(m1, m3, comps, va)
    bv = M.density_bucket(cv["n1km"].values)
    W = M.fit_stacker(Pv, va["y"].values, bv)
    yv = M.stack(Pv, bv, W)
    ost = ood_stats(tr)
    oodv, _ = ood_score(va, ost, cv["nearest"].values)
    Fv = M.sigma_features(Pv, cv["disp"].values, cv["n1km"].values, va)
    sig = M.fit_sigma(Fv, (va["y"].values - yv) / (1 + 0.5 * oodv))
    print("stacker", {k: v.round(2).tolist() for k, v in W.items()}, "sigma coef", sig.round(3))

    eng = Engine()
    eng.m1, eng.m3, eng.comps, eng.W, eng.sig, eng.ood = m1, m3, comps, W, sig, ost

    def conformal_scores(e, calset):
        r = e.predict_frame(calset, before=True)
        return np.abs(calset["y"].values - r["yhat"]) / r["sigma"]

    def intervals(e, df, cal_scores, calset, r):
        lo, hi, neff, thin = [], [], [], []
        for i in range(len(df)):
            same = calset["city"].values == df["city"].values[i]
            q, n_eff, th = M.weighted_conformal_q(calset["lat"].values[same], calset["lon"].values[same],
                                                  cal_scores[same], df["lat"].values[i], df["lon"].values[i])
            lo.append(r["yhat"][i] - q * r["sigma"][i])
            hi.append(r["yhat"][i] + q * r["sigma"][i])
            neff.append(n_eff)
            thin.append(th)
        return np.array(lo), np.array(hi), np.array(neff), np.array(thin)

    cs = conformal_scores(eng, cal)
    rt = eng.predict_frame(te, before=True)
    lo, hi, neff, thin = intervals(eng, te, cs, cal, rt)
    tshift = np.array([data["time_index"][c][m] for c, m in zip(te["city"], te["month_idx"])])
    area = te["carpet_area"].values
    true_v = te["label_value"].values
    pred_v = np.exp(rt["yhat"] + tshift) * area
    lo_v, hi_v = np.exp(lo + tshift) * area, np.exp(hi + tshift) * area
    hw = (hi_v - lo_v) / (2 * pred_v)
    grades = grade_of(hw)
    covered = (true_v >= lo_v) & (true_v <= hi_v)
    results = {}
    for city in ("Pune", "Hyderabad"):
        mk = te["city"].values == city
        res = M.metrics(pred_v[mk], true_v[mk])
        res["coverage_90"] = round(float(covered[mk].mean() * 100), 1)
        res["median_halfwidth_pct"] = round(float(np.median(hw[mk]) * 100), 1)
        res["hit_rate_AB"] = round(float(np.isin(grades[mk], ["A", "B"]).mean() * 100), 1)
        res["by_grade"] = {}
        for g in "ABCD":
            mg = mk & (grades == g)
            if mg.sum() >= 5:
                res["by_grade"][g] = {"share": round(float(mg.sum() / mk.sum() * 100), 1),
                                      "MdAPE": round(float(np.median(np.abs(pred_v[mg] / true_v[mg] - 1)) * 100), 2),
                                      "coverage_90": round(float(covered[mg].mean() * 100), 1), "n": int(mg.sum())}
        # per micro-market coverage (min / max over markets with >= 15 test deeds)
        cov_mm = pd.Series(covered[mk]).groupby(te["micro_market"].values[mk]).agg(["mean", "count"])
        cov_mm = cov_mm[cov_mm["count"] >= 15]
        res["coverage_by_micro_market"] = {k: round(float(v * 100), 1) for k, v in cov_mm["mean"].items()}
        results[city] = res
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "coverage_by_micro_market"} for k, v in results.items()}, indent=1))

    # ---------------- baselines + ablation ladder (Pune registered deeds: primary market) ----------------
    PRIMARY = "Pune"
    hy = te["city"].values == PRIMARY
    teh = te[hy]
    t_true = true_v[hy]
    t_shift = tshift[hy]
    t_area = area[hy]
    ladder = {}
    raw = data["raw_listings"].copy()
    raw = raw.assign(under_construction=raw["under_construction"].fillna(0))
    raw = M.enrich(raw.assign(label_source="listing", row_ref=""), proj)
    rtr = split(raw, "train")
    tix = data["time_index"]
    rs = np.array([tix[c][m] for c, m in zip(rtr["city"], rtr["month_idx"])])

    # baselines
    ladder["Government rate × area"] = M.metrics(teh["govt_rate_inr_sqft"].values * t_area, t_true)
    recent = raw[(raw["month_idx"] >= 27) & (raw["month_idx"] <= 38)]
    loc_med = recent.groupby("micro_market")["ask_ppsf_reported"].median()
    ladder["Locality median ₹/sq ft × area"] = M.metrics(teh["micro_market"].map(loc_med).values * t_area, t_true)

    base_feats = ["loc_code", "city_code", "bhk", "log_area", "floor_c", "total_floors", "age_years"]
    # L0 portal-style random forest: asking prices on reported (mixed) area
    rtr0 = rtr.assign(log_area=np.log(rtr["area_value"]))
    y0 = np.log(rtr0["ask_ppsf_reported"].values) - rs
    rf = M.M1(base_feats).fit(rtr0, y0, rf=True)
    ladder["Portal-style RF (asking prices)"] = M.metrics(np.exp(rf.predict(teh) + t_shift) * t_area, t_true)

    # L1 + deed-calibrated labels, still on reported area
    tr1 = tr.assign(log_area=np.log(tr["area_value"]))
    y1 = np.log(tr["label_value"].values / tr["area_value"].values) - np.array(
        [tix[c][m] for c, m in zip(tr["city"], tr["month_idx"])])
    l1 = M.M1(base_feats).fit(tr1, y1, rf=True)
    ladder["+ Deed-calibrated labels"] = M.metrics(np.exp(l1.predict(teh) + t_shift) * t_area, t_true)

    # L2 + carpet basis (gradient boosting on clean carpet labels)
    l2 = M.M1(M.F_BASE).fit(tr, tr["y"].values, tr["w"].values)
    ladder["+ Carpet-area basis"] = M.metrics(np.exp(l2.predict(teh) + t_shift) * t_area, t_true)

    # L3 + learned comps (no project identity in comps)
    comps_np = M.Comps(sales, m3.beta_raw, params={**comps.params, "g": 0.0})
    cv_np = comps_np.predict(va)
    ct_np = comps_np.predict(teh)
    def two_stack(m_va, m_te, cva, cte):
        Pv2 = np.column_stack([m_va, cva["pred"].values])
        Pt2 = np.column_stack([m_te, cte["pred"].values])
        W2 = M.fit_stacker(Pv2, va["y"].values, M.density_bucket(cva["n1km"].values))
        return M.stack(Pt2, M.density_bucket(cte["n1km"].values), W2)
    y3 = two_stack(l2.predict(va), l2.predict(teh), cv_np, ct_np)
    ladder["+ Learned comparables"] = M.metrics(np.exp(y3 + t_shift) * t_area, t_true)

    # L4 + satellite / infrastructure / location features
    l4 = M.M1(M.F_BASE + M.F_LOC).fit(tr, tr["y"].values, tr["w"].values)
    y4 = two_stack(l4.predict(va), l4.predict(teh), cv_np, ct_np)
    ladder["+ Satellite & infrastructure"] = M.metrics(np.exp(y4 + t_shift) * t_area, t_true)

    # L5 + project identity = Nirdhar
    ladder["+ Project identity = Nirdhar"] = M.metrics(pred_v[hy], t_true)
    for k, v in ladder.items():
        print(f"{k:38s} MdAPE {v['MdAPE']:6.2f}  PE10 {v['PE10']:5.1f}  PE20 {v['PE20']:5.1f}  COD {v['COD']:5.2f}  PRD {v['PRD']:.3f}")

    # ---------------- production refit (all rows except calibration split) ----------------
    fit_rows = sales[~sales.index.isin(split(sales, "cal").index)]
    m3p = M.M3().fit(fit_rows, fit_rows["y"].values, fit_rows["w"].values)
    m1p = M.M1(feats_all).fit(fit_rows, fit_rows["y"].values, fit_rows["w"].values)
    compsp = M.Comps(sales, m3p.beta_raw, params=comps.params)
    prod = Engine()
    prod.m1, prod.m3, prod.comps, prod.W, prod.sig, prod.ood = m1p, m3p, compsp, W, sig, ost
    cs_p = conformal_scores(prod, cal)
    prod.cal = dict(lat=cal["lat"].values, lon=cal["lon"].values, city=cal["city"].values, score=cs_p)

    # reference ("typical flat") rows per micro-market for explanations
    refs = {}
    cells = data["cells"]
    for loc, g in sales.groupby("micro_market"):
        r = {}
        for c in g.columns:
            if g[c].dtype.kind in "fi":
                r[c] = float(g[c].median())
        r.update(micro_market=loc, city=g["city"].iloc[0], project_id="", month_idx=44, under_construction=0.0,
                 sale_new=0.0, parking=1.0, east_facing=0.0, label_source="ref", row_ref="ref")
        for c in M.PROJ_COLS:
            col = c if c != "delay_months" else "project_delay_months"
            r[col] = float(g[col].median())
        refs[loc] = r

    prod.meta = {
        "model_version": MODEL_VERSION,
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "data_snapshot_hash": snapshot_hash(),
        "as_of": geo.AS_OF,
        "splits": {k: f"{pipeline.START + v[0]} to {pipeline.START + v[1]}" for k, v in SPLITS.items()},
        "rows": log["rows"],
        "comps_params": {k: round(v, 4) for k, v in comps.params.items()},
        "stacker_weights": {["low", "mid", "high"][k]: dict(zip(["M1 GBM", "M2 comps", "M3 hedonic"], v.round(2).tolist())) for k, v in W.items()},
        "m3_coefficients": {k: round(float(v), 5) for k, v in m3p.beta_raw.items()},
        "grade_cuts": GRADE_CUTS,
        "results": results,
        "ladder": ladder,
        "ladder_city": PRIMARY,
        "data_provenance": geo.DATA_PROVENANCE,
        "pipeline_report": data["report"],
        "data_is_synthetic": True,
        "train_seconds": round(time.time() - t0, 1),
    }
    prod.refs = refs
    prod.projects = proj
    prod.builders = data["builders"]
    prod.cells = cells
    prod.time_index = data["time_index"]
    prod.listings = data["listings_clean"][["listing_id", "project_id", "micro_market", "city", "month_idx", "ask_inr",
                                            "carpet_area", "ask_ppsf", "bhk", "floor", "source"]].copy()
    prod.grade_cuts = GRADE_CUTS
    with open(ART / "engine.pkl", "wb") as f:
        pickle.dump(prod, f)
    (ART / "model_card.json").write_text(json.dumps(prod.meta, indent=2, default=str))
    print("saved", ART / "engine.pkl", f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    # run via the package module so pickled classes resolve to nirdhar.train.Engine
    from nirdhar import train as _train
    _train.main()
