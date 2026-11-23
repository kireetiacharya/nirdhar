"""Nirdhar model layer (deck slide 8).

M1  LightGBM with monotonic constraints                     -> log price/sqft (carpet, as-of)
M2  Comparable-sales attention: softmax over the 30 best comps with valuer-style hedonic
    adjustments; attention parameters are tuned on a validation split
M3  Hierarchical hedonic: city -> micro-market -> project random effects (ridge-shrunk)
Stacker  convex weights per local-data-density bucket
Conformal  spatially weighted split-conformal on normalised residuals (Hjort et al.)
"""
from __future__ import annotations

import itertools

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import geo

LOCS = list(geo.LOCALITIES)
LOC_CODE = {k: i for i, k in enumerate(LOCS)}

F_BASE = ["loc_code", "city_code", "bhk", "log_area", "floor_c", "total_floors", "age_years",
          "parking", "east_facing", "sale_new", "under_construction"]
F_LOC = ["lat", "lon", "log_dist_metro", "log_dist_upcoming", "metro_stage", "log_dist_it", "poi_density_1km",
         "ndvi", "night_light_level", "night_light_trend", "built_volume_growth", "flood_hazard", "log_govt_rate",
         "log_rent"]
F_PROJ = ["builder_reliability", "builder_avg_delay", "project_delay_months", "oc_received", "complaints",
          "amenities_score", "rera_registered"]
MONO = {"age_years": -1, "log_dist_metro": -1, "builder_reliability": 1, "flood_hazard": -1, "parking": 1,
        "oc_received": 1, "log_govt_rate": 1}

PROJ_COLS = ["builder_reliability", "builder_avg_delay", "delay_months", "oc_received", "complaints",
             "amenities_score", "rera_registered"]


def enrich(df: pd.DataFrame, projects: pd.DataFrame) -> pd.DataFrame:
    """Attach project/builder features and derived numeric columns."""
    df = df.copy()
    p = projects.set_index("project_id")
    pid = df["project_id"].fillna("")
    for c in PROJ_COLS:
        df[c if c != "delay_months" else "project_delay_months"] = pid.map(p[c]).astype(float)
    return derive(df)


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Derived numeric columns only (does not touch project/builder fields)."""
    df = df.copy()
    df["oc_received"] = np.where(df["under_construction"] == 1, np.nan, df["oc_received"])
    df["loc_code"] = df["micro_market"].map(LOC_CODE).astype(float)
    df["city_code"] = (df["city"] == "Pune").astype(float)
    df["log_area"] = np.log(df["carpet_area"])
    df["floor_c"] = np.minimum(df["floor"], 25)
    df["log_dist_metro"] = np.log1p(df["dist_metro_m"])
    df["log_dist_upcoming"] = np.log1p(df["dist_upcoming_metro_m"])
    df["log_dist_it"] = np.log1p(df["dist_it_hub_m"])
    df["log_govt_rate"] = np.log(df["govt_rate_inr_sqft"])
    df["log_rent"] = np.log(df["rent_psf_month"]) if "rent_psf_month" in df else np.nan
    return df


# ====================================================================== M1
class M1:
    def __init__(self, feats, params=None):
        self.feats = list(feats)
        self.params = params or {}

    def fit(self, df, y, w=None, rf=False):
        mono = [MONO.get(f, 0) for f in self.feats]
        if rf:
            prm = dict(boosting_type="rf", n_estimators=300, num_leaves=127, bagging_freq=1, bagging_fraction=0.632,
                       feature_fraction=0.6, min_child_samples=5, verbose=-1)
        else:
            prm = dict(n_estimators=900, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                       feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                       monotone_constraints=mono, monotone_constraints_method="advanced", verbose=-1)
        prm.update(self.params)
        cat = ["loc_code"] if "loc_code" in self.feats else "auto"
        self.model = lgb.LGBMRegressor(**prm)
        self.model.fit(df[self.feats], y, sample_weight=w, categorical_feature=cat)
        return self

    def predict(self, df):
        return self.model.predict(df[self.feats])

    def contrib(self, df):
        return self.model.booster_.predict(df[self.feats], pred_contrib=True)


# ====================================================================== M3
M3_CONT = ["log_area", "bhk", "floor_c", "age_years", "parking_f", "east_f", "sale_new", "under_construction",
           "metro_prox", "upcoming_prox", "it_prox", "flood_hazard", "ndvi", "poi_density_1km",
           "built_volume_growth", "night_light_trend", "rel_f", "delay_f", "oc_f", "amenities_f", "rent_f"]


def m3_design(df):
    d = pd.DataFrame(index=df.index)
    d["log_area"] = df["log_area"]
    d["bhk"] = df["bhk"]
    d["floor_c"] = df["floor_c"]
    d["age_years"] = df["age_years"]
    d["parking_f"] = df["parking"].fillna(0.8)
    d["east_f"] = df["east_facing"].fillna(0.35)
    d["sale_new"] = df["sale_new"]
    d["under_construction"] = df["under_construction"]
    d["metro_prox"] = np.exp(-df["dist_metro_m"] / 700)
    stage_w = df["metro_stage"].map({0: 0, 1: 0.3, 2: 0.6, 3: 1.0}).fillna(0)
    d["upcoming_prox"] = stage_w * np.exp(-df["dist_upcoming_metro_m"] / 1000)
    d["it_prox"] = np.exp(-df["dist_it_hub_m"] / 3000)
    d["flood_hazard"] = df["flood_hazard"]
    d["ndvi"] = df["ndvi"]
    d["poi_density_1km"] = df["poi_density_1km"]
    d["built_volume_growth"] = df["built_volume_growth"]
    d["night_light_trend"] = df["night_light_trend"]
    d["rel_f"] = df["builder_reliability"].fillna(0.65)
    d["delay_f"] = df["project_delay_months"].fillna(12)
    d["oc_f"] = df["oc_received"].fillna(0.0)
    d["amenities_f"] = df["amenities_score"].fillna(0.6)
    d["rent_f"] = (np.log(df["rent_psf_month"] / 21) if "rent_psf_month" in df else pd.Series(0.0, index=df.index)).fillna(0.0)
    return d[M3_CONT].values.astype(float)


class M3:
    """Penalised least squares with random effects: y = a + city + loc_RE + proj_RE + X b."""

    def __init__(self, lam_loc=4.0, lam_proj=12.0, lam_x=1e-3, use_project=True):
        self.lam_loc, self.lam_proj, self.lam_x, self.use_project = lam_loc, lam_proj, lam_x, use_project

    def _mat(self, df):
        X = m3_design(df)
        Xs = (X - self.mu) / self.sd
        n = len(df)
        loc = np.zeros((n, len(LOCS)))
        lc = df["loc_code"].values
        ok = ~np.isnan(lc)
        loc[np.where(ok)[0], lc[ok].astype(int)] = 1
        mats = [np.ones((n, 1)), df[["city_code"]].values, Xs, loc]
        if self.use_project:
            pr = np.zeros((n, len(self.projects)))
            pi = df["project_id"].fillna("").map(self.pidx).values
            okp = ~pd.isna(pi)
            pr[np.where(okp)[0], pi[okp].astype(int)] = 1
            mats.append(pr)
        return np.hstack(mats)

    def fit(self, df, y, w=None):
        X = m3_design(df)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.projects = sorted(set(df["project_id"].fillna("")) - {""})
        self.pidx = {p: i for i, p in enumerate(self.projects)}
        A = self._mat(df)
        w = np.ones(len(df)) if w is None else w
        pen = np.r_[0, 0, np.full(len(M3_CONT), self.lam_x), np.full(len(LOCS), self.lam_loc),
                    np.full(len(self.projects) if self.use_project else 0, self.lam_proj)]
        AtW = A.T * w
        self.coef = np.linalg.solve(AtW @ A + np.diag(pen), AtW @ y)
        k = 2 + len(M3_CONT)
        self.beta_raw = dict(zip(M3_CONT, self.coef[2:k] / self.sd))
        self.loc_effect = dict(zip(LOCS, self.coef[k:k + len(LOCS)]))
        self.proj_effect = dict(zip(self.projects, self.coef[k + len(LOCS):])) if self.use_project else {}
        return self

    def predict(self, df):
        return self._mat(df) @ self.coef


# ====================================================================== M2
class Comps:
    """Learned comparable-sales model. Attention weight of comp c for subject q:
        s_qc = -(d/ld)^2 - (dlogA/la)^2 - (dAge/lg)^2 - (dFloor/lf)^2 - lb|dBHK| + g*sameProject + e*isDeed - dT/lt
        w = softmax(s) over the K best; value = sum w * (y_c + hedonic adjustment to the subject)."""

    K = 30
    RADIUS = 5000

    def __init__(self, pool: pd.DataFrame, beta: dict, params=None):
        self.params = params or dict(ld=900.0, la=0.25, lg=6.0, lf=8.0, lb=0.8, g=1.0, e=0.3, lt=24.0)
        self.beta = beta
        self.set_pool(pool)

    def set_pool(self, pool):
        self.pool = pool.reset_index(drop=True)
        self.by_city = {}
        for city, g in self.pool.groupby("city"):
            idx = g.index.values
            self.by_city[city] = dict(
                idx=idx, lat=g["lat"].values, lon=g["lon"].values, y=g["y"].values, la=np.log(g["carpet_area"].values),
                age=g["age_years"].values, floor=np.minimum(g["floor"].values, 25), bhk=g["bhk"].values,
                park=g["parking"].values, uc=g["under_construction"].values, month=g["month_idx"].values,
                proj=g["project_id"].fillna("").values, deed=(g["label_source"].values == "deed").astype(float),
                ref=g["row_ref"].values)

    def candidates(self, q, before=True):
        c = self.by_city[q["city"]]
        d = geo.haversine_m(q["lat"], q["lon"], c["lat"], c["lon"])
        m = d < self.RADIUS
        if before:
            m &= c["month"] < q["month_idx"]
            m &= c["ref"] != q.get("row_ref", None)
        ii = np.where(m)[0]
        b = self.beta
        dla = np.log(q["carpet_area"]) - c["la"][ii]
        dage = q["age_years"] - c["age"][ii]
        dfl = min(q["floor"], 25) - c["floor"][ii]
        qp = q.get("parking", np.nan)
        dpk = np.where(np.isnan(c["park"][ii]) | (qp != qp), 0.0, (qp if qp == qp else 0) - np.nan_to_num(c["park"][ii]))
        duc = q["under_construction"] - c["uc"][ii]
        adj = (c["y"][ii] + b["log_area"] * dla + b["age_years"] * dage + b["floor_c"] * dfl
               + b["parking_f"] * dpk + b["under_construction"] * duc)
        same = (c["proj"][ii] == (q.get("project_id") or "##none##")).astype(float)
        dt = (q["month_idx"] - c["month"][ii]) if before else (44 - c["month"][ii])
        return dict(ii=ii, d=d[ii], dla=dla, dage=dage, dfl=dfl, dbhk=np.abs(q["bhk"] - c["bhk"][ii]), same=same,
                    deed=c["deed"][ii], dt=np.abs(dt), adj=adj, n1km=int((d[ii] < 1000).sum()))

    def score(self, cd, p=None):
        p = p or self.params
        return (-(cd["d"] / p["ld"]) ** 2 - (cd["dla"] / p["la"]) ** 2 - (cd["dage"] / p["lg"]) ** 2
                - (cd["dfl"] / p["lf"]) ** 2 - p["lb"] * cd["dbhk"] + p["g"] * cd["same"] + p["e"] * cd["deed"]
                - cd["dt"] / p["lt"])

    def predict_one(self, cd, p=None, detail=False):
        if len(cd["ii"]) == 0:
            return dict(pred=np.nan, disp=np.nan, n_eff=0.0, n1km=0, nearest=np.inf, top=None)
        s = self.score(cd, p)
        k = min(self.K, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        st = s[top] - s[top].max()
        w = np.exp(st)
        w /= w.sum()
        pred = float(np.sum(w * cd["adj"][top]))
        disp = float(np.sqrt(np.sum(w * (cd["adj"][top] - pred) ** 2)))
        out = dict(pred=pred, disp=disp, n_eff=float(1 / np.sum(w ** 2)), n1km=cd["n1km"],
                   nearest=float(cd["d"].min()), top=None)
        if detail:
            order = np.argsort(-w)
            out["top"] = (cd["ii"][top][order], w[order], cd["adj"][top][order], cd["d"][top][order],
                          cd["same"][top][order])
        return out

    def predict(self, Q: pd.DataFrame, before=True, cache=None):
        rows = []
        for i, (_, q) in enumerate(Q.iterrows()):
            cd = cache[i] if cache is not None else self.candidates(q, before)
            rows.append(self.predict_one(cd))
        return pd.DataFrame(rows).drop(columns="top")

    def tune(self, Q: pd.DataFrame, n_iter=40, seed=0):
        rng = np.random.default_rng(seed)
        cache = [self.candidates(q, True) for _, q in Q.iterrows()]
        y = Q["y"].values

        def loss(p):
            pr = np.array([self.predict_one(cd, p)["pred"] for cd in cache])
            ok = ~np.isnan(pr)
            return float(np.median(np.abs(pr[ok] - y[ok]))) + 0.05 * (1 - ok.mean())

        best_p, best = dict(self.params), loss(self.params)
        for _ in range(n_iter):
            p = {k: v * float(np.exp(rng.normal(0, 0.45))) for k, v in best_p.items()}
            l = loss(p)
            if l < best:
                best, best_p = l, p
        self.params = best_p
        return best


# ====================================================================== stacker + conformal
def density_bucket(n1km):
    n = np.asarray(n1km, float)
    return np.where(n < 15, 0, np.where(n < 60, 1, 2))


def fit_stacker(P: np.ndarray, y: np.ndarray, bucket: np.ndarray, step=0.05):
    """Convex weights over the model columns per density bucket, minimising mean |log error|.
    Rows where a model has no prediction (e.g. no comps) are scored with renormalised weights."""
    m = P.shape[1]
    if m == 1:
        grid = [(1.0,)]
    elif m == 2:
        grid = [(a, 1 - a) for a in np.arange(0, 1 + 1e-9, step)]
    else:
        grid = [(a, b, 1 - a - b) for a in np.arange(0, 1 + 1e-9, step) for b in np.arange(0, 1 - a + 1e-9, step)]
    W = {}
    for k in (0, 1, 2):
        msk = bucket == k
        if msk.sum() < 30:
            msk = np.ones_like(bucket, bool)
        Pk, yk = P[msk], y[msk]
        best = min(grid, key=lambda w: np.mean(np.abs(stack(Pk, np.zeros(len(Pk), int), {0: np.array(w)}) - yk)))
        W[k] = np.array(best, float)
    return W


def stack(P, bucket, W):
    out = np.empty(len(P))
    for k, w in W.items():
        m = bucket == k
        Pk = P[m]
        # missing comps prediction -> renormalise over the other models
        ww = np.tile(w, (m.sum(), 1)).astype(float)
        ww[np.isnan(Pk)] = 0
        zero = ww.sum(1) == 0
        ww[zero] = (~np.isnan(Pk[zero])).astype(float)
        ww /= ww.sum(1, keepdims=True)
        out[m] = np.nansum(Pk * ww, axis=1)
    return out


def sigma_features(P, disp, n1km, df):
    """Heteroscedasticity drivers for the normalised conformal score."""
    dis = np.nanstd(P, axis=1)
    return np.column_stack([
        np.ones(len(P)), dis, np.nan_to_num(disp, nan=0.15), 1 / np.sqrt(1 + np.asarray(n1km, float)),
        df["under_construction"].values, (df["flood_hazard"].values == 2).astype(float),
        np.clip(np.nanmean(P, axis=1) - 9.2, 0, None), (df["bhk"].values >= 4).astype(float),
        np.isnan(df["builder_reliability"].values).astype(float) + (df["rera_registered"].values == 0).astype(float)])


def sigma_of(F, coef, ood):
    """OOD widens the range monotonically on top of the fitted scale model."""
    return np.exp(F @ coef) * (1 + 0.5 * np.asarray(ood, float))


def fit_sigma(F, resid):
    t = np.log(np.abs(resid) + 0.004)
    A = F.T @ F + 1e-3 * np.eye(F.shape[1])
    return np.linalg.solve(A, F.T @ t)


def weighted_conformal_q(cal_lat, cal_lon, cal_score, lat, lon, alpha=0.10, h=4000.0, floor=0.02):
    """Spatially weighted conformal quantile with the test point's own mass at +inf."""
    d = geo.haversine_m(lat, lon, cal_lat, cal_lon)
    w = np.exp(-0.5 * (d / h) ** 2) + floor
    order = np.argsort(cal_score)
    s, ws = cal_score[order], w[order]
    p = ws / (ws.sum() + 1.0)
    cum = np.cumsum(p)
    j = np.searchsorted(cum, 1 - alpha)
    n_eff = float(w.sum() ** 2 / np.sum(w ** 2))
    if j >= len(s):
        return float(s[-1] * 1.25), n_eff, True
    return float(s[j]), n_eff, False


# ====================================================================== metrics
def metrics(pred_value, true_value):
    pred_value, true_value = np.asarray(pred_value, float), np.asarray(true_value, float)
    ok = ~np.isnan(pred_value)
    p, t = pred_value[ok], true_value[ok]
    ape = np.abs(p / t - 1)
    ratio = p / t
    medr = np.median(ratio)
    return {
        "n": int(ok.sum()),
        "MdAPE": round(float(np.median(ape) * 100), 2),
        "PE10": round(float(np.mean(ape <= 0.10) * 100), 1),
        "PE20": round(float(np.mean(ape <= 0.20) * 100), 1),
        "COD": round(float(np.mean(np.abs(ratio - medr)) / medr * 100), 2),
        "PRD": round(float(np.mean(ratio) / (p.sum() / t.sum())), 3),
        "median_ratio": round(float(medr), 3),
    }


# ====================================================================== explanation groups
GROUPS = {
    "metro": ["dist_metro_m", "dist_upcoming_metro_m", "metro_stage"],
    "size": ["carpet_area", "bhk"],
    "floor": ["floor", "total_floors"],
    "age": ["age_years", "sale_new", "under_construction"],
    "builder": ["project_id", "builder_reliability", "builder_avg_delay", "project_delay_months", "oc_received",
                "complaints", "amenities_score", "rera_registered"],
    "parking": ["parking", "east_facing"],
    "flood": ["flood_hazard"],
    "neighbourhood": ["lat", "lon", "cell_id", "dist_it_hub_m", "poi_density_1km", "ndvi", "night_light_level",
                      "night_light_trend", "built_volume_growth", "govt_rate_inr_sqft", "rent_psf_month"],
}


def shapley_groups(value_fn, subject: pd.Series, reference: pd.Series, groups=GROUPS):
    """Exact grouped Shapley values of value_fn relative to a reference property.
    value_fn: DataFrame -> np.ndarray (log price per sqft). 2^G evaluations, batched."""
    names = list(groups)
    G = len(names)
    combos = list(itertools.product([0, 1], repeat=G))
    rows = []
    for mask in combos:
        r = reference.copy()
        for gi, on in enumerate(mask):
            if on:
                for c in groups[names[gi]]:
                    r[c] = subject[c]
        rows.append(r)
    vals = value_fn(pd.DataFrame(rows))
    v = {mask: vals[i] for i, mask in enumerate(combos)}
    from math import factorial
    phi = np.zeros(G)
    for gi in range(G):
        for mask in combos:
            if mask[gi]:
                continue
            s = sum(mask)
            wgt = factorial(s) * factorial(G - s - 1) / factorial(G)
            with_g = list(mask)
            with_g[gi] = 1
            phi[gi] += wgt * (v[tuple(with_g)] - v[mask])
    base = v[tuple([0] * G)]
    full = v[tuple([1] * G)]
    return dict(zip(names, phi)), float(base), float(full)
