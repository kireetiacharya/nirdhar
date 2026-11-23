"""Valuation service: one request in, one explainable, auditable valuation out."""
from __future__ import annotations

import datetime as dt
import json
import pickle
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from . import geo, models as M
from .train import Engine, grade_of  # noqa: F401  (Engine needed to unpickle)

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
ASOF_IDX = 44
SYNTHETIC_NOTE = ("Prototype trained on a synthetic dataset that mirrors the structure and defects of Index-II, "
                  "IGRS, RERA and portal data. Values, ranges and grades are illustrative of the method, "
                  "not valuations of real properties.")
FLOOD_TXT = {0: "low", 1: "medium", 2: "high"}
STAGE_W = {0: 0, 1: 0.3, 2: 0.6, 3: 1.0}


def lakh(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x) / 1e5, 2)


def month_label(idx):
    return str(pd.Period("2023-01", "M") + int(idx))


class Valuer:
    def __init__(self, path: Path = ROOT / "artifacts" / "engine.pkl"):
        with open(path, "rb") as f:
            self.e: Engine = pickle.load(f)
        self.meta_info = self.e.meta
        self.proj = self.e.projects.set_index("project_id")
        self.bld = self.e.builders.set_index("builder_id")
        self.cells = {c: g.reset_index(drop=True) for c, g in self.e.cells.groupby("city")}
        pool = self.e.comps.pool
        self.pool = pool
        # typical market / guidance ratio per micro-market (for reconciliation text)
        r = (pool["label_ppsf"] / pool["govt_rate_inr_sqft"]).groupby(pool["micro_market"]).median()
        self.mkt_to_govt = r.to_dict()
        self.listings = self.e.listings
        rc = geo.REAL_DIR / "pune_osm_cells.csv"
        self.real_cells = pd.read_csv(rc) if rc.exists() else None
        # "typical flat" reference: age of completed stock, not of new launches
        done = pool[pool["under_construction"] == 0]
        for loc, ref in self.e.refs.items():
            g = done[done["micro_market"] == loc]
            if len(g):
                ref["age_years"] = float(g["age_years"].median())
        AUDIT.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ metadata for the UI
    def meta(self):
        locs = {}
        for name, (city, lat, lon, *_rest) in geo.LOCALITIES.items():
            locs.setdefault(city, []).append({"name": name, "lat": lat, "lon": lon})
        projects = []
        for pid, p in self.proj.iterrows():
            projects.append({
                "project_id": pid, "name": p["project_name"], "city": p["city"], "micro_market": p["micro_market"],
                "builder": p["builder_name"], "floors": int(p["approved_floors"]),
                "possession": p["revised_possession"], "completed": bool(p["completed"]),
                "rera_id": p["rera_id"], "lat": p["lat"], "lon": p["lon"]})
        projects.sort(key=lambda x: (x["city"], x["micro_market"], x["name"]))
        return {"localities": locs, "projects": projects, "model": self.model_info(), "presets": self.presets()}

    def presets(self):
        def pid(name):
            m = self.proj[self.proj["project_name"] == name]
            return m.index[0] if len(m) else ""
        return [
            {"key": "happy", "title": "Happy path", "subtitle": "Yerawada 2 BHK, reliable builder",
             "req": {"city": "Pune", "micro_market": "Yerawada", "project_id": pid("Kesar Yerawada Courtyard"),
                     "property_type": "apartment", "bhk": 2, "carpet_area": 850, "area_basis": "carpet", "floor": 7,
                     "parking": True, "facing": "East", "agreement_value": 8200000, "loan_amount": 6000000}},
            {"key": "hard", "title": "Hard case", "subtitle": "Delayed riverside project, Kharadi",
             "req": {"city": "Pune", "micro_market": "Kharadi", "project_id": pid("Ojas Riverside Residency"),
                     "property_type": "apartment", "bhk": 3, "carpet_area": 1150, "area_basis": "carpet", "floor": 3,
                     "total_floors": 16, "parking": True, "facing": "North", "agreement_value": 10500000,
                     "loan_amount": 7500000}},
            {"key": "fraud", "title": "Fraud catch", "subtitle": "Agreement ~30% above estimate, Wakad",
             "req": {"city": "Pune", "micro_market": "Wakad", "project_id": pid("Prithvi Wakad Vista"),
                     "property_type": "apartment", "bhk": 2, "carpet_area": 780, "area_basis": "carpet", "floor": 8,
                     "total_floors": 18, "parking": True, "facing": "East", "agreement_value": 10600000,
                     "loan_amount": 8000000}},
            {"key": "shop", "title": "Commercial", "subtitle": "Ground-floor shop, Kothrud (LAP)",
             "req": {"city": "Pune", "micro_market": "Kothrud", "property_type": "commercial",
                     "commercial_kind": "shop", "bhk": 1, "carpet_area": 420, "area_basis": "carpet", "floor": 0,
                     "total_floors": 4, "loan_amount": 2200000}},
            {"key": "hyd", "title": "Hyderabad", "subtitle": "Expansion market (synthetic)",
             "req": {"city": "Hyderabad", "micro_market": "Kondapur", "project_id": pid("Meghana Kondapur Heights"),
                     "property_type": "apartment", "bhk": 2, "carpet_area": 1050, "area_basis": "carpet", "floor": 11,
                     "total_floors": 14, "parking": False, "facing": "East", "agreement_value": 9400000,
                     "loan_amount": 6000000}},
        ]

    def model_info(self):
        m = self.meta_info
        return {k: m[k] for k in ("model_version", "trained_at", "data_snapshot_hash", "as_of", "splits", "rows",
                                  "stacker_weights", "grade_cuts", "results", "ladder", "pipeline_report",
                                  "comps_params", "data_is_synthetic", "ladder_city", "data_provenance")}

    # ------------------------------------------------------------------ helpers
    def _cell(self, city, lat, lon):
        c = self.cells[city]
        d = geo.haversine_m(lat, lon, c["lat"].values, c["lon"].values)
        i = int(d.argmin())
        return c.iloc[i], float(d[i])

    def _row(self, req):
        city, loc = req["city"], req["micro_market"]
        pid = req.get("project_id") or ""
        p = self.proj.loc[pid] if pid in self.proj.index else None
        if p is not None:
            lat, lon = float(p["lat"]), float(p["lon"])
        elif req.get("lat") and req.get("lon"):
            lat, lon = float(req["lat"]), float(req["lon"])
        else:
            lat, lon = geo.LOCALITIES[loc][1], geo.LOCALITIES[loc][2]
        cell, cell_d = self._cell(city, lat, lon)
        # area basis -> carpet
        area_in = float(req["carpet_area"])
        basis = req.get("area_basis", "carpet")
        loading = float(p["loading_factor"]) if p is not None else 1.33
        carpet = area_in if basis == "carpet" else (area_in / 1.12 if basis == "builtup" else area_in / loading)
        total_floors = int(req.get("total_floors") or (p["approved_floors"] if p is not None else 10))
        if p is not None:
            uc = float(p["possession_idx"] > ASOF_IDX)
            derived_age = max(0.0, (ASOF_IDX - p["possession_idx"]) / 12)
        else:
            uc, derived_age = 0.0, np.nan
        age = req.get("age_years")
        age = float(age) if age not in (None, "") else (derived_age if derived_age == derived_age else 5.0)
        if uc:
            age = 0.0
        row = {
            "row_ref": "query", "label_source": "query", "city": city, "micro_market": loc, "project_id": pid,
            "month_idx": ASOF_IDX, "lat": lat, "lon": lon, "cell_id": cell["cell_id"], "bhk": int(req["bhk"]),
            "carpet_area": round(carpet, 0), "floor": int(req["floor"]), "total_floors": total_floors,
            "age_years": age, "under_construction": uc, "sale_new": float(uc or age < 1),
            "parking": float(bool(req.get("parking"))) if req.get("parking") is not None else np.nan,
            "east_facing": float(req.get("facing") == "East") if req.get("facing") else np.nan,
        }
        for c in ("govt_rate_inr_sqft", "poi_density_1km", "ndvi", "night_light_level", "night_light_trend",
                  "built_volume_growth", "flood_hazard", "dist_metro_m", "dist_upcoming_metro_m", "metro_stage",
                  "dist_it_hub_m", "rent_psf_month"):
            row[c] = float(cell[c])
        df = M.enrich(pd.DataFrame([row]), self.e.projects)
        ctx = dict(project=p, cell_dist=cell_d, carpet=carpet, area_in=area_in, basis=basis, loading=loading,
                   derived_age=derived_age, uc=bool(uc))
        return df, ctx

    def _value_fn(self, df):
        return self.e.predict_frame(M.derive(df), before=False)["yhat"]

    # ------------------------------------------------------------------ main entry
    def value(self, req: dict, valuation_id: str | None = None, save=True) -> dict:
        req = {k: v for k, v in req.items() if v not in ("", None)}
        ptype = req.get("property_type", "apartment")
        vid = valuation_id or str(uuid.uuid4())
        base = {"valuation_id": vid, "requested_at": dt.datetime.now().isoformat(timespec="seconds"),
                "inputs": req, "synthetic_note": SYNTHETIC_NOTE,
                "model": {"model_version": self.meta_info["model_version"],
                          "data_snapshot_hash": self.meta_info["data_snapshot_hash"],
                          "trained_at": self.meta_info["trained_at"], "as_of": self.meta_info["as_of"]}}
        base["data_provenance"] = geo.DATA_PROVENANCE.get(req.get("city"), {})
        if ptype == "commercial":
            out = self._commercial(req, base)
            if save:
                self._audit(out)
            return out
        if ptype != "apartment":
            out = base | {"supported": False, "route": {
                "route": "Full physical valuation", "code": "field",
                "reason": f"'{ptype.replace('_', ' ')}' is outside the prototype's first segment (RERA-registered "
                          "apartments). The roadmap adds builder floors and houses in phase 2, plots in phase 3 and "
                          "commercial (income approach) in phase 4."}}
            if save:
                self._audit(out)
            return out
        if req.get("micro_market") not in geo.LOCALITIES:
            raise ValueError("Unknown micro-market")

        df, ctx = self._row(req)
        q = df.iloc[0]
        r = self.e.predict_frame(df, before=False)
        yhat, sigma = float(r["yhat"][0]), float(r["sigma"][0])
        cal = self.e.cal
        same = cal["city"] == q["city"]
        qhat, n_eff, thin = M.weighted_conformal_q(cal["lat"][same], cal["lon"][same], cal["score"][same],
                                                   q["lat"], q["lon"])
        area = float(q["carpet_area"])
        value = np.exp(yhat) * area
        lo, hi = np.exp(yhat - qhat * sigma) * area, np.exp(yhat + qhat * sigma) * area
        hw = (hi - lo) / (2 * value)
        grade = str(grade_of(np.array([hw]))[0])
        ood = float(r["ood"][0])
        grade_notes = []
        if ood > 1.0 and grade in "AB":
            grade = "C"
            grade_notes.append("Capped at C: property is unusual for this micro-market (out-of-distribution).")
        if thin or ctx["cell_dist"] > 1500:
            grade = "D"
            grade_notes.append("Set to D: too little calibration data or location outside precomputed coverage.")

        P = r["P"][0]
        model_vals = {"M1 Gradient boosting": P[0], "M2 Comparable sales": P[1], "M3 Hierarchical hedonic": P[2]}
        spread = float(np.nanstd(P))
        disagreement = {
            "models": [{"name": k, "value_inr": None if np.isnan(v) else round(float(np.exp(v) * area), -3),
                        "ppsf": None if np.isnan(v) else round(float(np.exp(v)), 0)} for k, v in model_vals.items()],
            "spread_pct": round(spread * 100, 1),
            "level": "low" if spread < 0.04 else ("medium" if spread < 0.08 else "high"),
            "weights": dict(zip(model_vals, self.e.W[int(r["bucket"][0])].round(2).tolist())),
        }
        comps = self._comps(q)
        drivers = self._drivers(df, q, value, area, ctx)
        recon = self._reconcile(q, value, lo, hi, area, req, ctx)
        project = self._project(q, ctx)
        flags = self._flags(q, req, value, lo, hi, recon, project, comps, disagreement, ood, ctx, thin)
        route = self._route(grade, flags, req, lo)
        dq = {
            "comps_within_1km": int(r["comps"]["n1km"].values[0]),
            "nearest_comp_m": round(float(r["comps"]["nearest"].values[0]), 0),
            "comps_effective_n": round(float(r["comps"]["n_eff"].values[0]), 1),
            "comps_deed_share_top10": comps["deed_share"],
            "calibration_effective_n": round(n_eff, 1),
            "location_source": "RERA project coordinates" if ctx["project"] is not None else (
                "user coordinates" if req.get("lat") else "micro-market centroid"),
            "location_cell_distance_m": round(ctx["cell_dist"], 0),
            "ood_score": round(ood, 2),
            "ood_max_z": round(float(r["zmax"][0]), 2),
            "density_bucket": ["low", "mid", "high"][int(r["bucket"][0])],
        }
        out = base | {
            "supported": True,
            "property": {
                "city": q["city"], "micro_market": q["micro_market"], "project_id": q["project_id"],
                "project_name": None if ctx["project"] is None else ctx["project"]["project_name"],
                "bhk": int(q["bhk"]), "carpet_area": area, "area_input": ctx["area_in"], "area_basis": ctx["basis"],
                "floor": int(q["floor"]), "total_floors": int(q["total_floors"]), "age_years": round(float(q["age_years"]), 1),
                "under_construction": ctx["uc"], "parking": req.get("parking"), "facing": req.get("facing"),
                "lat": round(float(q["lat"]), 5), "lon": round(float(q["lon"]), 5),
            },
            "valuation": {
                "value_inr": round(value, -3), "p05_inr": round(lo, -3), "p95_inr": round(hi, -3),
                "ppsf": round(value / area, 0), "halfwidth_pct": round(hw * 100, 1), "grade": grade,
                "grade_label": {"A": "High confidence", "B": "Good confidence", "C": "Low confidence",
                                "D": "Very low confidence"}[grade],
                "grade_notes": grade_notes, "conformal_q": round(qhat, 3), "sigma": round(sigma, 4),
                "confidence_status": "Calibrated on held-out synthetic deeds (illustrative)",
            },
            "disagreement": disagreement, "comparables": comps, "drivers": drivers, "reconciliation": recon,
            "project_intel": project, "flags": flags, "route": route, "data_quality": dq,
        }
        if save:
            self._audit(out)
        return out

    # ------------------------------------------------------------------ commercial (income approach)
    COMM = {  # stated assumptions, to be calibrated with lease and sale data in the pilot
        "shop": {"floor_mult": {0: 4.5, 1: 2.5}, "upper_mult": 1.6, "cap": 0.080},
        "office": {"floor_mult": {}, "upper_mult": 1.8, "cap": 0.075},
        "vacancy": 0.08, "opex": 0.10, "cap_band": 0.01, "rent_band": 0.15, "lap_ltv": 0.60,
    }

    def _commercial(self, req, base):
        city, loc = req["city"], req["micro_market"]
        kind = req.get("commercial_kind") or "shop"
        A = self.COMM
        lat = float(req["lat"]) if req.get("lat") else geo.LOCALITIES[loc][1]
        lon = float(req["lon"]) if req.get("lon") else geo.LOCALITIES[loc][2]
        basis = req.get("area_basis", "carpet")
        area_in = float(req["carpet_area"])
        area = area_in if basis == "carpet" else (area_in / 1.12 if basis == "builtup" else area_in / 1.25)
        floor = int(req.get("floor") or 0)
        prop = {"city": city, "micro_market": loc, "kind": kind, "carpet_area": round(area, 0), "area_input": area_in,
                "area_basis": basis, "floor": floor, "lat": round(lat, 5), "lon": round(lon, 5)}
        rc = None
        if self.real_cells is not None and city == "Pune":
            d = geo.haversine_m(lat, lon, self.real_cells["lat"].values, self.real_cells["lon"].values)
            if d.min() < 600:
                rc = self.real_cells.iloc[int(d.argmin())]
        if rc is None or not np.isfinite(rc["rent_psf_month_1500m"]):
            return base | {"supported": True, "mode": "commercial_income", "property": prop, "valuation": None,
                           "flags": [{"severity": "red", "code": "NO_RENT_EVIDENCE", "title": "No rent evidence nearby",
                                      "detail": "Fewer than 5 real rent listings within 1.5 km; income approach not possible."}],
                           "route": {"route": "Full physical valuation", "code": "field",
                                     "reason": "No rent evidence for an income-approach estimate."}}
        rent_res = float(rc["rent_psf_month_1500m"])
        footfall = int(rc["poi_food_1km"] + rc["poi_retail_1km"])
        cfg = A[kind]
        mult = cfg["floor_mult"].get(floor, cfg["upper_mult"])
        uplift = (0.3 if footfall >= 60 else 0.15 if footfall >= 20 else 0.0) if kind == "shop" else 0.0
        mult += uplift
        rent_c = rent_res * mult
        gross = rent_c * 12 * area
        noi = gross * (1 - A["vacancy"] - A["opex"])
        cap = cfg["cap"]
        value = noi / cap
        lo = noi * (1 - A["rent_band"]) / (cap + A["cap_band"])
        hi = noi * (1 + A["rent_band"]) / (cap - A["cap_band"])
        hw = (hi - lo) / (2 * value)
        loan = float(req.get("loan_amount") or 0)
        eligible = A["lap_ltv"] * lo
        n = int(rc["rent_listings_1500m"])
        flags = [{"severity": "info", "code": "ASSUMPTION_DRIVEN", "title": "Income approach: assumption-driven",
                  "detail": "Multipliers and capitalisation rates are stated assumptions, not yet calibrated on "
                            "commercial lease or sale records."}]
        if n < 15:
            flags.append({"severity": "amber", "code": "THIN_RENT_EVIDENCE", "title": "Thin rent evidence",
                          "detail": f"Only {n} rent listings within 1.5 km."})
        if loan and loan > eligible:
            flags.append({"severity": "amber", "code": "LOAN_ABOVE_LIMIT", "title": "Requested loan above LAP limit",
                          "detail": f"Requested ₹{lakh(loan)} L vs ₹{lakh(eligible)} L at {A['lap_ltv']:.0%} of the lower bound."})
        steps = [
            {"step": "Residential rent evidence", "value": f"₹{rent_res:.1f}/sq ft/month",
             "source": f"median of {n} real Makaan rent listings within 1.5 km"},
            {"step": f"{kind.title()} multiplier", "value": f"× {mult:.2f}",
             "source": (f"floor {floor} base {cfg['floor_mult'].get(floor, cfg['upper_mult']):.2f}"
                        + (f" + footfall uplift {uplift:.2f} ({footfall} shops and eateries within 1 km, OSM)" if uplift else ""))
             + " (assumption)"},
            {"step": "Commercial rent", "value": f"₹{rent_c:.1f}/sq ft/month", "source": "evidence × multiplier"},
            {"step": "Gross annual rent", "value": f"₹{lakh(gross)} L", "source": f"× 12 × {area:.0f} sq ft carpet"},
            {"step": "Net operating income", "value": f"₹{lakh(noi)} L",
             "source": f"less vacancy {A['vacancy']:.0%} and costs {A['opex']:.0%} (assumption)"},
            {"step": "Capitalisation rate", "value": f"{cap:.1%}", "source": f"market yield for {kind}s (assumption, ±{A['cap_band']:.0%})"},
            {"step": "Value = NOI ÷ cap rate", "value": f"₹{lakh(value)} L", "source": "income approach"},
        ]
        return base | {
            "supported": True, "mode": "commercial_income", "property": prop,
            "valuation": {"value_inr": round(value, -3), "p05_inr": round(lo, -3), "p95_inr": round(hi, -3),
                          "ppsf": round(value / area, 0), "halfwidth_pct": round(hw * 100, 1), "grade": "C",
                          "grade_label": "Low confidence",
                          "grade_notes": ["Capped at C: income approach with uncalibrated assumptions."],
                          "confidence_status": "Range from rent ±15% and cap rate ±1%; not statistically calibrated"},
            "commercial": {"steps": steps, "assumptions": A[kind] | {k: A[k] for k in ("vacancy", "opex", "lap_ltv")},
                           "footfall_poi_1km": footfall, "rent_listings_1500m": n,
                           "nearest_metro": rc["nearest_metro_operational"],
                           "dist_metro_m": float(rc["dist_metro_operational_m"])},
            "flags": flags,
            "route": {"route": "Full physical valuation", "code": "field",
                      "reason": "Commercial valuations are routed to a valuer until the income model is calibrated; "
                                "Nirdhar supplies the rent evidence and a starting value.",
                      "ltv_base_inr": round(lo, -3), "ltv_base_label": "Lower bound of the income-approach range",
                      "max_eligible_loan_inr": round(eligible, -3),
                      "requested_loan_inr": round(loan, -3) if loan else None,
                      "requested_within_limit": None if not loan else bool(loan <= eligible + 1),
                      "ltv_rule": f"LAP at {A['lap_ltv']:.0%} of the lower bound (illustrative lender policy)"},
        }

    # ------------------------------------------------------------------ comparables
    def _comps(self, q):
        cd = self.e.comps.candidates(q, before=False)
        res = self.e.comps.predict_one(cd, detail=True)
        if res["top"] is None:
            return {"considered": 0, "selected": 0, "items": [], "deed_share": 0.0}
        ii, w, adj, d, same = res["top"]
        c = self.e.comps.by_city[q["city"]]
        rows = []
        for k in range(len(ii)):
            pr = self.pool.iloc[c["idx"][ii[k]]]
            pn = self.proj.loc[pr["project_id"], "project_name"] if pr["project_id"] in self.proj.index else "—"
            reasons = []
            if same[k]:
                reasons.append("same project")
            reasons.append(f"{d[k]:.0f} m away")
            da = pr["carpet_area"] / q["carpet_area"] - 1
            reasons.append("similar size" if abs(da) < 0.1 else f"{da:+.0%} size")
            if int(pr["bhk"]) == int(q["bhk"]):
                reasons.append(f"same {int(q['bhk'])} BHK")
            dage = pr["age_years"] - q["age_years"]
            if abs(dage) <= 2:
                reasons.append("similar age")
            mo = ASOF_IDX - int(pr["month_idx"])
            reasons.append("sold this month" if mo == 0 else f"{mo} mo ago")
            rows.append({
                "project": pn, "micro_market": pr["micro_market"], "distance_m": round(float(d[k]), 0),
                "month": month_label(pr["month_idx"]), "bhk": int(pr["bhk"]), "carpet_area": float(pr["carpet_area"]),
                "floor": int(pr["floor"]), "age_years": round(float(pr["age_years"]), 1),
                "source": "Registered deed" if pr["label_source"] == "deed" else "Listing (deed-calibrated)",
                "price_inr": round(float(pr["label_value"]), -3),
                "ppsf_raw": round(float(pr["label_ppsf"]), 0),
                "ppsf_time_adj": round(float(np.exp(pr["y"])), 0),
                "ppsf_adj_to_subject": round(float(np.exp(adj[k])), 0),
                "weight": round(float(w[k]), 3), "why": ", ".join(reasons),
                "lat": float(pr["lat"]), "lon": float(pr["lon"]),
            })
        top10 = rows[:10]
        # context layer for the map: every candidate project within 3 km, aggregated
        near = cd["d"] < 3000
        cand = self.pool.iloc[c["idx"][cd["ii"][near]]].assign(adj=np.exp(cd["adj"][near]))
        ctx_rows = []
        for pid, g in cand.groupby("project_id"):
            if not pid or pid not in self.proj.index:
                continue
            pp = self.proj.loc[pid]
            ctx_rows.append({"project": pp["project_name"], "lat": float(pp["lat"]), "lon": float(pp["lon"]),
                             "sales": int(len(g)), "median_adj_ppsf": round(float(g["adj"].median()), 0),
                             "selected": bool(pid in {self.pool.iloc[c["idx"][j]]["project_id"] for j in ii})})
        return {"considered": int(len(cd["ii"])), "selected": int(len(ii)), "items": rows, "context": ctx_rows,
                "deed_share": round(float(np.mean([r["source"] == "Registered deed" for r in top10])), 2),
                "comps_value_ppsf": round(float(np.exp(res["pred"])), 0)}

    # ------------------------------------------------------------------ explanations
    def _drivers(self, df, q, value, area, ctx):
        ref = pd.Series(self.e.refs[q["micro_market"]])
        subj = df.iloc[0].copy()
        cols = sorted(set(subj.index) | set(ref.index))
        subj = subj.reindex(cols)
        ref = ref.reindex(cols)
        for c in cols:  # fields the reference lacks default to the subject (not explained)
            if pd.isna(ref[c]) and not pd.isna(subj[c]):
                ref[c] = subj[c]
        ref["carpet_area"] = ref.get("carpet_area", subj["carpet_area"])
        phi, base, full = M.shapley_groups(self._value_fn, subj, ref)
        v0 = np.exp(base) * area
        v1 = np.exp(full) * area
        L = (v1 - v0) / (full - base) if abs(full - base) > 1e-9 else v1
        p = ctx["project"]
        lf = geo.location_features(q["city"], [q["lat"]], [q["lon"]])
        label = {
            "metro": f"Metro: {lf['nearest_metro'][0]} {q['dist_metro_m']:.0f} m"
                     + (f"; {lf['nearest_upcoming_metro'][0]} {geo.METRO_STAGE_LABEL[int(q['metro_stage'])]}"
                        if q["metro_stage"] > 0 else ""),
            "size": f"Size {q['carpet_area']:.0f} sq ft, {int(q['bhk'])} BHK (typical {ref['carpet_area']:.0f})",
            "floor": f"Floor {int(q['floor'])} of {int(q['total_floors'])}",
            "age": ("Under construction" + (f", possession {p['revised_possession']}" if p is not None else ""))
                   if ctx["uc"] else f"Age {q['age_years']:.1f} years (typical {ref['age_years']:.1f})",
            "builder": (f"Project & builder: {p['project_name']}'s own sales, builder reliability "
                        f"{q['builder_reliability']:.2f}, delay {q['project_delay_months']:.0f} mo")
                       if p is not None else "No RERA project linked",
            "parking": ("Covered parking" if q["parking"] == 1 else "No covered parking") if q["parking"] == q["parking"] else "Parking not stated",
            "flood": f"Flood hazard {FLOOD_TXT[int(q['flood_hazard'])]}",
            "neighbourhood": f"Exact location vs {q['micro_market']} average (amenities, greenery, growth)",
        }
        def typical(g):
            return all((subj[c] == ref[c]) or (pd.isna(subj[c]) and pd.isna(ref[c])) for c in M.GROUPS[g])
        items = [{"factor": label[k] + (" — typical for area" if typical(k) else ""), "group": k,
                  "inr": round(float(v * L), -3), "log_effect": round(float(v), 4)} for k, v in phi.items()]
        items.sort(key=lambda x: -abs(x["inr"]))
        return {"baseline_inr": round(float(v0), -3), "baseline_label": f"Typical flat of this size in {q['micro_market']}",
                "value_inr": round(float(v1), -3), "items": items,
                "method": "Exact grouped Shapley values of the full stacked ensemble (2^8 coalitions), relative to a "
                          "typical flat in the micro-market; converted to rupees with the logarithmic mean so they sum "
                          "exactly to the difference."}

    # ------------------------------------------------------------------ three-price reconciliation
    def _reconcile(self, q, value, lo, hi, area, req, ctx):
        govt = float(q["govt_rate_inr_sqft"]) * area
        typical_ratio = float(self.mkt_to_govt.get(q["micro_market"], np.nan))
        tix = np.array(self.e.time_index[q["city"]])
        if req.get("asking_price"):
            ask, ask_src, n_ask = float(req["asking_price"]), "Asking price supplied", None
        else:
            L = self.listings
            sel = L[(L["project_id"] == q["project_id"]) & (L["month_idx"] >= ASOF_IDX - 12)] if q["project_id"] else L.iloc[0:0]
            src = "Project asking (median, last 12 mo, time-adjusted)"
            if len(sel) < 3:
                sel = L[(L["micro_market"] == q["micro_market"]) & (L["month_idx"] >= ASOF_IDX - 12)]
                src = "Micro-market asking (median, last 12 mo, time-adjusted)"
            adj = sel["ask_ppsf"].values * np.exp(tix[ASOF_IDX] - tix[sel["month_idx"].values])
            ask, ask_src, n_ask = float(np.median(adj) * area), src, int(len(sel))
        agr = float(req["agreement_value"]) if req.get("agreement_value") else None
        notes = []
        ask_gap = ask / value - 1
        notes.append(f"Asking price is {ask_gap:+.0%} vs our estimate. On matched Pune listing-deed pairs, sale value "
                     f"averages {abs(self.meta_info['pipeline_report']['deed_calibration']['mean_listing_to_deed_gap']):.0%} "
                     "below asking: portals carry negotiation headroom.")
        notes.append(f"Estimated market value is {value / govt:.2f}× the government guidance value "
                     f"(typical for {q['micro_market']}: {typical_ratio:.2f}×). Guidance rates are a stamp-duty floor, "
                     "revised infrequently, so they sit well below market.")
        if agr:
            pos = "inside" if lo <= agr <= hi else ("above" if agr > hi else "below")
            notes.append(f"Agreement value is {agr / value - 1:+.1%} vs estimate, {pos} the 90% range.")
        items = [
            {"label": "Government guidance value", "inr": round(govt, -3), "source": f"Guidance rate ₹{q['govt_rate_inr_sqft']:,.0f}/sq ft × carpet area"},
            {"label": "Asking price", "inr": round(ask, -3), "source": ask_src + (f" · {n_ask} listings" if n_ask else "")},
            {"label": "Nirdhar estimated value", "inr": round(float(value), -3), "source": "Deed-calibrated ensemble", "range": [round(lo, -3), round(hi, -3)]},
        ]
        if agr:
            items.append({"label": "Agreement value", "inr": round(agr, -3), "source": "Supplied with application",
                          "status": "ok" if lo <= agr <= hi else ("high" if agr > hi else "low")})
        return {"items": items, "notes": notes, "ask_gap_pct": round(ask_gap * 100, 1),
                "value_to_guidance": round(value / govt, 2), "typical_value_to_guidance": round(typical_ratio, 2),
                "agreement_gap_pct": None if agr is None else round((agr / value - 1) * 100, 1)}

    # ------------------------------------------------------------------ project intelligence
    def _project(self, q, ctx):
        p = ctx["project"]
        if p is None:
            return {"linked": False, "message": "No RERA project linked. Location from micro-market centroid or "
                                                "user coordinates; builder signals unavailable."}
        b = self.bld.loc[p["builder_id"]]
        have = [a for a in str(p["approvals"]).split(";") if a]
        expected = ["Building permission", "Environmental clearance", "Fire NOC"] + (["Occupancy certificate"] if p["completed"] else [])
        if int(p["approved_floors"]) <= 6:
            expected.remove("Fire NOC")
        missing = [a for a in expected if a not in have]
        if not p["rera_registered"]:
            status = "Completed before RERA (not registered)"
        elif not p["completed"]:
            status = "Under construction"
        else:
            status = "Completed" + (" · OC received" if p["oc_received"] else " · OC not on record")
        return {
            "linked": True, "project_id": q["project_id"], "rera_id": p["rera_id"] or None,
            "rera_registered": bool(p["rera_registered"]), "name": p["project_name"], "micro_market": p["micro_market"],
            "status": status, "launch": p["launch_date"], "promised_possession": p["promised_possession"],
            "revised_possession": p["revised_possession"], "delay_months": int(p["delay_months"]),
            "total_units": int(p["total_units"]), "approved_floors": int(p["approved_floors"]),
            "oc_received": bool(p["oc_received"]), "complaints": None if pd.isna(p["complaints"]) else int(p["complaints"]),
            "approvals_on_record": have, "approvals_missing": missing, "loading_factor": float(p["loading_factor"]),
            "builder": {"name": b["name"], "projects_count": int(b["projects_count"]),
                        "avg_delay_months": round(float(b["avg_delay_months"]), 1),
                        "oc_rate": None if pd.isna(b["oc_rate"]) else round(float(b["oc_rate"]), 2),
                        "reliability_score": round(float(b["reliability_score"]), 2)},
            "transactions_in_pool": int((self.pool["project_id"] == q["project_id"]).sum()),
        }

    # ------------------------------------------------------------------ risk flags
    def _flags(self, q, req, value, lo, hi, recon, project, comps, dis, ood, ctx, thin):
        F = []

        def add(sev, code, title, detail):
            F.append({"severity": sev, "code": code, "title": title, "detail": detail})

        agr = req.get("agreement_value")
        if agr:
            agr = float(agr)
            gap = agr / value - 1
            if agr > hi:
                add("red", "INFLATED_AGREEMENT", "Agreement value above upper bound — refer",
                    f"₹{lakh(agr)} L is {gap:+.0%} vs estimate and above the 95th-percentile bound ₹{lakh(hi)} L. "
                    "Possible collusive over-valuation; refer before disbursal.")
            elif agr < lo:
                add("amber", "LOW_AGREEMENT", "Agreement value below range",
                    f"{gap:+.0%} vs estimate. Check for under-declaration or a distress sale.")
            govt = recon["items"][0]["inr"]
            if agr < govt:
                add("amber", "BELOW_GUIDANCE", "Agreement below government guidance value",
                    "Stamp duty is charged on the higher of the two; confirm duty paid on guidance value.")
        if abs(recon["ask_gap_pct"]) > 25:
            add("amber", "ASK_DEVIATION", "Asking price far from estimate",
                f"Asking is {recon['ask_gap_pct']:+.0f}% vs estimate; verify the listing and unit details.")
        if project["linked"]:
            if not project["rera_registered"]:
                add("amber", "NO_RERA", "No RERA registration on record",
                    "Pre-2017 project: verify building permission and OC manually.")
            if project["approvals_missing"]:
                add("amber" if "Occupancy certificate" in project["approvals_missing"] else "info", "MISSING_APPROVALS",
                    "Approvals missing from record", ", ".join(project["approvals_missing"]))
            if project["delay_months"] >= 12:
                add("amber", "BUILDER_DELAY", "Possession delayed",
                    f"Revised possession {project['revised_possession']} is {project['delay_months']} months after the promised date.")
            if (project["complaints"] or 0) >= 5:
                add("amber", "RERA_COMPLAINTS", "RERA complaints on record", f"{project['complaints']} complaints filed against the project.")
            if project["builder"]["reliability_score"] < 0.5:
                add("amber", "BUILDER_RISK", "Low builder reliability",
                    f"Builder delivery score {project['builder']['reliability_score']:.2f} (avg delay {project['builder']['avg_delay_months']:.0f} mo).")
        else:
            add("amber", "NO_PROJECT", "Property not linked to a RERA project", "Builder and approval checks not possible.")
        if int(q["flood_hazard"]) == 2:
            add("amber", "FLOOD", "High flood hazard", "Location falls in a high flood-hazard cell; consider insurance and physical check.")
        if q["dist_metro_m"] > 0 and comps["selected"] and dq_thin(comps):
            add("amber", "THIN_COMPS", "Few nearby comparable sales", "Fewer than 5 comparables within 1.5 km; range widened.")
        if dis["level"] == "high":
            add("amber", "MODEL_DISAGREEMENT", "Models disagree", f"Spread across the three models is {dis['spread_pct']}%.")
        if ood > 1.0:
            add("amber", "OUT_OF_DISTRIBUTION", "Unusual property for this micro-market",
                "Size, floor, age or location sits outside what the model has seen here; range widened.")
        if thin:
            add("red", "THIN_CALIBRATION", "Insufficient calibration data", "Confidence cannot be established; field visit required.")
        if ctx["derived_age"] == ctx["derived_age"] and req.get("age_years") not in (None, "") and not ctx["uc"]:
            if abs(float(req["age_years"]) - ctx["derived_age"]) > 2:
                add("info", "AGE_MISMATCH", "Stated age differs from RERA possession date",
                    f"Stated {float(req['age_years']):.0f} yrs vs {ctx['derived_age']:.1f} yrs implied by possession date.")
        order = {"red": 0, "amber": 1, "info": 2}
        F.sort(key=lambda f: order[f["severity"]])
        return F

    # ------------------------------------------------------------------ lender workflow
    def _route(self, grade, flags, req, lo):
        sev = {f["severity"] for f in flags}
        blocking = [f for f in flags if f["severity"] in ("red", "amber")]
        agr = float(req["agreement_value"]) if req.get("agreement_value") else None
        base_val = min(lo, agr) if agr else lo

        def ltv_cap(loan):
            return 0.90 if loan <= 30e5 else (0.80 if loan <= 75e5 else 0.75)

        eligible = 0.0
        for cap in (0.90, 0.80, 0.75):
            L = cap * base_val
            if ltv_cap(L) == cap:
                eligible = L
                break
        loan = float(req.get("loan_amount") or eligible)
        if grade in ("C", "D") or "red" in sev:
            code, route = "field", "Full physical valuation"
            why = [f"grade {grade}"] if grade in ("C", "D") else []
            why += [f["title"] for f in flags if f["severity"] == "red"]
            reason = ("Triggered by " + "; ".join(why) +
                      ". Valuer visits, starting from the Nirdhar report and comparables.")
        elif loan > 75e5:
            code, route = "dual", "Two valuers + Nirdhar"
            reason = "Loan above ₹75 L: two independent valuer reports required (HFC norm); Nirdhar is the third check."
        elif blocking:
            code, route = "field", "Full physical valuation"
            reason = "Risk flags present: " + "; ".join(f["title"] for f in blocking[:3]) + "."
        elif grade == "A":
            code, route = "desktop", "Desktop valuation"
            reason = "Grade A, loan ≤ ₹75 L, no flags: valuer signs off remotely on geo-tagged photos."
        else:
            code, route = "light", "Light-touch"
            reason = "Grade B, no flags: desktop review plus a quick photo check by a field agent."
        if grade in ("C", "D") and loan > 75e5:
            route += " (two valuers)"
        return {
            "route": route, "code": code, "reason": reason,
            "ltv_base_inr": round(base_val, -3),
            "ltv_base_label": "Lower of 5th-percentile value and agreement value" if agr else "5th-percentile value",
            "max_eligible_loan_inr": round(eligible, -3),
            "requested_loan_inr": round(loan, -3) if req.get("loan_amount") else None,
            "requested_within_limit": None if not req.get("loan_amount") else bool(loan <= eligible + 1),
            "ltv_rule": "RBI LTV caps: 90% up to ₹30 L, 80% up to ₹75 L, 75% above",
            "precedence": "Field (grade C/D or red flag) > two valuers (loan > ₹75 L) > field (amber flags) > desktop (A) > light-touch (B)",
        }

    # ------------------------------------------------------------------ audit / replay
    def _audit(self, out):
        (AUDIT / f"{out['valuation_id']}.json").write_text(json.dumps(out, indent=1, default=_json_default))

    def replay(self, vid: str):
        path = AUDIT / f"{vid}.json"
        if not path.exists():
            raise FileNotFoundError(vid)
        old = json.loads(path.read_text())
        new = self.value(old["inputs"], valuation_id=vid, save=False)
        same_model = old["model"]["model_version"] == new["model"]["model_version"] and \
            old["model"]["data_snapshot_hash"] == new["model"]["data_snapshot_hash"]
        ov = old.get("valuation", {}).get("value_inr")
        nv = new.get("valuation", {}).get("value_inr")
        return {"valuation_id": vid, "same_model_and_snapshot": same_model, "original_value_inr": ov,
                "replayed_value_inr": nv, "exact_match": ov == nv, "original": old}


def dq_thin(comps):
    near = [c for c in comps["items"] if c["distance_m"] <= 1500]
    return len(near) < 5


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
