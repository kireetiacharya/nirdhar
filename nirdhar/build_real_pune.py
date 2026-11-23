"""Build Nirdhar's REAL Pune layers from downloaded open data.

Inputs (paths as arguments, nothing fetched here):
  --gpkg      Geofabrik india/western-zone-latest-free .gpkg  (OpenStreetMap, ODbL 1.0)
  --stations  data/real/osm_pune_metro_stations.csv            (OSM via Overpass, ODbL 1.0)
  --rents     Indian_housing_Pune_data.csv                      (Makaan rental listings via Kaggle,
                                                                 bhavyadhingra00020; licence unclear)
Outputs in data/real/:
  pune_osm_cells.csv          per ~0.38 km cell: POI counts, metro distances, river distance, rent
  pune_metro_stations.csv     stations + stage used by the model (see STAGE RULE below)
  pune_localities.csv         micro-market centres taken from OSM place nodes
  pune_rents_clean.csv        cleaned real monthly rents (apartments)
  sources.json                provenance for every layer

STAGE RULE (team must verify against punemetrorail.org before submission):
  operational  = station names on Line 1 (Purple, PCMC-Swargate) or Line 2 (Aqua, Vanaz-Ramwadi)
  upcoming     = every other OSM station; stage 3 if OSM tags it under construction or it is on
                 the Pink Line (Line 3, Hinjewadi-Shivajinagar), else 2 (planned/approved).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sqlite3
import struct
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "real"
LAT0, LAT1, LON0, LON1, S = 18.40, 18.75, 73.65, 74.10, 0.0035

LINE_1_2 = {  # Purple and Aqua lines as built (names as spelled in OSM)
    "Pimpri Chinchwad Municipal Corporation", "Sant Tukaram Nagar", "Nashik Phata", "Kasarwadi", "Phugewadi",
    "Dapodi", "Bopodi", "Khadki", "Range Hills", "Shivaji Nagar", "District Court", "Kasba Peth कसबा पेठ",
    "Swargate", "Vanaz", "Anandnagar", "Nal Stop", "Garware College", "Deccan Gymkhana",
    "Chhatrapati Sambhaji Udyan", "PMC Bhavan", "Ruby Hall Clinic", "Bund Garden", "Yerwada", "Kalyani Nagar",
    "Ramwadi",
}

# Micro-markets for the Pune-first pilot. The OSM place node with this name supplies the centre;
# `prior` is ONLY a synthetic-generator prior (approx. carpet Rs/sq ft) and is never shown as data.
LOCALITIES = {
    "Koregaon Park": 17000, "Shivajinagar": 16000, "Kothrud": 13500, "Aundh": 13500, "Baner": 12500,
    "Viman Nagar": 12500, "Balewadi": 11500, "Bavdhan": 11000, "Pashan": 11000, "Kharadi": 11000,
    "Magarpatta": 11000, "Yerawada": 10000, "Wakad": 9500, "Pimple Saudagar": 9500, "Warje": 8800,
    "Hinjewadi": 8500, "Hadapsar": 8500, "Pimpri": 8000, "Dhanori": 7800, "Kondhwa": 7800, "Ravet": 7500,
    "Undri": 7200, "Wagholi": 6500,
}
FALLBACK = {"Kothrud": (18.5074, 73.8077), "Magarpatta": (18.5150, 73.9330), "Undri": (18.4570, 73.9050)}
OSM_ALIASES = {"Hinjewadi": ["Hinjawadi", "Hinjewadi"], "Yerawada": ["Yerawada", "Yerwada"],
               "Magarpatta": ["Magarpatta City", "Magarpatta"], "Pimple Saudagar": ["Pimple Saudagar"],
               "Shivajinagar": ["Shivajinagar", "Shivaji Nagar"]}


def _geom(blob):
    """Return (kind, coords ndarray[n,2]) from a GeoPackage geometry blob."""
    flags = blob[3]
    env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 7]
    w = memoryview(blob)[8 + env:]

    def read(off):
        bo = "<" if w[off] == 1 else ">"
        t = struct.unpack_from(bo + "I", w, off + 1)[0] % 1000
        off += 5
        if t == 1:
            return "point", [np.array(struct.unpack_from(bo + "dd", w, off)).reshape(1, 2)], off + 16
        if t == 2:
            n = struct.unpack_from(bo + "I", w, off)[0]
            return "line", [np.array(struct.unpack_from(bo + f"{2 * n}d", w, off + 4)).reshape(n, 2)], off + 4 + 16 * n
        if t == 5:
            n = struct.unpack_from(bo + "I", w, off)[0]
            off += 4
            parts = []
            for _ in range(n):
                _, p, off = read(off)
                parts += p
            return "line", parts, off
        return "other", [], off

    kind, parts, _ = read(0)
    return kind, (np.vstack(parts) if parts else np.empty((0, 2)))


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))


def nearest_dist(lat, lon, plat, plon, chunk=400):
    out = np.full(len(lat), np.inf)
    idx = np.zeros(len(lat), int)
    for s in range(0, len(plat), chunk):
        d = hav(lat[:, None], lon[:, None], plat[None, s:s + chunk], plon[None, s:s + chunk])
        j = d.argmin(1)
        m = d[np.arange(len(lat)), j] < out
        out[m] = d[np.arange(len(lat)), j][m]
        idx[m] = j[m] + s
    return out, idx


def stations(path):
    m = pd.read_csv(path)
    m = m[m["base_name"].fillna("").str.strip() != ""].copy()
    m["line_status"] = np.where(m["base_name"].isin(LINE_1_2), "operational", "upcoming")
    pink = m["network"].fillna("").str.contains("Pink", case=False)
    uc = m["stage"].eq("osm_tagged_under_construction")
    m["metro_stage"] = np.where(m["line_status"] == "operational", 0, np.where(uc | pink, 3, 2))
    m["stage_source"] = "rule: Line 1/2 names = operational; OSM construction tag or Pink Line = 3; else 2 (verify)"
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpkg", required=True)
    ap.add_argument("--stations", default=str(OUT / "osm_pune_metro_stations.csv"))
    ap.add_argument("--rents", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(a.gpkg)

    # ---- metro stations with stage
    st = stations(a.stations)
    st.to_csv(OUT / "pune_metro_stations.csv", index=False)
    op, up = st[st.line_status == "operational"], st[st.line_status == "upcoming"]
    print(f"stations: {len(op)} operational (Line 1/2), {len(up)} upcoming")

    # ---- locality centres from OSM place nodes
    places = []
    for fclass, name, blob in con.execute("select fclass, name, geom from gis_osm_places_free"):
        if not name:
            continue
        k, xy = _geom(blob)
        if k == "point" and LON0 <= xy[0, 0] <= LON1 and LAT0 <= xy[0, 1] <= LAT1:
            places.append((fclass, name, xy[0, 1], xy[0, 0]))
    P = pd.DataFrame(places, columns=["fclass", "name", "lat", "lon"])
    rank = {"suburb": 0, "neighbourhood": 1, "locality": 2, "quarter": 3, "village": 4, "town": 5, "hamlet": 6}
    locs = []
    for loc, prior in LOCALITIES.items():
        names = [n.lower() for n in OSM_ALIASES.get(loc, [loc])]
        cand = P[P["name"].str.lower().isin(names)].copy()
        if cand.empty:
            cand = P[P["name"].str.lower().str.contains(names[0].split()[0])].copy()
        if cand.empty:
            la_, lo_ = FALLBACK[loc]
            print("  ! no OSM place for", loc, "- using approximate manual centre")
            locs.append({"micro_market": loc, "lat": la_, "lon": lo_, "osm_name": "", "osm_fclass": "manual (approx)",
                         "prior_rate_synthetic": prior})
            continue
        cand["r"] = cand["fclass"].map(rank).fillna(9)
        c = cand.sort_values("r").iloc[0]
        locs.append({"micro_market": loc, "lat": round(c.lat, 5), "lon": round(c.lon, 5), "osm_name": c["name"],
                     "osm_fclass": c.fclass, "prior_rate_synthetic": prior})
    L = pd.DataFrame(locs)
    L.to_csv(OUT / "pune_localities.csv", index=False)
    print(f"localities matched to OSM places: {len(L)}/{len(LOCALITIES)}")

    # ---- rivers (flood-exposure proxy)
    rv = []
    for fclass, name, blob in con.execute("select fclass, name, geom from gis_osm_waterways_free where fclass = 'river'"):
        k, xy = _geom(blob)
        if len(xy) and ((xy[:, 0] >= LON0) & (xy[:, 0] <= LON1) & (xy[:, 1] >= LAT0) & (xy[:, 1] <= LAT1)).any():
            rv.append(xy)
    R = np.vstack(rv)
    print("river vertices in box:", len(R))

    # ---- cells: start from the POI grid built earlier, add river + rent + stage-aware metro distances
    C = pd.read_csv(OUT / "pune_osm_cells.csv")
    la, lo = C["lat"].values, C["lon"].values
    d_op, i_op = nearest_dist(la, lo, op.lat.values, op.lon.values)
    d_up, i_up = nearest_dist(la, lo, up.lat.values, up.lon.values)
    C["dist_metro_operational_m"] = d_op.round(0)
    C["nearest_metro_operational"] = op.base_name.values[i_op]
    C["dist_metro_upcoming_m"] = d_up.round(0)
    C["nearest_metro_upcoming"] = up.base_name.values[i_up]
    C["upcoming_metro_stage"] = np.where(d_up < 2500, up.metro_stage.values[i_up], 0)
    d_r, _ = nearest_dist(la, lo, R[:, 1], R[:, 0])
    C["dist_river_m"] = d_r.round(0)
    C["flood_proxy"] = np.where(d_r < 300, 2, np.where(d_r < 800, 1, 0))

    # ---- real rents (Makaan apartments) -> median monthly rent per sq ft within 1.5 km
    r = pd.read_csv(a.rents)
    r = r[r["house_type"].str.contains("Apartment", na=False) & r["city"].eq("Pune")].copy()
    r["sqft"] = r["house_size"].str.replace(",", "").str.extract(r"([\d.]+)")[0].astype(float)
    r = r[(r.sqft >= 300) & (r.sqft <= 5000) & r.latitude.between(LAT0, LAT1) & r.longitude.between(LON0, LON1)]
    r["rent_psf"] = r["price"] / r["sqft"]
    q1, q99 = r.rent_psf.quantile([0.01, 0.99])
    r = r[r.rent_psf.between(q1, q99)]
    r = r.drop_duplicates(["latitude", "longitude", "price", "sqft"])
    r[["house_type", "sqft", "price", "rent_psf", "latitude", "longitude", "location", "verificationDate", "Status"]] \
        .to_csv(OUT / "pune_rents_clean.csv", index=False)
    rent = np.full(len(C), np.nan)
    nrent = np.zeros(len(C), int)
    rl, ro, rp = r.latitude.values, r.longitude.values, r.rent_psf.values
    for s in range(0, len(C), 500):
        d = hav(la[s:s + 500, None], lo[s:s + 500, None], rl[None], ro[None])
        for k in range(d.shape[0]):
            m = d[k] < 1500
            nrent[s + k] = m.sum()
            if m.sum() >= 5:
                rent[s + k] = np.median(rp[m])
    C["rent_psf_month_1500m"] = np.round(rent, 2)
    C["rent_listings_1500m"] = nrent
    C.to_csv(OUT / "pune_osm_cells.csv", index=False)
    print(f"rents: {len(r)} clean apartment rents; {np.isfinite(rent).mean():.0%} of cells have >=5 within 1.5 km")

    src = {
        "built_at": dt.datetime.now().isoformat(timespec="seconds"),
        "layers": {
            "poi_counts, bus_stops, osm stations, places, rivers": {
                "source": "Geofabrik india/western-zone-latest-free (OpenStreetMap)", "osm_as_of": "2026-11-21T21:20:51Z",
                "licence": "ODbL 1.0, (c) OpenStreetMap contributors", "origin": "real"},
            "metro stations": {"source": "OpenStreetMap via Overpass API, 2026-11-22", "licence": "ODbL 1.0",
                               "origin": "real", "caveat": "operational/upcoming stage assigned by STAGE RULE; verify"},
            "flood_proxy": {"source": "distance to OSM rivers (Mula, Mutha, Pavana, Indrayani...)", "origin": "real-derived",
                            "caveat": "proxy, not an official hazard map"},
            "rent_psf_month_1500m": {"source": "Kaggle 'Indian Rental House Price' (Makaan listings), file "
                                               "Indian_housing_Pune_data.csv", "origin": "real (asking rents)",
                                     "caveat": "licence unclear; listing dates relative ('Posted N years ago')"},
            "locality priors": {"origin": "synthetic", "note": "generator prior only; never shown as data"},
        },
    }
    (OUT / "sources.json").write_text(json.dumps(src, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
