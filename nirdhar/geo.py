"""Static geography for the two pilot markets.

Coordinates are approximate public locations of micro-market centres, metro stations
and employment hubs. Market rates are prototype priors (carpet-area basis, Sep 2026) used
only by the synthetic data generator; the trained models never read them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

AS_OF = "2026-09"  # valuation as-of month (latest index month)

# name: (city, lat, lon, prior market rate Rs/sqft carpet, guidance ratio, flood base 0..1)
LOCALITIES = {
    # ---------------- Hyderabad ----------------
    "Gachibowli":      ("Hyderabad", 17.4401, 78.3489, 9800, 0.62, 0.05),
    "Kondapur":        ("Hyderabad", 17.4600, 78.3570, 8800, 0.66, 0.05),
    "Madhapur":        ("Hyderabad", 17.4483, 78.3915, 10800, 0.60, 0.10),
    "HITEC City":      ("Hyderabad", 17.4435, 78.3772, 10400, 0.61, 0.05),
    "Kokapet":         ("Hyderabad", 17.3964, 78.3270, 11200, 0.55, 0.05),
    "Narsingi":        ("Hyderabad", 17.3880, 78.3560, 8900, 0.58, 0.10),
    "Manikonda":       ("Hyderabad", 17.4040, 78.3870, 7600, 0.64, 0.10),
    "Puppalaguda":     ("Hyderabad", 17.3920, 78.3740, 8100, 0.60, 0.10),
    "Nallagandla":     ("Hyderabad", 17.4710, 78.3110, 8200, 0.60, 0.05),
    "Tellapur":        ("Hyderabad", 17.4700, 78.2850, 7400, 0.58, 0.10),
    "Kukatpally":      ("Hyderabad", 17.4849, 78.4138, 7300, 0.68, 0.20),
    "Miyapur":         ("Hyderabad", 17.4968, 78.3726, 6500, 0.68, 0.15),
    "Bachupally":      ("Hyderabad", 17.5440, 78.3650, 5800, 0.65, 0.15),
    "Kompally":        ("Hyderabad", 17.5360, 78.4870, 6000, 0.64, 0.10),
    "Uppal":           ("Hyderabad", 17.4018, 78.5590, 6000, 0.68, 0.35),
    "Nagole":          ("Hyderabad", 17.3730, 78.5590, 5900, 0.66, 0.60),
    "LB Nagar":        ("Hyderabad", 17.3457, 78.5522, 6200, 0.67, 0.35),
    "Banjara Hills":   ("Hyderabad", 17.4156, 78.4347, 14500, 0.58, 0.05),
    "Jubilee Hills":   ("Hyderabad", 17.4325, 78.4070, 15500, 0.57, 0.05),
    "Begumpet":        ("Hyderabad", 17.4440, 78.4620, 9300, 0.66, 0.25),
    "Shamshabad":      ("Hyderabad", 17.2510, 78.4370, 5000, 0.60, 0.10),
}

# ---------------- Pune (primary market) ----------------
# Centres come from OpenStreetMap place nodes (data/real/pune_localities.csv, built by
# nirdhar.build_real_pune). Rates there are synthetic-generator priors only.
REAL_DIR = Path(__file__).resolve().parents[1] / "data" / "real"
REAL_PUNE = (REAL_DIR / "pune_localities.csv").exists() and (REAL_DIR / "pune_metro_stations.csv").exists()
_PUNE_FALLBACK = {
    "Hinjewadi": (18.5913, 73.7389, 8500), "Wakad": (18.5975, 73.7700, 9500), "Baner": (18.5590, 73.7868, 12500),
    "Aundh": (18.5580, 73.8075, 13500), "Kothrud": (18.5074, 73.8077, 13500), "Kharadi": (18.5515, 73.9348, 11000),
    "Viman Nagar": (18.5679, 73.9143, 12500), "Hadapsar": (18.5089, 73.9260, 8500), "Wagholi": (18.5793, 73.9787, 6500),
}


def _pune_localities():
    if REAL_PUNE:
        import csv
        with open(REAL_DIR / "pune_localities.csv", encoding="utf-8") as f:
            return {r["micro_market"]: ("Pune", float(r["lat"]), float(r["lon"]), float(r["prior_rate_synthetic"]), 0.66, 0.0)
                    for r in csv.DictReader(f)}
    return {k: ("Pune", a, b, p, 0.66, 0.0) for k, (a, b, p) in _PUNE_FALLBACK.items()}


LOCALITIES = {**_pune_localities(), **LOCALITIES}   # Pune first: it is the primary market

# Operational metro stations (approximate locations)
METRO_OPERATIONAL = {
    "Hyderabad": [
        ("Miyapur", 17.4969, 78.3729), ("JNTU College", 17.4930, 78.3910), ("KPHB Colony", 17.4870, 78.3990),
        ("Kukatpally", 17.4847, 78.4115), ("Balanagar", 17.4770, 78.4430), ("Ameerpet", 17.4375, 78.4482),
        ("Begumpet", 17.4436, 78.4610), ("Paradise", 17.4420, 78.4890), ("Parade Ground", 17.4430, 78.4990),
        ("Raidurg", 17.4410, 78.3770), ("HITEC City", 17.4505, 78.3810), ("Durgam Cheruvu", 17.4430, 78.3880),
        ("Peddamma Gudi", 17.4310, 78.4060), ("Jubilee Hills Check Post", 17.4290, 78.4120),
        ("Road No 5 Jubilee Hills", 17.4300, 78.4200), ("Uppal", 17.4020, 78.5590), ("Nagole", 17.3920, 78.5600),
        ("LB Nagar", 17.3480, 78.5480), ("MGBS", 17.3780, 78.4800),
    ],
    "Pune": [
        ("PCMC", 18.6290, 73.8000), ("Sant Tukaram Nagar", 18.6170, 73.8130), ("Shivajinagar", 18.5310, 73.8530),
        ("Civil Court", 18.5270, 73.8550), ("Vanaz", 18.5070, 73.8050), ("Anand Nagar", 18.5080, 73.8150),
        ("Ramwadi", 18.5550, 73.9070), ("Kalyani Nagar", 18.5480, 73.9010), ("Yerawada", 18.5520, 73.8850),
    ],
}

# Upcoming stations: (name, lat, lon, stage) stage 1=proposed, 2=approval pending, 3=under construction
METRO_UPCOMING = {
    "Hyderabad": [
        ("Gachibowli (Ph-2)", 17.4400, 78.3530, 2), ("Nanakramguda (Airport line)", 17.4170, 78.3410, 2),
        ("Narsingi (Airport line)", 17.3920, 78.3570, 2), ("Kokapet (Airport line)", 17.3990, 78.3300, 2),
        ("Shamshabad (Airport line)", 17.2470, 78.4300, 2), ("Chanda Nagar (Ph-2)", 17.4960, 78.3330, 1),
        ("BHEL (Ph-2)", 17.4950, 78.3000, 1), ("Tellapur (Ph-2)", 17.4720, 78.2900, 1),
    ],
    "Pune": [
        ("Hinjewadi Ph-3", 18.5850, 73.6980, 3), ("Infosys Ph-2", 18.5900, 73.7200, 3),
        ("Wakad Chowk", 18.5990, 73.7650, 3), ("Balewadi", 18.5770, 73.7820, 3),
        ("Baner", 18.5600, 73.7900, 3), ("University Circle", 18.5530, 73.8250, 3),
        ("Kharadi (proposed)", 18.5520, 73.9400, 1),
    ],
}

METRO_STAGE_LABEL = {0: "none nearby", 1: "proposed", 2: "approval pending", 3: "under construction"}

IT_HUBS = {
    "Hyderabad": [("HITEC City", 17.4474, 78.3762), ("Financial District", 17.4150, 78.3400),
                  ("Mindspace Madhapur", 17.4417, 78.3820)],
    "Pune": [("Hinjewadi Phase 1", 18.5910, 73.7380), ("EON Kharadi", 18.5500, 73.9460),
             ("Magarpatta IT Park", 18.5140, 73.9290)],
}

CITY_STATE = {"Hyderabad": "Telangana", "Pune": "Maharashtra"}

# Pune: real employment hubs (approximate centres of the main IT parks)
IT_HUBS["Pune"] = [("Hinjewadi Phase 1", 18.5910, 73.7380), ("Hinjewadi Phase 3", 18.5880, 73.6970),
                   ("EON IT Park Kharadi", 18.5500, 73.9460), ("Magarpatta Cybercity", 18.5140, 73.9290),
                   ("Commerzone Yerwada", 18.5560, 73.8930), ("Baner-Balewadi offices", 18.5650, 73.7790)]

if REAL_PUNE:   # replace the approximate Pune metro lists with real OSM stations
    import csv as _csv
    with open(REAL_DIR / "pune_metro_stations.csv", encoding="utf-8") as _f:
        _rows = list(_csv.DictReader(_f))
    METRO_OPERATIONAL["Pune"] = [(r["base_name"], float(r["lat"]), float(r["lon"]))
                                 for r in _rows if r["line_status"] == "operational"]
    METRO_UPCOMING["Pune"] = [(r["base_name"], float(r["lat"]), float(r["lon"]), int(r["metro_stage"]))
                              for r in _rows if r["line_status"] == "upcoming"]

DATA_PROVENANCE = {
    "Pune": {"localities": "OpenStreetMap place nodes" if REAL_PUNE else "approximate",
             "metro": "OpenStreetMap stations, stage by rule (verify)" if REAL_PUNE else "approximate",
             "poi_density, flood proxy, rents": "real OSM / Makaan layers" if REAL_PUNE else "synthetic",
             "transactions": "synthetic (awaiting e-Search Index-II sample)"},
    "Hyderabad": {"all layers": "approximate / synthetic (expansion market)"},
}


def haversine_m(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in metres."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


def nearest(lat, lon, points):
    """Distance (m) and index of the nearest point for arrays of lat/lon."""
    lat = np.atleast_1d(np.asarray(lat, float))
    lon = np.atleast_1d(np.asarray(lon, float))
    plat = np.array([p[1] for p in points])
    plon = np.array([p[2] for p in points])
    d = haversine_m(lat[:, None], lon[:, None], plat[None, :], plon[None, :])
    idx = d.argmin(axis=1)
    return d[np.arange(len(lat)), idx], idx


def cell_id(lat, lon, size_deg=0.0035):
    """Square grid cell (~0.38 km side) used as the H3-r9 stand-in location key.

    (The h3 library's native DLL is blocked by local application-control policy, so the
    prototype uses an equivalent fixed grid; the schema keeps the same role.)"""
    return f"g{int(np.floor(lat / size_deg))}_{int(np.floor(lon / size_deg))}"


def location_features(city, lat, lon):
    """Location feature vector computed for any coordinate (vectorised)."""
    lat = np.atleast_1d(np.asarray(lat, float))
    lon = np.atleast_1d(np.asarray(lon, float))
    dm, im = nearest(lat, lon, METRO_OPERATIONAL[city])
    up = METRO_UPCOMING[city]
    du, iu = nearest(lat, lon, up)
    stage = np.array([up[i][3] for i in iu])
    stage = np.where(du < 2500, stage, 0)
    dh, ih = nearest(lat, lon, IT_HUBS[city])
    return {
        "dist_metro_m": dm,
        "nearest_metro": [METRO_OPERATIONAL[city][i][0] for i in im],
        "dist_upcoming_metro_m": du,
        "nearest_upcoming_metro": [up[i][0] for i in iu],
        "metro_stage": stage,
        "dist_it_hub_m": dh,
        "nearest_it_hub": [IT_HUBS[city][i][0] for i in ih],
    }
