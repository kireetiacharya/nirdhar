"""Nirdhar valuation API + web UI.  Run:  python -m uvicorn app:app --port 8000"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nirdhar.service import Valuer, _json_default

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Nirdhar valuation API", version="0.1.0")
valuer = Valuer()


def clean(obj):
    """numpy -> plain JSON types."""
    return json.loads(json.dumps(obj, default=_json_default))


class ValuationRequest(BaseModel):
    city: Literal["Pune", "Hyderabad"]
    micro_market: str
    project_id: Optional[str] = None
    property_type: Literal["apartment", "builder_floor", "independent_house", "plot", "commercial"] = "apartment"
    commercial_kind: Optional[Literal["shop", "office"]] = None
    bhk: int = Field(ge=1, le=6)
    carpet_area: float = Field(gt=150, lt=15000, description="Area in sq ft, in the basis given by area_basis")
    area_basis: Literal["carpet", "builtup", "superbuiltup"] = "carpet"
    floor: int = Field(ge=0, le=80)
    total_floors: Optional[int] = Field(default=None, ge=1, le=90)
    age_years: Optional[float] = Field(default=None, ge=0, le=80)
    parking: Optional[bool] = None
    facing: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    asking_price: Optional[float] = Field(default=None, gt=0)
    agreement_value: Optional[float] = Field(default=None, gt=0)
    loan_amount: Optional[float] = Field(default=None, gt=0)


@app.get("/api/meta")
def meta():
    return JSONResponse(clean(valuer.meta()))


@app.get("/api/model")
def model():
    return JSONResponse(clean(valuer.model_info()))


@app.post("/api/valuations")
def create_valuation(req: ValuationRequest):
    try:
        return JSONResponse(clean(valuer.value(req.model_dump())))
    except ValueError as ex:
        raise HTTPException(400, str(ex))


@app.get("/api/valuations/{vid}")
def get_valuation(vid: str):
    p = ROOT / "audit" / f"{vid}.json"
    if not p.exists() or "/" in vid or "\\" in vid:
        raise HTTPException(404, "Unknown valuation id")
    return JSONResponse(json.loads(p.read_text()))


@app.post("/api/valuations/{vid}/replay")
def replay(vid: str):
    if "/" in vid or "\\" in vid or ".." in vid:
        raise HTTPException(400, "bad id")
    try:
        r = valuer.replay(vid)
    except FileNotFoundError:
        raise HTTPException(404, "Unknown valuation id")
    r.pop("original", None)
    return JSONResponse(clean(r))


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
