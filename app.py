import math
import os
import time
import uuid
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="Vaimo Comparable Map API", version="1.1.1")


# -------------------------
# Static
# -------------------------
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# -------------------------
# Config
# -------------------------
DATA_PATHS = [
    "data/immoscout24.csv",
    "data/immoscout.csv",
    "immoscout24.csv",
]

GEOCODE_CACHE: Dict[str, Dict] = {}
REQUESTS: Dict[str, Dict] = {}

DATA_CACHE = None


# -------------------------
# Helpers
# -------------------------
def normalize_text(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def enqueue_geocode(address: str) -> str:
    return normalize_text(address)


def cache_get(key: str):
    item = GEOCODE_CACHE.get(key)
    if not item:
        return None
    return item["lat"], item["lon"]


def cache_set_ok(key: str, query: str, lat: float, lon: float):
    GEOCODE_CACHE[key] = {
        "query": query,
        "lat": lat,
        "lon": lon,
        "ts": time.time(),
    }


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# -------------------------
# Photon Geocoder
# -------------------------
def geocode_photon(query: str):

    try:
        url = "https://photon.komoot.io/api/"
        params = {
            "q": query,
            "limit": 1,
            "lang": "de",
        }

        r = requests.get(url, params=params, timeout=10)

        print("PHOTON STATUS:", r.status_code, query)

        if r.status_code == 429:
            return {"error": "rate_limit"}

        r.raise_for_status()

        data = r.json()
        features = data.get("features", [])

        if not features:
            return {"error": "no_result"}

        lon, lat = features[0]["geometry"]["coordinates"]

        return {"lat": float(lat), "lon": float(lon)}

    except requests.exceptions.Timeout:
        return {"error": "timeout"}

    except Exception as e:
        return {"error": str(e)}


# -------------------------
# Dataset
# -------------------------
def load_dataset():

    global DATA_CACHE

    if DATA_CACHE is not None:
        return DATA_CACHE

    for path in DATA_PATHS:
        if os.path.exists(path):
            df = pd.read_csv(path)
            DATA_CACHE = df
            print("Loaded dataset:", path, len(df))
            return DATA_CACHE

    raise HTTPException(status_code=500, detail="CSV dataset not found")


# -------------------------
# Request Model
# -------------------------
class ValuationRequest(BaseModel):

    location: str
    area_sqm: float
    tolerance: float = Field(default=0.2)
    radius_km: float = Field(default=10.0)


# -------------------------
# Create Request
# -------------------------
@app.post("/valuation/request")
def create_request(req: ValuationRequest):

    key = enqueue_geocode(req.location)
    subj = cache_get(key)

    if not subj:

        q = req.location
        if "," not in q:
            q = f"{q}, Schweiz"

        geo = geocode_photon(q)

        print("GEOCODE RESULT:", geo)

        if "error" in geo:

            raise HTTPException(
                status_code=503,
                detail=f"Geocoding fehlgeschlagen: {geo['error']}"
            )

        cache_set_ok(key, q, geo["lat"], geo["lon"])
        subj = (geo["lat"], geo["lon"])

    request_id = "req_" + uuid.uuid4().hex[:10]

    REQUESTS[request_id] = {
        "location": req.location,
        "area_sqm": req.area_sqm,
        "tolerance": req.tolerance,
        "radius_km": req.radius_km,
        "subject_lat": subj[0],
        "subject_lon": subj[1],
    }

    return {
        "request_id": request_id,
        "map_url": f"https://map-skur.onrender.com/map?request_id={request_id}",
        "subject_lat": subj[0],
        "subject_lon": subj[1],
    }


# -------------------------
# Debug Request
# -------------------------
@app.get("/debug/request/{request_id}")
def debug_request(request_id: str):

    r = REQUESTS.get(request_id)

    if not r:
        raise HTTPException(status_code=404, detail="unknown request")

    return r


# -------------------------
# Map Page
# -------------------------
@app.get("/map", response_class=HTMLResponse)
def map_page(request_id: str = ""):

    html = open("map.html").read()

    return html.replace("__REQUEST_ID__", request_id)


# -------------------------
# Map Prices
# -------------------------
@app.get("/map/prices")
def map_prices(bbox: str = Query(...), request_id: str = Query("")):

    try:
        west, south, east, north = [float(x) for x in bbox.split(",")]
    except:
        raise HTTPException(status_code=400, detail="invalid bbox")

    df = load_dataset()

    filt = REQUESTS.get(request_id)

    if filt:

        target = filt["area_sqm"]
        tol = filt["tolerance"]

        lo = target * (1 - tol)
        hi = target * (1 + tol)

        df = df[(df["area_sqm"] >= lo) & (df["area_sqm"] <= hi)]

    features = []

    for _, row in df.iterrows():

        addr = str(row["address"])

        key = enqueue_geocode(addr)
        coord = cache_get(key)

        if not coord:

            geo = geocode_photon(addr)

            if "error" in geo:
                continue

            cache_set_ok(key, addr, geo["lat"], geo["lon"])
            coord = (geo["lat"], geo["lon"])

        lat, lon = coord

        if not (west <= lon <= east and south <= lat <= north):
            continue

        if filt:

            subj_lat = filt["subject_lat"]
            subj_lon = filt["subject_lon"]
            rad = filt["radius_km"]

            if haversine_km(subj_lat, subj_lon, lat, lon) > rad:
                continue

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "title": row["title"],
                "address": row["address"],
                "price_chf": int(row["price_chf"]),
                "area_sqm": float(row["area_sqm"]),
                "url": row.get("url", "")
            }
        })

        if len(features) > 80:
            break

    return JSONResponse({
        "type": "FeatureCollection",
        "features": features
    })
