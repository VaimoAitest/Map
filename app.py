import json
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field


# -------------------------
# Config / Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MAP_URL = os.getenv("BASE_MAP_URL", "https://map-skur.onrender.com")
GEOCODE_CACHE_FILE = os.path.join(BASE_DIR, "geocode_cache.json")

DATA_CANDIDATES = [
    os.path.join(BASE_DIR, "data", "immoscout24_master.csv"),
    os.path.join(BASE_DIR, "immoscout24_master.csv"),
    os.path.join(BASE_DIR, "data", "immoscout_master.csv"),
    os.path.join(BASE_DIR, "immoscout_master.csv"),
]


# -------------------------
# In-memory state
# -------------------------
REQUESTS: Dict[str, Dict] = {}
GEOCODE_CACHE: Dict[str, Dict] = {}

DATA_CACHE = None
DATA_CACHE_ERROR = None


# -------------------------
# Helpers
# -------------------------
def normalize_text(s: str) -> str:
    return " ".join(str(s or "").strip().lower().split())


def cache_key(address: str) -> str:
    return normalize_text(address)


def load_geocode_cache() -> None:
    global GEOCODE_CACHE
    if os.path.exists(GEOCODE_CACHE_FILE):
        try:
            with open(GEOCODE_CACHE_FILE, "r", encoding="utf-8") as f:
                GEOCODE_CACHE = json.load(f)
            print(f"GEOCODE CACHE LOADED: {len(GEOCODE_CACHE)} entries")
        except Exception as e:
            print("GEOCODE CACHE LOAD ERROR:", repr(e))
            GEOCODE_CACHE = {}
    else:
        GEOCODE_CACHE = {}


def save_geocode_cache() -> None:
    try:
        tmp_path = GEOCODE_CACHE_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(GEOCODE_CACHE, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, GEOCODE_CACHE_FILE)
    except Exception as e:
        print("GEOCODE CACHE SAVE ERROR:", repr(e))


def cache_get(key: str) -> Optional[Tuple[float, float]]:
    item = GEOCODE_CACHE.get(key)
    if not item:
        return None
    try:
        return float(item["lat"]), float(item["lon"])
    except Exception:
        return None


def cache_set_ok(key: str, query: str, lat: float, lon: float) -> None:
    GEOCODE_CACHE[key] = {
        "query": query,
        "lat": float(lat),
        "lon": float(lon),
        "ts": int(time.time()),
    }
    save_geocode_cache()


def geocode_nominatim_ch(query: str) -> Optional[Tuple[float, float]]:
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": query,
            "format": "jsonv2",
            "limit": 1,
            "countrycodes": "ch",
            "addressdetails": 0,
        }
        headers = {
            "User-Agent": "VaimoAI/1.0"
        }

        r = requests.get(url, params=params, headers=headers, timeout=15)
        print("NOMINATIM STATUS:", r.status_code, "QUERY:", query)
        r.raise_for_status()

        data = r.json()
        if not data:
            print("NOMINATIM NO RESULT:", query)
            return None

        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
        print("NOMINATIM HIT:", query, lat, lon)
        return lat, lon

    except Exception as e:
        print("NOMINATIM ERROR:", query, repr(e))
        return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# -------------------------
# Data loading
# -------------------------
def load_dataset_safe() -> pd.DataFrame:
    global DATA_CACHE, DATA_CACHE_ERROR

    if DATA_CACHE is not None:
        return DATA_CACHE

    found = None
    for path in DATA_CANDIDATES:
        if os.path.exists(path):
            found = path
            break

    if not found:
        DATA_CACHE_ERROR = f"Keine CSV gefunden. Erwartet eine von: {', '.join(DATA_CANDIDATES)}"
        raise HTTPException(status_code=500, detail=DATA_CACHE_ERROR)

    try:
        df = pd.read_csv(found, sep=None, engine="python")

        lower_map = {str(c).strip().lower(): c for c in df.columns}

        def col(*names):
            for n in names:
                if n.lower() in lower_map:
                    return lower_map[n.lower()]
            return None

        title_col = col("title", "titel")
        address_col = col("address", "adresse")
        price_col = col("price_chf", "price", "preis", "preis_chf")
        area_col = col(
            "area_sqm",
            "living_area_sqm",
            "living_area",
            "fläche",
            "flaeche",
            "wohnfläche",
            "wohnflaeche",
        )
        url_col = col("url", "link")
        lat_col = col("lat", "latitude")
        lon_col = col("lon", "lng", "longitude")
        address_key_col = col("address_key")

        required = [title_col, address_col, price_col, area_col, lat_col, lon_col]
        if not all(required):
            raise ValueError(
                "CSV braucht mindestens diese Spalten: title, address, price_chf, area_sqm, lat, lon"
            )

        clean = pd.DataFrame({
            "title": df[title_col].astype(str).fillna(""),
            "address": df[address_col].astype(str).fillna(""),
            "price_chf": pd.to_numeric(df[price_col], errors="coerce"),
            "area_sqm": pd.to_numeric(df[area_col], errors="coerce"),
            "url": df[url_col].astype(str).fillna("") if url_col else "",
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
            "address_key": (
                df[address_key_col].astype(str).fillna("")
                if address_key_col
                else df[address_col].astype(str).apply(normalize_text)
            ),
        })

        clean = clean.dropna(subset=["price_chf", "area_sqm", "lat", "lon"])
        clean = clean[clean["price_chf"] > 0]
        clean = clean[clean["area_sqm"] > 0]

        # ungefähr Schweiz
        clean = clean[(clean["lat"] >= 45) & (clean["lat"] <= 49)]
        clean = clean[(clean["lon"] >= 5) & (clean["lon"] <= 12)]

        clean = clean.drop_duplicates(
            subset=["title", "address", "price_chf", "area_sqm", "lat", "lon"]
        ).reset_index(drop=True)

        DATA_CACHE = clean
        DATA_CACHE_ERROR = None
        print("DATASET LOADED:", found, len(clean))
        return DATA_CACHE

    except Exception as e:
        DATA_CACHE_ERROR = str(e)
        raise HTTPException(status_code=500, detail=f"CSV konnte nicht geladen werden: {e}")


def find_location_from_dataset(query: str) -> Optional[Tuple[float, float]]:
    try:
        df = load_dataset_safe()
        q = normalize_text(query)

        if not q:
            return None

        addr_norm = df["address"].astype(str).apply(normalize_text)

        matches = df[addr_norm.str.contains(q, na=False)]

        if matches.empty:
            # Zusätzliche grobe Suche für Ortsteile / PLZ-Fragmente
            tokens = [t for t in q.replace(",", " ").split() if len(t) >= 2]
            if tokens:
                mask = pd.Series([True] * len(df))
                for token in tokens:
                    mask = mask & addr_norm.str.contains(token, na=False)
                matches = df[mask]

        if matches.empty:
            print("DATASET NO MATCH:", query)
            return None

        lat = float(matches["lat"].median())
        lon = float(matches["lon"].median())
        print("DATASET MATCH:", query, lat, lon, "rows:", len(matches))
        return lat, lon

    except Exception as e:
        print("DATASET LOOKUP ERROR:", query, repr(e))
        return None


# -------------------------
# Lifespan
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_geocode_cache()
    try:
        load_dataset_safe()
    except Exception as e:
        print("DATA PRELOAD ERROR:", repr(e))
    yield


# -------------------------
# App
# -------------------------
app = FastAPI(
    title="Vaimo Comparable Map API",
    version="3.0.0",
    lifespan=lifespan,
)


# -------------------------
# Models
# -------------------------
class ValuationRequest(BaseModel):
    location: str = Field(..., description="Ort oder volle Adresse (Schweiz)")
    area_sqm: float = Field(..., gt=0)
    tolerance: float = Field(0.2, ge=0.05, le=0.6)
    radius_km: float = Field(10.0, ge=0.5, le=50.0)


# -------------------------
# API routes
# -------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "Vaimo Comparable Map API"}


@app.get("/health")
def health():
    rows = 0
    try:
        rows = len(load_dataset_safe())
    except Exception:
        pass

    return {
        "ok": True,
        "dataset_rows": rows,
        "csv_error": DATA_CACHE_ERROR,
        "requests_in_memory": len(REQUESTS),
        "geocode_cache_entries": len(GEOCODE_CACHE),
    }


@app.post("/valuation/request")
def create_request(req: ValuationRequest):
    key = cache_key(req.location)
    subj = cache_get(key)

    # 1) Cache
    if not subj:
        # 2) Erst in eigener CSV suchen
        subj = find_location_from_dataset(req.location)
        if subj:
            cache_set_ok(key, req.location, subj[0], subj[1])

    # 3) Nur wenn nicht gefunden: extern geocoden
    if not subj:
        raw = req.location.strip()
        candidates = [
            raw,
            f"{raw}, Schweiz",
            f"{raw}, Switzerland",
        ]

        seen = set()
        candidates = [c for c in candidates if c and not (c in seen or seen.add(c))]

        for query in candidates:
            coord = geocode_nominatim_ch(query)
            if coord:
                cache_set_ok(key, query, coord[0], coord[1])
                subj = coord
                break
            time.sleep(0.8)

    if not subj:
        raise HTTPException(
            status_code=503,
            detail=(
                "Geocoding fehlgeschlagen. Bitte gib den Ort genauer ein, "
                "z. B. 'Berg TG, Schweiz' oder eine vollständige Adresse."
            )
        )

    request_id = "req_" + uuid.uuid4().hex[:10]
    REQUESTS[request_id] = {
        "location": req.location,
        "area_sqm": float(req.area_sqm),
        "tolerance": float(req.tolerance),
        "radius_km": float(req.radius_km),
        "subject_lat": float(subj[0]),
        "subject_lon": float(subj[1]),
        "ts": int(time.time()),
    }

    return {
        "request_id": request_id,
        "map_url": f"{BASE_MAP_URL}/map?request_id={request_id}",
        "subject_lat": float(subj[0]),
        "subject_lon": float(subj[1]),
    }


@app.get("/valuation/map/{request_id}")
def get_comparable_map(request_id: str):
    r = REQUESTS.get(request_id)
    if not r:
        raise HTTPException(status_code=404, detail="unknown request_id")

    url = f"{BASE_MAP_URL}/map?request_id={request_id}"
    return {
        "request_id": request_id,
        "map_url": url,
        "embed_url": url,
        "image_url": None,
    }


@app.get("/debug/request/{request_id}")
def debug_request(request_id: str):
    r = REQUESTS.get(request_id)
    if not r:
        raise HTTPException(status_code=404, detail="unknown request_id")
    return r


# -------------------------
# HTML
# -------------------------
MAP_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Vaimo – Comparable Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <style>
    html, body, #map {
      height: 100%;
      margin: 0;
      background: #0b0b0b;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    }
    .topbar {
      position: absolute;
      top: 14px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 1000;
      width: min(980px, calc(100% - 28px));
      background: rgba(15,15,17,0.55);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 22px;
      padding: 12px 12px;
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      box-shadow: 0 16px 40px rgba(0,0,0,0.35);
    }
    .row {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .pill {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border-radius: 16px;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.12);
    }
    .pill label {
      font-size: 12px;
      color: rgba(255,255,255,0.70);
      white-space: nowrap;
    }
    .pill input {
      width: 200px;
      border: none;
      outline: none;
      background: transparent;
      color: white;
      font-size: 14px;
    }
    .pill input[type="number"] {
      width: 115px;
    }
    .btn {
      border: none;
      cursor: pointer;
      padding: 10px 14px;
      border-radius: 16px;
      font-weight: 800;
      background: rgba(255,255,255,0.92);
      color: #0b0b0b;
    }
    .btn:disabled {
      opacity: 0.65;
      cursor: default;
    }
    .ghost {
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(255,255,255,0.06);
      color: rgba(255,255,255,0.92);
      padding: 10px 12px;
      border-radius: 16px;
      cursor: pointer;
      font-weight: 700;
    }
    .advanced {
      display: none;
      margin-top: 10px;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .hint {
      margin-top: 8px;
      font-size: 12px;
      color: rgba(255,255,255,0.70);
      padding: 0 8px;
    }
    .price-label {
      background: rgba(255,255,255,0.96);
      border-radius: 16px;
      padding: 6px 10px;
      font-weight: 800;
      box-shadow: 0 8px 24px rgba(0,0,0,0.22);
      white-space: nowrap;
    }
  </style>
</head>
<body>
  <div class="topbar">
    <div class="row">
      <div class="pill"><label>Ort</label><input id="loc" placeholder="z.B. Zürich / Berg TG / Adresse"/></div>
      <div class="pill"><label>m²</label><input id="sqm" type="number" placeholder="z.B. 100"/></div>
      <button class="btn" id="apply" type="button">Request</button>
      <button class="ghost" id="toggleAdv" type="button">Erweitert</button>
    </div>

    <div class="advanced" id="adv">
      <div class="pill"><label>Tol</label><input id="tol" type="number" step="0.05" value="0.20"/></div>
      <div class="pill"><label>km</label><input id="rad" type="number" step="0.5" value="10.0"/></div>
    </div>

    <div class="hint" id="hint">Daten mit gespeicherten Koordinaten geladen.</div>
  </div>

  <div id="map"></div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const map = L.map('map').setView([47.3769, 8.5417], 12);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap'
    }).addTo(map);

    let layer;
    let requestId = "__REQUEST_ID__" || "";
    let isLoading = false;

    function bboxStr() {
      const b = map.getBounds();
      return [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].join(",");
    }

    async function loadData() {
      if (isLoading) return;
      isLoading = true;

      try {
        const url = requestId
          ? `/map/prices?request_id=${encodeURIComponent(requestId)}&bbox=${encodeURIComponent(bboxStr())}`
          : `/map/prices?bbox=${encodeURIComponent(bboxStr())}`;

        const res = await fetch(url);
        const geo = await res.json();

        if (layer) layer.remove();

        layer = L.geoJSON(geo, {
          pointToLayer: (feature, latlng) => {
            const p = feature.properties || {};
            const html = `<div class="price-label">${(p.price_chf || 0).toLocaleString('de-CH')} CHF</div>`;
            const icon = L.divIcon({ html: html, className: "", iconSize: [1, 1] });

            return L.marker(latlng, { icon }).bindPopup(
              `<b>${p.title || ''}</b><br/>` +
              `${(p.price_chf || 0).toLocaleString('de-CH')} CHF • ${(p.area_sqm || 0).toFixed(0)} m²<br/>` +
              `${p.address || ''}<br/>` +
              (p.url ? `<a href="${p.url}" target="_blank">Link</a>` : "")
            );
          }
        }).addTo(map);
      } catch (e) {
        console.error("loadData error:", e);
      } finally {
        isLoading = false;
      }
    }

    let moveTimer = null;
    function scheduleLoad() {
      clearTimeout(moveTimer);
      moveTimer = setTimeout(() => {
        loadData();
      }, 150);
    }

    map.on('moveend zoomend', scheduleLoad);

    document.getElementById('toggleAdv').addEventListener('click', () => {
      const adv = document.getElementById('adv');
      adv.style.display = (adv.style.display === 'flex') ? 'none' : 'flex';
    });

    document.getElementById('apply').addEventListener('click', async () => {
      const btn = document.getElementById('apply');
      const hint = document.getElementById('hint');
      btn.disabled = true;
      const old = btn.textContent;
      btn.textContent = "Loading…";

      try {
        const loc = document.getElementById('loc').value.trim();
        const sqm = parseFloat(document.getElementById('sqm').value);
        let tol = parseFloat(document.getElementById('tol').value);
        let rad = parseFloat(document.getElementById('rad').value);

        if (!isFinite(tol) || tol < 0.05) tol = 0.20;
        if (!isFinite(rad) || rad < 0.5) rad = 10.0;

        if (!loc) {
          alert("Bitte Ort/Adresse eingeben.");
          return;
        }
        if (!isFinite(sqm) || sqm <= 0) {
          alert("Bitte m² eingeben.");
          return;
        }

        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 15000);

        const res = await fetch("/valuation/request", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            location: loc,
            area_sqm: sqm,
            tolerance: tol,
            radius_km: rad
          }),
          signal: controller.signal
        });

        clearTimeout(timeout);

        const data = await res.json().catch(() => ({}));

        if (!res.ok) {
          const msg = (typeof data.detail === "string")
            ? data.detail
            : JSON.stringify(data.detail || data);
          alert(msg || "Fehler beim Request");
          return;
        }

        requestId = data.request_id;
        history.replaceState(null, "", `/map?request_id=${requestId}`);

        if (data.subject_lat && data.subject_lon) {
          map.setView([data.subject_lat, data.subject_lon], 13);
        }

        hint.textContent = `Request geladen: ${requestId}`;
        await loadData();
      } catch (e) {
        alert(
          e.name === "AbortError"
            ? "Timeout (15s) – Free Render kann schlafen. Nochmal klicken."
            : ("Error: " + e.message)
        );
      } finally {
        btn.disabled = false;
        btn.textContent = old;
      }
    });

    async function init() {
      const hint = document.getElementById("hint");

      try {
        if (requestId) {
          const res = await fetch(`/debug/request/${encodeURIComponent(requestId)}`);
          if (res.ok) {
            const data = await res.json();

            if (data.location) document.getElementById("loc").value = data.location;
            if (data.area_sqm) document.getElementById("sqm").value = data.area_sqm;
            if (data.tolerance) document.getElementById("tol").value = data.tolerance;
            if (data.radius_km) document.getElementById("rad").value = data.radius_km;

            if (data.subject_lat && data.subject_lon) {
              map.setView([data.subject_lat, data.subject_lon], 13);
            }

            hint.textContent = `Request geladen: ${requestId}`;
          } else {
            hint.textContent = "Request-ID gefunden, aber nicht mehr im Speicher.";
          }
        }

        await loadData();
      } catch (e) {
        console.error(e);
        hint.textContent = "Fehler beim Initialisieren der Map.";
      }
    }

    init();
  </script>
</body>
</html>
"""


@app.get("/map", response_class=HTMLResponse)
def map_page(request_id: str = ""):
    return MAP_HTML.replace("__REQUEST_ID__", request_id or "")


@app.get("/map/prices")
def map_prices(
    bbox: str = Query(...),
    request_id: str = Query("")
):
    try:
        west, south, east, north = [float(x) for x in bbox.split(",")]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bbox")

    df = load_dataset_safe()
    filt = REQUESTS.get(request_id) if request_id else None

    if filt:
        target = float(filt["area_sqm"])
        tol = float(filt["tolerance"])
        lo = target * (1 - tol)
        hi = target * (1 + tol)
        df = df[(df["area_sqm"] >= lo) & (df["area_sqm"] <= hi)]

    # zuerst bbox, dann optional radius, dann sortieren, dann limitieren
    df = df[
        (df["lon"] >= west) &
        (df["lon"] <= east) &
        (df["lat"] >= south) &
        (df["lat"] <= north)
    ]

    if filt and not df.empty:
        subj_lat = float(filt["subject_lat"])
        subj_lon = float(filt["subject_lon"])
        rad = float(filt["radius_km"])

        df = df.copy()
        df["distance_km"] = df.apply(
            lambda row: haversine_km(subj_lat, subj_lon, float(row["lat"]), float(row["lon"])),
            axis=1
        )
        df = df[df["distance_km"] <= rad]
        df = df.sort_values(["distance_km", "price_chf"])
    else:
        df = df.sort_values(["price_chf", "area_sqm"])

    df = df.head(80)

    features = []
    for _, row in df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row["lon"]), float(row["lat"])]
            },
            "properties": {
                "title": row["title"],
                "address": row["address"],
                "price_chf": int(float(row["price_chf"])),
                "area_sqm": float(row["area_sqm"]),
                "url": row.get("url", "")
            }
        })

    return JSONResponse({"type": "FeatureCollection", "features": features})
