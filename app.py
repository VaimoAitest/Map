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

app = FastAPI(title="Vaimo Comparable Map API", version="1.1.0")

# -------------------------
# Static files
# -------------------------
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------
# Config
# -------------------------
DATA_PATHS = [
    "data/immoscout.csv",
    "data/immoscout24.csv",
    "data/immoscout24 (2).csv",
]

GEOCODE_TTL_SECONDS = 60 * 60 * 24 * 14  # 14 Tage
REQUEST_TTL_SECONDS = 60 * 60 * 12        # 12 Stunden

# -------------------------
# In-memory caches
# -------------------------
GEOCODE_CACHE: Dict[str, Dict] = {}
REQUESTS: Dict[str, Dict] = {}

DATA_CACHE: Optional[pd.DataFrame] = None
DATA_CACHE_ERROR: Optional[str] = None


# -------------------------
# Helpers
# -------------------------
def normalize_text(s: str) -> str:
    return " ".join(str(s or "").strip().lower().split())


def enqueue_geocode(address: str) -> str:
    return normalize_text(address)


def cache_get(key: str) -> Optional[Tuple[float, float]]:
    item = GEOCODE_CACHE.get(key)
    if not item:
        return None
    if (time.time() - item["ts"]) > GEOCODE_TTL_SECONDS:
        return None
    return (item["lat"], item["lon"])


def cache_set_ok(key: str, query: str, lat: float, lon: float) -> None:
    GEOCODE_CACHE[key] = {
        "query": query,
        "lat": float(lat),
        "lon": float(lon),
        "ts": int(time.time()),
    }


def geocode_photon_ch(query: str) -> Optional[Tuple[float, float]]:
    try:
        url = "https://photon.komoot.io/api/"
        params = {
            "q": query,
            "limit": 1,
            "lang": "de",
        }
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        features = data.get("features", [])
        if not features:
            return None
        coords = features[0]["geometry"]["coordinates"]
        lon, lat = coords[0], coords[1]
        return float(lat), float(lon)
    except Exception:
        return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def cleanup_requests() -> None:
    now = int(time.time())
    stale = [
        req_id for req_id, payload in REQUESTS.items()
        if (now - int(payload.get("ts", now))) > REQUEST_TTL_SECONDS
    ]
    for req_id in stale:
        REQUESTS.pop(req_id, None)


def guess_column(df: pd.DataFrame, candidates) -> Optional[str]:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_dataset_safe() -> pd.DataFrame:
    global DATA_CACHE, DATA_CACHE_ERROR

    if DATA_CACHE is not None:
        return DATA_CACHE

    last_error = None
    found_path = None

    for path in DATA_PATHS:
        if os.path.exists(path):
            found_path = path
            break

    if not found_path:
        DATA_CACHE_ERROR = f"Keine CSV gefunden. Erwartet in: {', '.join(DATA_PATHS)}"
        raise HTTPException(status_code=500, detail=DATA_CACHE_ERROR)

    try:
        df = pd.read_csv(found_path)

        title_col = guess_column(df, ["title", "titel"])
        address_col = guess_column(df, ["address", "adresse"])
        price_col = guess_column(df, ["price_chf", "price", "preis", "preis_chf"])
        area_col = guess_column(df, ["area_sqm", "living_area_sqm", "living_area", "fläche", "flaeche", "wohnfläche", "wohnflaeche"])
        url_col = guess_column(df, ["url", "link"])

        missing = [name for name, col in {
            "title": title_col,
            "address": address_col,
            "price_chf": price_col,
            "area_sqm": area_col,
        }.items() if col is None]

        if missing:
            raise ValueError(f"Fehlende CSV-Spalten: {', '.join(missing)}")

        clean = pd.DataFrame({
            "title": df[title_col].astype(str).fillna(""),
            "address": df[address_col].astype(str).fillna(""),
            "price_chf": pd.to_numeric(df[price_col], errors="coerce"),
            "area_sqm": pd.to_numeric(df[area_col], errors="coerce"),
            "url": df[url_col].astype(str).fillna("") if url_col else "",
        })

        clean = clean.dropna(subset=["price_chf", "area_sqm"])
        clean = clean[clean["price_chf"] > 0]
        clean = clean[clean["area_sqm"] > 0]
        clean = clean.drop_duplicates(subset=["title", "address", "price_chf", "area_sqm"]).reset_index(drop=True)

        DATA_CACHE = clean
        DATA_CACHE_ERROR = None
        return DATA_CACHE

    except Exception as e:
        last_error = str(e)
        DATA_CACHE_ERROR = last_error
        raise HTTPException(status_code=500, detail=f"CSV konnte nicht geladen werden: {last_error}")


# -------------------------
# Models
# -------------------------
class ValuationRequest(BaseModel):
    location: str = Field(..., description="Ort oder volle Adresse (Schweiz)")
    area_sqm: float = Field(..., gt=0)
    tolerance: float = Field(0.2, ge=0.05, le=0.6)
    radius_km: float = Field(10.0, ge=0.5, le=25.0)


# -------------------------
# API
# -------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "Vaimo Comparable Map API"}


@app.get("/health")
def health():
    try:
        df = load_dataset_safe()
        count = int(len(df))
    except Exception:
        count = 0

    return {
        "ok": True,
        "dataset_rows": count,
        "csv_error": DATA_CACHE_ERROR,
        "requests_in_memory": len(REQUESTS),
    }


@app.post("/valuation/request")
def create_request(req: ValuationRequest):
    cleanup_requests()

    key = enqueue_geocode(req.location)
    subj = cache_get(key)

    if not subj:
        q2 = req.location.strip()
        q2 = q2 if ("," in q2) else f"{q2}, Schweiz"
        coord = geocode_photon_ch(q2)
        if coord:
            cache_set_ok(key, q2, coord[0], coord[1])
            subj = coord

    if not subj:
        raise HTTPException(
            status_code=503,
            detail="Geocoding ist gerade limitiert. Bitte in 30–60 Sekunden erneut versuchen."
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
        "map_url": f"https://map-skur.onrender.com/map?request_id={request_id}",
        "subject_lat": float(subj[0]),
        "subject_lon": float(subj[1]),
    }


@app.get("/valuation/map/{request_id}")
def get_comparable_map(request_id: str):
    cleanup_requests()

    r = REQUESTS.get(request_id)
    if not r:
        raise HTTPException(status_code=404, detail="unknown request_id")

    url = f"https://map-skur.onrender.com/map?request_id={request_id}"
    return {
        "request_id": request_id,
        "map_url": url,
        "embed_url": url,
        "image_url": None,
    }


@app.get("/debug/request/{request_id}")
def debug_request(request_id: str):
    cleanup_requests()

    r = REQUESTS.get(request_id)
    if not r:
        raise HTTPException(status_code=404, detail="unknown request_id")
    return r


# -------------------------
# Frontend HTML
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
      height:100%;
      margin:0;
      background:#0b0b0b;
      font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;
    }
    .topbar {
      position:absolute;
      top:14px;
      left:50%;
      transform:translateX(-50%);
      z-index:1000;
      width:min(980px, calc(100% - 28px));
      background:rgba(15,15,17,0.55);
      border:1px solid rgba(255,255,255,0.12);
      border-radius:22px;
      padding:12px 12px;
      backdrop-filter:blur(14px);
      -webkit-backdrop-filter:blur(14px);
      box-shadow:0 16px 40px rgba(0,0,0,0.35);
    }
    .row {
      display:flex;
      gap:10px;
      align-items:center;
      flex-wrap:wrap;
    }
    .brand {
      display:flex;
      align-items:center;
      padding:6px 10px;
      border-radius:16px;
      background:rgba(255,255,255,0.06);
      border:1px solid rgba(255,255,255,0.10);
    }
    .brand img {
      width:84px;
      height:auto;
      display:block;
      background:transparent;
      padding:0;
      border-radius:0;
    }
    .pill {
      display:flex;
      align-items:center;
      gap:8px;
      padding:8px 10px;
      border-radius:16px;
      background:rgba(255,255,255,0.06);
      border:1px solid rgba(255,255,255,0.12);
    }
    .pill label {
      font-size:12px;
      color:rgba(255,255,255,0.70);
      white-space:nowrap;
    }
    .pill input {
      width:200px;
      border:none;
      outline:none;
      background:transparent;
      color:white;
      font-size:14px;
    }
    .pill input[type="number"] {
      width:115px;
    }
    .btn {
      border:none;
      cursor:pointer;
      padding:10px 14px;
      border-radius:16px;
      font-weight:800;
      background:rgba(255,255,255,0.92);
      color:#0b0b0b;
    }
    .btn:disabled {
      opacity:0.65;
      cursor:default;
    }
    .ghost {
      border:1px solid rgba(255,255,255,0.14);
      background:rgba(255,255,255,0.06);
      color:rgba(255,255,255,0.92);
      padding:10px 12px;
      border-radius:16px;
      cursor:pointer;
      font-weight:700;
    }
    .advanced {
      display:none;
      margin-top:10px;
      gap:10px;
      align-items:center;
      flex-wrap:wrap;
    }
    .hint {
      margin-top:8px;
      font-size:12px;
      color:rgba(255,255,255,0.70);
      padding:0 8px;
    }
    .price-label {
      background:rgba(255,255,255,0.96);
      border-radius:16px;
      padding:6px 10px;
      font-weight:800;
      box-shadow:0 8px 24px rgba(0,0,0,0.22);
      white-space:nowrap;
    }
  </style>
</head>
<body>
  <div class="topbar">
    <div class="row">
      <div class="brand">
        <img src="/static/vaimo.png" onerror="this.style.display='none'" alt="VAIMO"/>
      </div>

      <div class="pill">
        <label>Ort</label>
        <input id="loc" placeholder="z.B. Zürich / Berg TG / Adresse"/>
      </div>

      <div class="pill">
        <label>m²</label>
        <input id="sqm" type="number" placeholder="z.B. 100"/>
      </div>

      <button class="btn" id="apply" type="button">Request</button>
      <button class="ghost" id="toggleAdv" type="button">Erweitert</button>
    </div>

    <div class="advanced" id="adv">
      <div class="pill">
        <label>Tol</label>
        <input id="tol" type="number" step="0.05" value="0.20"/>
      </div>
      <div class="pill">
        <label>km</label>
        <input id="rad" type="number" step="0.5" value="10.0"/>
      </div>
    </div>

    <div class="hint" id="hint">Auto-Geocode läuft im Hintergrund.</div>
  </div>

  <div id="map"></div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const map = L.map('map').setView([47.3769, 8.5417], 12);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap'
    }).addTo(map);

    let layer = null;
    let requestId = "__REQUEST_ID__" || "";

    function bboxStr() {
      const b = map.getBounds();
      return [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].join(",");
    }

    async function loadData() {
      const url = requestId
        ? `/map/prices?request_id=${encodeURIComponent(requestId)}&bbox=${encodeURIComponent(bboxStr())}`
        : `/map/prices?bbox=${encodeURIComponent(bboxStr())}`;

      const res = await fetch(url);
      const geo = await res.json();

      if (layer) {
        layer.remove();
      }

      layer = L.geoJSON(geo, {
        pointToLayer: (feature, latlng) => {
          const p = feature.properties || {};
          const html = `<div class="price-label">${(p.price_chf || 0).toLocaleString('de-CH')} CHF</div>`;
          const icon = L.divIcon({ html: html, className: "", iconSize: [1,1] });

          return L.marker(latlng, { icon }).bindPopup(
            `<b>${p.title || ''}</b><br/>` +
            `${(p.price_chf || 0).toLocaleString('de-CH')} CHF • ${(p.area_sqm || 0).toFixed(0)} m²<br/>` +
            `${p.address || ''}<br/>` +
            (p.url ? `<a href="${p.url}" target="_blank">Link</a>` : "")
          );
        }
      }).addTo(map);
    }

    map.on('moveend zoomend', async () => {
      try {
        await loadData();
      } catch (e) {
        console.error(e);
      }
    });

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

        if (!isFinite(tol) || tol < 0.05) tol = 0.2;
        if (!isFinite(rad) || rad < 0.5) rad = 10.0;

        if (!loc) {
          alert("Bitte Ort/Adresse eingeben.");
          return;
        }
        if (!isFinite(sqm) || sqm <= 0) {
          alert("Bitte m² eingeben.");
          return;
        }

        hint.textContent = "Request läuft…";

        const res = await fetch("/valuation/request", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            location: loc,
            area_sqm: sqm,
            tolerance: tol,
            radius_km: rad
          })
        });

        const data = await res.json().catch(() => ({}));

        if (!res.ok) {
          const msg = (typeof data.detail === "string")
            ? data.detail
            : JSON.stringify(data.detail || data);
          alert(msg || "Fehler beim Request");
          hint.textContent = "Fehler beim Request.";
          return;
        }

        requestId = data.request_id;

        history.replaceState(null, "", `/map?request_id=${encodeURIComponent(requestId)}`);

        if (data.subject_lat && data.subject_lon) {
          map.setView([data.subject_lat, data.subject_lon], 13);
        }

        hint.textContent = `Request geladen: ${requestId}`;
        await loadData();

      } catch (e) {
        alert("Error: " + e.message);
        hint.textContent = "Fehler beim Laden.";
      } finally {
        btn.disabled = false;
        btn.textContent = old;
      }
    });

    async function init() {
      const hint = document.getElementById("hint");

      try {
        if (requestId) {
          hint.textContent = `Lade Request ${requestId}…`;

          const res = await fetch(`/debug/request/${encodeURIComponent(requestId)}`);
          if (res.ok) {
            const data = await res.json();

            if (data.location) {
              document.getElementById("loc").value = data.location;
            }
            if (data.area_sqm) {
              document.getElementById("sqm").value = data.area_sqm;
            }
            if (data.tolerance) {
              document.getElementById("tol").value = data.tolerance;
            }
            if (data.radius_km) {
              document.getElementById("rad").value = data.radius_km;
            }

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


# -------------------------
# Frontend routes
# -------------------------
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

    cleanup_requests()
    df = load_dataset_safe()
    filt = REQUESTS.get(request_id) if request_id else None

    if filt:
        target = float(filt["area_sqm"])
        tol = float(filt["tolerance"])
        lo = target * (1 - tol)
        hi = target * (1 + tol)
        df = df[(df["area_sqm"] >= lo) & (df["area_sqm"] <= hi)]

    df = df.head(500)
    features = []

    for _, row in df.iterrows():
        addr = str(row["address"])
        key = enqueue_geocode(addr)
        coord = cache_get(key)

        if not coord:
          q = addr if ("," in addr) else f"{addr}, Schweiz"
          geo = geocode_photon_ch(q)
          if geo:
              cache_set_ok(key, q, geo[0], geo[1])
              coord = geo

        if not coord:
            continue

        lat, lon = coord

        if not (west <= lon <= east and south <= lat <= north):
            continue

        if filt:
            subj_lat = float(filt["subject_lat"])
            subj_lon = float(filt["subject_lon"])
            rad = float(filt["radius_km"])
            if haversine_km(subj_lat, subj_lon, lat, lon) > rad:
                continue

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "title": str(row["title"]),
                "address": str(row["address"]),
                "price_chf": int(row["price_chf"]),
                "area_sqm": float(row["area_sqm"]),
                "url": str(row.get("url", "")),
            }
        })

        if len(features) >= 80:
            break

    return JSONResponse({
        "type": "FeatureCollection",
        "features": features
    })
