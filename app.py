import os
import math
import random
import requests
from typing import Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="VaimoAI Map Minimal Stable")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
PHOTON_API = os.getenv("PHOTON_API", "https://photon.komoot.io/api/")
PHOTON_LANG = os.getenv("PHOTON_LANG", "de")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def allow_iframe_embedding(request: Request, call_next):
    response = await call_next(request)
    response.headers.pop("X-Frame-Options", None)
    response.headers["Content-Security-Policy"] = "frame-ancestors *;"
    return response


# -------------------------
# Simple in-memory request store
# -------------------------
REQUESTS: dict[str, dict] = {}


# -------------------------
# Health
# -------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "photon_api": PHOTON_API,
        "static_exists": os.path.isdir(STATIC_DIR),
    }


# -------------------------
# Photon geocoding
# -------------------------
def _is_ch_feature(props: dict) -> bool:
    cc = (props.get("countrycode") or props.get("countryCode") or "").upper()
    country = (props.get("country") or "").lower()
    return (cc == "CH") or ("switzerland" in country) or ("schweiz" in country)


def geocode_photon_ch(q: str) -> Optional[Tuple[float, float]]:
    if not q or not q.strip():
        return None

    q = q.strip()
    q2 = q if "," in q else f"{q}, Schweiz"

    try:
        r = requests.get(
            PHOTON_API,
            params={"q": q2, "limit": 5, "lang": PHOTON_LANG},
            headers={"User-Agent": "VaimoAI/1.0"},
            timeout=12,
        )

        if r.status_code != 200:
            return None

        data = r.json() or {}
        features = data.get("features") or []
        if not features:
            return None

        chosen = None
        for f in features:
            props = f.get("properties") or {}
            if _is_ch_feature(props):
                chosen = f
                break

        if not chosen:
            chosen = features[0]

        geom = chosen.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            return None

        lon, lat = float(coords[0]), float(coords[1])
        return (lat, lon)

    except Exception:
        return None


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


# -------------------------
# Request model
# -------------------------
class ValuationRequest(BaseModel):
    location: str = Field(..., description="Ort oder Adresse in der Schweiz")
    area_sqm: float = Field(..., gt=0)
    tolerance: float = Field(0.2, ge=0.05, le=0.6)
    radius_km: float = Field(10.0, ge=0.5, le=25.0)


@app.post("/valuation/request")
def create_request(req: ValuationRequest):
    subj = geocode_photon_ch(req.location)
    if not subj:
        raise HTTPException(
            status_code=503,
            detail="Geocoding fehlgeschlagen. Bitte genauere Adresse eingeben oder später erneut versuchen."
        )

    request_id = f"req_{random.randint(100000, 999999)}"
    REQUESTS[request_id] = {
        "location": req.location,
        "area_sqm": float(req.area_sqm),
        "tolerance": float(req.tolerance),
        "radius_km": float(req.radius_km),
        "subject_lat": float(subj[0]),
        "subject_lon": float(subj[1]),
    }

    return {
        "request_id": request_id,
        "map_url": f"/map?request_id={request_id}",
        "subject_lat": float(subj[0]),
        "subject_lon": float(subj[1]),
    }


@app.get("/debug/request/{request_id}")
def debug_request(request_id: str):
    if request_id not in REQUESTS:
        raise HTTPException(status_code=404, detail="unknown request_id")
    return REQUESTS[request_id]


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
    html, body, #map { height:100%; margin:0; background:#0b0b0b; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; }
    .topbar {
      position:absolute; top:14px; left:50%; transform:translateX(-50%);
      z-index:1000; width:min(980px, calc(100% - 28px));
      background:rgba(15,15,17,0.55); border:1px solid rgba(255,255,255,0.12);
      border-radius:22px; padding:12px;
      backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
      box-shadow:0 16px 40px rgba(0,0,0,0.35);
    }
    .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
    .brand { display:flex; align-items:center; padding:6px 10px; border-radius:16px; background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.10); }
    .brand img { width:84px; height:auto; display:block; background:transparent; padding:0; border-radius:0; }
    .pill { display:flex; align-items:center; gap:8px; padding:8px 10px; border-radius:16px; background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12); }
    .pill label { font-size:12px; color:rgba(255,255,255,0.70); white-space:nowrap; }
    .pill input { width:200px; border:none; outline:none; background:transparent; color:white; font-size:14px; }
    .pill input[type="number"] { width:115px; }
    .btn { border:none; cursor:pointer; padding:10px 14px; border-radius:16px; font-weight:800; background:rgba(255,255,255,0.92); color:#0b0b0b; }
    .btn:disabled { opacity:0.65; cursor:default; }
    .ghost { border:1px solid rgba(255,255,255,0.14); background:rgba(255,255,255,0.06); color:rgba(255,255,255,0.92); padding:10px 12px; border-radius:16px; cursor:pointer; font-weight:700; }
    .advanced { display:none; margin-top:10px; gap:10px; align-items:center; flex-wrap:wrap; }
    .hint { margin-top:8px; font-size:12px; color:rgba(255,255,255,0.70); padding:0 8px; }
    .price-label { background:rgba(255,255,255,0.96); border-radius:16px; padding:6px 10px; font-weight:800; box-shadow:0 8px 24px rgba(0,0,0,0.22); white-space:nowrap; }
  </style>
</head>
<body>
  <div class="topbar">
    <div class="row">
      <div class="brand"><img src="/static/vaimo.png" onerror="this.style.display='none'" alt="VAIMO"/></div>
      <div class="pill"><label>Ort</label><input id="loc" placeholder="z.B. Zürich / Berg TG / Adresse"/></div>
      <div class="pill"><label>m²</label><input id="sqm" type="number" placeholder="z.B. 100"/></div>
      <button class="btn" id="apply" type="button">Request</button>
      <button class="ghost" id="toggleAdv" type="button">Erweitert</button>
    </div>

    <div class="advanced" id="adv">
      <div class="pill"><label>Tol</label><input id="tol" type="number" step="0.05" value="0.20"/></div>
      <div class="pill"><label>km</label><input id="rad" type="number" step="0.5" value="10.0"/></div>
    </div>

    <div class="hint">Stabile Minimalversion. Marker sind Demo-Marker rund um den Standort.</div>
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

      if (layer) layer.remove();
      layer = L.geoJSON(geo, {
        pointToLayer: (feature, latlng) => {
          const p = feature.properties || {};
          const html = `<div class="price-label">${(p.price_chf||0).toLocaleString('de-CH')} CHF</div>`;
          const icon = L.divIcon({ html: html, className: "", iconSize: [1,1] });
          return L.marker(latlng, { icon }).bindPopup(
            `<b>${p.title||''}</b><br/>` +
            `${(p.price_chf||0).toLocaleString('de-CH')} CHF • ${(p.area_sqm||0).toFixed(0)} m²<br/>` +
            `${p.address||''}`
          );
        }
      }).addTo(map);
    }

    map.on('moveend zoomend', loadData);

    document.getElementById('toggleAdv').addEventListener('click', () => {
      const adv = document.getElementById('adv');
      adv.style.display = (adv.style.display === 'flex') ? 'none' : 'flex';
    });

    document.getElementById('apply').addEventListener('click', async () => {
      const btn = document.getElementById('apply');
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

        if (!loc) { alert("Bitte Ort/Adresse eingeben."); return; }
        if (!isFinite(sqm) || sqm <= 0) { alert("Bitte m² eingeben."); return; }

        const res = await fetch("/valuation/request", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ location: loc, area_sqm: sqm, tolerance: tol, radius_km: rad })
        });

        const data = await res.json().catch(() => ({}));

        if (!res.ok) {
          const msg = (typeof data.detail === "string") ? data.detail : JSON.stringify(data.detail || data);
          alert(msg || "Fehler beim Request");
          return;
        }

        requestId = data.request_id;
        history.replaceState(null, "", `/map?request_id=${requestId}`);

        if (data.subject_lat && data.subject_lon) {
          map.setView([data.subject_lat, data.subject_lon], 13);
        }

        await loadData();
      } catch (e) {
        alert("Error: " + e.message);
      } finally {
        btn.disabled = false;
        btn.textContent = old;
      }
    });

    loadData();
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

    features = []

    if request_id and request_id in REQUESTS:
        req = REQUESTS[request_id]
        center_lat = req["subject_lat"]
        center_lon = req["subject_lon"]
        area = req["area_sqm"]

        for i in range(20):
            lat = center_lat + random.uniform(-0.03, 0.03)
            lon = center_lon + random.uniform(-0.03, 0.03)

            if not (west <= lon <= east and south <= lat <= north):
                continue

            price = random.randint(700000, 1800000)
            sqm = area * random.uniform(0.85, 1.15)

            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "title": f"Vergleichsobjekt {i+1}",
                    "address": "Demo-Marker",
                    "price_chf": price,
                    "area_sqm": sqm
                }
            })

    return JSONResponse({"type": "FeatureCollection", "features": features})
