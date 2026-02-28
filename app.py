import os, json, time, re, uuid, math
import pandas as pd
import requests

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="VaimoAI Comparable Map")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_CSV = os.getenv("DATA_CSV", os.path.join(BASE_DIR, "data", "immoscout.csv"))
CACHE_PATH = os.path.join(BASE_DIR, "geocode_cache.json")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -------------------------
# Parse helpers
# -------------------------
def parse_int_price(value: str):
    if not isinstance(value, str):
        return None
    cleaned = value.replace("’", "").replace("'", "")
    digits = re.findall(r"\d+", cleaned)
    return int("".join(digits)) if digits else None

def parse_float_area(value: str):
    if not isinstance(value, str):
        return None
    v = value.replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)\s*m", v)
    return float(m.group(1)) if m else None

# -------------------------
# Geocode cache (best effort)
# -------------------------
def load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache):
    try:
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        os.replace(tmp, CACHE_PATH)
    except Exception:
        pass  # best effort

GEOCODE_CACHE = load_cache()

def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def nominatim_geocode(q: str):
    """
    Returns (lat, lon) or None.
    For production: move to paid geocoder / own data / provider coords (ImmoScout).
    """
    key = norm_key(q)
    if not key:
        return None
    cached = GEOCODE_CACHE.get(key)
    if cached and "lat" in cached:
        return (cached["lat"], cached["lon"])

    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": q, "format": "json", "limit": 1}
    headers = {"User-Agent": "VaimoAI/1.0"}
    r = requests.get(url, params=params, headers=headers, timeout=12)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    GEOCODE_CACHE[key] = {"lat": lat, "lon": lon, "ts": int(time.time())}
    save_cache(GEOCODE_CACHE)
    time.sleep(0.15)
    return (lat, lon)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# -------------------------
# Load dataset (your CSV)
# -------------------------
def load_dataset():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"CSV not found: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)

    # Update these if your CSV columns differ
    col_address = "HgListingCard_address_3884e"
    col_title = "HgListingDescription_title_fa343"
    col_area = "HgListingRoomsLivingSpacePrice_roomsLivingSpacePrice_ab258 3"
    col_price = "HgListingRoomsLivingSpacePrice_price_ad2cb"
    col_url = "HgCardElevated_content_900d9 href"

    for c in [col_address, col_title, col_area, col_price]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df = df.copy()
    df["address"] = df[col_address].astype(str)
    df["title"] = df[col_title].astype(str)
    df["area_sqm"] = df[col_area].apply(parse_float_area)
    df["price_chf"] = df[col_price].apply(parse_int_price)
    df["url"] = df[col_url].astype(str) if col_url in df.columns else ""

    df = df.dropna(subset=["area_sqm", "price_chf"])
    df["area_sqm"] = df["area_sqm"].astype(float)
    df["price_chf"] = df["price_chf"].astype(int)
    return df

DATA = load_dataset()

# -------------------------
# Requests (MVP store)
# -------------------------
REQUESTS = {}

class ValuationRequest(BaseModel):
    location: str = Field(..., description="Ort oder volle Adresse, z.B. 'Zürich' oder 'Bahnhofstrasse 1, Zürich'")
    area_sqm: float = Field(..., gt=0)
    tolerance: float = Field(0.2, ge=0.05, le=0.6)
    radius_km: float = Field(3.0, ge=0.5, le=25.0)

@app.post("/valuation/request")
def create_request(req: ValuationRequest):
    subj = nominatim_geocode(req.location)
    if not subj:
        raise HTTPException(status_code=400, detail="Could not geocode location")

    request_id = "req_" + uuid.uuid4().hex[:10]
    REQUESTS[request_id] = {
        "location": req.location,
        "area_sqm": float(req.area_sqm),
        "tolerance": float(req.tolerance),
        "radius_km": float(req.radius_km),
        "subject_lat": subj[0],
        "subject_lon": subj[1],
        "ts": int(time.time())
    }
    return {"request_id": request_id, "map_url": f"/map?request_id={request_id}"}

@app.get("/map", response_class=HTMLResponse)
def map_page(request_id: str = ""):
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Vaimo – Comparable Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <style>
    html, body, #map {{ height: 100%; margin: 0; background:#0b0b0b; }}
    .hud {{
      position:absolute; top:14px; left:14px; z-index:1000;
      display:flex; gap:12px; align-items:center;
      background: rgba(12,12,12,0.55);
      border: 1px solid rgba(255,255,255,0.12);
      padding: 10px 12px; border-radius: 16px; backdrop-filter: blur(10px);
      color: white; font-family: Arial;
    }}
    .hud img {{ width: 92px; height:auto; background:#fff; padding:6px; border-radius:12px; }}
    .hud input {{
      background: rgba(255,255,255,0.10);
      border: 1px solid rgba(255,255,255,0.18);
      color: white; padding: 10px 12px; border-radius: 12px; outline: none;
      font-size: 14px; width: 170px;
    }}
    .hud input[type="number"] {{ width: 105px; }}
    .hud button {{
      background: #ffffff; color:#000; font-weight:800; border:none;
      padding: 10px 12px; border-radius: 12px; cursor:pointer;
    }}
    .price-label {{
      background: rgba(255,255,255,0.96);
      border-radius: 16px;
      padding: 6px 10px;
      font-family: Arial;
      font-weight: 800;
      box-shadow: 0 6px 18px rgba(0,0,0,.22);
      white-space: nowrap;
    }}
    .hint {{ opacity:.7; font-size:12px; }}
  </style>
</head>
<body>

<div class="hud">
  <img src="/static/vaimo.png" onerror="this.style.display='none'" alt="VAIMO"/>
  <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
    <input id="loc" placeholder="Ort/Adresse" />
    <input id="sqm" type="number" placeholder="m²" />
    <input id="tol" type="number" step="0.05" value="0.20" placeholder="tol" />
    <input id="rad" type="number" step="0.5" value="3.0" placeholder="km" />
    <button id="apply">Request</button>
    <span class="hint">oder URL mit request_id nutzen</span>
  </div>
</div>

<div id="map"></div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
  const map = L.map('map').setView([47.3769, 8.5417], 12);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    maxZoom: 19, attribution: '&copy; OpenStreetMap'
  }}).addTo(map);

  let layer;
  let requestId = "{request_id}" || "";

  function bboxStr() {{
    const b = map.getBounds();
    return [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].join(",");
  }}

  async function loadData() {{
    const url = requestId
      ? `/map/prices?request_id=${{encodeURIComponent(requestId)}}&bbox=${{encodeURIComponent(bboxStr())}}`
      : `/map/prices?bbox=${{encodeURIComponent(bboxStr())}}`;

    const res = await fetch(url);
    const geo = await res.json();

    if (layer) layer.remove();
    layer = L.geoJSON(geo, {{
      pointToLayer: (feature, latlng) => {{
        const p = feature.properties;
        const html = `<div class="price-label">${{(p.price_chf||0).toLocaleString('de-CH')}} CHF</div>`;
        const icon = L.divIcon({{ html, className:"", iconSize:[1,1] }});
        return L.marker(latlng, {{ icon }}).bindPopup(
          `<b>${{p.title||''}}</b><br/>` +
          `${{(p.price_chf||0).toLocaleString('de-CH')}} CHF • ${{(p.area_sqm||0).toFixed(0)}} m²<br/>` +
          `${{p.address||''}}<br/>` +
          (p.url ? `<a href="${{p.url}}" target="_blank">Link</a>` : "")
        );
      }}
    }}).addTo(map);
  }}

  map.on('moveend zoomend', loadData);

  document.getElementById('apply').addEventListener('click', async () => {{
    const loc = document.getElementById('loc').value.trim();
    const sqm = parseFloat(document.getElementById('sqm').value);
    const tol = parseFloat(document.getElementById('tol').value);
    const rad = parseFloat(document.getElementById('rad').value);

    if (!loc || !sqm) {{
      alert("Bitte Ort/Adresse und m² eingeben");
      return;
    }}

    const res = await fetch("/valuation/request", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{
        location: loc,
        area_sqm: sqm,
        tolerance: isFinite(tol) ? tol : 0.2,
        radius_km: isFinite(rad) ? rad : 3.0
      }})
    }});
    const data = await res.json();
    if (!res.ok) {{
      alert(data.detail || "Fehler");
      return;
    }}

    requestId = data.request_id;
    history.replaceState(null, "", `/map?request_id=${{requestId}}`);
    loadData();
  }});

  loadData();
</script>
</body>
</html>
"""

@app.get("/map/prices")
def map_prices(
    bbox: str = Query(..., description="west,south,east,north"),
    request_id: str = Query("", description="Created by /valuation/request")
):
    try:
        west, south, east, north = [float(x) for x in bbox.split(",")]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bbox")

    filt = REQUESTS.get(request_id) if request_id else None

    # Limit geocoding work per call
    df = DATA.head(150)

    # Geocode listing addresses (cached)
    rows = []
    for _, row in df.iterrows():
        coord = nominatim_geocode(row["address"])
        if coord:
            rows.append((row, coord[0], coord[1]))

    features = []
    if filt:
        subj_lat = filt["subject_lat"]
        subj_lon = filt["subject_lon"]
        target = filt["area_sqm"]
        tol = filt["tolerance"]
        rad = filt["radius_km"]

        lo = target * (1 - tol)
        hi = target * (1 + tol)

        for row, lat, lon in rows:
            if not (west <= lon <= east and south <= lat <= north):
                continue
            if not (lo <= float(row["area_sqm"]) <= hi):
                continue
            if haversine_km(subj_lat, subj_lon, lat, lon) > rad:
                continue

            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "title": row["title"],
                    "address": row["address"],
                    "price_chf": int(row["price_chf"]),
                    "area_sqm": float(row["area_sqm"]),
                    "url": row.get("url", "")
                }
            })
    else:
        for row, lat, lon in rows[:40]:
            if not (west <= lon <= east and south <= lat <= north):
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "title": row["title"],
                    "address": row["address"],
                    "price_chf": int(row["price_chf"]),
                    "area_sqm": float(row["area_sqm"]),
                    "url": row.get("url", "")
                }
            })

    return JSONResponse({"type": "FeatureCollection", "features": features})
