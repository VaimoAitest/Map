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
        pass


GEOCODE_CACHE = load_cache()


def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def nominatim_geocode_ch(q: str):
    """
    Geocode only in Switzerland (countrycodes=ch).
    If user enters a short place like "Berg", we auto-append ", Schweiz".
    Returns (lat, lon) or None.
    """
    if not q or not q.strip():
        return None

    q = q.strip()
    # make ambiguous queries more Swiss-specific
    q2 = q if ("," in q) else f"{q}, Schweiz"

    cache_key = norm_key(q2)
    cached = GEOCODE_CACHE.get(cache_key)
    if cached and "lat" in cached:
        return (cached["lat"], cached["lon"])

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": q2,
        "format": "json",
        "limit": 1,
        "countrycodes": "ch",
    }
    headers = {"User-Agent": "VaimoAI/1.0 (render)"}

    r = requests.get(url, params=params, headers=headers, timeout=12)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None

    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])

    GEOCODE_CACHE[cache_key] = {"lat": lat, "lon": lon, "ts": int(time.time())}
    save_cache(GEOCODE_CACHE)
    time.sleep(0.12)
    return (lat, lon)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# -------------------------
# Load dataset (your CSV)
# -------------------------
def load_dataset():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"CSV not found: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)

    # Columns from your exported CSV (adjust if yours differs)
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
    location: str = Field(..., description="Ort oder volle Adresse (Schweiz)")
    area_sqm: float = Field(..., gt=0)
    tolerance: float = Field(0.2, ge=0.05, le=0.6)  # min 0.05 avoids tol=0 issues
    radius_km: float = Field(3.0, ge=0.5, le=25.0)


@app.post("/valuation/request")
def create_request(req: ValuationRequest):
    subj = nominatim_geocode_ch(req.location)
    if not subj:
        raise HTTPException(status_code=400, detail="Could not geocode location in Switzerland")

    request_id = "req_" + uuid.uuid4().hex[:10]
    REQUESTS[request_id] = {
        "location": req.location,
        "area_sqm": float(req.area_sqm),
        "tolerance": float(req.tolerance),
        "radius_km": float(req.radius_km),
        "subject_lat": subj[0],
        "subject_lon": subj[1],
        "ts": int(time.time()),
    }
    return {
        "request_id": request_id,
        "map_url": f"/map?request_id={request_id}",
        "subject_lat": subj[0],
        "subject_lon": subj[1],
    }


@app.get("/debug/request/{request_id}")
def debug_request(request_id: str):
    r = REQUESTS.get(request_id)
    if not r:
        raise HTTPException(404, "unknown request_id")
    return r


# -------------------------
# HTML template (NO f-string!)
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
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Arial, sans-serif;
    }

    .topbar {
      position: absolute;
      top: 14px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 1000;
      width: min(980px, calc(100% - 28px));
      background: rgba(15, 15, 17, 0.55);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 22px;
      padding: 12px 12px;
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      box-shadow: 0 16px 40px rgba(0,0,0,0.35);
    }

    .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }

    .brand {
      display:flex; align-items:center; gap:10px;
      padding: 6px 10px;
      border-radius: 16px;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.10);
    }

    .brand img {
      width: 84px;
      height: auto;
      display: block;
      background: transparent;
      padding: 0;
      border-radius: 0;
    }

    .pill {
      display:flex; align-items:center; gap:8px;
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
      width: 190px;
      border: none;
      outline: none;
      background: transparent;
      color: white;
      font-size: 14px;
    }

    .pill input[type="number"] { width: 115px; }

    .btn {
      border: none;
      cursor: pointer;
      padding: 10px 14px;
      border-radius: 16px;
      font-weight: 800;
      background: rgba(255,255,255,0.92);
      color: #0b0b0b;
    }

    .btn:disabled { opacity: 0.65; cursor: default; }
    .btn:active { transform: translateY(1px); }

    .ghost {
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(255,255,255,0.06);
      color: rgba(255,255,255,0.92);
      padding: 10px 12px;
      border-radius: 16px;
      cursor: pointer;
      font-weight: 700;
    }

    .hint {
      margin-top: 8px;
      font-size: 12px;
      color: rgba(255,255,255,0.70);
      padding: 0 8px;
    }

    .advanced {
      display: none;
      margin-top: 10px;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
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
      <div class="brand">
        <img src="/static/vaimo.png" onerror="this.style.display='none'" alt="VAIMO"/>
      </div>

      <div class="pill" title="Ort oder genaue Adresse (Schweiz)">
        <label>Ort</label>
        <input id="loc" placeholder="z.B. Berg TG / Zürich / Adresse" />
      </div>

      <div class="pill" title="Wohnfläche deines Objekts">
        <label>m²</label>
        <input id="sqm" type="number" placeholder="z.B. 100" />
      </div>

      <button class="btn" id="apply" type="button" title="Filter anwenden">Request</button>
      <button class="ghost" id="toggleAdv" type="button" title="Toleranz & Radius anzeigen">Erweitert</button>
    </div>

    <div class="advanced" id="adv">
      <div class="pill" title="0.20 = ±20% (100m² -> 80 bis 120m²)">
        <label>Tol</label>
        <input id="tol" type="number" step="0.05" value="0.20" placeholder="0.20" />
      </div>

      <div class="pill" title="Umkreis in km um den Standort">
        <label>km</label>
        <input id="rad" type="number" step="0.5" value="10.0" placeholder="10.0" />
      </div>
    </div>

    <div class="hint">
      Beispiel: <b>Zürich</b> + <b>100</b> m² → zeigt ähnliche Objekte (m² ± Tol, Umkreis km).
    </div>
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
            `${p.address||''}<br/>` +
            (p.url ? `<a href="${p.url}" target="_blank">Link</a>` : "")
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

        if (!loc) { alert("Bitte Ort/Adresse eingeben (Schweiz)."); return; }
        if (!isFinite(sqm) || sqm <= 0) { alert("Bitte Wohnfläche in m² eingeben (z.B. 100)."); return; }

        // timeout so it never hangs silently
        const controller = new AbortController();
        const t = setTimeout(() => controller.abort(), 15000);

        const res = await fetch("/valuation/request", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ location: loc, area_sqm: sqm, tolerance: tol, radius_km: rad }),
          signal: controller.signal
        });

        clearTimeout(t);

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

        // If 0 results, give a helpful hint
        // (We keep it simple: user can increase km/tol.)
        // You can later auto-expand if needed.
      } catch (e) {
        alert(e.name === "AbortError" ? "Timeout (15s) – Free Render kann schlafen. Nochmal klicken." : ("JS Error: " + e.message));
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
    bbox: str = Query(..., description="west,south,east,north"),
    request_id: str = Query("", description="Created by /valuation/request")
):
    try:
        west, south, east, north = [float(x) for x in bbox.split(",")]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bbox")

    filt = REQUESTS.get(request_id) if request_id else None

    # MVP approach:
    # 1) Pre-filter by area if request exists
    # 2) Geocode only the first N candidates to keep it usable on free tier
    df = DATA

    if filt:
        target = filt["area_sqm"]
        tol = filt["tolerance"]
        lo = target * (1 - tol)
        hi = target * (1 + tol)
        df = df[(df["area_sqm"] >= lo) & (df["area_sqm"] <= hi)]

    # limit candidates to reduce geocode load per request
    df = df.head(350)

    # Geocode listing addresses (cached, Switzerland-only)
    rows = []
    for _, row in df.iterrows():
        coord = nominatim_geocode_ch(row["address"])
        if coord:
            rows.append((row, coord[0], coord[1]))

    features = []

    if filt:
        subj_lat = filt["subject_lat"]
        subj_lon = filt["subject_lon"]
        rad = filt["radius_km"]

        for row, lat, lon in rows:
            if not (west <= lon <= east and south <= lat <= north):
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
        # preview mode: show some points in view
        for row, lat, lon in rows[:80]:
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
