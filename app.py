import os
import re
import uuid
import math
import time
import sqlite3
import threading
from typing import Optional, Tuple

import pandas as pd
import requests
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="VaimoAI Comparable Map")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_CSV = os.getenv("DATA_CSV", os.path.join(BASE_DIR, "data", "immoscout.csv"))
DB_PATH = os.getenv("GEO_DB", os.path.join(BASE_DIR, "geocache.sqlite"))

PHOTON_API = os.getenv("PHOTON_API", "https://photon.komoot.io/api/")
PHOTON_LANG = os.getenv("PHOTON_LANG", "de")

GEOCODE_MIN_INTERVAL_SEC = float(os.getenv("GEOCODE_MIN_INTERVAL_SEC", "1.2"))
GEOCODE_QUEUE_MAX = int(os.getenv("GEOCODE_QUEUE_MAX", "5000"))

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

    if "x-frame-options" in response.headers:
        del response.headers["x-frame-options"]

    response.headers["Content-Security-Policy"] = "frame-ancestors *;"
    return response


# -------------------------
# Health
# -------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "csv_exists": os.path.exists(DATA_CSV),
        "db_path": DB_PATH,
        "photon_api": PHOTON_API,
    }


# -------------------------
# SQLite helpers
# -------------------------
def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def db_init():
    conn = db_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS geocache (
        key TEXT PRIMARY KEY,
        q TEXT NOT NULL,
        lat REAL,
        lon REAL,
        status TEXT NOT NULL DEFAULT 'ok',
        tries INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        updated_at INTEGER NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS queue (
        key TEXT PRIMARY KEY,
        q TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'queued',
        tries INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        updated_at INTEGER NOT NULL
    );
    """)

    conn.commit()
    conn.close()


db_init()


# -------------------------
# Helpers
# -------------------------
def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def parse_float_area(value: str):
    if not isinstance(value, str):
        return None
    v = value.replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)\s*m", v)
    return float(m.group(1)) if m else None


def parse_int_price(value: str):
    if not isinstance(value, str):
        return None
    cleaned = value.replace("’", "").replace("'", "")
    digits = re.findall(r"\d+", cleaned)
    return int("".join(digits)) if digits else None


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
# Dataset (lazy load, safe)
# -------------------------
DATA_CACHE = None
DATA_CACHE_ERROR = None


def load_dataset_safe():
    global DATA_CACHE, DATA_CACHE_ERROR

    if DATA_CACHE is not None:
        return DATA_CACHE

    if not os.path.exists(DATA_CSV):
        DATA_CACHE_ERROR = f"CSV not found: {DATA_CSV}"
        DATA_CACHE = pd.DataFrame(columns=["address", "title", "area_sqm", "price_chf", "url"])
        return DATA_CACHE

    try:
        df = pd.read_csv(DATA_CSV)

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

        DATA_CACHE = df
        DATA_CACHE_ERROR = None
        return DATA_CACHE

    except Exception as e:
        DATA_CACHE_ERROR = str(e)
        DATA_CACHE = pd.DataFrame(columns=["address", "title", "area_sqm", "price_chf", "url"])
        return DATA_CACHE


@app.get("/debug/data")
def debug_data():
    df = load_dataset_safe()
    return {
        "rows": int(len(df)),
        "error": DATA_CACHE_ERROR,
        "csv_path": DATA_CSV,
        "csv_exists": os.path.exists(DATA_CSV),
    }


# -------------------------
# Cache / Queue
# -------------------------
def cache_get(key: str) -> Optional[Tuple[float, float]]:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT lat, lon, status FROM geocache WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    lat, lon, status = row
    if status == "ok" and lat is not None and lon is not None:
        return (float(lat), float(lon))
    return None


def cache_set_ok(key: str, q: str, lat: float, lon: float):
    now = int(time.time())
    conn = db_conn()
    conn.execute("""
      INSERT INTO geocache(key,q,lat,lon,status,tries,last_error,updated_at)
      VALUES(?,?,?,?, 'ok', 0, NULL, ?)
      ON CONFLICT(key) DO UPDATE SET
        q=excluded.q,
        lat=excluded.lat,
        lon=excluded.lon,
        status='ok',
        last_error=NULL,
        updated_at=excluded.updated_at;
    """, (key, q, lat, lon, now))
    conn.commit()
    conn.close()


def cache_set_fail(key: str, q: str, err: str, tries_inc: int = 1):
    now = int(time.time())
    conn = db_conn()
    conn.execute("""
      INSERT INTO geocache(key,q,lat,lon,status,tries,last_error,updated_at)
      VALUES(?,?,?,?, 'fail', ?, ?, ?)
      ON CONFLICT(key) DO UPDATE SET
        q=excluded.q,
        status='fail',
        tries=geocache.tries+?,
        last_error=?,
        updated_at=?;
    """, (key, q, None, None, tries_inc, err[:500], now, tries_inc, err[:500], now))
    conn.commit()
    conn.close()


def queue_size() -> int:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM queue WHERE status IN ('queued','working')")
    n = cur.fetchone()[0]
    conn.close()
    return int(n)


def enqueue_geocode(q: str) -> str:
    q = (q or "").strip()
    if not q:
        return ""

    q2 = q if ("," in q) else f"{q}, Schweiz"
    key = norm_key(q2)

    if cache_get(key):
        return key

    if queue_size() >= GEOCODE_QUEUE_MAX:
        return key

    now = int(time.time())
    conn = db_conn()
    conn.execute("""
      INSERT INTO queue(key,q,status,tries,last_error,updated_at)
      VALUES(?, ?, 'queued', 0, NULL, ?)
      ON CONFLICT(key) DO UPDATE SET
        updated_at=excluded.updated_at;
    """, (key, q2, now))
    conn.commit()
    conn.close()
    return key


def queue_take_one() -> Optional[Tuple[str, str]]:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
      SELECT key, q FROM queue
      WHERE status='queued'
      ORDER BY updated_at ASC
      LIMIT 1
    """)
    row = cur.fetchone()
    if not row:
        conn.close()
        return None

    key, q = row
    now = int(time.time())
    cur.execute("UPDATE queue SET status='working', updated_at=? WHERE key=?", (now, key))
    conn.commit()
    conn.close()
    return key, q


def queue_mark_done(key: str):
    conn = db_conn()
    now = int(time.time())
    conn.execute("UPDATE queue SET status='done', updated_at=? WHERE key=?", (now, key))
    conn.commit()
    conn.close()


def queue_mark_fail(key: str, err: str):
    conn = db_conn()
    now = int(time.time())
    conn.execute("""
      UPDATE queue
      SET status='fail', tries=tries+1, last_error=?, updated_at=?
      WHERE key=?
    """, (err[:500], now, key))
    conn.commit()
    conn.close()


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
    q2 = q if ("," in q) else f"{q}, Schweiz"

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


# -------------------------
# Worker
# -------------------------
_worker_started = False
_worker_lock = threading.Lock()
_last_geocode_ts = 0.0


def geocode_worker_loop():
    global _last_geocode_ts

    while True:
        item = queue_take_one()
        if not item:
            time.sleep(1.0)
            continue

        key, q = item

        if cache_get(key):
            queue_mark_done(key)
            continue

        with _worker_lock:
            now = time.time()
            wait = GEOCODE_MIN_INTERVAL_SEC - (now - _last_geocode_ts)
            if wait > 0:
                time.sleep(wait)
            _last_geocode_ts = time.time()

        coord = geocode_photon_ch(q)

        if coord:
            lat, lon = coord
            cache_set_ok(key, q, lat, lon)
            queue_mark_done(key)
        else:
            cache_set_fail(key, q, "geocode_failed", tries_inc=1)
            queue_mark_fail(key, "geocode_failed")
            time.sleep(1.0)


@app.on_event("startup")
def startup():
    global _worker_started
    if not _worker_started:
        t = threading.Thread(target=geocode_worker_loop, daemon=True)
        t.start()
        _worker_started = True


@app.get("/admin/queue")
def admin_queue():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT status, COUNT(*) FROM queue GROUP BY status")
    q_stats = {row[0]: row[1] for row in cur.fetchall()}
    cur.execute("SELECT status, COUNT(*) FROM geocache GROUP BY status")
    c_stats = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()

    return {
        "queue": q_stats,
        "cache": c_stats,
        "db_path": DB_PATH,
        "csv_error": DATA_CACHE_ERROR,
    }


# -------------------------
# Requests
# -------------------------
REQUESTS = {}


class ValuationRequest(BaseModel):
    location: str = Field(..., description="Ort oder volle Adresse (Schweiz)")
    area_sqm: float = Field(..., gt=0)
    tolerance: float = Field(0.2, ge=0.05, le=0.6)
    radius_km: float = Field(10.0, ge=0.5, le=25.0)


@app.post("/valuation/request")
def create_request(req: ValuationRequest):
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
        "map_url": f"/map?request_id={request_id}",
        "subject_lat": float(subj[0]),
        "subject_lon": float(subj[1]),
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
    html, body, #map { height:100%; margin:0; background:#0b0b0b; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; }
    .topbar {
      position:absolute; top:14px; left:50%; transform:translateX(-50%);
      z-index:1000; width:min(980px, calc(100% - 28px));
      background:rgba(15,15,17,0.55); border:1px solid rgba(255,255,255,0.12);
      border-radius:22px; padding:12px 12px;
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

    <div class="hint">Auto-Geocode läuft im Hintergrund.</div>
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

    df = load_dataset_safe()
    filt = REQUESTS.get(request_id) if request_id else None

    if filt:
        target = filt["area_sqm"]
        tol = filt["tolerance"]
        lo = target * (1 - tol)
        hi = target * (1 + tol)
        df = df[(df["area_sqm"] >= lo) & (df["area_sqm"] <= hi)]

    df = df.head(300)
    features = []

    for _, row in df.iterrows():
        addr = str(row["address"])
        key = enqueue_geocode(addr)
        coord = cache_get(key)

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
                "title": row["title"],
                "address": row["address"],
                "price_chf": int(row["price_chf"]),
                "area_sqm": float(row["area_sqm"]),
                "url": row.get("url", "")
            }
        })

        if len(features) >= 80:
            break

    return JSONResponse({"type": "FeatureCollection", "features": features})
