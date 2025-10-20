import os
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dateutil import parser
import pytz
from typing import List, Dict, Tuple
from math import fmod

# --- Skyfield for astronomy ---
from skyfield.api import load

app = FastAPI(title="Synastry One-Pack (API + UI)")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------- Input Schemas ----------
class Person(BaseModel):
    name: str
    date: str   # "YYYY-MM-DD"
    time: str   # "HH:MM"
    tz: str     # "Asia/Seoul"
    lat: float | None = None  # future: ASC/houses
    lon: float | None = None

class SynastryReq(BaseModel):
    personA: Person
    personB: Person

# ---------- Config ----------
PLANETS = ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"]
ASPECTS = {
    "conjunction": 0,
    "sextile": 60,
    "square": 90,
    "trine": 120,
    "opposition": 180,
}
DEFAULT_ORB = {
    "Sun": 6.0, "Moon": 6.0, "Mercury": 5.0, "Venus": 5.0, "Mars": 5.0,
    "Jupiter": 4.0, "Saturn": 4.0
}
PAIR_WEIGHT = {
    ("Sun","Moon"): 20, ("Venus","Mars"): 18, ("Sun","Venus"): 12, ("Sun","Mars"): 10,
    ("Moon","Venus"): 12, ("Moon","Mars"): 12, ("Mercury","Mercury"): 8,
    ("Venus","Venus"): 6, ("Mars","Mars"): 6,
}
ASPECT_MULT = {
    "conjunction": 1.00,
    "trine": 0.85,
    "sextile": 0.70,
    "square": -0.65,
    "opposition": -0.80,
}

# ---------- Utils ----------
def _angle_wrap(deg: float) -> float:
    x = fmod(deg, 360.0)
    return x + 360.0 if x < 0 else x

def angle_diff(a: float, b: float) -> float:
    d = abs(_angle_wrap(a) - _angle_wrap(b))
    return d if d <= 180 else 360 - d

def orb_allow(p1: str, p2: str) -> float:
    return (DEFAULT_ORB.get(p1, 4.0) + DEFAULT_ORB.get(p2, 4.0)) / 2

def linear_falloff(delta: float, allow: float) -> float:
    k = max(0.0, 1.0 - (delta / allow))
    return k

def ordered_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted([a, b]))

# ---------- Skyfield init ----------
ts = load.timescale()
eph = load("de421.bsp")
EARTH = eph["earth"]
SUN = eph["sun"]
MOON = eph["moon"]
MERCURY = eph["mercury"]
VENUS = eph["venus"]
MARS = eph["mars"]
JUPITER = eph["jupiter barycenter"]
SATURN = eph["saturn barycenter"]

PLANET_MAP = {
    "Sun": SUN, "Moon": MOON, "Mercury": MERCURY, "Venus": VENUS, "Mars": MARS,
    "Jupiter": JUPITER, "Saturn": SATURN,
}

def to_ts(date_str: str, time_str: str, tz_name: str):
    local = pytz.timezone(tz_name)
    dt_local = local.localize(parser.parse(f"{date_str} {time_str}"))
    dt_utc = dt_local.astimezone(pytz.utc)
    return ts.utc(dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour, dt_utc.minute, dt_utc.second)

def ecliptic_longitudes(t):
    # MVP: 적경(RA)을 15배하여 황경 근사. 추후 ecliptic 변환/하우스 추가 예정.
    longs = {}
    observer = EARTH.at(t)
    for name, body in PLANET_MAP.items():
        app = observer.observe(body).apparent()
        ra, dec, _ = app.radec()
        longs[name] = _angle_wrap(ra.hours * 15.0)
    return longs

def detect_aspects(longsA: Dict[str, float], longsB: Dict[str, float]):
    aspects = []
    for pA in PLANETS:
        for pB in PLANETS:
            d = angle_diff(longsA[pA], longsB[pB])
            allow = orb_allow(pA, pB)
            for aspect_name, deg in ASPECTS.items():
                delta = abs(d - deg)
                if delta <= allow:
                    k = linear_falloff(delta, allow)
                    aspects.append({
                        "bodies": [f"{pA}A", f"{pB}B"],
                        "type": aspect_name,
                        "exact_deg": round(d, 2),
                        "orb": round(delta, 2),
                        "strength": round(k, 3)
                    })
    aspects.sort(key=lambda x: x["strength"], reverse=True)
    return aspects

def score_synastry(aspects: List[Dict]) -> tuple[float, List[Dict]]:
    total = 0.0
    scored = []
    for a in aspects:
        p1 = a["bodies"][0].replace("A","")
        p2 = a["bodies"][1].replace("B","")
        pair = ordered_pair(p1, p2)
        base = PAIR_WEIGHT.get(pair, 4)
        mult = ASPECT_MULT.get(a["type"], 0.4)
        s = base * mult * a["strength"]
        a_with = dict(a)
        a_with["pair_weight"] = base
        a_with["aspect_mult"] = mult
        a_with["score_contrib"] = round(s, 3)
        scored.append(a_with)
        total += s
    normalized = max(0, min(100, round(50 + total, 1)))
    scored.sort(key=lambda x: abs(x["score_contrib"]), reverse=True)
    return normalized, scored

def summarize(scored_top: List[Dict]) -> str:
    def label(a):
        t = a["type"]
        pA, pB = a["bodies"]
        return f"{pA} {t} {pB} (orb {a['orb']}°, {a['score_contrib']:+.2f})"
    lines = [f"- {label(x)}" for x in scored_top[:6]]
    return "핵심 상호작용:\\n" + "\\n".join(lines)

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/compute-synastry")
def compute_synastry(req: SynastryReq):
    tA = to_ts(req.personA.date, req.personA.time, req.personA.tz)
    tB = to_ts(req.personB.date, req.personB.time, req.personB.tz)

    longsA = ecliptic_longitudes(tA)
    longsB = ecliptic_longitudes(tB)

    aspects = detect_aspects(longsA, longsB)
    score, scored = score_synastry(aspects)

    top = scored[:8]
    summary = summarize(top)

    return JSONResponse({
        "score": score,
        "aspects_top": top,
        "aspects_all_count": len(scored),
        "longitudes": {"A": longsA, "B": longsB},
        "summary": summary,
        "notes": "MVP: RA-as-longitude approximation. Upgrade to true ecliptic + houses later."
    })
from fastapi import HTTPException

class ReadingReq(BaseModel):
    score: float
    aspects_top: List[Dict]

@app.post("/generate-reading")
def generate_reading(req: ReadingReq):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    # 프롬프트: 간결/안전
    bullets = []
    for x in req.aspects_top[:6]:
        pA, pB = x["bodies"]
        bullets.append(f"{pA} {x['type']} {pB} (orb {x['orb']}°, {x['score_contrib']:+.2f})")
    bullet_text = "\n".join([f"- {b}" for b in bullets])

    prompt = f"""
너는 시나스트리(점성 궁합) 해석가다. 아래 데이터를 참고해 한국어로 4~6개 핵심 포인트를 간결히 설명하고,
마지막에 2줄 요약을 넣어라. 단정적 예언/운명 규정은 피하고, ‘경향’과 ‘활용 팁’ 중심으로 말해라.

[데이터]
- 전체 스코어: {req.score}
- 상위 상호작용:
{bullet_text}
"""

    # 모델은 네 플랜에서 사용 가능한 것으로 지정
    resp = client.responses.create(
        model="gpt-4o-mini",  # 가능 모델로 바꿔도 됨
        input=prompt,
        max_output_tokens=500,
        temperature=0.7,
    )
    text = resp.output_text
    return {"reading": text}
