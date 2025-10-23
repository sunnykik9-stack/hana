import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dateutil import parser
from typing import List, Dict
from math import fmod
import pytz

# --- OpenAI 연결 ---
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Skyfield for astronomy ---
from skyfield.api import load

# ---------- FastAPI ----------
app = FastAPI(title="Synastry MVP")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------- Input Schemas ----------
class Person(BaseModel):
    name: str
    date: str
    time: str
    tz: str
    lat: float | None = None
    lon: float | None = None

class SynastryReq(BaseModel):
    personA: Person
    personB: Person

# ---------- Config ----------
# 외행성 포함
PLANETS = [
    "Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn",
    "Uranus","Neptune","Pluto"
]

ASPECTS = {"conjunction": 0, "sextile": 60, "square": 90, "trine": 120, "opposition": 180}

# 기본 허용 오브(대략적 권장치)
DEFAULT_ORB = {
    "Sun": 6.0, "Moon": 6.0, "Mercury": 5.0, "Venus": 5.0, "Mars": 5.0,
    "Jupiter": 4.0, "Saturn": 4.0,
    "Uranus": 3.5, "Neptune": 3.5, "Pluto": 3.0
}

# 쌍 가중치(개인행성↑, 사회/외행성은 중간; 플루토/금·화는 강하게)
PAIR_WEIGHT = {
    # 개인행성–개인행성
    ("Sun","Moon"): 20, ("Venus","Mars"): 18, ("Sun","Venus"): 12, ("Sun","Mars"): 10,
    ("Moon","Venus"): 12, ("Moon","Mars"): 12, ("Mercury","Mercury"): 8,
    ("Venus","Venus"): 6, ("Mars","Mars"): 6,
    # 개인–사회/외행성 (대표 조합만 가중치 명시, 나머지는 디폴트 5 사용)
    ("Sun","Jupiter"): 10, ("Sun","Saturn"): 11,
    ("Moon","Jupiter"): 10, ("Moon","Saturn"): 11,
    ("Venus","Jupiter"): 9, ("Mars","Jupiter"): 9,
    ("Venus","Saturn"): 11, ("Mars","Saturn"): 11,

    ("Sun","Uranus"): 10, ("Sun","Neptune"): 10, ("Sun","Pluto"): 14,
    ("Moon","Uranus"): 10, ("Moon","Neptune"): 12, ("Moon","Pluto"): 14,
    ("Venus","Uranus"): 12, ("Venus","Neptune"): 12, ("Venus","Pluto"): 15,
    ("Mars","Uranus"): 12, ("Mars","Neptune"): 10, ("Mars","Pluto"): 16,
}

ASPECT_MULT = {
    "conjunction": 1.00, "trine": 0.85, "sextile": 0.70,
    "square": -0.65, "opposition": -0.80
}


# ---------- Utility ----------
def _angle_wrap(deg: float) -> float:
    x = fmod(deg, 360.0)
    return x + 360.0 if x < 0 else x

def angle_diff(a: float, b: float) -> float:
    d = abs(_angle_wrap(a) - _angle_wrap(b))
    return d if d <= 180 else 360 - d

def orb_allow(p1: str, p2: str) -> float:
    return (DEFAULT_ORB.get(p1, 4.0) + DEFAULT_ORB.get(p2, 4.0)) / 2

def linear_falloff(delta: float, allow: float) -> float:
    return max(0.0, 1.0 - (delta / allow))

def ordered_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted([a, b]))

# ---------- Skyfield init (외행성 포함) ----------
from functools import lru_cache
from skyfield.api import load

@lru_cache(maxsize=1)
def get_ephem():
    ts = load.timescale()
    eph = load("de421.bsp")
    earth = eph["earth"]
    planet_map = {
        "Sun": eph["sun"],
        "Moon": eph["moon"],
        "Mercury": eph["mercury"],
        "Venus": eph["venus"],
        "Mars": eph["mars"],
        "Jupiter": eph["jupiter barycenter"],
        "Saturn": eph["saturn barycenter"],
        "Uranus": eph["uranus barycenter"],
        "Neptune": eph["neptune barycenter"],
        "Pluto": eph["pluto barycenter"],
    }
    return ts, earth, planet_map

# 아래는 기존 코드의 ecliptic_longitudes 함수 일부 수정
def ecliptic_longitudes(t):
    ts, earth, planet_map = get_ephem()
    longs = {}
    observer = earth.at(t)
    for name, body in planet_map.items():
        app = observer.observe(body).apparent()
        ra, dec, _ = app.radec()
        longs[name] = _angle_wrap(ra.hours * 15.0)
    return longs

def to_ts(date_str: str, time_str: str, tz_name: str):
    local = pytz.timezone(tz_name)
    dt_local = local.localize(parser.parse(f"{date_str} {time_str}"))
    dt_utc = dt_local.astimezone(pytz.utc)
    return ts.utc(dt_utc.year, dt_utc.month, dt_utc.day,
                  dt_utc.hour, dt_utc.minute, dt_utc.second)

def ecliptic_longitudes(t):
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
        return f"{a['bodies'][0]} {a['type']} {a['bodies'][1]} (orb {a['orb']}°, {a['score_contrib']:+.2f})"
    lines = [f"- {label(x)}" for x in scored_top[:6]]
    return "핵심 상호작용:\n" + "\n".join(lines)

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
        "summary": summary,
        "notes": "MVP version"
    })

# ---------- GPT 해석 엔드포인트 ----------
class ReadingReq(BaseModel):
    score: float
    aspects_top: List[Dict]

# ---------- GPT 감성 해석 (Tumblr-style, 외행성 포함) ----------
class ReadingReq(BaseModel):
    score: float
    aspects_top: List[Dict]

# ---------- GPT 감성 해석 (Tumblr-style, 외행성 포함, per-aspect + overall) ----------
class ReadingReq(BaseModel):
    score: float
    aspects_top: List[Dict]

@app.post("/generate-reading")
def generate_reading(req: ReadingReq):
    try:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")

        topN = min(6, len(req.aspects_top))
        picked = req.aspects_top[:topN]

        lines = []
        for i, x in enumerate(picked, start=1):
            pA, pB = x["bodies"]
            lines.append(f"{i}. {pA} {x['type']} {pB} | orb {x['orb']}° | contrib {x['score_contrib']:+.2f}")
        bullet_text = "\n".join(lines)

        prompt = f"""
너는 시나스트리(점성 궁합)를 시적으로 해석하는 작가다. 아래 제공된 상위 상호작용 각각을
**최소 1,000자 이상**의 풍부한 한국어 문단으로 해석하라. Tumblr의 점성술 블로그처럼 감정선과 은유가 살아 있어야 한다.
과장/운명 단정 금지. 각 문단에는 (1) 관계가 주는 느낌, (2) 심리/행동 패턴, (3) 배움/조언을 포함한다.
또한 외행성(천왕성·해왕성·명왕성)의 상징은 ‘자유/각성’, ‘영감/포용’, ‘변화/재생’ 맥락으로 자연스럽게 녹여쓴다.

마지막에는 **전체 해석**을 별도 문단으로 작성하라(최소 8문장).
전체 해석에는 개별 요소들이 한 흐름으로 어떻게 엮이는지, 관계가 남기는 정서적 의미와 성장의 방향을 제시한다.

출력 형식(마크다운):
### 1. [bodies] — [aspect]
(최소 1,000자 문단)

### 2. ...
(각 상호작용마다 위와 같은 형식으로 문단 생성)

### 전체 해석
(최소 8문장 이상의 마무리 문단)

[관계 데이터]
- 전체 스코어: {req.score}
- 상위 상호작용:
{bullet_text}
"""

        resp = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=3200,
            temperature=0.85,
        )
        return {"reading": resp.output_text}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"reading": None, "error": str(e)}, status_code=500)
