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
PLANETS = ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"]
ASPECTS = {"conjunction": 0, "sextile": 60, "square": 90, "trine": 120, "opposition": 180}
DEFAULT_ORB = {
    "Sun": 6.0, "Moon": 6.0, "Mercury": 5.0, "Venus": 5.0,
    "Mars": 5.0, "Jupiter": 4.0, "Saturn": 4.0
}
PAIR_WEIGHT = {
    ("Sun","Moon"): 20, ("Venus","Mars"): 18, ("Sun","Venus"): 12, ("Sun","Mars"): 10,
    ("Moon","Venus"): 12, ("Moon","Mars"): 12, ("Mercury","Mercury"): 8,
    ("Venus","Venus"): 6, ("Mars","Mars"): 6,
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

# ---------- Skyfield ----------
ts = load.timescale()
eph = load("de421.bsp")
EARTH = eph["earth"]
PLANET_MAP = {
    "Sun": eph["sun"], "Moon": eph["moon"], "Mercury": eph["mercury"],
    "Venus": eph["venus"], "Mars": eph["mars"],
    "Jupiter": eph["jupiter barycenter"], "Saturn": eph["saturn barycenter"]
}

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

@app.post("/generate-reading")
def generate_reading(req: ReadingReq):
    try:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")

        # 핵심 상호작용 정리
        bullets = []
        for x in req.aspects_top[:10]:  # 최대 10개까지 반영
            pA, pB = x["bodies"]
            bullets.append(f"{pA} {x['type']} {pB} (orb {x['orb']}°, {x['score_contrib']:+.2f})")
        bullet_text = "\n".join([f"- {b}" for b in bullets])

        # ✨ 감성형 프롬프트
        prompt = f"""
너는 시나스트리(점성 궁합)를 예술적으로 해석하는 점성 작가다.
아래 데이터를 참고하여 두 사람의 관계를 시적이고 감정적으로 해석하되, 과장하거나 운명론적으로 단정 짓지 말아라.
해석은 **최소 10문장 이상**으로, 각 문장은 감정과 상징이 어우러지되, 실제 관계의 심리적 흐름을 드러내야 한다.

규칙:
1. 문체는 Tumblr의 점성술 블로그처럼 부드럽고 은유적이며 감정선이 있다.
2. 각 상호작용은 관계의 느낌, 감정의 방향, 배움의 의미를 중심으로 설명한다.
3. 외행성(천왕성·해왕성·명왕성)의 영향은 ‘변화, 영감, 재생’의 상징으로 표현하라.
4. 결론 부분에는 이 관계가 서로에게 남길 정서적 의미를 2~3문장으로 요약한다.
5. 해석 전체는 자연스럽게 이어지는 하나의 이야기처럼 구성한다.

[관계 데이터]
- 전체 스코어: {req.score}
- 주요 상호작용:
{bullet_text}

출력 예시(참고 스타일):
두 사람의 별자리는 오래된 별빛이 다시 만나는 것처럼 서로를 알아본다.
태양과 달의 만남은 내면의 중심을 따뜻하게 밝히고, 금성과 화성의 교차는 숨겨진 열정을 일깨운다.
토성과 목성은 서로의 꿈을 다르게 해석하지만, 그 차이 속에서 성장의 씨앗이 싹튼다.
천왕성과 해왕성의 전류는 관계에 자유와 신비를 더하고, 명왕성은 서로의 그림자를 비추며 변화의 불을 붙인다.
결국 이 관계는 서로를 통해 자신을 발견하는 긴 여정이다.
"""

        # GPT 요청
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=900,
            temperature=0.8,
        )
        return {"reading": resp.output_text}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"reading": None, "error": str(e)}, status_code=500)
