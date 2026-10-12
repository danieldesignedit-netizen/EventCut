"""Google AI (Gemini) as EventCut's story brain.

The engine sends Google a small copy of each clip (360p, one frame a second, mono sound: a few MB per minute;
the full-quality footage never leaves the PC). Gemini watches and listens, and returns complete soundbites with
times, quotes, who is speaking (by position in the frame), and how well each works as an opening hook or a
closing payoff. Results are saved per clip, so regenerating or changing settings costs nothing.
"""
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import media
from .config import HOME, load_settings

API = os.environ.get("EVENTCUT_GEMINI_API", "https://generativelanguage.googleapis.com")  # override only for testing
CACHE = HOME / "Projects" / "_ai"
PROMPT_VERSION = "p4"
# Tokens Google counts per second of video at low media resolution, one frame a second (70) plus sound (25).
VIDEO_TOK_S, AUDIO_TOK_S = 70, 25
# US$ per million tokens. Prices change: they can be edited in Settings (google_prices).
DEFAULT_PRICES = {"flash": {"in": 0.30, "audio": 1.00, "out": 2.50}, "pro": {"in": 1.25, "audio": 1.25, "out": 10.00}}


class AIError(RuntimeError):
    pass


def cfg():
    s = load_settings()
    return {"key": (s.get("google_api_key") or "").strip(), "tier": s.get("google_model", "flash"),
            "cap": float(s.get("google_monthly_cap_usd", 10)), "use": bool(s.get("use_google_ai", True)),
            "prices": {**DEFAULT_PRICES, **(s.get("google_prices") or {})}}


def enabled() -> bool:
    c = cfg()
    return bool(c["key"]) and c["use"]


# ------------------------------------------------------------------ HTTP
def _req(method, url, key, body=None, headers=None, timeout=120, raw=False):
    h = {"x-goog-api-key": key, **(headers or {})}
    data = None
    if body is not None:
        if isinstance(body, (bytes, bytearray)):
            data = bytes(body)
        else:
            data = json.dumps(body).encode()
            h.setdefault("Content-Type", "application/json")
    r = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp if raw else json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace")
        try:
            msg = json.loads(msg).get("error", {}).get("message", msg)
        except Exception:
            pass
        if e.code in (401, 403) or "API key not valid" in msg:
            raise AIError("Google rejected the API key. Check it in Settings, and that the Gemini API is enabled for it.")
        if e.code == 429:
            raise AIError("Google AI's usage limit was reached. Wait a minute, or turn on billing for the key to raise limits.")
        raise AIError(f"Google AI error ({e.code}): {msg[:300]}")
    except urllib.error.URLError as e:
        raise AIError(f"Couldn't reach Google AI. Check the internet connection. ({e.reason})")


# ------------------------------------------------------------------ models
def pick_model(key, tier="flash"):
    """Newest generally available Flash or Pro model this key can use (Google retires old names over time)."""
    cache = CACHE / "models.json"
    CACHE.mkdir(parents=True, exist_ok=True)
    try:
        c = json.loads(cache.read_text())
        if time.time() - c.get("at", 0) < 86400 and c.get("tier") == tier:
            return c["model"]
    except Exception:
        pass
    best, best_v = None, -1.0
    try:
        page = ""
        for _ in range(5):
            d = _req("GET", f"{API}/v1beta/models?pageSize=200" + (f"&pageToken={page}" if page else ""), key, timeout=30)
            for m in d.get("models", []):
                name = m.get("name", "").split("/")[-1]
                if "generateContent" not in m.get("supportedGenerationMethods", ["generateContent"]):
                    continue
                mm = re.fullmatch(rf"gemini-(\d+(?:\.\d+)?)-{tier}", name)
                if mm and float(mm.group(1)) > best_v:
                    best, best_v = name, float(mm.group(1))
            page = d.get("nextPageToken", "")
            if not page:
                break
    except AIError:
        raise
    except Exception:
        pass
    model = best or f"gemini-{tier}-latest"
    cache.write_text(json.dumps({"at": time.time(), "tier": tier, "model": model}))
    return model


# ------------------------------------------------------------------ cost
def estimate(seconds, tier=None):
    c = cfg()
    tier = tier or c["tier"]
    p = c["prices"].get(tier, DEFAULT_PRICES["flash"])
    vid, aud, out = seconds * VIDEO_TOK_S, seconds * AUDIO_TOK_S, 6000 + 40 * seconds / 60
    usd = vid / 1e6 * p["in"] + aud / 1e6 * p["audio"] + out / 1e6 * p["out"]
    return {"tokens": int(vid + aud), "usd": round(usd, 3), "tier": tier}


def spend_file():
    return CACHE / "spend.json"


def month_spend():
    try:
        return float(json.loads(spend_file().read_text()).get(time.strftime("%Y-%m"), 0))
    except Exception:
        return 0.0


def log_spend(usage, tier):
    p = cfg()["prices"].get(tier, DEFAULT_PRICES["flash"])
    prompt = usage.get("promptTokenCount", 0)
    audio_t = sum(d.get("tokenCount", 0) for d in usage.get("promptTokensDetails", []) if d.get("modality") == "AUDIO")
    out = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
    usd = (prompt - audio_t) / 1e6 * p["in"] + audio_t / 1e6 * p["audio"] + out / 1e6 * p["out"]
    CACHE.mkdir(parents=True, exist_ok=True)
    try:
        d = json.loads(spend_file().read_text())
    except Exception:
        d = {}
    m = time.strftime("%Y-%m")
    d[m] = round(d.get(m, 0) + usd, 4)
    spend_file().write_text(json.dumps(d))
    return usd


# ------------------------------------------------------------------ upload
def make_ai_copy(path, out):
    """360p, 1 frame a second, mono 48 kb/s: what Gemini needs, a few MB per minute."""
    if out.exists() and out.stat().st_size > 1000:
        return out
    ff, _ = media.tools()
    media.run_decoding([ff, "-y", "-v", "error"] + media.decode_args() + ["-i", path, "-vf", "fps=1,scale=-2:360", "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "32", "-c:a", "aac", "-b:a", "48k", "-ac", "1", "-movflags", "+faststart", str(out)])
    return out


def upload(key, file_path, progress=None):
    size = file_path.stat().st_size
    start = _req("POST", f"{API}/upload/v1beta/files", key, body={"file": {"display_name": file_path.name}}, raw=True,
                 headers={"X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                          "X-Goog-Upload-Header-Content-Length": str(size), "X-Goog-Upload-Header-Content-Type": "video/mp4"})
    up_url = start.headers.get("x-goog-upload-url") or start.headers.get("X-Goog-Upload-URL")
    if not up_url:
        raise AIError("Google AI didn't accept the upload.")
    chunk, off, info = 8 * 1024 * 1024, 0, None
    with open(file_path, "rb") as f:
        while off < size:
            data = f.read(chunk)
            last = off + len(data) >= size
            r = _req("POST", up_url, key, body=data, timeout=300, headers={
                "X-Goog-Upload-Offset": str(off), "X-Goog-Upload-Command": "upload, finalize" if last else "upload",
                "Content-Type": "application/octet-stream"}, raw=not last)
            off += len(data)
            if progress:
                progress(off / size)
            if last:
                info = r
    f_ = (info or {}).get("file", info or {})
    name = f_.get("name")
    for _ in range(240):  # Google prepares the video (usually seconds to a few minutes)
        if f_.get("state") == "ACTIVE":
            return f_
        if f_.get("state") == "FAILED":
            raise AIError("Google AI couldn't read this clip.")
        time.sleep(3)
        f_ = _req("GET", f"{API}/v1beta/{name}", key, timeout=30)
    raise AIError("Google AI took too long to prepare the video.")


# ------------------------------------------------------------------ the request
SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {"type": "STRING"},
        "title": {"type": "STRING"},
        "people": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "position": {"type": "STRING"}, "description": {"type": "STRING"}}}},
        "bites": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "start": {"type": "STRING"}, "end": {"type": "STRING"}, "quote": {"type": "STRING"},
            "position": {"type": "STRING"}, "topic": {"type": "STRING"}, "why": {"type": "STRING"},
            "score": {"type": "NUMBER"}, "hook": {"type": "NUMBER"}, "payoff": {"type": "NUMBER"}},
            "required": ["start", "end", "quote", "score", "hook", "payoff"]}},
    },
    "required": ["bites"],
}


def _prompt(event_type, duration_s):
    return f"""You are a senior video editor who cuts viral short-form videos from long recordings.
This video is a {event_type or 'company event'} recording, {int(duration_s // 60)} min {int(duration_s % 60)} s long.

Language: speakers may use Nigerian Pidgin English, English, or a mix, with some Yoruba, Igbo or Hausa words.
Understand all of it, judge humour and meaning in that context, and quote exactly what is said in the speakers'
own words using common Nigerian Pidgin spelling (for example: wetin, abeg, dey, na, oya, sabi, wahala, comot,
una, abi, sha, o). Never translate or "correct" Pidgin into standard English.

Find the strongest soundbites for short social videos (Reels, TikTok, Shorts). Rules:
- Each soundbite is a COMPLETE thought: it starts at the first word of a sentence and ends at the last word of a
  sentence. Never start mid-sentence or with a filler ("and", "so", "um"). It must make sense on its own.
- 3 to 20 seconds each. Prefer ones that are funny, surprising, bold, emotional, controversial, quotable or useful.
- Skip greetings, housekeeping, crosstalk and anything that needs earlier context.
- Return up to 40, best first, spread across the whole recording.

For each soundbite give:
- start, end: seconds from the start of THIS video as decimals (e.g. "754.2"), at sentence boundaries.
- quote: the exact words spoken.
- position: where the speaker is in the frame: "left", "center", "right", or "single" if only one person is shown.
- topic: 2 to 5 words. why: one short reason it works.
- score 0-10: how strongly it would stop someone scrolling and keep them watching.
- hook 0-10: how well it works as the very FIRST line of a video (bold claim, surprising, funny, provocative).
- payoff 0-10: how well it works as the LAST line (punchline, big laugh, strong conclusion, memorable).

Also give: people (each person's position and a short neutral description of their clothing), a 2-sentence summary,
and a title idea under 8 words."""


def _parse_t(v):
    s = str(v).strip()
    if re.fullmatch(r"\d+(?::\d{1,2}){1,2}(\.\d+)?", s):
        parts = [float(x) for x in s.split(":")]
        return sum(p * 60 ** (len(parts) - 1 - i) for i, p in enumerate(parts))
    try:
        return float(re.sub(r"[^0-9.]", "", s) or 0)
    except ValueError:
        return 0.0


def generate(key, model, file_uri, prompt, schema=True):
    body = {"contents": [{"role": "user", "parts": [{"file_data": {"mime_type": "video/mp4", "file_uri": file_uri}}, {"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "response_mime_type": "application/json", "media_resolution": "MEDIA_RESOLUTION_LOW"}}
    if schema:
        body["generationConfig"]["response_schema"] = SCHEMA
    try:
        return _req("POST", f"{API}/v1beta/models/{model}:generateContent", key, body=body, timeout=900)
    except AIError as e:  # older/newer models may not take every option: retry plainly
        if "(400)" in str(e):
            body["generationConfig"].pop("media_resolution", None)
            body["generationConfig"].pop("response_schema", None)
            return _req("POST", f"{API}/v1beta/models/{model}:generateContent", key, body=body, timeout=900)
        raise


def _json_from(resp):
    parts = (resp.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text:
        reason = (resp.get("candidates") or [{}])[0].get("finishReason") or resp.get("promptFeedback", {}).get("blockReason")
        raise AIError(f"Google AI returned nothing for this clip ({reason or 'no answer'}).")
    text = text[text.index("{"): text.rindex("}") + 1] if "{" in text else text
    return json.loads(text)


def analyse_clip(path, event_type, duration_s, key_for_cache, progress=None):
    """Soundbites for one clip. Cached by clip + model tier + prompt version."""
    c = cfg()
    if not c["key"]:
        raise AIError("Add your Google AI key in Settings first.")
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha1(f"{key_for_cache}|{c['tier']}|{PROMPT_VERSION}|{event_type}".encode()).hexdigest()[:16]
    out = CACHE / f"{tag}.json"
    if out.exists():
        return json.loads(out.read_text("utf-8"))
    if month_spend() >= c["cap"]:
        raise AIError(f"Google AI has reached this month's spending cap (US${c['cap']:.0f}). Raise it in Settings.")
    model = pick_model(c["key"], c["tier"])
    say = progress or (lambda f, note="": None)
    say(0.02, "Making a small copy for Google AI")
    copy = make_ai_copy(path, CACHE / f"{key_for_cache}_ai.mp4")
    say(0.1, "Sending it to Google AI")
    f = upload(c["key"], copy, lambda x: say(0.1 + 0.3 * x, f"Sending to Google AI, {int(x * 100)}%"))
    say(0.45, "Google AI is watching and listening")
    resp = generate(c["key"], model, f["uri"], _prompt(event_type, duration_s))
    usd = log_spend(resp.get("usageMetadata", {}), c["tier"])
    data = _json_from(resp)
    bites = []
    for b in data.get("bites", []):
        s, e = _parse_t(b.get("start")), _parse_t(b.get("end"))
        if e - s >= 1.5 and e <= duration_s + 2:
            bites.append({**b, "start": round(s, 2), "end": round(e, 2), "score": float(b.get("score", 5)),
                          "hook": float(b.get("hook", 0)), "payoff": float(b.get("payoff", 0))})
    result = {"model": model, "bites": bites, "people": data.get("people", []), "summary": data.get("summary", ""),
              "title": data.get("title", ""), "usd": round(usd, 4)}
    out.write_text(json.dumps(result), "utf-8")
    try:  # tidy up on Google's side (files expire anyway after 48 h)
        _req("DELETE", f"{API}/v1beta/{f['name']}", c["key"], timeout=20)
    except Exception:
        pass
    say(1.0, "Google AI is done")
    return result


def test_key(key, tier=None):
    """Quick check from Settings: the key works and which model will be used."""
    model = pick_model(key, tier or cfg()["tier"])
    _req("GET", f"{API}/v1beta/models/{model}", key, timeout=20)
    return model


# ------------------------------------------------------------------ colourist: per-clip grading decisions
COLOUR_VERSION = "c1"
COLOUR_SCHEMA = {"type": "OBJECT", "properties": {
    "temperature": {"type": "NUMBER"}, "tint": {"type": "NUMBER"}, "exposure": {"type": "NUMBER"},
    "contrast": {"type": "NUMBER"}, "highlights": {"type": "NUMBER"}, "shadows": {"type": "NUMBER"},
    "saturation": {"type": "NUMBER"}, "reduce_green": {"type": "NUMBER"}, "note": {"type": "STRING"}},
    "required": ["temperature", "tint", "exposure", "contrast", "highlights", "shadows", "saturation", "reduce_green"]}


def _colour_prompt(look, event_type):
    return f"""You are a veteran film colourist grading a {event_type or 'company event'} video.
This frame is from one clip, after basic white balance and exposure correction. The look "{look}" will be applied
on top of your adjustments, so do not recreate the look: make this shot sit well and look expensive before it.

Return gentle adjustments (0 means leave it):
- temperature -1..1 (positive warmer), tint -1..1 (positive magenta, negative green)
- exposure -0.7..0.7 stops, contrast -0.5..0.5
- highlights -0.5..0.5 (negative recovers bright skies, screens and stage lights)
- shadows -0.5..0.5 (positive opens up dark areas; keep blacks rich)
- saturation -0.5..0.5
- reduce_green 0..1: remove a green cast from LED or fluorescent light, or tame neon-green walls. Keep real greens
  such as grass, brand colours or deliberate set lighting.
- note: one short sentence on what you changed.

Priorities, in order: natural, healthy, rich skin for darker skin tones (never grey, ashy, orange or washed out);
no clipped highlights on faces; a clean, cinematic image; consistency with other shots of the same event.
Be subtle. Most values should be between -0.3 and 0.3. Use 0 when unsure."""


def colourist(frame_jpg, look, event_type, cache_tag):
    """Adjustments for one clip from one representative frame. Cached; a fraction of a cent per clip."""
    import base64
    c = cfg()
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha1(f"{cache_tag}|{look}|{c['tier']}|{COLOUR_VERSION}".encode()).hexdigest()[:16]
    out = CACHE / f"colour_{tag}.json"
    if out.exists():
        return json.loads(out.read_text("utf-8"))
    if month_spend() >= c["cap"]:
        raise AIError("Google AI has reached this month's spending cap.")
    model = pick_model(c["key"], c["tier"])
    data = base64.b64encode(Path(frame_jpg).read_bytes()).decode()
    body = {"contents": [{"role": "user", "parts": [{"inline_data": {"mime_type": "image/jpeg", "data": data}},
                                                     {"text": _colour_prompt(look, event_type)}]}],
            "generationConfig": {"temperature": 0.2, "response_mime_type": "application/json", "response_schema": COLOUR_SCHEMA}}
    try:
        resp = _req("POST", f"{API}/v1beta/models/{model}:generateContent", c["key"], body=body, timeout=120)
    except AIError as e:
        if "(400)" not in str(e):
            raise
        body["generationConfig"].pop("response_schema", None)
        resp = _req("POST", f"{API}/v1beta/models/{model}:generateContent", c["key"], body=body, timeout=120)
    log_spend(resp.get("usageMetadata", {}), c["tier"])
    d = _json_from(resp)
    lim = {"temperature": 1, "tint": 1, "exposure": 0.7, "contrast": 0.5, "highlights": 0.5, "shadows": 0.5, "saturation": 0.5}
    p = {k: round(max(-v, min(v, float(d.get(k, 0) or 0))), 3) for k, v in lim.items()}
    p["reduce_green"] = round(max(0.0, min(1.0, float(d.get("reduce_green", 0) or 0))), 3)
    p["note"] = str(d.get("note", ""))[:200]
    out.write_text(json.dumps(p), "utf-8")
    return p
