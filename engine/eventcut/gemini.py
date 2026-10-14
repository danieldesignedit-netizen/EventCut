"""Google AI (Gemini) as EventCut's story brain.

The engine sends Google a small copy of each clip (360p, one frame a second, mono sound: a few MB per minute;
the full-quality footage never leaves the PC). Gemini watches and listens, and returns complete soundbites with
times, quotes, who is speaking (by position in the frame), and how well each works as an opening hook or a
closing payoff. Results are saved per clip, so regenerating or changing settings costs nothing.
"""
import hashlib
import json
import os
import threading
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import media
from .config import HOME, WORK, load_settings

API = os.environ.get("EVENTCUT_GEMINI_API", "https://generativelanguage.googleapis.com")  # override only for testing
CACHE = WORK / "_ai"
PROMPT_VERSION = "p4"
# Tokens Google counts per second of video at low media resolution, one frame a second (70) plus sound (25).
VIDEO_TOK_S, AUDIO_TOK_S = 70, 25
# US$ per million tokens. Prices change: they can be edited in Settings (google_prices).
DEFAULT_PRICES = {"flash": {"in": 0.50, "audio": 1.00, "out": 3.00}, "pro": {"in": 2.00, "audio": 2.00, "out": 12.00}}


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
_wait_note = threading.local()  # set by the caller to show "waiting for internet" in the progress bar
NET_WAIT_MAX = 45 * 60  # keep waiting for a lost connection for up to 45 minutes
# The current job (shared by all threads): its progress note, a time limit and a way to stop at once when cancelled.
RUN = {"note": None, "deadline": None, "cancelled": None}


DEAD = {"until": 0, "why": ""}  # set when the prepaid balance is empty: no more requests for a while


def _stop_check():
    c = RUN.get("cancelled")
    if c and c():
        raise InterruptedError
    d = RUN.get("deadline")
    if d and time.time() > d:
        raise AIError("Google AI took longer than the time allowed for this step.")


def _req(method, url, key, body=None, headers=None, timeout=120, raw=False):
    """A request that survives internet drops: it waits and tries again rather than failing the video.
    The time spent waiting for answers counts too, so a slow connection can't stretch a step forever."""
    t0, delay = time.time(), 5.0
    while True:
        _stop_check()
        d = RUN.get("deadline")
        t_out = max(15, min(timeout, d - time.time())) if d else timeout
        try:
            return _req_once(method, url, key, body, headers, t_out, raw)
        except AIError as e:
            waited = time.time() - t0
            if not str(e).startswith("Couldn't reach Google AI") or waited >= getattr(_wait_note, "max", NET_WAIT_MAX):
                raise
            note = getattr(_wait_note, "fn", None) or RUN.get("note")
            if note:
                note(f"Waiting for the internet to come back (tried for {int(waited // 60)} min)")
            for _ in range(int(delay)):
                _stop_check()
                time.sleep(1)
            delay = min(60.0, delay * 1.6)


def _req_once(method, url, key, body=None, headers=None, timeout=120, raw=False):
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
        if e.code == 402 or "prepayment" in msg.lower() or "credits are depleted" in msg.lower() or "billing" in msg.lower() and e.code in (400, 403):
            DEAD.update(until=time.time() + 1800, why="credit")
            raise AIError("Your Google AI prepaid credit has run out. Top it up in Google AI Studio → Billing.")
        if e.code == 429:
            raise AIError("Google AI's usage limit was reached. Wait a minute, or turn on billing for the key to raise limits.")
        raise AIError(f"Google AI error ({e.code}): {msg[:300]}")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        raise AIError(f"Couldn't reach Google AI. Check the internet connection. ({getattr(e, 'reason', e)})")


# ------------------------------------------------------------------ models
BAD_FILE = CACHE / "models_closed.json"
_switches = []  # plain notes about model changes during this run, shown on the video
_lock_m = threading.Lock()


def _bad():
    try:
        d = json.loads(BAD_FILE.read_text())
        return {k for k, t in d.items() if time.time() - t < 30 * 86400}
    except Exception:
        return set()


def _mark_bad(model):
    with _lock_m:
        try:
            d = json.loads(BAD_FILE.read_text())
        except Exception:
            d = {}
        d[model] = time.time()
        CACHE.mkdir(parents=True, exist_ok=True)
        BAD_FILE.write_text(json.dumps(d))


def _model_names(key):
    """All Gemini model names this key can use (saved for a day)."""
    cache = CACHE / "models_list.json"
    CACHE.mkdir(parents=True, exist_ok=True)
    try:
        c = json.loads(cache.read_text())
        if time.time() - c.get("at", 0) < 86400 and c.get("names"):
            return c["names"]
    except Exception:
        pass
    names, page = [], ""
    try:
        for _ in range(5):
            d = _req("GET", f"{API}/v1beta/models?pageSize=200" + (f"&pageToken={page}" if page else ""), key, timeout=30)
            for m in d.get("models", []):
                if "generateContent" in m.get("supportedGenerationMethods", ["generateContent"]):
                    names.append(m.get("name", "").split("/")[-1])
            page = d.get("nextPageToken", "")
            if not page:
                break
    except AIError:
        raise
    except Exception:
        pass
    if names:
        cache.write_text(json.dumps({"at": time.time(), "names": names}))
    return names


def pretty(model):
    """gemini-3.1-pro-preview -> Gemini 3.1 Pro (preview)"""
    m = re.match(r"gemini-([\d.]+)-(\w+)(-preview)?", model or "")
    if not m:
        return model or "the Google AI model"
    return f"Gemini {m.group(1)} {m.group(2).capitalize()}" + (" (preview)" if m.group(3) else "")


def pick_model(key, tier="flash"):
    """Newest Flash or Pro model this key can use. A stable model is preferred; a preview is used when it is newer
    than every stable one (Google sometimes offers only a preview of its newest model). Models Google has closed are
    skipped."""
    bad = _bad()
    forced = _forced.get(tier)
    if forced and forced not in bad:
        return forced
    best, best_k = None, None
    for name in _model_names(key):
        mm = re.fullmatch(rf"gemini-(\d+(?:\.\d+)?)-{tier}(-preview(?:-[0-9a-z-]+)?)?", name)
        if not mm or name in bad:
            continue
        k = (float(mm.group(1)), 0 if mm.group(2) else 1, name)
        if best_k is None or k > best_k:
            best, best_k = name, k
    if best:
        return best
    fallback = f"gemini-{tier}-latest"
    return fallback if fallback not in bad else "gemini-flash-latest"


_forced = {}  # the model Google itself told us to use instead of a closed one


def _gone(msg):
    m = msg.lower()
    return ("(404)" in m or "no longer available" in m or "not found for api version" in m or "is not supported for generatecontent" in m
            or "has been deprecated" in m or "is not found" in m)


def take_switch_notes():
    with _lock_m:
        out = list(dict.fromkeys(_switches))
        _switches.clear()
    return out


def call_model(c, body, timeout=900, fallback_flash=True):
    """generateContent with the right model. If Google has closed a model, EventCut moves to the model Google
    names (or the newest one available), and from Pro to Flash if Pro can't be used; the change is reported."""
    if DEAD["until"] > time.time():
        raise AIError("Your Google AI prepaid credit has run out. Top it up in Google AI Studio → Billing.")
    tier = c["tier"]
    body.setdefault("generationConfig", {}).setdefault("maxOutputTokens", 32768)  # room for the whole answer
    first = None
    for attempt in range(5):
        model = pick_model(c["key"], tier)
        first = first or model
        url = f"{API}/v1beta/models/{model}:generateContent"
        try:
            try:
                resp = _req("POST", url, c["key"], body=body, timeout=timeout)
            except AIError as e:  # older/newer models may not take every option: retry plainly
                if "(400)" not in str(e) or _gone(str(e)):
                    raise
                gc = body.get("generationConfig", {})
                if not any(k in gc for k in ("media_resolution", "response_schema")):
                    raise
                gc.pop("media_resolution", None)
                gc.pop("response_schema", None)
                resp = _req("POST", url, c["key"], body=body, timeout=timeout)
            if model != first:
                with _lock_m:
                    _switches.append(f"{pretty(first)} isn't available to your key any more, so EventCut used {pretty(model)} instead.")
            USED.update(model=model, tier=tier, at=time.time())
            return resp, model, tier
        except AIError as e:
            msg = str(e)
            if not _gone(msg):
                raise
            _mark_bad(model)
            sug = re.search(r"use\s+(?:models/)?(gemini-[0-9a-z.\-]+[0-9a-z])", msg)
            if sug and sug.group(1) not in _bad():
                _forced[tier] = sug.group(1)
            elif _forced.get(tier) == model:
                _forced.pop(tier, None)
            if attempt >= 2 and tier != "flash" and fallback_flash:
                tier = "flash"
    raise AIError(f"No Google AI model could be used: {pretty(first)} and the models after it were refused.")


USED = {}  # the model actually used last (shown in Settings)


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


# US$ per million tokens by model (Google's published prices; a request over 200,000 tokens costs more on Pro).
MODEL_PRICES = [
    (r"gemini-3(\.\d+)?-pro", {"in": 2.00, "audio": 2.00, "out": 12.00, "in_long": 4.00, "out_long": 18.00}),
    (r"gemini-2\.5-pro", {"in": 1.25, "audio": 1.25, "out": 10.00, "in_long": 2.50, "out_long": 15.00}),
    (r"gemini-3\.5-flash", {"in": 1.50, "audio": 1.50, "out": 9.00}),
    (r"gemini-3(\.\d+)?-flash", {"in": 0.50, "audio": 1.00, "out": 3.00}),
    (r"gemini-2\.5-flash", {"in": 0.30, "audio": 1.00, "out": 2.50}),
]
RUN_COST = {"usd": 0.0, "reused": 0, "calls": 0}  # this video's Google cost (reset by the engine for each new video)
_cost_lock = threading.Lock()


def price_for(model, tier):
    for pat, pr in MODEL_PRICES:
        if model and re.match(pat, model):
            return pr
    return cfg()["prices"].get(tier, DEFAULT_PRICES["flash"])


def note_reused():
    with _cost_lock:
        RUN_COST["reused"] += 1


def log_spend(usage, tier, model=None):
    p = price_for(model or USED.get("model"), tier)
    prompt = usage.get("promptTokenCount", 0)
    if prompt > 200_000 and "in_long" in p:  # Google bills the whole request at the long-context rate
        p = dict(p, **{"in": p["in_long"], "audio": p["in_long"], "out": p["out_long"]})
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
    with _cost_lock:
        RUN_COST["usd"] += usd
        RUN_COST["calls"] += 1
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


def upload(key, file_path, progress=None, on_wait=None):
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
        if on_wait and f_.get("state") != "ACTIVE":
            on_wait()
        if f_.get("state") == "ACTIVE":
            return f_
        if f_.get("state") == "FAILED":
            raise AIError("Google AI couldn't read this clip.")
        _stop_check()
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


def generate(c, file_uri, prompt, schema=True):
    body = {"contents": [{"role": "user", "parts": [{"file_data": {"mime_type": "video/mp4", "file_uri": file_uri}}, {"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "response_mime_type": "application/json", "media_resolution": "MEDIA_RESOLUTION_LOW"}}
    if schema:
        body["generationConfig"]["response_schema"] = SCHEMA
    return call_model(c, body, timeout=900)


def _repair_json(text):
    """A cut-off JSON answer: keep every complete item, close what is still open."""
    stack, in_str, esc, last_ok = [], False, False, -1
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            if stack and stack[-1] == "[":
                last_ok = i  # an item inside a list has just finished
            if not stack:
                return json.loads(text[:i + 1])
    if last_ok < 0:
        raise ValueError("nothing complete")
    head = text[:last_ok + 1]
    # close whatever is still open at that point
    st, in_s, es = [], False, False
    for ch in head:
        if in_s:
            if es:
                es = False
            elif ch == "\\":
                es = True
            elif ch == '"':
                in_s = False
            continue
        if ch == '"':
            in_s = True
        elif ch in "{[":
            st.append(ch)
        elif ch in "}]" and st:
            st.pop()
    return json.loads(head + "".join("]" if c == "[" else "}" for c in reversed(st)))


def _json_from(resp, with_status=False):
    """The JSON answer. with_status=True returns (data, complete): a cut-off answer is repaired, keeping every
    complete item, and reported as incomplete."""
    cand = (resp.get("candidates") or [{}])[0]
    parts = cand.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text:
        reason = cand.get("finishReason") or resp.get("promptFeedback", {}).get("blockReason")
        raise AIError(f"Google AI returned nothing for this clip ({reason or 'no answer'}).")
    start = text.find("{")
    body = text[start:] if start >= 0 else text
    try:
        data = json.loads(body[: body.rindex("}") + 1] if "}" in body else body)
        complete = cand.get("finishReason") not in ("MAX_TOKENS",)
    except ValueError:
        try:
            data, complete = _repair_json(body), False
        except ValueError:
            raise AIError("Google AI's answer was incomplete.")
    return (data, complete) if with_status else data


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
    say = progress or (lambda f, note="": None)
    _wait_note.fn = lambda note: say(0.05, note)
    _wait_note.max = NET_WAIT_MAX  # picking the moments is worth waiting for
    say(0.02, "Making a small copy for Google AI")
    copy = make_ai_copy(path, CACHE / f"{key_for_cache}_ai.mp4")
    say(0.1, "Sending it to Google AI")
    f = upload(c["key"], copy, lambda x: say(0.1 + 0.3 * x, f"Sending to Google AI, {int(x * 100)}%"))
    say(0.45, "Google AI is watching and listening")
    resp, model, tier = generate(c, f["uri"], _prompt(event_type, duration_s))
    usd = log_spend(resp.get("usageMetadata", {}), tier, model)
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
    """Check from Settings and Check everything: a real, tiny request with exactly the model EventCut will use
    (a model can be listed but closed to new users; only a real request shows that). Returns the model used."""
    DEAD.update(until=0, why="")  # a test always really asks Google
    c = dict(cfg(), key=key.strip(), tier=tier or cfg()["tier"])
    body = {"contents": [{"role": "user", "parts": [{"text": "Reply with the single word OK."}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 20}}
    _, model, _ = call_model(c, body, timeout=60)
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
    c = dict(cfg(), tier="flash")  # colour is a simple job: Flash does it in about a second, whatever the main setting
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha1(f"{cache_tag}|{look}|{c['tier']}|{COLOUR_VERSION}".encode()).hexdigest()[:16]
    out = CACHE / f"colour_{tag}.json"
    if out.exists():
        note_reused()
        return json.loads(out.read_text("utf-8"))
    if month_spend() >= c["cap"]:
        raise AIError("Google AI has reached this month's spending cap.")
    _wait_note.max = 120  # colour is a finishing touch: wait a couple of minutes for the internet, then skip
    data = base64.b64encode(Path(frame_jpg).read_bytes()).decode()
    body = {"contents": [{"role": "user", "parts": [{"inline_data": {"mime_type": "image/jpeg", "data": data}},
                                                     {"text": _colour_prompt(look, event_type)}]}],
            "generationConfig": {"temperature": 0.2, "response_mime_type": "application/json", "response_schema": COLOUR_SCHEMA}}
    resp, model, tier = call_model(c, body, timeout=120)
    log_spend(resp.get("usageMetadata", {}), tier, model)
    d = _json_from(resp)
    lim = {"temperature": 1, "tint": 1, "exposure": 0.7, "contrast": 0.5, "highlights": 0.5, "shadows": 0.5, "saturation": 0.5}
    p = {k: round(max(-v, min(v, float(d.get(k, 0) or 0))), 3) for k, v in lim.items()}
    p["reduce_green"] = round(max(0.0, min(1.0, float(d.get("reduce_green", 0) or 0))), 3)
    p["note"] = str(d.get("note", ""))[:200]
    out.write_text(json.dumps(p), "utf-8")
    return p


# ------------------------------------------------------------------ sorting mixed-up footage by event
SORT_VERSION = "s1"


def _images_request(images, text, schema):
    """One request with several JPEG images. Returns parsed JSON."""
    import base64
    c = cfg()
    if not c["key"]:
        raise AIError("Add a Google AI key in Settings to sort with Google AI.")
    if month_spend() >= c["cap"]:
        raise AIError("Google AI has reached this month's spending cap.")
    _wait_note.max = 300
    parts = [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(b).decode()}} for b in images]
    parts.append({"text": text})
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.1, "response_mime_type": "application/json", "response_schema": schema}}
    resp, model, tier = call_model(c, body, timeout=180)
    log_spend(resp.get("usageMetadata", {}), tier, model)
    return _json_from(resp)


def _project_line(event, hint):
    return f'The project is a video of this event: "{event or "a company event"}".' + (f' The editor describes it as: "{hint}".' if hint else "")


OVERVIEW_SCHEMA = {"type": "OBJECT", "properties": {
    "events": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "id": {"type": "STRING"}, "label": {"type": "STRING"}, "clues": {"type": "STRING"},
        "matches": {"type": "STRING", "enum": ["yes", "no", "unsure"]}, "why": {"type": "STRING"},
        "groups": {"type": "ARRAY", "items": {"type": "STRING"}}},
        "required": ["id", "label", "clues", "matches", "why", "groups"]}}},
    "required": ["events"]}


def sort_overview(overview, event, hint):
    """overview: [(group, sheet_jpeg)]. Names the distinct events in the footage, e.g.
    [{"id": "e1", "label": "CSR school visit", "clues": "...", "matches": "yes", "why": "...", "groups": ["g1"]}]"""
    text = f"""EVENTCUT_SORT_OVERVIEW
{_project_line(event, hint)}
Someone mixed footage from different events into one folder. Each image is a contact sheet of one recording session;
the label on every thumbnail is the session id ({", ".join(g["id"] for g, _ in overview)}). Sessions, in the same order as the images:
""" + "\n".join(f'- {g["id"]}: {time.strftime("%a %d %b %Y %H:%M", time.localtime(g["start"]))} to {time.strftime("%H:%M", time.localtime(g["end"]))}, '
                f'folders: {", ".join(f for f, _ in g.get("folders", [])) or "-"}' for g, _ in overview) + """

Work out how many different events this footage shows (different venue, decor, banners, dress code, activity or audience).
Several sessions can be the same event. Give each event:
- id: e1, e2, ...
- label: a short name a person would recognise ("Beach clean-up, branded green tees", "CST week quiz in the office hall")
- clues: what tells this event apart from the others (backdrop, banners, clothing, venue, activity). Be concrete; this is used to check every file later.
- matches: "yes" if this is the event of the project, "no" if it clearly is not, "unsure" otherwise
- why: one short sentence
- groups: the session ids that belong to it
Folder names are useful clues but can be wrong."""
    d = _images_request([s for _, s in overview], text, OVERVIEW_SCHEMA)
    evs = []
    for i, e in enumerate(d.get("events") or []):
        evs.append({"id": str(e.get("id") or f"e{i + 1}"), "label": str(e.get("label", ""))[:80], "clues": str(e.get("clues", ""))[:300],
                    "matches": e.get("matches") if e.get("matches") in ("yes", "no", "unsure") else "unsure",
                    "why": str(e.get("why", ""))[:200], "groups": [str(x) for x in e.get("groups") or []]})
    if not evs:
        raise AIError("Google AI couldn't tell the events apart.")
    return evs


FILES_SCHEMA = {"type": "OBJECT", "properties": {
    "files": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "n": {"type": "INTEGER"}, "event": {"type": "STRING"}}, "required": ["n", "event"]}}},
    "required": ["files"]}


def sort_files(sheets, nums, events, event, hint):
    """Assigns every numbered thumbnail to one of the named events. Returns {number: event_id or "unsure"}."""
    text = f"""EVENTCUT_SORT_FILES
{_project_line(event, hint)}
The footage contains these events:
""" + "\n".join(f'- {e["id"]}: {e["label"]}. Clues: {e["clues"]}' for e in events) + f"""

Each image is a contact sheet; every thumbnail has a number. For every number ({nums[0]} to {nums[-1]}, {len(nums)} in total),
say which event it shows: one of {", ".join(e["id"] for e in events)}, or "unsure" when the picture has no clue
(dark, blurry, a close-up with no background). Return every number exactly once."""
    d = _images_request(sheets, text, FILES_SCHEMA)
    ok = {e["id"] for e in events}
    out = {}
    for f in d.get("files") or []:
        try:
            n = int(f.get("n"))
        except Exception:
            continue
        ev = str(f.get("event", "unsure"))
        out[n] = ev if ev in ok else "unsure"
    return out


# ------------------------------------------------------------------ picking the best moments of a big event
PICK_VERSION = "m3"
PICK_SCHEMA = {"type": "OBJECT", "properties": {
    "moments": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "start": {"type": "STRING"}, "end": {"type": "STRING"}, "what": {"type": "STRING"},
        "kind": {"type": "STRING"}, "score": {"type": "NUMBER"}, "hook": {"type": "NUMBER"}},
        "required": ["start", "end", "what", "score"]}},
    "avoid": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "start": {"type": "STRING"}, "end": {"type": "STRING"}, "why": {"type": "STRING"}}, "required": ["start", "end"]}},
    "statements": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "start": {"type": "STRING"}, "end": {"type": "STRING"}, "quote": {"type": "STRING"},
        "speaker": {"type": "STRING"}, "score": {"type": "NUMBER"}}, "required": ["start", "end", "quote", "score"]}},
    "music": {"type": "OBJECT", "properties": {
        "search": {"type": "STRING"}, "alternatives": {"type": "ARRAY", "items": {"type": "STRING"}},
        "bpm_min": {"type": "NUMBER"}, "bpm_max": {"type": "NUMBER"}, "vocals": {"type": "BOOLEAN"}}},
    "sfx": {"type": "ARRAY", "items": {"type": "STRING"}},
    "summary": {"type": "STRING"}},
    "required": ["moments"]}


def why_not(e) -> str:
    """Plain reason Google AI wasn't used, for the notice on the video (never Google's raw technical text)."""
    m = str(e)
    if "Add your Google AI key" in m or "No Google AI key" in m:
        return "there is no Google AI key in Settings"
    if "switched off" in m:
        return "Google AI is switched off in Settings"
    if "spending cap" in m:
        return "this month's Google AI spending cap has been reached (raise it in Settings)"
    if "rejected the API key" in m:
        return "Google rejected the API key (check it in Settings)"
    if "usage limit" in m or "(429)" in m:
        return "Google's usage limit was reached (wait a few minutes, or turn on billing for the key)"
    if "Couldn't reach Google AI" in m:
        return "the internet connection to Google AI was lost"
    if "prepaid credit" in m:
        return "your Google AI prepaid credit has run out (top it up in Google AI Studio → Billing)"
    if "incomplete" in m or "delimiter" in m or "Expecting" in m:
        return "Google AI's answer came back incomplete"
    if "time allowed" in m:
        return "Google AI was too slow to answer in time (usually a slow upload), so EventCut didn't wait any longer"
    if "No Google AI model could be used" in m:
        return "none of Google's models would accept the request from your key"
    if "couldn't read this clip" in m or "prepare the video" in m:
        return "Google couldn't read the preview EventCut sent"
    if "returned nothing" in m:
        return "Google AI gave no answer for this footage"
    if _gone(m):
        return "Google has closed the Google AI model EventCut asked for, and no other model could be used"
    if "(500)" in m or "(503)" in m or "(502)" in m:
        return "Google AI was having problems on its side (try again later)"
    if "(400)" in m:
        return "Google AI refused the request"
    clean = re.sub(r"\{.*\}", "", m).strip()
    first = re.split(r"(?<=[.!?])\s", clean)[0]
    return first[:160] or "Google AI returned an error"


def _pick_prompt(event_type, direction, duration, part, parts, minutes):
    return f"""You are a senior editor cutting a {int(duration)}-second social video of a {event_type or 'company event'}.
This video is part {part} of {parts} of the raw footage ({minutes} minutes), made of many separate camera clips joined end to end
(each jump in picture is a new clip). It is a small preview: 1 picture a second, low quality. Judge content, not image quality.

Editor's direction (follow it closely): {direction or '(none: choose what makes the best story for this event)'}

1. MOMENTS. Find what people would want to see: the main action and emotion of this event (for example goals and
near-misses, celebrations, trophies and medals being lifted, cheering crowds, gifts or cheques being handed over,
handshakes, hugs, laughter, group photos). Skip setup, idle waiting, people standing or posing with nothing happening,
backs of heads, medical checks and other routine admin, blurry or empty shots and repeats.
Return up to 25 moments, best first. For each:
- start, end: seconds from the start of THIS preview (decimals, e.g. "754.5"); 2 to 10 seconds long, inside one clip.
- what: what happens, in under 12 words. kind: one word (goal, celebration, crowd, handshake, gift, speech, trophy, group, other).
- score 1-10: how much it belongs in the video given the direction.
- hook 0-10: how strong it is as one of the first 3 seconds of the video, the shot that stops people scrolling
  (peak action or emotion, instantly understandable; 0 for anything calm or unclear).

2. AVOID. List the stretches (5 seconds or longer) that must NOT be used even if the picture is sharp: nothing
happening (people standing, posing or waiting), cluttered or messy frames (crowded backgrounds, people blocking the
view, backs of heads, equipment in the way), routine admin (registration, medical checks, setting up), off-topic or
unflattering moments. start, end (seconds in THIS preview) and why (a few words). Up to 30.

3. STATEMENTS. Listen for spoken lines worth hearing in the video: complete sentences that tell the story or show
feeling (why the company did this, thanks from someone who received help, the coach or captain on the win, a short
strong line from a speech). Each must be a whole sentence or two, 3 to 9 seconds, clearly audible, not cut mid-word,
in English or Nigerian Pidgin. Skip greetings, filler, unclear speech and anything off-topic. Up to 8, best first:
start, end (seconds in THIS preview, starting just before the first word and ending just after the last),
quote (the exact words), speaker (who, e.g. "company representative", "business owner", "coach"), score 1-10.

4. Also suggest the music for the whole video (music.search: a short search for a music library like "uplifting afrobeats
celebration", music.alternatives: 2 other searches, bpm_min and bpm_max, vocals: true only if vocals suit the video),
up to 6 short sound-effect searches that fit (e.g. "fast whoosh transition", "crowd cheer swell", "cinematic impact"),
and a one-sentence summary of this part."""


def build_montage(parts, out: Path):
    """Join skims into one small preview (426x240, 1 picture a second: what Google AI looks at anyway) for Google AI."""
    if out.exists() and out.stat().st_size > 1000:
        return out
    ff, _ = media.tools()
    lst = out.with_suffix(".txt")
    lst.write_text("".join("file '" + str(p).replace("'", "'\\''") + "'\n" for p in parts), "utf-8")
    tmp = out.with_suffix(".part.mp4")
    media.run([ff, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
               "-vf", "fps=1,scale=426:240:force_original_aspect_ratio=decrease,pad=426:240:(ow-iw)/2:(oh-ih)/2,setsar=1",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "33", "-g", "10", "-c:a", "aac", "-b:a", "24k", "-ac", "1", "-ar", "16000",
               "-movflags", "+faststart", str(tmp)])
    os.replace(tmp, out)
    try:
        lst.unlink()
    except OSError:
        pass
    return out


def step_limit(parts):
    """How long the whole Google AI step may take before EventCut carries on without the unfinished parts."""
    return min(25 * 60, 6 * 60 + 90 * parts)


def pick_moments(clips, event_type, direction, duration, progress=None, chunk_s=2400, cancelled=None):
    """clips: [(path, skim_path, seconds)] in the order things happened. Returns
    {"picks": [{path, start, end, what, kind, score, hook}], "statements": [{path, start, end, quote, speaker, score}],
     "music": {...}, "sfx": [...], "usd": float, "model": str, "partial": bool, "partial_reason": str}.
    Raises AIError with a plain reason when Google AI can't be used at all."""
    import concurrent.futures as cf
    c = cfg()
    if not c["key"]:
        raise AIError("No Google AI key in Settings.")
    if not c["use"]:
        raise AIError("Google AI is switched off in Settings.")
    if month_spend() >= c["cap"]:
        raise AIError(f"Google AI has reached this month's spending cap (US${c['cap']:.0f}).")
    say = progress or (lambda f, note="": None)
    # split the event into parts of about 40 minutes
    chunks, cur, cur_s = [], [], 0.0
    for item in clips:
        if cur and cur_s + item[2] > chunk_s:
            chunks.append(cur)
            cur, cur_s = [], 0.0
        cur.append(item)
        cur_s += item[2]
    if cur:
        chunks.append(cur)
    n = len(chunks)
    folder = CACHE / "picks"
    folder.mkdir(parents=True, exist_ok=True)
    results = [None] * n
    state = ["waiting"] * n
    lock = threading.Lock()
    up_lock = threading.Lock()  # one upload at a time: parts don't fight over the connection
    t0 = time.time()
    RUN.update(deadline=t0 + step_limit(n), cancelled=cancelled, note=lambda note: say(_frac(), note))

    def _frac():
        done = sum(1 for x in state if x in ("done", "failed"))
        return min(0.99, (done + 0.5 * sum(1 for x in state if x == "watching")) / n)

    def show(k, text):
        say(_frac(), text if n == 1 else f"Part {k + 1} of {n}: {text}")

    def tag_of(k):
        ch = chunks[k]
        return hashlib.sha1(json.dumps([[Path(sk).name for _, sk, _ in ch], event_type, direction, int(duration), c["tier"],
                                        PICK_VERSION, k, n]).encode()).hexdigest()[:16]

    def one(k):
        ch = chunks[k]
        cached = folder / f"{tag_of(k)}.json"
        if cached.exists():
            results[k] = json.loads(cached.read_text("utf-8"))
            note_reused()
            state[k] = "done"
            say(_frac(), f"Google AI already watched part {k + 1} of {n} (saved)")
            return
        offs, t = [], 0.0
        for path, sk, _ in ch:
            d = float(media.probe(str(sk)).get("duration") or 0)
            offs.append((t, t + d, path))
            t += d
        state[k] = "preparing"
        show(k, "making a small preview")
        mont = build_montage([sk for _, sk, _ in ch], folder / f"{tag_of(k)}.mp4")
        size = mont.stat().st_size
        with up_lock:
            state[k] = "uploading"
            t_up = time.time()

            def up_prog(x):
                sent = x * size / 1e6
                speed = sent / max(0.5, time.time() - t_up)
                fmt = (lambda v: f"{v:.1f}") if size < 10e6 else (lambda v: f"{v:.0f}")
                show(k, f"sending the preview, {fmt(sent)} of {fmt(size / 1e6)} MB ({speed:.1f} MB/s)")
            f = upload(c["key"], mont, up_prog, on_wait=lambda: show(k, "Google is preparing the preview"))
        state[k] = "watching"
        t_w = time.time()
        stop = threading.Event()

        def ticker():
            while not stop.wait(5):
                show(k, f"Google AI is watching ({int(time.time() - t_w)} s)")
        threading.Thread(target=ticker, daemon=True).start()
        spent = 0.0
        try:
            data = None
            for attempt in range(2):  # an answer that came back cut off is asked for once more
                resp, model, tier = generate_with(c, f["uri"], _pick_prompt(event_type, direction, duration, k + 1, n, int(t // 60)), PICK_SCHEMA)
                spent += log_spend(resp.get("usageMetadata", {}), tier, model)
                try:
                    got, complete = _json_from(resp, with_status=True)
                except AIError:
                    got, complete = None, False
                if got is not None and (data is None or len(got.get("moments") or []) > len(data.get("moments") or [])):
                    data = got
                if complete:
                    break
                show(k, "Google AI's answer was cut off, asking again")
            if data is None:
                raise AIError(f"Google AI's answer for part {k + 1} of {n} was incomplete.")
        finally:
            stop.set()
            try:
                _req("DELETE", f"{API}/v1beta/{f['name']}", c["key"], timeout=20)
            except Exception:
                pass

        def where(s_, e_):
            mid = (s_ + e_) / 2
            for a0, a1, path in offs:
                if a0 <= mid < a1:
                    return path, round(max(0.0, s_ - a0), 2), round(min(a1 - a0, e_ - a0), 2)
            return None
        picks, says, avoid = [], [], []
        for m in data.get("avoid") or []:
            s_, e_ = _parse_t(m.get("start")), _parse_t(m.get("end"))
            for a0, a1, path in offs:  # an avoided stretch may run across clips
                if s_ < a1 and a0 < e_:
                    avoid.append({"path": path, "start": round(max(0.0, s_ - a0), 2), "end": round(min(a1, e_) - a0, 2),
                                  "why": str(m.get("why", ""))[:80]})
        for m in data.get("moments") or []:
            s_, e_ = _parse_t(m.get("start")), _parse_t(m.get("end"))
            if e_ <= s_:
                e_ = s_ + 4
            w = where(s_, e_)
            if w:
                picks.append({"path": w[0], "start": w[1], "end": w[2], "what": str(m.get("what", ""))[:120],
                              "kind": str(m.get("kind", "other"))[:20], "score": float(m.get("score") or 5),
                              "hook": float(m.get("hook") or 0)})
        for m in data.get("statements") or []:
            s_, e_ = _parse_t(m.get("start")), _parse_t(m.get("end"))
            if e_ - s_ < 1.5:
                continue
            w = where(s_, e_)
            if w and w[2] - w[1] >= 1.5:
                says.append({"path": w[0], "start": w[1], "end": w[2], "quote": str(m.get("quote", ""))[:300],
                             "speaker": str(m.get("speaker", ""))[:60], "score": float(m.get("score") or 5)})
        results[k] = {"picks": picks, "statements": says, "avoid": avoid, "music": data.get("music") or {},
                      "sfx": [str(x)[:60] for x in (data.get("sfx") or [])][:8],
                      "summary": str(data.get("summary", ""))[:300], "usd": round(spent, 4), "model": model}
        cached.write_text(json.dumps(results[k]), "utf-8")
        results[k] = dict(results[k], fresh=True)
        state[k] = "done"
        say(_frac(), f"Google AI watched part {k + 1} of {n}")
        try:
            mont.unlink()
        except OSError:
            pass

    errors, failed = [], []
    ex = cf.ThreadPoolExecutor(3)
    futs = {ex.submit(one, k): k for k in range(n)}
    pending = set(futs)
    try:
        while pending:
            done_now, pending = cf.wait(pending, timeout=1.0)
            for fu in done_now:
                k = futs[fu]
                try:
                    fu.result()
                except InterruptedError:
                    raise
                except AIError as e:
                    state[k] = "failed"
                    errors.append(e)
                    failed.append(k + 1)
                except Exception as e:
                    state[k] = "failed"
                    errors.append(AIError(str(e).splitlines()[0][:200] if str(e) else type(e).__name__))
                    failed.append(k + 1)
            if cancelled and cancelled():
                raise InterruptedError
            if time.time() > RUN["deadline"] + 20:  # a stuck answer: carry on without it
                errors.append(AIError("Google AI took longer than the time allowed for this step."))
                break
    finally:
        RUN.update(deadline=None, cancelled=None, note=None)
        ex.shutdown(wait=False, cancel_futures=True)
    ok = [r for r in results if r]
    if not ok:
        raise errors[0] if errors else AIError("Google AI returned nothing.")
    picks = [p for r in ok for p in r["picks"]]
    statements = [p for r in ok for p in r.get("statements", [])]
    avoid = [p for r in ok for p in r.get("avoid", [])]
    music = next((r["music"] for r in ok if r.get("music") and r["music"].get("search")), {})
    sfx = []
    for r in ok:
        for x in r.get("sfx", []):
            if x.lower() not in [y.lower() for y in sfx]:
                sfx.append(x)
    return {"picks": picks, "statements": statements, "avoid": avoid, "music": music, "sfx": sfx[:10],
            "summaries": [r.get("summary", "") for r in ok], "model": next((r.get("model") for r in ok if r.get("model")), ""),
            "usd": round(sum(r.get("usd", 0) for r in ok if r.get("fresh")), 4), "partial": len(ok) < n,
            "parts": n, "parts_done": len(ok), "seconds": round(time.time() - t0, 1),
            "partial_reason": _partial_reason(errors, failed, n) if errors and len(ok) < n else ""}


def _partial_reason(errors, failed, n):
    """e.g. "Google AI's answer for part 2 of 6 was incomplete" (no technical text)."""
    which = (f"part {failed[0]} of {n}" if len(failed) == 1 else
             f"parts {', '.join(str(x) for x in sorted(failed)[:-1])} and {sorted(failed)[-1]} of {n}") if failed else "some parts"
    m = str(errors[0])
    if "incomplete" in m:
        return f"Google AI's answer for {which} was incomplete"
    return f"Google AI couldn't finish {which}: {why_not(errors[0])}"


def generate_with(c, file_uri, prompt, schema):
    body = {"contents": [{"role": "user", "parts": [{"file_data": {"mime_type": "video/mp4", "file_uri": file_uri}}, {"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "response_mime_type": "application/json", "media_resolution": "MEDIA_RESOLUTION_LOW",
                                 "response_schema": schema}}
    return call_model(c, body, timeout=900)