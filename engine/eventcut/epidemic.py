"""Epidemic Sound: professionally produced, licensed music for company channels.

Uses the company's Epidemic Sound API key (Partner Content API, "epidemic_live_..."). Searching and previews are
free; every new track downloaded counts against the plan's downloads, so downloaded tracks are kept and reused.
The track's most popular section (Epidemic's "highlights") is used for short videos, so a 30 or 60 second edit
gets the hook of the song, not the slow intro.
"""
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import media
from .config import EPIDEMIC_MUSIC, EPIDEMIC_SFX, HOME, WORK, load_settings

API = os.environ.get("EVENTCUT_EPIDEMIC_API", "https://api.epidemicsound.com")  # override only for testing
FOLDER = EPIDEMIC_MUSIC
SFX_FOLDER = EPIDEMIC_SFX
PREVIEWS = WORK / "_previews_audio"

# what to search for when EventCut chooses the music itself
FAMILY_TERMS = {
    "upbeat": ["afrobeats", "afro house", "amapiano", "upbeat celebration", "afro pop"],
    "cinematic": ["cinematic epic", "trailer build", "dramatic orchestral", "inspiring cinematic"],
    "corporate": ["uplifting corporate", "inspiring", "motivational", "positive afro"],
    "chill": ["chill", "lofi", "acoustic chill", "afro chill"],
    "emotional": ["emotional piano", "hopeful strings", "inspirational", "heartfelt"],
    "uplifting": ["uplifting celebration", "feel good afro", "hopeful uplifting", "inspiring celebration", "joyful"],
}
# genre words in the Direction box win over the family
PROMPT_TERMS = ["amapiano", "afrobeats", "afrobeat", "afro house", "afro", "highlife", "hip hop", "trap", "gospel",
                "house", "funk", "jazz", "rock", "drill", "dancehall", "reggae", "orchestral", "piano", "lofi"]


class EpidemicError(Exception):
    pass


def cfg():
    s = load_settings()
    return {"key": (s.get("epidemic_api_key") or "").strip(), "use": bool(s.get("use_epidemic", True)),
            "auto": bool(s.get("epidemic_auto", True)), "quality": s.get("epidemic_quality", "high")}


def enabled() -> bool:
    c = cfg()
    return bool(c["key"]) and c["use"]


def _user_id():
    # a stable, anonymous id for this PC (Epidemic uses it for per-user reporting)
    return "eventcut-" + hashlib.sha1(str(HOME).encode()).hexdigest()[:16]


def _req(path, params=None, key=None, timeout=30, method="GET", body=None):
    key = key or cfg()["key"]
    if not key:
        raise EpidemicError("Add your Epidemic Sound API key in Settings.")
    q = ""
    if params:
        pairs = []
        for k, v in params.items():
            if v is None or v == "":
                continue
            for item in (v if isinstance(v, (list, tuple)) else [v]):
                pairs.append((k, str(item)))
        q = "?" + urllib.parse.urlencode(pairs)
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json", "x-partner-user-id": _user_id(), "User-Agent": "EventCut"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(f"{API}{path}{q}", data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace")[:300]
        if e.code == 401:
            raise EpidemicError("Epidemic Sound rejected the API key. Check it in Settings.")
        if e.code == 402 or (e.code in (403, 429) and re.search(r"quota|allowance|limit reached|exceeded|download limit|upgrade", msg, re.I)):
            raise EpidemicError(ALLOWANCE)
        if e.code == 403:
            raise EpidemicError("Your Epidemic Sound plan doesn't include this track or action.")
        if e.code == 429:
            raise EpidemicError("Epidemic Sound's limit was reached (too many requests or downloads). Try again later.")
        detail = ""
        try:  # Epidemic's own explanation, in plain words, never the raw JSON
            j = json.loads(msg)
            errs = j.get("errors") or []
            detail = "; ".join(" ".join(x.get("messages") or []) for x in errs if isinstance(x, dict)) or j.get("message") or ""
        except Exception:
            detail = re.sub(r"<[^>]+>", " ", msg).strip()
        if e.code >= 500:
            raise EpidemicError("Epidemic Sound was having problems on its side. Try again later.")
        raise EpidemicError(f"Epidemic Sound refused the request ({e.code}{': ' + detail[:120] if detail else ''}).")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        raise EpidemicError(f"Couldn't reach Epidemic Sound. Check the internet connection. ({getattr(e, 'reason', e)})")


def _track(t):
    imgs = t.get("images") or {}
    return {"id": str(t.get("id")), "title": t.get("title", "Track"),
            "artist": ", ".join(a.get("name", "") for a in (t.get("mainArtists") or []) if isinstance(a, dict)) or "Epidemic Sound",
            "bpm": t.get("bpm"), "length": t.get("length") or 0,
            "moods": [m.get("name") for m in t.get("moods") or [] if isinstance(m, dict)][:3],
            "genres": [g.get("name") for g in t.get("genres") or [] if isinstance(g, dict)][:3],
            "image": imgs.get("S") or imgs.get("default") or "", "vocals": bool(t.get("hasVocals")),
            "downloaded": bool(_local(str(t.get("id"))))}


PAGE = 40  # Epidemic only accepts result offsets of 0, 40, 80, ...


def _page_offset(offset):
    return (max(0, int(offset)) // PAGE) * PAGE


def search(term="", limit=PAGE, offset=0, instrumental=False, bpm_min=None, bpm_max=None, sort=None):
    p = {"term": term or None, "limit": min(60, int(limit)), "offset": _page_offset(offset), "bpmMin": bpm_min, "bpmMax": bpm_max,
         "sort": sort}
    if instrumental:
        p["vocalType"] = ["NONE", "PRESENCE"]
    d = _req("/v0/tracks/search", p)
    tracks = d.get("tracks") or d.get("data") or d.get("items") or []
    pag = d.get("pagination") or {}
    return {"results": [_track(t) for t in tracks if isinstance(t, dict) and t.get("id")],
            "total": pag.get("totalCount") or pag.get("total") or len(tracks)}


def _local(track_id):
    if not FOLDER.exists():
        return None
    for f in FOLDER.glob(f"*[[]{track_id}[]].mp3"):
        if f.stat().st_size > 10000:
            return f
    return None


def highlight_start(track_id, duration):
    """Seconds into the track where its most popular section for this length begins (0 if unknown)."""
    try:
        d = _req(f"/v0/tracks/{track_id}/highlights", {"duration": int(max(5, min(60, round(duration))))}, timeout=15)
        hs = d.get("highlights") or []
        if hs:
            return round(float(hs[0].get("from", 0)) / 1000.0, 2)
    except EpidemicError:
        pass
    return 0.0


def preview(track_id, start=0.0):
    """A 45-second preview made on this PC from Epidemic's stream (doesn't use a download)."""
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    mp3 = media.has_encoder("libmp3lame")
    out = PREVIEWS / f"es_{track_id}.{'mp3' if mp3 else 'm4a'}"
    if out.exists() and out.stat().st_size > 5000:
        return str(out)
    d = _req(f"/v0/tracks/{track_id}/stream")
    url = d.get("url")
    if not url:
        raise EpidemicError("Epidemic Sound didn't return a preview for this track.")
    ff, _ = media.tools()
    tmp = out.with_suffix(".part" + out.suffix)
    try:
        media.run([ff, "-y", "-v", "error", "-ss", f"{max(0.0, start):.2f}", "-i", url, "-t", "45", "-vn", "-c:a", "libmp3lame" if mp3 else "aac", "-b:a", "128k", str(tmp)],
                  timeout=120)
    except Exception:
        raise EpidemicError("The preview couldn't be loaded. Check the internet connection and try again.")
    os.replace(tmp, out)
    return str(out)


def download(track_id, title="", artist=""):
    """The full track as MP3, kept in EventCut/Epidemic Sound/Music (one download per track, ever)."""
    hit = _local(track_id)
    if hit:
        return str(hit)
    d = _req(f"/v0/tracks/{track_id}/download", {"format": "mp3", "quality": cfg()["quality"]})
    url = d.get("url")
    if not url:
        raise EpidemicError("Epidemic Sound didn't return a download for this track.")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "EventCut"}), timeout=180) as r:
        data = r.read()
    if len(data) < 10000:
        raise EpidemicError("The Epidemic Sound download was incomplete. Try again.")
    FOLDER.mkdir(parents=True, exist_ok=True)
    dest = FOLDER / f"{_safe(f'{artist} - {title}')} [{track_id}].mp3"
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    dest.with_suffix(".licence.txt").write_text(
        f"Epidemic Sound: licensed through your company's Epidemic Sound plan. No on-screen credit needed.\n"
        f"Track {track_id}: {title} by {artist}. Downloaded {time.strftime('%Y-%m-%d')}.\n"
        f"For YouTube and other channels, make sure each channel is cleared (safelisted) in your Epidemic Sound account.", "utf-8")
    return str(dest)


def _fetch(url, dest: Path, min_bytes=2000):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "EventCut"}), timeout=180) as r:
        data = r.read()
    if len(data) < min_bytes:
        raise EpidemicError("The Epidemic Sound download was incomplete. Try again.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return dest


def _safe(text):
    return re.sub(r"[^A-Za-z0-9 ._()&-]+", "", str(text)).strip(" -")[:90] or "Track"


def create_version(track_id, seconds, title="", artist="", progress=None):
    """Epidemic's own edit of a track to an exact length (Create versions, beta): it ends properly instead of fading.
    Returns the file path. One version per track and length is kept and reused."""
    secs = int(round(max(1, min(300, seconds))))
    dest = FOLDER / f"{_safe(f'{artist} - {title}')} [{track_id}] {secs}s.mp3"
    if dest.exists() and dest.stat().st_size > 10000:
        return str(dest)
    job = _req("/v0/tracks/versions", method="POST", body={"trackId": track_id, "durationMs": secs * 1000}, timeout=60)
    jid = job.get("jobId") or job.get("id")
    if not jid:
        raise EpidemicError("Epidemic Sound didn't start the exact-length edit.")
    t0 = time.time()
    while True:
        st = _req(f"/v0/tracks/versions/{jid}", timeout=30)
        status = str(st.get("status", "")).upper()
        if status == "COMPLETED":
            res = st.get("results") or st.get("result") or {}
            if isinstance(res, list):
                res = res[0] if res else {}
            url = res.get("highQualityUrl") or res.get("previewUrl")
            if not url:
                raise EpidemicError("Epidemic Sound finished the edit but sent no file.")
            _fetch(url, dest, 10000)
            dest.with_suffix(".licence.txt").write_text(
                f"Epidemic Sound: {secs}-second version of track {track_id} ({title} by {artist}), licensed through your company's "
                f"Epidemic Sound plan. Made {time.strftime('%Y-%m-%d')}.", "utf-8")
            return str(dest)
        if status == "FAILED":
            raise EpidemicError(f"Epidemic Sound couldn't make the exact-length edit ({st.get('message') or 'failed'}).")
        if time.time() - t0 > 180:
            raise EpidemicError("Epidemic Sound took too long to make the exact-length edit.")
        if progress:
            progress(min(0.9, (time.time() - t0) / 60))
        time.sleep(3)


def similar(track_id, limit=12):
    d = _req(f"/v0/tracks/{track_id}/similar", {"limit": min(60, int(limit))})
    tracks = d.get("tracks") or d.get("data") or d.get("items") or []
    return [_track(t) for t in tracks if isinstance(t, dict) and t.get("id")]


# ------------------------------------------------------------------ sound effects
def _sfx_item(x):
    ln = x.get("length") or x.get("duration") or 0
    try:
        ln = float(ln)
    except (TypeError, ValueError):
        ln = 0.0
    if ln > 600:  # some responses give milliseconds
        ln = ln / 1000.0
    return {"id": str(x.get("id")), "title": str(x.get("title") or "Sound effect"), "length": round(ln, 2),
            "downloaded": bool(_local_sfx(str(x.get("id"))))}


def sfx_search(term, limit=PAGE, offset=0):
    d = _req("/v0/sound-effects/search", {"term": term[:500], "limit": min(60, int(limit)), "offset": _page_offset(offset),
                                           "sort": "best-match", "order": "desc"})
    items = d.get("soundEffects") or d.get("sound_effects") or d.get("tracks") or d.get("items") or d.get("data") or []
    return [_sfx_item(x) for x in items if isinstance(x, dict) and x.get("id")]


def _local_sfx(sid):
    if not SFX_FOLDER.exists():
        return None
    for f in SFX_FOLDER.glob(f"*[[]{sid}[]].mp3"):
        if f.stat().st_size > 500:
            return f
    return None


ALLOWANCE = ("Your Epidemic Sound downloads allowance is used up (the free developer plan allows 50 downloads in total, "
             "and they can't be published). Ask Epidemic Sound for a paid API plan to download more")


def downloaded_sfx():
    """Every Epidemic sound effect already on this PC: [{path, title, id}] (reusing them costs no downloads)."""
    out = []
    if not SFX_FOLDER.exists():
        return out
    for f in SFX_FOLDER.glob("*.mp3"):
        m = re.match(r"(.*) \[([^\]]+)\]$", f.stem)
        if m and f.stat().st_size > 500:
            out.append({"path": str(f), "title": m.group(1), "id": m.group(2), "source": "Epidemic Sound"})
    return out


def sfx_download(sid, title=""):
    hit = _local_sfx(sid)
    if hit:
        return str(hit)
    d = _req(f"/v0/sound-effects/{sid}/download", {"format": "mp3", "quality": "high"})
    url = d.get("url")
    if not url:
        raise EpidemicError("Epidemic Sound didn't return this sound effect.")
    dest = SFX_FOLDER / f"{_safe(title)} [{sid}].mp3"
    _fetch(url, dest, 500)
    return str(dest)


def auto_pick(family, duration, rng, recent, prompt="", brief=None):
    """Choose a fresh Epidemic track for this video: Google AI's music suggestion first (from the footage and the
    Direction box), else the event's mood. Cut to the exact length by Epidemic when the plan allows, otherwise the
    track starts at its most popular section. Returns the music dict EventCut uses, or raises."""
    brief = brief or {}
    p = (prompt or "").lower()
    terms = [t for t in PROMPT_TERMS if re.search(rf"\b{re.escape(t)}\b", p)]
    fam = list(FAMILY_TERMS.get(family, FAMILY_TERMS["upbeat"]))
    rng.shuffle(fam)
    pool = [x for x in [brief.get("search")] + list(brief.get("alternatives") or []) if x]
    pool += [" ".join(terms[:2])] if terms else []
    pool += fam
    bmin, bmax = brief.get("bpm_min"), brief.get("bpm_max")
    try:
        bmin, bmax = (int(bmin) if bmin else None), (int(bmax) if bmax else None)
        if bmin and bmax and bmin > bmax:
            bmin, bmax = bmax, bmin
    except (TypeError, ValueError):
        bmin = bmax = None
    instrumental = not bool(brief.get("vocals"))
    tried = []
    for term in pool[:5]:
        tried.append(term)
        res = search(term, limit=PAGE, offset=int(rng.integers(0, 2)) * PAGE, instrumental=instrumental, bpm_min=bmin, bpm_max=bmax)
        if not res["results"]:  # fewer than 40 matches: the first page
            res = search(term, limit=PAGE, instrumental=instrumental, bpm_min=bmin, bpm_max=bmax)
        if not res["results"] and (bmin or bmax):
            res = search(term, limit=40, instrumental=instrumental)
        good = [t for t in res["results"] if (t["length"] or 0) >= min(duration * 0.6, duration + 5) and f"es:{t['id']}" not in recent]
        if not good:
            continue
        good.sort(key=lambda t: rng.random())
        t = good[int(rng.integers(0, min(len(good), 10)))]
        base = {"title": f"{t['title']} by {t['artist']}", "creator": t["artist"], "license": "Epidemic Sound",
                "source": "epidemic", "es_id": t["id"], "search": term, "bpm": t.get("bpm")}
        try:  # Epidemic's exact-length edit: the song starts and ends musically, right on the video's length
            path = create_version(t["id"], duration, t["title"], t["artist"])
            return dict(base, path=path, start=0.0, version=True, fade_in=0.0, fade_out=0.25)
        except EpidemicError as e:
            note = str(e)
        path = download(t["id"], t["title"], t["artist"])
        start = highlight_start(t["id"], duration) if duration <= 75 else 0.0
        if start and (t["length"] or 0) - start < duration:
            start = max(0.0, (t["length"] or 0) - duration - 1)
        return dict(base, path=path, start=start, version=False, fade_in=0.15 if start else 0.0, fade_out=1.2, version_note=note)
    raise EpidemicError("No Epidemic Sound track matched " + ", ".join(f'"{x}"' for x in tried[:3]) + ".")


def test_key(key):
    _req("/v0/tracks/search", {"limit": 1}, key=key.strip(), timeout=20)
    return True
