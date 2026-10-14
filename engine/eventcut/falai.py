"""Fast speech-to-text on fal (Whisper on cloud graphics cards).

Only the sound is sent (small MP3, never the video). Word timings come back in the same shape as the PC's own
Whisper, so captions, podcasts and the caption editor work the same. If fal can't be reached or the key is wrong,
EventCut falls back to listening on this PC.
"""
import base64
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request

from . import media
from .config import HOME, WORK, load_settings

API = os.environ.get("EVENTCUT_FAL_API", "https://queue.fal.run")  # override only for testing
MODEL = "fal-ai/whisper"
LOG = WORK / "_ai" / "fal_usage.json"
MAX_BYTES = 40 * 1024 * 1024  # longer recordings than ~2.5 hours per clip are heard on this PC instead
_lock = threading.Lock()


class FalError(Exception):
    pass


def cfg():
    s = load_settings()
    return {"key": (s.get("fal_api_key") or "").strip(), "use": bool(s.get("use_fal", True))}


_broken = {"until": 0, "why": ""}  # a bad key or empty balance: stop asking fal for a while, listen on the PC


def enabled() -> bool:
    c = cfg()
    return bool(c["key"]) and c["use"] and time.time() > _broken["until"]


def mark_broken(why):
    _broken.update(until=time.time() + 600, why=why)


def _req(method, url, key, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Authorization": f"Key {key}", "Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace")
        try:
            j = json.loads(msg)
            msg = j.get("detail") or j.get("error") or msg
            if isinstance(msg, list):
                msg = "; ".join(str(x.get("msg", x)) if isinstance(x, dict) else str(x) for x in msg)
        except Exception:
            pass
        if e.code in (401, 403):
            raise FalError("fal rejected the API key. Check it in Settings.")
        if e.code == 402 or "balance" in str(msg).lower() or "credit" in str(msg).lower():
            raise FalError("The fal account is out of credit. Top it up at fal.ai, or EventCut will listen on this PC.")
        raise FalError(f"fal error ({e.code}): {str(msg)[:200]}")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        raise FalError(f"Couldn't reach fal. Check the internet connection. ({getattr(e, 'reason', e)})")


def _req_retry(method, url, key, body=None, timeout=120, max_wait=180, note=None):
    """Survives short internet drops; gives up after max_wait so the PC can listen instead."""
    waited, delay = 0.0, 3.0
    while True:
        try:
            return _req(method, url, key, body, timeout)
        except FalError as e:
            if not str(e).startswith("Couldn't reach fal") or waited >= max_wait:
                raise
            if note:
                note(f"Waiting for the internet to come back ({int(waited)} s)")
            time.sleep(delay)
            waited += delay
            delay = min(30.0, delay * 1.6)


def _audio_mp3(path):
    """Mono 16 kHz AAC (M4A) of the clip's sound: about 0.25 MB per minute. (AAC is built into every FFmpeg.)"""
    ff, _ = media.tools()
    fd, tmp = tempfile.mkstemp(suffix=".m4a")
    os.close(fd)
    try:
        media.run([ff, "-y", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "32k", tmp])
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def log_usage(seconds):
    with _lock:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        try:
            d = json.loads(LOG.read_text())
        except Exception:
            d = {}
        m = time.strftime("%Y-%m")
        d[m] = round(float(d.get(m, 0)) + seconds / 60, 2)
        LOG.write_text(json.dumps(d))


def month_minutes():
    try:
        return float(json.loads(LOG.read_text()).get(time.strftime("%Y-%m"), 0))
    except Exception:
        return 0.0


def transcribe(path, prompt="", progress=None, cancelled=None):
    """Word list [{word, start, end, p}] for one clip."""
    c = cfg()
    if not c["key"]:
        raise FalError("No fal key in Settings.")
    note = (lambda n: progress(0.0, n)) if progress else None
    if progress:
        progress(0.02, "Taking out the sound")
    audio = _audio_mp3(path)
    if len(audio) > MAX_BYTES:
        raise FalError("This recording is too long to send in one piece.")
    if len(audio) < 2000:
        return []
    secs = len(audio) / 4000.0  # 32 kbit/s
    body = {"audio_url": "data:audio/mp4;base64," + base64.b64encode(audio).decode(), "task": "transcribe",
            "language": "en", "chunk_level": "word", "batch_size": 64}
    if prompt:
        body["prompt"] = prompt
    if progress:
        progress(0.1, "Sending the sound to fal")
    sub = _req_retry("POST", f"{API}/{MODEL}", c["key"], body, timeout=300, note=note)
    status_url = sub.get("status_url") or f"{API}/{MODEL}/requests/{sub['request_id']}/status"
    result_url = sub.get("response_url") or f"{API}/{MODEL}/requests/{sub['request_id']}"
    t0 = time.time()
    while True:
        if cancelled and cancelled():
            try:
                _req("PUT", sub.get("cancel_url") or result_url + "/cancel", c["key"], timeout=15)
            except Exception:
                pass
            raise InterruptedError
        st = _req_retry("GET", status_url, c["key"], timeout=30, note=note)
        s = st.get("status")
        if s == "COMPLETED":
            if st.get("error"):
                raise FalError(f"fal couldn't transcribe this clip: {str(st.get('error'))[:200]}")
            break
        if progress:
            if s == "IN_QUEUE":
                progress(0.15, f"Waiting at fal ({st.get('queue_position', 0)} ahead)")
            else:
                progress(min(0.9, 0.2 + (time.time() - t0) / max(20.0, secs / 20)), "fal is listening")
        if time.time() - t0 > 1800:
            raise FalError("fal took too long.")
        time.sleep(1.5)
    out = _req_retry("GET", result_url, c["key"], timeout=120, note=note)
    words = []
    for ch in out.get("chunks") or []:
        ts = ch.get("timestamp") or [None, None]
        txt = str(ch.get("text") or "").strip()
        if not txt or ts[0] is None:
            continue
        st_, en = float(ts[0]), float(ts[1] if ts[1] is not None else ts[0] + 0.3)
        if en <= st_:
            en = st_ + 0.15
        words.append({"word": " " + txt, "start": round(st_, 2), "end": round(en, 2), "p": 0.9})
    # Whisper sometimes "hears" a phrase over and over in music or silence: drop long runs of the same word
    clean = []
    for w in words:
        if len(clean) >= 3 and all(x["word"].lower() == w["word"].lower() for x in clean[-3:]):
            continue
        clean.append(w)
    words = clean
    log_usage(secs)
    if progress:
        progress(1.0, "Heard")
    return words


def test_key(key):
    """Cheap check: ask for a request that doesn't exist. 401/403 = bad key, 404 etc = key accepted."""
    try:
        _req("GET", f"{API}/{MODEL}/requests/00000000-0000-0000-0000-000000000000/status", key.strip(), timeout=20)
    except FalError as e:
        m = str(e)
        if "rejected" in m or "reach" in m:
            raise
    return True
