"""EventCut Engine local API (http://127.0.0.1:7788).

Only the official EventCut site may call it (Origin check + Chrome Private Network Access),
and only for a signed-in user whose access the admin has approved.
"""
import base64
import hashlib
import json
import mimetypes
import os
import platform
import re
import shutil
import subprocess
import threading
import time
import traceback
import urllib.request
import uuid

import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import epidemic, falai, analyze, audio, gemini, grade, media, render, smart, story
from .config import (FINISHED, FONTS, HOME, LOGOS, LOGS, LOOKS, MUSIC_UPLOADS, PORT, VERSION, WORK, ensure_folders,
                     load_settings, save_settings)

JOBS = {}
ALLOWED_ROOTS = set()
ROOTS_FILE = WORK / "allowed_folders.json"  # folders you chose before: their files stay usable after a restart


def _load_roots():
    try:
        ALLOWED_ROOTS.update(json.loads(ROOTS_FILE.read_text("utf-8"))[-200:])
    except Exception:
        pass


def allow(p):
    r = p if os.path.isdir(p) else os.path.dirname(p)
    if r and r not in ALLOWED_ROOTS:
        ALLOWED_ROOTS.add(r)
        try:
            ROOTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            ROOTS_FILE.write_text(json.dumps(sorted(ALLOWED_ROOTS)[-200:]), "utf-8")
        except OSError:
            pass


_load_roots()
_auth_cache = {}  # token hash -> (expires, user)
SPEED_FILE = WORK / "speed.json"


# ------------------------------------------------------------------ access control
def _jwt_payload(token):
    try:
        part = token.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
    except Exception:
        return {}


def verify_access(token):
    """Returns the user dict if approved, else raises PermissionError with a plain message."""
    cfg = load_settings()
    if not cfg.get("require_approval", True):
        return {"id": "local", "email": "local", "role": "admin"}
    if not token:
        raise PermissionError("Please sign in on the EventCut site.")
    h = hashlib.sha256(token.encode()).hexdigest()
    hit = _auth_cache.get(h)
    if hit and hit[0] > time.time():
        return hit[1]
    url, anon = cfg.get("supabase_url"), cfg.get("supabase_anon_key")
    if not url or not anon:
        raise PermissionError("Engine isn't linked to the EventCut site yet. Open the site and sign in once.")
    claims = _jwt_payload(token)
    if claims.get("exp", 0) < time.time():
        raise PermissionError("Your session expired. Sign in again.")
    req = urllib.request.Request(f"{url}/rest/v1/profiles?select=id,email,full_name,status,role&id=eq.{claims.get('sub')}",
                                 headers={"apikey": anon, "Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            rows = json.loads(r.read())
    except Exception:
        # offline: honour a recent approval for up to 3 days
        offline = WORK / "_access.json"
        if offline.exists():
            d = json.loads(offline.read_text())
            if d.get("sub") == claims.get("sub") and d.get("until", 0) > time.time():
                return d["user"]
        raise PermissionError("Can't reach the EventCut site to check your access. Check your internet connection.")
    if not rows or rows[0].get("status") != "approved":
        status = rows[0].get("status") if rows else "unknown"
        raise PermissionError({"pending": "Your access request is waiting for approval.",
                               "declined": "Your access request was declined.",
                               "revoked": "Your access has been removed by the admin."}.get(status, "You don't have access yet."))
    user = rows[0]
    _auth_cache[h] = (time.time() + 600, user)  # re-check every 10 minutes so revoking takes effect quickly
    (WORK / "_access.json").write_text(json.dumps({"sub": claims.get("sub"), "user": user, "until": time.time() + 3 * 86400}))
    return user


def safe_name(name: str, fallback: str) -> str:
    """Keep a readable file name (letters in any language) without path tricks."""
    name = os.path.basename(str(name).replace("\\", "/"))
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "", name).strip(" .")
    return name[:120] or fallback


def path_allowed(p: str) -> bool:
    rp = os.path.realpath(p)
    roots = {os.path.realpath(str(HOME))} | {os.path.realpath(r) for r in ALLOWED_ROOTS}
    return any(rp == r or rp.startswith(r + os.sep) for r in roots)


# ------------------------------------------------------------------ speed model for time-left estimates
def load_speed():
    d = {"analyse": 0.35, "transcribe": 0.12, "transcribe_fal": 0.02, "render": 6.0, "finish": 1.2}  # seconds per second of footage/output
    if SPEED_FILE.exists():
        try:
            d.update(json.loads(SPEED_FILE.read_text()))
        except Exception:
            pass
    return d


def update_speed(stage, rate):
    d = load_speed()
    d[stage] = round(d.get(stage, rate) * 0.6 + rate * 0.4, 4)  # learns this PC's speed over time
    SPEED_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPEED_FILE.write_text(json.dumps(d))


def log(msg):
    """One line in Logs/engine.log (the engine's output goes there)."""
    try:
        print(time.strftime("%H:%M:%S"), msg, flush=True)
    except Exception:
        pass


def _base(key):
    return re.sub(r"_(\d+x\d+|\d+:\d+|\d+)$", "", key or "")


# ------------------------------------------------------------------ jobs
class Job:
    def __init__(self, kind, payload, user):
        self.id = uuid.uuid4().hex[:10]
        self.kind, self.payload, self.user = kind, payload, user
        self.state, self.stage, self.note, self.error = "queued", "Starting", "", None
        self.progress = 0.0
        self.started = time.time()
        self.eta = None
        self.result = None
        self.cancelled = False
        self.skip_stage = False  # set from the site: "Skip listening"
        st = (payload or {}).get("settings") or {}
        self.title = st.get("title") or st.get("name") or st.get("event_type") or kind
        self.clips = len((payload or {}).get("media") or (payload or {}).get("paths") or [])
        self.events = []
        self.weights = {}
        self.done_w = 0.0
        self.stage_key = None
        self.stage_t0 = time.time()
        self.notices = []
        self.timings = []  # [{"step", "seconds"}] in order: shown on the finished video

    def plan_stages(self, weights):
        """weights: ordered {stage: expected_seconds}. Each is corrected by how long that step really took on this
        PC (and this internet connection) in earlier runs."""
        sp = load_speed()
        self.weights = {k: v * float(sp.get(f"f_{_base(k)}", 1.0)) for k, v in weights.items()}
        self.planned = dict(self.weights)
        self.done_w = 0.0

    def close_stage(self):
        if not self.stage_key:
            return
        secs = time.time() - self.stage_t0
        self.timings.append({"step": self.stage, "key": self.stage_key, "seconds": round(secs, 1)})
        log(f"[{self.id}] done: {self.stage} in {secs:.1f} s")
        planned = getattr(self, "planned", {}).get(self.stage_key, 0)
        raw = planned / max(1e-6, float(load_speed().get(f"f_{_base(self.stage_key)}", 1.0)))
        if raw >= 5 and secs > 1 and not self.cancelled:
            f = min(5.0, max(0.2, secs / raw))
            d = load_speed()
            k = f"f_{_base(self.stage_key)}"
            d[k] = round(float(d.get(k, f)) * 0.5 + f * 0.5, 3)
            try:
                SPEED_FILE.parent.mkdir(parents=True, exist_ok=True)
                SPEED_FILE.write_text(json.dumps(d))
            except OSError:
                pass

    def set_stage(self, key, label):
        self.close_stage()
        if self.stage_key and self.stage_key in self.weights:
            self.done_w += self.weights[self.stage_key]
        self.stage_key, self.stage, self.note = key, label, ""
        self.stage_t0 = time.time()
        log(f"[{self.id}] step: {label}")
        self.update(0.0)

    def update(self, frac, note=""):
        total = sum(self.weights.values()) or 1
        cur = self.weights.get(self.stage_key, 0)
        self.progress = min(0.999, (self.done_w + cur * frac) / total)
        if note:
            self.note = note
        # live estimate for the current stage, prior for the rest
        elapsed_stage = time.time() - self.stage_t0
        stage_left = (elapsed_stage / frac - elapsed_stage) if frac > 0.05 else cur
        later = 0.0
        seen = False
        for k, v in self.weights.items():
            if seen:
                later += v
            if k == self.stage_key:
                seen = True
        self.eta = {"stage_left": round(max(0, stage_left)), "total_left": round(max(0, stage_left + later)),
                    "elapsed": round(time.time() - self.started)}
        self.push()

    def push(self):
        self.events.append(self.snapshot())
        self.events = self.events[-50:]

    def snapshot(self):
        return {"id": self.id, "kind": self.kind, "title": self.title, "clips": self.clips, "state": self.state, "stage": self.stage, "stage_key": self.stage_key, "note": self.note,
                "progress": round(self.progress, 4), "eta": self.eta, "error": self.error, "result": self.result,
                "notices": list(self.notices)}


def play_sound(ok=True):
    cfg = load_settings()
    if not cfg.get("notify_sound", True):
        return
    f = WORK / "Sounds" / ("eventcut_done.wav" if ok else "eventcut_failed.wav")
    if platform.system() == "Windows" and f.exists():
        try:
            import winsound
            winsound.PlaySound(str(f), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            pass


def project_dir(name):
    safe = re.sub(r"[^A-Za-z0-9 _-]+", "", name or "Event").strip() or "Event"
    base = FINISHED / safe
    base.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split(" ")[-1]) for p in base.glob("Version *") if p.name.split(" ")[-1].isdigit()] or [0])
    d = base / f"Version {n}"
    d.mkdir()
    return d


def feed_copy(src):
    """Web copy of a finished video for the homepage feed: 720p (1280 px on the long side), at most 30 s, about 6 MB.

    Sharp enough for the large hero player on the homepage, still quick to load."""
    if not src or not path_allowed(src) or not os.path.isfile(src):
        raise ValueError("Video not found")
    ff, _ = media.tools()
    folder = WORK / "_feed"
    folder.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{src}|{os.path.getmtime(src)}|hd2".encode()).hexdigest()[:16]
    vid, img = folder / f"{key}.mp4", folder / f"{key}.jpg"
    info = media.probe(src)
    dur = min(30.0, float(info.get("duration") or 30))
    w, h = info.get("width") or 1080, info.get("height") or 1920
    long_side = 1280 if max(w, h) >= 1280 else max(w, h) // 2 * 2
    scale = (f"scale=-2:{long_side}" if h >= w else f"scale={long_side}:-2") + ":flags=lanczos"
    if not vid.exists():
        media.run([ff, "-y", "-v", "error", "-i", src, "-t", f"{dur:.2f}", "-vf", f"{scale},fps=25,format=yuv420p",
                   "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-maxrate", "1800k", "-bufsize", "3600k",
                   "-profile:v", "high", "-c:a", "aac", "-b:a", "96k", "-ac", "2", "-movflags", "+faststart",
                   "-af", f"afade=t=out:st={max(0, dur - 0.6):.2f}:d=0.6", str(vid)])
    if not img.exists():
        thumb = Path(src[:-4] + "_thumbnail.jpg")
        media.run([ff, "-y", "-v", "error"] + (["-i", str(thumb)] if thumb.exists() else ["-ss", f"{min(1.5, dur / 3):.2f}", "-i", src])
                  + ["-frames:v", "1", "-vf", scale, "-q:v", "3", str(img)])
    return {"video": str(vid), "poster": str(img), "seconds": round(dur, 1), "width": w, "height": h,
            "bytes": vid.stat().st_size}


WIDE_CROP = "scale=960:540:force_original_aspect_ratio=increase,crop=960:540:(iw-960)/2:(ih-540)*0.3"  # same frame, any source shape


def grade_frame(p, t, preset, intensity=1.0, log_profile=None, width=960, wide=False):
    """One frame of a source clip with the full colour pipeline (correction + look). Cached as a JPG."""
    ff, _ = media.tools()
    key = analyze.cache_key(p)
    aj = analyze.CACHE / f"{key}.json"
    a = json.loads(aj.read_text("utf-8")) if aj.exists() else {}
    log = a.get("log") or {}
    lut = grade.lut_for(log_profile=log_profile or (log.get("profile") if log.get("is_log") else None),
                        correction=grade.correction_params(a.get("frames", [])) if a.get("frames") else None,
                        preset_name=preset, intensity=float(intensity), transfer=None, theme_hex=None)
    tag = json.dumps([p, round(t, 2), preset, round(float(intensity), 2), log_profile, width, wide])
    out = WORK / "_thumbs" / f"grade_{hashlib.sha1(tag.encode()).hexdigest()[:12]}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        media.run([ff, "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", p, "-vf",
                   f"{WIDE_CROP if wide else f'scale={width}:-2'},lut3d=file='{render.fpath(lut)}'", "-frames:v", "1", "-q:v", "3", str(out)])
    return out


def raw_frame(p, t, width=960, wide=False):
    """The frame exactly as the camera recorded it (for 'straight from camera')."""
    ff, _ = media.tools()
    out = WORK / "_thumbs" / f"raw_{hashlib.sha1(json.dumps([p, round(t, 2), width, wide]).encode()).hexdigest()[:12]}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        media.run([ff, "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", p, "-vf", WIDE_CROP if wide else f"scale={width}:-2",
                   "-frames:v", "1", "-q:v", "3", str(out)])
    return out


def ai_colour_pass(analyses, s, job=None, notices=None):
    """Ask the Google AI colourist about one representative, corrected frame per clip (in parallel)."""
    from concurrent.futures import ThreadPoolExecutor
    ff, _ = media.tools()
    look = s.get("preset", "Vibrant Cinematic")
    vids = [a for a in analyses if a.get("kind") == "video" and a.get("moments")]
    offline = {"yes": False}  # no internet: one clip finds out, the rest skip straight away

    def one(a):
        if offline["yes"] or (job and job.cancelled):
            raise gemini.AIError("skipped")
        if job:
            gemini._wait_note.fn = lambda note: job.update(done / max(1, len(vids)), note)
        good = [m for m in a["moments"] if not m.get("reject")] or a["moments"]
        m = max(good, key=lambda m: m.get("score", 0))
        t = (m["start"] + m["end"]) / 2
        log = a.get("log") or {}
        lut = grade.lut_for(log_profile=log.get("profile") if log.get("is_log") else None,
                            correction=grade.correction_params(a.get("frames", [])), preset_name="Natural", intensity=0.0)
        jpg = WORK / "_thumbs" / f"aicol_{a['key']}_{round(t, 1)}.jpg"
        jpg.parent.mkdir(parents=True, exist_ok=True)
        if not jpg.exists():
            # the small skim copy when there is one: the same colours, without seeking through a 4K file on an external drive
            src = (s.get("_skim") or {}).get(a["path"]) or a["path"]
            media.run([ff, "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", src, "-vf", f"scale=768:-2,lut3d=file='{render.fpath(lut)}'",
                       "-frames:v", "1", "-q:v", "4", str(jpg)])
        return a["path"], gemini.colourist(jpg, look, s.get("event_type", ""), analyze.proxy_key(a["path"]))
    out, done = {}, 0
    with ThreadPoolExecutor(8) as pool:
        for fut in [pool.submit(one, a) for a in vids]:
            try:
                p, params = fut.result()
                out[p] = params
            except gemini.AIError as e:
                # anything but a one-off hiccup stops the colourist for every clip at once (no minutes of failing clip by clip)
                if str(e) != "skipped" and "returned nothing" not in str(e):
                    if not offline["yes"] and notices is not None:
                        notices.append(f"The Google AI colourist wasn't used: {gemini.why_not(e)}. "
                                       "Clips got EventCut's standard colour correction instead.")
                        log(f"colourist stopped: {e}")
                    offline["yes"] = True
                # this clip keeps the standard correction
            except Exception:
                traceback.print_exc()
            done += 1
            if job:
                job.update(done / max(1, len(vids)), f"Clip {done} of {len(vids)}")
    return out


def card_picture(o):
    """A small picture of a finished video for its card (made once, next to the video)."""
    try:
        if o.get("thumbnail") and os.path.isfile(o["thumbnail"]):
            return o["thumbnail"]
        v = o.get("video")
        if not v or not os.path.isfile(v):
            return None
        out = Path(v).with_name("_card.jpg")
        if not out.exists():
            ff, _ = media.tools()
            dur = float(media.probe(v).get("duration") or 4)
            media.run([ff, "-y", "-v", "error", "-ss", f"{min(dur * 0.3, 8):.2f}", "-i", v, "-vf", "scale=480:-2", "-frames:v", "1",
                       "-q:v", "5", str(out)], check=False)
        return str(out) if out.exists() else None
    except Exception:
        return None


def export_name(event, fmt, version_dir):
    """EventCut_Team-Bonding_9x16_v3_2026-10-11: sorts by event, then shape, version and date."""
    words = re.sub(r"[^A-Za-z0-9 -]+", " ", event or "Event").split()
    ev = "-".join(w[:1].upper() + w[1:] for w in words)[:40] or "Event"
    v = str(version_dir.name).split(" ")[-1]
    return f"EventCut_{ev}_{fmt.replace(':', 'x')}_v{v}_{time.strftime('%Y-%m-%d')}"


def analyse_workers():
    """How many clips to watch at once: about a third of the processor cores, 1 to 4 (setting: analyse_workers, 0 = auto)."""
    n = int(load_settings().get("analyse_workers", 0) or 0)
    if n <= 0:
        n = min(4, round((os.cpu_count() or 4) / 3))
    return max(1, min(6, n))


def run_generate(job: Job):
    p = job.payload
    s = dict(p.get("settings") or {})
    if not int(s.get("seed") or 0):  # every new video gets its own randomness: music, effects, picks
        import random
        s["seed"] = random.randint(1, 2_000_000_000)
    try:
        job.state = "running"
        items = p.get("media") or media.scan(p.get("paths", []))[0]
        # files that went missing, were moved, or can't be read are skipped instead of stopping the whole video
        bad = []

        def readable(m):
            if not os.path.isfile(m["path"]):
                bad.append((m["path"], "missing or moved"))
                return False
            if m["kind"] in ("video", "audio"):
                try:
                    media.probe(m["path"])
                except Exception:
                    bad.append((m["path"], "damaged or unreadable"))
                    return False
            return True
        if not items:
            raise ValueError("No footage was chosen for this video. Choose the files (or open the video again from Your videos and use Regenerate).")
        gemini.take_switch_notes()
        log(f"[{job.id}] new video: {len(items)} files, {s.get('duration')} s, {s.get('formats')}, direction: {(s.get('prompt') or '(none)')[:200]}")
        items = [m for m in items if readable(m)]
        vids = [m for m in items if m["kind"] in ("video", "photo")]
        if not vids:
            where = sorted({str(Path(b[0]).parent) for b in bad})[:2]
            raise ValueError("None of the chosen videos or photos could be read" + (f" (looked in {', '.join(where)})" if where else "") +
                             ". Check the drive is plugged in and the files are still in their folder.")
        foot = sum(float(media.probe(m["path"]).get("duration") or 0) for m in vids if m["kind"] == "video") or 10
        cached = all((analyze.CACHE / f"{analyze.cache_key(m['path'])}.json").exists() for m in vids)
        sp = load_speed()
        formats = s.get("formats") or [s.get("format", "9:16")]
        dur = float(s.get("duration", 60))
        talk = story.template_for(s.get("event_type", "")) == "talk"
        use_ai = talk and gemini.enabled() and s.get("google_ai", True)
        # Fast watching (always on): skim every clip from its keyframes, then study only the best stretches closely.
        # Podcasts and interviews (a few long recordings where every word matters) and small projects are studied in full.
        skim = not talk and foot > float(load_settings().get("skim_min_minutes", 15)) * 60
        notices = job.notices
        s["_notices"] = notices
        want_words = (s.get("captions") or (talk and not use_ai) or (s.get("prompt") and not use_ai and not skim)) and smart.speech_available()
        pick_ai = skim and s.get("google_ai", True)
        job.plan_stages({"skim": foot * sp.get("skim", 0.12) if skim else 0,
                         "picks": foot * 0.004 + 60 if pick_ai else 0,
                         "analyse": 1 if cached else (dur * 2.2 * 6 * sp["analyse"] if skim else foot * sp["analyse"]),
                         "statements": 20 if pick_ai else 0,
                         "sync": foot * 0.01 + 5 if any(m["kind"] == "audio" for m in items) else 0,
                         "transcribe": (dur * 40 if skim else foot) * sp["transcribe_fal" if falai.enabled() else "transcribe"] if want_words else 0,
                         "ai": foot * 0.05 + 40 if use_ai else 0,
                         "colour": 3 * min(len(vids), 40) if (bool(gemini.cfg()["key"]) and load_settings().get("google_colour", True) and s.get("preset") != "None") else 0,
                         "director": 25 if s.get("director") else 0, "plan": 2,
                         **{f"render_{f}": dur * sp["render"] for f in formats},
                         **{f"finish_{f}": dur * sp["finish"] for f in formats}})
        workers = analyse_workers()
        plock = threading.Lock()

        def run_parallel(fn, n, nworkers):
            if nworkers > 1 and n > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(nworkers) as ex:
                    for _ in ex.map(fn, range(n)):
                        pass
            else:
                for i in range(n):
                    fn(i)

        def checked(fn, path):
            """Run one clip's work; a damaged file is skipped and named at the end, never stops the video."""
            try:
                return fn()
            except InterruptedError:
                raise
            except Exception as e:
                print("skipped", path, e)
                with plock:
                    if path not in [b[0] for b in bad]:
                        bad.append((path, "damaged or unreadable"))
                return None
        t0 = time.time()
        if not skim:
            # 1. analyse every frame (small projects, podcasts)
            job.set_stage("analyse", "Watching every clip")
            analyses = [None] * len(vids)
            fr = [0.0] * len(vids)

            def one(i):
                m = vids[i]
                if job.cancelled:
                    raise InterruptedError

                def prog(f, i=i):
                    if job.cancelled:
                        raise InterruptedError
                    with plock:
                        fr[i] = f
                        done = sum(1 for x in fr if x >= 1)
                        job.update(sum(fr) / len(vids), f"Clip {min(len(vids), done + 1)} of {len(vids)}" + (f", {workers} at a time" if workers > 1 else ""))
                analyses[i] = checked(lambda: analyze.analyse(m["path"], m["kind"], prog), m["path"])
                prog(1.0)
            run_parallel(one, len(vids), workers)
            analyses = [a for a in analyses if a]
            if analyses and not cached:
                update_speed("analyse", (time.time() - t0) / foot)
        else:
            from . import quick
            # 1. skim every clip (keyframes only, about 2 pictures a second); clips studied fully in an earlier run are reused
            job.set_stage("skim", "Skimming every clip")
            full = {}
            quicks = [None] * len(vids)
            photos = [None] * len(vids)
            fr = [0.0] * len(vids)
            sworkers = max(workers, min(6, (os.cpu_count() or 4) // 2))

            def skim_one(i):
                m = vids[i]
                if job.cancelled:
                    raise InterruptedError

                def prog(f, i=i):
                    if job.cancelled:
                        raise InterruptedError
                    with plock:
                        fr[i] = f
                        done = sum(1 for x in fr if x >= 1)
                        job.update(sum(fr) / len(vids), f"Clip {min(len(vids), done + 1)} of {len(vids)}, {sworkers} at a time")
                if m["kind"] == "photo":
                    photos[i] = checked(lambda: analyze.analyse_photo(m["path"]), m["path"])
                else:
                    prev = quick.full_cached(m["path"])
                    if prev:
                        with plock:
                            full[m["path"]] = prev
                    quicks[i] = checked(lambda: quick.quick_analyse(m["path"], prog), m["path"])
                prog(1.0)
            run_parallel(skim_one, len(vids), sworkers)
            quicks = [q for q in quicks if q]
            photos = [a for a in photos if a]
            if quicks:
                update_speed("skim", (time.time() - t0) / foot)
            order = story.source_order(quicks)
            # 2. Google AI watches a small preview of the whole event and picks moments, following the Direction box
            picks, music_brief = [], {}
            if pick_ai:
                job.set_stage("picks", "Google AI is choosing the best moments")
                if not gemini.enabled() and not gemini.cfg()["key"]:
                    notices.append("Google AI didn't choose the moments: " + gemini.why_not("No Google AI key") + ". EventCut chose them itself.")
                elif not gemini.enabled():
                    notices.append("Google AI didn't choose the moments: " + gemini.why_not("switched off") + ". EventCut chose them itself.")
                else:
                    by_path = {q["path"]: q for q in quicks}
                    clips = [(p, by_path[p]["qproxy"], float(by_path[p]["info"].get("duration") or 0)) for p in order if p in by_path]

                    def gprog(f, note=""):
                        if job.cancelled:
                            raise InterruptedError
                        job.update(f, note)
                    try:
                        r = gemini.pick_moments(clips, s.get("event_type", ""), s.get("prompt", ""), dur, gprog,
                                                cancelled=lambda: job.cancelled)
                        picks, music_brief = r["picks"], r
                        s["_ai_picks"] = picks
                        s["_ai_statements"] = r.get("statements", [])
                        s["_ai_avoid"] = r.get("avoid", [])
                        log(f"Google AI ({r.get('model')}): {len(picks)} moments, {len(r.get('statements', []))} statements, "
                            f"{r.get('parts_done')}/{r.get('parts')} parts in {r.get('seconds')} s")
                        if r.get("partial"):
                            notices.append(f"Google AI watched only part of the footage ({r.get('partial_reason')}). "
                                           "EventCut chose moments for the rest itself.")
                    except InterruptedError:
                        raise
                    except gemini.AIError as e:
                        notices.append(f"Google AI didn't choose the moments: {gemini.why_not(e)}. EventCut chose them itself.")
                    except Exception as e:
                        traceback.print_exc()
                        notices.append(f"Google AI didn't choose the moments: {str(e).splitlines()[0][:160]}. EventCut chose them itself.")
            s["_music_brief"] = music_brief
            # 3. close look at the chosen stretches only
            job.set_stage("analyse", "Looking closely at the best moments")
            t1 = time.time()
            said = s.get("_ai_statements") or []
            wins = quick.plan_windows([q for q in quicks if q["path"] not in full], dur,
                                      picks + [{"path": x["path"], "start": max(0.0, x["start"] - 1.5), "end": x["end"] + 1.5, "score": 8}
                                               for x in said], order)
            todo = [q for q in quicks if q["path"] in wins]
            closes = [None] * len(todo)
            cf_ = [0.0] * len(todo)

            def close_one(i):
                q = todo[i]
                if job.cancelled:
                    raise InterruptedError

                def prog(f, i=i):
                    if job.cancelled:
                        raise InterruptedError
                    with plock:
                        cf_[i] = f
                        job.update(sum(cf_) / max(1, len(todo)), f"{sum(1 for x in cf_ if x >= 1)} of {len(todo)} clips, best moments only")
                closes[i] = checked(lambda: quick.close_analysis(q, wins[q["path"]], prog), q["path"])
                prog(1.0)
            run_parallel(close_one, len(todo), workers)
            analyses = list(full.values()) + [a for a in closes if a] + photos
            secs = sum(e - s0 for w in wins.values() for s0, e in w)
            if secs > 0:
                update_speed("analyse", (time.time() - t1) / secs)
            s["_skim"] = {q["path"]: q["qproxy"] for q in quicks}
            # 3b. the statements Google AI heard, snapped to whole sentences
            if s.get("_ai_statements") and story.template_for(s.get("event_type", "")) != "talk":
                job.set_stage("statements", "Listening closely to the best statements")
                from . import soundbites
                try:
                    s["_soundbites"] = soundbites.refine(s["_ai_statements"], s["_skim"], lambda f, n="": job.update(f, n),
                                                         lambda: job.cancelled)
                    log(f"[{job.id}] statements: " + " | ".join(f"{Path(b['path']).name} {b['start']}-{b['end']}: {b['quote'][:60]}"
                                                                for b in s["_soundbites"]))
                except InterruptedError:
                    raise
                except Exception as e:
                    traceback.print_exc()
                    s["_soundbites"] = []
        if not analyses:
            raise ValueError("None of the videos or photos could be read. Check the files are still in their folder.")
        # 1b. separately recorded sound (recorders, lapel mics, mixing desk) lined up with the camera clips
        recs = [m["path"] for m in items if m["kind"] == "audio"]
        if recs:
            job.set_stage("sync", "Matching recorded audio to the clips")
            from . import sync
            try:
                s["audio_sync"] = sync.match([a["path"] for a in analyses if a["kind"] == "video" and a["info"].get("has_audio")],
                                             recs, lambda f: job.update(f, "Lining up the sound"), sources=s.get("_skim"))
            except Exception as e:  # never stops the edit
                s["audio_sync"] = {}
                print("audio sync failed:", e)
        # 2. transcribe (with fast watching this happens after planning, only for the clips in the video)
        transcripts = {}
        unheard = []

        def listen_to(pool, footage_s):
            if True:
                job.set_stage("transcribe", "Listening to speeches")
                t0 = time.time()
                speak = [a for a in pool if a["kind"] == "video" and a["info"].get("has_audio")]
                lens = [float(a["info"].get("duration") or 1) for a in speak]
                on_fal = falai.enabled()
                heard = [0.0] * len(speak)
                hlock = threading.Lock()

                def listen(i):
                    a = speak[i]

                    def prog(f, note, i=i):
                        if job.cancelled:
                            raise InterruptedError
                        if job.skip_stage:
                            raise smart.Skipped
                        with hlock:
                            heard[i] = f
                            done = sum(1 for x in heard if x >= 1)
                            msg = f"{done} of {len(speak)} clips heard, on fal" if on_fal else f"Clip {i + 1} of {len(speak)}: {note}"
                            job.update(sum(h * l for h, l in zip(heard, lens)) / (sum(lens) or 1), msg)
                    if job.cancelled:
                        raise InterruptedError
                    if job.skip_stage:
                        return
                    try:
                        transcripts[a["path"]] = smart.transcribe(a["path"], analyze.proxy_key(a["path"]), prog,
                                                                  audio_src=(s.get("_skim") or {}).get(a["path"]))
                    except InterruptedError:
                        raise
                    except smart.Skipped:
                        return
                    except Exception as e:  # one clip's sound can't be read: carry on without its words
                        print("listening skipped", a["path"], e)
                        with hlock:
                            unheard.append(a["path"])
                    with hlock:
                        heard[i] = 1.0
                if on_fal:  # several clips at once: fal does the listening, this PC only takes out the sound
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(4) as ex:
                        for _ in ex.map(listen, range(len(speak))):
                            pass
                else:
                    for i in range(len(speak)):
                        if job.skip_stage:
                            break
                        listen(i)
                if not job.skip_stage:
                    update_speed("transcribe_fal" if on_fal else "transcribe", (time.time() - t0) / max(1.0, footage_s))
                job.skip_stage = False
        if want_words and not skim:
            listen_to(analyses, foot)
        # 2b. Google AI picks the soundbites (podcasts, interviews, panels)
        ai, ai_note = {}, None
        if use_ai:
            job.set_stage("ai", "Google AI is watching your footage")
            speak = [a for a in analyses if a["kind"] == "video" and a["info"].get("has_audio")]
            lens = [float(a["info"].get("duration") or 1) for a in speak]
            try:
                for i, a in enumerate(speak):
                    if job.cancelled:
                        raise InterruptedError
                    before = sum(lens[:i])

                    def prog(f, note="", i=i, before=before):
                        if job.cancelled:
                            raise InterruptedError
                        job.update((before + f * lens[i]) / (sum(lens) or 1), f"Clip {i + 1} of {len(speak)}: {note}")
                    try:
                        r = gemini.analyse_clip(a["path"], s.get("event_type", ""), lens[i], analyze.proxy_key(a["path"]), prog)
                        ai[a["path"]] = r
                    except gemini.AIError:
                        raise
                    except InterruptedError:
                        raise
                    except Exception as e:  # this clip couldn't be prepared for Google AI: the others still count
                        print("Google AI skipped a clip", a["path"], e)
                if not s.get("title") and s.get("ai_title") and ai:
                    s["title"] = next(iter(ai.values())).get("title") or ""
            except gemini.AIError as e:
                ai, ai_note = {}, f"Google AI wasn't used ({e}). EventCut chose the moments itself."
        s["_ai"] = ai
        if ai and transcripts:  # the AI's quotes fix Pidgin words the speech-to-text misheard (timings stay)
            from . import talk as _talk
            for pth, r_ in ai.items():
                if transcripts.get(pth):
                    transcripts[pth] = _talk.apply_quotes(transcripts[pth], r_.get("bites"))
        # 3. director (optional): never stops the video; without it EventCut chooses the story itself
        if s.get("director"):
            job.set_stage("director", "Director mode is choosing the story")
            cfg = load_settings()
            try:
                if smart.month_spend() >= float(cfg.get("director_monthly_cap_usd", 20)):
                    raise RuntimeError("it has reached this month's spending cap")
                d = smart.director_plan(analyses, transcripts, s)
                s["director_picks"] = {f"{k[0]}|{k[1]}": v for k, v in d["picks"].items()}
                if d.get("title") and not s.get("title"):
                    s["title"] = d["title"]
            except Exception as e:
                why = str(e).splitlines()[0][:160] if str(e) else "it couldn't be reached"
                ai_note = (ai_note + " " if ai_note else "") + f"Director mode wasn't used ({why}). EventCut chose the story itself."
        # 4. style + plan
        job.set_stage("plan", "Building the story")
        if s.get("style_name"):
            st = smart.load_style(s["style_name"])
            if st:
                s["style"] = {"median_shot": st["median_shot"]}
                if s.get("style_colour", True):
                    s["match_look"] = st["lab"]
                s.setdefault("lufs", st.get("lufs", -14))
        if s.get("look_ref"):
            s["match_look"] = smart.frame_lab(s["look_ref"]["path"], s["look_ref"].get("t"))
        taste = smart.load_taste() if s.get("learn_taste", True) else None
        music = s.get("music") or {}
        seed = int(s.get("seed", 0))
        if not music.get("path") and not music.get("mood"):
            # automatic: new music for every version (library track for the mood if online, else an original)
            family = s.get("music_family") or story.music_family(s)
            job.update(0.3, "Choosing music")
            music = audio.pick_auto_music(family, dur, seed, online=s.get("online_music", True), prompt=s.get("prompt", ""),
                                          brief=(s.get("_music_brief") or {}).get("music"))
            if music.get("epidemic_error"):
                what = "an EventCut original track" if music.get("source") == "generated" else "a track from the free library"
                notices.append(f"Epidemic Sound wasn't used for the music: {music['epidemic_error'].rstrip('.')}. EventCut used {what} instead.")
            elif music.get("source") == "epidemic" and not music.get("version") and music.get("version_note"):
                notices.append("Epidemic Sound's exact-length edit wasn't available (" + music["version_note"].rstrip(".") +
                               "), so the track starts at its most popular part and fades out at the end.")
            s["music"] = music
        if music.get("path") and "start" not in music and epidemic.enabled():
            # an Epidemic Sound track picked by hand: start at its most popular section for short videos
            mm = re.search(r"\[([^\]]+)\]\.mp3$", str(music["path"]))
            if mm and str(Path(music["path"]).parent) == str(epidemic.FOLDER) and dur <= 75:
                st0 = epidemic.highlight_start(mm.group(1), dur)
                try:
                    length = float(media.probe(music["path"]).get("duration") or 0)
                except Exception:
                    length = 0
                if st0 and length and length - st0 < dur:
                    st0 = max(0.0, length - dur - 1)
                music = dict(music, start=st0)
                s["music"] = music
        if music.get("path"):
            grid = audio.shift_grid(audio.detect_beats(music["path"]), float(music.get("start") or 0))
        else:
            grid = audio.music_grid(music["mood"], dur, seed)
        plan = p.get("plan") or story.plan_edit(analyses, s, grid, transcripts, taste)
        if skim and plan.get("shortened") and not p.get("plan"):
            # not enough strong moments in the stretches studied so far: look closely at more of the footage, once
            job.set_stage("analyse", "Looking closely at more moments")
            more = quick.plan_windows([q for q in quicks if q["path"] not in full], dur, picks, order, boost=3.0)
            redo = [q for q in quicks if q["path"] in more and more[q["path"]] != wins.get(q["path"])]
            extra = [None] * len(redo)

            def close_more(i):
                q = redo[i]
                if job.cancelled:
                    raise InterruptedError
                extra[i] = checked(lambda: quick.close_analysis(q, more[q["path"]]), q["path"])
                job.update((i + 1) / max(1, len(redo)), f"{i + 1} of {len(redo)} clips")
            run_parallel(close_more, len(redo), workers)
            if any(extra):
                newer = {a["path"]: a for a in extra if a}
                analyses = [newer.get(a["path"], a) for a in analyses] + [a for pth, a in newer.items()
                                                                           if pth not in {x["path"] for x in analyses}]
                plan = story.plan_edit(analyses, s, grid, transcripts, taste)
        if want_words and skim:  # fast watching: listen only to the clips that made it into the video
            used = {sh["src"] for sh in plan.get("shots", [])}
            pool = [a for a in analyses if a["path"] in used]
            listen_to(pool, sum(float(a["info"].get("duration") or 0) for a in pool))
        s.pop("_ai", None)
        # Google AI colourist: gentle, per-clip corrections before the look, only for clips in the edit
        if gemini.cfg()["key"] and load_settings().get("google_colour", True) and s.get("preset") != "None":
            job.set_stage("colour", "Google AI is grading each clip")
            used = {sh["src"] for sh in plan.get("shots", [])}
            s["ai_colour"] = ai_colour_pass([a for a in analyses if a["path"] in used], s, job, notices)
        if ai:
            plan["ai"] = {"used": True, "cost_usd": round(sum(v.get("usd", 0) for v in ai.values()), 4)}
        if ai_note:
            notices.append(ai_note)
        if unheard:
            names = ", ".join(Path(x).name for x in unheard[:3]) + (f" and {len(unheard) - 3} more" if len(unheard) > 3 else "")
            notices.append(f"The speech in {names} couldn't be heard (its sound track is damaged), so "
                           f"{'those clips have' if len(unheard) > 1 else 'that clip has'} no captions.")
        # the sound: music, sound effects and level changes, as editable data (the sound timeline edits it later)
        if not s.get("sound"):
            from . import soundplan
            s["_epidemic_expected"] = epidemic.enabled()
            s["sound"] = soundplan.build(plan, s, music, dur, np.random.default_rng((int(s.get("seed", 0)) * 7 + 3) % (2 ** 32)), notices)
        for n_ in gemini.take_switch_notes():
            if n_ not in notices:
                notices.insert(0, n_)
        plan["notices"] = notices
        if bad:
            names = ", ".join(Path(b[0]).name for b in bad[:4]) + (f" and {len(bad) - 4} more" if len(bad) > 4 else "")
            plan["warning"] = (plan.get("warning") + " " if plan.get("warning") else "") + \
                f"{len(bad)} file{'s were' if len(bad) > 1 else ' was'} skipped because {'they' if len(bad) > 1 else 'it'} couldn't be read: {names}."
        plan["music"] = {k: music.get(k) for k in ("path", "mood", "title", "creator", "license", "source")}
        job.update(1.0)
        # 5. render each format
        out_dir = project_dir(s.get("name") or s.get("event_type") or "Event")
        results = []
        for f in formats:
            fs = dict(s, format=f, name=export_name(s.get("name") or s.get("event_type") or "Event", f, out_dir))
            fdir = out_dir / f.replace(":", "x")
            t0 = time.time()
            job.set_stage(f"render_{f}", f"Rendering {f}")
            rstate = {"stage": "render"}

            def prog(stage, frac, note="", f=f):
                if stage != "render" and rstate["stage"] == "render":
                    rstate["stage"] = "finish"
                    job.set_stage(f"finish_{f}", f"Finishing {f}")
                order = {"assemble": 0.25, "graphics": 0.55, "audio": 0.8, "export": 0.95}
                if stage == "render":
                    job.update(frac, note)
                else:
                    job.update(min(1, order.get(stage, 0) + frac * 0.2), note)
            r = render.Renderer(plan, fs, analyses, fdir, prog, lambda: job.cancelled).run(transcripts if s.get("captions") else None)
            update_speed("render", (time.time() - t0) * 0.75 / dur)
            update_speed("finish", (time.time() - t0) * 0.25 / dur)
            results.append(r)
            if r.get("replaced") and not any("damaged part" in n for n in notices):
                names = sorted({Path(x["was"]).name for x in r["replaced"]})
                notices.append(f"{len(r['replaced'])} shot{'s' if len(r['replaced']) > 1 else ''} came from a damaged part of "
                               f"{', '.join(names[:3])}{' and others' if len(names) > 3 else ''} and {'were' if len(r['replaced']) > 1 else 'was'} "
                               "replaced with the next best moments.")
        plan["notices"] = notices
        job.close_stage()
        job.stage_key = None
        timing = {"total": round(time.time() - job.started, 1), "steps": job.timings}
        log(f"[{job.id}] finished in {timing['total']:.0f} s: " + ", ".join(f"{t['step']} {t['seconds']:.0f} s" for t in job.timings))
        meta = {"settings": {k: v for k, v in s.items() if k not in ("director_picks", "match_look") and not k.startswith("_")}, "plan": plan,
                "outputs": results, "created": time.strftime("%Y-%m-%d %H:%M"), "user": job.user.get("email"),
                "media": [{"path": m["path"], "kind": m["kind"]} for m in items], "timing": timing}
        (out_dir / "project.json").write_text(json.dumps(meta, indent=1), "utf-8")
        job.result = {"folder": str(out_dir), "outputs": results, "plan": plan, "version": out_dir.name, "settings": meta["settings"],
                      "timing": timing, "media": meta["media"]}
        job.state, job.stage, job.progress = "done", "Done", 1.0
        job.eta = {"stage_left": 0, "total_left": 0, "elapsed": round(time.time() - job.started)}
        job.push()
        play_sound(True)
    except InterruptedError:
        job.state, job.stage = "cancelled", "Cancelled"
        job.push()
    except Exception as e:
        fail_job(job, e)
        play_sound(False)


def friendly_error(e) -> str:
    """One plain sentence for the site, plus the useful line from FFmpeg when that's what failed."""
    raw = str(e) or e.__class__.__name__
    low = raw.lower()
    if isinstance(e, MemoryError) or "out of memory" in low or "cannot allocate memory" in low or "bad_alloc" in low:
        return "This PC ran out of memory. Close other apps, or split the footage into smaller projects, then resume."
    if "no space left" in low or "disk full" in low or "errno 28" in low:
        return "The disk is full. Free some space on the drive that holds the EventCut folder, then resume."
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    first = lines[0] if lines else raw
    detail = next((l for l in reversed(lines[1:]) if not l.startswith("Conversion failed")), "")
    return (first + (f" ({detail[:220]})" if detail else ""))[:400]


def fail_job(job, e):
    """Mark a job failed, keep which step it was on, and write the full details to Logs/ for support."""
    tb = traceback.format_exc()
    print(tb)
    job.failed_stage = job.stage
    job.state, job.error = "failed", friendly_error(e)
    job.stage = f"Stopped while {job.stage[0].lower() + job.stage[1:]}" if job.stage else "Something went wrong"
    try:
        logs = LOGS
        logs.mkdir(parents=True, exist_ok=True)
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "version": VERSION, "job": job.id, "kind": job.kind, "title": job.title,
               "clips": job.clips, "stage": job.failed_stage, "progress": round(job.progress, 3), "error": job.error, "detail": tb[-6000:]}
        (logs / "last_error.json").write_text(json.dumps(rec, indent=1), "utf-8")
        with open(logs / "errors.log", "a", encoding="utf-8") as f:
            f.write(f"\n===== {rec['time']}  {job.kind}  {job.title}  (engine {VERSION})\nStep: {job.failed_stage}\n{tb}\n")
    except Exception:
        pass
    job.push()


def waveform(path, points=600):
    """Peaks for drawing a track on the sound timeline (cached)."""
    points = max(50, min(3000, int(points)))
    out = WORK / "_waveforms" / (hashlib.sha1(f"{path}|{os.path.getsize(path)}|{points}".encode()).hexdigest()[:16] + ".json")
    if out.exists():
        return json.loads(out.read_text())
    ff, _ = media.tools()
    p = media.run([ff, "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "4000", "-f", "f32le", "-"])
    x = np.abs(np.frombuffer(p.stdout, np.float32))
    dur = len(x) / 4000.0
    if not len(x):
        return {"duration": 0, "peaks": []}
    step = max(1, len(x) // points)
    n = len(x) // step
    peaks = x[: n * step].reshape(n, step).max(axis=1)
    peaks = peaks / (peaks.max() + 1e-9)
    res = {"duration": round(dur, 3), "peaks": [round(float(v), 3) for v in peaks]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    return res


def run_remix(job: Job):
    """Audio-only re-render: new music, volumes or SFX without re-rendering the picture."""
    p = job.payload
    try:
        job.state = "running"
        job.plan_stages({"audio": 10, "export": 3})
        folder = Path(p["folder"])
        meta = json.loads((folder / "project.json").read_text("utf-8"))
        s = dict(meta["settings"], **(p.get("settings") or {}))
        outs = []
        for o in meta["outputs"]:
            fdir = Path(o["video"]).parent
            fs = dict(s, format=o["format"], name=Path(o["video"]).stem)
            r = render.Renderer(meta["plan"], fs, [], fdir)
            job.set_stage("audio", "Re-mixing audio")
            video = r.work / "video.mp4"
            body = next(r.work.glob("body.mov"), None) or next(r.work.glob("run_0.mov"))
            total = float(media.probe(str(video))["duration"])
            trans_at, t = [], 0.0
            for sh in meta["plan"]["shots"]:
                if sh["transition"] in render.XFADE:
                    trans_at.append((t, sh["transition"]))
                t += sh["length"]
            mix = r.mix_audio(body, total, trans_at)
            job.set_stage("export", "Exporting")
            outs.append(r.export(video, mix))
        meta["settings"].update(p.get("settings") or {})
        meta["outputs"] = outs
        (folder / "project.json").write_text(json.dumps(meta, indent=1), "utf-8")
        job.result = {"folder": str(folder), "outputs": outs}
        job.state, job.stage, job.progress = "done", "Done", 1.0
        job.eta = {"stage_left": 0, "total_left": 0, "elapsed": round(time.time() - job.started)}
        job.push()
        play_sound(True)
    except Exception as e:
        fail_job(job, e)
        play_sound(False)


def run_recaption(job: Job):
    """Caption edits: re-draw the captions on the finished picture (no re-edit, no re-render of shots)."""
    p = job.payload
    try:
        job.state = "running"
        folder = Path(p["folder"])
        meta = json.loads((folder / "project.json").read_text("utf-8"))
        s = dict(meta["settings"], **(p.get("settings") or {}), captions=True)
        words = [w for w in (p.get("words") or []) if str(w.get("word", "")).strip()]
        job.plan_stages({f"cap_{i}": 20 for i in range(len(meta["outputs"]))})
        outs = []
        for i, o in enumerate(meta["outputs"]):
            job.set_stage(f"cap_{i}", f"Updating captions, {o['format']}")
            fdir = Path(o["video"]).parent
            fs = dict(s, format=o["format"], name=Path(o["video"]).stem)
            r = render.Renderer(meta["plan"], fs, [], fdir)
            body = next(r.work.glob("body.mov"), None) or next(r.work.glob("run_0.mov"))
            end = r.work / "endcard.mov"
            end = end if end.exists() and s.get("end_card", True) else None
            caps = r.captions_ass(words) if words else None
            if words:
                r.write_srt(words)
                (fdir / "captions.json").write_text(json.dumps(words), "utf-8")
            video = r.finish_video(body, end, caps)
            mix = r.work / "mix.wav"
            outs.append(r.export(video, mix))
        # remember the corrections, so the next version of a video from this footage spells it right
        fixes = {}
        for w in words:
            if w.get("src") and w.get("s0") is not None and w.get("edited"):
                fixes.setdefault(w["src"], []).append(w)
        for src, ws in fixes.items():
            f = smart.transcript_path(analyze.proxy_key(src))
            if f.exists():
                tr = json.loads(f.read_text("utf-8"))
                for w in ws:
                    for t in tr:
                        if abs(t["start"] - float(w["s0"])) < 0.05:
                            t["word"] = " " + str(w["word"]).strip()
                            break
                f.write_text(json.dumps(tr), "utf-8")
        meta["settings"].update({k: v for k, v in (p.get("settings") or {}).items()})
        meta["settings"]["captions"] = bool(words)
        meta["outputs"] = outs
        (folder / "project.json").write_text(json.dumps(meta, indent=1), "utf-8")
        job.result = {"folder": str(folder), "outputs": outs}
        job.state, job.stage, job.progress = "done", "Done", 1.0
        job.eta = {"stage_left": 0, "total_left": 0, "elapsed": round(time.time() - job.started)}
        job.push()
        play_sound(True)
    except Exception as e:
        fail_job(job, e)
        play_sound(False)


# Big jobs take turns: two at once would each run at half speed and fight over memory and the graphics card,
# so the first one finishes sooner when they queue. Settings: max_parallel_jobs (1 recommended, 2 on strong PCs).
_QUEUE, _RUNNING = [], [0]
_QCOND = threading.Condition()


def run_sort(job: Job):
    """Google AI sorts mixed-up footage into events (nothing is deleted; the site removes files from the project)."""
    from . import sortfiles
    p = job.payload
    try:
        job.state = "running"
        job.plan_stages({"sort": 1})
        job.set_stage("sort", "Sorting your footage by event")

        def note(msg):
            job.note = msg
            job.push()
        gemini._wait_note.fn = note
        job.result = sortfiles.ai_sort(p.get("groups") or [], p.get("event", ""), p.get("hint", ""),
                                       lambda f, n="": job.update(f, n), lambda: job.cancelled)
        job.state, job.stage, job.progress = "done", "Done", 1.0
        job.push()
    except InterruptedError:
        job.state, job.stage = "cancelled", "Cancelled"
        job.push()
    except Exception as e:
        fail_job(job, e)
    finally:
        gemini._wait_note.fn = None


def _keep_awake(on: bool):
    """Stop Windows going to sleep while a video is being made (the screen may still turn off)."""
    if platform.system() != "Windows":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:
        pass


def _queued(job, target):
    _keep_awake(True)
    try:
        _queued_inner(job, target)
    finally:
        _keep_awake(False)


def _queued_inner(job, target):
    heavy = job.kind == "generate"
    if heavy:
        with _QCOND:
            _QUEUE.append(job)
            while True:
                limit = max(1, int(load_settings().get("max_parallel_jobs", 1)))
                if job.cancelled:
                    _QUEUE.remove(job)
                    job.state, job.stage = "cancelled", "Cancelled"
                    job.push()
                    _QCOND.notify_all()
                    return
                ahead = _QUEUE.index(job)
                if _RUNNING[0] < limit and ahead == 0:
                    _QUEUE.pop(0)
                    _RUNNING[0] += 1
                    break
                job.state = "queued"
                job.stage = f"Waiting for {_RUNNING[0] + ahead} other video{'s' if _RUNNING[0] + ahead != 1 else ''} to finish first"
                job.push()
                _QCOND.wait(3)
        job.started = time.time()
    try:
        target(job)
    finally:
        if heavy:
            with _QCOND:
                _RUNNING[0] -= 1
                _QCOND.notify_all()


def start_job(kind, payload, user):
    job = Job(kind, payload, user)
    JOBS[job.id] = job
    target = {"generate": run_generate, "remix": run_remix, "recaption": run_recaption, "sort": run_sort}[kind]
    threading.Thread(target=_queued, args=(job, target), daemon=True).start()
    return job


# ------------------------------------------------------------------ native pickers (browsers can't give real paths)
def pick(mode="folder"):
    if platform.system() != "Windows" and not os.environ.get("DISPLAY"):
        raise RuntimeError("No desktop available for the file picker.")
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if mode == "folder":
            p = filedialog.askdirectory(title="Choose an event folder")
            return [p] if p else []
        types = [("Media", " ".join(f"*{e}" for e in sorted(media.VIDEO_EXT | media.PHOTO_EXT | media.AUDIO_EXT))), ("All files", "*.*")]
        if mode == "audio":
            types = [("Audio", " ".join(f"*{e}" for e in sorted(media.AUDIO_EXT)))]
        if mode == "image":
            types = [("Images and animated logos", "*.png *.webp *.jpg *.gif *.mov *.webm")]
        if mode == "lut":
            types = [("LUT", "*.cube")]
        return list(filedialog.askopenfilenames(title="Choose files", filetypes=types))
    finally:
        root.destroy()


def open_in_explorer(path):
    if platform.system() == "Windows":
        p = Path(path)
        subprocess.Popen(["explorer", "/select,", str(p)] if p.is_file() else ["explorer", str(p)])
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", "-R", path])
    else:
        subprocess.Popen(["xdg-open", str(Path(path).parent if Path(path).is_file() else path)])


# ------------------------------------------------------------------ HTTP layer
class Handler(BaseHTTPRequestHandler):
    server_version = f"EventCutEngine/{VERSION}"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def origin_ok(self):
        origin = self.headers.get("Origin")
        if not origin:  # same-machine tools (curl) and media elements without CORS
            return True
        cfg = load_settings()
        allowed = set(cfg.get("allowed_origins", []))
        return origin in allowed or origin.endswith(".netlify.app") and cfg.get("allow_netlify_previews", False)

    def cors(self):
        origin = self.headers.get("Origin")
        if origin and self.origin_ok():
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Filename, X-Folder")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Access-Control-Max-Age", "600")

    def send_json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if self.headers.get("Content-Type", "").startswith("application/json"):
            return json.loads(raw or b"{}")
        return raw

    def user(self, q=None):
        auth = self.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else (q or {}).get("token", [""])[0]
        return verify_access(token)

    def do_OPTIONS(self):
        if not self.origin_ok():
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(204)
        self.cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        self.route("GET")

    def do_POST(self):
        self.route("POST")

    def route(self, method):
        if not self.origin_ok():
            return self.send_json({"error": "This site isn't allowed to use EventCut Engine."}, 403)
        u = urlparse(self.path)
        q = parse_qs(u.query)
        path = u.path.rstrip("/") or "/"
        try:
            if path == "/health":
                cfg = load_settings()
                return self.send_json({"ok": True, "version": VERSION, "gpu": media.gpu_available(), "gpu_label": media.hardware().get("label"),
                                       "gpu_notes": media.hardware().get("notes", []),
                                       "whisper": smart.speech_available(), "fal": falai.enabled(), "epidemic": epidemic.enabled(), "home": str(HOME),
                                       "linked": bool(cfg.get("supabase_url")), "cpu_threads": os.cpu_count()})
            if path == "/link" and method == "POST":
                # first sign-in on this PC: the site tells the engine which backend to check approvals against
                b = self.body()
                cfg = load_settings()
                if b.get("supabase_url", "").startswith("https://") and b.get("supabase_anon_key"):
                    cfg["supabase_url"], cfg["supabase_anon_key"] = b["supabase_url"].rstrip("/"), b["supabase_anon_key"]
                    save_settings(cfg)
                return self.send_json({"ok": True})
            user = self.user(q)
            return self.dispatch(method, path, q, user)
        except PermissionError as e:
            return self.send_json({"error": str(e), "access": False}, 401)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            traceback.print_exc()
            return self.send_json({"error": str(e).splitlines()[0][:300]}, 500)

    def dispatch(self, method, path, q, user):
        if path == "/me":
            return self.send_json({"user": user})
        if path == "/pick" and method == "POST":
            paths = pick(self.body().get("mode", "folder"))
            for p in paths:
                allow(p)
            return self.send_json({"paths": paths})
        if path == "/scan" and method == "POST":
            paths = self.body().get("paths", [])
            for p in paths:
                allow(p)
            found, skipped = media.scan(paths)
            out = []
            for m in found:
                try:
                    info = media.probe(m["path"]) if m["kind"] != "photo" else {}
                    m.update({"duration": info.get("duration", 0), "width": info.get("width"), "height": info.get("height"),
                              "fps": info.get("fps"), "hdr": media.is_hdr(info) if info else False})
                    out.append(m)
                except Exception:
                    skipped.append({"path": m["path"], "reason": "This file is damaged or can't be read, so it was skipped."})
            return self.send_json({"media": out, "skipped": skipped,
                                   "total_duration": round(sum(m.get("duration", 0) for m in out), 1),
                                   "total_size": sum(m["size"] for m in out)})
        if path == "/sort" and method == "POST":
            # group the project's files by when they were recorded (free, no internet)
            from . import sortfiles
            items = [m for m in (self.body().get("media") or []) if path_allowed(m.get("path", "")) and os.path.isfile(m.get("path", ""))]
            return self.send_json({"groups": sortfiles.group(items), "ai": bool(gemini.cfg()["key"])})
        if path == "/jobs" and method == "GET":
            # lets a reloaded page reconnect to renders still in progress
            return self.send_json({"jobs": [{k: v for k, v in j.snapshot().items() if k != "result"} for j in JOBS.values()
                                            if j.kind != "sort" and j.state in ("queued", "running") or time.time() - j.started < 86400]})
        if path == "/jobs" and method == "POST":
            b = self.body()
            job = start_job(b.get("kind", "generate"), b, user)
            return self.send_json(job.snapshot())
        m = re.match(r"^/jobs/([a-f0-9]+)(/cancel|/events|/skip)?$", path)
        if m:
            job = JOBS.get(m.group(1))
            if not job:
                return self.send_json({"error": "Job not found"}, 404)
            if m.group(2) == "/cancel":
                job.cancelled = True
                return self.send_json({"ok": True})
            if m.group(2) == "/skip":
                job.skip_stage = True  # stop listening, carry on with what was heard
                return self.send_json({"ok": True})
            if m.group(2) == "/events":
                return self.sse(job)
            return self.send_json(job.snapshot())
        if path == "/media":
            return self.stream_file(q.get("path", [""])[0], q.get("download", ["0"])[0] == "1")
        if path == "/preview":
            # a small, browser-friendly video for hover previews (4K camera files often can't play in a browser)
            p = q.get("path", [""])[0]
            if not p or not path_allowed(p) or not os.path.isfile(p):
                return self.send_json({"error": "File not available"}, 404)
            proxy = analyze.CACHE / f"{analyze.proxy_key(p)}_proxy.mp4"
            if proxy.exists() and proxy.stat().st_size > 1000 and media.probe(str(proxy)).get("pix_fmt") in ("yuv420p", "yuvj420p"):
                return self.send_json({"path": str(proxy), "start": 0.3})  # the working copy, if browsers can play it
            out = WORK / "_previews" / f"{analyze.proxy_key(p)}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                ff, _ = media.tools()
                dur = float(media.probe(p).get("duration") or 0)
                ss = max(0.0, dur * 0.35 - 1) if dur > 10 else 0.0
                media.run_decoding([ff, "-y", "-v", "error"] + media.decode_args() + ["-ss", f"{ss:.2f}", "-i", p, "-t", "8", "-an",
                                    "-vf", "scale=-2:360,fps=20", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p",
                                    "-movflags", "+faststart", str(out)])
            return self.send_json({"path": str(out), "start": 0})
        if path == "/thumb":
            return self.thumb(q.get("path", [""])[0], float(q.get("t", ["1"])[0]), int(q.get("w", ["480"])[0]))
        if path == "/grade-preview" and method == "POST":
            return self.grade_preview(self.body())
        if path == "/library/search":
            qq, cat, pg = q.get("q", [""])[0], q.get("category", ["music"])[0], int(q.get("page", ["1"])[0])
            originals = audio.originals_for(qq) if cat == "music" and pg == 1 else []
            try:
                res = audio.search_online(qq, cat, pg)
                res["originals"] = originals
                return self.send_json(res)
            except Exception:
                return self.send_json({"error": "The online library can't be reached. Check your internet connection.", "results": [],
                                       "originals": originals, "pages": 1}, 200 if originals else 502)
        if path == "/library/add" and method == "POST":
            b = self.body()
            return self.send_json({"path": audio.download_online(b["item"], b.get("category", "music"))})
        if path == "/library":
            return self.send_json(audio.ensure_library())
        if path == "/library/upload" and method == "POST":
            name = safe_name(unquote(self.headers.get("X-Filename", "upload.mp3")), "upload.mp3")
            folder = self.headers.get("X-Folder", "Uploads")
            dest_dir = {"Uploads": MUSIC_UPLOADS, "Logos": LOGOS, "Fonts": FONTS,
                        "LUTs": LOOKS, "Assets": WORK / "Brand kit"}.get(folder, MUSIC_UPLOADS)
            dest_dir.mkdir(parents=True, exist_ok=True)
            data = self.body()
            if folder == "Logos" and not render.logo_bytes_ok(data, name):
                return self.send_json({"error": "That logo file couldn't be read. Try a PNG with a transparent background."}, 400)
            dest = dest_dir / name
            dest.write_bytes(data)
            os.utime(dest, None)  # newest file = default logo
            return self.send_json({"path": str(dest), "name": dest.name})
        if path == "/ai/estimate" and method == "POST":
            paths = self.body().get("paths", [])
            secs = sum(float(media.probe(p)["duration"] or 0) for p in paths if path_allowed(p) and os.path.isfile(p))
            est = gemini.estimate(secs)
            est.update(seconds=round(secs), enabled=gemini.enabled(), month_spend=round(gemini.month_spend(), 3), cap=gemini.cfg()["cap"])
            return self.send_json(est)
        if path == "/selftest" and method == "POST":
            from . import selftest
            b = self.body()
            clip = b.get("clip")
            if clip and not path_allowed(clip):
                clip = None
            return self.send_json({"results": selftest.run(clip, bool(b.get("downloads")))})
        if path == "/last-error":
            f = LOGS / "last_error.json"
            return self.send_json(json.loads(f.read_text("utf-8")) if f.exists() else {})
        if path == "/open-logs" and method == "POST":
            LOGS.mkdir(parents=True, exist_ok=True)
            open_in_explorer(str(LOGS))
            return self.send_json({"ok": True})
        if path == "/epidemic/search":
            try:
                pg = max(1, int(q.get("page", ["1"])[0]))
                r = epidemic.search(q.get("q", [""])[0], limit=40, offset=(pg - 1) * 40, instrumental=q.get("instrumental", ["0"])[0] == "1")
                r["more"] = len(r["results"]) >= 30
                return self.send_json(r)
            except epidemic.EpidemicError as e:
                return self.send_json({"error": str(e), "results": []}, 502)
        if path == "/epidemic/preview" and method == "POST":
            b = self.body()
            try:
                start = epidemic.highlight_start(str(b["id"]), 45)
                return self.send_json({"path": epidemic.preview(str(b["id"]), start)})
            except epidemic.EpidemicError as e:
                return self.send_json({"error": str(e)}, 502)
        if path == "/epidemic/use" and method == "POST":
            b = self.body()
            try:
                pth = epidemic.download(str(b["id"]), b.get("title", ""), b.get("artist", ""))
                dur = float(b.get("duration") or 60)
                start = epidemic.highlight_start(str(b["id"]), dur) if dur <= 75 else 0.0
                length = float(b.get("length") or 0)
                if start and length and length - start < dur:
                    start = max(0.0, length - dur - 1)
                return self.send_json({"path": pth, "start": start, "title": f"{b.get('title', '')} by {b.get('artist', '')}",
                                       "source": "epidemic", "license": "Epidemic Sound"})
            except epidemic.EpidemicError as e:
                return self.send_json({"error": str(e)}, 502)
        if path == "/epidemic/similar":
            try:
                return self.send_json({"results": epidemic.similar(q.get("id", [""])[0], 15)})
            except epidemic.EpidemicError as e:
                return self.send_json({"error": str(e), "results": []}, 502)
        if path == "/epidemic/sfx":
            try:
                pg = max(1, int(q.get("page", ["1"])[0]))
                res = epidemic.sfx_search(q.get("q", ["whoosh"])[0] or "whoosh", 40, (pg - 1) * 40)
                return self.send_json({"results": res, "more": len(res) >= 30})
            except epidemic.EpidemicError as e:
                return self.send_json({"error": str(e), "results": []}, 502)
        if path == "/epidemic/sfx/use" and method == "POST":
            b = self.body()
            try:
                pth = epidemic.sfx_download(str(b["id"]), b.get("title", "Sound effect"))
                return self.send_json({"path": pth, "length": round(float(media.probe(pth).get("duration") or 0), 2),
                                       "source": "Epidemic Sound"})
            except epidemic.EpidemicError as e:
                return self.send_json({"error": str(e)}, 502)
        if path == "/epidemic/sfx/preview" and method == "POST":
            # effects are short: the download link doubles as the preview (Epidemic: "serves both download and preview")
            b = self.body()
            try:
                hit = epidemic._local_sfx(str(b["id"]))
                if hit:
                    return self.send_json({"path": str(hit)})
                d = epidemic._req(f"/v0/sound-effects/{b['id']}/download", {"format": "mp3", "quality": "normal"})
                return self.send_json({"url": d.get("url")})
            except epidemic.EpidemicError as e:
                return self.send_json({"error": str(e)}, 502)
        if path == "/waveform":
            pth = q.get("path", [""])[0]
            if not pth or not path_allowed(pth) or not os.path.isfile(pth):
                return self.send_json({"error": "File not available"}, 404)
            return self.send_json(waveform(pth, int(q.get("points", ["600"])[0])))
        if path == "/epidemic/test" and method == "POST":
            tb = self.body()
            try:
                epidemic.test_key((tb.get("key") or epidemic.cfg()["key"]).strip())
                return self.send_json({"ok": True})
            except epidemic.EpidemicError as e:
                return self.send_json({"ok": False, "error": str(e)}, 400)
        if path == "/fal/test" and method == "POST":
            tb = self.body()
            key = (tb.get("key") or falai.cfg()["key"]).strip()
            try:
                falai.test_key(key)
                falai._broken.update(until=0, why="")
                return self.send_json({"ok": True})
            except falai.FalError as e:
                return self.send_json({"ok": False, "error": str(e)}, 400)
        if path == "/ai/test" and method == "POST":
            tb = self.body()
            key = (tb.get("key") or gemini.cfg()["key"]).strip()
            try:
                model = gemini.test_key(key, tb.get("tier"))
                return self.send_json({"ok": True, "model": model, "name": gemini.pretty(model),
                                       "notes": gemini.take_switch_notes()})
            except gemini.AIError as e:
                return self.send_json({"ok": False, "error": str(e)}, 400)
        if path == "/captions":
            folder = q.get("folder", [""])[0]
            if not folder or not path_allowed(folder):
                return self.send_json({"words": []})
            f = next(Path(folder).glob("*/captions.json"), None)
            return self.send_json({"words": json.loads(f.read_text("utf-8")) if f else []})
        if path == "/feed-copy" and method == "POST":
            b = self.body()
            res = feed_copy(b.get("path", ""))
            src, t = b.get("frame_src"), b.get("frame_t")
            if src and t is not None and path_allowed(src) and os.path.isfile(src):
                looks = {"before": str(raw_frame(src, float(t), 960, wide=True))}
                for name in grade.PRESETS:
                    if name != "None":
                        looks[name] = str(grade_frame(src, float(t), name, 1.0, width=960, wide=True))
                res["looks"] = looks
            return self.send_json(res)
        if path == "/music-preview" and method == "POST":
            b = self.body()
            out = WORK / "_previews_audio" / f"{re.sub('[^A-Za-z]', '', b.get('mood', 'x'))}.wav"
            if not out.exists():
                audio.write_wav(out, audio.generate_music(b.get("mood", "Upbeat Afro House"), 20, 6))
            return self.send_json({"path": str(out)})
        if path == "/assets":
            return self.send_json(self.list_assets())
        if path == "/presets":
            return self.send_json({"presets": list(grade.PRESETS), "luts": grade.export_preset_luts()})
        if path == "/styles":
            if method == "POST":
                b = self.body()
                return self.send_json(smart.build_style(b["name"], b["paths"]))
            return self.send_json({"styles": smart.list_styles()})
        if path == "/taste" and method == "POST":
            b = self.body()
            t = story.record_taste(smart.load_taste(), b.get("action"), b.get("shot", {}), b.get("moment"))
            smart.save_taste(t)
            return self.send_json(t)
        if path == "/taste/reset" and method == "POST":
            smart.save_taste({})
            return self.send_json({"ok": True})
        if path == "/director/estimate" and method == "POST":
            paths = self.body().get("paths", [])
            an = []
            for p in paths:
                f = analyze.CACHE / f"{analyze.cache_key(p)}.json"
                if f.exists():
                    an.append(json.loads(f.read_text("utf-8")))
            est = smart.director_estimate(an, {}) if an else {"estimated_usd": 0.05, "data_notice": "About 60 small still frames and the transcript will be sent. Full videos never leave this PC."}
            est["month_spend"] = smart.month_spend()
            est["cap"] = load_settings().get("director_monthly_cap_usd", 20)
            return self.send_json(est)
        if path == "/transcript" and method == "POST":
            b = self.body()
            smart.save_transcript(analyze.proxy_key(b["path"]), b["words"])
            return self.send_json({"ok": True})
        if path == "/transcript":
            p = q.get("path", [""])[0]
            f = smart.transcript_path(analyze.proxy_key(p))
            return self.send_json({"words": json.loads(f.read_text("utf-8")) if f.exists() else []})
        if path == "/candidates":
            # alternative moments for swapping a clip in the review timeline
            p = q.get("path", [""])[0]
            f = analyze.CACHE / f"{analyze.cache_key(p)}.json"
            a = json.loads(f.read_text("utf-8")) if f.exists() else {"moments": []}
            return self.send_json({"moments": [dict(m, idx=i) for i, m in enumerate(a["moments"]) if not m.get("reject")]})
        if path == "/projects":
            return self.send_json({"projects": self.list_projects()})
        if path == "/open" and method == "POST":
            p = self.body().get("path")
            if p and path_allowed(p):
                open_in_explorer(p)
            return self.send_json({"ok": True})
        if path == "/settings":
            cfg = load_settings()
            if method == "POST":
                b = self.body()
                for k in ("notify_sound", "notify_volume", "director_mode_api_key", "director_monthly_cap_usd", "director_mode_model",
                          "google_api_key", "google_model", "google_monthly_cap_usd", "use_google_ai", "google_prices", "speech_language",
                          "google_colour", "max_parallel_jobs", "fal_api_key", "use_fal", "analyse_workers",
                          "epidemic_api_key", "use_epidemic", "epidemic_auto"):
                    if k in b:
                        cfg[k] = b[k]
                save_settings(cfg)
            safe = {k: v for k, v in cfg.items() if k not in ("director_mode_api_key", "supabase_anon_key", "google_api_key", "fal_api_key", "epidemic_api_key")}
            safe["epidemic_key_set"] = bool(cfg.get("epidemic_api_key"))
            safe["fal_key_set"] = bool(cfg.get("fal_api_key"))
            safe["fal_month_minutes"] = round(falai.month_minutes(), 1)
            safe["fal_note"] = falai._broken["why"] if not falai.enabled() and cfg.get("fal_api_key") else ""
            safe["cpu_count"] = os.cpu_count() or 0
            safe["analyse_workers_auto"] = min(4, max(1, round((os.cpu_count() or 4) / 3)))
            safe["director_key_set"] = bool(cfg.get("director_mode_api_key"))
            safe["google_key_set"] = bool(cfg.get("google_api_key"))
            safe["google_month_spend"] = round(gemini.month_spend(), 3)
            safe["google_prices"] = {**gemini.DEFAULT_PRICES, **(cfg.get("google_prices") or {})}
            return self.send_json(safe)
        if path == "/brand-kit" and method == "POST":
            # Team Brand Kit sync: the site sends logos (SVG already rasterised to PNG in the browser), fonts and theme
            kit = self.body()
            (WORK / "Brand kit").mkdir(parents=True, exist_ok=True)
            (WORK / "Brand kit" / "brand-kit.json").write_text(json.dumps({k: v for k, v in kit.items() if k != "files"}, indent=1))
            saved, bad = [], []
            for f in kit.get("files", []):
                ftype = f.get("type") or ("font" if str(f.get("name", "")).lower().endswith((".ttf", ".otf")) else "logo")
                folder = {"logo": LOGOS, "font": FONTS, "lut": LOOKS}.get(ftype, LOGOS)
                name = safe_name(f.get("name", ""), "logo.png")
                data = base64.b64decode(f.get("data") or "")
                if ftype == "logo" and not render.logo_bytes_ok(data, name):
                    bad.append(name)
                    continue
                dest = (folder / name).resolve()
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                saved.append(name)
            return self.send_json({"ok": True, "saved": saved, "unreadable": bad})
        return self.send_json({"error": "Not found"}, 404)

    def list_assets(self):
        out = {}
        for sub, d in (("Logos", LOGOS), ("Fonts", FONTS)):
            out[sub.lower()] = [{"name": f.name, "path": str(f)} for f in sorted(d.glob("*")) if f.is_file()] if d.exists() else []
        kit = WORK / "Brand kit" / "brand-kit.json"
        out["brand_kit"] = json.loads(kit.read_text()) if kit.exists() else None
        out["luts"] = [{"name": f.stem, "path": str(f)} for f in sorted(LOOKS.glob("*.cube"))]
        return out

    def list_projects(self):
        res = []
        for pj in sorted(FINISHED.glob("*/Version */project.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:200]:
            try:
                d = json.loads(pj.read_text("utf-8"))
                outs = d.get("outputs") or []
                for o in outs:  # a picture for the card in Your videos, whatever was chosen in Exports
                    o["card"] = card_picture(o)
                res.append({"event": pj.parent.parent.name, "version": pj.parent.name, "folder": str(pj.parent),
                            "created": d.get("created"), "outputs": outs, "settings": d.get("settings"),
                            "shots": len(d.get("plan", {}).get("shots", [])), "timing": d.get("timing")})
            except Exception:
                pass
        return res

    def sse(self, job):
        self.send_response(200)
        self.cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        last = None
        try:
            while True:
                snap = job.snapshot()
                if snap != last:
                    self.wfile.write(f"data: {json.dumps(snap)}\n\n".encode())
                    self.wfile.flush()
                    last = snap
                if job.state in ("done", "failed", "cancelled"):
                    break
                time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def stream_file(self, p, download=False):
        if not p or not path_allowed(p) or not os.path.isfile(p):
            return self.send_json({"error": "File not available"}, 404)
        size = os.path.getsize(p)
        ctype = mimetypes.guess_type(p)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else size - int(b)
            end = int(b) if a and b else size - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.cors()
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", ctype)
        # the file's real name, also for "Save video as" / the player's own download button (never "media")
        from urllib.parse import quote
        name = os.path.basename(p)
        ascii_name = re.sub(r'[^A-Za-z0-9._ -]+', "_", name)
        self.send_header("Content-Disposition", f"{'attachment' if download else 'inline'}; filename=\"{ascii_name}\"; "
                                                f"filename*=UTF-8''{quote(name)}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(p, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def thumb(self, p, t, w):
        if not p or not path_allowed(p):
            return self.send_json({"error": "File not available"}, 404)
        cache = WORK / "_thumbs"
        cache.mkdir(parents=True, exist_ok=True)
        out = cache / (hashlib.sha1(f"{p}|{t}|{w}".encode()).hexdigest()[:16] + ".jpg")
        if not out.exists():
            ff, _ = media.tools()
            if media.classify(Path(p)) == "photo":
                media.run([ff, "-y", "-v", "error", "-i", p, "-vf", f"scale={w}:-2", "-frames:v", "1", str(out)])
            else:
                for tt in (t, 1.0, 0.0):  # an unreadable spot gives no picture (FFmpeg doesn't even say so): try earlier
                    media.run([ff, "-y", "-v", "error", "-ss", f"{tt:.2f}", "-i", p, "-vf", f"scale={w}:-2", "-frames:v", "1", "-q:v", "4",
                               str(out)], check=False)
                    if out.exists() and out.stat().st_size > 0:
                        break
        if not out.exists():
            return self.send_json({"error": "No picture could be read from this file"}, 422)
        data = out.read_bytes()
        self.send_response(200)
        self.cors()
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "max-age=86400")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def grade_preview(self, b):
        """Before/after frame for the grading panel."""
        p = b.get("path")
        if not p or not path_allowed(p):
            return self.send_json({"error": "File not available"}, 404)
        t = float(b.get("t", 1))
        out, last = None, None
        for tt in (t, 1.0, 0.0):  # a damaged or unreadable spot: try the start of the clip instead
            try:
                out = grade_frame(p, tt, b.get("preset", "Vibrant Cinematic"), float(b.get("intensity", 1)), b.get("log_profile"))
                if out.exists() and out.stat().st_size > 0:
                    break
            except Exception as e:
                last = e
                out = None
        if out is None:
            log(f"look preview failed for {Path(p).name}: {str(last).splitlines()[-1][:160] if last else 'no frame'}")
            return self.send_json({"error": f"{Path(p).name} can't be previewed"}, 422)
        data = out.read_bytes()
        self.send_response(200)
        self.cors()
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve():
    ensure_folders()
    audio.ensure_library()
    grade.export_preset_luts()
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    httpd.daemon_threads = True
    print(f"EventCut Engine {VERSION} running at http://127.0.0.1:{PORT}  (folder: {HOME})")
    httpd.serve_forever()


if __name__ == "__main__":
    serve()
