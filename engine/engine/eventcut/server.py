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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import analyze, audio, grade, media, render, smart, story
from .config import HOME, PORT, VERSION, ensure_folders, load_settings, save_settings

JOBS = {}
ALLOWED_ROOTS = set()
_auth_cache = {}  # token hash -> (expires, user)
SPEED_FILE = HOME / "Projects" / "speed.json"


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
        offline = HOME / "Projects" / "_access.json"
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
    (HOME / "Projects" / "_access.json").write_text(json.dumps({"sub": claims.get("sub"), "user": user, "until": time.time() + 3 * 86400}))
    return user


def path_allowed(p: str) -> bool:
    rp = os.path.realpath(p)
    roots = {os.path.realpath(str(HOME))} | {os.path.realpath(r) for r in ALLOWED_ROOTS}
    return any(rp == r or rp.startswith(r + os.sep) for r in roots)


# ------------------------------------------------------------------ speed model for time-left estimates
def load_speed():
    d = {"analyse": 0.35, "transcribe": 0.25, "render": 6.0, "finish": 1.2}  # seconds per second of footage/output
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
        self.events = []
        self.weights = {}
        self.done_w = 0.0
        self.stage_key = None
        self.stage_t0 = time.time()

    def plan_stages(self, weights):
        """weights: ordered {stage: expected_seconds}."""
        self.weights = dict(weights)
        self.done_w = 0.0

    def set_stage(self, key, label):
        if self.stage_key and self.stage_key in self.weights:
            self.done_w += self.weights[self.stage_key]
        self.stage_key, self.stage, self.note = key, label, ""
        self.stage_t0 = time.time()
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
        return {"id": self.id, "kind": self.kind, "state": self.state, "stage": self.stage, "note": self.note,
                "progress": round(self.progress, 4), "eta": self.eta, "error": self.error, "result": self.result}


def play_sound(ok=True):
    cfg = load_settings()
    if not cfg.get("notify_sound", True):
        return
    f = audio.LIB / ("eventcut_done.wav" if ok else "eventcut_failed.wav")
    if platform.system() == "Windows" and f.exists():
        try:
            import winsound
            winsound.PlaySound(str(f), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            pass


def project_dir(name):
    safe = re.sub(r"[^A-Za-z0-9 _-]+", "", name or "Event").strip() or "Event"
    base = HOME / "Exports" / safe
    base.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split(" ")[-1]) for p in base.glob("Version *") if p.name.split(" ")[-1].isdigit()] or [0])
    d = base / f"Version {n}"
    d.mkdir()
    return d


def run_generate(job: Job):
    p = job.payload
    s = dict(p.get("settings") or {})
    try:
        job.state = "running"
        items = p.get("media") or media.scan(p.get("paths", []))[0]
        vids = [m for m in items if m["kind"] in ("video", "photo")]
        if not vids:
            raise ValueError("No videos or photos found in the selected folders.")
        foot = sum(media.probe(m["path"])["duration"] for m in vids if m["kind"] == "video") or 10
        cached = all((analyze.CACHE / f"{analyze.cache_key(m['path'])}.json").exists() for m in vids)
        sp = load_speed()
        formats = s.get("formats") or [s.get("format", "9:16")]
        dur = float(s.get("duration", 60))
        want_words = (s.get("captions") or s.get("prompt")) and smart.whisper_available()
        job.plan_stages({"analyse": 1 if cached else foot * sp["analyse"],
                         "transcribe": foot * sp["transcribe"] if want_words else 0,
                         "director": 25 if s.get("director") else 0, "plan": 2,
                         **{f"render_{f}": dur * sp["render"] for f in formats},
                         **{f"finish_{f}": dur * sp["finish"] for f in formats}})
        # 1. analyse
        job.set_stage("analyse", "Watching every clip")
        t0 = time.time()
        analyses = []
        for i, m in enumerate(vids):
            if job.cancelled:
                raise InterruptedError
            def prog(f, i=i):
                job.update((i + f) / len(vids), f"Clip {i + 1} of {len(vids)}")
            analyses.append(analyze.analyse(m["path"], m["kind"], prog))
            job.update((i + 1) / len(vids), f"Clip {i + 1} of {len(vids)}")
        if not cached:
            update_speed("analyse", (time.time() - t0) / foot)
        # 2. transcribe
        transcripts = {}
        if want_words:
            job.set_stage("transcribe", "Listening to speeches")
            t0 = time.time()
            for i, a in enumerate(a for a in analyses if a["kind"] == "video" and a["info"].get("has_audio")):
                transcripts[a["path"]] = smart.transcribe(a["path"], a["key"])
                job.update(min(1, (i + 1) / len(analyses)))
            update_speed("transcribe", (time.time() - t0) / foot)
        # 3. director (optional)
        if s.get("director"):
            job.set_stage("director", "Director mode is choosing the story")
            cfg = load_settings()
            if smart.month_spend() >= float(cfg.get("director_monthly_cap_usd", 20)):
                raise RuntimeError("Director mode has reached this month's spending cap. Ask the admin to raise it.")
            d = smart.director_plan(analyses, transcripts, s)
            s["director_picks"] = {f"{k[0]}|{k[1]}": v for k, v in d["picks"].items()}
            if d.get("title") and not s.get("title"):
                s["title"] = d["title"]
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
            family = s.get("music_family") or story.mood_from_prompt(s.get("prompt", ""), story.template_for(s.get("event_type", "")))
            job.update(0.3, "Choosing music")
            music = audio.pick_auto_music(family, dur, seed, online=s.get("online_music", True))
            s["music"] = music
        if music.get("path"):
            grid = audio.detect_beats(music["path"])
        else:
            grid = audio.music_grid(music["mood"], dur, seed)
        plan = p.get("plan") or story.plan_edit(analyses, s, grid, transcripts, taste)
        plan["music"] = {k: music.get(k) for k in ("path", "mood", "title", "creator", "license", "source")}
        job.update(1.0)
        # 5. render each format
        out_dir = project_dir(s.get("name") or s.get("event_type") or "Event")
        results = []
        for f in formats:
            fs = dict(s, format=f, name=f"{s.get('name') or 'EventCut'} {f.replace(':', 'x')}")
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
        meta = {"settings": {k: v for k, v in s.items() if k not in ("director_picks", "match_look")}, "plan": plan,
                "outputs": results, "created": time.strftime("%Y-%m-%d %H:%M"), "user": job.user.get("email")}
        (out_dir / "project.json").write_text(json.dumps(meta, indent=1), "utf-8")
        job.result = {"folder": str(out_dir), "outputs": results, "plan": plan, "version": out_dir.name}
        job.state, job.stage, job.progress = "done", "Done", 1.0
        job.eta = {"stage_left": 0, "total_left": 0, "elapsed": round(time.time() - job.started)}
        job.push()
        play_sound(True)
    except InterruptedError:
        job.state, job.stage = "cancelled", "Cancelled"
        job.push()
    except Exception as e:
        traceback.print_exc()
        job.state, job.stage, job.error = "failed", "Something went wrong", str(e).splitlines()[0][:300]
        job.push()
        play_sound(False)


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
        traceback.print_exc()
        job.state, job.error = "failed", str(e).splitlines()[0][:300]
        job.push()
        play_sound(False)


def start_job(kind, payload, user):
    job = Job(kind, payload, user)
    JOBS[job.id] = job
    target = {"generate": run_generate, "remix": run_remix}[kind]
    threading.Thread(target=target, args=(job,), daemon=True).start()
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
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Filename")
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
                                       "whisper": smart.whisper_available(), "home": str(HOME),
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
                ALLOWED_ROOTS.add(p if os.path.isdir(p) else os.path.dirname(p))
            return self.send_json({"paths": paths})
        if path == "/scan" and method == "POST":
            paths = self.body().get("paths", [])
            for p in paths:
                ALLOWED_ROOTS.add(p if os.path.isdir(p) else os.path.dirname(p))
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
        if path == "/jobs" and method == "GET":
            # lets a reloaded page reconnect to renders still in progress
            return self.send_json({"jobs": [{k: v for k, v in j.snapshot().items() if k != "result"} for j in JOBS.values()
                                            if j.state in ("queued", "running") or time.time() - j.started < 600]})
        if path == "/jobs" and method == "POST":
            b = self.body()
            job = start_job(b.get("kind", "generate"), b, user)
            return self.send_json(job.snapshot())
        m = re.match(r"^/jobs/([a-f0-9]+)(/cancel|/events)?$", path)
        if m:
            job = JOBS.get(m.group(1))
            if not job:
                return self.send_json({"error": "Job not found"}, 404)
            if m.group(2) == "/cancel":
                job.cancelled = True
                return self.send_json({"ok": True})
            if m.group(2) == "/events":
                return self.sse(job)
            return self.send_json(job.snapshot())
        if path == "/media":
            return self.stream_file(q.get("path", [""])[0], q.get("download", ["0"])[0] == "1")
        if path == "/thumb":
            return self.thumb(q.get("path", [""])[0], float(q.get("t", ["1"])[0]), int(q.get("w", ["480"])[0]))
        if path == "/grade-preview" and method == "POST":
            return self.grade_preview(self.body())
        if path == "/library/search":
            try:
                return self.send_json(audio.search_online(q.get("q", [""])[0], q.get("category", ["music"])[0], int(q.get("page", ["1"])[0])))
            except Exception:
                return self.send_json({"error": "The online library can't be reached. Check your internet connection.", "results": []}, 502)
        if path == "/library/add" and method == "POST":
            b = self.body()
            return self.send_json({"path": audio.download_online(b["item"], b.get("category", "music"))})
        if path == "/library":
            return self.send_json(audio.ensure_library())
        if path == "/library/upload" and method == "POST":
            name = re.sub(r"[^A-Za-z0-9 ._()-]+", "", self.headers.get("X-Filename", "upload.mp3"))
            folder = self.headers.get("X-Folder", "Uploads")
            dest_dir = {"Uploads": audio.LIB / "Uploads", "Logos": HOME / "Assets" / "Logos", "Fonts": HOME / "Assets" / "Fonts",
                        "LUTs": HOME / "Presets" / "LUTs", "Assets": HOME / "Assets"}.get(folder, audio.LIB / "Uploads")
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / name
            dest.write_bytes(self.body())
            return self.send_json({"path": str(dest)})
        if path == "/music-preview" and method == "POST":
            b = self.body()
            out = audio.LIB / "_previews" / f"{re.sub('[^A-Za-z]', '', b.get('mood', 'x'))}.wav"
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
            smart.save_transcript(analyze.cache_key(b["path"]), b["words"])
            return self.send_json({"ok": True})
        if path == "/transcript":
            p = q.get("path", [""])[0]
            f = smart.TRANS_CACHE / f"{analyze.cache_key(p)}.json"
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
                for k in ("notify_sound", "notify_volume", "director_mode_api_key", "director_monthly_cap_usd", "director_mode_model"):
                    if k in b:
                        cfg[k] = b[k]
                save_settings(cfg)
            safe = {k: v for k, v in cfg.items() if k not in ("director_mode_api_key", "supabase_anon_key")}
            safe["director_key_set"] = bool(cfg.get("director_mode_api_key"))
            return self.send_json(safe)
        if path == "/brand-kit" and method == "POST":
            # Team Brand Kit sync: the site sends logos (SVG already rasterised to PNG in the browser), fonts and theme
            kit = self.body()
            (HOME / "Assets" / "brand-kit.json").write_text(json.dumps({k: v for k, v in kit.items() if k != "files"}, indent=1))
            for f in kit.get("files", []):
                sub = {"logo": "Logos", "font": "Fonts", "lut": "../Presets/LUTs"}.get(f.get("type"), "")
                dest = (HOME / "Assets" / sub / re.sub(r"[^A-Za-z0-9 ._()-]+", "", f["name"])).resolve()
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(base64.b64decode(f["data"]))
            return self.send_json({"ok": True})
        return self.send_json({"error": "Not found"}, 404)

    def list_assets(self):
        out = {}
        for sub in ("Logos", "Fonts"):
            d = HOME / "Assets" / sub
            out[sub.lower()] = [{"name": f.name, "path": str(f)} for f in sorted(d.glob("*")) if f.is_file()] if d.exists() else []
        kit = HOME / "Assets" / "brand-kit.json"
        out["brand_kit"] = json.loads(kit.read_text()) if kit.exists() else None
        out["luts"] = [{"name": f.stem, "path": str(f)} for f in sorted((HOME / "Presets" / "LUTs").glob("*.cube"))]
        return out

    def list_projects(self):
        res = []
        for pj in sorted((HOME / "Exports").glob("*/Version */project.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:200]:
            try:
                d = json.loads(pj.read_text("utf-8"))
                res.append({"event": pj.parent.parent.name, "version": pj.parent.name, "folder": str(pj.parent),
                            "created": d.get("created"), "outputs": d.get("outputs"), "settings": d.get("settings"),
                            "shots": len(d.get("plan", {}).get("shots", []))})
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
        if download:
            from urllib.parse import quote
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(os.path.basename(p))}")
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
        cache = HOME / "Projects" / "_thumbs"
        cache.mkdir(parents=True, exist_ok=True)
        out = cache / (hashlib.sha1(f"{p}|{t}|{w}".encode()).hexdigest()[:16] + ".jpg")
        if not out.exists():
            ff, _ = media.tools()
            if media.classify(Path(p)) == "photo":
                media.run([ff, "-y", "-v", "error", "-i", p, "-vf", f"scale={w}:-2", "-frames:v", "1", str(out)])
            else:
                media.run([ff, "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", p, "-vf", f"scale={w}:-2", "-frames:v", "1", "-q:v", "4", str(out)])
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
        ff, _ = media.tools()
        key = analyze.cache_key(p)
        aj = analyze.CACHE / f"{key}.json"
        frames = json.loads(aj.read_text("utf-8"))["frames"] if aj.exists() else []
        log = (json.loads(aj.read_text("utf-8")).get("log") or {}) if aj.exists() else {}
        lut = grade.lut_for(log_profile=b.get("log_profile") or (log.get("profile") if log.get("is_log") else None),
                            correction=grade.correction_params(frames) if frames else None,
                            preset_name=b.get("preset", "Vibrant Cinematic"), intensity=float(b.get("intensity", 1)),
                            transfer=None, theme_hex=None)
        out = HOME / "Projects" / "_thumbs" / f"grade_{hashlib.sha1(json.dumps(b, sort_keys=True).encode()).hexdigest()[:12]}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            media.run([ff, "-y", "-v", "error", "-ss", f"{float(b.get('t', 1)):.2f}", "-i", p, "-vf",
                       f"scale=960:-2,lut3d=file='{render.fpath(lut)}'", "-frames:v", "1", "-q:v", "3", str(out)])
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
