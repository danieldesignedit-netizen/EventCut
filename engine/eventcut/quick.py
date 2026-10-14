"""Fast watching: every clip is skimmed from its keyframes only, then only the promising moments are studied closely.

Camera files store a complete picture every half second or so; the frames in between are only changes. Decoding just
those complete pictures (2 a second) is several times faster than decoding every frame of 4K footage, and is plenty to
judge faces, light, sharpness, movement and sound. The best stretches of each clip (picked from this skim, plus Google
AI's picks when it is on) are then analysed frame by frame, exactly as before, so framing and quality checks stay precise.
"""
import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from . import analyze, media
from .config import WORK

QFPS = 2
CACHE = WORK / "_quick"
VERSION = "q1"


def quick_key(path: str) -> str:
    st = Path(path).stat()
    return hashlib.sha1(f"{path}|{st.st_size}|{int(st.st_mtime)}|{VERSION}".encode()).hexdigest()[:16]


def full_cached(path: str):
    """A full frame-by-frame analysis from an earlier run, if there is one (always preferred)."""
    f = analyze.CACHE / f"{analyze.cache_key(path)}.json"
    if f.exists():
        try:
            return json.loads(f.read_text("utf-8"))
        except Exception:
            return None
    return None


# ------------------------------------------------------------------ the skim
def make_quick_proxy(src: str, dst: Path, info: dict) -> Path:
    """2 pictures a second from keyframes only, 360p, plus the sound as small mono AAC (always present, for montages)."""
    if dst.exists() and dst.stat().st_size > 500:
        return dst
    ff, _ = media.tools()
    dur = float(info.get("duration") or 0)
    vf = [f"fps={QFPS}", "scale=-2:360:flags=fast_bilinear", "format=yuv420p"]
    if media.is_hdr(info):
        vf = ["zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"] + vf
    tmp = dst.with_suffix(".part.mp4")

    def cmd(skip, tone=True):
        # "discard" drops the in-between frames before they are even read into the decoder (about 1.7x faster than
        # only skipping them in the decoder); "skip_frame" is the backup for formats where discard is ignored
        c = [ff, "-y", "-v", "error"] + (["-discard:v", "nokey", "-skip_frame:v", "nokey"] if skip else []) + ["-i", src]
        if info.get("has_audio"):
            c += ["-map", "0:v:0", "-map", "0:a:0"]
        else:
            c += ["-f", "lavfi", "-t", f"{max(0.5, dur):.3f}", "-i", "anullsrc=r=16000:cl=mono", "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
        filt = vf if tone else [x for x in vf if not x.startswith("zscale")]
        c += ["-vf", ",".join(filt), "-c:v", "libx264", "-preset", "ultrafast", "-crf", "32", "-g", str(QFPS * 10),
              "-c:a", "aac", "-b:a", "32k", "-ac", "1", "-ar", "16000", "-movflags", "+faststart", str(tmp)]
        return c
    errors = []
    for skip, tone in ((True, True), (False, True), (True, False), (False, False)):
        try:
            media.run(cmd(skip, tone))
            if tmp.exists() and tmp.stat().st_size > 500:
                os.replace(tmp, dst)
                return dst
        except Exception as e:  # some files/decoders refuse keyframe-only decoding: decode normally instead
            errors.append(str(e).splitlines()[0])
    raise RuntimeError("Couldn't skim this clip. " + (errors[-1] if errors else ""))


def _qscore(fr, loud):
    """Score a stretch of the skim the same way the close analysis scores moments (sharp, lit, faces, reactions)."""
    q = float(np.mean([analyze._quality(f) for f in fr]))
    faces = float(np.mean([f["faces"] for f in fr]))
    area = float(np.mean([f["face_area"] for f in fr]))
    motion = float(np.mean([f["motion"] for f in fr]))
    peak = float(loud.max()) if loud is not None and len(loud) else 0.0
    energy = float(loud.mean()) if loud is not None and len(loud) else 0.0
    dark = np.mean([f["luma"] for f in fr]) < 0.07
    blurry = np.mean([f["sharp"] for f in fr]) < 18
    score = 0.34 * q + 0.22 * min(1, faces / 2) + 0.12 * min(1, area * 25) + 0.18 * peak + 0.08 * energy + 0.06 * min(1, motion * 4)
    return score, bool(dark or blurry), max(energy, peak * 0.8), faces


def quick_analyse(path: str, progress=None) -> dict:
    """Skim one clip. Cached, so a stopped or repeated run never skims the same file twice."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = quick_key(path)
    out = CACHE / f"{key}.json"
    if out.exists():
        try:
            return json.loads(out.read_text("utf-8"))
        except Exception:
            pass
    info = media.probe(path)
    qproxy = make_quick_proxy(path, CACHE / f"{key}.mp4", info)
    if progress:
        progress(0.6)
    cap = cv2.VideoCapture(str(qproxy))
    frames, prev = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        m, prev = analyze._frame_metrics(frame, prev, detect_faces=True)
        m.pop("boxes", None)
        frames.append(m)
    cap.release()
    env = media.audio_envelope(str(qproxy), rate=analyze.FPS) if info.get("has_audio") else np.zeros(0)
    loud = None
    if len(env):
        base = np.percentile(env, 30)
        loud = np.clip((env - base) / 18.0, 0, 1)
    dur = float(info.get("duration") or len(frames) / QFPS)
    windows = []
    step = 2.5
    t = 0.0
    while t < max(dur - 0.4, 0.01):
        e = min(dur, t + step)
        fr = frames[int(t * QFPS): max(int(t * QFPS) + 1, int(math.ceil(e * QFPS)))]
        if fr:
            lw = loud[int(t * analyze.FPS): int(e * analyze.FPS)] if loud is not None else None
            sc, rej, en, fa = _qscore(fr, lw)
            windows.append({"start": round(t, 2), "end": round(e, 2), "score": round(sc, 3), "reject": rej,
                            "energy": round(en, 3), "faces": round(fa, 2)})
        t = e
    good = sorted((w["score"] for w in windows if not w["reject"]), reverse=True)
    res = {"path": path, "kind": "video", "quick": True, "key": key, "info": info, "qproxy": str(qproxy),
           "qframes": frames, "audio_db": [round(float(x), 2) for x in env], "windows": windows,
           "clip_score": round(float(np.mean(good[:3])) if good else 0.0, 3),
           "captured": analyze.capture_time(path, info)}
    out.write_text(json.dumps(res), "utf-8")
    return res


# ------------------------------------------------------------------ choosing what to look at closely
WIN = 6.0  # seconds studied closely around each promising moment


def plan_windows(quicks: list, duration: float, ai_picks=None, order=None, boost: float = 1.0) -> dict:
    """{clip path: [(start, end), ...]}: the stretches worth a close look. Google AI's picks first, then the best of the
    skim spread across the whole event, so a story told in order still has material from every part of the day."""
    ai_picks = ai_picks or []
    n_target = int(np.clip(round(duration * 2.2 * boost), 36, int(220 * boost)))
    chosen = {}

    def add(path, s, e, dur):
        s, e = max(0.0, s), min(dur, e)
        if e - s < 0.8:
            return False
        lst = chosen.setdefault(path, [])
        for i, (a, b) in enumerate(lst):
            if s <= b + 1.0 and e >= a - 1.0:  # overlaps: merge
                lst[i] = (min(a, s), max(b, e))
                return True
        lst.append((s, e))
        return True

    by_path = {q["path"]: q for q in quicks}
    for p in ai_picks:  # 1. Google AI's picks, as asked in the Direction box
        q = by_path.get(p["path"])
        if not q:
            continue
        dur = float(q["info"].get("duration") or 0)
        mid = (p["start"] + p["end"]) / 2
        half = max(WIN, p["end"] - p["start"] + 2) / 2
        add(p["path"], mid - half, mid + half, dur)
    count = lambda: sum(len(v) for v in chosen.values())
    # 2. the skim's best, spread over the event in recording order
    cands = []
    for q in quicks:
        for w in q["windows"]:
            if not w["reject"]:
                cands.append((w["score"], q["path"], w))
    if not cands:
        return chosen
    order = order or [q["path"] for q in quicks]
    pos = {p: i for i, p in enumerate(order)}
    bins = 12
    per_bin = max(1, int(n_target * 0.5 / bins))
    if boost > 1:
        per_bin *= 2
    by_bin = {}
    for sc, p, w in cands:
        b = min(bins - 1, int(pos.get(p, 0) / max(1, len(order)) * bins))
        by_bin.setdefault(b, []).append((sc, p, w))
    per_clip = {}

    def take(sc, p, w):
        q = by_path[p]
        dur = float(q["info"].get("duration") or 0)
        limit = int((2 + int(dur // 60)) * boost)
        if per_clip.get(p, 0) >= limit:
            return False
        mid = (w["start"] + w["end"]) / 2
        if add(p, mid - WIN / 2, mid + WIN / 2, dur):
            per_clip[p] = per_clip.get(p, 0) + 1
            return True
        return False
    for b in sorted(by_bin):
        got = 0
        for sc, p, w in sorted(by_bin[b], key=lambda x: -x[0]):
            if got >= per_bin:
                break
            if take(sc, p, w):
                got += 1
    for sc, p, w in sorted(cands, key=lambda x: -x[0]):
        if count() >= n_target:
            break
        take(sc, p, w)
    # whole clip when the windows cover most of a short clip anyway
    for p, lst in chosen.items():
        dur = float(by_path[p]["info"].get("duration") or 0)
        if dur and sum(b - a for a, b in lst) >= dur * 0.7:
            chosen[p] = [(0.0, dur)]
        else:
            chosen[p] = sorted(lst)
    return chosen


# ------------------------------------------------------------------ the close look
def _window_frames(path: str, start: float, end: float, info: dict, cache_tag: str):
    """Frame-by-frame metrics (10 a second) for one stretch of a clip, from a small proxy of just that stretch."""
    WCACHE = CACHE / "windows"
    WCACHE.mkdir(parents=True, exist_ok=True)
    f = WCACHE / f"{cache_tag}_{start:.2f}_{end:.2f}.json"
    if f.exists():
        try:
            return json.loads(f.read_text("utf-8"))
        except Exception:
            pass
    ff, _ = media.tools()
    tmp = WCACHE / f"{cache_tag}_{start:.2f}_{end:.2f}.mp4"
    vf = [f"fps={analyze.FPS}", "scale=-2:360:flags=fast_bilinear"]
    if media.is_hdr(info):
        vf.insert(0, "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
    cmd = [ff, "-y", "-v", "error"] + media.decode_args() + ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", path,
                                                             "-vf", ",".join(vf), "-an", "-c:v", "libx264", "-preset", "ultrafast",
                                                             "-crf", "28", "-pix_fmt", "yuv420p", str(tmp)]
    try:
        media.run_decoding(cmd)
    except RuntimeError:
        if not media.is_hdr(info):
            raise
        cmd = [x for x in cmd]
        i = cmd.index("-vf")
        cmd[i + 1] = f"fps={analyze.FPS},scale=-2:360"
        media.run_decoding(cmd)
    cap = cv2.VideoCapture(str(tmp))
    frames, prev, i = [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        m, prev = analyze._frame_metrics(frame, prev, detect_faces=(i % 2 == 0))
        frames.append(m)
        i += 1
    cap.release()
    for j in range(1, len(frames)):  # carry face data to the frames between face checks
        if frames[j]["faces"] == 0 and frames[j - 1]["faces"] > 0 and j % 2 == 1:
            for k in ("faces", "face_x", "face_y", "face_area", "skin_luma", "group_x", "group_y", "span", "fx", "fy", "focus", "boxes"):
                if k in frames[j - 1]:
                    frames[j][k] = frames[j - 1][k]
    cuts = [round(c + start, 3) for c in media.scene_cuts(str(tmp))]
    try:
        tmp.unlink()
    except OSError:
        pass
    res = {"frames": frames, "cuts": cuts}
    f.write_text(json.dumps(res), "utf-8")
    return res


def close_analysis(q: dict, windows: list, progress=None) -> dict:
    """A normal EventCut analysis of a clip, built from its skim plus close looks at the chosen stretches.
    Only moments inside the closely-studied stretches can be used in the edit."""
    path, info = q["path"], q["info"]
    dur = float(info.get("duration") or 0)
    n = max(1, int(round(dur * analyze.FPS)))
    # outside the close looks, every 10th of a second borrows the nearest skim picture (never used for shots)
    qf = q["qframes"] or [{"luma": 0.4, "p5": 0.05, "p95": 0.9, "sat": 0.3, "sharp": 50.0, "motion": 0.0, "shift": 0.0, "faces": 0,
                           "face_x": 0.5, "face_y": 0.4, "face_area": 0.0, "skin_luma": -1.0, "clip_hi": 0.0, "clip_lo": 0.0,
                           "rgb": [0.4, 0.4, 0.4], "fx": 0.5, "fy": 0.45, "focus": "centre"}]
    frames = []
    for i in range(n):
        src = dict(qf[min(len(qf) - 1, int(i / analyze.FPS * QFPS))])
        src["motion"], src["shift"], src["filler"] = 0.0, 0.0, True
        frames.append(src)
    cuts = []
    for k, (s, e) in enumerate(windows):
        w = _window_frames(path, s, e, info, q["key"])
        i0 = int(round(s * analyze.FPS))
        for j, fr in enumerate(w["frames"]):
            if 0 <= i0 + j < n:
                frames[i0 + j] = fr
        cuts += w["cuts"]
        if progress:
            progress((k + 1) / len(windows))
    a = {"path": path, "key": analyze.cache_key(path), "kind": "video", "info": info, "proxy": q["qproxy"],
         "frames": frames, "audio_db": q["audio_db"], "cuts": sorted(cuts), "captured": q.get("captured"),
         "log": analyze.detect_log([f for f in frames if not f.get("filler")], info), "skimmed": True,
         "windows": [[round(s, 2), round(e, 2)] for s, e in windows]}
    moments = analyze.build_moments(a)
    inside = []
    for m in moments:
        for s, e in windows:
            if m["start"] >= s - 0.05 and m["end"] <= e + 0.05:
                inside.append(m)
                break
    a["moments"] = inside
    analyze.add_signatures(a)
    return a
