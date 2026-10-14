"""Camera matching: footage from different cameras and phones is made to look like one shoot.

Clips are grouped by camera (their folder plus the camera model the file names). The camera with the most, and best
exposed, footage is the "hero"; every other camera gets a gentle colour match towards it before the look is applied.
HDR phone footage is converted to normal range first (the same way the video itself is made), so it compares fairly.
"""
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import grade, media
from .config import WORK

STATS_DIR = WORK / "_ai" / "camstats"
TONEMAP = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
SW, SH = 160, 90
VERSION = 1


def _model(info: dict, kind: str) -> str:
    tags = {**(info.get("tags") or {}), **(info.get("vtags") or {})}
    for k in ("com.apple.quicktime.model", "model", "com.android.model", "make", "com.apple.quicktime.make"):
        if tags.get(k):
            return str(tags[k]).strip()
    brand = str(tags.get("major_brand") or "").upper()
    if "XAVC" in brand:
        return "Sony"
    if kind == "photo":
        return "photos"
    return ""


def camera_key(a: dict) -> str:
    folder = os.path.basename(os.path.dirname(a["path"])) or "footage"
    return f"{folder}|{_model(a.get('info') or {}, a.get('kind', 'video'))}"


def label_for(key: str, keys) -> str:
    folder, model = key.split("|", 1)
    same_folder = [k for k in keys if k.split("|", 1)[0] == folder]
    if len(same_folder) > 1 and model:
        return f"{folder} ({model})"
    return folder


def _grab(src: str, t: float, hdr: bool):
    ff, _ = media.tools()
    vf = ([TONEMAP] if hdr else []) + [f"scale={SW}:{SH}"]
    args = [ff, "-v", "error"] + (["-ss", f"{t:.2f}"] if t else []) + ["-i", src, "-vf", ",".join(vf),
                                                                       "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = media.run(args, check=False, timeout=40)
    if p.returncode != 0 and hdr:  # zscale missing on this FFmpeg
        return _grab(src, t, False)
    buf = p.stdout or b""
    if len(buf) < SW * SH * 3:
        return None
    return np.frombuffer(buf[:SW * SH * 3], np.uint8).reshape(-1, 3).astype(np.float32) / 255


def clip_stats(a: dict, skim: str = None):
    """LAB colour statistics of a clip after its own correction (what the camera match starts from). Cached."""
    info = a.get("info") or {}
    log = a.get("log") or {}
    profile = log.get("profile") if log.get("is_log") else None
    corr = grade.correction_params(a.get("frames", [])) if a.get("frames") else None
    if profile and corr:
        corr = {"gains": corr["gains"], "gamma": 1.0}
    try:
        st = os.stat(a["path"])
        ident = f"{a['path']}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        ident = a["path"]
    tag = hashlib.sha1(json.dumps([ident, profile, corr, VERSION]).encode()).hexdigest()[:16]
    cf = STATS_DIR / f"{tag}.json"
    if cf.exists():
        try:
            return json.loads(cf.read_text("utf-8"))
        except Exception:
            pass
    if a.get("kind") == "photo":
        pix = [_grab(a["path"], 0, False)]
    else:
        dur = float(info.get("duration") or 0)
        # the skim copy is already converted from HDR; the original needs converting here
        src, hdr = (skim, False) if skim and os.path.exists(skim) else (a["path"], media.is_hdr(info))
        sdur = dur if src == a["path"] else float(a.get("_skim_dur") or dur)
        pix = [_grab(src, t, hdr) for t in ([sdur * f for f in (0.2, 0.5, 0.8)] if sdur > 3 else [min(1.0, sdur / 2)])]
    pix = [p for p in pix if p is not None]
    if not pix:
        return None
    x = np.concatenate(pix, 0)
    x = grade.decode_log(x, profile)
    if corr:
        x = grade.apply_correction(x, corr)
    y = grade.luma(x)
    out = {**grade.lab_stats(x), "luma": float(np.median(y)), "clip": float(np.mean((y > 0.97) | (y < 0.02)))}
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(out), "utf-8")
    return out


def _pool(stats: list) -> dict:
    """Combine clips: the mean of means, and the spread including how much the clips differ from each other."""
    m = np.array([s["mean"] for s in stats])
    v = np.array([np.square(s["std"]) for s in stats])
    mean = m.mean(axis=0)
    std = np.sqrt(v.mean(axis=0) + m.var(axis=0))
    return {"mean": mean.tolist(), "std": std.tolist(), "luma": float(np.median([s["luma"] for s in stats])),
            "clip": float(np.mean([s["clip"] for s in stats]))}


def measure(analyses: list, skims: dict = None, per_camera: int = 6, hero: str = None) -> dict:
    """Group the footage by camera and measure each camera's colours. Returns
    {"hero": key, "auto_hero": key, "cameras": [{key, label, clips, seconds, kind, stats}], "by_path": {path: key}}."""
    skims = skims or {}
    groups = {}
    for a in analyses:
        if a.get("kind") not in ("video", "photo"):
            continue
        groups.setdefault(camera_key(a), []).append(a)
    if not groups:
        return {}
    jobs = []
    for key, items in groups.items():
        # spread the samples across the event (first, middle, last...), not just the first few files
        idx = np.unique(np.linspace(0, len(items) - 1, min(per_camera, len(items))).round().astype(int))
        jobs += [(key, items[int(i)]) for i in idx]

    def one(j):
        try:
            return j[0], clip_stats(j[1], skims.get(j[1]["path"]))
        except Exception:
            return j[0], None
    found = {}
    with ThreadPoolExecutor(6) as pool:
        for key, st in pool.map(one, jobs):
            if st:
                found.setdefault(key, []).append(st)
    keys = list(groups)
    cams = []
    for key in keys:
        items = groups[key]
        secs = sum(float((a.get("info") or {}).get("duration") or 0) for a in items if a.get("kind") == "video")
        kind = "video" if any(a.get("kind") == "video" for a in items) else "photo"
        st = _pool(found[key]) if found.get(key) else None
        phone = bool(re.search(r"iphone|phone|android|samsung|pixel|tecno|infinix|redmi", key, re.I)) or \
            sum(1 for a in items if re.match(r"(IMG|VID|PXL)_", os.path.basename(a["path"]), re.I)) > len(items) / 2
        cams.append({"key": key, "label": label_for(key, keys), "clips": len(items), "seconds": round(secs, 1), "kind": kind,
                     "stats": st, "sample": items[len(items) // 2]["path"], "phone": phone})

    def hero_score(c):
        """Most footage, well exposed, rich colour; a dedicated camera is preferred to a phone when they're close."""
        if not c["stats"] or c["kind"] != "video":
            return -1
        st = c["stats"]
        exposure = max(0.2, 1 - abs(st["luma"] - 0.45) * 2) * max(0.3, 1 - st["clip"] * 3)
        colour = float(np.clip((st["std"][1] + st["std"][2]) / 2 / 14, 0.6, 1.25))
        phone = 0.8 if c.get("phone") else 1.0
        return c["seconds"] * exposure * colour * phone
    auto = max(cams, key=hero_score)["key"]
    chosen = hero if hero in groups and any(c["key"] == hero and c["stats"] for c in cams) else auto
    cams.sort(key=lambda c: (c["key"] != chosen, -c["seconds"], -c["clips"]))
    by_path = {a["path"]: key for key, items in groups.items() for a in items}
    return {"hero": chosen, "auto_hero": auto, "cameras": cams, "by_path": by_path}


def match_for(cams: dict, path: str, strength: float = 0.8):
    """The colour match for one clip (None for the hero camera, or when there is nothing to match)."""
    if not cams or not cams.get("cameras"):
        return None
    key = (cams.get("by_path") or {}).get(path)
    hero = cams.get("hero")
    if not key or key == hero:
        return None
    st = {c["key"]: c["stats"] for c in cams["cameras"]}
    if not st.get(key) or not st.get(hero):
        return None
    r = lambda d: {"mean": [round(v, 3) for v in d["mean"]], "std": [round(v, 3) for v in d["std"]]}
    return {"src": r(st[key]), "ref": r(st[hero]), "strength": strength}


def summary(cams: dict) -> list:
    """For the website: the cameras without the numbers."""
    return [{"key": c["key"], "label": c["label"], "clips": c["clips"], "seconds": c["seconds"], "kind": c["kind"],
             "hero": c["key"] == cams.get("hero"), "measured": bool(c["stats"]), "sample": c.get("sample")}
            for c in (cams or {}).get("cameras", [])]
