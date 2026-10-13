"""Footage analysis: per-frame quality, faces, motion, audio energy, shots and candidate moments.

Everything runs on small proxies so hours of footage stay manageable, and results are
cached so regenerating a video never re-analyses the same file.
"""
import hashlib
import threading
import json
from pathlib import Path

import cv2
import numpy as np

from . import media
from .config import HOME

CACHE = HOME / "Projects" / "_analysis"
FPS = 10  # proxy frame rate used for analysis
_tls = threading.local()  # one face detector per worker thread (several clips are analysed at once)


def _model_path(name: str) -> str:
    import sys
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return str(base / "models" / name)


def _detect_faces(img):
    """YuNet face detector (MIT licence, bundled). Returns list of (x, y, w, h)."""
    _face = getattr(_tls, "face", None)
    if _face is None:
        _face = _tls.face = cv2.FaceDetectorYN.create(_model_path("face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.72, 0.3, 50)
    h, w = img.shape[:2]
    _face.setInputSize((w, h))
    _, faces = _face.detect(img)
    if faces is None:
        return []
    return [tuple(int(v) for v in f[:4]) for f in faces if f[2] > 6 and f[3] > 6]


def cache_key(path: str) -> str:
    st = Path(path).stat()
    return hashlib.sha1(f"{path}|{st.st_size}|{int(st.st_mtime)}|v8".encode()).hexdigest()[:16]


def proxy_key(path: str) -> str:
    """Working copies don't change between analysis versions, so they keep their name (no re-encoding 4K files)."""
    st = Path(path).stat()
    return hashlib.sha1(f"{path}|{st.st_size}|{int(st.st_mtime)}|v6".encode()).hexdigest()[:16]


def _frame_metrics(frame, prev_gray, detect_faces):
    small = cv2.resize(frame, (min(480, frame.shape[1]), int(frame.shape[0] * min(480, frame.shape[1]) / frame.shape[1])))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lum = gray.astype(np.float32) / 255.0
    m = {
        "luma": float(lum.mean()),
        "p5": float(np.percentile(lum, 5)),
        "p95": float(np.percentile(lum, 95)),
        "sat": float(hsv[..., 1].mean() / 255.0),
        "sharp": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        "motion": 0.0, "shift": 0.0,
        "faces": 0, "face_x": 0.5, "face_y": 0.4, "face_area": 0.0, "skin_luma": -1.0,
        "clip_hi": float((lum > 0.98).mean()), "clip_lo": float((lum < 0.02).mean()),
    }
    mid = (lum > 0.15) & (lum < 0.85)
    if mid.sum() > 50:
        m["rgb"] = [round(float(small[..., c][mid].mean() / 255.0), 4) for c in (2, 1, 0)]
    else:
        m["rgb"] = [m["luma"]] * 3
    m["fx"], m["fy"], m["focus"] = 0.5, 0.45, "centre"
    diff = None
    if prev_gray is not None and prev_gray.shape == gray.shape:
        diff = np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16))
        m["motion"] = float(diff.mean() / 255.0)
        # where the action is: centroid of strong changes (used when there are no faces)
        mask = diff > 28
        if mask.mean() > 0.004:
            ys, xs = np.nonzero(mask)
            m["fx"], m["fy"], m["focus"] = float(xs.mean() / gray.shape[1]), float(ys.mean() / gray.shape[0]), "motion"
        (dx, dy), _ = cv2.phaseCorrelate(np.float32(prev_gray), np.float32(gray))
        m["shift"] = float(np.hypot(dx, dy) / gray.shape[1])
    if detect_faces:
        faces = [(max(0, x), max(0, y), w, h) for x, y, w, h in _detect_faces(small)]
        if len(faces):
            areas = [w * h for (_, _, w, h) in faces]
            i = int(np.argmax(areas))
            x, y, w, h = faces[i]
            m.update(faces=int(len(faces)), face_x=float((x + w / 2) / gray.shape[1]),
                     face_y=float((y + h / 2) / gray.shape[0]), face_area=float(areas[i] / gray.size),
                     skin_luma=float(lum[y + h // 4: y + 3 * h // 4, x + w // 4: x + 3 * w // 4].mean()))
            # weighted centre of all faces keeps groups in frame when reframing
            cx = sum((fx + fw / 2) * fw * fh for fx, fy, fw, fh in faces) / sum(areas)
            cy = sum((fy + fh / 2) * fw * fh for fx, fy, fw, fh in faces) / sum(areas)
            m["group_x"] = float(cx / gray.shape[1])
            m["group_y"] = float(cy / gray.shape[0])
            # spread of all faces, so a crop never cuts people out of a group
            m["span"] = [float(min(fx for fx, _, _, _ in faces) / gray.shape[1]), float(max(fx + fw for fx, _, fw, _ in faces) / gray.shape[1]),
                         float(min(fy for _, fy, _, _ in faces) / gray.shape[0]), float(max(fy + fh for _, fy, _, fh in faces) / gray.shape[0])]
            m["fx"], m["fy"], m["focus"] = m["group_x"], m["group_y"], "faces"
            # every face, for multi-person framing: centre x/y, width, height (0..1) and mouth movement,
            # which tells who is talking (lower-face change minus upper-face change, so nodding doesn't count)
            boxes = []
            for fx, fy, fw, fh in faces:
                mouth = 0.0
                if diff is not None:
                    lo = diff[fy + int(fh * 0.6): fy + fh, fx + fw // 5: fx + fw - fw // 5]
                    up = diff[fy: fy + int(fh * 0.4), fx + fw // 5: fx + fw - fw // 5]
                    if lo.size and up.size:
                        mouth = float(lo.mean()) / (float(up.mean()) + 2.0)  # mouth change relative to the head's own movement
                gw, gh = gray.shape[1], gray.shape[0]
                boxes.append([round((fx + fw / 2) / gw, 4), round((fy + fh / 2) / gh, 4), round(fw / gw, 4), round(fh / gh, 4), round(mouth, 5)])
            m["boxes"] = boxes
    return m, gray


def detect_log(frames: list, info: dict) -> dict:
    """Guess whether footage is flat/Log. Returns {is_log, confident, profile}."""
    tags = " ".join(str(v) for v in list(info.get("tags", {}).values()) + list(info.get("vtags", {}).values())).lower()
    for key, prof in [("s-log3", "slog3"), ("slog3", "slog3"), ("s-log2", "slog3"), ("v-log", "vlog"), ("vlog", "vlog"),
                      ("c-log", "clog3"), ("f-log", "flog"), ("n-log", "nlog"), ("d-log", "dlog"),
                      ("apple log", "applelog"), ("logc", "logc")]:
        if key in tags:
            return {"is_log": True, "confident": True, "profile": prof}
    if not frames:
        return {"is_log": False, "confident": True, "profile": None}
    sat = np.median([f["sat"] for f in frames])
    lo = np.median([f["p5"] for f in frames])
    hi = np.median([f["p95"] for f in frames])
    contrast = hi - lo
    score = (sat < 0.16) + (contrast < 0.55) + (lo > 0.07) + (hi < 0.85)
    if score >= 4:
        return {"is_log": True, "confident": info.get("bit_depth", 8) >= 10, "profile": "generic_log"}
    if score == 3 and sat < 0.2:
        return {"is_log": True, "confident": False, "profile": "generic_log"}
    return {"is_log": False, "confident": True, "profile": None}


def analyse_video(path: str, progress=None) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = cache_key(path)
    out_json = CACHE / f"{key}.json"
    if out_json.exists():
        return json.loads(out_json.read_text("utf-8"))
    info = media.probe(path)
    proxy = media.make_proxy(path, CACHE / f"{proxy_key(path)}_proxy.mp4", fps=FPS)
    cap = cv2.VideoCapture(str(proxy))
    frames, prev, i = [], None, 0
    total = max(1, int(info["duration"] * FPS))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        m, prev = _frame_metrics(frame, prev, detect_faces=(i % 2 == 0))
        frames.append(m)
        i += 1
        if progress and i % 25 == 0:
            progress(min(1.0, i / total))
    cap.release()
    # carry face data to the odd frames we skipped
    for j in range(1, len(frames)):
        if frames[j]["faces"] == 0 and frames[j - 1]["faces"] > 0 and j % 2 == 1:
            for k in ("faces", "face_x", "face_y", "face_area", "skin_luma", "group_x", "group_y", "span", "fx", "fy", "focus", "boxes"):
                if k in frames[j - 1]:
                    frames[j][k] = frames[j - 1][k]

    env = media.audio_envelope(path, rate=FPS) if info["has_audio"] else np.zeros(0)
    cuts = media.scene_cuts(str(proxy))
    result = {
        "path": path, "key": key, "kind": "video", "info": info, "proxy": str(proxy),
        "frames": frames, "audio_db": [round(float(x), 2) for x in env], "cuts": cuts,
        "log": detect_log(frames, info),
    }
    result["moments"] = build_moments(result)
    add_signatures(result)
    result["captured"] = capture_time(path, info)
    out_json.write_text(json.dumps(result), "utf-8")
    return result


def signature(img) -> list:
    """Tiny visual fingerprint: layout (16x9 grey) + colour (hue histogram), to spot near-identical scenes."""
    small = cv2.resize(img, (16, 9), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    g = (g - g.mean()) / (g.std() + 1e-6)
    hsv = cv2.cvtColor(cv2.resize(img, (64, 36)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], (hsv[..., 1] > 50).astype(np.uint8), [12], [0, 180]).ravel()
    hist = hist / (hist.sum() + 1e-6)
    return [round(float(x), 3) for x in g] + [round(float(x), 3) for x in hist]


def similarity(a, b) -> float:
    """0..1: how alike two moments look (same person in the same place scores high)."""
    if not a or not b:
        return 0.0
    ga, gb = np.array(a[:144]), np.array(b[:144])
    ha, hb = np.array(a[144:]), np.array(b[144:])
    layout = float(np.dot(ga, gb) / (np.linalg.norm(ga) * np.linalg.norm(gb) + 1e-6))
    colour = float(np.minimum(ha, hb).sum()) if ha.sum() > 0.2 and hb.sum() > 0.2 else 0.5
    return 0.55 * max(0.0, layout) + 0.45 * colour


def add_signatures(a: dict):
    cap = cv2.VideoCapture(a["proxy"])
    for m in a["moments"]:
        cap.set(cv2.CAP_PROP_POS_MSEC, (m["start"] + m["end"]) / 2 * 1000)
        ok, f = cap.read()
        if ok:
            m["sig"] = signature(f)
    cap.release()


def capture_time(path: str, info: dict):
    """When the clip was recorded (camera metadata), so clips can be put in the order things happened."""
    tags = {**info.get("tags", {}), **info.get("vtags", {})}
    for k in ("com.apple.quicktime.creationdate", "creation_time", "date"):
        v = tags.get(k)
        if v and not str(v).startswith("1970") and not str(v).startswith("1904"):
            return str(v)
    return None


def analyse_photo(path: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = cache_key(path)
    out_json = CACHE / f"{key}.json"
    if out_json.exists():
        return json.loads(out_json.read_text("utf-8"))
    img = cv2.imread(path)
    if img is None:  # HEIC etc: let ffmpeg convert
        tmp = CACHE / f"{key}.jpg"
        ffmpeg, _ = media.tools()
        media.run([ffmpeg, "-y", "-v", "error", "-i", path, str(tmp)])
        img = cv2.imread(str(tmp))
    m, _ = _frame_metrics(img, None, True)
    h, w = img.shape[:2]
    score = _quality(m) + 0.6 * min(1.0, m["faces"] / 3)
    psig = signature(img)
    result = {"path": path, "key": key, "kind": "photo", "info": {"width": w, "height": h, "duration": 0}, "captured": None,
              "frames": [m], "moments": [{"start": 0, "end": 3.0, "score": round(score, 3), "energy": 0.3,
                                           "faces": m["faces"], "face_x": m.get("group_x", m["face_x"]),
                                           "quality": round(_quality(m), 3), "photo": True, "scene": 0, "scene_len": 3.0, "sig": psig}]}
    out_json.write_text(json.dumps(result), "utf-8")
    return result


def _quality(m: dict) -> float:
    """0..1: sharp, well exposed, not clipped."""
    sharp = min(1.0, m["sharp"] / 180.0)
    expo = 1.0 - min(1.0, abs(m["luma"] - 0.45) * 2.2)
    clip = 1.0 - min(1.0, (m["clip_hi"] + m["clip_lo"]) * 3)
    return 0.45 * sharp + 0.35 * expo + 0.2 * clip


def build_moments(a: dict) -> list:
    """Split each shot into 0.6-4 s candidate moments and score them."""
    frames, env = a["frames"], np.array(a["audio_db"]) if a["audio_db"] else None
    n = len(frames)
    if n == 0:
        return []
    dur = n / FPS
    bounds = [0.0] + [c for c in a["cuts"] if 0.3 < c < dur - 0.3] + [dur]
    loud = None
    if env is not None and len(env):
        base = np.percentile(env, 30)
        loud = np.clip((env - base) / 18.0, 0, 1)  # reactions: cheering, laughter, applause
    moments = []
    for scene, (s, e) in enumerate(zip(bounds, bounds[1:])):
        length = e - s
        if length < 0.4:
            continue
        step = 2.5 if length > 5 else length
        t = s
        while t < e - 0.4:
            w_end = min(e, t + max(step, 0.6))
            i0, i1 = int(t * FPS), max(int(t * FPS) + 1, int(w_end * FPS))
            fr = frames[i0:i1]
            if not fr:
                break
            q = float(np.mean([_quality(f) for f in fr]))
            shake = float(np.std([f["shift"] for f in fr])) if len(fr) > 2 else 0.0
            motion = float(np.mean([f["motion"] for f in fr]))
            faces = float(np.mean([f["faces"] for f in fr]))
            face_area = float(np.mean([f["face_area"] for f in fr]))
            fx = [f.get("group_x", f["face_x"]) for f in fr if f["faces"]]
            energy = float(loud[i0:i1].mean()) if loud is not None and len(loud[i0:i1]) else 0.0
            peak = float(loud[i0:i1].max()) if loud is not None and len(loud[i0:i1]) else 0.0
            blurry = np.mean([f["sharp"] for f in fr]) < 25
            dark = np.mean([f["luma"] for f in fr]) < 0.07
            blown = np.mean([f["clip_hi"] for f in fr]) > 0.35
            floor = motion > 0.2 and shake > 0.02  # whip/floor/pocket shots
            reject = bool(blurry or dark or blown or floor or shake > 0.035)
            score = (0.34 * q + 0.22 * min(1, faces / 2) + 0.12 * min(1, face_area * 25) +
                     0.18 * peak + 0.08 * energy + 0.06 * min(1, motion * 12))
            moments.append({"start": round(t, 2), "end": round(w_end, 2), "score": round(score, 3), "scene": scene,
                            "scene_len": round(length, 2),
                            "quality": round(q, 3), "energy": round(max(energy, peak * 0.8), 3),
                            "faces": round(faces, 2), "face_x": round(float(np.median(fx)) if fx else 0.5, 3),
                            "motion": round(motion, 3), "shake": round(shake, 4), "reject": reject,
                            "reason": "blurry" if blurry else "too dark" if dark else "overexposed" if blown else "shaky" if (floor or shake > 0.035) else ""})
            t = w_end
    return moments


def analyse(path: str, kind: str, progress=None) -> dict:
    return analyse_photo(path) if kind == "photo" else analyse_video(path, progress)
