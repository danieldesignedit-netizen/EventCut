"""Sort mixed-up footage by event.

1. Without AI (free, instant): files are grouped by when they were recorded (camera date, photo EXIF date, else
   the file date). A new group starts on a new day or after a gap of more than 3 hours.
2. With Google AI (pennies): Google looks at a contact sheet of every group, names the events it sees, then checks
   every single file against those events, so a different event shot on the same day is still caught.
Nothing is ever deleted: the website only removes files from the project.
"""
import datetime as dt
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import analyze, media
from .config import HOME, WORK

GAP = 3 * 3600
TILE_W, TILE_H = 256, 144
CACHE = WORK / "_sort"


# ------------------------------------------------------------------ when was it recorded
def _parse_iso(v):
    try:
        v = str(v).strip().replace("Z", "+00:00").replace(" ", "T", 1)
        if "." in v:  # trim fractions to microseconds for older Pythons
            head, _, rest = v.partition(".")
            digits = re.match(r"\d*", rest).group(0)
            v = head + "." + (digits[:6] or "0") + rest[len(digits):]
        d = dt.datetime.fromisoformat(v)
        if d.tzinfo is None:
            return time.mktime(d.timetuple())
        return d.timestamp()
    except Exception:
        return None


def _exif_time(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            ex = im.getexif()
            v = None
            try:
                v = ex.get_ifd(0x8769).get(36867)  # DateTimeOriginal
            except Exception:
                pass
            v = v or ex.get(306)  # DateTime
        if v:
            return time.mktime(time.strptime(str(v).strip()[:19], "%Y:%m:%d %H:%M:%S"))
    except Exception:
        pass
    return None


def recorded_at(path, kind):
    """(epoch seconds, where the time came from)"""
    t = None
    if kind in ("video", "audio"):
        try:
            info = media.probe(path)
            for tags in (info.get("tags") or {}, info.get("vtags") or {}):
                for k in ("creation_time", "com.apple.quicktime.creationdate", "date"):
                    if tags.get(k):
                        t = _parse_iso(tags[k])
                        if t:
                            break
                if t:
                    break
        except Exception:
            pass
        if t and t < 946684800:  # cameras with an unset clock say 1970 or 2000
            t = None
        src = "camera"
    else:
        t = _exif_time(path)
        src = "photo"
    if not t:
        try:
            t, src = os.path.getmtime(path), "file"
        except OSError:
            t, src = 0, "file"
    return t, src


# ------------------------------------------------------------------ time groups (no AI)
def _folder(p):
    return Path(p).parent.name or str(Path(p).parent)


def _summary(gid, members):
    ts = [m["t"] for m in members]
    kinds = {}
    for m in members:
        kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
    folders = {}
    for m in members:
        f = _folder(m["path"])
        folders[f] = folders.get(f, 0) + 1
    vis = [m for m in members if m["kind"] in ("video", "photo")]
    vids = [m for m in vis if m["kind"] == "video"]
    pool = vids if len(vids) >= 4 else vis
    idx = np.linspace(0, len(pool) - 1, min(4, len(pool))).astype(int) if pool else []
    return {"id": gid, "start": min(ts), "end": max(ts), "counts": kinds,
            "size": sum(m.get("size", 0) for m in members),
            "folders": sorted(folders.items(), key=lambda x: -x[1])[:3],
            "samples": [pool[i]["path"] for i in idx],
            "paths": [m["path"] for m in members],
            "estimated": sum(m["src"] == "file" for m in members) > len(members) / 2}


def group(items):
    """items: [{path, kind, size}] -> time groups, oldest first."""
    rows = []
    with ThreadPoolExecutor(6) as ex:
        times = list(ex.map(lambda m: recorded_at(m["path"], m["kind"]), items))
    for m, (t, src) in zip(items, times):
        rows.append({**m, "t": t, "src": src})
    rows.sort(key=lambda r: r["t"])
    groups, cur = [], []
    for r in rows:
        if cur:
            last = cur[-1]["t"]
            new_day = time.localtime(last)[:3] != time.localtime(r["t"])[:3]
            if r["t"] - last > GAP or new_day:
                groups.append(cur)
                cur = []
        cur.append(r)
    if cur:
        groups.append(cur)
    return [_summary(f"g{i + 1}", g) for i, g in enumerate(groups)]


# ------------------------------------------------------------------ thumbnails and contact sheets
def tile(path, kind):
    """A small 16:9 picture of the file (cached)."""
    import cv2
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / (hashlib.sha1(f"{path}|{os.path.getsize(path)}".encode()).hexdigest()[:16] + ".jpg")
    if f.exists():
        img = cv2.imread(str(f))
        if img is not None:
            return img
    img = None
    try:
        if kind == "photo":
            from PIL import Image
            with Image.open(path) as im:
                im.draft("RGB", (TILE_W * 2, TILE_H * 2))
                try:
                    from PIL import ImageOps
                    im = ImageOps.exif_transpose(im)
                except Exception:
                    pass
                img = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:
            proxy = analyze.CACHE / f"{analyze.proxy_key(path)}_proxy.mp4"
            src = str(proxy) if proxy.exists() and proxy.stat().st_size > 1000 else path
            dur = float(media.probe(src).get("duration") or 0)
            ff, _ = media.tools()
            p = media.run([ff, "-v", "error", "-ss", f"{max(0.0, dur * 0.4):.2f}", "-i", src, "-frames:v", "1",
                           "-vf", f"scale={TILE_W}:-2", "-f", "image2pipe", "-vcodec", "mjpeg", "-"], check=False, timeout=60)
            if p.stdout:
                img = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        img = np.full((TILE_H, TILE_W, 3), 40, np.uint8)
    # letterbox into the tile
    h, w = img.shape[:2]
    s = min(TILE_W / w, TILE_H / h)
    r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((TILE_H, TILE_W, 3), np.uint8)
    y, x = (TILE_H - r.shape[0]) // 2, (TILE_W - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    cv2.imwrite(str(f), out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def sheet(tiles, labels, cols=5):
    """Numbered contact sheet as JPEG bytes."""
    import cv2
    rows = (len(tiles) + cols - 1) // cols
    W, H = cols * TILE_W, rows * TILE_H
    img = np.zeros((H, W, 3), np.uint8)
    for i, (t, lab) in enumerate(zip(tiles, labels)):
        y, x = (i // cols) * TILE_H, (i % cols) * TILE_W
        img[y:y + TILE_H, x:x + TILE_W] = t
        cv2.rectangle(img, (x, y), (x + TILE_W - 1, y + TILE_H - 1), (0, 0, 0), 2)
        (tw, th), _ = cv2.getTextSize(str(lab), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (x + 2, y + 2), (x + tw + 12, y + th + 14), (0, 0, 0), -1)
        cv2.putText(img, str(lab), (x + 7, y + th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return b.tobytes()


# ------------------------------------------------------------------ Google AI sorting (runs as a job)
def ai_sort(groups, event, hint, progress, cancelled):
    """groups: time groups (with paths and kinds). Returns {"events": [...], "files": {path: event_id}}."""
    from . import gemini
    kinds = {}
    for g in groups:
        for p in g["paths"]:
            kinds[p] = media.classify(Path(p))
    files = [p for g in groups for p in g["paths"] if kinds[p] in ("video", "photo")]
    num = {p: i + 1 for i, p in enumerate(files)}
    # 1. tiles for every picture
    tiles = {}
    done = [0]

    def one(p):
        tiles[p] = tile(p, kinds[p])
        done[0] += 1
        progress(0.45 * done[0] / max(1, len(files)), f"Looking at file {done[0]} of {len(files)}")
    with ThreadPoolExecutor(4) as ex:
        for _ in ex.map(one, files):
            if cancelled():
                raise InterruptedError
    # 2. overview: one sheet per time group, Google names the events
    progress(0.5, "Google AI is naming the events")
    overview = []
    for g in groups:
        ps = [p for p in g["paths"] if kinds[p] in ("video", "photo")]
        if not ps:
            continue
        pick = [ps[i] for i in np.linspace(0, len(ps) - 1, min(10, len(ps))).astype(int)]
        overview.append((g, sheet([tiles[p] for p in pick], [f"{g['id']}"] * len(pick))))
    events = gemini.sort_overview(overview, event, hint)
    if cancelled():
        raise InterruptedError
    # 3. every file against those events, 80 files per request
    batches = [files[i:i + 80] for i in range(0, len(files), 80)]
    assign = {}
    done[0] = 0

    def run(b):
        sheets = []
        for j in range(0, len(b), 20):
            part = b[j:j + 20]
            sheets.append(sheet([tiles[p] for p in part], [str(num[p]) for p in part]))
        nums = [num[p] for p in b]
        res = gemini.sort_files(sheets, nums, events, event, hint)
        done[0] += 1
        progress(0.55 + 0.45 * done[0] / len(batches), f"Checking every file ({done[0]} of {len(batches)})")
        return b, nums, res
    with ThreadPoolExecutor(3) as ex:
        for b, nums, res in ex.map(run, batches):
            for p, n in zip(b, nums):
                assign[p] = res.get(n, "unsure")
    # audio files follow the time group they were recorded in (majority event of that group)
    for g in groups:
        votes = {}
        for p in g["paths"]:
            if p in assign:
                votes[assign[p]] = votes.get(assign[p], 0) + 1
        top = max(votes, key=votes.get) if votes else "unsure"
        for p in g["paths"]:
            assign.setdefault(p, top)
    ids = {e["id"] for e in events}
    out_events = []
    for e in events + [{"id": "unsure", "label": "Not sure", "matches": "unsure", "why": "Google AI couldn't tell."}]:
        ps = [p for p, v in assign.items() if v == e["id"] or (e["id"] == "unsure" and v not in ids)]
        if ps:
            out_events.append({**e, "paths": ps, "samples": [p for p in ps if kinds[p] in ("video", "photo")][:4],
                               "counts": {k: sum(kinds[p] == k for p in ps) for k in ("video", "photo", "audio")}})
    return {"events": out_events}
