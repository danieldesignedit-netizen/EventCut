"""Speech transcription, reference Styles, Director mode (optional cloud AI) and the shoot guide."""
import base64
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from . import media, grade
from .config import HOME, load_settings

TRANS_CACHE = HOME / "Projects" / "_transcripts"
_whisper = None


# ------------------------------------------------------------------ transcription (local, free)
def whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def transcribe(path: str, key: str, progress=None) -> list:
    """Word-level transcript with faster-whisper on the GPU (CPU fallback). Cached per file."""
    TRANS_CACHE.mkdir(parents=True, exist_ok=True)
    out = TRANS_CACHE / f"{key}.json"
    if out.exists():
        return json.loads(out.read_text("utf-8"))
    if not whisper_available():
        return []
    global _whisper
    from faster_whisper import WhisperModel
    if _whisper is None:
        import sys
        device = "cuda" if media.nvidia_ok() else "cpu"
        bundled = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)) / "models" / "whisper"
        size = "medium" if device == "cuda" else "small"  # medium downloads once (about 1.5 GB) on GPU PCs
        root = str(bundled) if size == "small" and bundled.exists() else str(HOME / "Projects" / "_models")
        try:
            _whisper = WhisperModel(size, device=device, compute_type="float16" if device == "cuda" else "int8", download_root=root)
        except Exception:  # GPU libraries missing: fall back to the CPU
            _whisper = WhisperModel("small", device="cpu", compute_type="int8", download_root=str(bundled) if bundled.exists() else root)
    segments, info = _whisper.transcribe(path, word_timestamps=True, vad_filter=True)
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({"word": str(w.word), "start": round(float(w.start), 2), "end": round(float(w.end), 2),
                          "p": round(float(w.probability), 2)})
        if progress and info.duration:
            progress(min(1.0, float(seg.end) / float(info.duration)))
    out.write_text(json.dumps(words), "utf-8")
    return words


def save_transcript(key: str, words: list):
    """Edits from the transcript editor overwrite the cached transcript."""
    TRANS_CACHE.mkdir(parents=True, exist_ok=True)
    (TRANS_CACHE / f"{key}.json").write_text(json.dumps(words), "utf-8")


# ------------------------------------------------------------------ reference Styles
def sample_frames(path: str, n=12, size=(192, 108)):
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for i in np.linspace(total * 0.05, total * 0.95, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.resize(f, size))
    cap.release()
    return frames


def build_style(name: str, paths: list) -> dict:
    """Learn pacing, colour and loudness from up to 5 reference videos."""
    import cv2
    shots, lums, sats, frames, lufs = [], [], [], [], []
    for p in paths[:5]:
        info = media.probe(p)
        cuts = [0.0] + media.scene_cuts(p, 0.3) + [info["duration"]]
        shots += [b - a for a, b in zip(cuts, cuts[1:]) if b - a > 0.1]
        fr = sample_frames(p, 16)
        frames += fr
        for f in fr:
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            lums.append(hsv[..., 2].mean() / 255)
            sats.append(hsv[..., 1].mean() / 255)
        if info["has_audio"]:
            r = media.run([media.tools()[0], "-v", "info", "-i", p, "-af", "ebur128", "-f", "null", "-"], check=False)
            txt = (r.stderr or b"").decode("utf-8", "replace")
            try:
                lufs.append(float(txt[txt.rindex("I:"):].split()[1]))
            except Exception:
                pass
    img = np.concatenate(frames, 0) if frames else np.zeros((10, 10, 3), np.uint8)
    lab = grade.lab_stats(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255)
    shots = sorted(shots) or [1.0]
    style = {"name": name, "references": paths, "median_shot": round(float(np.median(shots)), 3),
             "p90_shot": round(float(shots[int(len(shots) * 0.9) - 1]), 3), "max_shot": round(float(max(shots)), 3),
             "luma": round(float(np.mean(lums)), 3), "saturation": round(float(np.mean(sats)), 3),
             "lab": lab, "lufs": round(float(np.mean(lufs)), 1) if lufs else -14.0, "created": time.strftime("%Y-%m-%d")}
    folder = HOME / "Styles"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(style, indent=1), "utf-8")
    return style


def list_styles() -> list:
    folder = HOME / "Styles"
    out = []
    for f in sorted(folder.glob("*.json")) if folder.exists() else []:
        try:
            s = json.loads(f.read_text("utf-8"))
            s.pop("lab", None)
            out.append(s)
        except Exception:
            pass
    return out


def load_style(name: str):
    f = HOME / "Styles" / f"{name}.json"
    return json.loads(f.read_text("utf-8")) if f.exists() else None


def frame_lab(path: str, t: float = None):
    """'Match this look' from a single reference frame or image."""
    import cv2
    if Path(path).suffix.lower() in media.PHOTO_EXT:
        img = cv2.imread(path)
    else:
        cap = cv2.VideoCapture(path)
        if t is not None:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        cap.release()
    img = cv2.resize(img, (320, int(320 * img.shape[0] / img.shape[1])))
    return grade.lab_stats(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255)


# ------------------------------------------------------------------ Director mode (optional, paid, cloud)
PRICE_PER_MTOK_IN = 3.0   # update to the provider's current price list
PRICE_PER_MTOK_OUT = 15.0


def director_candidates(analyses, limit=60):
    cands = []
    for a in analyses:
        for i, m in enumerate(a.get("moments", [])):
            if not m.get("reject"):
                cands.append((m["score"], a, i, m))
    cands.sort(key=lambda c: -c[0])
    return cands[:limit]


def director_estimate(analyses, transcripts) -> dict:
    n = len(director_candidates(analyses))
    words = sum(len(v) for v in transcripts.values())
    tok_in = n * 90 + int(words * 1.4) + 1500
    tok_out = 2500
    cost = tok_in / 1e6 * PRICE_PER_MTOK_IN + tok_out / 1e6 * PRICE_PER_MTOK_OUT
    return {"images": n, "input_tokens": tok_in, "estimated_usd": round(cost, 3),
            "data_notice": f"{n} small still frames (256 px) and the speech transcript will be sent to the AI provider. Full videos never leave this PC."}


def _thumb_b64(a, m):
    import cv2
    cap = cv2.VideoCapture(a.get("proxy") or a["path"])
    cap.set(cv2.CAP_PROP_POS_MSEC, (m["start"] + m["end"]) / 2 * 1000)
    ok, f = cap.read()
    cap.release()
    if not ok:
        return None
    f = cv2.resize(f, (256, int(256 * f.shape[0] / f.shape[1])))
    ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode()


def director_plan(analyses, transcripts, settings) -> dict:
    """Ask a top-tier vision model to pick and order moments like a senior editor.

    Returns {"picks": {(src, idx): {"boost": float, "section": str}}, "title": str, "notes": str}.
    """
    cfg = load_settings()
    key = cfg.get("director_mode_api_key")
    if not key:
        raise RuntimeError("Director mode needs an API key. Add it in Settings.")
    cands = director_candidates(analyses)
    content = [{"type": "text", "text": (
        "You are a senior event video editor with 30 years of experience cutting highlight films for top tech companies. "
        f"Event type: {settings.get('event_type')}. Target length: {settings.get('duration')} seconds. "
        f"Director's brief: {settings.get('prompt') or '(none: choose what makes the best story)'}.\n"
        "Below are candidate moments, each with an id, a still frame and what was said. Choose the moments for a story with "
        "a 3-second hook, a build, a peak, an emotional beat and a clean close. Favour genuine emotion, key speeches, "
        "reactions, inclusiveness (many different people), and avoid near-duplicates. "
        "Reply with JSON only: {\"title\": str, \"picks\": [{\"id\": int, \"section\": \"hook|build|peak|emotional|close\", \"why\": str}], \"notes\": str}")}]
    index = {}
    for k, (_, a, i, m) in enumerate(cands):
        said = " ".join(w["word"] for w in transcripts.get(a["path"], []) if m["start"] - 0.3 <= w["start"] <= m["end"] + 0.3).strip()
        b64 = _thumb_b64(a, m)
        if not b64:
            continue
        index[k] = (a["path"], i)
        content.append({"type": "text", "text": f"id {k} · {Path(a['path']).name} · {m['start']:.1f}s · faces {m.get('faces', 0)} · energy {m.get('energy', 0):.2f}" + (f" · said: \"{said[:200]}\"" if said else "")})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
    body = json.dumps({"model": cfg.get("director_mode_model", "claude-sonnet-5-5"), "max_tokens": 3000,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, method="POST", headers={
        "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.loads(r.read())
    text = "".join(c.get("text", "") for c in resp.get("content", []) if c.get("type") == "text")
    js = json.loads(text[text.index("{"): text.rindex("}") + 1])
    picks = {}
    n = max(1, len(js.get("picks", [])))
    for rank, p in enumerate(js.get("picks", [])):
        if p.get("id") in index:
            picks[index[p["id"]]] = {"boost": 1.0 - 0.5 * rank / n, "section": p.get("section", "peak"), "why": p.get("why", "")}
    usage = resp.get("usage", {})
    cost = usage.get("input_tokens", 0) / 1e6 * PRICE_PER_MTOK_IN + usage.get("output_tokens", 0) / 1e6 * PRICE_PER_MTOK_OUT
    log_spend(cost)
    return {"picks": picks, "title": js.get("title"), "notes": js.get("notes", ""), "cost_usd": round(cost, 4)}


def log_spend(usd):
    f = HOME / "Projects" / "director_spend.json"
    month = time.strftime("%Y-%m")
    data = json.loads(f.read_text()) if f.exists() else {}
    data[month] = round(data.get(month, 0) + usd, 4)
    f.write_text(json.dumps(data))


def month_spend():
    f = HOME / "Projects" / "director_spend.json"
    return json.loads(f.read_text()).get(time.strftime("%Y-%m"), 0) if f.exists() else 0


# ------------------------------------------------------------------ taste learning storage
def load_taste():
    f = HOME / "Projects" / "taste.json"
    return json.loads(f.read_text()) if f.exists() else {}


def save_taste(t):
    (HOME / "Projects" / "taste.json").write_text(json.dumps(t))
