"""Separately recorded audio (a recorder, lapel mic or mixing desk) matched to camera clips by sound.

Every camera clip with sound is compared with every recording: the loudness pattern over time (claps, laughter,
speech rhythm) lines up at exactly one offset when they captured the same moment. Matches are only used when
they are clearly unambiguous. The matched recording then replaces the camera's microphone in the edit,
with a little camera sound kept underneath for the room.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import media
from .config import HOME, WORK

RATE = 100  # loudness samples per second
CACHE = WORK / "_sync"


def _env(path, cache_tag):
    """Onset-style loudness curve: rises in energy (claps, syllables, laughs) stand out, steady noise doesn't."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"env_{cache_tag}.npy"
    if f.exists():
        return np.load(f)
    ff, _ = media.tools()
    p = media.run([ff, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "8000", "-f", "f32le", "-"])
    x = np.frombuffer(p.stdout, np.float32)
    hop = 8000 // RATE
    n = len(x) // hop
    if n < RATE * 2:
        e = np.zeros(0, np.float32)
    else:
        e = np.log1p(np.abs(x[: n * hop]).reshape(n, hop).mean(1) * 1000)
        d = np.diff(e, prepend=e[0])
        d = np.maximum(d, 0)
        k = RATE  # remove slow changes (1 s)
        c = np.cumsum(np.concatenate([[0], d]))
        avg = (c[np.clip(np.arange(n) + k // 2, 0, n)] - c[np.clip(np.arange(n) - k // 2, 0, n)]) / k
        e = (d - avg).astype(np.float32)
    np.save(f, e)
    return e


def _key(path):
    st = Path(path).stat()
    return hashlib.sha1(f"{path}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]


def match(videos, recordings, progress=None, sources=None):
    """{video_path: {"audio": recording, "offset": seconds into the recording where the clip starts, "score": z}}"""
    if not videos or not recordings:
        return {}
    tag = hashlib.sha1(json.dumps(sorted(_key(p) for p in list(videos) + list(recordings))).encode()).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    out_f = CACHE / f"match_{tag}.json"
    if out_f.exists():
        return json.loads(out_f.read_text("utf-8"))
    recs = []
    for r in recordings:
        e = _env(r, _key(r))
        if len(e):
            e = (e - e.mean()) / (e.std() + 1e-6)
            recs.append((r, e))
    result = {}
    for i, v in enumerate(videos):
        if progress:
            progress(i / max(1, len(videos)))
        ve = _env((sources or {}).get(v) or v, _key(v))  # the skim's sound when there is one (no need to read the 4K file again)
        if len(ve) < RATE * 3:
            continue
        ve = (ve - ve.mean()) / (ve.std() + 1e-6)
        best = None
        for r, re in recs:
            if len(re) < len(ve):
                continue
            n = 1 << int(np.ceil(np.log2(len(re) + len(ve))))
            corr = np.fft.irfft(np.fft.rfft(re, n) * np.conj(np.fft.rfft(ve, n)), n)[: len(re) - len(ve) + 1] / len(ve)
            k = int(np.argmax(corr))
            peak = float(corr[k])
            # how far the best match stands out from every other possible position
            mask = np.ones(len(corr), bool)
            mask[max(0, k - RATE): k + RATE] = False
            rest = corr[mask]
            z = (peak - float(rest.mean())) / (float(rest.std()) + 1e-6) if len(rest) > 10 else 0
            second = float(rest.max()) if len(rest) else 0
            if z > 6 and peak > 0.08 and peak > second * 1.5 and (best is None or z > best["score"]):
                frac = 0.0  # sub-sample peak position (parabola through the three best points)
                if 0 < k < len(corr) - 1:
                    a_, b_, c_ = corr[k - 1], corr[k], corr[k + 1]
                    den = a_ - 2 * b_ + c_
                    frac = float(0.5 * (a_ - c_) / den) if den < 0 else 0.0
                best = {"audio": r, "offset": round((k + max(-0.5, min(0.5, frac))) / RATE, 3), "score": round(z, 1)}
        if best:
            result[v] = best
    out_f.write_text(json.dumps(result), "utf-8")
    return result
