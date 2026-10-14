"""Statements for the video: Google AI hears the whole event and names the best complete sentences; EventCut then
listens closely to just those few seconds (fal speech-to-text) and moves the edges to the exact first and last word,
so a statement never starts or stops mid-word."""
import hashlib
import json
import os
import re
from pathlib import Path

from . import falai, media
from .config import WORK

CACHE = WORK / "_ai" / "bites"
END = re.compile(r"[.!?…]['\")\]]*$")


def _excerpt(src, t0, t1, out):
    ff, _ = media.tools()
    media.run([ff, "-y", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
               "-c:a", "pcm_s16le", str(out)])
    return out


def snap(st, words):
    """st: {start, end, ...}; words: [{word, start, end}] in clip time. Returns the statement on whole sentences."""
    s0, e0 = float(st["start"]), float(st["end"])
    ws = [w for w in words if s0 - 1.5 <= w["start"] <= e0 + 1.5]
    if len(ws) < 3:
        return None  # nothing clearly said here
    allw = words

    def ends(i):
        w = allw[i]
        nxt = allw[i + 1] if i + 1 < len(allw) else None
        return bool(END.search(w["word"].strip())) or nxt is None or nxt["start"] - w["end"] > 0.7

    def begins(i):
        return i == 0 or ends(i - 1)
    idx = [allw.index(w) for w in ws]
    starts = [i for i in idx if begins(i)] or [idx[0]]
    i0 = min(starts, key=lambda i: abs(allw[i]["start"] - s0))
    stops = [i for i in idx if i >= i0 and ends(i)] or [idx[-1]]
    j0 = min(stops, key=lambda j: abs(allw[j]["end"] - e0))
    # too long: end at an earlier sentence end (at least 3 s in)
    if allw[j0]["end"] - allw[i0]["start"] > 9.5:
        shorter = [j for j in stops if 3.0 <= allw[j]["end"] - allw[i0]["start"] <= 9.5]
        if not shorter:
            return None
        j0 = max(shorter)
    if allw[j0]["end"] - allw[i0]["start"] < 1.8:
        return None
    quote = " ".join(w["word"].strip() for w in allw[i0:j0 + 1])
    return dict(st, start=round(max(0.0, allw[i0]["start"] - 0.15), 2), end=round(allw[j0]["end"] + 0.3, 2), quote=quote[:300], snapped=True,
                words=[{"word": w["word"], "start": round(w["start"], 3), "end": round(w["end"], 3)} for w in allw[i0:j0 + 1]])


def refine(statements, sources, progress=None, cancelled=None, limit=6):
    """Snap the best statements to whole sentences. sources: {clip path: file to take the sound from (the skim)}.
    Without fal, Google AI's own times are used with a little room either side."""
    best = sorted(statements or [], key=lambda x: -float(x.get("score", 0)))[:limit]
    if not best:
        return []
    use_fal = falai.enabled()
    CACHE.mkdir(parents=True, exist_ok=True)
    done = [0]

    def one(st):
        if cancelled and cancelled():
            raise InterruptedError
        if not use_fal:
            return dict(st, start=max(0.0, float(st["start"]) - 0.3), end=float(st["end"]) + 0.4, snapped=False)
        src = sources.get(st["path"]) or st["path"]
        t0 = max(0.0, float(st["start"]) - 3.0)
        t1 = float(st["end"]) + 3.0
        tag = hashlib.sha1(f"{st['path']}|{t0:.2f}|{t1:.2f}".encode()).hexdigest()[:16]
        cf = CACHE / f"{tag}.json"
        try:
            if cf.exists():
                words = json.loads(cf.read_text("utf-8"))
            else:
                wav = _excerpt(src, t0, t1, CACHE / f"{tag}.wav")
                try:
                    words = falai.transcribe(str(wav), st.get("quote", ""), cancelled=cancelled)
                finally:
                    try:
                        os.remove(wav)
                    except OSError:
                        pass
                cf.write_text(json.dumps(words), "utf-8")
        except InterruptedError:
            raise
        except Exception as e:  # can't listen closely: keep Google AI's times
            print("statement not refined:", e)
            return dict(st, start=max(0.0, float(st["start"]) - 0.3), end=float(st["end"]) + 0.4, snapped=False)
        finally:
            done[0] += 1
            if progress:
                progress(done[0] / len(best), f"{done[0]} of {len(best)} statements heard")
        words = [dict(w, start=w["start"] + t0, end=w["end"] + t0) for w in words]
        return snap(st, words)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(6) as ex:  # all at once: a few seconds in total
        out = [r for r in ex.map(one, best) if r]
    if progress:
        progress(1.0, "Statements ready")
    return out
