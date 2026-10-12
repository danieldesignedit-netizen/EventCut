"""Podcast, interview and panel editing: complete soundbites, the speaker in frame, a hook and a cliffhanger.

How a senior editor cuts a short from a podcast, and what this module does:
- Only whole thoughts: every clip starts at the beginning of a sentence and ends at the end of one.
- Opens on the most gripping line (3-6 s), then the best moments in the order they were said.
- The person talking is on screen: vertical versions follow the speaker; when several people talk or laugh,
  all hosts are shown stacked.
- Ends on a payoff: the line that got the biggest laugh or reaction, so the video lands instead of trailing off.

Moments can come from Google AI (gemini.py) or, without it, from the transcript and the room's sound.
"""
import re

import numpy as np

from .analyze import FPS

WEAK_START = {"and", "so", "but", "because", "or", "um", "uh", "like", "yeah", "okay", "ok", "then", "also", "which",
              "that", "cause", "'cause", "mm", "hmm", "erm", "right"}
FILLERS = {"um", "uh", "erm", "hmm", "mm", "like", "you know", "i mean"}
XF = {"zoom_in": 0.4, "blurwhip": 0.3, "flash": 0.2, "dip": 0.4, "dissolve": 0.5}


# ------------------------------------------------------------------ sentences
def sentences(words, max_len=22.0):
    """Group words into sentences: break after . ? ! or a long pause. Very long run-ons split at commas."""
    out, cur = [], []

    def flush():
        if cur:
            out.append({"start": cur[0]["start"], "end": cur[-1]["end"], "text": " ".join(w["word"].strip() for w in cur).strip(),
                        "n": len(cur)})
    for i, w in enumerate(words):
        if cur and (w["start"] - cur[-1]["end"] > 0.9 or (cur[-1]["word"].strip()[-1:] in ".?!" ) or
                    (w["start"] - cur[0]["start"] > max_len and cur[-1]["word"].strip()[-1:] in ",;:")):
            flush()
            cur = []
        cur.append(w)
    flush()
    return [s for s in out if s["end"] - s["start"] >= 0.6]


def pause_sentences(a, min_gap=0.45):
    """No transcript: treat stretches of speech between pauses as 'sentences' (from the room's sound)."""
    env = np.array(a.get("audio_db") or [])
    if not len(env):
        return []
    thr = np.percentile(env, 35)
    speech = env > thr
    k = max(1, int(min_gap * FPS))
    out, start, quiet = [], None, 0
    for i, v in enumerate(speech):
        if v:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= k:
                end = i - quiet + 1
                if (end - start) / FPS >= 1.0:
                    out.append({"start": start / FPS, "end": end / FPS, "text": "", "n": int((end - start) / FPS * 2.5)})
                start, quiet = None, 0
    if start is not None:
        out.append({"start": start / FPS, "end": len(env) / FPS, "text": "", "n": 0})
    return out


def snap(t, sents, edge, tol=1.6):
    """Move a time from the AI (roughly right) onto the nearest real sentence start or end."""
    key = "start" if edge == "start" else "end"
    best = min(sents, key=lambda s: abs(s[key] - t), default=None)
    return best[key] if best and abs(best[key] - t) <= tol else t


# ------------------------------------------------------------------ who is where, who is talking
def seats(a, t0, t1):
    """Where each person sits during t0..t1: clusters of face positions, left to right."""
    frames = a.get("frames", [])
    fr = frames[int(t0 * FPS): max(int(t0 * FPS) + 1, int(t1 * FPS))]
    pts = [b for f in fr for b in (f.get("boxes") or [])]
    with_faces = sum(1 for f in fr if f.get("boxes")) or 1
    if not pts:
        return []
    pts.sort(key=lambda b: b[0])
    gap = max(0.05, float(np.median([b[2] for b in pts])) * 0.9)
    groups, cur = [], [pts[0]]
    for b in pts[1:]:
        if b[0] - cur[-1][0] > gap:
            groups.append(cur)
            cur = []
        cur.append(b)
    groups.append(cur)
    out = []
    for g in groups:
        if len(g) < with_faces * 0.25:
            continue  # someone walking past, a poster, a false detection
        arr = np.array(g)
        out.append({"cx": float(np.median(arr[:, 0])), "cy": float(np.median(arr[:, 1])),
                    "w": float(np.median(arr[:, 2])), "h": float(np.median(arr[:, 3]))})
    return out


def _smooth(x, n):
    if n <= 1 or len(x) < 2:
        return x
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def _activity(a, i0, i1, st):
    """Mouth movement per person per frame, each against their own normal level."""
    frames = a.get("frames", [])
    fr = frames[i0:i1]
    act = np.full((len(fr), len(st)), np.nan)
    for i, f in enumerate(fr):
        for b in f.get("boxes") or []:
            j = int(np.argmin([abs(b[0] - s["cx"]) for s in st]))
            if abs(b[0] - st[j]["cx"]) < st[j]["w"] * 1.2 and len(b) > 4:
                act[i, j] = b[4] if np.isnan(act[i, j]) else max(act[i, j], b[4])
    allv = []
    for f in frames:  # the person's normal level over the whole clip
        allv.append([max([b[4] for b in (f.get("boxes") or []) if len(b) > 4 and abs(b[0] - s["cx"]) < s["w"] * 1.2] or [np.nan]) for s in st])
    base = np.nanmedian(np.array(allv, dtype=float), 0) if allv else np.ones(len(st))
    base = np.where(np.isfinite(base) & (base > 0), base, 1.0)
    return act / base


def speaker_runs(a, t0, t1, st, sents=None, min_hold=1.4):
    """Who is talking, as runs [(start, end, seat index or -1 for 'several / unclear')].

    Decided once per sentence (one sentence = one voice), from mouth movement averaged over the sentence."""
    i0, i1 = int(t0 * FPS), max(int(t0 * FPS) + 1, int(t1 * FPS))
    if len(st) < 2:
        return [(t0, t1, 0 if st else -1)]
    act = _activity(a, i0, i1, st)
    if not len(act):
        return [(t0, t1, -1)]
    # sentence boundaries inside this stretch (else ~2.5 s windows)
    parts = [(max(t0, s["start"]), min(t1, s["end"])) for s in (sents or []) if s["end"] > t0 and s["start"] < t1]
    if not parts:
        n = max(1, int(round((t1 - t0) / 2.5)))
        edges = np.linspace(t0, t1, n + 1)
        parts = list(zip(edges[:-1], edges[1:]))
    labels = []
    for p0, p1 in parts:
        seg = act[max(0, int((p0 - t0) * FPS)): max(1, int((p1 - t0) * FPS))]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a person not visible in this sentence: empty column
            m = np.nanmean(seg, 0) if len(seg) and np.isfinite(seg).any() else np.zeros(len(st))
        m = np.nan_to_num(m)
        order = np.argsort(m)
        lead, second = m[order[-1]], m[order[-2]]
        labels.append((p0, p1, int(order[-1]) if lead > 1.05 and lead > second * 1.18 else -1))
    # runs: fill the gaps between sentences, merge equal neighbours, no flashes shorter than min_hold
    runs = []
    for k, (p0, p1, sp) in enumerate(labels):
        start = t0 if k == 0 else runs[-1][1]
        end = t1 if k == len(labels) - 1 else (p1 + labels[k + 1][0]) / 2
        if runs and runs[-1][2] == sp:
            runs[-1][1] = end
        else:
            runs.append([start, end, sp])
    merged = []
    for r in runs:
        if merged and r[1] - r[0] < min_hold:
            merged[-1][1] = r[1]
        elif merged and merged[-1][1] - merged[-1][0] < min_hold:
            merged[-1][1], merged[-1][2] = r[1], r[2]
        else:
            merged.append(r)
    return [tuple(r) for r in merged]


# ------------------------------------------------------------------ scoring soundbites without AI
def _loud(a):
    env = np.array(a.get("audio_db") or [])
    if not len(env):
        return env
    base = np.percentile(env, 30)
    return np.clip((env - base) / 18.0, 0, 1)


def local_bites(a, sents, terms, max_len):
    """Candidate soundbites (1-4 whole sentences) scored on liveliness, reactions and the brief."""
    loud = _loud(a)
    out = []
    for i in range(len(sents)):
        for k in range(1, 5):
            if i + k > len(sents):
                break
            grp = sents[i:i + k]
            s, e = grp[0]["start"], grp[-1]["end"]
            d = e - s
            if d < 3.0:
                continue
            if d > max_len:
                break
            text = " ".join(g["text"] for g in grp).strip()
            first = re.sub(r"[^a-z']", "", (text.split() or [""])[0].lower())
            seg = loud[int(s * FPS): int(e * FPS)] if len(loud) else np.zeros(1)
            after = loud[int(e * FPS): int((e + 1.8) * FPS)] if len(loud) else np.zeros(1)
            wps = sum(g["n"] for g in grp) / max(0.5, d)
            words = text.lower().split()
            filler = sum(1 for w in words if w.strip(",.?!") in FILLERS) / max(1, len(words))
            score = (0.30 * float(seg.mean() if len(seg) else 0) + 0.22 * min(1.0, wps / 3.0)
                     + 0.22 * float(after.max() if len(after) else 0)  # laughter or reaction right after
                     + 0.12 * (1 if re.search(r"[?!]", text) else 0)
                     + 0.10 * min(1.0, d / 9.0))
            if first in WEAK_START:
                score -= 0.25  # starts mid-thought ("and so...")
            score -= 0.6 * filler
            if terms and text:
                score += 0.35 * sum(1 for t in terms if t in text.lower())
            out.append({"start": s, "end": e, "score": score, "quote": text, "question": text.rstrip().endswith("?"),
                        "position": None, "hook": 0.0, "payoff": float(after.max() if len(after) else 0)})
    return out


# ------------------------------------------------------------------ the plan
def _layout_runs(a, s, e, bite):
    """Split a soundbite into shots with framings: for tall outputs (9:16, 4:5, 1:1) the speaker close-up or all
    hosts stacked; for 16:9 the wide shot with punch-ins on the speaker. Returns [(start, end, {"v": .., "h": ..})]."""
    st = seats(a, s, e)
    if not st:
        return [(s, e, {"v": None, "h": None})]
    runs = speaker_runs(a, s, e, st, bite.get("sents"))
    pos = bite.get("position")  # the AI's 'left / centre / right' for the person speaking
    if pos and len(st) >= 2:
        idx = {"left": 0, "center": len(st) // 2, "centre": len(st) // 2, "middle": len(st) // 2, "right": len(st) - 1}.get(str(pos).lower())
        if idx is not None:
            runs = [(r0, r1, idx if sp == -1 else sp) for r0, r1, sp in runs]
    boxes = [[x["cx"], x["cy"], x["w"], x["h"]] for x in st]
    out = []
    for k, (r0, r1, sp) in enumerate(runs):
        if len(st) == 1:
            v = {"mode": "speaker", "box": boxes[0]}
        elif sp >= 0:
            v = {"mode": "speaker", "box": boxes[sp]}
        elif len(st) <= 3:
            v = {"mode": "stack", "boxes": boxes}
        else:
            v = {"mode": "wide"}
        h = {"mode": "punch", "box": boxes[sp]} if sp >= 0 and len(st) >= 2 and k % 2 == 1 else None
        out.append((r0, r1, {"v": v, "h": h}))
    # open a longer bite on everyone (stacked) for a beat, so viewers see who is in the room
    if len(st) >= 2 and len(st) <= 3 and bite.get("establish") and out and out[0][1] - out[0][0] > 3.2:
        r0, r1, lay = out[0]
        out[0:1] = [(r0, r0 + 1.4, {"v": {"mode": "stack", "boxes": boxes}, "h": None}), (r0 + 1.4, r1, lay)]
    return out


def plan_talk(analyses, settings, grid, transcripts=None, ai=None, rng=None, source_order=None):
    """Shot list for a podcast/interview short. ai: {path: {"bites": [...], "selection": [...]}} from gemini.py."""
    transcripts, ai = transcripts or {}, ai or {}
    rng = rng or np.random.default_rng(int(settings.get("seed", 0)))
    duration = float(settings.get("duration", 60))
    ending = "none" if settings.get("ending") == "none" or settings.get("end_card", True) is False else "card"
    end_len = 0.0 if ending == "none" else min(2.5, max(1.5, duration * 0.06))
    body = duration - end_len
    terms = [t for t in re.findall(r"[a-z][a-z'-]{2,}", (settings.get("prompt") or "").lower())
             if t not in {"the", "and", "with", "focus", "make", "video", "about", "this", "that", "from", "want", "please"}]
    max_bite = float(np.clip(body * 0.45, 6.0, 20.0))

    by_path = {a["path"]: a for a in analyses if a.get("kind") == "video"}
    sents = {p: (sentences(transcripts[p]) if transcripts.get(p) else pause_sentences(a)) for p, a in by_path.items()}

    # candidate soundbites: the AI's picks (snapped to real sentence edges) or local scoring
    cands = []
    for p, a in by_path.items():
        ss = sents[p]
        if ai.get(p, {}).get("bites"):
            for b in ai[p]["bites"]:
                s0, e0 = snap(float(b["start"]), ss, "start"), snap(float(b["end"]), ss, "end")
                if e0 - s0 < 2.0:
                    continue
                cands.append({"src": p, "start": s0, "end": e0, "score": float(b.get("score", 5)) / 10 + 0.15 * float(b.get("hook", 0)) / 10,
                              "quote": b.get("quote", ""), "question": str(b.get("quote", "")).rstrip().endswith("?"),
                              "position": b.get("position"), "hook": float(b.get("hook", 0)) / 10,
                              "payoff": float(b.get("payoff", 0)) / 10, "ai": True})
        else:
            for c in local_bites(a, ss, terms, max_bite):
                c["src"] = p
                cands.append(c)
    if not cands:
        raise ValueError("No clear speech found in this footage. Turn on Captions or Google AI, or choose a non-podcast event type.")
    seed = int(settings.get("seed", 0))
    for c in cands:  # regenerate = a different selection of equally good moments
        c["rank"] = c["score"] + (rng.gumbel() * 0.06 if seed else 0)

    def overlaps(c, picked, pad=1.5):
        return any(o["src"] == c["src"] and c["start"] < o["end"] + pad and o["start"] < c["end"] + pad for o in picked)

    # 1. hook: the single most gripping complete line, 3-6 s (up to 8 s if nothing shorter is strong)
    hook = None
    if settings.get("hook", True) is not False:
        pool = [c for c in cands if 2.5 <= c["end"] - c["start"] <= 6.5] or [c for c in cands if c["end"] - c["start"] <= 8.5]
        if pool:
            hook = max(pool, key=lambda c: c["rank"] + 0.6 * c["hook"] + (0.1 if not c["question"] else 0))
    picked = [hook] if hook else []

    # 2. the closer: a payoff (biggest laugh or reaction right after, or the AI's pick), so the video lands
    cliff = None
    closers = [c for c in cands if not overlaps(c, picked) and c["end"] - c["start"] <= 12 and not c["question"]]
    if closers:
        cliff = max(closers, key=lambda c: c["rank"] + 0.5 * c.get("payoff", 0))
        picked.append(cliff)

    # 3. the story: best remaining bites until the length is filled, then in the order they were said
    used = sum(c["end"] - c["start"] for c in picked)
    story = []
    for c in sorted(cands, key=lambda c: -c["rank"]):
        if used >= body * 0.97:
            break
        d = c["end"] - c["start"]
        if d > max_bite or overlaps(c, picked + story):
            continue
        if used + d > body * 1.06:
            continue  # too long for what's left: a shorter bite may fit
        story.append(c)
        used += d
    order = {p: i for i, p in enumerate(source_order or list(by_path))}
    story.sort(key=lambda c: (order.get(c["src"], 0), c["start"]))

    # 4. shots, with framing per speaker
    shots = []
    seq = ([("hook", hook)] if hook else []) + [("story", c) for c in story] + ([("close", cliff)] if cliff else [])
    for n, (section, c) in enumerate(seq):
        a = by_path[c["src"]]
        info = a.get("info", {})
        pre = 0.12  # a breath before the first word, a beat after the last
        tail = 0.9 if section == "close" else 0.3  # let the closing laugh or reaction ring out
        s, e = max(0.0, c["start"] - pre), min(float(info.get("duration") or c["end"] + 1), c["end"] + tail)
        trans = "cut"
        if n > 0:
            if section == "story" and seq[n - 1][0] == "hook":
                trans = "zoom_in"
            elif section == "close":
                trans = "cut"
            elif rng.random() < 0.25:
                trans = str(rng.choice(["blurwhip", "zoom_in"]))
        if trans in XF:
            s = max(0.0, s - XF[trans])  # the transition overlaps the start, so start that much earlier: no words lost
        c["establish"] = section == "story" and n == (1 if hook else 0)
        c["sents"] = [x for x in sents[c["src"]] if x["end"] > s and x["start"] < e]
        for k, (r0, r1, lay) in enumerate(_layout_runs(a, s, e, c)):
            shots.append({"src": c["src"], "kind": "video", "in": round(r0, 3), "out": round(r1, 3), "length": round(r1 - r0, 4),
                          "section": section, "face_x": 0.5, "fps": info.get("fps", 25), "log": (a.get("log") or {}).get("profile"),
                          "layouts": lay, "transition": trans if k == 0 else "cut", "quote": c.get("quote", "") if k == 0 else "",
                          "score": round(c["score"], 3), "bite": n})
    from .story import tame_moves
    tame_moves(shots, "celebration", rng, settings.get("transition_amount", "less"))
    total = sum(s["length"] for s in shots)
    warning, shortened = None, False
    if total < body * 0.8:
        shortened = True
        warning = (f"There were only enough complete, strong moments for about {int(total + end_len)} seconds, "
                   "so the video is shorter rather than padded with weak lines.")
    first_story = next((i for i, s in enumerate(shots) if s["section"] == "story"), 0)
    return {"template": "talk", "order": "talk", "bpm": grid["bpm"], "shots": shots, "body": round(total, 3),
            "sections": [{"name": "hook", "start": 0, "end": round(sum(s["length"] for s in shots[:first_story]), 3)}],
            "end_card": round(max(end_len, duration - total) if not shortened else end_len, 3), "ending": ending,
            "drop_at": round(sum(s["length"] for s in shots[:first_story]), 3), "mood": settings.get("music_mood") or "chill",
            "leaks": [], "warning": warning, "shortened": shortened,
            "bites": [{"section": sec, "src": c["src"], "start": round(c["start"], 2), "end": round(c["end"], 2), "quote": c.get("quote", "")}
                      for sec, c in seq]}
