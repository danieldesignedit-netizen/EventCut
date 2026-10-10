"""Edit planning: turns analysed moments into a beat-synced, story-shaped shot list.

The plan is pure data (JSON), so the review timeline can show it, lock clips, swap clips and
Director mode can replace it with an AI-written plan before rendering.
"""
import math
import re

import numpy as np

# Section: share of duration, wanted energy, wanted faces, shot length in beats, montage burst chance
TEMPLATES = {
    "celebration": [
        ("hook", 0.10, 0.9, 0.6, 1.5, 0.3),
        ("build", 0.27, 0.45, 0.5, 2.0, 0.0),
        ("peak", 0.35, 1.0, 0.5, 1.0, 0.45),
        ("emotional", 0.16, 0.35, 0.9, 3.0, 0.0),
        ("close", 0.12, 0.6, 0.8, 2.0, 0.0),
    ],
    "corporate": [
        ("hook", 0.10, 0.7, 0.7, 1.5, 0.2),
        ("build", 0.30, 0.4, 0.6, 2.5, 0.0),
        ("peak", 0.30, 0.8, 0.6, 1.5, 0.25),
        ("emotional", 0.18, 0.3, 0.9, 3.5, 0.0),
        ("close", 0.12, 0.5, 0.8, 2.5, 0.0),
    ],
    "talk": [  # podcast, vox pop, interview, panel: longer holds on people speaking
        ("hook", 0.15, 0.6, 1.0, 4.0, 0.0),
        ("build", 0.35, 0.4, 1.0, 6.0, 0.0),
        ("peak", 0.30, 0.6, 1.0, 5.0, 0.0),
        ("close", 0.20, 0.4, 1.0, 5.0, 0.0),
    ],
}

EVENT_TYPES = {
    "Watch party": "celebration", "Team bonding": "celebration", "Anniversary": "celebration",
    "End-of-year party": "celebration", "CSR": "corporate", "Product launch": "celebration",
    "Conference": "corporate", "Award night": "celebration", "Podcast": "talk", "Vox pop": "talk",
    "Interview": "talk", "Panel": "talk",
}

TALK_WORDS = ("podcast", "vox", "interview", "panel", "talk", "speech", "testimon", "q&a", "conversation")
CORP_WORDS = ("csr", "conference", "summit", "seminar", "training", "townhall", "town hall", "workshop", "outreach", "charity")


def template_for(event_type: str) -> str:
    if event_type in EVENT_TYPES:
        return EVENT_TYPES[event_type]
    t = (event_type or "").lower()
    if any(w in t for w in TALK_WORDS):
        return "talk"
    if any(w in t for w in CORP_WORDS):
        return "corporate"
    return "celebration"


def prompt_terms(prompt: str):
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", (prompt or "").lower())
    stop = {"the", "and", "with", "focus", "end", "start", "make", "video", "show", "more", "less", "about", "that",
            "this", "from", "into", "mood", "upbeat", "should", "want", "include", "please", "our", "for", "then"}
    return [w for w in words if w not in stop]


def mood_from_prompt(prompt: str, template: str) -> str:
    p = (prompt or "").lower()
    if any(w in p for w in ("emotional", "touching", "sentimental", "heartfelt", "farewell", "tribute")):
        return "Emotional Piano"
    if any(w in p for w in ("cinematic", "epic", "dramatic", "night")):
        return "Cinematic Night"
    if any(w in p for w in ("corporate", "professional", "clean")) or template == "corporate":
        return "Corporate Bright"
    if any(w in p for w in ("chill", "relaxed", "calm", "soft")) or template == "talk":
        return "Chill Celebration"
    return "Upbeat Afro House"


def _moment_pool(analyses, terms, transcripts, taste, rng, randomness, director=None):
    director = director or {}
    pool = []
    for a in analyses:
        name = a["path"].lower()
        words = transcripts.get(a["path"], [])
        for i, m in enumerate(a.get("moments", [])):
            if m.get("reject"):
                continue
            s = m["score"]
            hit = 0
            if terms:
                hit += sum(1 for t in terms if t in name) * 0.3
                said = " ".join(w["word"].lower() for w in words if m["start"] - 0.5 <= w["start"] <= m["end"] + 0.5)
                hit += sum(1 for t in terms if t in said) * 0.6
            s += min(0.6, hit)
            if taste:
                s += taste.get("faces", 0) * min(1, m.get("faces", 0) / 2) + taste.get("energy", 0) * m.get("energy", 0) \
                     + taste.get("motion", 0) * min(1, m.get("motion", 0) * 12) + taste.get("photo", 0) * (1 if m.get("photo") else 0)
                s += taste.get("sources", {}).get(a["path"], 0)
            s += rng.gumbel() * randomness  # regenerate = different picks
            d = director.get(f"{a['path']}|{i}")
            if d:
                s += d["boost"]
            pool.append({"director_section": d["section"] if d else None,"src": a["path"], "kind": a["kind"], "idx": i, "m": m, "score": s, "prompt_hit": hit,
                         "fps": a.get("info", {}).get("fps", 25), "w": a["info"].get("width"), "h": a["info"].get("height"),
                         "log": (a.get("log") or {}).get("profile")})
    return pool


def plan_edit(analyses, settings, grid, transcripts=None, taste=None):
    """Return {sections, shots, total} where every shot length is a whole number of beats."""
    transcripts = transcripts or {}
    duration = float(settings.get("duration", 60))
    end_card = min(2.5, max(1.5, duration * 0.06))
    body = duration - end_card
    tmpl_name = template_for(settings.get("event_type", ""))
    tmpl = TEMPLATES[tmpl_name]
    style = settings.get("style") or {}
    seed = int(settings.get("seed", 0))
    rng = np.random.default_rng(seed)
    terms = prompt_terms(settings.get("prompt", ""))
    pool = _moment_pool(analyses, terms, transcripts, taste, rng, randomness=0.12 if seed else 0.02,
                        director=settings.get("director_picks"))
    if not pool:
        raise ValueError("No usable shots found. The footage may be too dark, blurry or shaky.")

    beat = 60.0 / grid["bpm"]
    # Style pacing: scale shot lengths toward the reference median shot length
    pace = 1.0
    if style.get("median_shot"):
        pace = float(np.clip(style["median_shot"] / 0.55, 0.5, 3.0))
    if tmpl_name == "talk":
        pace = max(pace, 1.0)

    # section boundaries snapped to bars
    bar = 4 * beat
    bounds, t = [], 0.0
    for name, share, energy, faces, beats_per_shot, burst in tmpl:
        length = max(bar, round(body * share / bar) * bar)
        bounds.append([name, t, t + length, energy, faces, beats_per_shot, burst])
        t += length
    bounds[-1][2] = body  # last section absorbs rounding

    used, used_src_time, used_order = set(), [], []
    locks = {(l["src"], round(l["start"], 2)): l for l in settings.get("locks", [])}
    shots = []
    last_src = None
    # place locked clips first into their recorded sections
    locked_by_section = {}
    for l in locks.values():
        locked_by_section.setdefault(l.get("section", "peak"), []).append(l)

    for name, s0, s1, w_energy, w_faces, bps, burst in bounds:
        t = s0
        section_shots = []
        pending_locks = list(locked_by_section.get(name, []))
        while t < s1 - beat * 0.5:
            nb = bps * pace
            if burst and rng.random() < burst:
                nb = 0.5 if name == "peak" else 1.0  # rapid-fire montage
            nb = max(0.5 if name == "peak" else 1.0, round(nb * 2) / 2)
            length = min(s1 - t, nb * beat)
            if length < beat * 0.45:
                break
            if pending_locks:
                l = pending_locks.pop(0)
                need = min(length, l["end"] - l["start"])
                section_shots.append({"src": l["src"], "kind": l.get("kind", "video"), "in": l["start"], "out": l["start"] + need,
                                      "length": need, "section": name, "face_x": l.get("face_x", 0.5), "locked": True,
                                      "fps": l.get("fps", 25), "log": l.get("log")})
                used.add((l["src"], l.get("idx", -1)))
                t += need
                continue
            best, best_s = None, -1e9
            for c in pool:
                key = (c["src"], c["idx"])
                if key in used:
                    continue
                m = c["m"]
                avail = m["end"] - m["start"]
                if c["kind"] != "photo" and avail < length * 0.98:
                    continue
                fit = c["score"] - 0.35 * abs(m.get("energy", 0) - w_energy) - 0.25 * abs(min(1, m.get("faces", 0) / 2) - w_faces)
                if c.get("director_section"):
                    fit += 0.45 if c["director_section"] == name else -0.2
                if name == "emotional":
                    fit -= 0.3 * min(1, m.get("motion", 0) * 15)
                if c["kind"] == "photo":
                    fit -= 0.08 if name in ("hook", "peak") else -0.04
                if c["src"] == last_src:
                    fit -= 0.18
                near = sum(1 for (src, tt) in used_src_time if src == c["src"] and abs(tt - m["start"]) < 6)
                fit -= 0.12 * near
                if fit > best_s:
                    best, best_s = c, fit
            if best is None:
                # nothing long enough: take the best unused moment and shorten the shot to fit it
                unused = [c for c in pool if (c["src"], c["idx"]) not in used and c["src"] != last_src] or \
                         [c for c in pool if (c["src"], c["idx"]) not in used]
                if unused:
                    best = max(unused, key=lambda c: c["score"] + 0.2 * min(4, c["m"]["end"] - c["m"]["start"]))
                else:  # every moment used once: reuse the one used longest ago
                    order = {k: i for i, k in enumerate(used_order)}
                    best = min((c for c in pool if c["src"] != last_src) or pool, key=lambda c: order.get((c["src"], c["idx"]), -1))
                if best["kind"] != "photo":
                    avail = best["m"]["end"] - best["m"]["start"]
                    length = min(length, max(beat * 0.5, math.floor(avail / (beat * 0.5)) * beat * 0.5))
            m = best["m"]
            used.add((best["src"], best["idx"]))
            used_order.append((best["src"], best["idx"]))
            used_src_time.append((best["src"], m["start"]))
            if best["kind"] == "photo":
                cin = 0.0
            else:
                # take the most energetic part: centre of the moment
                slack = (m["end"] - m["start"]) - length
                cin = m["start"] + max(0.0, slack / 2)
            section_shots.append({"src": best["src"], "kind": best["kind"], "in": round(cin, 3), "out": round(cin + length, 3),
                                  "length": round(length, 4), "section": name, "face_x": m.get("face_x", 0.5),
                                  "fps": best["fps"], "log": best["log"], "moment": best["idx"],
                                  "score": round(best["score"], 3), "prompt_hit": best["prompt_hit"] > 0})
            last_src = best["src"]
            t += length
        # keep a sense of time: build/emotional/close follow the order things happened
        if name in ("build", "emotional", "close"):
            section_shots.sort(key=lambda s: (s["src"], s["in"]))
        shots.extend(section_shots)

    # transitions: hard cuts on beat, with accents at section changes
    transitions = {"hook": "flash", "build": "whip", "peak": "flash", "emotional": "dissolve", "close": "whip"}
    prev = None
    for i, s in enumerate(shots):
        s["transition"] = "cut"
        if i > 0 and s["section"] != prev:
            s["transition"] = transitions.get(s["section"], "cut")
            if tmpl_name == "talk":
                s["transition"] = "dissolve"
        prev = s["section"]
    total = sum(s["length"] for s in shots)
    peak_start = next((b[1] for b in bounds if b[0] == "peak"), body * 0.35)
    return {"template": tmpl_name, "bpm": grid["bpm"], "sections": [{"name": b[0], "start": b[1], "end": b[2]} for b in bounds],
            "shots": shots, "body": round(total, 3), "end_card": round(duration - total, 3), "drop_at": round(peak_start, 3),
            "mood": settings.get("music_mood") or mood_from_prompt(settings.get("prompt", ""), tmpl_name)}


def record_taste(taste: dict, action: str, shot: dict, moment: dict = None) -> dict:
    """Learn from what users keep (lock) and throw away (swap/delete)."""
    taste = dict(taste or {})
    sign = {"lock": 1, "keep": 0.5, "swap": -1, "delete": -1}.get(action, 0)
    if not sign or not moment:
        return taste
    lr = 0.03
    taste["faces"] = float(np.clip(taste.get("faces", 0) + sign * lr * (min(1, moment.get("faces", 0) / 2) - 0.5), -0.3, 0.3))
    taste["energy"] = float(np.clip(taste.get("energy", 0) + sign * lr * (moment.get("energy", 0) - 0.5), -0.3, 0.3))
    taste["motion"] = float(np.clip(taste.get("motion", 0) + sign * lr * (min(1, moment.get("motion", 0) * 12) - 0.5), -0.3, 0.3))
    taste["photo"] = float(np.clip(taste.get("photo", 0) + sign * lr * (1 if shot.get("kind") == "photo" else -0.2), -0.3, 0.3))
    taste["events"] = taste.get("events", 0) + 1
    return taste
