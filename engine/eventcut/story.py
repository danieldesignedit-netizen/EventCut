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
    "Conference": "talk", "Award night": "celebration", "Podcast": "talk", "Vox pop": "talk",
    "Interview": "talk", "Panel": "talk",
}

# Each event type edits differently: what counts as a great moment, pacing, music, look, transitions.
# w: how much energy (cheers, applause, laughter), faces (people, emotion) and motion (dancing, action) matter.
TRANSITION_SETS = {
    "energetic": ["whip_l", "whip_r", "flash", "zoom_in", "blurwhip"],
    "playful": ["whip_l", "whip_r", "slide_l", "pan_l", "pan_u", "zoom_in"],
    "soft": ["dissolve", "dissolve", "dip", "zoom_out"],
    "flashy": ["flash", "whip_u", "zoom_in", "tilt", "blurwhip", "whip_l"],
    "documentary": ["dissolve", "pan_l", "slide_l"],
    "cinematic": ["dip", "dissolve", "zoom_in", "tilt"],
    "sleek": ["zoom_in", "tilt", "pan_l", "blurwhip"],
    "corporate": ["blurwhip", "slide_l", "dissolve", "zoom_out", "pan_l"],
}
PROFILES = {
    "Watch party":       {"template": "celebration", "music": "upbeat", "look": "Vibrant Cinematic", "pace": 0.85,
                          "w": {"energy": 1.7, "faces": 1.2, "motion": 0.9}, "trans": "energetic", "slowmo": False, "variety": 0.93, "leaks": 3},
    "Team bonding":      {"template": "celebration", "music": "upbeat", "look": "Warm Celebration", "pace": 1.0,
                          "w": {"energy": 1.3, "faces": 1.3, "motion": 1.3}, "trans": "playful", "slowmo": False, "variety": 0.86, "leaks": 2},
    "Anniversary":       {"template": "celebration", "music": "emotional", "look": "Film Print", "pace": 1.6,
                          "w": {"energy": 0.8, "faces": 1.7, "motion": 0.5}, "trans": "soft", "slowmo": True, "variety": 0.9, "leaks": 2},
    "End-of-year party": {"template": "celebration", "music": "upbeat", "look": "Vibrant Cinematic", "pace": 0.75,
                          "w": {"energy": 1.4, "faces": 1.0, "motion": 1.7}, "trans": "flashy", "slowmo": False, "variety": 0.92, "leaks": 4},
    "CSR":               {"template": "corporate", "music": "corporate", "look": "Documentary Soft", "pace": 1.5,
                          "w": {"energy": 0.6, "faces": 1.6, "motion": 0.8}, "trans": "documentary", "slowmo": False, "variety": 0.88, "leaks": 0},
    "Award night":       {"template": "celebration", "music": "cinematic", "look": "Cinematic Teal & Orange", "pace": 1.1,
                          "w": {"energy": 1.7, "faces": 1.4, "motion": 0.7}, "trans": "cinematic", "slowmo": True, "variety": 0.92, "leaks": 3},
    "Conference":        {"template": "talk", "music": "corporate", "look": "Clean Corporate", "pace": 1.0,
                          "w": {"energy": 0.8, "faces": 1.3, "motion": 0.6}, "trans": "corporate", "slowmo": False, "variety": 0.92, "leaks": 0},
    "Product launch":    {"template": "celebration", "music": "cinematic", "look": "Cinematic Teal & Orange", "pace": 1.0,
                          "w": {"energy": 1.4, "faces": 0.9, "motion": 1.4}, "trans": "sleek", "slowmo": False, "variety": 0.92, "leaks": 2,
                          "reveal": True},
}
DEFAULT_PROFILE = {"template": "celebration", "music": "upbeat", "look": "Vibrant Cinematic", "pace": 1.0,
                   "w": {"energy": 1.0, "faces": 1.0, "motion": 1.0}, "trans": "energetic", "slowmo": True, "variety": 0.93, "leaks": 3}
TALK_PROFILE = {"template": "talk", "music": "chill", "look": "Natural", "pace": 1.0, "w": {"energy": 1.0, "faces": 1.0, "motion": 1.0},
                "trans": "documentary", "slowmo": False, "variety": 0.93, "leaks": 0}
CORP_PROFILE = dict(PROFILES["CSR"], look="Clean Corporate", trans="corporate")


def photo_rules(settings, analyses):
    """Videos first, photos as occasional accents. Returns (mode, max photo shots for this length)."""
    mode = settings.get("photos", "rarely")
    has_video = any(a.get("kind") == "video" for a in analyses)
    if not has_video:
        return "normal", 10 ** 6
    body = float(settings.get("duration", 60))
    cap = 0 if mode == "never" else 10 ** 6 if mode == "normal" else max(1, int(body / 1.6 * 0.1))
    if mode == "rarely":
        # photos fill in only where the videos run short: never cut a video short when photos could finish it
        n_video = sum(1 for a in analyses if a.get("kind") == "video" for m in a.get("moments", []) if not m.get("reject"))
        need = int(body / 1.6) + 1
        if n_video < need:
            cap += need - n_video
    return mode, cap


def profile_for(event_type: str) -> dict:
    if event_type in PROFILES:
        return PROFILES[event_type]
    t = template_for(event_type)
    return TALK_PROFILE if t == "talk" else CORP_PROFILE if t == "corporate" else DEFAULT_PROFILE


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


MOOD_WORDS = {
    "emotional": ("emotional", "touching", "sentimental", "heartfelt", "farewell", "tribute", "memorial", "inspiring"),
    "cinematic": ("cinematic", "epic", "dramatic", "night", "trailer", "powerful"),
    "corporate": ("corporate", "professional", "clean", "conference", "award"),
    "chill": ("chill", "relaxed", "calm", "soft", "lofi", "lo-fi", "acoustic"),
    "upbeat": ("upbeat", "party", "dance", "afro", "amapiano", "energetic", "fun", "hype"),
}


def music_family(settings) -> str:
    """The direction box wins ("emotional", "party"...), otherwise the event type's own music."""
    p = (settings.get("prompt") or "").lower()
    for mood, words in MOOD_WORDS.items():
        if any(w in p for w in words):
            return mood
    return profile_for(settings.get("event_type", ""))["music"]


def mood_from_prompt(prompt: str, template: str, rng=None) -> str:
    """Mood family for the music: upbeat, cinematic, corporate, chill or emotional."""
    p = (prompt or "").lower()
    for mood, words in MOOD_WORDS.items():
        if any(w in p for w in words):
            return mood
    if template == "corporate":
        return "corporate"
    if template == "talk":
        return "chill"
    return "upbeat"


def _moment_pool(analyses, terms, transcripts, taste, rng, randomness, director=None, avoid=None, prof=None):
    director = director or {}
    avoid = avoid or []
    w = (prof or DEFAULT_PROFILE)["w"]
    photo_mode = (prof or {}).get("_photos", "rarely")
    has_video = any(a.get("kind") == "video" for a in analyses)
    pool = []
    for a in analyses:
        name = a["path"].lower()
        words = transcripts.get(a["path"], [])
        for i, m in enumerate(a.get("moments", [])):
            if m.get("reject"):
                continue
            s = m["score"]
            if a.get("kind") == "photo":
                if photo_mode == "never" and has_video:
                    continue
                if photo_mode != "normal" and has_video:
                    s -= 0.35  # videos first: a photo only wins when it's clearly better than the video moments left
            # what this kind of event cares about: cheers and applause, faces and emotion, dancing and action
            s += 0.22 * (w["energy"] - 1) * m.get("energy", 0) + 0.18 * (w["faces"] - 1) * min(1, m.get("faces", 0) / 2) \
                + 0.15 * (w["motion"] - 1) * min(1, m.get("motion", 0) * 12)
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
            if any(v.get("src") == a["path"] and m["start"] < float(v["end"]) and float(v["start"]) < m["end"] for v in avoid):
                s -= 0.45  # used in the previous version: prefer something new
            d = director.get(f"{a['path']}|{i}")
            if d:
                s += d["boost"]
            pool.append({"director_section": d["section"] if d else None,"src": a["path"], "kind": a["kind"], "idx": i, "m": m, "score": s, "prompt_hit": hit,
                         "fps": a.get("info", {}).get("fps", 25), "w": a["info"].get("width"), "h": a["info"].get("height"),
                         "log": (a.get("log") or {}).get("profile")})
    return pool


def plan_edit(analyses, settings, grid, transcripts=None, taste=None):
    """Return {sections, shots, total} where every shot length is a whole number of beats."""
    if template_for(settings.get("event_type", "")) == "talk" and settings.get("talk_mode", True):
        from . import talk
        try:
            return talk.plan_talk(analyses, settings, grid, transcripts, settings.get("_ai"),
                                  source_order=source_order([a for a in analyses if a.get("kind") == "video"]))
        except ValueError:
            if settings.get("event_type") in ("Podcast", "Interview", "Panel", "Vox pop"):
                raise
            # e.g. a conference with no clear speech: edit it like an event instead of failing
            settings = dict(settings, event_type="corporate summit")
    if settings.get("order", "chronological") == "chronological":
        return plan_chronological(analyses, settings, grid, transcripts, taste)
    transcripts = transcripts or {}
    duration = float(settings.get("duration", 60))
    end_card = min(2.5, max(1.5, duration * 0.06)) if settings.get("end_card", True) else 0.0
    body = duration - end_card
    tmpl_name = template_for(settings.get("event_type", ""))
    tmpl = TEMPLATES[tmpl_name]
    style = settings.get("style") or {}
    seed = int(settings.get("seed", 0))
    rng = np.random.default_rng(seed)
    terms = prompt_terms(settings.get("prompt", ""))
    pool = _moment_pool(analyses, terms, transcripts, taste, rng, randomness=0.12 if seed else 0.02,
                        director=settings.get("director_picks"), avoid=settings.get("avoid"),
                        prof=dict(profile_for(settings.get("event_type", "")), _photos=photo_rules(settings, analyses)[0]))
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

    used, used_src_time, used_order, used_sigs = set(), [], [], []
    _, photo_cap = photo_rules(settings, analyses)
    from .analyze import similarity
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
            n_photos = sum(1 for x in shots + section_shots if x.get("kind") == "photo")
            for c in pool:
                key = (c["src"], c["idx"])
                if key in used:
                    continue
                if c["kind"] == "photo" and (n_photos >= photo_cap or (name == "hook" and photo_cap < 10 ** 6)):
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
                if c["m"].get("sig") and any(similarity(c["m"]["sig"], u) > 0.93 for u in used_sigs):
                    continue  # near-identical to a shot already in the video
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
                else:  # every moment used once: stop rather than repeat shots
                    break
                if best["kind"] != "photo":
                    avail = best["m"]["end"] - best["m"]["start"]
                    length = min(length, max(beat * 0.5, math.floor(avail / (beat * 0.5)) * beat * 0.5))
            m = best["m"]
            used.add((best["src"], best["idx"]))
            used_order.append((best["src"], best["idx"]))
            if best["m"].get("sig"):
                used_sigs.append(best["m"]["sig"])
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
            keys = [(x["src"], round(x["in"], 2)) for x in section_shots]
            if len(set(keys)) == len(keys):  # only when no shot repeats, or repeats would end up side by side
                section_shots.sort(key=lambda s: (s["src"], s["in"]))
        shots.extend(section_shots)

    # transitions: hard cuts on beat, with accents at section changes
    assign_transitions(shots, tmpl_name, rng, settings.get("transition_amount", "normal"))
    tame_moves(shots, tmpl_name, rng, settings.get("transition_amount", "normal"))
    leaks = []
    total = sum(s["length"] for s in shots)
    usable = sum(c["m"]["end"] - c["m"]["start"] for c in pool if c["kind"] != "photo") + 3 * sum(1 for c in pool if c["kind"] == "photo")
    warning = None
    if usable < body * 0.8:
        warning = (f"There are only about {int(usable)} seconds of usable footage for a {int(duration)}-second video, "
                   "so some shots are repeated. Add more footage or choose a shorter length.")
    peak_start = next((b[1] for b in bounds if b[0] == "peak"), body * 0.35)
    return {"template": tmpl_name, "bpm": grid["bpm"], "sections": [{"name": b[0], "start": b[1], "end": b[2]} for b in bounds],
            "shots": shots, "body": round(total, 3), "end_card": round(duration - total, 3), "drop_at": round(peak_start, 3),
            "mood": settings.get("music_mood") or music_family(settings),
            "leaks": pick_leaks(shots, rng, tmpl_name), "warning": warning, "shortened": bool(warning), "order": "highlights"}


# Transition families. Names map to an FFmpeg xfade style and duration in render.XFADE.
SECTION_ACCENTS = {
    "celebration": {"hook": ["flash", "zoom", "zoom_in"], "build": ["whip_l", "whip_r", "blurwhip", "tilt", "pan_l"], "peak": ["flash", "zoom_in", "whip_u", "tilt"],
                    "emotional": ["dissolve", "dip", "zoom_out"], "close": ["dissolve", "blurwhip", "dip", "zoom_out"]},
    "corporate": {"hook": ["zoom", "dissolve", "zoom_in"], "build": ["slide_l", "blurwhip", "tilt", "pan_l"], "peak": ["zoom_in", "whip_r", "tilt"],
                  "emotional": ["dissolve", "zoom_out"], "close": ["dissolve", "dip", "zoom_out"]},
    "talk": {"hook": ["dissolve"], "build": ["dissolve"], "peak": ["dissolve", "dip"], "close": ["dissolve"]},
}
IN_SECTION = {
    "celebration": {"build": ["whip_l", "whip_r", "blurwhip", "slide_l", "pan_l", "pan_u"], "peak": ["flash", "zoom", "whip_l", "whip_r", "whip_u", "blurwhip", "zoom_in", "pan_u"],
                    "hook": ["zoom", "flash"], "emotional": ["dissolve"], "close": ["dissolve"]},
    "corporate": {"build": ["slide_l", "blurwhip", "dissolve"], "peak": ["zoom", "blurwhip", "whip_r"], "emotional": ["dissolve"], "close": ["dissolve"]},
    "talk": {},
}


CAMERA_MOVES = ("tilt", "zoom_in", "zoom_out", "pan_l", "pan_u")
CLASSIC = {"celebration": ["whip_l", "whip_r", "blurwhip"], "corporate": ["blurwhip", "slide_l", "dissolve"], "talk": ["dissolve"]}


def tame_moves(shots, tmpl_name, rng, amount="normal"):
    """Keep the camera-style moves rare and deliberate: at most one tilt (two in long, busy edits), a couple of
    zooms or pans, never two in a row, and only where both shots are long enough to show the move."""
    total = sum(s["length"] for s in shots)
    tilt_cap = 2 if amount == "more" and total > 45 else 1
    move_cap = {"less": 1, "normal": 2, "more": 4}.get(amount, 2) + (1 if total > 60 else 0)
    classic = CLASSIC.get(tmpl_name, CLASSIC["celebration"])
    tilts = moves = 0
    last_move = -9
    need = {"tilt": 1.35, "zoom_in": 0.95, "zoom_out": 1.15, "pan_l": 1.05, "pan_u": 1.05}
    accents = [0] + [i for i, s in enumerate(shots) if i and s.get("transition", "cut") != "cut"] + [len(shots)]

    def run_len(a, b):
        return sum(x["length"] for x in shots[a:b])

    for i, s in enumerate(shots):
        t = s.get("transition")
        if t not in CAMERA_MOVES:
            continue
        k = accents.index(i)
        # the move needs footage on both sides: the stretch since the last transition and until the next one
        roomy = i > 0 and run_len(accents[k - 1], i) >= need[t] and run_len(i, accents[k + 1]) >= need[t]
        ok = roomy and i - last_move > 3 and (tilts < tilt_cap if t == "tilt" else moves < move_cap)
        if ok:
            last_move = i
            if t == "tilt":
                tilts += 1
            else:
                moves += 1
        else:
            s["transition"] = str(rng.choice(classic)) if i > 0 and shots[i - 1]["length"] >= 0.45 and s["length"] >= 0.45 else "cut"


def assign_transitions(shots, tmpl_name, rng, amount="normal"):
    """Hard cuts on the beat stay the backbone; accents land at section changes and every few cuts,
    chosen at random from the right family so each regenerate looks different."""
    every = {"less": (7, 11), "normal": (3, 6), "more": (2, 4)}.get(amount, (3, 6))
    accents, inner = SECTION_ACCENTS.get(tmpl_name, SECTION_ACCENTS["celebration"]), IN_SECTION.get(tmpl_name, {})
    prev, since, gap = None, 0, int(rng.integers(*every))
    for i, sh in enumerate(shots):
        sh["transition"] = "cut"
        if i == 0:
            prev = sh["section"]
            continue
        if sh["section"] != prev:
            choices = accents.get(sh["section"], ["dissolve"])
            sh["transition"] = str(rng.choice(choices))
            since, gap = 0, int(rng.integers(*every))
        else:
            since += 1
            pool = inner.get(sh["section"])
            # only on shots long enough to carry a transition without feeling rushed
            if pool and since >= gap and shots[i - 1]["length"] >= 0.45 and sh["length"] >= 0.45:
                sh["transition"] = str(rng.choice(pool))
                since, gap = 0, int(rng.integers(*every))
        prev = sh["section"]


def pick_leaks(shots, rng, tmpl_name):
    """Light-leak flares over a few key moments (section changes and the drop)."""
    if tmpl_name == "talk":
        return []
    out, t = [], 0.0
    for i, sh in enumerate(shots):
        if i and shots[i - 1]["section"] != sh["section"] and rng.random() < 0.75:
            out.append({"at": round(max(0.0, t - 0.25), 3), "style": int(rng.integers(0, 4))})
        t += sh["length"]
    return out[:5]


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


# ---------------------------------------------------------------- "As it happened" editor
def _natural_key(path: str):
    import os
    name = os.path.basename(path).lower()
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def source_order(analyses) -> list:
    """Order clips the way the event happened: camera recording time when most clips have it, else file names
    (cameras and phones number files in order: IMG_1234, C0001...)."""
    paths = [a["path"] for a in analyses]
    stamped = {a["path"]: a.get("captured") for a in analyses if a.get("captured")}
    if len(stamped) >= max(1, len(paths) * 0.6):
        return sorted(paths, key=lambda p: (stamped.get(p) or "9999", _natural_key(p)))
    return sorted(paths, key=_natural_key)


def _shot_len(m, beat, pace, calm=False):
    """Seconds for a shot: energetic moments cut fast, faces and calm moments breathe. Always whole half-beats."""
    e = m.get("energy", 0)
    beats = 1.0 if e >= 0.55 else 2.0 if e >= 0.25 else 3.0
    if calm or m.get("faces", 0) >= 1.5 and e < 0.4:
        beats += 1.0
    beats *= pace
    want = max(0.5, round(beats * 2) / 2) * beat
    avail = m["end"] - m["start"]
    if m.get("photo"):
        return max(want, 2 * beat)
    if avail < want:
        want = (avail // (beat * 0.5)) * beat * 0.5
    return want if want >= max(0.45, beat * 0.5) else 0.0


def plan_chronological(analyses, settings, grid, transcripts=None, taste=None):
    """Hook of the very best moments (3-5 s), then the event in the order it happened, using only
    interesting moments, each used once, without near-identical shots."""
    from .analyze import similarity
    transcripts = transcripts or {}
    duration = float(settings.get("duration", 60))
    end_card = min(2.5, max(1.5, duration * 0.06)) if settings.get("end_card", True) else 0.0
    body = duration - end_card
    tmpl_name = template_for(settings.get("event_type", ""))
    seed = int(settings.get("seed", 0))
    rng = np.random.default_rng(seed)
    pool = _moment_pool(analyses, prompt_terms(settings.get("prompt", "")), transcripts, taste, rng,
                        randomness=0.12 if seed else 0.02, director=settings.get("director_picks"), avoid=settings.get("avoid"),
                        prof=dict(profile_for(settings.get("event_type", "")), _photos=photo_rules(settings, analyses)[0]))
    if not pool:
        raise ValueError("No usable shots found. The footage may be too dark, blurry or shaky.")
    beat = 60.0 / grid["bpm"]
    style = settings.get("style") or {}
    prof = profile_for(settings.get("event_type", ""))
    pace = float(np.clip(style.get("median_shot", 0.55) / 0.55, 0.5, 3.0)) if style.get("median_shot") else 1.0
    pace *= prof["pace"]
    if tmpl_name == "talk":
        pace = max(pace, 2.5)

    # only interesting, engaging moments: drop the weakest third (but keep everything if footage is scarce)
    scores = np.array([c["score"] for c in pool])
    cutoff = max(0.3, float(np.percentile(scores, 35))) if len(pool) > 12 else 0.0
    good = [c for c in pool if c["score"] >= cutoff or c["prompt_hit"] > 0]

    chosen, hook = [], []
    _, photo_cap = photo_rules(settings, analyses)

    def too_similar(c, picked, limit):
        sig = c["m"].get("sig")
        return any(similarity(sig, p["m"].get("sig")) > limit for p in picked)

    # 1. hook: the very best moments from across the event, 3-5 seconds, varied sources and looks
    want_hook = 0.0 if settings.get("hook", True) is False or tmpl_name == "talk" else float(np.clip(body * 0.12, 3.0, 5.0))
    hook_len = 0.0
    hb = beat if beat >= 0.42 else beat * 2  # one beat per hook shot (two if the music is very fast)
    for c in sorted(good, key=lambda c: -(c["score"] + 0.25 * c["m"].get("energy", 0))):
        if hook_len >= want_hook - 0.05:
            break
        avail = c["m"]["end"] - c["m"]["start"]
        if (c["kind"] != "photo" and avail < hb) or too_similar(c, hook, 0.82):
            continue
        if c["kind"] == "photo" and photo_cap < 10 ** 6:
            continue  # the opening is always moving footage
        if sum(1 for h in hook if h["src"] == c["src"]) >= max(1, len(hook) // 2):
            continue  # spread the hook across clips
        hook.append(c)
        hook_len += hb

    # 2. the story: best moments first, then laid out in the order they happened
    remaining = body - hook_len
    hook_keys = {(h["src"], h["idx"]) for h in hook}
    per_scene, picked_len = {}, 0.0
    for c in sorted(good, key=lambda c: -c["score"]):
        if picked_len >= remaining:
            break
        if (c["src"], c["idx"]) in hook_keys or too_similar(c, hook, 0.9):
            continue  # never show the hook moments again
        m = c["m"]
        L = _shot_len(m, beat, pace)
        if L <= 0:
            continue
        if c["kind"] == "photo" and sum(1 for o in chosen if o["kind"] == "photo") >= photo_cap:
            continue
        key = (c["src"], m.get("scene", 0))
        cap = 1 + int(m.get("scene_len", 0) // 7)  # long takes may give a few moments, short scenes one
        if per_scene.get(key, 0) >= cap:
            continue
        if any(o["src"] == c["src"] and abs(o["m"]["start"] - m["start"]) < 2.5 for o in chosen):
            continue  # moments from the same stretch of a clip look like a repeat
        if too_similar(c, [o for o in chosen if (o["src"], o["m"].get("scene", 0)) != key], prof.get("variety", 0.93)):
            continue  # same person/place in another clip
        c["len"] = L
        chosen.append(c)
        per_scene[key] = per_scene.get(key, 0) + 1
        picked_len += L
    # too long: drop the weakest picks (keeping each clip represented if possible)
    while picked_len > remaining + beat * 0.5 and len(chosen) > 1:
        counts = {}
        for o in chosen:
            counts[o["src"]] = counts.get(o["src"], 0) + 1
        drop = min((o for o in chosen if counts[o["src"]] > 1) or chosen, key=lambda o: o["score"])
        chosen.remove(drop)
        picked_len -= drop["len"]

    order = {p: i for i, p in enumerate(source_order(analyses))}
    chosen.sort(key=lambda o: (order.get(o["src"], 0), o["m"]["start"]))
    if prof.get("reveal") and len(chosen) > 3:
        # product launch: build up, then land on the biggest moment (the reveal) at the end
        big = max(chosen, key=lambda o: o["m"].get("energy", 0) + min(1, o["m"].get("motion", 0) * 12) + 0.5 * o["score"])
        chosen.remove(big)
        chosen.append(big)

    def shot(c, length, section):
        m = c["m"]
        cin = 0.0 if c["kind"] == "photo" else m["start"] + max(0.0, ((m["end"] - m["start"]) - length) / 2)
        return {"src": c["src"], "kind": c["kind"], "in": round(cin, 3), "out": round(cin + length, 3), "length": round(length, 4),
                "section": section, "face_x": m.get("face_x", 0.5), "fps": c["fps"], "log": c["log"], "moment": c["idx"],
                "score": round(c["score"], 3), "prompt_hit": c["prompt_hit"] > 0}

    shots = [shot(h, hb, "hook") for h in hook]
    total_story = sum(o["len"] for o in chosen) or 1
    t = 0.0
    for o in chosen:
        f = t / total_story
        section = "build" if f < 0.35 else "peak" if f < 0.72 else "emotional" if f < 0.9 else "close"
        shots.append(shot(o, o["len"], section))
        t += o["len"]

    total = sum(s["length"] for s in shots)
    warning, shortened = None, False
    if total < body - beat * 2:
        shortened = True
        warning = (f"There were only enough interesting moments for about {int(round(total + end_card))} seconds, "
                   f"so the video is shorter than {int(duration)} seconds instead of repeating shots. Add more footage for a longer cut.")
    assign_transitions_chrono(shots, tmpl_name, rng, settings.get("transition_amount", "normal"), TRANSITION_SETS.get(prof["trans"]))
    tame_moves(shots, tmpl_name, rng, settings.get("transition_amount", "normal"))
    leaks, tt = [], 0.0
    for i, s in enumerate(shots):
        if i and s["transition"] != "cut" and len(leaks) < prof.get("leaks", 4) and (s["section"] != shots[i - 1]["section"] or rng.random() < 0.3):
            leaks.append({"at": round(max(0.0, tt - 0.25), 3), "style": int(rng.integers(0, 4))})
        tt += s["length"]
    if tmpl_name == "talk":
        leaks = []
    return {"template": tmpl_name, "order": "chronological", "bpm": grid["bpm"], "slowmo": prof.get("slowmo", True), "profile": prof["trans"],
            "sections": [{"name": "hook", "start": 0, "end": round(hook_len, 3)}, {"name": "story", "start": round(hook_len, 3), "end": round(total, 3)}],
            "shots": shots, "body": round(total, 3), "end_card": round(end_card if shortened else max(end_card, duration - total), 3),
            "drop_at": round(hook_len, 3) if hook_len else round(min(total * 0.3, 8 * beat * 4), 3),
            "mood": settings.get("music_mood") or music_family(settings),
            "leaks": leaks, "warning": warning, "shortened": shortened}


def assign_transitions_chrono(shots, tmpl_name, rng, amount="normal", fam=None):
    """Cuts on the beat inside a clip; accents where the story moves to the next clip; a punchy accent out of the hook."""
    fam = fam or {"celebration": ["whip_l", "whip_r", "blurwhip", "zoom", "flash", "slide_l", "tilt", "zoom_in", "zoom_out", "pan_l", "pan_u"],
                  "corporate": ["blurwhip", "slide_l", "dissolve", "zoom", "tilt", "zoom_out", "pan_l"], "talk": ["dissolve"]}.get(tmpl_name, ["whip_l", "blurwhip", "zoom"])
    within_every = {"less": 0, "normal": 0, "more": 4}.get(amount, 0)
    accent_chance = {"less": 0.5, "normal": 0.85, "more": 1.0}.get(amount, 0.85)
    since = 0
    for i, s in enumerate(shots):
        s["transition"] = "cut"
        if i == 0:
            continue
        prev = shots[i - 1]
        if prev["section"] == "hook" and s["section"] != "hook":
            s["transition"] = str(rng.choice([t for t in fam if t not in ("dissolve", "dip")] or fam)) if tmpl_name != "talk" else "dissolve"
            since = 0
        elif s["src"] != prev["src"] and prev["section"] != "hook" and s["section"] != "hook":
            if rng.random() < accent_chance and prev["length"] >= 0.45 and s["length"] >= 0.45:
                s["transition"] = "dissolve" if s["section"] in ("emotional", "close") and rng.random() < 0.5 else str(rng.choice(fam))
            since = 0
        else:
            since += 1
            if within_every and since >= within_every and s["section"] != "hook" and s["length"] >= 0.45:
                s["transition"] = str(rng.choice(fam))
                since = 0
