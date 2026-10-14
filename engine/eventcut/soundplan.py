"""The sound of a video, as editable data: the music (which track, which part, fades, level changes) and every
sound effect (which one, where, how loud). EventCut fills it in automatically; the sound timeline on the site edits
it; the mix is rebuilt from it in seconds without re-rendering the picture.

Every item says exactly where it came from (Epidemic Sound, EventCut original, free library, your upload), so the
Sound list on each video is never a guess.
"""
import json
import re
import time
from pathlib import Path

import numpy as np

from . import audio, epidemic, media
from .config import EPIDEMIC_MUSIC, EPIDEMIC_SFX, MUSIC, MUSIC_LIBRARY, MUSIC_UPLOADS, SFX_LIBRARY, SFX_ORIGINALS, WORK

SFX_HISTORY = WORK / "sfx_history.json"
# what EventCut asks Epidemic Sound for, per kind of moment
SFX_TERMS = {
    "whoosh": ["fast whoosh transition", "swoosh transition", "air whoosh swipe", "cinematic whoosh", "quick swish"],
    "riser": ["riser build up", "tension riser", "cinematic riser short", "uplifter build"],
    "impact": ["cinematic impact hit", "deep boom impact", "punchy hit drop", "trailer hit"],
    "sting": ["dramatic sting", "logo sting", "short cinematic sting", "orchestral hit ending"],
}
SFX_LEN = {"whoosh": (0.2, 3.0), "riser": (1.0, 6.0), "impact": (0.2, 4.0), "sting": (0.4, 5.0)}
SFX_GAIN = {"whoosh": 0.34, "riser": 0.30, "impact": 0.48, "sting": 0.42}
XFADE_KINDS = ("whip_l", "whip_r", "whip_u", "whip", "blurwhip", "zoom", "zoom_in", "zoom_out", "slide_l", "pan_l", "pan_u",
               "tilt", "flash", "dissolve", "dip")


def source_of(path: str) -> str:
    """Where a sound file came from, from its folder: never guessed from its name."""
    p = Path(str(path))
    pairs = [(EPIDEMIC_MUSIC, "Epidemic Sound"), (EPIDEMIC_SFX, "Epidemic Sound"), (SFX_ORIGINALS, "EventCut original"),
             (MUSIC_LIBRARY, "Free library"), (SFX_LIBRARY, "Free library"), (MUSIC_UPLOADS, "Your upload"), (MUSIC, "Your music")]
    for folder, label in pairs:
        try:
            p.resolve().relative_to(folder.resolve())
            return label
        except Exception:
            continue
    return "Your file"


def epidemic_id(path: str):
    m = re.search(r"\[([^\]]+)\](?: \d+s)?\.mp3$", str(path))
    return m.group(1) if m else None


def _history():
    try:
        return json.loads(SFX_HISTORY.read_text("utf-8"))
    except Exception:
        return []


def _remember(ids):
    h = (_history() + list(ids))[-80:]
    SFX_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    SFX_HISTORY.write_text(json.dumps(h), "utf-8")


def _length(path):
    try:
        return float(media.probe(str(path)).get("duration") or 0)
    except Exception:
        return 0.0


def bite_spans(plan: dict) -> list:
    """(start, end) on the video's timeline of every statement."""
    out, t = [], 0.0
    for s in plan.get("shots", []):
        if s.get("soundbite"):
            out.append((t, t + s["length"]))
        t += s["length"]
    return out


def cues(plan: dict, amount: str = "normal") -> list:
    """Where effects belong: whooshes on accented transitions, a riser into the drop, an impact on it, a sting at the end."""
    from .render import XFADE
    shots = plan.get("shots", [])
    out, t = [], 0.0
    trans = []
    for i, s in enumerate(shots):
        if i and s.get("transition") in XFADE:
            trans.append((t, s["transition"], s.get("section")))
        t += s["length"]
    body = t
    limit = {"less": 3, "normal": 8, "more": 99}.get(amount, 8)
    if plan.get("template") == "talk":
        limit = min(limit, 3)  # podcasts and interviews: a few gentle whooshes, never busy
    for tt, kind, sec in trans:
        if kind in ("dissolve", "dip"):
            continue  # soft transitions stay silent
        if sum(1 for c in out if c["kind"] == "whoosh") >= limit:
            break
        out.append({"kind": "whoosh", "at": round(tt, 3)})
    if plan.get("template") != "talk":
        drop = float(plan.get("drop_at") or 0)
        if drop >= 2.0:
            out.append({"kind": "riser", "at": round(drop, 3), "end_at": True})  # ends on the drop
            out.append({"kind": "impact", "at": round(drop, 3)})
        # the biggest moments of the peak get a hit (more when the edit is busy)
        peaks = []
        t = 0.0
        for s in shots:
            if s.get("section") in ("peak",) and s.get("score", 0) > 0:
                peaks.append((s.get("score", 0), t))
            t += s["length"]
        extra = {"less": 0, "normal": 1, "more": 3}.get(amount, 1)
        for _, tt in sorted(peaks, reverse=True)[:extra]:
            if all(abs(tt - c["at"]) > 2.0 for c in out):
                out.append({"kind": "impact", "at": round(tt, 3)})
        out.append({"kind": "sting", "at": round(max(0.0, body - 0.1), 3)})
    # nothing over someone speaking: statements stay clean
    spans = bite_spans(plan)
    out = [c for c in out if not any(a - 0.3 <= c["at"] <= b - 0.2 for a, b in spans) or c["kind"] == "sting"]
    # no two effects on top of each other
    out.sort(key=lambda c: c["at"])
    clean = []
    for c in out:
        if clean and abs(c["at"] - clean[-1]["at"]) < 0.35 and c["kind"] == clean[-1]["kind"]:
            continue
        clean.append(c)
    return clean


def _pick_epidemic_sfx(kind, rng, recent, ideas, n=1):
    """Up to n different, fresh Epidemic effects of this kind (Google AI's ideas first when they match the kind).
    Several are taken from each search, so a video with ten transitions still needs only a search or two."""
    words = {"whoosh": ("whoosh", "swoosh", "swish", "swipe", "transition"), "riser": ("riser", "rise", "build", "uplift"),
             "impact": ("impact", "hit", "boom", "drop", "punch"), "sting": ("sting", "ending", "logo")}[kind]
    terms = [x for x in ideas if any(w in x.lower() for w in words)] + list(SFX_TERMS[kind])
    lo, hi = SFX_LEN[kind]
    picked, seen = [], set(recent)
    for term, page in [(t, 0) for t in terms[:6]] + [(t, 40) for t in terms[:3]]:  # a second page only when the first ran out
        if len(picked) >= n:
            break
        items = epidemic.sfx_search(term, limit=40, offset=page)
        ok = [x for x in items if (lo <= x["length"] <= hi or not x["length"]) and f"esfx:{x['id']}" not in seen]
        if not ok:
            continue
        order = list(rng.permutation(len(ok)))
        for i in order[: n - len(picked)]:
            x = ok[int(i)]
            seen.add(f"esfx:{x['id']}")
            picked.append(x)
    out = []
    if picked:
        from concurrent.futures import ThreadPoolExecutor

        def dl(x):
            try:
                return {"path": epidemic.sfx_download(x["id"], x["title"]), "title": x["title"], "id": x["id"], "source": "Epidemic Sound"}
            except epidemic.EpidemicError:
                if len(picked) == 1:
                    raise
                return None
        with ThreadPoolExecutor(6) as pool:
            out = [r for r in pool.map(dl, picked) if r]
    return out


def build(plan: dict, settings: dict, music: dict, total: float, rng=None, notices=None) -> dict:
    """The automatic sound plan for a new video (new effects every time; recent ones are not repeated)."""
    rng = rng or np.random.default_rng(int(settings.get("seed", 0)) or None)
    notices = notices if notices is not None else []
    amount = settings.get("transition_amount", "normal")
    want_sfx = settings.get("auto_sfx", True)
    cs = cues(plan, amount) if want_sfx else []
    ideas = list((settings.get("_music_brief") or {}).get("sfx") or [])
    recent = set(_history())
    chosen, used_ids = {}, []
    epi_ok, epi_err = epidemic.enabled(), None
    builtin = audio.sfx_families()
    for kind in ("whoosh", "riser", "impact", "sting"):
        if not any(c["kind"] == kind for c in cs):
            continue
        # a different effect at every cue: fresh Epidemic effects first, EventCut's own to fill any gap, and only then a repeat
        n_need = sum(1 for c in cs if c["kind"] == kind)
        files = []
        if epi_ok:
            try:
                files = _pick_epidemic_sfx(kind, rng, recent | set(used_ids), ideas, n_need)
            except epidemic.EpidemicError as e:
                epi_ok, epi_err = False, str(e)
        used_ids += [f"esfx:{f['id']}" for f in files]
        if len(files) < n_need:
            fam = {"sting": "impact"}.get(kind, kind)
            opts = [str(o) for o in (builtin.get(fam) or builtin.get("whoosh") or [])]
            for i in rng.permutation(len(opts))[: n_need - len(files)] if opts else []:
                p = opts[int(i)]
                files.append({"path": p, "title": Path(p).stem, "id": None, "source": source_of(p)})
        if files:
            files = [files[int(i)] for i in rng.permutation(len(files))]  # Epidemic and built-in effects mixed through the video
        chosen[kind] = files
    if want_sfx and settings.get("_epidemic_expected") and epi_err:
        notices.append(f"Epidemic Sound sound effects weren't used: {epi_err.rstrip('.')}. EventCut used its own effects instead.")
    sfx, counters = [], {}
    for c in cs:
        files = chosen.get(c["kind"]) or []
        if not files:
            continue
        k = counters.get(c["kind"], 0)
        counters[c["kind"]] = k + 1
        f = files[k % len(files)]
        at = c["at"]
        ln = _length(f["path"])
        if c["kind"] == "riser" and ln:
            at = max(0.0, c["at"] - ln)  # a riser finishes exactly on the drop
        elif c["kind"] == "whoosh":
            at = max(0.0, at - min(0.25, ln / 2 if ln else 0.25))  # the whoosh peaks on the cut
        sfx.append({"kind": c["kind"], "path": f["path"], "title": f["title"], "source": f["source"], "id": f.get("id"),
                    "at": round(at, 3), "gain": SFX_GAIN[c["kind"]], "length": round(ln, 2)})
    _remember(used_ids)
    # music level through the video: lifts for the peak, softer for emotional moments (speech ducking is automatic)
    gains = []
    t = 0.0
    lift = {"hook": 1.0, "peak": 2.0, "emotional": -3.0, "close": 0.0, "build": 0.0, "story": 0.0}
    last = None
    for s in plan.get("shots", []):
        db = lift.get(s.get("section"), 0.0)
        if db != last:
            gains.append({"t": round(t, 3), "db": db})
            last = db
        t += s["length"]
    src = music.get("source")
    mpath = music.get("path")
    m = {"path": mpath, "start": float(music.get("start") or 0), "fade_in": float(music.get("fade_in", 0.0) or 0.0),
         "fade_out": float(music.get("fade_out", 1.2) if music.get("fade_out") is not None else 1.2), "gain_db": 0.0,
         "title": music.get("title") or (Path(mpath).stem if mpath else music.get("mood")), "id": music.get("es_id") or (epidemic_id(mpath) if mpath else None),
         "source": ("Epidemic Sound" if src == "epidemic" else source_of(mpath)) if mpath else "EventCut original",
         "mood": music.get("mood"), "version": bool(music.get("version"))}
    statements, t = [], 0.0
    for s_ in plan.get("shots", []):
        if s_.get("soundbite"):
            statements.append({"at": round(t, 3), "length": round(s_["length"], 3), "quote": s_.get("quote", ""), "speaker": s_.get("speaker", "")})
        t += s_["length"]
    return {"music": m, "gains": gains, "sfx": sfx, "statements": statements, "total": round(total, 3),
            "made": time.strftime("%Y-%m-%d %H:%M")}


def sound_list(sound: dict) -> list:
    """What the Sound list shows: every item with its exact source."""
    out = []
    m = sound.get("music") or {}
    if m:
        out.append({"type": "Music", "title": m.get("title") or "Music", "source": m.get("source") or "Unknown",
                    "id": m.get("id"), "at": 0.0,
                    "detail": ("exact-length edit by Epidemic Sound" if m.get("version") else
                               f"starts {m.get('start', 0):.1f} s into the track" if m.get("start") else "")})
    for x in sorted(sound.get("sfx", []), key=lambda x: x["at"]):
        out.append({"type": f"Sound effect ({x['kind']})", "title": x["title"], "source": x["source"], "id": x.get("id"),
                    "at": x["at"], "detail": ""})
    for b in sound.get("statements", []):
        out.append({"type": "Statement", "title": f"\u201c{b.get('quote', '')}\u201d", "source": "Original event sound", "id": None,
                    "at": b.get("at", 0.0), "detail": (b.get("speaker") or "") + ", music lowered"})
    out.append({"type": "Event sound", "title": "Speech and crowd from the footage", "source": "Original event sound", "id": None,
                "at": 0.0, "detail": "kept low under the music" if sound.get("statements") else "lowered under music, raised for speeches"})
    return out


def sound_list_text(sound: dict, video_name: str) -> str:
    lines = [f"Sound list for {video_name}", ""]
    for it in sound_list(sound):
        when = f"at {int(it['at'] // 60)}:{it['at'] % 60:04.1f}  " if it["type"].startswith(("Sound effect", "Statement")) else ""
        ident = f"  (Epidemic ID {it['id']})" if it.get("id") and it["source"] == "Epidemic Sound" else ""
        lines.append(f"- {it['type']}: {it['title']}  |  {it['source']}{ident}  {when}{it['detail']}".rstrip())
    return "\n".join(lines) + "\n"
