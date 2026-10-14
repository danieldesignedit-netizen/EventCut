"""Check everything: a one-minute self-test of this PC and every connected service, in plain language."""
import os
import shutil
import time
import uuid
from pathlib import Path

from . import epidemic, falai, gemini, media, smart
from .config import FOLDERS, HOME, VERSION, WORK


def _check(name, fn):
    t0 = time.time()
    try:
        detail = fn()
        ok = True
        if isinstance(detail, tuple):
            ok, detail = detail
    except Exception as e:
        ok, detail = False, str(e).splitlines()[0][:300] or e.__class__.__name__
    return {"name": name, "ok": ok, "detail": detail, "ms": int((time.time() - t0) * 1000)}


def _sample_clip():
    """A 3-second 4K test clip (made once), for decoding speed when no footage is given."""
    out = WORK / "_selftest" / "test-4k.mp4"
    if out.exists() and out.stat().st_size > 1000:
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff, _ = media.tools()
    media.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=3840x2160:r=25:d=3", "-f", "lavfi", "-i", "sine=f=440:d=3",
               "-c:v", "libx264", "-preset", "ultrafast", "-g", "12", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)])
    return str(out)


def run(clip=None, use_downloads=False):
    res = []
    ff_info = {}

    def tools():
        ff, fp = media.tools()
        ff_info["ff"] = ff
        v = media.run([ff, "-version"]).stdout.decode("utf-8", "replace").splitlines()[0]
        return f"Engine {VERSION}. {v[:80]}"
    res.append(_check("EventCut Engine and FFmpeg", tools))

    def folders():
        missing = []
        for d in FOLDERS:
            d.mkdir(parents=True, exist_ok=True)
            f = d / f".eventcut-check-{uuid.uuid4().hex[:6]}"
            try:
                f.write_text("ok")
                f.unlink()
            except Exception:
                missing.append(d.name)
        free = shutil.disk_usage(str(HOME)).free / 1e9
        if missing:
            return False, "Can't write to: " + ", ".join(missing)
        if free < 20:
            return False, f"All folders work, but only {free:.0f} GB is free on this drive. Big projects need 20 GB or more."
        return f"All EventCut folders work. {free:.0f} GB free in {HOME}."
    res.append(_check("EventCut folder", folders))

    test_clip = clip if clip and os.path.isfile(clip) else None

    def decode():
        src = test_clip or _sample_clip()
        info = media.probe(src)
        dur = float(info.get("duration") or 3)
        dst = WORK / "_selftest" / f"skim-{uuid.uuid4().hex[:6]}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        from . import quick
        t0 = time.time()
        quick.make_quick_proxy(src, dst, info)
        took = time.time() - t0
        try:
            dst.unlink()
        except OSError:
            pass
        speed = dur / max(0.01, took)
        name = Path(src).name if test_clip else "a 4K test clip"
        return f"Fast watching works on {name}: {speed:.0f}x faster than real time ({info.get('width')}x{info.get('height')}, {info.get('codec')})."
    res.append(_check("Fast watching (keyframe skim)", decode))

    def gpu():
        hw = media.hardware()
        if hw.get("kind"):
            return f"{hw['label']} is used for encoding."
        return False, "No graphics card encoder works, so the processor is used (slower). " + " ".join(hw.get("notes") or [])[:200]
    res.append(_check("Graphics card", gpu))

    # Google AI
    def google():
        c = gemini.cfg()
        if not c["key"]:
            return False, "No Google AI key in Settings."
        if not c["use"]:
            return False, "Google AI is switched off in Settings."
        model = gemini.test_key(c["key"], c["tier"])
        # the same path a real video takes: a tiny preview uploaded and watched with the moment-picking request
        src = test_clip or _sample_clip()
        prev = gemini.CACHE / "_check_preview.mp4"
        try:
            prev.unlink()
        except OSError:
            pass
        ff, _ = media.tools()
        prev.parent.mkdir(parents=True, exist_ok=True)
        media.run([ff, "-y", "-v", "error", "-t", "6", "-i", str(src), "-vf", "fps=1,scale=-2:240", "-c:v", "libx264", "-preset", "veryfast",
                   "-c:a", "aac", "-b:a", "24k", "-ac", "1", str(prev)])
        f = gemini.upload(c["key"], prev)
        try:
            resp, model, tier = gemini.generate_with(c, f["uri"], gemini._pick_prompt("test", "", 15, 1, 1, 1), gemini.PICK_SCHEMA)
        finally:
            try:
                gemini._req("DELETE", f"{gemini.API}/v1beta/{f['name']}", c["key"], timeout=20)
            except Exception:
                pass
        gemini.log_spend(resp.get("usageMetadata", {}), tier)
        gemini._json_from(resp)
        spent = gemini.month_spend()
        notes = " ".join(gemini.take_switch_notes())
        return (f"Works with {gemini.pretty(model)}: it watched a test preview and chose moments. {notes} "
                f"Spent this month: US${spent:.2f} of {c['cap']:.0f}.").replace("  ", " ")
    res.append(_check("Google AI (Gemini)", google))

    # speech
    def speech():
        if falai.enabled():
            falai.test_key(falai.cfg()["key"])
            src = test_clip or _sample_clip()
            falai.transcribe(src, "")
            return "fal speech-to-text works (a short test was transcribed)."
        if smart.whisper_available():
            return "Speech is heard on this PC (no fal key, slower)."
        return False, "No speech-to-text: add a fal key in Settings."
    res.append(_check("Speech-to-text", speech))

    # Epidemic Sound
    def es_music():
        if not epidemic.enabled():
            return False, "No Epidemic Sound key in Settings (or it's switched off)."
        r = epidemic.search("uplifting afrobeats", limit=5, instrumental=True)
        if not r["results"]:
            return False, "Search worked but found nothing."
        t = r["results"][0]
        epidemic.highlight_start(t["id"], 30)
        sim = epidemic.similar(t["id"], 3)
        epidemic.preview(t["id"], 0)
        return f"Music search, previews and similar songs work (e.g. \"{t['title']}\", {len(sim)} similar)."
    res.append(_check("Epidemic Sound music", es_music))

    def es_sfx():
        if not epidemic.enabled():
            return False, "No Epidemic Sound key in Settings."
        items = epidemic.sfx_search("fast whoosh transition", limit=5)
        if not items:
            return False, "Sound-effect search found nothing."
        return f"Sound-effect search works ({items[0]['title']}, and more)."
    res.append(_check("Epidemic Sound effects", es_sfx))

    def es_downloads():
        if not epidemic.enabled():
            return False, "No Epidemic Sound key in Settings."
        if not use_downloads:
            return True, "Not tested (it uses downloads from your plan). Tick the box to include it."
        r = epidemic.search("uplifting corporate", limit=3, instrumental=True)
        t = r["results"][0]
        p = epidemic.download(t["id"], t["title"], t["artist"])
        notes = [f"track download works ({Path(p).name})"]
        try:
            v = epidemic.create_version(t["id"], 15, t["title"], t["artist"])
            notes.append(f"exact-length edits work ({Path(v).name})")
        except epidemic.EpidemicError as e:
            notes.append(f"exact-length edits are not available: {e}")
        fx = epidemic.sfx_search("cinematic impact hit", limit=3)
        if fx:
            epidemic.sfx_download(fx[0]["id"], fx[0]["title"])
            notes.append("sound-effect download works")
        out = "; ".join(notes)
        return out[:1].upper() + out[1:] + "."
    res.append(_check("Epidemic Sound downloads", es_downloads))
    return res
