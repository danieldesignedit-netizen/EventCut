"""Export to Premiere Pro: an editable sequence (Final Cut Pro 7 XML, which Premiere opens with File > Import).

Every shot points to the full-quality original file, so the editor can trim, re-order and re-grade from the source:
  V1  the shots (original files, with the in and out points EventCut chose)
  V2  the finished EventCut video, switched off (a reference to compare against)
  A1  statements (the people speaking), at full level
  A2  the clips' own sound under everything else, low
  A3  the music, with its fades
  A4  the sound effects
Section markers (hook, build, peak...) sit on the sequence. The colour grade, titles and captions are in the
finished video on V2; they are not rebuilt as Premiere effects.
"""
import os
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape

from . import media

AMBIENT = 0.16      # the clips' own sound under music, as in the EventCut mix
SFX_LEVEL = 1.0


def _url(p: str) -> str:
    p = str(Path(p).resolve()).replace("\\", "/")
    if not p.startswith("/"):
        p = "/" + p  # C:/... -> /C:/...
    return "file://localhost" + quote(p, safe="/:")


def _rate(fps):
    fps = float(fps or 25)
    ntsc = abs(fps - round(fps)) > 0.01
    tb = int(round(fps))
    return f"<rate><timebase>{tb}</timebase><ntsc>{'TRUE' if ntsc else 'FALSE'}</ntsc></rate>"


def _level(gain):
    return ("<filter><effect><name>Audio Levels</name><effectid>audiolevels</effectid><effectcategory>audiolevels</effectcategory>"
            "<effecttype>audiolevels</effecttype><mediatype>audio</mediatype><parameter><parameterid>level</parameterid>"
            f"<name>Level</name><valuemin>0</valuemin><valuemax>3.98109</valuemax><value>{max(0.0, gain):.4f}</value></parameter>"
            "</effect></filter>")


def _probe(p):
    try:
        return media.probe(p) if media.classify(Path(p)) == "video" else {}
    except Exception:
        return {}


class _Files:
    """Each source file is described once; later uses refer to it by id."""

    def __init__(self, fps):
        self.ids, self.fps = {}, fps

    def ref(self, path, kind, info=None):
        if path in self.ids:
            return f'<file id="{self.ids[path]}"/>'
        fid = f"file-{len(self.ids) + 1}"
        self.ids[path] = fid
        info = info or {}
        name = escape(Path(path).name)
        dur = int(round(float(info.get("duration") or 0) * self.fps)) if kind != "photo" else self.fps * 3600  # a still lasts as long as needed
        media_xml = ""
        if kind in ("video", "photo"):
            w, h = info.get("width") or 1920, info.get("height") or 1080
            media_xml += (f"<video><samplecharacteristics>{_rate(info.get('fps') or self.fps)}<width>{w}</width><height>{h}</height>"
                          "</samplecharacteristics></video>")
        if kind == "audio" or (kind == "video" and info.get("has_audio", True)):
            media_xml += "<audio><samplecharacteristics><depth>16</depth><samplerate>48000</samplerate></samplecharacteristics><channelcount>2</channelcount></audio>"
        return (f'<file id="{fid}"><name>{name}</name><pathurl>{_url(path)}</pathurl>{_rate(info.get("fps") or self.fps)}'
                f"<duration>{max(1, dur)}</duration><media>{media_xml}</media></file>")


def build(plan: dict, sound: dict, settings: dict, analyses: dict, video: str, out_path: Path, width: int, height: int, fps: int = 25):
    fps = int(fps or 25)
    f = lambda s: int(round(float(s) * fps))
    files = _Files(fps)
    shots = plan.get("shots", [])
    total_s = float((media.probe(video) or {}).get("duration") or sum(s["length"] for s in shots)) if video and os.path.isfile(video) else \
        sum(s["length"] for s in shots)
    n = 0

    def cid():
        nonlocal n
        n += 1
        return f"clipitem-{n}"

    v1, a1, a2 = [], [], []
    t = 0.0
    markers = []
    last_sec = None
    for i, s in enumerate(shots):
        a = analyses.get(s["src"], {})
        info = a.get("info") or _probe(s["src"])
        kind = s.get("kind") or a.get("kind") or "video"
        start, end = f(t), f(t + s["length"])
        if end <= start:
            t += s["length"]
            continue
        if s.get("section") and s["section"] != last_sec:
            markers.append((start, s["section"].capitalize()))
            last_sec = s["section"]
        src_in = f(s.get("in") or 0) if kind != "photo" else 0
        src_out = src_in + (end - start)
        name = escape(Path(s["src"]).name)
        note = []
        if s.get("soundbite") and s.get("quote"):
            note.append(f"Statement: {s['quote'][:200]}")
        if s.get("what"):
            note.append(str(s["what"])[:200])
        comment = f"<comments><mastercomment1>{escape(' | '.join(note))}</mastercomment1></comments>" if note else ""
        v1.append(f'<clipitem id="{cid()}"><name>{name}</name><enabled>TRUE</enabled>{_rate(fps)}<start>{start}</start><end>{end}</end>'
                  f"<in>{src_in}</in><out>{src_out}</out>{files.ref(s['src'], 'photo' if kind == 'photo' else 'video', info)}{comment}</clipitem>")
        if kind != "photo" and info.get("has_audio", True):
            item = (f'<clipitem id="{cid()}"><name>{name}</name><enabled>TRUE</enabled>{_rate(fps)}<start>{start}</start><end>{end}</end>'
                    f"<in>{src_in}</in><out>{src_out}</out>{files.ref(s['src'], 'video', info)}"
                    f"<sourcetrack><mediatype>audio</mediatype><trackindex>1</trackindex></sourcetrack>")
            if s.get("soundbite"):
                a1.append(item + _level(1.0) + "</clipitem>")
            else:
                a2.append(item + _level(AMBIENT) + "</clipitem>")
        t += s["length"]
    body_end = f(t)

    # music: the part EventCut used, with its fades
    a3 = []
    m = (sound or {}).get("music") or {}
    mp = m.get("path")
    if mp and os.path.isfile(mp):
        try:
            minfo = media.probe(mp)
        except Exception:
            minfo = {}
        mdur = float(minfo.get("duration") or total_s)
        ms = float(m.get("start") or 0)
        length = min(total_s, max(0.5, mdur - ms))
        gain = 10 ** (float(m.get("gain_db") or 0) / 20)
        fade_in, fade_out = f(m.get("fade_in") or 0), f(m.get("fade_out") or 0)
        trans = ""
        if fade_out:
            trans = (f"<transitionitem><start>{f(length) - fade_out}</start><end>{f(length)}</end><alignment>end-black</alignment>"
                     "<effect><name>Constant Power</name><effectid>KGAudioTransCrossFade3dB</effectid><effecttype>transition</effecttype>"
                     "<mediatype>audio</mediatype></effect></transitionitem>")
        lead = ""
        if fade_in:
            lead = ("<transitionitem><start>0</start><end>" + str(fade_in) + "</end><alignment>start-black</alignment>"
                    "<effect><name>Constant Power</name><effectid>KGAudioTransCrossFade3dB</effectid><effecttype>transition</effecttype>"
                    "<mediatype>audio</mediatype></effect></transitionitem>")
        a3.append(lead + f'<clipitem id="{cid()}"><name>{escape(m.get("title") or Path(mp).stem)}</name><enabled>TRUE</enabled>{_rate(fps)}'
                  f"<start>0</start><end>{f(length)}</end><in>{f(ms)}</in><out>{f(ms + length)}</out>{files.ref(mp, 'audio', minfo)}"
                  f"{_level(gain)}</clipitem>" + trans)

    # sound effects
    a4 = []
    for x in (sound or {}).get("sfx", []):
        p = x.get("path")
        if not p or not os.path.isfile(p):
            continue
        ln = float(x.get("length") or 0)
        if ln <= 0:
            try:
                ln = float(media.probe(p).get("duration") or 1)
            except Exception:
                ln = 1.0
        st = f(x["at"])
        a4.append(f'<clipitem id="{cid()}"><name>{escape(x.get("title") or Path(p).stem)}</name><enabled>TRUE</enabled>{_rate(fps)}'
                  f"<start>{st}</start><end>{st + max(1, f(ln))}</end><in>0</in><out>{max(1, f(ln))}</out>"
                  f"{files.ref(p, 'audio', {'duration': ln})}{_level(float(x.get('gain') or SFX_LEVEL) * 2.5)}</clipitem>")

    # V2 / A5: the finished EventCut video for reference (switched off)
    v2, a5 = [], []
    if video and os.path.isfile(video):
        vinfo = {"duration": total_s, "fps": fps, "width": width, "height": height, "has_audio": True}
        v2.append(f'<clipitem id="{cid()}"><name>{escape(Path(video).name)}</name><enabled>FALSE</enabled>{_rate(fps)}<start>0</start>'
                  f"<end>{f(total_s)}</end><in>0</in><out>{f(total_s)}</out>{files.ref(video, 'video', vinfo)}</clipitem>")
        a5.append(f'<clipitem id="{cid()}"><name>{escape(Path(video).name)}</name><enabled>FALSE</enabled>{_rate(fps)}<start>0</start>'
                  f"<end>{f(total_s)}</end><in>0</in><out>{f(total_s)}</out>{files.ref(video, 'video', vinfo)}"
                  "<sourcetrack><mediatype>audio</mediatype><trackindex>1</trackindex></sourcetrack></clipitem>")

    def track(items, enabled=True, locked=False):
        return f"<track>{''.join(items)}<enabled>{'TRUE' if enabled else 'FALSE'}</enabled><locked>{'TRUE' if locked else 'FALSE'}</locked></track>"
    mk = "".join(f"<marker><name>{escape(nm)}</name><comment></comment><in>{at}</in><out>-1</out></marker>" for at, nm in markers)
    if body_end < f(total_s):
        mk += f"<marker><name>End card</name><comment>In the finished video on V2</comment><in>{body_end}</in><out>-1</out></marker>"
    title = escape(Path(out_path).stem)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="4">'
        f'<sequence id="sequence-1"><name>{title}</name><duration>{f(total_s)}</duration>{_rate(fps)}'
        "<timecode><string>00:00:00:00</string><frame>0</frame><displayformat>NDF</displayformat>" + _rate(fps) + "</timecode>"
        f"{mk}<media>"
        f"<video><format><samplecharacteristics>{_rate(fps)}<width>{width}</width><height>{height}</height>"
        "<pixelaspectratio>square</pixelaspectratio><fielddominance>none</fielddominance></samplecharacteristics></format>"
        f"{track(v1)}{track(v2, enabled=False)}</video>"
        "<audio><numOutputChannels>2</numOutputChannels>"
        f"{track(a1)}{track(a2)}{track(a3)}{track(a4)}{track(a5, enabled=False)}</audio>"
        "</media></sequence></xmeml>\n")
    Path(out_path).write_text(xml, "utf-8")
    return str(out_path)
