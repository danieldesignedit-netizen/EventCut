"""Rendering: shot list -> graded, reframed segments -> transitions -> titles/logo/captions -> mixed audio -> exports."""
import json
import math
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np

from . import audio, grade, media
from .config import HOME, tools

FORMATS = {"9:16": (1080, 1920), "16:9": (1920, 1080), "1:1": (1080, 1080), "4:5": (1080, 1350)}
XFADE = {"whip": ("smoothleft", 0.22), "flash": ("fadewhite", 0.18), "dissolve": ("fade", 0.45)}


def _font(name="Poppins-Bold.ttf"):
    import sys
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    user = HOME / "Assets" / "Fonts"
    for d in (user, base / "fonts"):
        p = d / name
        if p.exists():
            return p
    return base / "fonts" / "Poppins-Bold.ttf"


def fpath(p) -> str:
    """Escape a path for use inside an ffmpeg filter argument (handles Windows drive colons)."""
    s = str(Path(p).resolve()).replace("\\", "/")
    return s.replace(":", r"\:").replace("'", r"\'").replace(",", r"\,").replace("[", r"\[").replace("]", r"\]")


def ftext(t: str) -> str:
    return t.replace("\\", "\\\\").replace(":", r"\:").replace("'", "’").replace("%", r"\%").replace(",", r"\,")


def _hex(h, alpha=None):
    h = (h or "#FFFFFF").strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return f"0x{h.upper()}" + (f"@{alpha}" if alpha is not None else "")


def _ass_color(h, alpha=0):
    h = (h or "#FFFFFF").strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def _moving_avg(x, win):
    """O(n) centred moving average (np.convolve with a long window is far too slow on minutes of audio)."""
    if win <= 1 or len(x) == 0:
        return x
    c = np.cumsum(np.concatenate([[0.0], x.astype(np.float64)]))
    half = win // 2
    idx = np.arange(len(x))
    lo = np.clip(idx - half, 0, len(x))
    hi = np.clip(idx + half + 1, 0, len(x))
    return ((c[hi] - c[lo]) / np.maximum(1, hi - lo)).astype(np.float32)


class Renderer:
    def __init__(self, plan, settings, analyses, out_dir: Path, progress=None, cancel=None):
        self.plan, self.s, self.out = plan, settings, Path(out_dir)
        self.an = {a["path"]: a for a in analyses}
        self.progress = progress or (lambda stage, frac, note="": None)
        self.cancel = cancel or (lambda: False)
        self.W, self.H = FORMATS.get(settings.get("format", "9:16"), FORMATS["9:16"])
        self.fps = int(settings.get("fps", 25))
        self.work = self.out / "_work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.ffmpeg, _ = tools()
        theme = [c for c in (settings.get("theme") or []) if c]
        self.primary = theme[0] if theme else "#FFFFFF"
        self.accent = theme[2] if len(theme) > 2 else theme[1] if len(theme) > 1 else "#FF3D6E"
        self.preview = settings.get("quality") == "preview"
        if self.preview:
            scale = 540 / min(self.W, self.H)
            self.W, self.H = int(self.W * scale) // 2 * 2, int(self.H * scale) // 2 * 2

    def check(self):
        if self.cancel():
            raise InterruptedError("Cancelled")

    # ------------------------------------------------------------- colour per source
    def lut_for_source(self, src):
        a = self.an.get(src, {})
        log = a.get("log") or {}
        profile = self.s.get("log_overrides", {}).get(src) or (log.get("profile") if log.get("is_log") else None)
        corr = grade.correction_params(a.get("frames", [])) if self.s.get("auto_correct", True) else None
        if profile and corr:
            corr = {"gains": corr["gains"], "gamma": 1.0}  # exposure judged after Log decode
        transfer = None
        ref = self.s.get("match_look")
        if ref and a.get("frames"):
            transfer = {"src": self._source_lab_stats(src), "ref": ref, "strength": float(self.s.get("match_strength", 0.7))}
        custom = self.s.get("custom_lut")
        return grade.lut_for(log_profile=profile, correction=corr, preset_name=self.s.get("preset", "Vibrant Cinematic"),
                             intensity=float(self.s.get("intensity", 1.0)), transfer=transfer,
                             theme_hex=self.primary if self.s.get("theme_tint") else None,
                             custom_cube=custom if custom and Path(custom).exists() else None)

    def _source_lab_stats(self, src):
        import cv2
        a = self.an[src]
        if a["kind"] == "photo":
            img = cv2.imread(src)
        else:
            cap = cv2.VideoCapture(a["proxy"])
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            frames = []
            for i in np.linspace(0, n - 1, 8).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, f = cap.read()
                if ok:
                    frames.append(cv2.resize(f, (160, 90)))
            cap.release()
            img = np.concatenate(frames, 0) if frames else np.zeros((90, 160, 3), np.uint8)
        return grade.lab_stats(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255)

    # ------------------------------------------------------------- one shot
    def render_shot(self, i, shot, extra_tail=0.0):
        out = self.work / f"shot_{i:04d}.mp4"
        src = shot["src"]
        a = self.an.get(src, {})
        info = a.get("info", {})
        length = shot["length"] + extra_tail
        lut = self.lut_for_source(src)
        W, H = self.W, self.H
        vf = []
        args = [self.ffmpeg, "-y", "-v", "error"]
        if shot["kind"] == "photo":
            args += ["-loop", "1", "-t", f"{length:.3f}", "-i", src]
            sw, sh = info.get("width", W), info.get("height", H)
            z = max(W / sw, H / sh) * 1.12
            fx = shot.get("face_x", 0.5)
            frames = max(1, int(round(length * self.fps)))
            # gentle push-in toward faces (Ken Burns)
            vf.append(f"scale={int(sw * z * 1.1) // 2 * 2}:-2")
            vf.append(f"zoompan=z='min(1+0.10*on/{frames},1.10)':x='({fx}*iw-iw/zoom/2)*(1-0)+0*on':y='ih/2-ih/zoom/2':d={frames}:s={W}x{H}:fps={self.fps}")
        else:
            args += ["-ss", f"{shot['in']:.3f}", "-t", f"{length:.3f}", "-i", src]
            if media.is_hdr(info):
                vf.append("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
            sw, sh = info.get("width", W), info.get("height", H)
            # slow-motion for high-frame-rate clips in emotional/close sections
            src_fps = float(info.get("fps") or 25)
            slow = shot["section"] in ("emotional", "close") and src_fps >= 48 and not self.preview
            if slow:
                factor = min(2.0, src_fps / self.fps)
                args[args.index("-t") + 1] = f"{length / factor:.3f}"
                vf.append(f"setpts={factor:.3f}*PTS")
            m = a.get("moments", [{}])[shot.get("moment", 0)] if a.get("moments") else {}
            if 0.012 < m.get("shake", 0) <= 0.035 and not self.preview:
                vf.append("deshake=rx=32:ry=32")
            scale = max(W / sw, H / sh)
            if shot["section"] == "emotional":
                scale *= 1.04  # slight room for a slow push
            # crop in source pixels first (fast), then scale to the output size
            cw, ch = min(sw, W / scale), min(sh, H / scale)
            cx = min(max(0, shot.get("face_x", 0.5) * sw - cw / 2), sw - cw)
            cy = (sh - ch) / 2
            vf.append(f"crop={int(cw) // 2 * 2}:{int(ch) // 2 * 2}:{int(cx)}:{int(cy)}")
            vf.append(f"scale={W}:{H}:flags=lanczos")
            vf.append(f"fps={self.fps}")
        vf.append(f"lut3d=file='{fpath(lut)}'")
        if self.s.get("preset", "Vibrant Cinematic") in ("Vibrant Cinematic", "Cinematic Teal & Orange", "Warm Celebration") and not self.preview:
            # soft highlight glow (halation) for the cinematic look
            # computed at quarter resolution for speed
            vf.append(f"split[a][b];[b]scale={W // 4}:{H // 4},gblur=sigma={max(3, W // 360)},curves=all='0/0 0.7/0.05 1/0.8',scale={W}:{H}[g];[a][g]blend=all_mode=screen:all_opacity=0.22")
        if self.s.get("grain", True) and not self.preview:
            vf.append("noise=alls=3:allf=t")
        vf += ["setsar=1", "format=yuv420p", f"trim=duration={length:.3f}", "setpts=PTS-STARTPTS"]
        has_audio = shot["kind"] != "photo" and info.get("has_audio")
        if has_audio:
            af = f"aresample=48000,aformat=channel_layouts=stereo,apad,atrim=duration={length:.3f}"
            args += ["-filter_complex", f"[0:v]{','.join(vf)}[v];[0:a]{af}[a]", "-map", "[v]", "-map", "[a]"]
        else:
            args += ["-f", "lavfi", "-t", f"{length:.3f}", "-i", "anullsrc=r=48000:cl=stereo",
                     "-filter_complex", f"[0:v]{','.join(vf)}[v]", "-map", "[v]", "-map", "1:a"]
        args += ["-r", str(self.fps)] + media.intermediate_args()
        args += ["-c:a", "pcm_s16le", "-t", f"{length:.3f}", str(out.with_suffix(".mov"))]
        media.run(args)
        return out.with_suffix(".mov")

    # ------------------------------------------------------------- assembly
    def assemble(self, files):
        """Concatenate runs of hard cuts, then xfade between runs at accented transitions."""
        shots = self.plan["shots"]
        runs, cur = [], []
        for i, (f, s) in enumerate(zip(files, shots)):
            if i > 0 and s["transition"] in XFADE:
                runs.append(cur)
                cur = []
            cur.append((f, s))
        runs.append(cur)
        run_files = []
        for r_i, run in enumerate(runs):
            lst = self.work / f"run_{r_i}.txt"
            lst.write_text("".join(f"file '{Path(f).resolve().as_posix()}'\n" for f, _ in run), "utf-8")
            out = self.work / f"run_{r_i}.mov"
            media.run([self.ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)])
            run_files.append(out)
        if len(run_files) == 1:
            return run_files[0]
        # xfade chain; earlier runs were rendered with an extra tail equal to the transition duration
        inputs, fc, offset = [], [], 0.0
        for f in run_files:
            inputs += ["-i", str(f)]
        prev_v, prev_a = "0:v", "0:a"
        lengths = [float(media.probe(str(f))["duration"]) for f in run_files]
        offset = lengths[0]
        for k in range(1, len(run_files)):
            trans = runs[k][0][1]["transition"]
            kind, d = XFADE[trans]
            offset -= d
            fc.append(f"[{prev_v}][{k}:v]xfade=transition={kind}:duration={d}:offset={offset:.3f}[v{k}]")
            fc.append(f"[{prev_a}][{k}:a]acrossfade=d={d}:c1=tri:c2=tri[a{k}]")
            prev_v, prev_a = f"v{k}", f"a{k}"
            offset += lengths[k]
        out = self.work / "body.mov"
        media.run([self.ffmpeg, "-y", "-v", "error"] + inputs + ["-filter_complex", ";".join(fc), "-map", f"[{prev_v}]", "-map", f"[{prev_a}]"]
                  + media.intermediate_args() + ["-c:a", "pcm_s16le", str(out)])
        return out

    # ------------------------------------------------------------- graphics
    def logo_file(self):
        logo = self.s.get("logo")
        if logo and Path(logo).exists():
            return Path(logo)
        folder = HOME / "Assets" / "Logos"
        for ext in ("*.png", "*.webp", "*.mov", "*.webm", "*.gif", "*.jpg"):
            found = sorted(folder.glob(ext))
            if found:
                return found[0]
        return None

    def end_card(self, dur):
        out = self.work / "endcard.mov"
        W, H = self.W, self.H
        logo = self.logo_file()
        title = self.s.get("end_text") or self.s.get("title") or ""
        bg = self.s.get("end_bg") or "#05060A"
        args = [self.ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-t", f"{dur:.3f}", "-i", f"color=c={_hex(bg)}:s={W}x{H}:r={self.fps}",
                "-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        fc = "[0:v]format=yuv420p"
        # accent line that draws in
        fc += f",drawbox=x=(iw-w)/2:y=ih*0.62:w='min(iw*0.28\\,iw*0.28*t/0.5)':h=max(2\\,ih/400):color={_hex(self.accent)}:t=fill"
        if title:
            fs = int(min(W, H) * 0.045)
            fc += (f",drawtext=fontfile='{fpath(_font('Poppins-Medium.ttf'))}':text='{ftext(title)}':fontsize={fs}:fontcolor=white"
                   f":x=(w-tw)/2:y=h*0.66:alpha='min(1,max(0,(t-0.35)/0.4))'")
        fc += "[bg]"
        if logo:
            args += (["-stream_loop", "-1"] if logo.suffix.lower() in (".mov", ".webm", ".gif") else ["-loop", "1"]) + ["-t", f"{dur:.3f}", "-i", str(logo)]
            lw = int(min(W, H) * 0.42)
            fc += (f";[2:v]scale={lw}:-1,format=rgba,fade=in:st=0.1:d=0.5:alpha=1[lg];"
                   f"[bg][lg]overlay=(W-w)/2:(H*0.5-h)/2+H*0.08:shortest=1,fade=out:st={max(0, dur - 0.4):.2f}:d=0.4[v]")
        else:
            wm = self.s.get("title") or "EventCut"
            fc += (f";[bg]drawtext=fontfile='{fpath(_font())}':text='{ftext(wm)}':fontsize={int(min(W, H) * 0.08)}:fontcolor=white"
                   f":x=(w-tw)/2:y=h*0.5-th:alpha='min(1,t/0.5)',fade=out:st={max(0, dur - 0.4):.2f}:d=0.4[v]")
        args += ["-filter_complex", fc, "-map", "[v]", "-map", "1:a", "-r", str(self.fps)] + media.intermediate_args() + ["-c:a", "pcm_s16le", "-t", f"{dur:.3f}", str(out)]
        media.run(args)
        return out

    def captions_ass(self, words_timeline):
        """Bold, theme-coloured, word-by-word highlighted captions inside platform safe zones."""
        W, H = self.W, self.H
        fs = int(min(W, H) * (0.13 if H > W else 0.085))
        margin_v = int(H * (0.24 if H > W else 0.08))  # stay above TikTok/Reels bottom UI
        hl = _ass_color(self.accent)
        style = (f"Style: Cap,Poppins,{fs},&H00FFFFFF,{hl},&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,{max(2, fs // 14)},0,2,"
                 f"{int(W * 0.08)},{int(W * 0.08)},{margin_v},1")
        lines, cur = [], []
        for w in words_timeline:
            if cur and (len(cur) >= 4 or w["start"] - cur[-1]["end"] > 0.6 or w["start"] - cur[0]["start"] > 2.2):
                lines.append(cur)
                cur = []
            cur.append(w)
        if cur:
            lines.append(cur)

        def ts(t):
            t = max(0, t)
            return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"
        ev = []
        for ln in lines:
            # one event per word: the whole line in white, the word being spoken in the accent colour
            for k, w in enumerate(ln):
                start = w["start"]
                end = ln[k + 1]["start"] if k + 1 < len(ln) else ln[-1]["end"] + 0.15
                parts = []
                for j, ww in enumerate(ln):
                    t = ww["word"].strip().upper()
                    parts.append(f"{{\\1c{_ass_color(self.accent)}&\\fscx106\\fscy106}}{t}{{\\1c&H00FFFFFF&\\fscx100\\fscy100}}" if j == k else t)
                fade = "{\\fad(80,0)}" if k == 0 else ""
                ev.append(f"Dialogue: 0,{ts(start)},{ts(end)},Cap,,0,0,0,,{fade}{' '.join(parts)}")
        txt = ("[Script Info]\nScriptType: v4.00+\nPlayResX: %d\nPlayResY: %d\nWrapStyle: 0\n\n[V4+ Styles]\n"
               "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
               "%s\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n%s\n") % (W, H, style, "\n".join(ev))
        p = self.work / "captions.ass"
        p.write_text(txt, "utf-8")
        return p

    def words_on_timeline(self, transcripts):
        """Map source-time words to the edited timeline."""
        out, t = [], 0.0
        talk = self.plan.get("template") == "talk"
        for s in self.plan["shots"]:
            if not talk and s["length"] < 1.4:  # skip speech fragments in fast montage cuts
                t += s["length"]
                continue
            for w in transcripts.get(s["src"], []):
                if s["kind"] != "photo" and s["in"] <= w["start"] and w["end"] <= s["out"]:
                    out.append({"word": w["word"], "start": t + w["start"] - s["in"], "end": t + w["end"] - s["in"]})
            t += s["length"]
        return out

    def finish_video(self, body, end, captions):
        W, H = self.W, self.H
        out = self.work / "video.mp4"
        logo = self.logo_file()
        args = [self.ffmpeg, "-y", "-v", "error", "-i", str(body), "-i", str(end)]
        fc = "[0:v]format=yuv420p"
        title = self.s.get("title")
        if title:
            fs = int(min(W, H) * (0.085 if H > W else 0.07))
            y = "h*0.40" if H > W else "h*0.42"
            fc += (f",drawtext=fontfile='{fpath(_font())}':text='{ftext(title.upper())}':fontsize={fs}:fontcolor={_hex(self.primary)}"
                   f":borderw={max(2, fs // 18)}:bordercolor=black@0.35:x=(w-tw)/2:y={y}"
                   f":alpha='if(lt(t,0.15),0,if(lt(t,0.45),(t-0.15)/0.3,if(lt(t,2.3),1,max(0,1-(t-2.3)/0.3))))':enable='lt(t,2.7)'")
            sub = self.s.get("subtitle")
            if sub:
                fc += (f",drawtext=fontfile='{fpath(_font('Poppins-Medium.ttf'))}':text='{ftext(sub)}':fontsize={int(fs * 0.38)}:fontcolor=white"
                       f":x=(w-tw)/2:y={y}+{int(fs * 1.25)}:alpha='if(lt(t,0.4),0,if(lt(t,0.7),(t-0.4)/0.3,if(lt(t,2.3),1,max(0,1-(t-2.3)/0.3))))':enable='lt(t,2.7)'")
        if captions:
            fc += f",ass='{fpath(captions)}':fontsdir='{fpath(_font().parent)}'"
        fc += "[b]"
        if logo and self.s.get("watermark", True):
            loop = ["-stream_loop", "-1"] if logo.suffix.lower() in (".mov", ".webm", ".gif") else ["-loop", "1"]
            args += loop + ["-i", str(logo)]
            lw = int(min(W, H) * 0.14)
            mx, my = int(W * 0.06), int(H * (0.06 if H > W else 0.05))
            fc += f";[2:v]scale={lw}:-1,format=rgba,colorchannelmixer=aa=0.85[lg];[b][lg]overlay={mx}:{my}:shortest=1[bb]"
            body_label = "bb"
        else:
            body_label = "b"
        fc += f";[{body_label}][1:v]concat=n=2:v=1:a=0[v]"
        codec = self.s.get("codec", "h264")
        enc = media.video_encoder_args("draft") if self.preview else media.video_encoder_args("high", codec)
        bitrate = [] if self.preview else ["-maxrate", "16M" if max(W, H) >= 1920 else "10M", "-bufsize", "24M"]
        args += ["-filter_complex", fc, "-map", "[v]", "-r", str(self.fps)] + enc + bitrate + ["-pix_fmt", "yuv420p", "-an", str(out)]
        media.run(args)
        return out

    # ------------------------------------------------------------- audio
    def mix_audio(self, body, total, transitions_at):
        sr = 48000
        plan = self.plan
        # ambience/speech from the footage, cleaned
        p = media.run([self.ffmpeg, "-v", "error", "-i", str(body), "-vn", "-af", "highpass=f=80,afftdn=nr=12:nf=-35",
                       "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"])
        amb = np.frombuffer(p.stdout, dtype=np.float32).reshape(-1, 2)
        n = int(total * sr)
        amb = np.pad(amb, ((0, max(0, n - len(amb))), (0, 0)))[:n]
        # music
        m = self.s.get("music") or {}
        if m.get("path") and Path(m["path"]).exists():
            mp = media.run([self.ffmpeg, "-v", "error", "-ss", str(m.get("start", 0)), "-i", m["path"], "-t", f"{total:.3f}",
                            "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"])
            music = np.frombuffer(mp.stdout, dtype=np.float32).reshape(-1, 2).copy()
            music = np.pad(music, ((0, max(0, n - len(music))), (0, 0)))[:n]
            fade = int(1.2 * sr)
            music[-fade:] *= np.linspace(1, 0, fade)[:, None]
        else:
            mood = m.get("mood") or plan.get("mood", "Upbeat Afro House")
            gen = audio.generate_music(mood, total, plan.get("drop_at"), seed=int(self.s.get("seed", 0)))
            idx = np.linspace(0, len(gen) - 1, int(len(gen) * sr / audio.SR)).astype(int)  # 44.1k -> 48k
            music = gen[idx][:n]
            music = np.pad(music, ((0, max(0, n - len(music))), (0, 0)))
        # duck music under speech: speech-band energy of the ambience
        env = np.abs(amb).mean(axis=1)
        envs = _moving_avg(env, int(0.25 * sr))
        talk = plan.get("template") == "talk"
        thresh = np.percentile(envs, 60 if talk else 85)
        speech = (envs > thresh).astype(np.float32)
        speech = _moving_avg(speech, int(0.4 * sr))
        vols = self.s.get("volumes") or {}
        v_music = float(vols.get("music", 1.0))
        v_speech = float(vols.get("speech", 1.0))
        v_sfx = float(vols.get("sfx", 1.0))
        duck_db = 12 if talk else 7
        duck = 10 ** (-duck_db * speech / 20)
        amb_gain = (0.9 if talk else 0.32) * v_speech
        mix = amb * amb_gain + music * (0.55 if talk else 0.9) * v_music * duck[:, None]
        # sound effects at transitions + custom placements
        sfx_dir = audio.LIB / "SFX"
        cache = {}

        def load(name):
            if name not in cache:
                f = sfx_dir / f"EventCut {name}.wav"
                if not f.exists():
                    f = Path(name)
                r = media.run([self.ffmpeg, "-v", "error", "-i", str(f), "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"])
                cache[name] = np.frombuffer(r.stdout, dtype=np.float32).reshape(-1, 2)
            return cache[name]
        placements = list(self.s.get("sfx") or [])
        if self.s.get("auto_sfx", True):
            for t, kind in transitions_at:
                name = {"whip": "Whoosh", "flash": "Impact", "dissolve": "Reverse Whoosh"}.get(kind)
                if name:
                    lead = 0.28 if name.endswith("Whoosh") else 0.0
                    placements.append({"name": name, "at": max(0, t - lead), "gain": 0.45 if name == "Impact" else 0.35})
        for pl in placements:
            try:
                clip = load(pl["name"]) * float(pl.get("gain", 0.4)) * v_sfx
            except Exception:
                continue
            i = int(pl["at"] * sr)
            if i < n:
                seg = clip[: n - i]
                mix[i:i + len(seg)] += seg
        raw = self.work / "mix_raw.wav"
        audio.write_wav_sr(raw, mix, sr)
        # loudness: two-pass EBU R128 to the platform target
        target = float(self.s.get("lufs", -14))
        p = media.run([self.ffmpeg, "-v", "info", "-i", str(raw), "-af", f"loudnorm=I={target}:TP=-1:LRA=11:print_format=json", "-f", "null", "-"], check=False)
        err = (p.stderr or b"").decode("utf-8", "replace")
        out = self.work / "mix.wav"
        try:
            js = json.loads(err[err.rindex("{"): err.rindex("}") + 1])
            af = (f"loudnorm=I={target}:TP=-1:LRA=11:measured_I={js['input_i']}:measured_TP={js['input_tp']}:"
                  f"measured_LRA={js['input_lra']}:measured_thresh={js['input_thresh']}:offset={js['target_offset']}:linear=true")
        except Exception:
            af = f"loudnorm=I={target}:TP=-1:LRA=11"
        media.run([self.ffmpeg, "-y", "-v", "error", "-i", str(raw), "-af", af + ",aresample=48000", "-c:a", "pcm_s16le", str(out)])
        return out

    # ------------------------------------------------------------- main
    def run(self, transcripts=None):
        t0 = time.time()
        shots = self.plan["shots"]
        files = []
        # shots before an accented transition get an extra tail so xfade keeps the beat timing
        tails = {i - 1: XFADE[s["transition"]][1] for i, s in enumerate(shots) if i > 0 and s["transition"] in XFADE}
        from concurrent.futures import ThreadPoolExecutor, as_completed
        workers = max(1, min(4, (os.cpu_count() or 2) // 2))
        # build LUTs first (shared per source), then render shots in parallel
        for src in {s["src"] for s in shots}:
            self.lut_for_source(src)
        files = [None] * len(shots)
        done = 0
        with ThreadPoolExecutor(workers) as pool:
            futs = {pool.submit(self.render_shot, i, s, tails.get(i, 0.0)): i for i, s in enumerate(shots)}
            for f in as_completed(futs):
                self.check()
                files[futs[f]] = f.result()
                done += 1
                self.progress("render", done / len(shots), f"Grading and reframing shot {done} of {len(shots)}")
        self.check()
        self.progress("assemble", 0.2, "Adding transitions")
        body = self.assemble(files)
        body_len = float(media.probe(str(body))["duration"])
        self.progress("assemble", 1.0)
        end_len = max(1.5, float(self.s.get("duration", 60)) - body_len)
        end = self.end_card(end_len)
        captions = None
        if self.s.get("captions") and transcripts:
            words = self.words_on_timeline(transcripts)
            if words:
                captions = self.captions_ass(words)
                self.write_srt(words)
        self.check()
        self.progress("graphics", 0.3, "Adding titles, logo and captions")
        video = self.finish_video(body, end, captions)
        self.progress("graphics", 1.0)
        total = float(media.probe(str(video))["duration"])
        trans_at, t = [], 0.0
        for s in shots:
            if s["transition"] in XFADE:
                trans_at.append((t, s["transition"]))
            t += s["length"]
        self.check()
        self.progress("audio", 0.3, "Mixing music, speech and sound effects")
        mix = self.mix_audio(body, total, trans_at)
        self.progress("audio", 1.0)
        self.progress("export", 0.2, "Exporting")
        outputs = self.export(video, mix)
        self.progress("export", 1.0)
        outputs["seconds"] = round(time.time() - t0, 1)
        return outputs

    def write_srt(self, words):
        lines, cur = [], []
        for w in words:
            if cur and (len(cur) >= 7 or w["start"] - cur[-1]["end"] > 0.7):
                lines.append(cur)
                cur = []
            cur.append(w)
        if cur:
            lines.append(cur)

        def ts(t):
            ms = int(round(t * 1000))
            return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
        srt = "\n".join(f"{i + 1}\n{ts(l[0]['start'])} --> {ts(l[-1]['end'] + 0.1)}\n{' '.join(w['word'].strip() for w in l)}\n" for i, l in enumerate(lines))
        (self.out / "captions.srt").write_text(srt, "utf-8")
        vtt = "WEBVTT\n\n" + srt.replace(",", ".")
        (self.out / "captions.vtt").write_text(vtt, "utf-8")

    def export(self, video, mix):
        name = self.s.get("name") or "EventCut"
        stem = self.out / name
        full = Path(f"{stem}.mp4")
        media.run([self.ffmpeg, "-y", "-v", "error", "-i", str(video), "-i", str(mix), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                   "-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart", "-shortest", str(full)])
        out = {"video": str(full), "format": self.s.get("format"), "width": self.W, "height": self.H}
        if self.preview:
            return out
        wants = set(self.s.get("exports") or ["video", "video_only", "audio", "thumbnail"])
        if "video_only" in wants:
            p = Path(f"{stem} (video only).mp4")
            media.run([self.ffmpeg, "-y", "-v", "error", "-i", str(full), "-map", "0:v", "-c", "copy", "-movflags", "+faststart", str(p)])
            out["video_only"] = str(p)
        if "audio" in wants:
            pw, pm = Path(f"{stem} (audio).wav"), Path(f"{stem} (audio).mp3")
            shutil.copy(mix, pw)
            media.run([self.ffmpeg, "-y", "-v", "error", "-i", str(mix), "-c:a", "libmp3lame", "-b:a", "320k", str(pm)])
            out["audio_wav"], out["audio_mp3"] = str(pw), str(pm)
        if (self.out / "captions.srt").exists():
            out["captions_srt"] = str(self.out / "captions.srt")
            out["captions_vtt"] = str(self.out / "captions.vtt")
        if "thumbnail" in wants:
            out["thumbnail"] = self.thumbnail(full)
        return out

    def thumbnail(self, video):
        """Best frame: highest-scoring shot with faces, sharp, mid-shot; adds the title."""
        shots = self.plan["shots"]
        best_t, best_s, t = 1.0, -1, 0.0
        for s in shots:
            sc = s.get("score", 0) + (0.2 if s["section"] in ("hook", "peak") else 0)
            if sc > best_s and s["length"] >= 0.4:
                best_s, best_t = sc, t + s["length"] / 2
            t += s["length"]
        p = Path(str(video).replace(".mp4", " (thumbnail).jpg"))
        vf = "scale=iw:ih"
        title = self.s.get("title")
        if title:
            fs = int(min(self.W, self.H) * 0.09)
            vf += (f",drawbox=y=ih*0.62:w=iw:h=ih*0.38:color=black@0.35:t=fill,drawtext=fontfile='{fpath(_font())}':text='{ftext(title.upper())}'"
                   f":fontsize={fs}:fontcolor={_hex(self.primary)}:x=(w-tw)/2:y=h*0.72:borderw=3:bordercolor=black@0.4")
        media.run([self.ffmpeg, "-y", "-v", "error", "-ss", f"{best_t:.2f}", "-i", str(video), "-frames:v", "1", "-vf", vf, "-q:v", "2", str(p)])
        return str(p)
