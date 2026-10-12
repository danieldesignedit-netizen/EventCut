"""Rendering: shot list -> graded, reframed segments -> transitions -> titles/logo/captions -> mixed audio -> exports."""
import json
import math
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np

from . import audio, fxtrans, grade, media
from .config import HOME, tools

FORMATS = {"9:16": (1080, 1920), "16:9": (1920, 1080), "1:1": (1080, 1080), "4:5": (1080, 1350)}
# transition name -> (FFmpeg xfade style, duration in seconds)
XFADE = {
    "whip_l": ("smoothleft", 0.24), "whip_r": ("smoothright", 0.24), "whip_u": ("smoothup", 0.24), "whip": ("smoothleft", 0.24),
    "blurwhip": ("hblur", 0.3), "slide_l": ("slideleft", 0.3), "zoom": ("zoomin", 0.32),
    "flash": ("fadewhite", 0.2), "dip": ("fadeblack", 0.4), "dissolve": ("fade", 0.5),
    # camera-style moves drawn by fxtrans (built-in look used only if a shot is too short)
    "tilt": ("zoomin", 0.6), "zoom_in": ("zoomin", 0.4), "zoom_out": ("fade", 0.5), "pan_l": ("smoothleft", 0.44), "pan_u": ("smoothup", 0.44),
}
# which sound family goes with each transition
# Caption looks. size_v/size_h: font size as a share of the short side for tall/wide videos.
CAPTION_STYLES = {
    "bold":    {"font": "Poppins", "bold": True, "upper": True, "size_v": 0.12, "size_h": 0.08, "outline": 0.075, "shadow": 0, "border": 1, "words": 3, "max_s": 2.2, "highlight": True},
    "clean":   {"font": "Poppins Medium", "bold": False, "upper": False, "size_v": 0.085, "size_h": 0.06, "outline": 0.04, "shadow": 2, "border": 1, "words": 5, "max_s": 2.8, "highlight": False},
    "boxed":   {"font": "Poppins", "bold": True, "upper": False, "size_v": 0.085, "size_h": 0.058, "outline": 0.16, "shadow": 0, "border": 3, "words": 4, "max_s": 2.6, "highlight": True, "outline_col": "&H40000000"},
    "karaoke": {"font": "Poppins", "bold": True, "upper": True, "size_v": 0.105, "size_h": 0.072, "outline": 0.07, "shadow": 0, "border": 1, "words": 4, "max_s": 2.6, "highlight": True},
    "neon":    {"font": "Poppins", "bold": True, "upper": True, "size_v": 0.11, "size_h": 0.075, "outline": 0.05, "shadow": 0, "border": 1, "words": 3, "max_s": 2.2, "highlight": True, "blur": 2},
    "minimal": {"font": "Poppins Medium", "bold": False, "upper": False, "size_v": 0.06, "size_h": 0.045, "outline": 0.0, "shadow": 2, "border": 1, "words": 7, "max_s": 3.2, "highlight": False},
}

TRANS_SFX = {"whip_l": "whoosh", "whip_r": "whoosh", "whip_u": "whoosh", "whip": "whoosh", "blurwhip": "whoosh",
             "slide_l": "whoosh", "zoom": "impact", "flash": "impact", "dip": "reverse", "dissolve": None,
             "tilt": "whoosh", "zoom_in": "whoosh", "zoom_out": "reverse", "pan_l": "whoosh", "pan_u": "whoosh"}


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


LOGO_EXT = {".png", ".webp", ".gif", ".mov", ".webm", ".jpg", ".jpeg"}


def logo_bytes_ok(data: bytes, name: str) -> bool:
    """True when the bytes really are a logo FFmpeg can place (not an error page saved as .png)."""
    ext = Path(name).suffix.lower()
    if ext not in LOGO_EXT or len(data) < 32:
        return False
    if ext == ".gif":
        return data[:4] == b"GIF8"
    if ext == ".mov":
        return b"ftyp" in data[:16] or b"moov" in data[:4096] or b"mdat" in data[:4096] or b"wide" in data[:16]
    if ext == ".webm":
        return data[:4] == b"\x1a\x45\xdf\xa3"
    try:
        import cv2
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        return img is not None and img.size > 0
    except Exception:
        # no decoder available: fall back to signatures
        return data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff" or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")


_logo_checked = {}


def logo_path_ok(p: Path) -> bool:
    try:
        key = (str(p), p.stat().st_mtime, p.stat().st_size)
        if key not in _logo_checked:
            with open(p, "rb") as f:
                head = f.read(8192) if p.suffix.lower() in (".mov", ".webm", ".gif") else f.read()
            _logo_checked[key] = logo_bytes_ok(head, p.name)
        return _logo_checked[key]
    except OSError:
        return False


def _rgb(h):
    h = (h or "#FFFFFF").strip().lstrip("#")
    h = "".join(c * 2 for c in h) if len(h) == 3 else h
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def fx_leak(work: Path, style: int = 0, colors=None) -> Path:
    """A 0.9 s light-leak flare with transparency (generated, so it's original and free to use).

    Its colours come from the brand's hex codes; without them it's a neutral warm white, never a preset hue."""
    cols = [c for c in (colors or []) if c and c.upper() not in ("#FFFFFF", "#000000")]
    tag = "-".join(c.strip("#") for c in cols) or "neutral"
    out = work / f"leak_{style}_{tag}.mov"
    if out.exists():
        return out
    W, H, fps, n = 480, 270, 25, 23
    if cols:
        a_, b_ = _rgb(cols[style % len(cols)]), _rgb(cols[(style + 1) % len(cols)])
        # lighten the brand colours so the flare reads as light, not paint
        pal = [[min(255, int(v * 0.75 + 255 * 0.25)) for v in a_], [min(255, int(v * 0.75 + 255 * 0.25)) for v in b_]]
    else:
        pal = [(255, 236, 214), (255, 214, 170)]
    pal = np.array(pal, np.float32) / 255
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ff, _ = tools()
    p = __import__("subprocess").Popen([ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                                        "-c:v", "qtrle", str(out)], stdin=__import__("subprocess").PIPE)
    for i in range(n):
        t = i / (n - 1)
        env = np.sin(np.pi * t) ** 1.2
        cx = W * (-0.3 + 1.6 * t) if style % 2 == 0 else W * (1.3 - 1.6 * t)
        # a soft vertical band of light sweeping across, brighter at the top like a lens flare
        band = np.exp(-((xx - cx) / (W * 0.28)) ** 2)
        glow = np.exp(-((xx - cx * 0.8 - W * 0.1) / (W * 0.5)) ** 2) * np.exp(-((yy - H * 0.15) / (H * 0.9)) ** 2)
        vert = 0.55 + 0.45 * np.exp(-((yy - H * 0.1) / (H * 0.7)) ** 2)
        a = np.clip((band * vert * 0.55 + glow * 0.3) * env, 0, 0.7)
        rgb = pal[0] * band[..., None] + pal[1] * (glow * 0.9)[..., None] + 0.06
        rgb = np.clip(rgb, 0, 1)
        frame = np.concatenate([rgb, a[..., None]], axis=2)
        p.stdin.write((frame * 255).astype(np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    return out


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
        ai = (self.s.get("ai_colour") or {}).get(src)  # the Google AI colourist's notes for this clip
        if ai:
            ai = {k: v for k, v in ai.items() if k != "note"}
        return grade.lut_for(log_profile=profile, correction=corr, preset_name=self.s.get("preset", "Vibrant Cinematic"),
                             intensity=float(self.s.get("intensity", 1.0)), transfer=transfer,
                             theme_hex=self.primary if self.s.get("theme_tint") else None,
                             custom_cube=custom if custom and Path(custom).exists() else None, ai=ai or None)

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
            args += media.decode_args() + ["-ss", f"{shot['in']:.3f}", "-t", f"{length:.3f}", "-i", src]
            if media.is_hdr(info):
                vf.append("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
            sw, sh = info.get("width", W), info.get("height", H)
            # slow-motion for high-frame-rate clips in emotional/close sections
            src_fps = float(info.get("fps") or 25)
            slow = (shot["section"] in ("emotional", "close") or (self.plan.get("slowmo") and shot["section"] == "peak" and shot.get("score", 0) > 0.75)) \
                and self.plan.get("slowmo", True) and src_fps >= 48 and not self.preview and not shot.get("layouts")
            if slow:
                factor = min(2.0, src_fps / self.fps)
                args[args.index("-t") + 1] = f"{length / factor:.3f}"
                vf.append(f"setpts={factor:.3f}*PTS")
            m = a.get("moments", [{}])[shot.get("moment", 0)] if a.get("moments") else {}
            vf += self.framing(shot, a, sw, sh, length)
            if 0.012 < m.get("shake", 0) <= 0.035 and not self.preview:
                # stabilise after the reframe: same result, far less work on 4K footage
                vf.append("deshake=rx=32:ry=32")
            vf.append(f"fps={self.fps}")
        vf.append(f"lut3d=file='{fpath(lut)}'")
        if self.s.get("preset", "Vibrant Cinematic") in ("Vibrant Cinematic", "Cinematic Teal & Orange", "Warm Celebration") and not self.preview:
            # soft highlight glow (halation) for the cinematic look
            # computed at quarter resolution for speed
            vf.append(f"split[a][b];[b]scale={W // 4}:{H // 4},gblur=sigma={max(3, W // 360)},curves=all='0/0 0.7/0.05 1/0.8',scale={W}:{H}[g];[a][g]blend=all_mode=screen:all_opacity=0.22")
        if self.s.get("grain", True) and not self.preview and self.s.get("preset") != "None":
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
        media.run_decoding(args)
        return out.with_suffix(".mov")

    # ------------------------------------------------------------- framing
    def focus_track(self, shot, a, length):
        """Where the subject is during this shot: faces first, then motion. Returns (start, end, faces_span_width)."""
        from .analyze import FPS
        frames = a.get("frames", [])
        i0, i1 = int(shot["in"] * FPS), max(int(shot["in"] * FPS) + 1, int((shot["in"] + length) * FPS))
        fr = frames[i0:i1] or frames[:1]
        faces = [f for f in fr if f.get("focus") == "faces"]
        pool = faces if len(faces) >= max(1, len(fr) // 4) else [f for f in fr if f.get("focus") == "motion"]
        if not pool:
            return (0.5, 0.42), (0.5, 0.42), 0.0
        half = max(1, len(pool) // 2)
        p0 = (float(np.median([f["fx"] for f in pool[:half]])), float(np.median([f["fy"] for f in pool[:half]])))
        p1 = (float(np.median([f["fx"] for f in pool[half:] or pool])), float(np.median([f["fy"] for f in pool[half:] or pool])))
        spans = [f["span"][1] - f["span"][0] for f in faces if f.get("span")]
        return p0, p1, float(np.median(spans)) if spans else 0.0

    def framing(self, shot, a, sw, sh, length):
        """Smart reframe for the output shape.

        - Faces (or, without faces, the action) are kept in frame on both axes, with headroom.
        - If the subject moves, the crop pans smoothly with it.
        - Vertical footage going into a wide frame (or a group too wide to crop) is placed whole
          over a blurred, enlarged copy of itself, the way broadcast editors handle phone video."""
        W, H = self.W, self.H
        mode = self.s.get("framing", "auto")
        if shot.get("layouts") and mode == "auto":
            lay = shot["layouts"].get("v" if H / W > 1.05 or abs(H - W) < 4 else "h")
            if lay:
                return self.layout_filters(lay, sw, sh)
            if W / H > 1.3 and sw / sh > 1.3:  # landscape into landscape: the full wide shot
                return [f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos", f"crop={W}:{H}", f"fps={self.fps}"]
        src_ar, dst_ar = sw / sh, W / H
        p0, p1, span = self.focus_track(shot, a, length)
        scale = max(W / sw, H / sh)
        cw, ch = min(sw, W / scale), min(sh, H / scale)
        fit = mode == "fit" or (mode == "auto" and (
            (src_ar < 0.9 and dst_ar > 1.15) or (dst_ar >= 1.0 and span > 0 and span * sw > cw * 0.92)))
        if fit and abs(src_ar - dst_ar) > 0.05:
            return [f"split=2[bga][fga];[bga]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                    f"gblur=sigma={max(20, W // 40)},eq=brightness=-0.10:saturation=1.15[bgb];"
                    f"[fga]scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos[fgb];"
                    f"[bgb][fgb]overlay=(W-w)/2:(H-h)/2", f"fps={self.fps}"]

        def place(p):
            x = min(max(0.0, p[0] * sw - cw / 2), sw - cw)
            y = min(max(0.0, p[1] * sh - ch * 0.42), sh - ch)  # faces sit a little above centre
            return x, y
        x0, y0 = place(p0)
        x1, y1 = place(p1)
        moving = abs(x1 - x0) > cw * 0.08 or abs(y1 - y0) > ch * 0.08
        if moving:
            d = max(0.1, length)
            ease = f"(0.5-0.5*cos(PI*min(t/{d:.3f},1)))"  # smooth start and stop
            xs, ys = f"'{x0:.1f}+({x1 - x0:.1f})*{ease}'", f"'{y0:.1f}+({y1 - y0:.1f})*{ease}'"
        else:
            xs, ys = f"{(x0 + x1) / 2:.1f}", f"{(y0 + y1) / 2:.1f}"
        out = [f"crop={int(cw) // 2 * 2}:{int(ch) // 2 * 2}:{xs}:{ys}", f"scale={W}:{H}:flags=lanczos", f"fps={self.fps}"]
        # slow push-in on longer, calmer shots (varies with every regenerate)
        import zlib
        rng = np.random.default_rng(zlib.crc32(f"{self.s.get('seed', 0)}|{shot['src']}|{shot['in']:.2f}".encode()))
        if not self.preview and length >= 1.2 and shot["section"] in ("build", "emotional", "close", "hook") and rng.random() < 0.7:
            z = 0.05 + 0.05 * rng.random()
            out += [f"scale=w='trunc({W}*(1+{z:.3f}*t/{length:.3f})/2)*2':h=-2:eval=frame:flags=bicubic", f"crop={W}:{H}"]
        return out

    def _face_crop(self, box, sw, sh, out_w, out_h, face_frac, eye_line):
        """Crop (x, y, w, h) in source pixels around one face, at the output's shape, the face filling face_frac
        of the width and its centre eye_line of the way down (headroom like a camera operator would leave)."""
        cx, cy, fw = box[0] * sw, box[1] * sh, max(box[2] * sw, 8)
        ar = out_w / out_h
        cw = fw / face_frac
        cw = max(cw, out_w / 2.2, fw * 1.9)          # never enlarge a small image more than 2.2x
        ch = cw / ar
        if ch > sh:
            ch, cw = sh, sh * ar
        if cw > sw:
            cw, ch = sw, sw / ar
        x = min(max(0.0, cx - cw / 2), sw - cw)
        y = min(max(0.0, cy - ch * eye_line), sh - ch)
        return int(x) // 2 * 2, int(y) // 2 * 2, int(cw) // 2 * 2, int(ch) // 2 * 2

    def layout_filters(self, lay, sw, sh):
        """Podcast framings: the speaker in close-up, all hosts stacked, a punch-in, or the wide shot."""
        W, H, fps = self.W, self.H, f"fps={self.fps}"
        mode = lay.get("mode")
        if mode == "speaker":
            frac = (0.36 if H > W else 0.22) * (1.3 if lay.get("tight") else 1.0)  # tight = punch-in on the same speaker
            x, y, cw, ch = self._face_crop(lay["box"], sw, sh, W, H, frac, 0.34)
            return [f"crop={cw}:{ch}:{x}:{y}", f"scale={W}:{H}:flags=lanczos", fps]
        if mode == "punch":
            cw = sw / 1.45
            x, y, cw, ch = self._face_crop(lay["box"], sw, sh, W, H, max(0.08, lay["box"][2] * sw / cw), 0.40)
            return [f"crop={cw}:{ch}:{x}:{y}", f"scale={W}:{H}:flags=lanczos", fps]
        if mode == "stack":
            boxes = (lay["boxes"] if lay.get("order") == "given" else sorted(lay["boxes"], key=lambda b: b[0]))[:3]
            n = len(boxes)
            hs = [H // n // 2 * 2] * (n - 1)
            hs.append(H - sum(hs))
            parts = [f"split={n}" + "".join(f"[s{i}]" for i in range(n))]
            for i, (b, ph) in enumerate(zip(boxes, hs)):
                x, y, cw, ch = self._face_crop(b, sw, sh, W, ph, 0.24, 0.45)
                parts.append(f"[s{i}]crop={cw}:{ch}:{x}:{y},scale={W}:{ph}:flags=lanczos,setsar=1[t{i}]")
            chain = ";".join(parts) + ";" + "".join(f"[t{i}]" for i in range(n)) + f"vstack=inputs={n}"
            lines = ",".join(f"drawbox=x=0:y={sum(hs[:i + 1]) - 3}:w={W}:h=6:color=black@0.85:t=fill" for i in range(n - 1))
            return [chain + ("," + lines if lines else ""), fps]
        # wide: whole frame over a blurred copy (tall outputs) or simply scaled (wide outputs)
        if W / H > 1.3:
            return [f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos", f"crop={W}:{H}", fps]
        return [f"split=2[bga][fga];[bga]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                f"gblur=sigma={max(20, W // 40)},eq=brightness=-0.10:saturation=1.15[bgb];"
                f"[fga]scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos[fgb];"
                f"[bgb][fgb]overlay=(W-w)/2:(H-h)/2", fps]

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
        # xfade needs identical frame rates and timebases on both sides, so normalise every run first
        for k in range(len(run_files)):
            fc.append(f"[{k}:v]fps={self.fps},format=yuv420p,setsar=1,settb=AVTB,setpts=PTS-STARTPTS[n{k}]")
            fc.append(f"[{k}:a]aresample=48000,asetpts=PTS-STARTPTS[m{k}]")
        prev_v, prev_a = "n0", "m0"
        lengths = [float(media.probe(str(f))["duration"]) for f in run_files]
        offset = lengths[0]
        for k in range(1, len(run_files)):
            trans = runs[k][0][1]["transition"]
            kind, d = XFADE[trans]
            made = None
            if trans in fxtrans.CUSTOM:
                dd = fxtrans.frames_for(d, self.fps) / self.fps
                if lengths[k - 1] >= 2 * dd + 0.1 and lengths[k] >= 2 * dd + 0.1:
                    try:
                        made, _ = fxtrans.make(trans, self.ffmpeg, run_files[k - 1], lengths[k - 1], run_files[k], self.W, self.H, self.fps,
                                               self.work / f"trans_{k}.mov", seed=int(self.s.get("seed") or 0) + k)
                        d = dd
                    except Exception:
                        made = None
            offset -= d
            if made:
                ti = len(inputs) // 2
                inputs += ["-i", str(made)]
                fc.append(f"[{ti}:v]fps={self.fps},format=yuv420p,setsar=1,settb=AVTB,setpts=PTS-STARTPTS[tt{k}]")
                fc.append(f"[{prev_v}]trim=duration={offset:.4f},setpts=PTS-STARTPTS[pa{k}]")
                fc.append(f"[n{k}]trim=start={d:.4f},setpts=PTS-STARTPTS[nb{k}]")
                fc.append(f"[pa{k}][tt{k}][nb{k}]concat=n=3:v=1:a=0[v{k}]")
            else:
                fc.append(f"[{prev_v}][n{k}]xfade=transition={kind}:duration={d}:offset={offset:.3f}[v{k}]")
            fc.append(f"[{prev_a}][m{k}]acrossfade=d={d:.4f}:c1=tri:c2=tri[a{k}]")
            prev_v, prev_a = f"v{k}", f"a{k}"
            offset += lengths[k]
        out = self.work / "body.mov"
        media.run([self.ffmpeg, "-y", "-v", "error"] + inputs + ["-filter_complex", ";".join(fc), "-map", f"[{prev_v}]", "-map", f"[{prev_a}]"]
                  + media.intermediate_args() + ["-c:a", "pcm_s16le", str(out)])
        return out

    # ------------------------------------------------------------- graphics
    def logo_file(self):
        """The chosen logo, else the most recently added readable one in Assets/Logos (images or animated logos)."""
        logo = self.s.get("logo")
        if logo and Path(logo).is_file() and logo_path_ok(Path(logo)):
            return Path(logo)
        folder = HOME / "Assets" / "Logos"
        files = [f for f in folder.glob("*") if f.is_file() and f.suffix.lower() in LOGO_EXT] if folder.exists() else []
        for f in sorted(files, key=lambda f: f.stat().st_mtime, reverse=True):
            if logo_path_ok(f):
                return f
        return None

    def logo_overlay(self, idx, base_label, out_label):
        """Filter text that places the logo input #idx as a corner watermark."""
        W, H = self.W, self.H
        size = {"small": 0.10, "medium": 0.15, "large": 0.22}.get(self.s.get("logo_size", "medium"), 0.15)
        lw = int(min(W, H) * size) // 2 * 2
        mx, my = int(W * 0.05), int(H * (0.05 if H > W else 0.06))
        pos = self.s.get("logo_pos", "top-left")
        x = f"{mx}" if "left" in pos else f"W-w-{mx}"
        y = f"{my}" if "top" in pos else f"H-h-{int(H * (0.16 if H > W else 0.07))}"  # stay above social app buttons
        op = float(self.s.get("logo_opacity", 0.9))
        return f"[{idx}:v]scale={lw}:-1,format=rgba,colorchannelmixer=aa={op:.2f}[lg];[{base_label}][lg]overlay={x}:{y}:shortest=1[{out_label}]"

    def end_card(self, dur):
        out = self.work / "endcard.mov"
        W, H = self.W, self.H
        logo = self.logo_file() if self.s.get("end_logo", True) else None
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
        """Captions in one of several styles, kept inside the platform safe zones.

        Styles: bold (word pops in the accent colour), clean, boxed, karaoke (words fill as spoken),
        neon (glowing accent text) and minimal (small lower-third subtitles)."""
        W, H = self.W, self.H
        tall = H > W
        name = self.s.get("caption_style", "bold")
        cfg = CAPTION_STYLES.get(name, CAPTION_STYLES["bold"])
        size = {"small": 0.82, "medium": 1.0, "large": 1.2}.get(self.s.get("caption_size", "medium"), 1.0)
        fs = int(min(W, H) * (cfg["size_v"] if tall else cfg["size_h"]) * size)
        pos = self.s.get("caption_position", "bottom")
        align = {"top": 8, "middle": 5}.get(pos, 2)
        if pos == "top":
            margin_v = int(H * (0.14 if tall else 0.07))  # below the platform's top bar
        elif pos == "middle":
            margin_v = 0
        else:
            margin_v = int(H * (0.24 if tall else 0.08))  # above TikTok/Reels/Shorts buttons and captions
        accent = self.s.get("caption_color") or self.accent
        acc = _ass_color(accent)
        white = "&H00FFFFFF"
        upper = cfg["upper"]
        font = cfg["font"]
        bold = -1 if cfg["bold"] else 0
        outline, shadow, border = max(1, int(fs * cfg["outline"])), cfg["shadow"], cfg["border"]
        primary = acc if name == "neon" else white
        outline_col = cfg.get("outline_col") or "&H00000000"
        if name == "neon":
            outline_col = _ass_color(accent, 0x30)
        style = (f"Style: Cap,{font},{fs},{primary},{acc},{outline_col},&H64000000,{bold},0,0,0,100,100,{cfg.get('spacing', 0)},0,"
                 f"{border},{outline},{shadow},{align},{int(W * 0.08)},{int(W * 0.08)},{margin_v},1")
        per_line = cfg["words"] if tall else cfg["words"] + 2
        lines, cur = [], []
        for w in words_timeline:
            if cur and (len(cur) >= per_line or w["start"] - cur[-1]["end"] > 0.6 or w["start"] - cur[0]["start"] > cfg["max_s"]
                        or cur[-1]["word"].strip()[-1:] in ".?!"):
                lines.append(cur)
                cur = []
            cur.append(w)
        if cur:
            lines.append(cur)

        def ts(t):
            t = max(0, t)
            return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"

        def word(t):
            t = t.strip()
            return t.upper() if upper else t
        ev = []
        for li, ln in enumerate(lines):
            # a line never lingers into the next one (two lines on screen at once made captions jump)
            nxt_start = lines[li + 1][0]["start"] if li + 1 < len(lines) else None
            end_all = ln[-1]["end"] + 0.15
            if nxt_start is not None:
                end_all = max(ln[-1]["end"], min(end_all, nxt_start - 0.01))
            if name == "karaoke":
                # one event per line; each word fills from white to the accent colour as it is spoken
                parts = []
                for k, w in enumerate(ln):
                    nxt = ln[k + 1]["start"] if k + 1 < len(ln) else w["end"]
                    parts.append(f"{{\\kf{max(1, int(round((nxt - w['start']) * 100)))}}}{word(w['word'])}")
                ev.append(f"Dialogue: 0,{ts(ln[0]['start'])},{ts(end_all)},Cap,,0,0,0,,{{\\1c{acc}&\\2c{white}&\\fad(60,60)}}" + " ".join(parts))
                continue
            if not cfg["highlight"]:
                ev.append(f"Dialogue: 0,{ts(ln[0]['start'])},{ts(end_all)},Cap,,0,0,0,,{{\\fad(100,80)}}" + " ".join(word(w["word"]) for w in ln))
                continue
            for k, w in enumerate(ln):  # one event per word: the line, with the spoken word highlighted
                start = ln[0]["start"] if k == 0 else w["start"]  # no gap before the first word lights up
                end = ln[k + 1]["start"] if k + 1 < len(ln) else end_all
                if end <= start:
                    continue
                parts = []
                for j2, ww in enumerate(ln):
                    t = word(ww["word"])
                    if j2 == k:
                        if name == "boxed":
                            parts.append(f"{{\\1c{acc}&}}{t}{{\\1c{white}&}}")
                        elif name == "neon":
                            parts.append(f"{{\\1c{white}&\\blur1}}{t}{{\\1c{acc}&\\blur{cfg.get('blur', 0)}}}")
                        else:  # colour only: changing size made the whole line shift on every word
                            parts.append(f"{{\\1c{acc}&}}{t}{{\\1c{white}&}}")
                    else:
                        parts.append(t)
                fx = "{\\fad(80,0)}" if k == 0 else ""
                if cfg.get("blur"):
                    fx += f"{{\\blur{cfg['blur']}}}"
                ev.append(f"Dialogue: 0,{ts(start)},{ts(end)},Cap,,0,0,0,,{fx}{' '.join(parts)}")
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
                    out.append({"word": w["word"], "start": round(t + w["start"] - s["in"], 3), "end": round(t + w["end"] - s["in"], 3),
                                "src": s["src"], "s0": w["start"]})  # where it came from, so corrections can be remembered
            t += s["length"]
        return out

    def finish_video(self, body, end, captions):
        W, H = self.W, self.H
        out = self.work / "video.mp4"
        logo = self.logo_file()
        # input 1 is the end card; without one, a tiny placeholder keeps input numbering the same
        args = [self.ffmpeg, "-y", "-v", "error", "-i", str(body)] + (["-i", str(end)] if end is not None else ["-f", "lavfi", "-t", "0.1", "-i", "color=c=black:s=16x16"])
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
        label, nxt = "b", 2
        # light leaks: warm flares over a few key moments
        for k, lk in enumerate(self.plan.get("leaks", []) if self.s.get("light_leaks", True) and not self.preview else []):
            leak = fx_leak(self.work, lk.get("style", 0), self.s.get("theme") or [])
            args += ["-i", str(leak)]
            fc += (f";[{nxt}:v]scale={W}:{H},format=rgba,setpts=PTS-STARTPTS+{lk['at']:.3f}/TB[lk{k}];"
                   f"[{label}][lk{k}]overlay=eof_action=pass:format=auto[lo{k}]")
            label, nxt = f"lo{k}", nxt + 1
        if logo and self.s.get("watermark", True):
            loop = ["-stream_loop", "-1"] if logo.suffix.lower() in (".mov", ".webm", ".gif") else ["-loop", "1"]
            args += loop + ["-i", str(logo)]
            fc += ";" + self.logo_overlay(nxt, label, "bb")
            label, nxt = "bb", nxt + 1
        if end is not None:
            fc += f";[{label}][1:v]concat=n=2:v=1:a=0[v]"
        else:
            fc += f";[{label}]null[v]"
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
                f = Path(name)
                if not f.exists():
                    f = sfx_dir / f"EventCut {name}.wav"
                if not f.exists():
                    hits = sorted(sfx_dir.glob(f"EventCut {name}*.wav"))
                    f = hits[0] if hits else f
                r = media.run([self.ffmpeg, "-v", "error", "-i", str(f), "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"])
                cache[name] = np.frombuffer(r.stdout, dtype=np.float32).reshape(-1, 2)
            return cache[name]
        placements = list(self.s.get("sfx") or [])
        if self.s.get("auto_sfx", True):
            fams = audio.sfx_families()
            import zlib
            rng = np.random.default_rng(zlib.crc32(f"sfx|{self.s.get('seed', 0)}".encode()))
            for t, kind in transitions_at:
                fam = TRANS_SFX.get(kind)
                if not fam or not fams.get(fam):
                    continue
                path = str(rng.choice(fams[fam]))
                lead = 0.25 if fam in ("whoosh", "reverse") else 0.0
                placements.append({"name": path, "at": max(0, t - lead), "gain": 0.45 if fam == "impact" else 0.35})
        self.used_sfx = [pl["name"] for pl in placements if Path(str(pl["name"])).exists()]
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
        end = None
        if self.s.get("end_card", True) and self.s.get("ending") != "none":
            end_len = max(1.5, float(self.s.get("duration", 60)) - body_len)
            if self.plan.get("shortened"):  # not enough footage: keep a normal end card, don't stretch it
                end_len = max(1.5, float(self.plan.get("end_card") or 2.5))
            end = self.end_card(end_len)
        captions = None
        if self.s.get("captions") and transcripts:
            words = self.words_on_timeline(transcripts)
            if words:
                captions = self.captions_ass(words)
                self.write_srt(words)
                (self.out / "captions.json").write_text(json.dumps(words), "utf-8")  # for the caption editor
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
            p = Path(f"{stem}_video-only.mp4")
            media.run([self.ffmpeg, "-y", "-v", "error", "-i", str(full), "-map", "0:v", "-c", "copy", "-movflags", "+faststart", str(p)])
            out["video_only"] = str(p)
        if "audio" in wants:
            pw, pm = Path(f"{stem}_audio.wav"), Path(f"{stem}_audio.mp3")
            shutil.copy(mix, pw)
            media.run([self.ffmpeg, "-y", "-v", "error", "-i", str(mix), "-c:a", "libmp3lame", "-b:a", "320k", str(pm)])
            out["audio_wav"], out["audio_mp3"] = str(pw), str(pm)
        if (self.out / "captions.srt").exists():
            out["captions_srt"] = str(self.out / "captions.srt")
            out["captions_vtt"] = str(self.out / "captions.vtt")
        if "thumbnail" in wants:
            out["thumbnail"] = self.thumbnail(full)
        # credits for any library music or effects that need attribution (CC BY)
        used = [p for p in [(self.s.get("music") or {}).get("path")] + list(getattr(self, "used_sfx", [])) if p]
        credits = audio.credits_for(used)
        if credits:
            cf = Path(f"{stem}_music-credits.txt")
            cf.write_text("Music and sound credits (please include these in the post caption or description):\n" + credits + "\n", "utf-8")
            out["credits"] = str(cf)
        if (self.s.get("music") or {}).get("title"):
            out["music"] = (self.s.get("music") or {}).get("title")
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
        p = Path(str(video)[:-4] + "_thumbnail.jpg")
        vf = "scale=iw:ih"
        title = self.s.get("title")
        if title:
            fs = int(min(self.W, self.H) * 0.09)
            vf += (f",drawbox=y=ih*0.62:w=iw:h=ih*0.38:color=black@0.35:t=fill,drawtext=fontfile='{fpath(_font())}':text='{ftext(title.upper())}'"
                   f":fontsize={fs}:fontcolor={_hex(self.primary)}:x=(w-tw)/2:y=h*0.72:borderw=3:bordercolor=black@0.4")
        media.run([self.ffmpeg, "-y", "-v", "error", "-ss", f"{best_t:.2f}", "-i", str(video), "-frames:v", "1", "-vf", vf, "-q:v", "2", str(p)])
        return str(p)
