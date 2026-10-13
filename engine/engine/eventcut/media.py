"""Thin wrappers around ffprobe/ffmpeg: probing, proxies, frames, audio envelopes."""
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import tools

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".mts", ".m2ts", ".mxf", ".webm", ".3gp", ".m4v"}
PHOTO_EXT = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tif", ".tiff"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".aiff", ".aif"}
RAW_EXT = {".braw", ".r3d", ".ari", ".arx"}

CREATE_NO_WINDOW = 0x08000000


def run(cmd, check=True, capture=True, timeout=None):
    kw = {}
    import platform
    if platform.system() == "Windows":
        kw["creationflags"] = CREATE_NO_WINDOW
    p = subprocess.run(cmd, stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.PIPE if capture else None, timeout=timeout, **kw)
    if check and p.returncode != 0:
        err = (p.stderr or b"").decode("utf-8", "replace")[-1500:]
        raise RuntimeError(f"Command failed: {Path(cmd[0]).name}\n{err}")
    return p


def classify(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in VIDEO_EXT:
        return "video"
    if ext in PHOTO_EXT:
        return "photo"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in RAW_EXT:
        return "raw"
    return "other"


def scan(paths):
    """Expand files/folders into a sorted media list with a plain reason for skips."""
    found, skipped = [], []
    for p in paths:
        p = Path(p)
        items = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
        for f in items:
            kind = classify(f)
            if f.name.startswith("."):
                continue
            if kind == "raw":
                skipped.append({"path": str(f), "reason": "Camera RAW video isn't supported yet. Export it to ProRes or H.265 first."})
            elif kind in ("video", "photo", "audio"):
                found.append({"path": str(f), "kind": kind, "size": f.stat().st_size})
    return found, skipped


@lru_cache(maxsize=512)
def probe(path: str) -> dict:
    _, ffprobe = tools()
    p = run([ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    data = json.loads(p.stdout)
    v = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    info = {"duration": float(data["format"].get("duration", 0) or 0), "has_audio": a is not None,
            "tags": {k.lower(): v for k, v in (data["format"].get("tags") or {}).items()}}
    if v:
        w, h = int(v.get("width", 0)), int(v.get("height", 0))
        rot = 0
        for sd in v.get("side_data_list", []) or []:
            if "rotation" in sd:
                rot = int(sd["rotation"])
        if "rotate" in (v.get("tags") or {}):
            rot = int(v["tags"]["rotate"])
        if abs(rot) in (90, 270):
            w, h = h, w
        num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1").split("/")
        fps = float(num) / float(den) if float(den or 0) else 0.0
        info.update({"width": w, "height": h, "fps": round(fps, 3), "codec": v.get("codec_name"),
                     "color_transfer": v.get("color_transfer"), "color_primaries": v.get("color_primaries"),
                     "pix_fmt": v.get("pix_fmt"), "bit_depth": int(v.get("bits_per_raw_sample") or 8),
                     "vtags": {k.lower(): val for k, val in (v.get("tags") or {}).items()}})
    return info


def is_hdr(info: dict) -> bool:
    return info.get("color_transfer") in ("arib-std-b67", "smpte2084")


_HWDEC = {"ok": True}


def decode_args() -> list:
    """Let the graphics chip decode 4K/HEVC footage (much faster than the processor). Off again if it ever fails."""
    return ["-hwaccel", "auto"] if _HWDEC["ok"] else []


def run_decoding(cmd: list):
    """Run an FFmpeg command that uses decode_args(); retry on the processor if hardware decoding fails."""
    try:
        return run(cmd)
    except RuntimeError:
        if "-hwaccel" not in cmd:
            raise
        i = cmd.index("-hwaccel")
        plain = cmd[:i] + cmd[i + 2:]
        r = run(plain)
        _HWDEC["ok"] = False  # this PC's decoder had trouble: stay on the processor for this session
        return r


def make_proxy(src: str, dst: Path, height: int = 360, fps: int = 10) -> Path:
    """Small, fast working copy used for analysis and preview."""
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    ffmpeg, _ = tools()
    info = probe(src)
    vf = [f"fps={fps}", f"scale=-2:{height}:flags=fast_bilinear"]
    if is_hdr(info):
        vf.insert(0, "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
    cmd = [ffmpeg, "-y", "-v", "error"] + decode_args() + ["-i", src, "-vf", ",".join(vf), "-an",
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p", str(dst)]  # 8-bit: plays in any browser
    try:
        run_decoding(cmd)
    except RuntimeError:
        if is_hdr(info):  # zscale missing: fall back to plain scaling
            run([ffmpeg, "-y", "-v", "error", "-i", src, "-vf", f"fps={fps},scale=-2:{height}", "-an",
                 "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", str(dst)])
        else:
            raise
    return dst


def audio_envelope(src: str, rate: int = 50) -> np.ndarray:
    """RMS loudness envelope (dBFS) at `rate` values per second."""
    ffmpeg, _ = tools()
    sr = 8000
    p = run([ffmpeg, "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"], check=False)
    raw = np.frombuffer(p.stdout or b"", dtype=np.int16).astype(np.float32) / 32768.0
    hop = sr // rate
    n = len(raw) // hop
    if n == 0:
        return np.zeros(0)
    frames = raw[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-10)
    return 20 * np.log10(rms + 1e-10)


def load_audio(src: str, sr: int = 22050) -> np.ndarray:
    ffmpeg, _ = tools()
    p = run([ffmpeg, "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"])
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def scene_cuts(src: str, threshold: float = 0.32) -> list:
    ffmpeg, _ = tools()
    p = run([ffmpeg, "-v", "info", "-i", src, "-vf", f"select='gt(scene,{threshold})',showinfo", "-an", "-f", "null", "-"], check=False)
    return [float(x) for x in re.findall(r"pts_time:([0-9.]+)", (p.stderr or b"").decode("utf-8", "replace"))]


def has_encoder(name: str) -> bool:
    ffmpeg, _ = tools()
    p = run([ffmpeg, "-hide_banner", "-encoders"], check=False)
    return name in (p.stdout or b"").decode()


_HW = None


def _try_encoder(enc: str, extra=()) -> tuple:
    ffmpeg, _ = tools()
    p = run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=25:d=0.5", "-pix_fmt", "yuv420p",
             "-c:v", enc, *extra, "-f", "null", "-"], check=False, timeout=60)
    lines = (p.stderr or b"").decode("utf-8", "replace").strip().splitlines()
    import re as _re
    # the first line from the encoder itself explains the cause (e.g. "Driver does not support the required nvenc API version")
    useful = [_re.sub(r"^\[[^\]]+\]\s*", "", l) for l in lines if not l.startswith("[out#") and "Nothing was written" not in l]
    return p.returncode == 0, (useful[0] if useful else (lines[-1] if lines else ""))


def hardware() -> dict:
    """Find a working hardware video encoder: NVIDIA (NVENC), then AMD (AMF), then Intel (Quick Sync).

    Each one is test-encoded, because an encoder can be present in FFmpeg yet unusable
    (for example an NVIDIA driver that is too old for this FFmpeg build)."""
    global _HW
    if _HW is not None:
        return _HW
    notes = []
    for kind, h264, hevc, label in [("nvenc", "h264_nvenc", "hevc_nvenc", "NVIDIA GPU"),
                                    ("amf", "h264_amf", "hevc_amf", "AMD GPU"),
                                    ("qsv", "h264_qsv", "hevc_qsv", "Intel GPU")]:
        if not has_encoder(h264):
            continue
        ok, err = _try_encoder(h264)
        if ok:
            _HW = {"kind": kind, "h264": h264, "hevc": hevc, "label": label, "notes": notes}
            return _HW
        hint = ""
        if kind == "nvenc" and ("API" in err or "driver" in err.lower() or "nvenc" in err.lower()):
            hint = " Update the NVIDIA driver (NVIDIA app or nvidia.com/drivers) to the latest version."
        notes.append(f"{label}: {err[:160]}{hint}")
    _HW = {"kind": None, "label": "Processor only", "notes": notes}
    return _HW


def nvidia_ok() -> bool:
    return hardware().get("kind") == "nvenc"


def gpu_available() -> bool:
    return hardware().get("kind") is not None


def video_encoder_args(quality: str = "high", codec: str = "h264") -> list:
    hw = hardware()
    k = hw.get("kind")
    if k:
        enc = hw["hevc"] if codec == "h265" else hw["h264"]
        if k == "nvenc":
            return ["-c:v", enc, "-preset", "p5", "-rc", "vbr", "-cq", "19" if quality == "high" else "24", "-b:v", "0"]
        if k == "amf":
            q = "18" if quality == "high" else "24"
            return ["-c:v", enc, "-quality", "quality" if quality == "high" else "speed", "-rc", "cqp", "-qp_i", q, "-qp_p", q, "-qp_b", q]
        if k == "qsv":
            return ["-c:v", enc, "-preset", "slower" if quality == "high" else "fast", "-global_quality", "20" if quality == "high" else "26"]
    enc = "libx265" if codec == "h265" and has_encoder("libx265") else "libx264"
    return ["-c:v", enc, "-preset", "medium" if quality == "high" else "veryfast", "-crf", "18" if quality == "high" else "23"]


def intermediate_args() -> list:
    """Near-lossless, very fast codec for working files (only the final export is compressed properly)."""
    hw = hardware()
    if hw.get("kind") == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p2", "-rc", "constqp", "-qp", "14"]
    if hw.get("kind") == "amf":
        return ["-c:v", "h264_amf", "-quality", "speed", "-rc", "cqp", "-qp_i", "14", "-qp_p", "14", "-qp_b", "14"]
    if hw.get("kind") == "qsv":
        return ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "16"]
    return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "12"]
