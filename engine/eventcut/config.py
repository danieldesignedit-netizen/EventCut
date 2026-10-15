"""EventCut Engine configuration and the local EventCut folder layout."""
import json
import os
import platform
import shutil
import sys
from pathlib import Path

VERSION = "1.9.1"
PORT = int(os.environ.get("EVENTCUT_PORT", "7788"))

# Origins allowed to drive this engine. The installer writes the real Netlify URL
# into settings.json; localhost entries are for local development of the site.
DEFAULT_ORIGINS = [
    "https://eventcut.netlify.app",
    "http://localhost:8888",
    "http://127.0.0.1:8888",
    "http://localhost:5500",
]


def home_folder() -> Path:
    override = os.environ.get("EVENTCUT_HOME")
    if override:
        return Path(override)
    docs = Path.home() / "Documents"
    return (docs if docs.exists() else Path.home()) / "EventCut"


HOME = home_folder()
SETTINGS_FILE = HOME / "settings.json"

# One EventCut folder, with plain-language subfolders people can find their way around.
FINISHED = HOME / "Finished videos"        # every project and its versions
LOGOS = HOME / "Logos"
FONTS = HOME / "Fonts"
MUSIC = HOME / "Music"                      # drop your own licensed tracks straight in here
MUSIC_UPLOADS = MUSIC / "Your uploads"
MUSIC_LIBRARY = MUSIC / "Free library"
SFX = HOME / "Sound effects"
SFX_ORIGINALS = SFX / "EventCut originals"
SFX_LIBRARY = SFX / "Free library"
EPIDEMIC = HOME / "Epidemic Sound"
EPIDEMIC_MUSIC = EPIDEMIC / "Music"
EPIDEMIC_SFX = EPIDEMIC / "Sound effects"
LOOKS = HOME / "Looks"                      # LUTs (.cube) go straight in here
STYLES = LOOKS / "Styles"
LOGS = HOME / "Logs"
WORK = HOME / "Working files"               # EventCut's own caches: safe to ignore (or delete to free space)

FOLDERS = [FINISHED, LOGOS, FONTS, MUSIC, MUSIC_UPLOADS, MUSIC_LIBRARY, SFX, SFX_ORIGINALS, SFX_LIBRARY,
           EPIDEMIC, EPIDEMIC_MUSIC, EPIDEMIC_SFX, LOOKS, STYLES, LOGS, WORK]

# where things lived before version 1.6 (moved automatically, once)
OLD_LAYOUT = [
    ("Exports", FINISHED), ("Assets/Logos", LOGOS), ("Assets/Fonts", FONTS), ("Assets", WORK / "Brand kit"),
    ("Presets/LUTs", LOOKS), ("Presets", LOOKS), ("Styles", STYLES),
    ("Music & SFX/Epidemic Sound", EPIDEMIC_MUSIC), ("Music & SFX/Uploads", MUSIC_UPLOADS),
    ("Music & SFX/Library/Music", MUSIC_LIBRARY), ("Music & SFX/Library/SFX", SFX_LIBRARY), ("Music & SFX/Library", MUSIC_LIBRARY),
    ("Music & SFX/Music", MUSIC), ("Music & SFX/SFX", SFX_ORIGINALS), ("Music & SFX/_previews", WORK / "_previews_audio"),
    ("Music & SFX", WORK), ("Projects", WORK),
]
_migrated = [False]


def _merge_move(src: Path, dst: Path):
    """Move everything in src into dst (never overwriting), then remove src if it ended up empty."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in list(src.iterdir()):
        target = dst / item.name
        try:
            if target.exists():
                if item.is_dir() and target.is_dir():
                    _merge_move(item, target)
                continue  # keep what is already in the new place
            shutil.move(str(item), str(target))
        except Exception:
            pass
    try:
        src.rmdir()
    except OSError:
        pass


def _rewrite_paths(file: Path, pairs):
    try:
        txt = file.read_text("utf-8")
        new = txt
        for a, b in pairs:
            new = new.replace(a, b)
        if new != txt:
            file.write_text(new, "utf-8")
    except Exception:
        pass


def migrate_layout() -> None:
    """Move an older EventCut folder (Assets, Presets, Music & SFX, Projects, Exports) into the new layout.
    Saved projects and the analysis cache are updated to point at the new places."""
    if _migrated[0]:
        return
    _migrated[0] = True
    moved = []
    for old, new in OLD_LAYOUT:
        src = HOME / old
        if src.is_dir():
            moved.append((src, new))
            _merge_move(src, new)
    if not moved:
        return
    pairs = []
    for src, new in moved:
        for a, b in ((str(src), str(new)), (str(src).replace("\\", "\\\\"), str(new).replace("\\", "\\\\")),
                     (str(src).replace("\\", "/"), str(new).replace("\\", "/"))):
            if a != b and (a, b) not in pairs:
                pairs.append((a, b))
    pairs.sort(key=lambda ab: -len(ab[0]))  # most specific first (Music & SFX/Epidemic Sound before Music & SFX)
    for pj in FINISHED.glob("*/Version */project.json"):
        _rewrite_paths(pj, pairs)
    an = WORK / "_analysis"
    if an.is_dir():
        for f in an.glob("*.json"):
            _rewrite_paths(f, pairs)
    _rewrite_paths(SETTINGS_FILE, pairs)


def ensure_folders() -> None:
    migrate_layout()
    for d in FOLDERS:
        d.mkdir(parents=True, exist_ok=True)


def bin_path(name: str) -> str:
    """Find ffmpeg/ffprobe: bundled next to the engine first, then PATH."""
    exe = name + (".exe" if platform.system() == "Windows" else "")
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    for cand in [base / "bin" / exe, base / exe]:
        if cand.exists():
            return str(cand)
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} not found. Reinstall EventCut Engine.")
    return found


FFMPEG = None
FFPROBE = None


def tools():
    global FFMPEG, FFPROBE
    if FFMPEG is None:
        FFMPEG, FFPROBE = bin_path("ffmpeg"), bin_path("ffprobe")
    return FFMPEG, FFPROBE


DEFAULT_SETTINGS = {
    "allowed_origins": DEFAULT_ORIGINS,
    "supabase_url": "",
    "supabase_anon_key": "",
    "require_approval": True,
    "notify_sound": True,
    "notify_volume": 0.8,
    "director_mode_api_key": "",
    "director_mode_model": "claude-sonnet-5-5",
    "director_monthly_cap_usd": 20,
}


# Keys are stored locked to this Windows user on this PC (Windows DPAPI): settings.json shows unreadable text,
# and a copy of the file on another PC or another Windows account can't be unlocked.
SECRET_KEYS = ("google_api_key", "fal_api_key", "epidemic_api_key", "director_mode_api_key")
_ENC = "enc:"


def _dpapi(data: bytes, protect: bool) -> bytes:
    import ctypes
    from ctypes import wintypes

    class BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]
    buf = ctypes.create_string_buffer(data, len(data))
    blob_in, blob_out = BLOB(len(data), buf), BLOB()
    entropy = b"EventCut-engine-keys"
    ebuf = ctypes.create_string_buffer(entropy, len(entropy))
    blob_e = BLOB(len(entropy), ebuf)
    crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
    fn = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    ok = fn(ctypes.byref(blob_in), None, ctypes.byref(blob_e), None, None, 0x01, ctypes.byref(blob_out))  # 0x01: no UI
    if not ok:
        raise OSError("DPAPI failed")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(blob_out.pbData)


def _machine_secret() -> bytes:
    """Fallback lock (only off Windows, e.g. for development): tied to this user account and machine."""
    import getpass
    import hashlib
    import socket
    return hashlib.sha256(f"{socket.gethostname()}|{getpass.getuser()}|{Path.home()}|EventCut".encode()).digest()


def _xor(data: bytes, key: bytes) -> bytes:
    import hashlib
    stream, i = b"", 0
    while len(stream) < len(data):
        stream += hashlib.sha256(key + i.to_bytes(4, "big")).digest()
        i += 1
    return bytes(a ^ b for a, b in zip(data, stream))


def lock_secret(text: str) -> str:
    import base64
    if not text or text.startswith(_ENC):
        return text
    raw = ("EC1:" + text).encode("utf-8")  # the marker proves a key was unlocked correctly
    if platform.system() == "Windows":
        return _ENC + "w" + base64.b64encode(_dpapi(raw, True)).decode()
    return _ENC + "x" + base64.b64encode(_xor(raw, _machine_secret())).decode()


def unlock_secret(text: str) -> str:
    import base64
    if not isinstance(text, str) or not text.startswith(_ENC):
        return text or ""
    kind, body = text[len(_ENC)], base64.b64decode(text[len(_ENC) + 1:])
    try:
        raw = _dpapi(body, False) if kind == "w" else _xor(body, _machine_secret())
        out = raw.decode("utf-8")
    except Exception:
        return None
    return out[4:] if out.startswith("EC1:") else None  # None: copied from another PC or account, so it can't be read here


_UNREADABLE = {}  # keys that can't be unlocked here are left as they are in the file (never wiped by a save)


def load_settings() -> dict:
    ensure_folders()
    data = dict(DEFAULT_SETTINGS)
    stored = {}
    if SETTINGS_FILE.exists():
        try:
            stored = json.loads(SETTINGS_FILE.read_text("utf-8"))
            data.update(stored)
        except Exception:
            pass
    for k in SECRET_KEYS:
        v = unlock_secret(data.get(k) or "")
        if v is None:
            _UNREADABLE[k] = data.get(k)
            v = ""
        else:
            _UNREADABLE.pop(k, None)
        data[k] = v
    changed = False
    # one time: earlier versions could leave "pro" saved; Flash is the default (Pro is only chosen again on purpose)
    if not stored.get("flash_default_done"):
        if data.get("google_model") not in (None, "", "flash"):
            data["google_model"] = "flash"
        data["flash_default_done"] = True
        changed = True
    # keys saved in plain text by an earlier version are locked now
    if any(stored.get(k) and not str(stored[k]).startswith(_ENC) for k in SECRET_KEYS) and not _UNREADABLE:
        changed = True
    if changed and SETTINGS_FILE.exists():
        try:
            save_settings(data)
        except Exception:
            pass
    return data


def save_settings(data: dict) -> None:
    ensure_folders()
    out = dict(data)
    for k in SECRET_KEYS:
        if not out.get(k) and _UNREADABLE.get(k):
            out[k] = _UNREADABLE[k]
            continue
        if out.get(k):
            try:
                out[k] = lock_secret(out[k])
            except Exception:
                pass  # never lose a key because locking failed
    SETTINGS_FILE.write_text(json.dumps(out, indent=2), "utf-8")


def _json_default(o):
    """Let numpy numbers (float32, int64, arrays) be saved as JSON."""
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def install_json_numpy_support():
    import json as _json
    _orig = _json.dumps

    def dumps(obj, *a, **kw):
        kw.setdefault("default", _json_default)
        return _orig(obj, *a, **kw)
    _json.dumps = dumps
