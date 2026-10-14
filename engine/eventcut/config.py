"""EventCut Engine configuration and the local EventCut folder layout."""
import json
import os
import platform
import shutil
import sys
from pathlib import Path

VERSION = "1.7.0"
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


def load_settings() -> dict:
    ensure_folders()
    data = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            data.update(json.loads(SETTINGS_FILE.read_text("utf-8")))
        except Exception:
            pass
    return data


def save_settings(data: dict) -> None:
    ensure_folders()
    SETTINGS_FILE.write_text(json.dumps(data, indent=2), "utf-8")


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
