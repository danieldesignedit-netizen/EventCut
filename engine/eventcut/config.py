"""EventCut Engine configuration and the local EventCut folder layout."""
import json
import os
import platform
import shutil
import sys
from pathlib import Path

VERSION = "1.0.0"
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
SUBFOLDERS = ["Assets", "Presets", "Music & SFX", "Projects", "Exports", "Styles"]


def ensure_folders() -> None:
    for name in SUBFOLDERS:
        (HOME / name).mkdir(parents=True, exist_ok=True)
    for sub in ["Assets/Logos", "Assets/Fonts", "Presets/LUTs", "Music & SFX/Music", "Music & SFX/SFX", "Music & SFX/Uploads"]:
        (HOME / sub).mkdir(parents=True, exist_ok=True)


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


SETTINGS_FILE = HOME / "settings.json"
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
