"""EventCut Engine for Windows: runs the local engine in the background with a system-tray icon."""
import os
import sys
import threading
import webbrowser
from pathlib import Path

from eventcut import config, server


def site_url():
    for base in (Path(getattr(sys, "_MEIPASS", "")), Path(sys.executable).parent, Path(__file__).parent):
        f = base / "site.txt"
        if f.exists():
            return f.read_text("utf-8").strip()
    return "https://eventcut.netlify.app"


def ensure_origin(url):
    cfg = config.load_settings()
    origins = set(cfg.get("allowed_origins", []))
    if url and url not in origins:
        cfg["allowed_origins"] = sorted(origins | {url.rstrip("/")})
        config.save_settings(cfg)


def tray(url):
    try:
        import pystray
        from PIL import Image
    except Exception:
        return False
    icon_path = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "eventcut.ico"
    image = Image.open(icon_path) if icon_path.exists() else Image.new("RGB", (64, 64), (139, 92, 255))

    def open_site(icon=None, item=None):
        webbrowser.open(url + "/app.html")

    def open_folder(icon=None, item=None):
        os.startfile(str(config.HOME)) if hasattr(os, "startfile") else webbrowser.open(str(config.HOME))

    def quit_app(icon, item):
        icon.stop()
        os._exit(0)

    menu = pystray.Menu(
        pystray.MenuItem("Open EventCut", open_site, default=True),
        pystray.MenuItem("Open the EventCut folder", open_folder),
        pystray.MenuItem(f"Engine {config.VERSION} is running", None, enabled=False),
        pystray.MenuItem("Quit", quit_app),
    )
    pystray.Icon("EventCut", image, "EventCut Engine", menu).run()
    return True


def start_log():
    """The tray app has no console: keep everything the engine prints in EventCut/Logs/engine.log (for support)."""
    try:
        logs = config.HOME / "Logs"
        logs.mkdir(parents=True, exist_ok=True)
        f = logs / "engine.log"
        if f.exists() and f.stat().st_size > 5 * 1024 * 1024:
            os.replace(f, logs / "engine.old.log")
        out = open(f, "a", encoding="utf-8", buffering=1)
        out.write(f"\n===== EventCut Engine {config.VERSION} started {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}\n")
        sys.stdout = sys.stderr = out
        import faulthandler
        faulthandler.enable(out)  # even a hard crash leaves a trace
    except Exception:
        pass


def main():
    start_log()
    url = site_url()
    ensure_origin(url)
    t = threading.Thread(target=server.serve, daemon=True)
    t.start()
    if "--no-tray" in sys.argv or not tray(url):
        t.join()


if __name__ == "__main__":
    main()
