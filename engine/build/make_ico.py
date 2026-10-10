"""Turns the site's icon PNG into a Windows .ico for the engine and its tray icon."""
from pathlib import Path
from PIL import Image
src = Path(__file__).resolve().parents[2] / "web" / "icons" / "icon-512.png"
Image.open(src).save(Path(__file__).resolve().parents[1] / "eventcut.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
