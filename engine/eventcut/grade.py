"""Colour pipeline baked into one 3D LUT per clip:

    Log/flat -> normal colour  ->  auto correction (white balance, exposure for faces)
    -> optional "match this look" transfer -> grading preset (with intensity) -> hex-theme tint

LUTs are written as standard .cube files, so any custom .cube in Presets/LUTs also works.
"""
import colorsys
import hashlib
from pathlib import Path

import numpy as np

from .config import HOME, LOOKS, WORK

SIZE = 33
LUT_DIR = WORK / "_luts"


def grid(size=SIZE):
    r = np.linspace(0, 1, size)
    b, g, rr = np.meshgrid(r, r, r, indexing="ij")  # .cube order: red fastest
    return np.stack([rr, g, b], axis=-1).reshape(-1, 3)


def luma(x):
    return x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722


def clamp(x):
    return np.clip(x, 0, 1)


# ------------------------------------------------------------------ Log decoding
def _slog3_to_lin(x):
    return np.where(x >= 171.2102946929 / 1023, (10 ** ((x * 1023 - 420) / 261.5)) * (0.18 + 0.01) - 0.01,
                    (x * 1023 - 95) * 0.01125 / (171.2102946929 - 95))


def _vlog_to_lin(x):
    return np.where(x < 0.181, (x - 0.125) / 5.6, 10 ** ((x - 0.598206) / 0.241514) - 0.00873)


def _generic_log_to_lin(x):
    # Works well for most flat profiles (C-Log, F-Log, N-Log, D-Log, Apple Log, LogC) as a starting point
    return np.maximum(0, (10 ** ((x - 0.41) / 0.24)) * 0.18 - 0.004)


def filmic(lin):
    """Smooth highlight roll-off tone map from scene-linear to display."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    x = lin * 1.35
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e)) ** (1 / 2.2)


def decode_log(x, profile):
    if not profile:
        return x
    f = {"slog3": _slog3_to_lin, "vlog": _vlog_to_lin}.get(profile, _generic_log_to_lin)
    lin = f(x)
    out = filmic(lin)
    # restore saturation typical of a Rec.709 conversion
    y = luma(out)[..., None]
    return clamp(y + (out - y) * 1.06)


# ------------------------------------------------------------------ auto correction
def correction_params(frames: list, skin_target=0.47) -> dict:
    """White balance gains and exposure from analysis frames; exposes for faces when present."""
    if not frames:
        return {"gains": [1, 1, 1], "gamma": 1.0}
    rgb = np.median(np.array([f.get("rgb", [f["luma"]] * 3) for f in frames]), axis=0)
    gray = rgb.mean()
    gains = np.clip(gray / np.maximum(rgb, 1e-3), 0.85, 1.18)
    gains = 1 + (gains - 1) * 0.6  # partial: keep the mood of coloured stage light
    skins = [f["skin_luma"] for f in frames if f.get("skin_luma", -1) > 0]
    lum = float(np.median([f["luma"] for f in frames]))
    if skins:
        s = float(np.median(skins))
        gamma = float(np.clip(np.log(skin_target) / np.log(max(0.05, s)), 0.65, 1.35))
    else:
        gamma = float(np.clip(np.log(0.42) / np.log(max(0.05, lum)), 0.75, 1.25))
    return {"gains": [round(float(g), 4) for g in gains], "gamma": round(gamma, 4)}


def apply_correction(x, p):
    x = clamp(x * np.array(p["gains"]))
    g = p["gamma"]
    if abs(g - 1) > 0.01:
        # gamma applied to luminance only, protecting highlights (no clipping)
        y = luma(x)[..., None]
        y2 = clamp(y ** g)
        x = clamp(x * (y2 / np.maximum(y, 1e-4)))
    return x


# ------------------------------------------------------------------ preset building blocks
def s_curve(x, strength=0.5, pivot=0.45):
    x = clamp(x)
    k = 1 + strength * 4
    up = 1 / (1 + np.exp(-k * (x - pivot)))
    lo, hi = 1 / (1 + np.exp(k * pivot)), 1 / (1 + np.exp(-k * (1 - pivot)))
    return clamp((up - lo) / (hi - lo))


def saturate(x, amount, vibrance=0.0, protect_skin=0.0):
    y = luma(x)[..., None]
    chroma = x - y
    sat = np.abs(chroma).max(axis=-1, keepdims=True)
    amt = amount + vibrance * (1 - np.clip(sat * 3, 0, 1))
    if protect_skin:
        # hue near orange (skin): limit boost
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        skin = ((r > g) & (g > b) & ((r - b) > 0.05) & ((g - b) / np.maximum(r - b, 1e-4) < 0.75)).astype(float)[..., None]
        amt = amt * (1 - skin * protect_skin) + skin * protect_skin * (1 + (amount - 1) * 0.35)
    return clamp(y + chroma * amt)


def split_tone(x, shadows, highlights, amount=0.08, balance=0.5):
    y = luma(x)[..., None]
    w_s = clamp((balance - y) / balance)
    w_h = clamp((y - balance) / (1 - balance))
    return clamp(x + (np.array(shadows) - 0.5) * amount * 2 * w_s + (np.array(highlights) - 0.5) * amount * 2 * w_h)


def lift_gamma_gain(x, lift=0.0, gamma=1.0, gain=1.0):
    return clamp((clamp(x * gain + lift * (1 - x))) ** (1 / gamma))


def black_crush(x, level=0.03):
    return clamp((x - level) / (1 - level))


def highlight_rolloff(x, start=0.8):
    over = np.maximum(0, x - start)
    return clamp(np.minimum(x, start) + over * (1 - over / (2 * (1 - start) + 1e-6)))


PRESETS = {"None": lambda x: x}  # original colours, no look


def preset(name):
    def deco(fn):
        PRESETS[name] = fn
        return fn
    return deco


@preset("Vibrant Cinematic")
def _vibrant_cinematic(x):
    # Matched to the reference Style: deep blacks, punchy neon colour, glowing warm highlights, clean skin
    x = black_crush(x, 0.035)
    x = s_curve(x, 0.55, 0.42)
    x = highlight_rolloff(x, 0.82)
    x = split_tone(x, (0.45, 0.5, 0.57), (0.6, 0.53, 0.42), 0.07, 0.45)  # cool, not violet, shadows
    x = saturate(x, 1.28, vibrance=0.25, protect_skin=0.6)
    return x


@preset("Natural")
def _natural(x):
    return saturate(s_curve(x, 0.2), 1.06, 0.08, 0.3)


@preset("Warm Celebration")
def _warm(x):
    x = s_curve(x, 0.35, 0.45)
    x = split_tone(x, (0.52, 0.47, 0.43), (0.62, 0.55, 0.4), 0.1)
    return saturate(x, 1.18, 0.15, 0.4)


@preset("Cinematic Teal & Orange")
def _teal_orange(x):
    x = black_crush(s_curve(x, 0.45), 0.02)
    x = split_tone(x, (0.38, 0.52, 0.58), (0.64, 0.52, 0.38), 0.12, 0.5)
    return saturate(x, 1.12, 0.1, 0.5)


@preset("Clean Corporate")
def _corporate(x):
    x = lift_gamma_gain(x, 0.015, 1.08, 1.02)
    x = s_curve(x, 0.22, 0.5)
    return saturate(x, 1.04, 0.06, 0.4)


@preset("Moody Night")
def _moody(x):
    x = lift_gamma_gain(x, 0.0, 0.88, 0.95)
    x = black_crush(s_curve(x, 0.5, 0.5), 0.02)
    x = split_tone(x, (0.4, 0.47, 0.6), (0.55, 0.5, 0.46), 0.1, 0.5)
    return saturate(x, 0.9, 0.1, 0.3)


@preset("Black & White")
def _bw(x):
    y = x[..., 0] * 0.3 + x[..., 1] * 0.59 + x[..., 2] * 0.11
    y = s_curve(y[..., None].repeat(3, -1), 0.5, 0.45)
    return black_crush(y, 0.02)


# ------------------------------------------------------------------ film looks: restrained, the way a film colourist works
def hue_sat(x, hue_lo, hue_hi, amount):
    """Change saturation of one hue band only (hue in 0..1), e.g. calm down neon greens."""
    import colorsys as _c  # noqa: F401 (documentation of hue convention)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    d = np.maximum(mx - mn, 1e-6)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
    w = clamp(1 - np.abs(((h - (hue_lo + hue_hi) / 2 + 0.5) % 1) - 0.5) / max(1e-3, (hue_hi - hue_lo) / 2))[..., None]
    y = luma(x)[..., None]
    return clamp(y + (x - y) * (1 + (amount - 1) * w))


def matte(x, black=0.02, white=0.98):
    return clamp(black + x * (white - black))


@preset("Film Print")
def _film_print(x):
    # classic cinema print stock: dense but soft blacks, warm highlights, cyan-leaning shadows, calm greens
    x = s_curve(x, 0.42, 0.44)
    x = highlight_rolloff(x, 0.74)
    x = split_tone(x, (0.45, 0.52, 0.54), (0.6, 0.54, 0.45), 0.08, 0.48)
    x = hue_sat(x, 0.22, 0.45, 0.72)
    x = saturate(x, 1.04, 0.06, 0.65)
    return matte(x, 0.025, 0.97)


@preset("Muted Film")
def _muted_film(x):
    # soft, low-saturation film look with gentle contrast: grown-up and quiet
    x = s_curve(x, 0.26, 0.46)
    x = highlight_rolloff(x, 0.78)
    x = split_tone(x, (0.47, 0.52, 0.51), (0.55, 0.53, 0.49), 0.06, 0.5)
    x = saturate(x, 0.8, 0.05, 0.7)
    return matte(x, 0.03, 0.96)


@preset("Documentary Soft")
def _documentary(x):
    # honest and natural, slightly lifted shadows, true skin: for CSR, interviews and real stories
    x = lift_gamma_gain(x, 0.012, 1.04, 1.0)
    x = s_curve(x, 0.15, 0.48)
    x = split_tone(x, (0.5, 0.5, 0.5), (0.56, 0.53, 0.48), 0.04, 0.5)
    return saturate(x, 0.92, 0.08, 0.7)


@preset("Golden Hour")
def _golden(x):
    # late-afternoon warmth: honey highlights, soft contrast, glowing skin
    x = s_curve(x, 0.3, 0.46)
    x = highlight_rolloff(x, 0.8)
    x = split_tone(x, (0.5, 0.48, 0.45), (0.66, 0.56, 0.4), 0.11, 0.45)
    return saturate(x, 1.08, 0.12, 0.55)


@preset("Bleach Bypass")
def _bleach(x):
    # silver retained in the print: low colour, hard contrast, gritty and dramatic
    y = luma(x)[..., None].repeat(3, -1)
    y = s_curve(y, 0.65, 0.45)
    x = clamp(x * 0.5 + y * 0.5)
    x = s_curve(x, 0.35, 0.45)
    return saturate(x, 0.62, 0.0, 0.4)


# ------------------------------------------------------------------ Google AI colourist: per-clip adjustments
def apply_ai(x, p):
    """Small, clip-specific corrections a colourist would make before the look (all values are gentle)."""
    if not p:
        return x
    t = float(np.clip(p.get("temperature", 0), -1, 1)) * 0.06   # warm (+) / cool (-)
    tint = float(np.clip(p.get("tint", 0), -1, 1)) * 0.04        # magenta (+) / green (-)
    x = clamp(x * np.array([1 + t + tint * 0.5, 1 - tint, 1 - t + tint * 0.5]))
    ev = float(np.clip(p.get("exposure", 0), -0.7, 0.7))
    if abs(ev) > 0.01:
        x = clamp(x * (2 ** ev))
    c = float(np.clip(p.get("contrast", 0), -0.5, 0.5))
    if abs(c) > 0.01:
        x = s_curve(x, 0.5 * c, 0.45) if c > 0 else clamp(0.45 + (x - 0.45) * (1 + c * 0.6))
    hl = float(np.clip(p.get("highlights", 0), -0.5, 0.5))
    if abs(hl) > 0.01:
        y = luma(x)[..., None]
        x = clamp(x + hl * 0.25 * clamp((y - 0.6) / 0.4))
    sh = float(np.clip(p.get("shadows", 0), -0.5, 0.5))
    if abs(sh) > 0.01:
        y = luma(x)[..., None]
        x = clamp(x + sh * 0.2 * clamp(1 - y / 0.35))
    s = float(np.clip(p.get("saturation", 0), -0.5, 0.5))
    if abs(s) > 0.01:
        x = saturate(x, 1 + s, 0.0, 0.6)
    gr = float(np.clip(p.get("reduce_green", 0), 0, 1))  # LED/fluorescent green casts on set walls
    if gr > 0.01:
        x = hue_sat(x, 0.25, 0.42, 1 - 0.5 * gr)
    return x


# ------------------------------------------------------------------ match this look (colour transfer)
def _rgb2lab(x):
    import cv2
    return cv2.cvtColor(x.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_RGB2Lab).reshape(x.shape)


def _lab2rgb(x):
    import cv2
    return clamp(cv2.cvtColor(x.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_Lab2RGB).reshape(x.shape))


def lab_stats(img_rgb01: np.ndarray) -> dict:
    lab = _rgb2lab(img_rgb01.reshape(-1, 3))
    return {"mean": lab.mean(axis=0).tolist(), "std": lab.std(axis=0).tolist()}


def apply_transfer(x, src_stats, ref_stats, strength=0.7):
    lab = _rgb2lab(x)
    sm, ss = np.array(src_stats["mean"]), np.array(src_stats["std"])
    rm, rs = np.array(ref_stats["mean"]), np.array(ref_stats["std"])
    ratio = np.clip(rs / np.maximum(ss, 1e-3), 0.6, 1.8)
    out = (lab - sm) * ratio + rm
    return clamp(x * (1 - strength) + _lab2rgb(out) * strength)


# ------------------------------------------------------------------ hex theme tint
def hex_to_rgb(h):
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def theme_tint(x, hex_color, amount=0.06):
    c = np.array(hex_to_rgb(hex_color))
    y = luma(x)[..., None]
    w = clamp(1 - np.abs(y - 0.35) * 2.5)  # tint mostly shadows/mids
    return clamp(x + (c - c.mean()) * amount * w)


# ------------------------------------------------------------------ build + write
def build_lut(log_profile=None, correction=None, preset_name="Vibrant Cinematic", intensity=1.0,
              transfer=None, theme_hex=None, custom_cube=None, ai=None):
    x = grid()
    x = decode_log(x, log_profile)
    if preset_name == "None":
        # no grading at all: the footage's own colours (Log footage is still converted so it doesn't look grey)
        return clamp(x)
    if correction:
        x = apply_correction(x, correction)
    if ai:
        x = apply_ai(x, ai)
    if transfer:
        x = apply_transfer(x, transfer["src"], transfer["ref"], transfer.get("strength", 0.7))
    if custom_cube:
        graded = apply_cube(x, read_cube(custom_cube))
    else:
        graded = PRESETS.get(preset_name, _natural)(x.copy())
    x = clamp(x + (graded - x) * float(intensity))
    if theme_hex:
        x = theme_tint(x, theme_hex)
    return x


def read_cube(path):
    size, rows = None, []
    for line in Path(path).read_text("utf-8", "ignore").splitlines():
        line = line.strip()
        if line.upper().startswith("LUT_3D_SIZE"):
            size = int(line.split()[-1])
        elif line and (line[0].isdigit() or line[0] in "-."):
            rows.append([float(v) for v in line.split()[:3]])
    return np.array(rows).reshape(size, size, size, 3), size


def apply_cube(x, cube):
    from scipy.ndimage import map_coordinates
    table, size = cube
    idx = clamp(x) * (size - 1)
    out = np.stack([map_coordinates(table[..., c], [idx[:, 2], idx[:, 1], idx[:, 0]], order=1) for c in range(3)], -1)
    return clamp(out)


def write_cube(x, path: Path, title="EventCut"):
    """Written to a temporary file and swapped in, so a clip rendering in parallel never reads half a table."""
    import os
    import threading
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {SIZE}", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 1 1 1"]
    lines += [f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in x]
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text("\n".join(lines) + "\n", "utf-8")
    os.replace(tmp, path)
    return path


LUT_VERSION = 3  # bump when looks change, so old cached colour tables aren't reused


def lut_for(**kw) -> Path:
    key = hashlib.sha1((f"v{LUT_VERSION}" + repr(sorted((k, repr(v)) for k, v in kw.items()))).encode()).hexdigest()[:14]
    p = LUT_DIR / f"{key}.cube"
    if not p.exists():
        write_cube(build_lut(**kw), p, kw.get("preset_name", "EventCut"))
    return p


def export_preset_luts():
    """Write the built-in presets into the Looks folder so users can reuse them in Premiere/Resolve too."""
    out = LOOKS
    out.mkdir(parents=True, exist_ok=True)
    for name in PRESETS:
        p = out / f"EventCut - {name}.cube"
        if not p.exists():
            write_cube(build_lut(preset_name=name), p, name)
    return sorted(str(p) for p in out.glob("*.cube"))
