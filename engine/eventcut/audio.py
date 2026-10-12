"""Beat detection, plus EventCut's own procedurally generated music beds and sound effects.

All generated audio is original (synthesised here), so it is free for commercial use.
Users can also drop licensed tracks into "Music & SFX/Music" or upload their own.
"""
import json
import wave
from pathlib import Path

import numpy as np
from scipy import signal

from .config import HOME
from . import media

SR = 44100
LIB = HOME / "Music & SFX"


# ---------------------------------------------------------------- beat detection
def detect_beats(path: str) -> dict:
    """Tempo, beat times, downbeats and an energy curve (for finding the drop)."""
    cache = Path(str(path) + ".beats.json") if str(path).startswith(str(LIB)) else None
    if cache and cache.exists():
        return json.loads(cache.read_text())
    sr = 22050
    y = media.load_audio(path, sr)
    hop = 256
    f, t, Z = signal.stft(y, sr, nperseg=1024, noverlap=1024 - hop)
    mag = np.log1p(np.abs(Z) * 100)
    flux = np.maximum(0, np.diff(mag, axis=1)).sum(axis=0)
    flux = (flux - flux.mean()) / (flux.std() + 1e-9)
    fps = sr / hop
    ac = np.correlate(flux, flux, "full")[len(flux) - 1:]
    lags = np.arange(len(ac))
    bpm_axis = 60 * fps / np.maximum(lags, 1)
    valid = (bpm_axis > 70) & (bpm_axis < 180)
    weight = np.exp(-0.5 * (np.log2(np.maximum(bpm_axis, 1) / 115) / 0.6) ** 2)  # prefer ~115 BPM
    lag = int(lags[valid][np.argmax((ac * weight)[valid])])
    # parabolic refinement for sub-frame tempo accuracy
    if 1 <= lag < len(ac) - 1:
        a0, a1, a2 = ac[lag - 1], ac[lag], ac[lag + 1]
        lagf = float(lag + 0.5 * (a0 - a2) / (a0 - 2 * a1 + a2 + 1e-12))
    else:
        lagf = float(lag)
    period = lag
    phases = [flux[p::period].sum() for p in range(period)]
    phase = int(np.argmax(phases))
    beats = (phase + np.arange(0, (len(flux) - phase) / lagf) * lagf) / fps
    # downbeat = beat offset (0..3) with the strongest onsets
    bidx = np.clip((beats * fps).astype(int), 0, len(flux) - 1)
    strengths = [flux[bidx[k::4]].sum() for k in range(4)]
    db = int(np.argmax(strengths))
    rms = np.sqrt(signal.convolve(y ** 2, np.ones(sr) / sr, mode="same")[::sr // 4])
    out = {"bpm": round(float(60 * fps / lagf), 1), "beats": [round(float(b), 3) for b in beats],
           "downbeats": [round(float(b), 3) for b in beats[db::4]],
           "energy": [round(float(x), 4) for x in rms / (rms.max() + 1e-9)], "energy_rate": 4,
           "duration": round(len(y) / sr, 2)}
    if cache:
        cache.write_text(json.dumps(out))
    return out


# ---------------------------------------------------------------- synthesis helpers
def _env(n, a=0.002, d=0.1, s=0.0, r=0.05, sustain_len=0.0):
    a_n, d_n, s_n, r_n = int(a * SR), int(d * SR), int(sustain_len * SR), int(r * SR)
    e = np.concatenate([np.linspace(0, 1, max(1, a_n)), np.linspace(1, s, max(1, d_n)),
                        np.full(s_n, s), np.linspace(s, 0, max(1, r_n))])
    return np.pad(e, (0, max(0, n - len(e))))[:n]


def _lp(x, hz, order=2):
    b, a = signal.butter(order, min(0.99, hz / (SR / 2)), "low")
    return signal.lfilter(b, a, x)


def _hp(x, hz, order=2):
    b, a = signal.butter(order, min(0.99, hz / (SR / 2)), "high")
    return signal.lfilter(b, a, x)


def _bp(x, lo, hi):
    b, a = signal.butter(2, [lo / (SR / 2), min(0.99, hi / (SR / 2))], "band")
    return signal.lfilter(b, a, x)


rng = np.random.default_rng(7)


def kick(dur=0.45):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 48 + 110 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t * 7.5) + 0.25 * np.sin(ph * 2) * np.exp(-t * 40)


def clap(dur=0.3):
    n = int(dur * SR)
    noise = _bp(rng.standard_normal(n), 900, 4200)
    e = np.zeros(n)
    for off in (0, 0.011, 0.022):
        e += _env(n, 0.001, 0.012, 0, 0.001) if off == 0 else np.roll(_env(n, 0.001, 0.012, 0, 0.001), int(off * SR))
    e += np.roll(_env(n, 0.001, 0.18, 0, 0.01), int(0.03 * SR)) * 0.8
    return noise * e * 0.6


def hat(dur=0.06, open_=False):
    n = int((0.25 if open_ else dur) * SR)
    return _hp(rng.standard_normal(n), 7000) * _env(n, 0.001, 0.2 if open_ else 0.04, 0, 0.01) * 0.25


def shaker(dur=0.09):
    n = int(dur * SR)
    return _bp(rng.standard_normal(n), 4000, 11000) * _env(n, 0.012, 0.06, 0, 0.01) * 0.18


def tone(freq, dur, kind="saw", detune=0.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for d in ([0] if not detune else [-detune, 0, detune]):
        ph = 2 * np.pi * freq * (1 + d) * t
        out += signal.sawtooth(ph) if kind == "saw" else np.sin(ph) if kind == "sine" else signal.square(ph, 0.5) * 0.6
    return out / (1 if not detune else 3)


def midi(n):
    return 440 * 2 ** ((n - 69) / 12)


MOODS = {
    # name: bpm, key root (midi), progression (semitones from root, minor/major), groove, family
    "Upbeat Afro House": {"bpm": 120, "root": 45, "prog": [(0, "m"), (8, "M"), (3, "M"), (10, "M")], "swing": 0.12, "perc": "afro", "family": "upbeat"},
    "Afrobeats Groove": {"bpm": 104, "root": 44, "prog": [(0, "m"), (5, "m"), (10, "M"), (3, "M")], "swing": 0.18, "perc": "afro", "family": "upbeat"},
    "Amapiano Glow": {"bpm": 112, "root": 41, "prog": [(0, "M"), (9, "m"), (4, "m"), (5, "M")], "swing": 0.1, "perc": "amapiano", "family": "upbeat"},
    "Amapiano Sunset": {"bpm": 113, "root": 44, "prog": [(0, "m"), (5, "m"), (10, "M"), (8, "M")], "swing": 0.14, "perc": "amapiano", "family": "upbeat"},
    "Amapiano Jazz Night": {"bpm": 110, "root": 39, "prog": [(0, "m"), (3, "M"), (8, "M"), (7, "M")], "swing": 0.16, "perc": "amapiano", "family": "upbeat"},
    "Dance Pop Lift": {"bpm": 124, "root": 47, "prog": [(0, "m"), (8, "M"), (10, "M"), (5, "m")], "swing": 0.0, "perc": "pop", "family": "upbeat"},
    "Cinematic Night": {"bpm": 96, "root": 38, "prog": [(0, "m"), (5, "m"), (8, "M"), (7, "M")], "swing": 0.0, "perc": "cinematic", "family": "cinematic"},
    "Epic Rise": {"bpm": 90, "root": 40, "prog": [(0, "m"), (8, "M"), (3, "M"), (10, "M")], "swing": 0.0, "perc": "cinematic", "family": "cinematic"},
    "Dark Pulse": {"bpm": 100, "root": 37, "prog": [(0, "m"), (0, "m"), (8, "M"), (7, "M")], "swing": 0.0, "perc": "cinematic", "family": "cinematic"},
    "Corporate Bright": {"bpm": 112, "root": 48, "prog": [(0, "M"), (7, "M"), (9, "m"), (5, "M")], "swing": 0.0, "perc": "pop", "family": "corporate"},
    "Tech Optimism": {"bpm": 118, "root": 46, "prog": [(0, "M"), (5, "M"), (9, "m"), (7, "M")], "swing": 0.0, "perc": "pop", "family": "corporate"},
    "Clean Motion": {"bpm": 108, "root": 43, "prog": [(0, "M"), (4, "m"), (5, "M"), (7, "M")], "swing": 0.05, "perc": "chill", "family": "corporate"},
    "Chill Celebration": {"bpm": 100, "root": 43, "prog": [(0, "M"), (9, "m"), (5, "M"), (7, "M")], "swing": 0.08, "perc": "chill", "family": "chill"},
    "Lo-fi Evening": {"bpm": 84, "root": 42, "prog": [(0, "M"), (4, "m"), (9, "m"), (5, "M")], "swing": 0.22, "perc": "chill", "family": "chill"},
    "Sunset Groove": {"bpm": 94, "root": 45, "prog": [(0, "m"), (3, "M"), (5, "m"), (10, "M")], "swing": 0.14, "perc": "chill", "family": "chill"},
    "Emotional Piano": {"bpm": 76, "root": 41, "prog": [(0, "M"), (7, "M"), (9, "m"), (5, "M")], "swing": 0.0, "perc": "none", "family": "emotional"},
    "Gentle Hope": {"bpm": 70, "root": 43, "prog": [(0, "M"), (5, "M"), (9, "m"), (7, "M")], "swing": 0.0, "perc": "none", "family": "emotional"},
    "Tribute": {"bpm": 66, "root": 38, "prog": [(0, "m"), (8, "M"), (3, "M"), (10, "M")], "swing": 0.0, "perc": "none", "family": "emotional"},
}
FAMILIES = {}
for _n, _c in MOODS.items():
    FAMILIES.setdefault(_c["family"], []).append(_n)


def style_cfg(mood: str, seed: int = 0) -> dict:
    """A style plus per-seed variation (key, tempo, chord order), so every regenerate sounds different."""
    if mood in FAMILIES:  # a family was given: pick one of its styles
        mood = FAMILIES[mood][np.random.default_rng(seed).integers(len(FAMILIES[mood]))]
    cfg = dict(MOODS.get(mood, MOODS["Upbeat Afro House"]))
    cfg["name"] = mood
    if seed:
        r = np.random.default_rng(seed * 7919 + 13)
        cfg["root"] = cfg["root"] + int(r.integers(-3, 4))
        cfg["bpm"] = round(cfg["bpm"] * (1 + r.uniform(-0.04, 0.04)), 1)
        k = int(r.integers(0, len(cfg["prog"])))
        cfg["prog"] = cfg["prog"][k:] + cfg["prog"][:k]
        cfg["arp"] = [int(x) for x in r.integers(0, 3, 8)]
    return cfg


def _chord(root, quality):
    third = 3 if quality == "m" else 4
    return [root, root + third, root + 7]


def music_grid(mood: str, duration: float, seed: int = 0) -> dict:
    """Exact beat grid of a generated EventCut track (no detection needed)."""
    bpm = style_cfg(mood, seed)["bpm"]
    beats = np.arange(0, duration + 0.01, 60 / bpm)
    return {"bpm": bpm, "beats": [round(float(b), 4) for b in beats], "downbeats": [round(float(b), 4) for b in beats[::4]], "duration": duration}


def generate_music(mood: str, duration: float, drop_at: float = None, seed: int = 0) -> np.ndarray:
    """Arranged music bed: intro, build with riser, drop at `drop_at`, clean ending at `duration`."""
    cfg = style_cfg(mood, seed)
    r = np.random.default_rng(seed)
    arp = cfg.get("arp", [0, 2, 1, 2, 0, 2, 1, 2])
    bpm = cfg["bpm"]
    beat = 60 / bpm
    bar = 4 * beat
    total = int((duration + 1.5) * SR)
    L = np.zeros(total)
    R = np.zeros(total)
    drop_at = drop_at if drop_at is not None else min(duration * 0.3, 8 * bar)
    drop_at = round(drop_at / bar) * bar or bar * 2
    end = duration

    def place(sig, t, gain=1.0, pan=0.0):
        i = int(t * SR)
        if i >= total or i < 0:
            return
        seg = sig[: total - i] * gain
        L[i:i + len(seg)] += seg * (1 - max(0, pan))
        R[i:i + len(seg)] += seg * (1 + min(0, pan))

    k, c, h, sh = kick(), clap(), hat(), shaker()
    nbars = int(np.ceil(end / bar)) + 1
    prog = cfg["prog"]
    for b in range(nbars):
        t0 = b * bar
        if t0 >= end:
            break
        section = "intro" if t0 < drop_at else "main"
        if end - t0 <= bar * 1.01:
            section = "outro"
        root_off, q = prog[b % len(prog)]
        notes = _chord(cfg["root"] + 12 + root_off, q)
        # pads: always
        pad = sum(tone(midi(n + 12), bar, "saw", 0.006) for n in notes) / 3
        pad = _lp(pad, 900 if section == "intro" else 2200) * _env(len(pad), 0.25, 0.1, 0.8, 0.3, bar - 0.6)
        place(pad, t0, 0.16, -0.2)
        place(pad, t0 + 0.012, 0.16, 0.2)
        if cfg["perc"] == "none":
            # piano-ish arpeggio
            for i in range(8):
                n = notes[[0, 1, 2, 1, 2, 0, 1, 2][i]] + 24
                pl = tone(midi(n), 1.2, "sine") + 0.3 * tone(midi(n + 12), 1.2, "sine")
                place(pl * _env(len(pl), 0.003, 1.1, 0, 0.05), t0 + i * beat / 2, 0.12, 0.3 * (i % 2 * 2 - 1))
            bassn = tone(midi(cfg["root"] + root_off), bar, "sine")
            place(bassn * _env(len(bassn), 0.02, 0.2, 0.6, 0.3, bar - 0.6), t0, 0.22)
            continue
        if section != "outro":
            # bass: root on offbeats/8ths
            for i in range(8):
                if section == "intro" and i % 4:
                    continue
                bn = tone(midi(cfg["root"] + root_off), beat / 2 * 0.9, "saw")
                bn = _lp(bn, 420) * _env(len(bn), 0.004, 0.18, 0.4, 0.04, beat / 2 * 0.5)
                place(bn, t0 + i * beat / 2 + (beat / 2 if cfg["perc"] == "pop" and i % 2 == 0 else 0) * 0, 0.42)
            for i in range(4):
                tb = t0 + i * beat
                if section == "main" or i == 0:
                    place(k, tb, 0.9 if section == "main" else 0.5)
                if section == "main" and i in (1, 3) and cfg["perc"] != "cinematic":
                    place(c, tb, 0.55)
                if cfg["perc"] == "cinematic" and section == "main" and i == 2:
                    place(_lp(rng.standard_normal(int(0.5 * SR)), 300) * _env(int(0.5 * SR), 0.001, 0.4), tb, 0.5)
            if section == "main":
                for i in range(16):
                    ts = t0 + i * beat / 4 + (cfg["swing"] * beat / 4 if i % 2 else 0)
                    if cfg["perc"] == "amapiano" and i in (0, 5, 8, 11, 14):
                        # sliding log-drum bass, the signature amapiano sound
                        f0 = midi(cfg["root"] + root_off - 12)
                        nn = int(0.32 * SR)
                        tt = np.arange(nn) / SR
                        ld = np.sin(2 * np.pi * np.cumsum(f0 * (1.6 - 0.6 * np.exp(-tt * 18))) / SR) * np.exp(-tt * 7)
                        place(_lp(ld, 500), ts, 0.55)
                    if cfg["perc"] in ("afro", "amapiano"):
                        place(sh, ts, 1.0 if i % 4 == 2 else 0.55, 0.4)
                        if i in (3, 6, 10, 14):  # log-drum / conga style accent
                            ld = _lp(tone(midi(cfg["root"] + root_off + 12), 0.18, "sine"), 600) * _env(int(0.18 * SR), 0.002, 0.15)
                            place(ld, ts, 0.35, -0.3)
                    elif i % 2 == 0:
                        place(hat(open_=(i % 8 == 4)), ts, 0.7 if i % 4 == 2 else 0.4, 0.3)
                # top arp in main
                if cfg["perc"] in ("pop", "chill", "afro", "amapiano"):
                    for i in range(8):
                        n = notes[arp[i] % 3] + 24 + r.integers(0, 2) * 12
                        a = _lp(tone(midi(n), beat / 2, "square"), 2600) * _env(int(beat / 2 * SR), 0.002, 0.2, 0, 0.02)
                        place(a, t0 + i * beat / 2, 0.06, 0.5 * (1 if i % 2 else -1))
        else:
            # outro: one long chord + final kick hit
            place(k, t0, 0.7)
    # riser into drop, impact on drop
    rs = riser(min(4 * beat, drop_at))
    place(rs, drop_at - len(rs) / SR, 0.35)
    place(impact(), drop_at, 0.6)
    st = np.stack([L, R], axis=1)[: int(end * SR)]
    fade = int(min(1.2, end * 0.08) * SR)
    st[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 1.5
    st /= np.max(np.abs(st)) + 1e-9
    return st * 0.89


def whoosh(dur=0.6, up=True):
    n = int(dur * SR)
    t = np.linspace(0, 1, n)
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    # sweep a band through the noise in chunks
    chunk = 512
    for i in range(0, n, chunk):
        x = t[i] if up else 1 - t[i]
        cf = 300 + 5000 * x ** 2
        seg = noise[max(0, i - 2048): i + chunk]
        out[i:i + chunk] = _bp(seg, cf * 0.6, cf * 1.4)[-len(out[i:i + chunk]):]
    env = np.sin(np.pi * t) ** 1.5
    return out * env / (np.max(np.abs(out * env)) + 1e-9)


def riser(dur=2.0):
    n = int(dur * SR)
    t = np.linspace(0, 1, n)
    noise = _hp(rng.standard_normal(n), 800) * t ** 2
    f = 200 + 1200 * t ** 2
    s = signal.sawtooth(2 * np.pi * np.cumsum(f) / SR) * t ** 3 * 0.3
    out = _lp(noise + s, 9000)
    return out / (np.max(np.abs(out)) + 1e-9)


def impact(dur=1.6):
    n = int(dur * SR)
    t = np.arange(n) / SR
    boom = np.sin(2 * np.pi * np.cumsum(40 + 60 * np.exp(-t * 12)) / SR) * np.exp(-t * 2.6)
    crack = _hp(rng.standard_normal(n), 2000) * np.exp(-t * 30) * 0.5
    out = boom + crack
    return out / (np.max(np.abs(out)) + 1e-9)


def pop(dur=0.12):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.sin(2 * np.pi * (900 - 3000 * t) * t) * np.exp(-t * 45)
    return out / np.max(np.abs(out))


def shutter():
    n = int(0.18 * SR)
    a = _bp(rng.standard_normal(n), 1500, 9000) * (_env(n, 0.001, 0.02) + np.roll(_env(n, 0.001, 0.03), int(0.07 * SR)))
    return a / np.max(np.abs(a))


def signature_chime(success=True):
    """EventCut's own render-complete sound: a bright rising two-note glint (or a soft low tone on failure)."""
    if not success:
        n = int(0.7 * SR)
        x = tone(midi(57), 0.7, "sine") + 0.5 * tone(midi(53), 0.7, "sine")
        return x * _env(n, 0.01, 0.6, 0, 0.05) * 0.5
    out = np.zeros(int(1.3 * SR))
    for i, (nn, t) in enumerate([(76, 0.0), (83, 0.11), (88, 0.22)]):
        x = tone(midi(nn), 1.0, "sine") + 0.35 * tone(midi(nn + 12), 1.0, "sine") + 0.1 * tone(midi(nn + 19), 1.0, "sine")
        x *= _env(len(x), 0.003, 0.95, 0, 0.02)
        s = int(t * SR)
        out[s:s + len(x)] += x * (0.6 + 0.2 * i)
    sweep = whoosh(0.35) * 0.12
    out[: len(sweep)] += sweep
    return out / np.max(np.abs(out)) * 0.8


def write_wav(path: Path, data: np.ndarray):
    data = np.asarray(data)
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


def glitch(seed=0, dur=0.35):
    r = np.random.default_rng(seed)
    n = int(dur * SR)
    out = np.zeros(n)
    i = 0
    while i < n:
        L = int(r.integers(300, 3000))
        seg = r.standard_normal(min(L, n - i)) if r.random() < 0.5 else signal.square(2 * np.pi * r.uniform(80, 900) * np.arange(min(L, n - i)) / SR)
        out[i:i + len(seg)] = seg * r.uniform(0.2, 1.0)
        i += L + int(r.integers(0, 800))
    out = np.round(out * 6) / 6  # bit-crush
    return out * _env(n, 0.001, dur * 0.8, 0, 0.02)


def sparkle(seed=0):
    r = np.random.default_rng(seed)
    out = np.zeros(int(0.9 * SR))
    for k in range(7):
        f = midi(84 + int(r.choice([0, 3, 5, 7, 10, 12, 15])))
        x = tone(f, 0.5, "sine") * _env(int(0.5 * SR), 0.002, 0.45)
        st = int(k * 0.06 * SR)
        out[st:st + len(x)] += x * (0.8 - k * 0.08)
    return out / np.max(np.abs(out))


def swipe(seed=0):
    r = np.random.default_rng(seed)
    d = r.uniform(0.15, 0.3)
    n = int(d * SR)
    x = _hp(r.standard_normal(n), r.uniform(1500, 4000)) * np.sin(np.pi * np.linspace(0, 1, n)) ** 2
    return x / np.max(np.abs(x))


def bass_drop(seed=0):
    n = int(1.4 * SR)
    t = np.arange(n) / SR
    f = 120 * np.exp(-t * (1.6 + seed * 0.4)) + 30
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.8)
    return x / np.max(np.abs(x))


def _sfx_set():
    """Around 45 original effects in families, so automatic sound design varies between versions."""
    items = {}
    for i in range(10):
        d = 0.35 + i * 0.07
        items[f"Whoosh {i + 1:02d}"] = (lambda d=d, up=(i % 3 != 2): whoosh(d, up))
    for i in range(4):
        items[f"Reverse Whoosh {i + 1:02d}"] = (lambda d=0.5 + i * 0.15: whoosh(d, up=False)[::-1].copy())
    for i in range(6):
        items[f"Impact {i + 1:02d}"] = (lambda d=1.0 + i * 0.25: impact(d))
    for i in range(4):
        items[f"Riser {i + 1:02d}"] = (lambda d=1.0 + i * 0.75: riser(d))
    for i in range(4):
        items[f"Swipe {i + 1:02d}"] = (lambda k=i: swipe(k))
    for i in range(3):
        items[f"Glitch {i + 1:02d}"] = (lambda k=i: glitch(k))
    for i in range(3):
        items[f"Sparkle {i + 1:02d}"] = (lambda k=i: sparkle(k))
    for i in range(2):
        items[f"Bass Drop {i + 1:02d}"] = (lambda k=i: bass_drop(k))
    for i in range(3):
        items[f"Pop {i + 1:02d}"] = (lambda d=0.08 + i * 0.04: pop(d))
    items["Shutter 01"] = shutter
    items["Shutter 02"] = lambda: np.concatenate([shutter(), np.zeros(int(0.05 * SR)), shutter()])
    return items


def ensure_library() -> dict:
    """Write the original SFX set and the chime once; music is generated per video to fit its length."""
    sfx_dir = LIB / "SFX"
    sfx_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in _sfx_set().items():
        p = sfx_dir / f"EventCut {name}.wav"
        if not p.exists():
            write_wav(p, fn())
    for ok in (True, False):
        p = LIB / ("eventcut_done.wav" if ok else "eventcut_failed.wav")
        if not p.exists():
            write_wav(p, signature_chime(ok))
    return library_listing()


SFX_FAMILY_WORDS = {
    "whoosh": ("whoosh", "swoosh", "swish", "swipe", "woosh", "transition"),
    "reverse": ("reverse",),
    "impact": ("impact", "hit", "boom", "drop", "thud", "slam", "punch"),
    "riser": ("riser", "rise", "build"),
}


def sfx_families() -> dict:
    """All available effects (originals + downloaded library effects) grouped by family."""
    files = []
    for d in (LIB / "SFX", LIB / "Library" / "SFX"):
        if d.exists():
            files += [f for f in d.iterdir() if f.suffix.lower() in media.AUDIO_EXT]
    fams = {k: [] for k in SFX_FAMILY_WORDS}
    for f in sorted(files):
        n = f.stem.lower()
        for fam, words in SFX_FAMILY_WORDS.items():
            if fam == "whoosh" and "reverse" in n:
                continue
            if any(w in n for w in words):
                fams[fam].append(str(f))
                break
    return fams


# ---------------------------------------------------------------- online library (Openverse)
OPENVERSE = "https://api.openverse.org/v1/audio/"
FAMILY_QUERIES = {
    "upbeat": ["upbeat", "afrobeat", "happy dance", "party", "energetic pop", "funk", "house music"],
    "cinematic": ["cinematic", "epic", "trailer", "dramatic", "orchestral"],
    "corporate": ["corporate", "inspiring", "technology", "motivational", "positive"],
    "chill": ["chill", "lofi", "relaxed", "acoustic", "smooth"],
    "emotional": ["emotional piano", "inspirational", "sentimental", "strings", "hopeful"],
}


def _http_json(url):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "EventCut/1.0 (+https://eventcut.netlify.app)"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


# Genres the open library barely tags get searched with close relatives too
RELATED = {
    "amapiano": ["amapiano", "log drum", "afro house", "south african house", "deep house", "afrobeat"],
    "afrobeat": ["afrobeat", "afrobeats", "afro", "african", "highlife", "afro house"],
    "hip hop": ["hip hop", "hiphop", "rap beat", "boom bap", "trap"],
    "party": ["party", "dance", "club", "edm", "house music"],
    "emotional piano": ["emotional piano", "piano", "sad piano", "sentimental"],
}


def originals_for(q: str) -> list:
    """EventCut's own styles that match a search word (always available, made to fit each video)."""
    ql = (q or "").lower().strip()
    if not ql:
        return []
    words = {"afrobeat": "Afro", "afro": "Afro", "party": "upbeat", "upbeat": "upbeat", "dance": "upbeat", "hip hop": "upbeat",
             "inspiring": "corporate", "corporate": "corporate", "cinematic": "cinematic", "epic": "cinematic",
             "chill": "chill", "lofi": "chill", "emotional piano": "emotional", "emotional": "emotional", "piano": "emotional"}
    out = []
    for name, cfg in MOODS.items():
        key = words.get(ql)
        if ql in name.lower() or (key and (key == cfg["family"] or key.lower() in name.lower())):
            out.append({"name": name, "bpm": cfg["bpm"], "family": cfg["family"]})
    return out


def _openverse(q, category, page, page_size):
    from urllib.parse import urlencode
    params = {"q": q, "category": category, "license": "cc0,by", "page": max(1, int(page)), "page_size": min(20, int(page_size))}
    return _http_json(OPENVERSE + "?" + urlencode(params))


def search_online(q: str = "", category: str = "music", page: int = 1, page_size: int = 20) -> dict:
    """Search openly licensed audio that can be used commercially and edited (CC0 and CC BY only:
    share-alike licences would force the whole company video under the same licence)."""
    q = q or ("music" if category == "music" else "sound effect")
    page = max(1, int(page))
    data = _openverse(q, category, page, page_size)
    raw = list(data.get("results", []))
    count, pages = data.get("result_count", 0), data.get("page_count", 1)
    related = RELATED.get(q.lower().strip(), []) if category == "music" else []
    if related and len(raw) < 12:
        seen = {r.get("id") for r in raw}
        for alt in related:
            if alt == q.lower():
                continue
            try:
                d2 = _openverse(alt, category, page, 10)
            except Exception:
                continue
            for r in d2.get("results", []):
                if r.get("id") not in seen:
                    seen.add(r.get("id"))
                    raw.append(r)
            count += d2.get("result_count", 0)
            pages = max(pages, d2.get("page_count", 1))
            if len(raw) >= 30:
                break
    data = {"results": raw, "result_count": count, "page_count": pages}
    params = {"page": page}
    items = []
    for r in data.get("results", []):
        if not r.get("url"):
            continue
        items.append({"id": r.get("id"), "title": (r.get("title") or "Untitled")[:120], "creator": r.get("creator") or "Unknown",
                      "url": r["url"], "duration": round((r.get("duration") or 0) / 1000, 1),
                      "license": (r.get("license") or "").upper(), "license_version": r.get("license_version") or "",
                      "license_url": r.get("license_url") or "", "source": r.get("source") or r.get("provider") or "",
                      "page": r.get("foreign_landing_url") or "", "attribution": r.get("attribution") or "",
                      "tags": [t.get("name") for t in (r.get("tags") or [])[:6] if isinstance(t, dict)], "filetype": r.get("filetype") or "mp3"})
    return {"results": items, "count": data.get("result_count", len(items)), "page": params["page"],
            "pages": data.get("page_count", 1)}


def download_online(item: dict, category: str = "music") -> str:
    """Save a library track into Music & SFX/Library with its licence and attribution next to it."""
    import re
    import urllib.request
    folder = LIB / "Library" / ("Music" if category == "music" else "SFX")
    folder.mkdir(parents=True, exist_ok=True)
    base = re.sub(r"[^A-Za-z0-9 ._()-]+", "", f"{item.get('title', 'Track')} - {item.get('creator', '')}").strip()[:90] or "Track"
    ext = "." + (item.get("filetype") or "mp3").lower().strip(".")
    if ext not in media.AUDIO_EXT and ext not in (".ogg", ".opus"):
        ext = ".mp3"
    dest = folder / f"{base}{ext}"
    if not dest.exists():
        req = urllib.request.Request(item["url"], headers={"User-Agent": "EventCut/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        if len(data) < 2000:
            raise RuntimeError("That track couldn't be downloaded. Try another one.")
        dest.write_bytes(data)
    lic = item.get("license", "")
    credit = item.get("attribution") or f"\"{item.get('title')}\" by {item.get('creator')}, licensed under {lic} {item.get('license_version', '')}".strip()
    (folder / f"{base}.licence.txt").write_text(
        f"{'Public domain (CC0). No credit needed.' if lic == 'CC0' else 'Credit required: ' + credit}\n{item.get('license_url', '')}\n{item.get('page', '')}", "utf-8")
    (folder / f"{base}.json").write_text(json.dumps(item), "utf-8")
    return str(dest)


HISTORY = LIB / "music_history.json"


def _history():
    try:
        return json.loads(HISTORY.read_text("utf-8"))
    except Exception:
        return []


def _remember(key):
    h = [k for k in _history() if k != key][-24:] + [key]
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    HISTORY.write_text(json.dumps(h), "utf-8")


def pick_auto_music(family: str, duration: float, seed: int, online: bool = True):
    """A different song every new video and every regenerate: a library track for the mood (online), otherwise an
    EventCut original. The last couple of dozen picks are remembered and never repeated."""
    rng = np.random.default_rng(seed * 104729 + 7)
    recent = set(_history())
    if online:
        try:
            queries = list(FAMILY_QUERIES.get(family, FAMILY_QUERIES["upbeat"]))
            rng.shuffle(queries)
            for q in queries[:3]:
                res = search_online(str(q), "music", page=int(rng.integers(1, 6)))
                good = [r for r in res["results"] if r["duration"] >= min(duration * 0.8, duration + 5) and r["duration"] <= 600
                        and f"lib:{r.get('id')}" not in recent]
                if good:
                    item = good[int(rng.integers(len(good)))]
                    _remember(f"lib:{item.get('id')}")
                    return {"path": download_online(item, "music"), "title": item["title"], "creator": item["creator"],
                            "license": item["license"], "source": "library"}
        except Exception:
            pass
    # EventCut originals: a style not used recently, with a fresh key, tempo and melody from the seed
    styles = FAMILIES.get(family) or list(MOODS)
    fresh = [s for s in styles if f"gen:{s}" not in recent] or styles
    style = str(rng.choice(fresh))
    _remember(f"gen:{style}")
    return {"mood": style, "title": f"{style} (EventCut original)", "source": "generated"}


def library_listing() -> dict:
    music = [{"name": m, "source": "EventCut original, made to fit each video", "licence": "Free for commercial use", "generated": True,
              "bpm": cfg["bpm"], "family": cfg["family"]} for m, cfg in MOODS.items()]
    for folder, tag in [(LIB / "Library" / "Music", "Library"), (LIB / "Music", "Library"), (LIB / "Uploads", "Your upload")]:
        for f in sorted(folder.glob("*")) if folder.exists() else []:
            if f.suffix.lower() in media.AUDIO_EXT | {".ogg", ".opus"}:
                lic = f.with_suffix(".licence.txt")
                music.append({"name": f.stem, "path": str(f), "source": tag,
                              "licence": lic.read_text("utf-8").splitlines()[0] if lic.exists() else ("You confirmed you hold the rights" if tag == "Your upload" else "See library notes")})
    sfx = []
    for folder, tag in [(LIB / "SFX", "EventCut original"), (LIB / "Library" / "SFX", "Library")]:
        for f in sorted(folder.glob("*")) if folder.exists() else []:
            if f.suffix.lower() in media.AUDIO_EXT | {".ogg", ".opus"}:
                sfx.append({"name": f.stem, "path": str(f), "source": tag})
    return {"music": music, "sfx": sfx}


def credits_for(paths) -> str:
    """Attribution text for every library track used in a video (CC BY requires it)."""
    lines = []
    for p in paths:
        lic = Path(p).with_suffix(".licence.txt")
        if lic.exists():
            first = lic.read_text("utf-8").splitlines()[0]
            if first.startswith("Credit required: "):
                lines.append(first.replace("Credit required: ", ""))
    return "\n".join(dict.fromkeys(lines))


def write_wav_sr(path: Path, data: np.ndarray, sr: int):
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    peak = np.max(np.abs(data)) if len(data) else 0
    if peak > 0.99:  # leave headroom; loudnorm sets the final level
        data = data / peak * 0.95
    pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path
