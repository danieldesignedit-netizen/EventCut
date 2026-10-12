"""Camera-style transitions rendered frame by frame with OpenCV: tilting card reveal, zoom in, zoom out, eased pans.

FFmpeg's built-in xfade can't do a 3D card tilt, and its per-pixel custom expressions are far too slow,
so these few frames (about half a second) are drawn here and spliced into the timeline.
"""
import subprocess
from pathlib import Path

import numpy as np

from . import media

# name -> (built-in fallback for very short shots, duration, sound family)
CUSTOM = {
    "tilt": ("zoomin", 0.6, "whoosh"),
    "zoom_in": ("zoomin", 0.4, "whoosh"),
    "zoom_out": ("fade", 0.5, "reverse"),
    "pan_l": ("smoothleft", 0.44, "whoosh"),
    "pan_u": ("smoothup", 0.44, "whoosh"),
}


def _ease(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def _ease_io(t):  # stronger ease in-out, the curve camera operators use for whips and pushes
    t = min(1.0, max(0.0, t))
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def read_frames(ffmpeg, path, start, n, W, H, fps):
    cmd = [ffmpeg, "-v", "error"] + (["-ss", f"{start:.3f}"] if start > 0 else []) + ["-i", str(path), "-vf", f"fps={fps},scale={W}:{H},setsar=1",
                                                                                     "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = media.run(cmd)
    raw = np.frombuffer(p.stdout, np.uint8)
    k = len(raw) // (W * H * 3)
    frames = [raw[i * W * H * 3:(i + 1) * W * H * 3].reshape(H, W, 3) for i in range(k)]
    while frames and len(frames) < n:
        frames.append(frames[-1])
    return frames


def write_frames(ffmpeg, frames, out, W, H, fps):
    cmd = [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-pix_fmt", "yuv420p"] + media.intermediate_args() + [str(out)]
    kw = {}
    import platform
    if platform.system() == "Windows":
        kw["creationflags"] = media.CREATE_NO_WINDOW
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, **kw)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    err = p.stderr.read()
    if p.wait() != 0:
        raise RuntimeError("Transition render failed: " + err.decode("utf-8", "replace")[-400:])
    return out


def _zoom(img, z, cx=0.5, cy=0.5, cv2=None):
    H, W = img.shape[:2]
    M = np.float32([[z, 0, (1 - z) * W * cx], [0, z, (1 - z) * H * cy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _blend(a, b, m):
    return (a.astype(np.float32) * (1 - m) + b.astype(np.float32) * m).astype(np.uint8)


def tilt(fa, fb, seed=0):
    """Outgoing shot becomes a card that tilts away in 3D and shrinks, revealing the next scene behind it."""
    import cv2
    H, W = fa[0].shape[:2]
    n = len(fa)
    rng = np.random.default_rng(seed)
    side = 1 if rng.random() < 0.5 else -1  # which way the card swings
    asp, d = W / H, 1.6
    corners = np.float32([[-asp / 2, -0.5], [asp / 2, -0.5], [asp / 2, 0.5], [-asp / 2, 0.5]])
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    out = []
    for i in range(n):
        e = _ease((i + 0.5) / n)
        th, ph = side * 1.15 * e, 0.22 * e  # yaw and a little pitch
        s = 1 - 0.38 * e
        dst = []
        for x, y in corners:
            # rotate around the vertical axis, then tip back slightly
            X, Z = x * np.cos(th), x * np.sin(th)
            Y, Z = y * np.cos(ph), Z - y * np.sin(ph)  # top edge leans back
            k = d / (d + Z)
            px = (X * k * s - side * 0.22 * asp * e) * H + W / 2
            py = (Y * k * s) * H + H / 2
            dst.append([px, py])
        M = cv2.getPerspectiveTransform(src, np.float32(dst))
        card = cv2.warpPerspective(fa[i], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.warpPerspective(np.ones((H, W), np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
        alpha = mask * (1 - _ease(((i + 0.5) / n - 0.78) / 0.22))
        # next scene settles from a slight zoom, brightening as the card leaves
        bg = _zoom(fb[i], 1.12 - 0.12 * e, cv2=cv2).astype(np.float32) * (0.55 + 0.45 * e)
        # soft drop shadow under the card for depth
        sh = cv2.GaussianBlur(mask, (0, 0), max(3, W / 60))
        sh = np.roll(sh, int(H * 0.02), axis=0)[..., None] * 0.45 * (1 - e * 0.5)
        bg = bg * (1 - sh)
        a3 = alpha[..., None]
        out.append(np.clip(card.astype(np.float32) * a3 + bg * (1 - a3), 0, 255).astype(np.uint8))
    return out


def zoom_in(fa, fb, seed=0):
    """Punch through: push into the outgoing shot, land in the next one as it settles."""
    import cv2
    n, out = len(fa), []
    for i in range(n):
        t = (i + 0.5) / n
        e = _ease_io(t)
        za, zb = 1 + 1.1 * e, 1.7 - 0.7 * e
        m = _ease((t - 0.38) / 0.3)
        # motion blur: average three nearby zoom levels while moving fast
        speed = 6 * t * (1 - t)
        steps = [0] if speed < 0.6 else [-0.04, 0, 0.04]
        a = np.mean([_zoom(fa[i], za * (1 + s * speed), cv2=cv2).astype(np.float32) for s in steps], 0) if m < 1 else 0
        b = np.mean([_zoom(fb[i], zb * (1 + s * speed), cv2=cv2).astype(np.float32) for s in steps], 0) if m > 0 else 0
        out.append(np.clip(a * (1 - m) + b * m, 0, 255).astype(np.uint8))
    return out


def zoom_out(fa, fb, seed=0):
    """Pull back: the outgoing shot shrinks into the frame while the next scene eases out from a close crop."""
    import cv2
    H, W = fa[0].shape[:2]
    n, out = len(fa), []
    ones = np.ones((H, W), np.float32)
    for i in range(n):
        t = (i + 0.5) / n
        e = _ease_io(t)
        za = 1 - 0.5 * e
        M = np.float32([[za, 0, (1 - za) * W / 2], [0, za, (1 - za) * H / 2]])
        a = cv2.warpAffine(fa[i], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.warpAffine(ones, M, (W, H), flags=cv2.INTER_LINEAR) * (1 - _ease((t - 0.5) / 0.5))
        b = _zoom(fb[i], 1.3 - 0.3 * e, cv2=cv2).astype(np.float32) * (0.7 + 0.3 * e)
        m3 = mask[..., None]
        out.append(np.clip(a.astype(np.float32) * m3 + b * (1 - m3), 0, 255).astype(np.uint8))
    return out


def pan(fa, fb, axis="x", seed=0):
    """Eased camera pan: both shots travel together with motion blur at full speed, like a real whip-pan."""
    import cv2
    H, W = fa[0].shape[:2]
    L = W if axis == "x" else H
    n, out = len(fa), []
    for i in range(n):
        t = (i + 0.5) / n
        off = int(round(_ease_io(t) * L))
        strip = np.concatenate([fa[i], fb[i]], axis=1 if axis == "x" else 0)
        f = strip[:, off:off + W] if axis == "x" else strip[off:off + H, :]
        vel = abs(_ease_io(min(1, t + 0.5 / n)) - _ease_io(max(0, t - 0.5 / n))) * L
        k = int(vel * 0.4)
        if k >= 3:
            f = cv2.blur(f, (k, 1) if axis == "x" else (1, k))
        out.append(f)
    return out


RENDER = {"tilt": tilt, "zoom_in": zoom_in, "zoom_out": zoom_out,
          "pan_l": lambda a, b, seed=0: pan(a, b, "x", seed), "pan_u": lambda a, b, seed=0: pan(a, b, "y", seed)}


def frames_for(d, fps):
    return max(4, int(round(d * fps)))


def make(kind, ffmpeg, run_a, len_a, run_b, W, H, fps, out: Path, seed=0):
    """Render the transition between the tail of run_a and the head of run_b. Returns (path, exact duration)."""
    d = CUSTOM[kind][1]
    n = frames_for(d, fps)
    dur = n / fps
    fa = read_frames(ffmpeg, run_a, max(0.0, len_a - dur), n, W, H, fps)
    fb = read_frames(ffmpeg, run_b, 0.0, n, W, H, fps)
    if len(fa) < n or len(fb) < n:
        raise RuntimeError("not enough frames for transition")
    write_frames(ffmpeg, RENDER[kind](fa, fb, seed=seed), out, W, H, fps)
    return out, dur
