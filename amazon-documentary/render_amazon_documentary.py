#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMAZON - Cinematic Documentary Engine  (v3 "Field Journal")
============================================================
Renders the 90-scene Amazon documentary from illustrated stills + voice-over with
After-Effects-style motion design, using only Python / NumPy / OpenCV / Pillow + FFmpeg.

What it does
  * Reads the scene number printed on every image (built-in OCR), maps image -> scene,
    and removes the printed label from the frame (Poisson patch fill).
  * 2.5D depth-parallax camera: push-in / pull-out / pans / tilts / orbit / crane / dolly-zoom /
    rotate / handheld.  Uses Depth-Anything-V2 when available (GPU or CPU), heuristic depth otherwise.
  * Per-scene atmosphere: volumetric light rays, mist, dust bokeh, rain, lightning, fireflies,
    insects, caustics, bubbles, heat haze, falling leaves, drips, cloud shadows, electric pulse.
  * Cinematic grade per mood, bloom / halation, vignette, film grain, chromatic aberration.
  * Arabic-first (RTL) motion typography: opening title, chapter cards, lower thirds,
    species files, survival-log HUD, journal map card, kinetic statements, end card.
  * Transitions: dissolve, dips, whip pans with motion blur, zoom-through, light leak,
    ink bleed, flash - driven by the storyboard.
  * Procedural sound-design bus (whooshes, booms, risers, UI blips, heartbeat, thunder,
    crackle, paper) mixed under the voice-over, two-pass loudness normalisation.
  * Cuts are snapped to pauses in the voice-over.
  * Resumable: renders per-scene segments into a work folder; re-running skips finished ones.

CLI (backwards compatible with the previous engine):
  python render_amazon_documentary.py --images DIR --audio VO.wav --output OUT.mp4 [--manifest M.json] [--preview]
Extra:
  --check                 only map images -> scenes and write a contact sheet, then exit
  --scenes 1-6,17,66-67   render only these scenes (full quality) with their audio
  --work-dir DIR          where segments / caches go (default: next to the output)
  --depth auto|model|heuristic|off
  --no-sfx  --sfx-db -3   --music FILE  --fps 24  --crf 18  --preset medium
  --export-manifest FILE  write the built-in cinematic manifest to edit it
"""

import argparse
import bisect
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.stderr.write("OpenCV is required:  pip install opencv-python-headless\n")
    raise

from PIL import Image, ImageDraw, ImageFont, features

cv2.setNumThreads(max(1, os.cpu_count() or 1))

VERSION = "3.0.0-fieldjournal"
REF_W = 1920.0
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ----------------------------------------------------------------------------------------------
# Palette (linear 0..1 RGB) - "field journal": ink, cream paper, explorer-orange accent
# ----------------------------------------------------------------------------------------------
INK = (0.055, 0.060, 0.052)
CREAM = (0.955, 0.925, 0.855)
WHITE = (0.975, 0.970, 0.950)
ACCENT = (0.925, 0.470, 0.190)
GOLD = (0.890, 0.735, 0.440)
MUTED = (0.760, 0.790, 0.735)
DANGER = (0.950, 0.330, 0.240)
TEAL = (0.470, 0.800, 0.740)
SEPIA = (0.230, 0.180, 0.120)


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------------------------
# Math helpers & easing
# ----------------------------------------------------------------------------------------------
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def seg(t, a, b):
    if b <= a:
        return 1.0 if t >= b else 0.0
    return clamp((t - a) / (b - a))


def ease_out_expo(t):
    t = clamp(t)
    return 1.0 if t >= 1 else 1 - 2 ** (-10 * t)


def ease_in_expo(t):
    t = clamp(t)
    return 0.0 if t <= 0 else 2 ** (10 * (t - 1))


def ease_in_out_expo(t):
    t = clamp(t)
    if t <= 0 or t >= 1:
        return t
    return 2 ** (20 * t - 10) / 2 if t < 0.5 else (2 - 2 ** (-20 * t + 10)) / 2


def ease_out_cubic(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def ease_in_cubic(t):
    t = clamp(t)
    return t * t * t


def ease_in_out_cubic(t):
    t = clamp(t)
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_in_out_sine(t):
    t = clamp(t)
    return -(math.cos(math.pi * t) - 1) / 2


def ease_out_back(t, s=1.5):
    t = clamp(t) - 1
    return t * t * ((s + 1) * t + s) + 1


def smooth(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def wobble(t, seed, speed=1.0):
    """Smooth pseudo-random signal in ~[-1, 1] (sum of incommensurate sines)."""
    r = np.random.default_rng(seed)
    ph = r.random(4) * 6.283
    f = np.array([0.31, 0.53, 0.97, 1.71]) * speed
    w = np.array([0.45, 0.3, 0.17, 0.08])
    return float(np.sum(w * np.sin(f * t * 6.283 + ph)))


def fractal_noise(h, w, rng, octaves=4, base=3.0, persistence=0.5):
    """Tileable fractal value noise in [0, 1]."""
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    big = max(h, w)
    for o in range(octaves):
        gh = max(2, int(round(base * (2 ** o) * h / big)))
        gw = max(2, int(round(base * (2 ** o) * w / big)))
        g = rng.random((gh, gw)).astype(np.float32)
        t = np.tile(g, (3, 3))
        up = cv2.resize(t, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)[h:2 * h, w:2 * w]
        acc += up * amp
        tot += amp
        amp *= persistence
    acc /= tot
    lo, hi = np.percentile(acc, 1), np.percentile(acc, 99)
    return np.clip((acc - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def luma(img):
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


# ----------------------------------------------------------------------------------------------
# Arabic text helpers
# ----------------------------------------------------------------------------------------------
AR_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
AR_ORD = ["الأول", "الثاني", "الثالث", "الرابع", "الخامس", "السادس", "السابع", "الثامن", "التاسع",
          "العاشر", "الحادي عشر", "الثاني عشر", "الثالث عشر", "الرابع عشر", "الخامس عشر", "السادس عشر"]
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI"]


def has_ar(s):
    return bool(AR_RE.search(s or ""))


def ar_num(n):
    return "".join(AR_DIGITS[int(c)] if c.isdigit() else c for c in str(n))


# ----------------------------------------------------------------------------------------------
# Fonts
# ----------------------------------------------------------------------------------------------
FONT_BASE = "https://raw.githubusercontent.com/google/fonts/main/"
FONT_SOURCES = {
    "ruqaa": "ofl/arefruqaa/ArefRuqaa-Bold.ttf",
    "ar_black": "ofl/tajawal/Tajawal-ExtraBold.ttf",
    "ar_bold": "ofl/tajawal/Tajawal-Bold.ttf",
    "ar_med": "ofl/tajawal/Tajawal-Medium.ttf",
    "ar_light": "ofl/tajawal/Tajawal-Light.ttf",
    "mono": "ofl/ibmplexmono/IBMPlexMono-Medium.ttf",
    "serif_it": "ofl/cormorantgaramond/CormorantGaramond-Italic%5Bwght%5D.ttf",
    "cinzel": "ofl/cinzel/Cinzel%5Bwght%5D.ttf",
}
FONT_WEIGHT = {"serif_it": 600, "cinzel": 600}
SYS_FONTS = {
    "arabic": ["/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
               "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "latin": ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"],
}


class FontBook:
    def __init__(self, extra_dirs=(), download=True):
        here = Path(__file__).resolve().parent
        self.cache_dir = Path(os.environ.get("AMAZON_FONT_DIR", Path.home() / ".cache" / "amazon_doc_fonts"))
        self.dirs = [Path(d) for d in extra_dirs if d] + [here / "fonts", self.cache_dir]
        self.download = download
        self.paths, self.cache = {}, {}
        self.raqm = bool(features.check("raqm"))
        if hasattr(ImageFont, "Layout"):
            self.layout = ImageFont.Layout.RAQM if self.raqm else ImageFont.Layout.BASIC
        else:  # very old Pillow
            self.layout = ImageFont.LAYOUT_RAQM if self.raqm else ImageFont.LAYOUT_BASIC
        self.reshaper = None
        if not self.raqm:
            try:
                import arabic_reshaper
                from bidi.algorithm import get_display
                self.reshaper = (arabic_reshaper, get_display)
            except Exception:
                log("WARNING: Pillow has no RAQM and arabic-reshaper/python-bidi are missing -> "
                    "Arabic may render incorrectly.  pip install arabic-reshaper python-bidi")
        for k in FONT_SOURCES:
            self.paths[k] = self._resolve(k)
        if not self.raqm:
            # Without RAQM, Arabic is drawn from presentation-form glyphs: use fonts that contain them.
            amiri = self._resolve_extra("Amiri-Bold.ttf", "ofl/amiri/Amiri-Bold.ttf")
            amiri_r = self._resolve_extra("Amiri-Regular.ttf", "ofl/amiri/Amiri-Regular.ttf")
            for k in ("ruqaa", "ar_black", "ar_bold"):
                self.paths[k] = amiri or self.paths[k]
            for k in ("ar_med", "ar_light"):
                self.paths[k] = amiri_r or amiri or self.paths[k]
        log("Fonts: " + ", ".join(f"{k}={Path(p).name}" for k, p in self.paths.items()) +
            f" | Arabic shaping: {'RAQM' if self.raqm else ('reshaper' if self.reshaper else 'NONE')}")

    def _resolve(self, key):
        fname = FONT_SOURCES[key].split("/")[-1].replace("%5B", "[").replace("%5D", "]")
        for d in self.dirs:
            p = d / fname
            if p.exists() and p.stat().st_size > 10000:
                return str(p)
        if self.download:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                dst = self.cache_dir / fname
                urllib.request.urlretrieve(FONT_BASE + FONT_SOURCES[key], dst)
                if dst.stat().st_size > 10000:
                    return str(dst)
            except Exception as e:
                log(f"  font download failed for {fname}: {e}")
        group = "latin" if key in ("mono", "serif_it", "cinzel") else "arabic"
        for p in SYS_FONTS[group]:
            if Path(p).exists():
                return p
        return None

    def _resolve_extra(self, fname, src):
        for d in self.dirs:
            p = d / fname
            if p.exists() and p.stat().st_size > 10000:
                return str(p)
        if self.download:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                dst = self.cache_dir / fname
                urllib.request.urlretrieve(FONT_BASE + src, dst)
                if dst.stat().st_size > 10000:
                    return str(dst)
            except Exception:
                pass
        try:
            import matplotlib
            p = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
            if p.exists():
                return str(p)
        except Exception:
            pass
        return None

    def get(self, key, size):
        size = max(6, int(round(size)))
        ck = (key, size)
        if ck not in self.cache:
            p = self.paths.get(key)
            if p:
                f = ImageFont.truetype(p, size, layout_engine=self.layout)
                if key in FONT_WEIGHT:
                    try:
                        f.set_variation_by_axes([FONT_WEIGHT[key]])
                    except Exception:
                        pass
            else:
                f = ImageFont.load_default()
            self.cache[ck] = f
        return self.cache[ck]

    def shape(self, text):
        if not has_ar(text):
            return text, {}
        if self.raqm:
            return text, {"direction": "rtl", "language": "ar"}
        if self.reshaper:
            rs, gd = self.reshaper
            return gd(rs.reshape(text)), {}
        return text, {}


class Sprite:
    __slots__ = ("a", "w", "h", "pad", "base", "ink_w")

    def __init__(self, a, pad, base, ink_w):
        self.a, self.pad, self.base, self.ink_w = a, pad, base, ink_w
        self.h, self.w = a.shape[:2]


_SPRITES = {}


def text_sprite(fb, text, key, size, color=WHITE, shadow=0.55, glow=0.0, glow_color=None, tracking=0.0, halo=0.0):
    """Premultiplied RGBA float32 sprite. Anchor helpers: .pad (left ink edge), .base (baseline row)."""
    ck = (text, key, int(round(size)), tuple(round(c, 3) for c in color), round(shadow, 2), round(glow, 2),
          tuple(glow_color or ()), round(tracking, 3), round(halo, 2))
    if ck in _SPRITES:
        return _SPRITES[ck]
    font = fb.get(key, size)
    asc, desc = font.getmetrics()
    pad = int(size * 0.5) + 6
    if tracking and not has_ar(text):
        advs = [font.getlength(ch) for ch in text]
        ink_w = sum(advs) + tracking * size * max(0, len(text) - 1)
        W = int(math.ceil(ink_w)) + 2 * pad
        H = asc + desc + 2 * pad
        img = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(img)
        x = float(pad)
        for ch, adv in zip(text, advs):
            d.text((x, pad + asc), ch, font=font, fill=255, anchor="ls")
            x += adv + tracking * size
    else:
        t, kw = fb.shape(text)
        bb = font.getbbox(t, anchor="ls", **kw)
        ink_w = bb[2] - bb[0]
        W = int(math.ceil(ink_w)) + 2 * pad
        H = asc + desc + 2 * pad
        img = Image.new("L", (W, H), 0)
        ImageDraw.Draw(img).text((pad - bb[0], pad + asc), t, font=font, fill=255, anchor="ls", **kw)
    A = np.asarray(img, np.float32) / 255.0
    col = np.asarray(color, np.float32)
    rgb = A[..., None] * col
    alpha = A.copy()
    if glow > 0:
        g = cv2.GaussianBlur(A, (0, 0), max(1.0, size * 0.18)) * glow
        gc = np.asarray(glow_color or color, np.float32)
        rgb = rgb + g[..., None] * gc * (1 - alpha[..., None])
        alpha = alpha + g * (1 - alpha)
    if halo > 0:
        hl = cv2.GaussianBlur(cv2.dilate(A, np.ones((3, 3), np.uint8)), (0, 0), max(1.0, size * 0.16))
        hl = np.clip(hl * 2.2, 0, 1) * halo
        alpha = alpha + hl * (1 - alpha)
    if shadow > 0:
        s = cv2.GaussianBlur(A, (0, 0), max(1.0, size * 0.09))
        dy = max(1, int(size * 0.035))
        s = np.roll(s, dy, axis=0) * shadow
        alpha = alpha + s * (1 - alpha)
    spr = Sprite(np.dstack([rgb, alpha]).astype(np.float32), pad, pad + asc, ink_w)
    _SPRITES[ck] = spr
    return spr


# ----------------------------------------------------------------------------------------------
# Compositing primitives
# ----------------------------------------------------------------------------------------------
def blit(dst, arr, x, y, opacity=1.0, scale=1.0, ax=0.0, ay=0.0, mode="over"):
    """Composite premultiplied RGBA `arr` so that its local point (ax, ay) lands on dst (x, y)."""
    if opacity <= 0.003 or arr is None:
        return
    h, w = arr.shape[:2]
    H, W = dst.shape[:2]
    ox, oy = x - ax * scale, y - ay * scale
    ix, iy = int(math.floor(ox)), int(math.floor(oy))
    fx, fy = ox - ix, oy - iy
    if abs(scale - 1.0) > 1e-4 or fx > 1e-3 or fy > 1e-3:
        bw, bh = int(math.ceil(w * scale)) + 2, int(math.ceil(h * scale)) + 2
        M = np.float32([[scale, 0, fx], [0, scale, fy]])
        arr = cv2.warpAffine(arr, M, (bw, bh), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    else:
        bw, bh = w, h
    x0, y0, x1, y1 = max(ix, 0), max(iy, 0), min(ix + bw, W), min(iy + bh, H)
    if x1 <= x0 or y1 <= y0:
        return
    s = arr[y0 - iy:y1 - iy, x0 - ix:x1 - ix]
    d = dst[y0:y1, x0:x1]
    if mode == "add":
        d += s[..., :3] * opacity
    else:
        a = s[..., 3:4] * opacity
        d *= (1.0 - a)
        d += s[..., :3] * opacity


def sprite_wipe(spr_a, progress, soft, rtl=True, exit_progress=0.0):
    """Return a copy of a sprite revealed by a soft edge; exit hides it continuing the same direction."""
    w = spr_a.shape[1]
    xs = np.arange(w, dtype=np.float32)
    soft = max(1.0, soft)
    e = progress * (w + soft)
    m = np.clip((xs - (w - e)) / soft, 0, 1) if rtl else np.clip((e - xs) / soft, 0, 1)
    if exit_progress > 0:
        e2 = (1 - exit_progress) * (w + soft)
        m = m * (np.clip((e2 - xs) / soft, 0, 1) if rtl else np.clip((xs - (w - e2)) / soft, 0, 1))
    return spr_a * m[None, :, None]


def blur_sprite(a, sigma):
    if sigma < 0.35:
        return a
    return cv2.GaussianBlur(a, (0, 0), sigma)


SHIFT, ONE = 4, 16


def _pt(x, y):
    return (int(round(x * ONE)), int(round(y * ONE)))


def fill_mask(dst, m, x0, y0, color, opacity=1.0):
    H, W = dst.shape[:2]
    h, w = m.shape[:2]
    ax0, ay0, ax1, ay1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if ax1 <= ax0 or ay1 <= ay0 or opacity <= 0.003:
        return
    mm = m[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
    a = mm.astype(np.float32)[..., None] * (opacity / 255.0)
    d = dst[ay0:ay1, ax0:ax1]
    d += (np.asarray(color, np.float32) - d) * a


def draw_polyline(dst, pts, color, thick=2.0, opacity=1.0, closed=False, frac=1.0, dashed=0.0, mode="over"):
    """Antialiased (sub-pixel) polyline; frac<1 draws only the first part (stroke draw-on)."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 2 or opacity <= 0.003 or frac <= 0:
        return
    if closed:
        pts = np.vstack([pts, pts[:1]])
    if frac < 1.0 or dashed > 0:
        seglen = np.hypot(*np.diff(pts, axis=0).T)
        cum = np.concatenate([[0], np.cumsum(seglen)])
        L = cum[-1] * frac
        k = int(np.searchsorted(cum, L))
        if k <= 0:
            return
        k = min(k, len(pts) - 1)
        r = (L - cum[k - 1]) / max(seglen[k - 1], 1e-9)
        tail = pts[k - 1] + (pts[k] - pts[k - 1]) * min(1.0, r)
        pts = np.vstack([pts[:k], tail])
        if dashed > 0:
            # resample into dash segments
            seglen = np.hypot(*np.diff(pts, axis=0).T)
            cum = np.concatenate([[0], np.cumsum(seglen)])
            out = []
            s = 0.0
            while s < cum[-1]:
                e = min(s + dashed, cum[-1])
                out.append((np.array([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])]),
                            np.array([np.interp(e, cum, pts[:, 0]), np.interp(e, cum, pts[:, 1])])))
                s += dashed * 2
            for a, b in out:
                draw_polyline(dst, [a, b], color, thick, opacity, mode=mode)
            return
    pad = thick + 3
    x0, y0 = int(math.floor(pts[:, 0].min() - pad)), int(math.floor(pts[:, 1].min() - pad))
    x1, y1 = int(math.ceil(pts[:, 0].max() + pad)), int(math.ceil(pts[:, 1].max() + pad))
    H, W = dst.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    p = np.array([_pt(px - x0, py - y0) for px, py in pts], np.int32)
    cv2.polylines(m, [p], False, 255, max(1, int(round(thick))), cv2.LINE_AA, SHIFT)
    if mode == "add":
        dst[y0:y1, x0:x1] += m.astype(np.float32)[..., None] * (opacity / 255.0) * np.asarray(color, np.float32)
    else:
        fill_mask(dst, m, x0, y0, color, opacity)


def draw_line(dst, x0, y0, x1, y1, color, thick=2.0, opacity=1.0):
    draw_polyline(dst, [(x0, y0), (x1, y1)], color, thick, opacity)


def draw_circle(dst, cx, cy, r, color, thick=-1, opacity=1.0, mode="over"):
    if opacity <= 0.003 or r <= 0.2:
        return
    pad = int(r + abs(thick) + 3)
    x0, y0 = int(cx) - pad, int(cy) - pad
    m = np.zeros((2 * pad + 1, 2 * pad + 1), np.uint8)
    cv2.circle(m, _pt(cx - x0, cy - y0), int(round(r * ONE)), 255,
               -1 if thick < 0 else max(1, int(round(thick))), cv2.LINE_AA, SHIFT)
    if mode == "add":
        H, W = dst.shape[:2]
        ax0, ay0, ax1, ay1 = max(x0, 0), max(y0, 0), min(x0 + m.shape[1], W), min(y0 + m.shape[0], H)
        if ax1 > ax0 and ay1 > ay0:
            mm = m[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0].astype(np.float32)[..., None] / 255.0
            dst[ay0:ay1, ax0:ax1] += mm * opacity * np.asarray(color, np.float32)
    else:
        fill_mask(dst, m, x0, y0, color, opacity)


def draw_rect(dst, x0, y0, x1, y1, color, opacity=1.0):
    """Axis aligned filled rectangle with fractional (anti-aliased) edges."""
    if opacity <= 0.003 or x1 <= x0 or y1 <= y0:
        return
    H, W = dst.shape[:2]
    ix0, iy0 = max(int(math.floor(x0)), 0), max(int(math.floor(y0)), 0)
    ix1, iy1 = min(int(math.ceil(x1)), W), min(int(math.ceil(y1)), H)
    if ix1 <= ix0 or iy1 <= iy0:
        return
    xs = np.arange(ix0, ix1, dtype=np.float32)
    ys = np.arange(iy0, iy1, dtype=np.float32)
    cx = np.clip(np.minimum(xs + 1, x1) - np.maximum(xs, x0), 0, 1)
    cy = np.clip(np.minimum(ys + 1, y1) - np.maximum(ys, y0), 0, 1)
    a = (cy[:, None] * cx[None, :] * opacity)[..., None]
    d = dst[iy0:iy1, ix0:ix1]
    d += (np.asarray(color, np.float32) - d) * a


_RR = {}


def rrect_mask(w, h, r):
    w, h, r = int(w), int(h), int(min(r, w // 2, h // 2))
    k = (w, h, r)
    if k not in _RR:
        m = np.zeros((h, w), np.uint8)
        if w > 0 and h > 0:
            cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1)
            cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 255, -1)
            for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
                cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
        _RR[k] = m
    return _RR[k]


def glass(dst, x0, y0, w, h, S, opacity=1.0, radius=14, tint=(0.018, 0.032, 0.028), tint_a=0.56,
          border=0.22, blur=20, clip_w=None):
    """Frosted-glass panel (background blur + tint + hairline border). clip_w reveals from the right."""
    x0, y0, w, h = int(round(x0)), int(round(y0)), int(round(w)), int(round(h))
    if w < 4 or h < 4 or opacity <= 0.003:
        return
    H, W = dst.shape[:2]
    m = rrect_mask(w, h, radius * S).astype(np.float32) / 255.0
    if clip_w is not None:
        cw = clamp(clip_w, 0, 1) * w
        xs = np.arange(w, dtype=np.float32)
        m = m * np.clip((xs - (w - cw)) / 2.0, 0, 1)[None, :]
    ax0, ay0, ax1, ay1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if ax1 <= ax0 or ay1 <= ay0:
        return
    reg = dst[ay0:ay1, ax0:ax1]
    rh, rw = reg.shape[:2]
    small = cv2.resize(reg, (max(1, rw // 4), max(1, rh // 4)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(1.0, blur * S / 4))
    bl = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR)
    g = bl * (1 - tint_a) + np.asarray(tint, np.float32) * tint_a
    sheen = np.linspace(0.035, 0.0, rh, dtype=np.float32)[:, None, None]
    g = g + sheen
    mm = m[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0][..., None] * opacity
    reg += (g - reg) * mm
    if border > 0:
        inner = np.zeros((h, w), np.uint8)
        t = max(1, int(round(1.2 * S)))
        if w > 2 * t and h > 2 * t:
            inner[t:h - t, t:w - t] = rrect_mask(w - 2 * t, h - 2 * t, max(1, radius * S - t))
        ring = np.clip(rrect_mask(w, h, radius * S).astype(np.int16) - inner, 0, 255).astype(np.uint8)
        if clip_w is not None:
            ring = (ring.astype(np.float32) * np.clip((np.arange(w) - (w - clamp(clip_w, 0, 1) * w)) / 2.0, 0, 1)[None, :]).astype(np.uint8)
        fill_mask(dst, ring, x0, y0, (1, 1, 1), border * opacity)


def soft_shadow_sprite(w, h, strength=1.0):
    """Elliptical soft dark blob used behind lower thirds."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    a = np.clip(1 - d, 0, 1) ** 1.6 * strength
    return np.dstack([np.zeros((h, w, 3), np.float32), a]).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# Image analysis: printed scene label (detect + OCR + remove), subject (explorer head)
# ----------------------------------------------------------------------------------------------
def imread_rgb(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


_DIGIT_T = None


def _norm_glyph(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    g = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    h, w = g.shape
    th, tw = 32, 26
    sc = th / h
    nw = max(1, min(tw, int(round(w * sc))))
    g = cv2.resize(g, (nw, th), interpolation=cv2.INTER_AREA)
    out = np.zeros((th, tw), np.float32)
    x0 = (tw - nw) // 2
    out[:, x0:x0 + nw] = g
    out = cv2.GaussianBlur(out, (0, 0), 0.8)
    v = out - out.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def _digit_templates(fb=None):
    global _DIGIT_T
    if _DIGIT_T is not None:
        return _DIGIT_T
    cands = ["/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
             "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
             "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
             "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf"]
    if fb:
        for k in ("ar_bold", "ar_black", "mono"):
            if fb.paths.get(k):
                cands.append(fb.paths[k])
    T = []
    for p in cands:
        if not Path(p).exists():
            continue
        try:
            f = ImageFont.truetype(p, 64)
        except Exception:
            continue
        for d in "0123456789":
            img = Image.new("L", (90, 110), 0)
            ImageDraw.Draw(img).text((10, 90), d, font=f, fill=255, anchor="ls")
            v = _norm_glyph(np.asarray(img) > 127)
            if v is not None:
                T.append((int(d), v))
    _DIGIT_T = T
    return T


def find_label(img, fb=None):
    """Detect the printed scene number (white digits on a dark box near the top-left)."""
    H, W = img.shape[:2]
    rh, rw = int(H * 0.26), int(W * 0.24)
    roi = img[:rh, :rw].astype(np.int16)
    mn, mx = roi.min(2), roi.max(2)
    white = ((mn > 175) & (mx - mn < 50)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(white, 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h < H * 0.025 or h > H * 0.15 or w > h * 1.15 or w < h * 0.12 or a < 0.10 * w * h:
            continue
        comps.append((x, y, w, h, i))
    best = None
    for c in comps:
        grp = [d for d in comps if abs(d[1] - c[1]) < c[3] * 0.22 and abs(d[3] - c[3]) < c[3] * 0.22]
        grp.sort(key=lambda d: d[0])
        # keep a horizontally contiguous run
        run = [grp[0]]
        for d in grp[1:]:
            if d[0] - (run[-1][0] + run[-1][2]) < c[3] * 0.8:
                run.append(d)
            elif len(run) < 2:
                run = [d]
            else:
                break
        if 1 <= len(run) <= 4 and (best is None or len(run) > len(best)):
            best = run
    if not best:
        return None
    x0 = min(d[0] for d in best)
    y0 = min(d[1] for d in best)
    x1 = max(d[0] + d[2] for d in best)
    y1 = max(d[1] + d[3] for d in best)
    dh = y1 - y0
    pad = int(dh * 0.62) + 5
    bx0, by0, bx1, by1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    # validate: the box around digits must be dark & neutral
    box = img[by0:by1, bx0:bx1].astype(np.int16)
    nonwhite = ~((box.min(2) > 175) & (box.max(2) - box.min(2) < 50))
    if nonwhite.sum() == 0 or box.max(2)[nonwhite].mean() > 135:
        return None
    # OCR
    T = _digit_templates(fb)
    digits, conf = [], 1.0
    for (x, y, w, h, i) in best:
        v = _norm_glyph(lab[y:y + h, x:x + w] == i)
        if v is None or not T:
            digits = None
            break
        scores = {}
        for d, t in T:
            s = float((v * t).sum())
            if s > scores.get(d, -9):
                scores[d] = s
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        digits.append(ranked[0][0])
        margin = ranked[0][1] - ranked[1][1]
        conf = min(conf, ranked[0][1] * 0.5 + min(margin, 0.3))
    number = int("".join(map(str, digits))) if digits else None
    return {"box": (bx0, by0, bx1, by1), "number": number, "conf": conf, "ndigits": len(best)}


def remove_label(img, box):
    """Content-aware fill of the printed label (frequency separation):
    low frequencies are inpainted from the surroundings, fine texture is borrowed from the
    most similar nearby patch (never from bright/white areas such as the explorer's head)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    e = 3
    x0e, y0e, x1e, y1e = max(0, x0 - e), max(0, y0 - e), min(W, x1 + e), min(H, y1 + e)
    bw, bh = x1e - x0e, y1e - y0e
    f = img.astype(np.float32)
    mask = np.zeros((H, W), np.uint8)
    mask[y0e:y1e, x0e:x1e] = 255
    ys, xs = np.arange(y0e, y1e), np.arange(x0e, x1e)
    # mirror of the area just below the box (continues vertical structures) and just right of it
    vm = f[np.clip(2 * y1e - 1 - ys, 0, H - 1)][:, x0e:x1e]
    hm = f[y0e:y1e][:, np.clip(2 * x1e - 1 - xs, 0, W - 1)]
    # low-frequency fallback
    q = 4
    sm = cv2.resize(img, (W // q, H // q), interpolation=cv2.INTER_AREA)
    msm = cv2.dilate(cv2.resize(mask, (W // q, H // q), interpolation=cv2.INTER_NEAREST), np.ones((3, 3), np.uint8))
    low = cv2.resize(cv2.inpaint(sm, msm, 6, cv2.INPAINT_TELEA), (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    low = cv2.GaussianBlur(low, (0, 0), 3.0)[y0e:y1e, x0e:x1e]
    dv = (y1e - 1 - ys).astype(np.float32)[:, None]
    dh = (x1e - 1 - xs).astype(np.float32)[None, :]
    soft = 0.12 * max(bw, bh)
    # which mirror is more consistent with the *other* boundary? (e.g. sky must not get forest)
    err_v = float(np.abs(vm[:, -1] - f[y0e:y1e, min(W - 1, x1e)]).mean()) if x1e < W else 0.0
    err_h = float(np.abs(hm[-1, :] - f[min(H - 1, y1e), x0e:x1e]).mean()) if y1e < H else 0.0
    bias = (err_h - err_v) / (err_h + err_v + 1e-6)
    wv = 1.0 / (1.0 + np.exp(-((dh - dv) + bias * max(bw, bh) * 1.6) / soft))
    wh = 1.0 - wv
    # never copy the explorer's white head / very bright pixels into the corner
    def _ok(a):
        white = (a.min(2) > 226) & (a.max(2) - a.min(2) < 20)
        return cv2.erode((~white).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)
    okv, okh = _ok(vm), _ok(hm)
    okv, okh = cv2.GaussianBlur(okv, (0, 0), 2), cv2.GaussianBlur(okh, (0, 0), 2)
    wv, wh = wv * okv, wh * okh
    wl = np.full_like(wv, 0.02) + (1 - np.clip(wv + wh, 0, 1)) * 0.98
    tot = wv + wh + wl
    fill = (vm * wv[..., None] + hm * wh[..., None] + low * wl[..., None]) / tot[..., None]
    out = f.copy()
    out[y0e:y1e, x0e:x1e] = fill
    # match the remaining boundaries (when the box does not touch the frame edge) with Poisson blending
    if True:
        try:
            m = np.zeros((H, W), np.uint8)
            m[max(1, y0e):min(H - 1, y1e), max(1, x0e):min(W - 1, x1e)] = 255
            ctr = ((max(1, x0e) + min(W - 1, x1e)) // 2, (max(1, y0e) + min(H - 1, y1e)) // 2)
            out = cv2.seamlessClone(np.clip(out, 0, 255).astype(np.uint8), img, m, ctr, cv2.NORMAL_CLONE).astype(np.float32)
        except Exception:
            pass
    m2 = cv2.dilate(mask, np.ones((3, 3), np.uint8)).astype(np.float32) / 255.0
    mf = cv2.GaussianBlur(m2, (0, 0), 1.5)[..., None]
    out = f * (1 - mf) + out * mf
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def find_head(img, exclude=None):
    """Locate the explorer's white round head (normalised x, y, r) - used as camera focus."""
    H, W = img.shape[:2]
    sc = 480.0 / W
    sm = cv2.resize(img, (480, int(H * sc)), interpolation=cv2.INTER_AREA).astype(np.int16)
    mn, mx = sm.min(2), sm.max(2)
    m = ((mn > 212) & (mx - mn < 32)).astype(np.uint8)
    if exclude is not None:
        x0, y0, x1, y1 = [int(v * sc) for v in exclude]
        m[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    lum = sm.mean(2)
    best, bs = None, 0
    tot = m.shape[0] * m.shape[1]
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < tot * 0.00025 or w < 4 or h < 4:
            continue
        asp = w / float(h)
        if asp < 0.6 or asp > 1.6:
            continue
        fill = a / (math.pi / 4 * w * h)
        if fill < 0.68:
            continue
        # needs a dark ink outline around it
        pad = max(2, int(0.18 * max(w, h)))
        X0, Y0, X1, Y1 = max(0, x - pad), max(0, y - pad), min(m.shape[1], x + w + pad), min(m.shape[0], y + h + pad)
        ringm = np.ones((Y1 - Y0, X1 - X0), bool)
        ringm[(y - Y0):(y - Y0 + h), (x - X0):(x - X0 + w)] = False
        if ringm.sum() == 0:
            continue
        rl = lum[Y0:Y1, X0:X1][ringm].mean()
        if rl > 165:
            continue
        score = a * fill
        if score > bs:
            bs, best = score, (cen[i][0] / sm.shape[1], cen[i][1] / sm.shape[0], max(w, h) / 2 / sm.shape[1])
    return best


# ----------------------------------------------------------------------------------------------
# Depth
# ----------------------------------------------------------------------------------------------
class DepthEstimator:
    def __init__(self, mode="auto"):
        self.mode, self.pipe = mode, None
        if mode in ("auto", "model"):
            try:
                import torch
                from transformers import pipeline
                dev = 0 if torch.cuda.is_available() else -1
                self.pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=dev)
                log(f"Depth: Depth-Anything-V2-Small on {'GPU' if dev == 0 else 'CPU'}")
            except Exception as e:
                if mode == "model":
                    raise
                log(f"Depth: model unavailable ({type(e).__name__}) -> heuristic depth")
        else:
            log(f"Depth: {mode}")
        self.is_model = self.pipe is not None

    def __call__(self, img):
        """img: RGB uint8. Returns float32 depth in [0,1] (1 = near), same HxW."""
        H, W = img.shape[:2]
        if self.mode == "off":
            return np.full((H, W), 0.5, np.float32)
        d = None
        if self.pipe is not None:
            try:
                out = self.pipe(Image.fromarray(img))
                d = np.asarray(out["depth"], np.float32)
                if d.shape[:2] != (H, W):
                    d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                log(f"  depth model failed on image ({e}); heuristic used")
                d = None
        if d is None:
            d = heuristic_depth(img)
        lo, hi = np.percentile(d, 2), np.percentile(d, 98)
        d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
        k = max(3, int(W * 0.012)) | 1
        d = cv2.dilate(d, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        d = cv2.GaussianBlur(d, (0, 0), W * 0.010)
        return d.astype(np.float32)


def heuristic_depth(img):
    H, W = img.shape[:2]
    sw = 384
    sm = cv2.resize(img, (sw, int(H * sw / W)), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    h, w = sm.shape[:2]
    lum = luma(sm)
    haze = cv2.GaussianBlur(lum, (0, 0), w * 0.04)
    sat = cv2.GaussianBlur(sm.max(2) - sm.min(2), (0, 0), w * 0.03)
    mu = cv2.GaussianBlur(lum, (0, 0), w * 0.02)
    var = np.maximum(cv2.GaussianBlur(lum * lum, (0, 0), w * 0.02) - mu * mu, 0)
    con = np.sqrt(var)

    def nz(x):
        lo, hi = np.percentile(x, 2), np.percentile(x, 98)
        return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)

    yy = (np.linspace(0, 1, h, dtype=np.float32) ** 1.25)[:, None] * np.ones((1, w), np.float32)
    near = 0.36 * (1 - nz(haze)) + 0.20 * nz(con) + 0.12 * nz(sat) + 0.32 * yy
    return cv2.resize(near, (W, H), interpolation=cv2.INTER_CUBIC)


# ----------------------------------------------------------------------------------------------
# Grading
# ----------------------------------------------------------------------------------------------
LOOKS = {
    #            exp  contrast sat   gain(r,g,b)            lift(r,g,b)              shadow tint          hi tint            bloom vig
    "day":       (1.00, 1.08, 1.05, (1.02, 1.00, 0.97), (0.010, 0.016, 0.014), (-0.01, 0.01, 0.015), (0.03, 0.015, -0.01), 0.16, 0.30),
    "day_mist":  (1.02, 1.04, 1.00, (1.01, 1.00, 0.98), (0.018, 0.024, 0.022), (-0.01, 0.01, 0.02), (0.03, 0.02, 0.00), 0.24, 0.30),
    "warm":      (1.02, 1.06, 1.06, (1.05, 1.00, 0.93), (0.016, 0.014, 0.008), (0.00, 0.005, 0.01), (0.04, 0.02, -0.02), 0.20, 0.28),
    "heat":      (1.03, 1.00, 0.96, (1.07, 1.01, 0.90), (0.03, 0.025, 0.012), (0.01, 0.005, 0.00), (0.05, 0.03, -0.02), 0.26, 0.34),
    "tension":   (0.95, 1.14, 0.88, (1.00, 1.00, 1.00), (0.004, 0.010, 0.012), (-0.015, 0.0, 0.02), (0.02, 0.01, -0.01), 0.12, 0.48),
    "dusk":      (0.90, 1.08, 0.95, (1.08, 0.98, 0.88), (0.006, 0.008, 0.016), (-0.01, 0.0, 0.025), (0.05, 0.02, -0.03), 0.20, 0.42),
    "night":     (0.80, 1.10, 0.78, (0.84, 0.93, 1.10), (0.000, 0.010, 0.026), (-0.02, 0.0, 0.035), (0.00, 0.01, 0.03), 0.14, 0.52),
    "rain":      (0.92, 1.00, 0.86, (0.95, 0.99, 1.04), (0.012, 0.018, 0.026), (-0.01, 0.005, 0.025), (0.00, 0.01, 0.02), 0.16, 0.40),
    "underwater": (0.94, 1.02, 0.92, (0.86, 1.02, 0.98), (0.004, 0.024, 0.026), (-0.02, 0.01, 0.03), (0.00, 0.03, 0.02), 0.26, 0.42),
    "dawn":      (1.02, 1.04, 1.02, (1.07, 1.00, 0.91), (0.024, 0.020, 0.016), (0.00, 0.005, 0.02), (0.06, 0.03, -0.02), 0.30, 0.30),
    "aerial":    (1.00, 1.06, 1.00, (1.00, 1.00, 1.00), (0.016, 0.020, 0.024), (-0.01, 0.0, 0.02), (0.02, 0.015, 0.0), 0.16, 0.26),
    "interior":  (1.00, 1.06, 1.02, (1.10, 0.99, 0.86), (0.012, 0.010, 0.006), (0.00, 0.0, 0.01), (0.05, 0.02, -0.03), 0.30, 0.36),
}
MIST_COLOR = {"night": (0.30, 0.36, 0.44), "dusk": (0.62, 0.58, 0.55), "rain": (0.62, 0.66, 0.68),
              "dawn": (0.90, 0.84, 0.74), "aerial": (0.84, 0.86, 0.86)}


def grade(img, mood, W_ref):
    L = LOOKS.get(mood, LOOKS["day"])
    exp, con, sat, gain, lift, sh_t, hi_t, bloom, _ = L
    x = img.astype(np.float32) / 255.0
    x *= exp
    x = x * np.asarray(gain, np.float32) + np.asarray(lift, np.float32) * (1 - x)
    l = luma(x)[..., None]
    x = l + (x - l) * sat
    # filmic S-curve around pivot
    piv = 0.46
    x = piv + (x - piv) * con
    x = np.clip(x, 0, 1)
    x = x * x * (3 - 2 * x) * 0.22 + x * 0.78
    l = luma(x)[..., None]
    x = x + np.asarray(sh_t, np.float32) * (1 - l) ** 2 + np.asarray(hi_t, np.float32) * l ** 2
    x = np.clip(x, 0, 1)
    if bloom > 0:
        h, w = x.shape[:2]
        small = cv2.resize(x, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        hi = np.clip((luma(small) - 0.62) / 0.38, 0, 1)[..., None] * small
        b1 = cv2.GaussianBlur(hi, (0, 0), W_ref * 0.006)
        b2 = cv2.GaussianBlur(hi, (0, 0), W_ref * 0.022)
        b = cv2.resize(b1 * 0.55 + b2 * 0.45, (w, h), interpolation=cv2.INTER_LINEAR)
        hal = cv2.resize(cv2.GaussianBlur(hi, (0, 0), W_ref * 0.003), (w, h), interpolation=cv2.INTER_LINEAR)
        x = 1 - (1 - x) * (1 - b * bloom)
        x = x + hal * np.float32([0.07, 0.02, 0.0]) * (bloom * 3)
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def vignette_map(W, H, strength):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.70)) ** 2)
    v = 1 - strength * np.clip(r - 0.42, 0, None) ** 1.7
    return np.clip(v, 1 - strength, 1)[..., None].astype(np.float32)


# ----------------------------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------------------------
class Camera:
    def __init__(self, spec, seed, focus, par_conf):
        self.move = spec.get("move", "push_in")
        self.amt = float(spec.get("amount", 1.0))
        self.hh = float(spec.get("handheld", 0.0))
        self.breathe = bool(spec.get("breathe", False))
        self.P = float(spec.get("parallax", 1.0)) * par_conf
        self.fx, self.fy = focus
        self.seed = seed

    def state(self, p, tl):
        m = clamp(p, -0.3, 1.3)
        mv, a, P = self.move, self.amt, self.P
        fdx, fdy = (self.fx - 0.5), (self.fy - 0.5)
        zn = zf = 1.04
        cx = cy = rot = px = py = 0.0
        if mv == "push_in":
            zf = 1.0 + a * m
            zn = 1.0 + a * m * (1 + 0.8 * P)
            cx, cy = fdx * 0.32 * m, fdy * 0.32 * m
        elif mv == "pull_out":
            zf = 1.0 + a * (1 - m)
            zn = 1.0 + a * (1 - m) * (1 + 0.8 * P)
            cx, cy = fdx * 0.32 * (1 - m), fdy * 0.32 * (1 - m)
        elif mv in ("pan_left", "pan_right"):
            d = 1 if mv == "pan_right" else -1
            zn = zf = 1.075
            cx = d * 0.045 * a * (m - 0.5) * 2
            px = d * 0.010 * a * P * (m - 0.5) * 2
            cy = fdy * 0.12
        elif mv in ("tilt_up", "tilt_down"):
            d = 1 if mv == "tilt_down" else -1
            zn = zf = 1.075
            cy = d * 0.042 * a * (m - 0.5) * 2
            py = d * 0.006 * P * (m - 0.5) * 2
            cx = fdx * 0.12
        elif mv in ("crane_up", "crane_down"):
            d = 1 if mv == "crane_down" else -1
            zf = 1.05 + 0.025 * m
            zn = zf + 0.02 * m * P
            py = d * 0.014 * a * P * (m - 0.5) * 2
            cy = d * 0.022 * a * (m - 0.5) * 2
            cx = fdx * 0.12
        elif mv in ("orbit_left", "orbit_right"):
            d = 1 if mv == "orbit_right" else -1
            zn = zf = 1.065 + 0.02 * m
            px = d * 0.017 * a * P * (m - 0.5) * 2
            cx = -d * 0.012 * a * (m - 0.5) * 2 + fdx * 0.12
            cy = fdy * 0.1
            rot = d * math.radians(0.45) * a * (m - 0.5) * 2
        elif mv == "dolly_zoom":
            zn = 1.05 + 0.01 * m
            zf = 1.05 + 0.14 * a * m * max(P, 0.5)
            cx, cy = fdx * 0.25 * m, fdy * 0.25 * m
        elif mv == "rotate":
            zn = zf = 1.08 + 0.035 * m
            rot = math.radians(1.6) * a * (m - 0.5) * 2
        elif mv == "drift":
            zf = 1.035 + 0.03 * m
            zn = zf + 0.012 * m * P
            cx = 0.016 * a * (m - 0.5)
            px = 0.007 * P * (m - 0.5) * 2
        else:  # static
            zn = zf = 1.035 + 0.01 * m
            cx = 0.004 * (m - 0.5)
        if self.breathe:
            b = 1 + 0.0045 * math.sin(tl * 6.283 * 0.21)
            zn *= b
            zf *= b
        hx = hy = 0.0
        if self.hh > 0:
            hx = self.hh * 0.0018 * wobble(tl, self.seed, 0.9)
            hy = self.hh * 0.0015 * wobble(tl, self.seed + 7, 0.8)
            rot += self.hh * math.radians(0.16) * wobble(tl, self.seed + 13, 0.6)
        return {"zn": zn, "zf": zf, "cx": cx, "cy": cy, "rot": rot, "px": px, "py": py, "hx": hx, "hy": hy}


# ----------------------------------------------------------------------------------------------
# Atmosphere FX
# ----------------------------------------------------------------------------------------------
class FXBase:
    needs_q = False   # needs quarter-res screen maps (rays) / depth

    def distort(self, mx, my, ctx):
        pass

    def lf(self, ctx, M, A):
        """Low-frequency contribution at quarter res: frame = frame * M + A."""
        pass

    def draw(self, frame, ctx):
        pass


def _soft_disc(r, sigma, core=0.0):
    R = int(math.ceil(r + 3 * sigma + 2))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    d = np.sqrt(xx * xx + yy * yy)
    a = np.clip(r - d + 0.5, 0, 1)
    if sigma > 0.3:
        a = cv2.GaussianBlur(a, (0, 0), sigma)
    if core > 0:
        a = a * (1 - core) + np.exp(-(d / max(1.0, r * 0.35)) ** 2) * core
    a = a / max(a.max(), 1e-6)
    return a, R


class Particles(FXBase):
    """Dust motes / pollen with depth-of-field bokeh, or fireflies."""

    def __init__(self, W, H, S, seed, kind="dust", n=None, strength=1.0, color=None):
        self.W, self.H, self.S, self.kind, self.strength = W, H, S, kind, strength
        r = np.random.default_rng(seed)
        if kind == "fireflies":
            n = n or 26
            self.color = np.float32(color or (0.85, 1.0, 0.45))
        else:
            n = n or 70
            self.color = np.float32(color or (1.0, 0.95, 0.80))
        self.n = n
        self.z = r.random(n) ** 1.7
        self.x0, self.y0 = r.random(n), r.random(n)
        self.vx = (r.random(n) - 0.5) * 0.010
        self.vy = -(0.002 + r.random(n) * 0.008) if kind != "fireflies" else (r.random(n) - 0.5) * 0.01
        self.ph = r.random(n) * 6.283
        self.fr = 0.15 + r.random(n) * 0.5
        self.bank = []
        for zb in range(8):
            z = (zb + 0.5) / 8
            if kind == "fireflies":
                rad, sig, core = (1.3 + z * 3.0) * S, (0.8 + z * 2.5) * S, 0.6
            else:
                rad, sig, core = (0.9 + z * 7.5) * S, (0.3 + z * 3.2) * S, 0.0
            a, R = _soft_disc(rad, sig, core)
            if kind == "fireflies":
                halo, R2 = _soft_disc(rad * 3.5, sig * 3, 0)
                pad = R2 - R
                a = np.pad(a, pad) + halo * 0.35
                R = R2
            self.bank.append((a.astype(np.float32), R))

    def draw(self, frame, ctx):
        t, W, H = ctx["tl"], self.W, self.H
        camx, camy = ctx["cam_dx"], ctx["cam_dy"]
        fade = ctx.get("fx_fade", 1.0) * self.strength
        for i in range(self.n):
            z = self.z[i]
            if self.kind == "fireflies":
                x = self.x0[i] + 0.035 * math.sin(self.ph[i] + t * 0.23 * (1 + z)) + self.vx[i] * t
                y = self.y0[i] + 0.03 * math.sin(self.ph[i] * 1.7 + t * 0.31) + self.vy[i] * t
                blink = max(0.0, math.sin(self.ph[i] + t * self.fr[i] * 2.2)) ** 3
                b = 0.9 * blink
            else:
                x = self.x0[i] + self.vx[i] * t + 0.012 * math.sin(self.ph[i] + t * 0.4)
                y = self.y0[i] + self.vy[i] * t
                b = (0.35 + 0.65 * (0.5 + 0.5 * math.sin(self.ph[i] + t * self.fr[i] * 6.283 * 0.25))) * (0.35 + 0.5 * z)
            X = ((x % 1.0) * 1.1 - 0.05) * W - camx * (0.4 + z * 1.6)
            Y = ((y % 1.0) * 1.1 - 0.05) * H - camy * (0.4 + z * 1.6)
            if b * fade < 0.02:
                continue
            a, R = self.bank[min(7, int(z * 8))]
            spr = np.dstack([a[..., None] * self.color, a])
            blit(frame, spr, round(X), round(Y), opacity=b * fade * (0.55 if self.kind == "dust" else 1.0), ax=R, ay=R, mode="add")


class Insects(FXBase):
    def __init__(self, W, H, S, seed, n=22):
        r = np.random.default_rng(seed)
        self.W, self.H, self.S, self.n = W, H, S, n
        self.cx, self.cy = r.random(n) * 0.9 + 0.05, r.random(n) * 0.7 + 0.1
        self.ax, self.ay = (0.01 + r.random(n) * 0.05), (0.01 + r.random(n) * 0.04)
        self.fx, self.fy = 1.5 + r.random(n) * 3.5, 1.2 + r.random(n) * 3.2
        self.ph = r.random(n) * 6.283
        self.sz = (0.9 + r.random(n) * 1.6)

    def draw(self, frame, ctx):
        t = ctx["tl"]
        dt = 1.0 / ctx["fps"]
        for i in range(self.n):
            def pos(tt):
                return ((self.cx[i] + self.ax[i] * math.sin(self.fx[i] * tt + self.ph[i]) + 0.02 * math.sin(tt * 0.3 + i)) * self.W - ctx["cam_dx"],
                        (self.cy[i] + self.ay[i] * math.sin(self.fy[i] * tt * 1.3 + self.ph[i] * 2) + 0.015 * math.cos(tt * 0.4 + i)) * self.H - ctx["cam_dy"])
            x0, y0 = pos(t - dt * 0.6)
            x1, y1 = pos(t)
            draw_polyline(frame, [(x0, y0), (x1, y1)], (0.03, 0.03, 0.025), max(1.0, self.sz[i] * self.S * 1.6), 0.7)


class Rays(FXBase):
    """Volumetric light rays derived from the plate's own highlights (CC Light Rays style)."""
    needs_q = True

    def __init__(self, W, H, S, seed, plate_q, strength=0.55, fade=False, color=(1.0, 0.94, 0.78)):
        self.strength, self.fade, self.seed = strength, fade, seed
        x = plate_q.astype(np.float32) / 255.0
        h, w = x.shape[:2]
        l = luma(x)
        hi = np.clip((l - 0.50) / 0.5, 0, 1) ** 1.4
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        hi = hi * np.clip(1.15 - yy * 1.1, 0, 1)
        # light source: brightest upper region
        col = cv2.GaussianBlur(hi, (0, 0), w * 0.06)
        top = col[: max(2, h // 3)]
        yx = np.unravel_index(np.argmax(top), top.shape)
        sx, sy = float(yx[1]), -h * 0.25
        acc = np.zeros_like(hi)
        N = 18
        for i in range(N):
            s = 1.0 + i * 0.05
            M = np.float32([[s, 0, (1 - s) * sx], [0, s, (1 - s) * sy]])
            acc += cv2.warpAffine(hi, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) * (1 - i / N)
        acc /= N * 0.5
        acc = cv2.GaussianBlur(acc, (0, 0), 1.5)
        self.layer = (np.clip(acc, 0, 1.5)[..., None] * np.float32(color)).astype(np.float32)

    def lf(self, ctx, M, A):
        q = ctx.get("qmaps")
        if q is None:
            return
        qx, qy = q
        r = cv2.remap(self.layer, qx, qy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        k = self.strength * 0.85 * (0.82 + 0.18 * math.sin(ctx["tl"] * 0.7 + self.seed))
        if self.fade:
            k *= 1.0 - 0.85 * clamp(ctx["p"])
        A += r * np.float32(k)


class Mist(FXBase):
    needs_q = True

    def __init__(self, W, H, S, seed, mood, strength=0.30):
        r = np.random.default_rng(seed)
        self.qw, self.qh = max(8, W // 4), max(8, H // 4)
        self.n1 = fractal_noise(self.qh, self.qw * 2, r, 4, 2.5)
        self.n2 = fractal_noise(self.qh, self.qw * 2, r, 3, 4.0)
        self.color = np.float32(MIST_COLOR.get(mood, (0.80, 0.84, 0.82)))
        self.strength = strength
        self.v1, self.v2 = 0.010 + r.random() * 0.01, 0.022 + r.random() * 0.01

    def lf(self, ctx, M, A):
        t = ctx["tl"]
        qh, qw = M.shape[:2]
        tw = self.n1.shape[1]
        o1 = int(t * self.v1 * qw + ctx["cam_dx"] / 4 * 0.3) % tw
        o2 = int(t * self.v2 * qw + ctx["cam_dx"] / 4 * 0.6) % tw
        i1 = (np.arange(qw) + o1) % tw
        i2 = (np.arange(qw) + o2) % tw
        n = self.n1[:qh, i1] * 0.6 + self.n2[:qh, i2] * 0.4
        a = np.clip((n - 0.38) * 1.7, 0, 1)
        dq = ctx.get("depth_q")
        if dq is not None:
            a = a * (0.25 + 0.75 * (1 - dq) ** 1.3)
        a = (a * np.float32(self.strength * ctx.get("fx_fade", 1.0))).astype(np.float32)
        M *= 1.0 - a
        A *= (1.0 - a)[..., None]
        A += a[..., None] * self.color


class CloudShadows(FXBase):
    def __init__(self, W, H, S, seed, strength=0.30):
        r = np.random.default_rng(seed)
        self.qw, self.qh = max(8, W // 4), max(8, H // 4)
        self.n = fractal_noise(self.qh * 2, self.qw * 2, r, 3, 2.0)
        self.strength = strength

    def lf(self, ctx, M, A):
        t = ctx["tl"]
        qh, qw = M.shape[:2]
        ox = int(t * 0.018 * self.qw) % (self.qw * 2)
        oy = int(t * 0.008 * self.qh) % (self.qh * 2)
        xi = (np.arange(qw) + ox) % (self.qw * 2)
        yi = (np.arange(qh) + oy) % (self.qh * 2)
        n = self.n[yi][:, xi]
        sh = (1 - np.clip((n - 0.45) * 2.2, 0, 1) * self.strength).astype(np.float32)
        M *= sh
        A *= sh[..., None]


class Rain(FXBase):
    def __init__(self, W, H, S, seed, heavy=True):
        r = np.random.default_rng(seed)
        self.W, self.H, self.S = W, H, S
        self.layers = []
        specs = [(0.55, 700 if heavy else 260, 0.022, 1, 1.25), (1.0, 220 if heavy else 70, 0.07, 2, 2.3)]
        ang = math.radians(9)
        for alpha, n, length, th, speed in specs:
            h2, w2 = H // 2, W // 2
            tex = np.zeros((h2 * 2, w2 * 2), np.uint8)
            for _ in range(int(n * 4)):
                x, y = r.random() * w2 * 2, r.random() * h2 * 2
                L = length * h2 * (0.6 + r.random() * 0.8)
                x2, y2 = x + math.sin(ang) * L, y + math.cos(ang) * L
                cv2.line(tex, _pt(x, y), _pt(x2, y2), int(120 + 135 * r.random()), max(1, int(th * S * 0.75)), cv2.LINE_AA, SHIFT)
            texf = cv2.GaussianBlur(tex.astype(np.float32) / 255.0, (0, 0), 0.5)
            self.layers.append((texf, alpha * (0.42 if heavy else 0.3), speed, ang))
        self.color = np.float32((0.80, 0.85, 0.90))

    def draw(self, frame, ctx):
        t = ctx["tl"]
        H, W = self.H // 2, self.W // 2
        acc = None
        for tex, alpha, speed, ang in self.layers:
            dy = int(t * speed * H) % H
            dx = int(t * speed * H * math.tan(ang) + ctx["cam_dx"] * 0.25) % W
            v = tex[H - dy:2 * H - dy, W - dx:2 * W - dx] * np.float32(alpha * ctx.get("fx_fade", 1.0))
            acc = v if acc is None else np.maximum(acc, v)
        a = cv2.resize(acc, (self.W, self.H), interpolation=cv2.INTER_LINEAR)[..., None]
        frame *= 1.0 - a
        frame += a * self.color


class Drips(FXBase):
    def __init__(self, W, H, S, seed, n=7):
        r = np.random.default_rng(seed)
        self.W, self.H, self.S = W, H, S
        self.x = r.random(n) * 0.9 + 0.05
        self.t0 = r.random(n) * 3.0
        self.per = 1.8 + r.random(n) * 2.5
        self.y0 = r.random(n) * 0.25

    def draw(self, frame, ctx):
        t = ctx["tl"]
        for i in range(len(self.x)):
            ph = ((t - self.t0[i]) % self.per[i]) / self.per[i]
            if ph > 0.45:
                continue
            q = ph / 0.45
            y = (self.y0[i] + q * q * 1.1) * self.H
            x = self.x[i] * self.W - ctx["cam_dx"] * 1.2
            L = (8 + 40 * q) * self.S
            draw_polyline(frame, [(x, y - L), (x, y)], (0.88, 0.92, 0.95), 1.6 * self.S, 0.55 * (1 - q * 0.5))


class Leaves(FXBase):
    def __init__(self, W, H, S, seed, n=12):
        r = np.random.default_rng(seed)
        self.W, self.H, self.S, self.n = W, H, S, n
        self.x0, self.y0 = r.random(n), r.random(n)
        self.sp = 0.03 + r.random(n) * 0.05
        self.ph = r.random(n) * 6.283
        self.sz = (7 + r.random(n) * 14) * S
        self.col = [(0.20 + r.random() * 0.15, 0.30 + r.random() * 0.15, 0.12 + r.random() * 0.06) for _ in range(n)]
        self.z = r.random(n)

    def draw(self, frame, ctx):
        t = ctx["tl"]
        for i in range(self.n):
            y = ((self.y0[i] + self.sp[i] * t) % 1.2 - 0.1) * self.H
            x = ((self.x0[i] + 0.04 * math.sin(t * 0.9 + self.ph[i]) + 0.01 * t) % 1.1 - 0.05) * self.W - ctx["cam_dx"] * (0.5 + self.z[i])
            ang = t * 1.7 + self.ph[i]
            flip = math.cos(t * 2.3 + self.ph[i])
            L, Wd = self.sz[i], self.sz[i] * 0.42 * abs(flip) + 0.6
            ca, sa = math.cos(ang), math.sin(ang)
            pts = []
            for k in range(12):
                a = k / 12 * 6.283
                px, py = math.cos(a) * L * 0.5, math.sin(a) * Wd * 0.5 * (1 - 0.6 * abs(math.cos(a)) ** 4)
                pts.append((x + px * ca - py * sa, y + px * sa + py * ca))
            pts = np.array(pts)
            pad = 3
            x0, y0 = int(pts[:, 0].min()) - pad, int(pts[:, 1].min()) - pad
            w, h = int(pts[:, 0].max()) - x0 + pad + 1, int(pts[:, 1].max()) - y0 + pad + 1
            if w <= 0 or h <= 0 or w > 400 or h > 400:
                continue
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [np.array([_pt(px - x0, py - y0) for px, py in pts], np.int32)], 255, cv2.LINE_AA, SHIFT)
            fill_mask(frame, m, x0, y0, self.col[i], 0.85)


class Bubbles(FXBase):
    def __init__(self, W, H, S, seed, n=34):
        r = np.random.default_rng(seed)
        self.W, self.H, self.S, self.n = W, H, S, n
        self.x0, self.y0 = r.random(n), r.random(n)
        self.sp = 0.05 + r.random(n) * 0.09
        self.r = (1.5 + r.random(n) ** 2 * 7) * S
        self.ph = r.random(n) * 6.283

    def draw(self, frame, ctx):
        t = ctx["tl"]
        for i in range(self.n):
            y = ((self.y0[i] - self.sp[i] * t) % 1.15) * self.H
            x = (self.x0[i] + 0.008 * math.sin(t * 3 + self.ph[i])) * self.W - ctx["cam_dx"] * 0.8
            draw_circle(frame, x, y, self.r[i], (0.85, 0.95, 0.95), thick=max(1.0, self.S * 1.2), opacity=0.42)
            draw_circle(frame, x - self.r[i] * 0.35, y - self.r[i] * 0.35, max(0.6, self.r[i] * 0.22), (1, 1, 1), opacity=0.5)


class Caustics(FXBase):
    def __init__(self, W, H, S, seed, strength=0.16):
        r = np.random.default_rng(seed)
        self.qw, self.qh = max(8, W // 4), max(8, H // 4)
        yy, xx = np.mgrid[0:self.qh, 0:self.qw].astype(np.float32)
        self.X, self.Y = xx / self.qw * 6.0, yy / self.qw * 6.0
        self.waves = [(r.random() * 6.283, 1.6 + r.random() * 1.8, 0.5 + r.random() * 0.9, r.random() * 6.283) for _ in range(5)]
        self.mask = np.clip(1.1 - np.linspace(0, 1, self.qh, dtype=np.float32) * 1.15, 0, 1)[:, None] ** 1.5
        self.strength = strength

    def draw_c(self, ctx):
        t = ctx["tl"]
        X = self.X + 0.35 * np.sin(self.Y * 1.7 + t * 0.6)
        Y = self.Y + 0.35 * np.cos(self.X * 1.3 + t * 0.5)
        c = np.zeros_like(X)
        for ang, k, w, ph in self.waves:
            c += np.cos((X * math.cos(ang) + Y * math.sin(ang)) * k + t * w + ph)
        c = (1 - np.abs(c / len(self.waves))) ** 7 * self.mask
        self._c = c.astype(np.float32)

    def lf(self, ctx, M, A):
        self.draw_c(ctx)
        qh, qw = M.shape[:2]
        c = self._c[:qh, :qw]
        A += c[..., None] * (np.float32((0.70, 0.95, 0.88)) * self.strength)

    def draw(self, frame, ctx):
        pass


class Wobble(FXBase):
    """Underwater / heat distortion applied to the camera maps."""

    def __init__(self, W, H, S, seed, amp=2.2, freq=2.6, speed=1.1):
        self.amp, self.freq, self.speed, self.seed = amp * S, freq, speed, seed

    def distort(self, mx, my, ctx):
        t = ctx["tl"]
        H, W = ctx["H"], ctx["W"]
        u = ctx["gu"] / W * 6.283 * self.freq
        v = ctx["gv"] / H * 6.283 * self.freq
        mx += self.amp * np.sin(v * 1.3 + t * self.speed * 2.1 + self.seed) * 1.0
        my += self.amp * np.cos(u * 1.1 + t * self.speed * 1.7) * 0.8


class Pulse(FXBase):
    """Electric-eel discharge: refractive ring + cyan glow + exposure blip (restrained)."""

    def __init__(self, W, H, S, seed, at=(0.68, 0.58), times=(0.34, 0.66)):
        self.at, self.times, self.S = at, times, S

    def _phase(self, ctx):
        for tp in self.times:
            q = (ctx["p"] - tp) * ctx["dur"] / 1.1
            if 0 <= q <= 1:
                return q
        return None

    def distort(self, mx, my, ctx):
        q = self._phase(ctx)
        if q is None:
            return
        W, H = ctx["W"], ctx["H"]
        cx, cy = self.at[0] * W, self.at[1] * H
        dx, dy = ctx["gu"] - cx, ctx["gv"] - cy
        r = np.sqrt(dx * dx + dy * dy) + 1e-3
        r0 = q * W * 0.55
        w = W * 0.035
        k = 9 * self.S * (1 - q) * np.exp(-((r - r0) / w) ** 2)
        mx += k * dx / r
        my += k * dy / r

    def draw(self, frame, ctx):
        q = self._phase(ctx)
        if q is None:
            return
        H, W = frame.shape[:2]
        cx, cy = self.at[0] * W, self.at[1] * H
        r0 = q * W * 0.55
        draw_circle(frame, cx, cy, r0, (0.45, 0.95, 1.0), thick=max(1.5, 3 * self.S), opacity=0.35 * (1 - q), mode="add")
        draw_circle(frame, cx, cy, r0 * 0.7, (0.45, 0.95, 1.0), thick=max(1.0, 2 * self.S), opacity=0.18 * (1 - q), mode="add")

    def lf(self, ctx, M, A):
        q = self._phase(ctx)
        if q is not None and q < 0.15:
            A += np.float32((1 - q / 0.15) * 0.10) * np.float32((0.6, 0.95, 1.0))


class Lightning(FXBase):
    def __init__(self, W, H, S, seed, times=(0.30, 0.74)):
        self.times = times

    def env(self, ctx):
        e = 0.0
        for tp in self.times:
            dt = (ctx["p"] - tp) * ctx["dur"]
            if 0 <= dt < 0.6:
                e = max(e, math.exp(-dt * 14) + 0.6 * math.exp(-max(0, dt - 0.16) * 16) * (dt > 0.16))
        return min(1.0, e)

    def lf(self, ctx, M, A):
        e = self.env(ctx)
        if e > 0.01:
            k = np.float32(0.42 * e)
            M *= 1 - k
            A *= 1 - k
            A += k * np.float32((0.85, 0.9, 1.0))


class Tension(FXBase):
    """Heartbeat vignette pulse."""

    def __init__(self, W, H, S, seed, bpm=74):
        self.per = 60.0 / bpm
        self.vig = None

    def lf(self, ctx, M, A):
        if self.vig is None or self.vig.shape[:2] != M.shape[:2]:
            H, W = M.shape[:2]
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            r = np.sqrt(((xx - W / 2) / (W * 0.55)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
            self.vig = np.clip(r - 0.35, 0, 1).astype(np.float32)
        ph = ctx["tl"] % self.per
        e = math.exp(-ph * 9) + 0.55 * math.exp(-max(0, ph - 0.27) * 10) * (ph > 0.27)
        k = 1 - self.vig * np.float32(0.30 * e + 0.10)
        M *= k
        A *= k[..., None]


def make_fx(name, W, H, S, seed, sc, plate_q, mood):
    nm = name.split(":")[0]
    if nm == "dust":
        return Particles(W, H, S, seed, "dust")
    if nm == "dust_warm":
        return Particles(W, H, S, seed, "dust", color=(1.0, 0.86, 0.62), n=90, strength=1.2)
    if nm == "fireflies":
        return Particles(W, H, S, seed, "fireflies")
    if nm == "insects":
        return Insects(W, H, S, seed)
    if nm in ("rays", "rays_fade"):
        col = (1.0, 0.86, 0.66) if mood in ("dusk", "dawn") else (1.0, 0.95, 0.80)
        return Rays(W, H, S, seed, plate_q, strength=0.5 if mood != "dawn" else 0.65, fade=(nm == "rays_fade"), color=col)
    if nm == "mist":
        return Mist(W, H, S, seed, mood, strength=0.30 if mood not in ("night",) else 0.22)
    if nm == "clouds":
        return CloudShadows(W, H, S, seed)
    if nm == "rain":
        return Rain(W, H, S, seed, heavy=True)
    if nm == "rain_light":
        return Rain(W, H, S, seed, heavy=False)
    if nm == "drips":
        return Drips(W, H, S, seed)
    if nm == "leaves":
        return Leaves(W, H, S, seed)
    if nm == "bubbles":
        return Bubbles(W, H, S, seed)
    if nm == "caustics":
        return Caustics(W, H, S, seed)
    if nm == "wobble":
        return Wobble(W, H, S, seed, amp=2.2, freq=2.4, speed=1.0)
    if nm == "heat":
        return Wobble(W, H, S, seed, amp=1.4, freq=7.0, speed=2.2)
    if nm == "pulse":
        return Pulse(W, H, S, seed, at=tuple(sc.get("pulse_at", (0.68, 0.58))))
    if nm == "lightning":
        return Lightning(W, H, S, seed)
    if nm == "tension":
        return Tension(W, H, S, seed)
    return None


# ----------------------------------------------------------------------------------------------
# Built-in cinematic manifest (from the 3 storyboard documents, 90 scenes)
# ----------------------------------------------------------------------------------------------
FILM = {"title_ar": "الأمازون", "title_en": "AMAZON", "kicker_ar": "فيلم وثائقي"}

CHAPTERS = [
    (1, "وحيدًا في المجهول", "ALONE IN THE UNKNOWN"),
    (10, "حياة لا تُرى", "UNSEEN LIFE"),
    (13, "النهر", "THE RIVER"),
    (21, "الضفة الأخرى", "THE FAR BANK"),
    (27, "العدو الصغير", "THE SMALLEST ENEMY"),
    (33, "حين يحلّ الظلام", "WHEN DARKNESS FALLS"),
    (47, "المطر", "THE RAIN"),
    (54, "الصباح الثاني", "THE SECOND MORNING"),
    (61, "الخيارات تتقلّص", "NARROWING OPTIONS"),
    (71, "أهل الغابة", "PEOPLE OF THE FOREST"),
    (76, "عالمٌ صنعناه", "THE WORLD WE BUILT"),
    (84, "السؤال الحقيقي", "THE REAL QUESTION"),
]

SPECIES = {
    "eel": ("الأنقليس الكهربائي", "Electrophorus electricus",
            ["ليس أنقليسًا حقيقيًا، بل من أسماك السكّين", "يولّد صدمات كهربائية للصيد والدفاع"]),
    "piranha": ("سمكة البيرانا", "Pygocentrus nattereri",
                ["تعيش وتتحرك في أسراب", "سمعتها في الأفلام مبالغ فيها كثيرًا"]),
    "jaguar": ("اليغور", "Panthera onca",
               ["أكبر القطط البرية في الأمريكتين", "صيّاد متخفٍّ يتجنب البشر غالبًا"]),
    "mosquito": ("بعوضة الأنوفيلة", "Anopheles spp.",
                 ["إناثها تنقل طفيل الملاريا", "البعوض أخطر الحيوانات على البشر عالميًا"]),
    "viper": ("أفعى الحُفَر", "Viperidae · Crotalinae",
              ["تستشعر حرارة فرائسها بحُفَر خاصة", "سامّة ومتقنة للتمويه"]),
    "anaconda": ("الأناكوندا الخضراء", "Eunectes murinus",
                 ["من أثقل الثعابين في العالم", "غير سامّة؛ تعصر فرائسها"]),
}

# n, title, seconds, camera(move, amount, handheld[, flags]), mood, fx, transition OUT, extra
_SC = [
    (1, "الاستيقاظ في المجهول", 10, ("push_in", .09, 0), "day_mist", ["rays", "mist", "dust"], "cut", {"hint": "explorer waking wet ground"}),
    (2, "لا أثر للبشر", 10, ("orbit_left", 1, 0), "day_mist", ["mist", "dust"], "cut", {"hint": "explorer scanning forest no people"}),
    (3, "الهاتف بلا شبكة", 10, ("push_in", .08, .15), "day_mist", ["dust"], "zoom_through", {"hint": "phone signal", "hud": {"signal": 0, "battery": 0.62, "hl": ["signal"]}}),
    (4, "المؤن القليلة", 10, ("rotate", 1, 0), "day", ["dust"], "cut", {"hint": "checking supplies bottle knife food", "hud": {"water": 1.0, "food": 0.4, "hl": ["water", "food"]}}),
    (5, "يوم واحد", 10, ("crane_up", 1, 0), "day", ["rays", "dust"], "slow_dissolve", {"hint": "tightens backpack straps", "kinetic": "يوم واحد"}),
    (6, "الغابة ليست طريقًا", 10, ("pull_out", .10, 0), "day_mist", ["mist", "dust"], "cut", {"hint": "choosing path direction"}),
    (7, "التقدم بصعوبة", 10, ("pan_left", 1, .35), "day", ["dust"], "whip_left", {"hint": "parts leaves vines"}),
    (8, "أرض لا تثق بها", 10, ("push_in", .08, .3), "day", ["drips"], "cut", {"hint": "boot mud"}),
    (9, "السماء المختفية", 10, ("tilt_up", 1, 0), "day_mist", ["rays", "dust", "drips"], "whip_down", {"hint": "tilting head upward canopy"}),
    (10, "لا تعتمد على عينيك", 10, ("dolly_zoom", 1, .2), "tension", ["dust", "tension"], "cut", {"hint": "freezes listening"}),
    (11, "حياة فوقك", 10, ("crane_up", 1, 0), "day", ["dust", "leaves"], "cut", {"hint": "branch animal silhouette"}),
    (12, "حياة عند قدميك", 10, ("pan_right", .7, 0), "day", ["insects"], "cut", {"hint": "lifts boot insects floor"}),
    (13, "صوت الماء", 10, ("push_in", .10, .15), "day", ["dust", "rays"], "whip_right", {"hint": "moving toward stream water"}),
    (14, "النهر الصغير", 10, ("push_in", .07, 0), "day_mist", ["mist", "dust"], "dissolve", {"hint": "emerges river edge"}),
    (15, "ماء يبدو هادئًا", 10, ("orbit_right", 1, 0), "day", ["dust"], "cut", {"hint": "hesitating near water bottle"}),
    (16, "ما تحت السطح", 10, ("tilt_down", 1, 0), "underwater", ["caustics", "bubbles", "wobble"], "dissolve", {"hint": "fish glide murky water"}),
    (17, "الأنقليس الكهربائي", 10, ("pan_right", .8, 0), "underwater", ["caustics", "bubbles", "wobble"], "cut", {"hint": "electric eel moving underwater", "species": "eel"}),
    (18, "قوة غير مرئية", 10, ("push_in", .07, 0), "underwater", ["caustics", "bubbles", "wobble", "pulse"], "cut", {"hint": "eel electrical pulse", "pulse_at": [0.70, 0.60]}),
    (19, "البيرانا والسمعة", 10, ("pan_left", .8, 0), "underwater", ["caustics", "bubbles", "wobble"], "cut", {"hint": "piranhas swimming roots", "species": "piranha"}),
    (20, "قرار عدم السباحة", 10, ("pull_out", .08, 0), "day", ["dust"], "cut", {"hint": "withdraws from river"}),
    (21, "السير بمحاذاة النهر", 10, ("pan_right", 1, .15), "day", ["dust", "mist"], "whip_right", {"hint": "walking riverbank"}),
    (22, "حركة على الضفة الأخرى", 10, ("orbit_left", 1, .15), "tension", ["mist", "tension"], "flash", {"hint": "watches shadow pass across"}),
    (23, "ما الذي رآه؟", 10, ("dolly_zoom", 1, .3), "tension", ["tension"], "cut", {"hint": "leans forward steps back"}),
    (24, "موطن اليغور", 10, ("pan_right", .6, 0), "day", ["dust", "mist"], "cut", {"hint": "jaguar walking rainforest", "species": "jaguar"}),
    (25, "مفترس لا تراه", 10, ("orbit_right", 1, .1), "day", ["dust"], "cut", {"hint": "explorer scanning forest turns"}),
    (26, "الابتعاد عن النهر", 10, ("push_in", .10, .2), "day", ["dust"], "light_leak", {"hint": "entering rainforest undergrowth"}),
    (27, "الإرهاق الأول", 10, ("drift", 1, 0), "heat", ["heat", "dust", "insects"], "slow_dissolve", {"hint": "resting exhausted wipes face", "fallback": 28}),
    (28, "هواء ثقيل", 10, ("push_in", .08, 0), "heat", ["heat", "dust"], "zoom_through", {"hint": "fanning shirt humid"}),
    (29, "الحشرات في كل مكان", 10, ("drift", 1, .2), "day", ["insects", "dust"], "cut", {"hint": "waving away insects mosquitoes"}),
    (30, "خطر صغير", 10, ("tilt_down", .6, 0), "day", ["insects"], "cut", {"hint": "checking arm bites"}),
    (31, "الملاريا", 10, ("push_in", .07, 0), "day", ["insects", "dust"], "cut", {"hint": "mosquito lifts off leaf", "species": "mosquito"}),
    (32, "لا ترى كل المخاطر", 10, ("orbit_left", 1, .2), "tension", ["insects", "dust"], "flash", {"hint": "brushing shoulder"}),
    (33, "صوت خلفه", 10, ("push_in", .06, .55), "tension", ["tension"], "cut", {"hint": "reacting to rustle"}),
    (34, "لا شيء ظاهر", 10, ("static", 1, 0), "tension", ["leaves"], "cut", {"hint": "leaves sway foliage"}),
    (35, "الضوء يتراجع", 10, ("tilt_up", 1, 0), "dusk", ["rays_fade", "dust"], "slow_dissolve", {"hint": "noticing light disappearing", "hud": {"time": "المساء", "hl": ["time"]}}),
    (36, "المساء يقترب", 10, ("orbit_right", 1, 0), "dusk", ["dust", "mist"], "cut", {"hint": "scanning ground forest"}),
    (37, "البحث عن مكان", 10, ("push_in", .08, 0), "dusk", ["dust"], "cut", {"hint": "steps on raised ground"}),
    (38, "تفقد مكان النوم", 10, ("pan_left", .6, 0), "dusk", ["insects"], "cut", {"hint": "moving leaves aside"}),
    (39, "مخابئ محتملة", 10, ("push_in", .10, 0), "dusk", ["dust"], "cut", {"hint": "insect disappears into crack log"}),
    (40, "الحذر من الثعابين", 10, ("orbit_right", 1, .1), "dusk", ["dust"], "cut", {"hint": "steps around fallen log"}),
    (41, "أفعى الحُفَر", 10, ("push_in", .07, 0), "dusk", ["dust"], "cut", {"hint": "pit viper flicking tongue", "species": "viper"}),
    (42, "ليست كل مواجهة", 10, ("dolly_zoom", .7, .15), "dusk", ["tension"], "cut", {"hint": "looking ahead worried"}),
    (43, "قرار البقاء", 10, ("pull_out", .08, 0), "night", ["fireflies"], "dip_night", {"hint": "sits roots bottle stays"}),
    (44, "أصوات الظلام", 10, ("orbit_left", 1, 0), "night", ["fireflies", "mist"], "cut", {"hint": "hearing sounds night", "hud": {"time": "الليل", "hl": ["time"]}}),
    (45, "عينان في العتمة؟", 10, ("push_in", .08, 0), "night", ["fireflies", "drips"], "cut", {"hint": "droplet falls darkness"}),
    (46, "الخوف لا يختفي", 10, ("push_in", .05, 0, "breathe"), "night", ["fireflies"], "slow_dissolve", {"hint": "closing eyes breathing"}),
    (47, "بداية المطر", 10, ("tilt_up", 1, 0), "rain", ["rain_light", "drips"], "cut", {"hint": "raising hand to rain"}),
    (48, "مطر غزير", 10, ("push_in", .07, .15), "rain", ["rain", "lightning", "mist"], "cut", {"hint": "hunches in heavy rain"}),
    (49, "الغابة المغمورة", 10, ("pan_right", .7, 0), "rain", ["rain_light", "drips"], "dissolve", {"hint": "water flowing around root"}),
    (50, "الغابة والماء", 10, ("crane_down", 1, 0), "rain", ["rain_light", "mist"], "cut", {"hint": "water ripples flooded forest"}),
    (51, "ماذا سيجد صباحًا؟", 10, ("push_in", .08, 0), "rain", ["rain_light"], "cut", {"hint": "protecting food supplies"}),
    (52, "خطأ البداية", 10, ("push_in", .07, 0), "rain", ["rain_light"], "dissolve", {"hint": "looking across rainforest regret"}),
    (53, "ليلة بلا نوم", 10, ("pull_out", .08, 0), "night", ["rain_light", "fireflies"], "light_leak", {"hint": "awake night resting curled", "fallback": 43}),
    (54, "الصباح الثاني", 10, ("crane_up", 1, 0), "dawn", ["rays", "mist", "dust"], "cut", {"hint": "waking up morning", "hud": {"day": 2, "time": "الفجر", "hl": ["day"]}}),
    (55, "مؤن أقل", 10, ("rotate", 1, 0), "dawn", ["dust"], "cut", {"hint": "checking water and food", "hud": {"water": 0.35, "food": 0.15, "hl": ["water", "food"]}}),
    (56, "هل تحركت فعلًا؟", 10, ("orbit_right", 1, 0), "day_mist", ["mist", "dust"], "whip_right", {"hint": "turning in rainforest landmarks"}),
    (57, "محاولة جديدة", 10, ("push_in", .08, .3), "day", ["dust"], "cut", {"hint": "walking through wet foliage"}),
    (58, "أرض منخفضة", 10, ("pan_left", .8, 0), "day_mist", ["mist", "insects"], "cut", {"hint": "examines flooded lowland"}),
    (59, "عمق مجهول", 10, ("push_in", .08, .15), "tension", ["tension", "insects"], "cut", {"hint": "peering into murky water"}),
    (60, "الأناكوندا", 10, ("pan_right", .6, 0), "day", ["mist"], "ink", {"hint": "anaconda wetland", "species": "anaconda"}),
    (61, "طريق آخر", 8, ("pan_right", .8, .15), "day", ["dust"], "cut", {"hint": "testing ground vegetation"}),
    (62, "كل طريق أصعب", 8, ("pull_out", .10, 0), "day_mist", ["mist", "dust"], "cut", {"hint": "looking at rainforest openings"}),
    (63, "ليست مواجهة واحدة", 8, ("push_in", .08, 0), "day", ["insects", "drips"], "dissolve", {"hint": "rain drips canopy hazards"}),
    (64, "الماء الذي تحتاجه", 8, ("push_in", .07, 0), "day", ["dust"], "cut", {"hint": "shakes nearly empty bottle", "hud": {"water": 0.12, "hl": ["water"]}}),
    (65, "الطريق الذي لا تعرفه", 8, ("tilt_up", 1, 0), "day", ["rays", "dust"], "whip_up", {"hint": "pointing in rainforest"}),
    (66, "اتساع الأمازون", 8, ("drift", 1, 0), "aerial", ["clouds", "mist"], "dissolve", {"hint": "cloud shadows drift over rainforest aerial", "map": 1}),
    (67, "البعد عن المساعدة", 8, ("pan_right", .6, 0), "aerial", ["mist"], "cut", {"hint": "mist drifting across forest landscape"}),
    (68, "لا لافتات ولا محطات", 8, ("push_in", .07, 0), "day", ["dust"], "cut", {"hint": "facing dense rainforest wall"}),
    (69, "الخيارات تتقلص", 8, ("rotate", 1, 0), "day", ["dust"], "cut", {"hint": "calculating remaining supplies", "hud": {"water": 0.08, "food": 0.06, "hl": ["water", "food"]}}),
    (70, "يومان فقط", 8, ("tilt_up", .6, 0), "tension", ["dust"], "light_leak", {"hint": "looking toward canopy", "kinetic": "يومان فقط"}),
    (71, "أهل الغابة", 8, ("pan_right", .6, 0), "warm", ["dust", "mist"], "dissolve", {"hint": "people moving amazon settlement"}),
    (72, "معرفة متوارثة", 8, ("orbit_left", 1, 0), "warm", ["dust"], "cut", {"hint": "guide pointing along river"}),
    (73, "معرفة النباتات", 8, ("push_in", .07, 0), "warm", ["dust"], "cut", {"hint": "hand pointing to plant leaves"}),
    (74, "الأنهار طرق حياة", 8, ("pan_left", .7, 0), "warm", ["mist", "dust"], "dissolve", {"hint": "canoe gliding amazon river"}),
    (75, "الفارق هو المعرفة", 8, ("orbit_right", 1, 0), "day", ["dust"], "zoom_through", {"hint": "watching passing river canoe"}),
    (76, "عالم صنعناه", 8, ("push_in", .07, 0), "interior", ["dust_warm"], "whip_left", {"hint": "water runs from tap indoors"}),
    (77, "اختفاء الامتيازات", 8, ("push_in", .07, 0), "day", ["dust"], "dip_black", {"hint": "phone battery dies", "hud": {"battery": 0.0, "signal": 0, "hl": ["battery"]}}),
    (78, "بلا شبكة", 8, ("push_in", .07, 0), "day_mist", ["mist", "dust"], "dissolve", {"hint": "turns in rainforest clearing"}),
    (79, "أين الماء الآمن؟", 8, ("orbit_left", 1, 0), "day", ["dust"], "cut", {"hint": "looking at water bottle river"}),
    (80, "كيف تعرف الطريق؟", 8, ("push_in", .06, 0), "day", ["dust"], "cut", {"hint": "studying paths"}),
    (81, "أين ستنام؟", 8, ("orbit_right", 1, 0), "dusk", ["dust"], "cut", {"hint": "checks ground and canopy"}),
    (82, "ماذا ستأكل؟", 8, ("push_in", .07, 0), "dusk", ["dust"], "cut", {"hint": "stores ration food", "hud": {"food": 0.03, "hl": ["food"]}}),
    (83, "ما الذي لا تعرفه؟", 8, ("pan_left", .6, 0), "day", ["drips", "dust"], "ink", {"hint": "droplet rolls off leaf plant"}),
    (84, "الغابة لا تطارده", 8, ("push_in", .07, 0), "day_mist", ["rays", "leaves", "dust"], "dissolve", {"hint": "explorer standing towering trees"}),
    (85, "لا تلاحظ وجودك", 8, ("pan_left", .7, 0), "day", ["leaves", "dust"], "cut", {"hint": "watching birds rainforest"}),
    (86, "فقدان السيطرة", 8, ("push_in", .05, 0), "dusk", ["dust"], "slow_dissolve", {"hint": "shifting expression"}),
    (87, "السؤال الحقيقي", 8, ("crane_up", 1, 0), "dusk", ["rays", "dust", "mist"], "dissolve", {"hint": "raising head canopy", "kinetic": "السؤال الحقيقي"}),
    (88, "ليس عليك أن تكون وحشًا", 8, ("drift", 1, 0), "aerial", ["clouds", "mist"], "dip_black", {"hint": "cloud shadows drift over rainforest"}),
    (89, "كم من الوقت؟", 8, ("pull_out", .10, 0), "dusk", ["mist", "dust"], "dip_black", {"hint": "explorer standing tiny silhouette", "kinetic": "كم من الوقت؟", "kinetic_pos": "top"}),
    (90, "الصمت بعد السؤال", 8, ("drift", 1, 0), "aerial", ["mist"], "fade_to_black", {"hint": "mist drifting above amazon canopy", "end": 1}),
]


def builtin_manifest():
    scenes = []
    ch_ids = [c[0] for c in CHAPTERS]
    for n, title, dur, cam, mood, fx, tout, extra in _SC:
        ch = bisect.bisect_right(ch_ids, n)
        camd = {"move": cam[0], "amount": cam[1], "handheld": cam[2]}
        if len(cam) > 3 and cam[3] == "breathe":
            camd["breathe"] = True
        s = {"scene": n, "title": title, "duration": dur, "chapter": ch, "camera": camd, "mood": mood,
             "fx": list(fx), "transition_out": tout}
        s.update(extra)
        scenes.append(s)
    return {
        "format": "amazon-cinematic-v3",
        "film": dict(FILM),
        "chapters": [{"start_scene": a, "title_ar": b, "title_en": c} for a, b, c in CHAPTERS],
        "species": {k: {"name_ar": v[0], "latin": v[1], "facts": v[2]} for k, v in SPECIES.items()},
        "scenes": scenes,
    }


OLD_CAMERA = {"push_in": "push_in", "zoom_in": "push_in", "pull_out": "pull_out", "zoom_out": "pull_out",
              "pan_left": "pan_left", "pan_right": "pan_right", "tilt_up": "tilt_up", "tilt_down": "tilt_down"}


def load_manifest(path):
    base = builtin_manifest()
    if not path or not Path(path).exists():
        log("Manifest: built-in cinematic manifest (90 scenes).")
        return base
    user = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(user, dict) and user.get("format", "").startswith("amazon-cinematic"):
        log(f"Manifest: cinematic manifest {path}")
        for k in ("film", "chapters", "species"):
            if k in user:
                base[k] = user[k]
        base["scenes"] = user["scenes"]
        return base
    scenes = user.get("scenes", []) if isinstance(user, dict) else user
    if not isinstance(scenes, list) or not scenes:
        raise RuntimeError("Manifest must be a list of scenes or {'scenes': [...]}.")
    log(f"Manifest: {path} ({len(scenes)} scenes) merged over the built-in cinematic directions.")
    bynum = {s["scene"]: s for s in base["scenes"]}
    merged = []
    for i, us in enumerate(scenes):
        n = int(us.get("scene", i + 1))
        s = dict(bynum.get(n, {"scene": n, "title": f"Scene {n}", "duration": 8, "chapter": len(base["chapters"]),
                               "camera": {"move": "push_in", "amount": 0.08, "handheld": 0}, "mood": "day",
                               "fx": ["dust"], "transition_out": "cut"}))
        for k in ("title", "duration", "start", "keywords"):
            if k in us and us[k] not in (None, ""):
                s[k] = us[k]
        for k in ("image", "filename", "file"):  # legacy names are only a fallback after OCR
            if us.get(k):
                s["legacy_image"] = us[k]
                break
        if isinstance(us.get("camera"), dict):
            s["camera"] = us["camera"]
        if n not in bynum and isinstance(us.get("camera"), str):
            s["camera"] = {"move": OLD_CAMERA.get(us["camera"], "push_in"), "amount": 0.08, "handheld": 0}
        for k in ("mood", "fx", "transition_out", "species", "hud", "kinetic", "map", "focus"):
            if k in us:
                s[k] = us[k]
        merged.append(s)
    base["scenes"] = merged
    return base


# ----------------------------------------------------------------------------------------------
# Image -> scene mapping
# ----------------------------------------------------------------------------------------------
STOP = {"the", "a", "an", "in", "of", "to", "and", "at", "on", "his", "he", "with", "from", "into", "for", "jpg", "png"}
SYN = {"stick": "explorer", "man": "explorer", "figure": "explorer", "character": "explorer", "stickman": "explorer"}


def _tokens(s):
    s = re.sub(r"_[0-9a-f]{6,10}$", "", s.lower())
    toks = re.split(r"[^a-z]+", s)
    out = set()
    for t in toks:
        if len(t) < 3 or t in STOP:
            continue
        t = SYN.get(t, t)
        for suf in ("ing", "es", "s", "ed"):
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                t = t[: -len(suf)]
                break
        out.add(t)
    return out


def map_images(manifest, folder, fb, work_dir):
    folder = Path(folder)
    files = sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not files:
        raise RuntimeError(f"No images found under: {folder}")
    scenes = manifest["scenes"]
    nums = [int(s["scene"]) for s in scenes]
    cache_p = Path(work_dir) / "ocr_cache.json"
    cache = {}
    if cache_p.exists():
        try:
            cache = json.loads(cache_p.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    info = {}
    excluded = []
    log(f"Scanning {len(files)} images (reading printed scene numbers)...")
    for p in files:
        key = f"{p.name}|{p.stat().st_size}"
        if key in cache:
            info[p] = cache[key]
            continue
        name_l = p.stem.lower()
        try:
            img = imread_rgb(p)
            lab = find_label(img, fb)
            rec = {"number": lab["number"] if lab else None, "conf": lab["conf"] if lab else 0.0,
                   "box": list(lab["box"]) if lab else None, "w": img.shape[1], "h": img.shape[0]}
        except Exception as e:
            rec = {"number": None, "conf": 0.0, "box": None, "error": str(e)}
        rec["reference"] = bool(re.search(r"reference|sheet|turnaround|model_sheet", name_l)) and rec["number"] is None
        info[p] = rec
        cache[key] = rec
    try:
        cache_p.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass
    by_name = {p.name: p for p in files}
    by_stem = {p.stem.lower(): p for p in files}
    assign = {}
    used = set()
    # 1) explicit image names in manifest
    for s in scenes:
        for k in ("image", "filename", "file"):
            v = s.get(k)
            if v:
                p = by_name.get(Path(v).name) or by_stem.get(Path(v).stem.lower())
                if p and p not in used:
                    assign[s["scene"]] = (p, "manifest")
                    used.add(p)
                    break
    # 2) OCR numbers
    groups = {}
    for p, r in info.items():
        if p in used or r.get("reference"):
            continue
        if r.get("number") is not None and r.get("conf", 0) > 0.35 and r["number"] in nums:
            groups.setdefault(r["number"], []).append(p)
    hints = {s["scene"]: _tokens(" ".join([s.get("hint", "")] + list(map(str, s.get("keywords", []))))) for s in scenes}
    for n, ps in groups.items():
        if n in assign:
            continue
        ps.sort(key=lambda q: (-len(_tokens(q.stem) & hints.get(n, set())), -info[q]["conf"]))
        assign[n] = (ps[0], "ocr")
        used.add(ps[0])
    # 2b) legacy manifest image names for scenes the OCR did not cover
    for s in scenes:
        v = s.get("legacy_image")
        if v and s["scene"] not in assign:
            p = by_name.get(Path(v).name) or by_stem.get(Path(v).stem.lower())
            if p and p not in used and not info[p].get("reference") and info[p].get("number") in (None, s["scene"]):
                assign[s["scene"]] = (p, "manifest-legacy")
                used.add(p)
    # 3) remaining images by name similarity (Hungarian if scipy present)
    rest_imgs = [p for p in files if p not in used and not info[p].get("reference")]
    rest_scn = [s["scene"] for s in scenes if s["scene"] not in assign]
    if rest_imgs and rest_scn:
        M = np.zeros((len(rest_scn), len(rest_imgs)), np.float32)
        for i, n in enumerate(rest_scn):
            for j, p in enumerate(rest_imgs):
                tk = _tokens(p.stem)
                h = hints.get(n, set())
                M[i, j] = len(tk & h) / (len(h | tk) + 1e-6)
        try:
            from scipy.optimize import linear_sum_assignment
            ri, cj = linear_sum_assignment(-M)
            pairs = list(zip(ri, cj))
        except Exception:
            pairs, taken = [], set()
            for i in np.argsort(-M.max(1)):
                j = int(np.argmax(np.where([k in taken for k in range(len(rest_imgs))], -1, M[i])))
                if j not in taken:
                    pairs.append((i, j))
                    taken.add(j)
        for i, j in pairs:
            if M[i, j] >= 0.22:
                assign[rest_scn[i]] = (rest_imgs[j], f"name:{M[i, j]:.2f}")
                used.add(rest_imgs[j])
    # 4) borrow for scenes still missing (storyboard fallback or nearest neighbour)
    final = {}
    for s in scenes:
        n = s["scene"]
        if n in assign:
            p, src = assign[n]
            final[n] = {"path": str(p), "source": src, "label_box": info[p].get("box"), "borrowed": None}
    for s in scenes:
        n = s["scene"]
        if n in final:
            continue
        cand = [s.get("fallback")] if s.get("fallback") else []
        cand += sorted([m for m in final if final[m]["borrowed"] is None], key=lambda m: abs(m - n))
        for c in cand:
            if c in final and final[c]["borrowed"] is None:
                final[n] = dict(final[c])
                final[n]["source"] = f"borrowed from {c}"
                final[n]["borrowed"] = c
                break
    missing = [s["scene"] for s in scenes if s["scene"] not in final]
    if missing:
        raise RuntimeError(f"No image available for scenes {missing}")
    spare = [p.name for p in files if p not in used and not info[p].get("reference")]
    refs = [p.name for p in files if info[p].get("reference")]
    return final, {"spare": spare, "reference": refs, "ocr": {p.name: info[p].get("number") for p in files}}


def contact_sheet(manifest, mapping, fb, out_path):
    scenes = manifest["scenes"]
    cols = 10
    tw, th = 288, 162
    rows = (len(scenes) + cols - 1) // cols
    sheet = np.zeros((rows * (th + 46) + 70, cols * tw, 3), np.float32) + 0.07
    t = text_sprite(fb, "AMAZON - image/scene contact sheet  (orange = borrowed, red = name-guess)", "mono", 26, WHITE, shadow=0)
    blit(sheet, t.a, 16, 50 - t.base + 0, 1.0)
    for i, s in enumerate(scenes):
        n = s["scene"]
        m = mapping[n]
        r, c = divmod(i, cols)
        x0, y0 = c * tw, 70 + r * (th + 46)
        try:
            img = imread_rgb(m["path"])
            if m.get("label_box"):
                img = remove_label(img, m["label_box"])
            im = cv2.resize(img, (tw - 6, th - 6), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            if m["borrowed"]:
                im = cv2.flip(im, 1) * 0.8
            sheet[y0 + 3:y0 + th - 3, x0 + 3:x0 + tw - 3] = im
        except Exception:
            pass
        col = ACCENT if m["borrowed"] else (DANGER if m["source"].startswith("name") else TEAL)
        draw_rect(sheet, x0 + 3, y0 + th - 6, x0 + tw - 3, y0 + th - 3, col)
        a = text_sprite(fb, f"{n:03d}", "mono", 22, col, shadow=0.8)
        blit(sheet, a.a, x0 + 8, y0 + 6 + a.pad, 1.0, ax=a.pad, ay=a.pad)
        tt = text_sprite(fb, s.get("title", ""), "ar_bold", 20, WHITE, shadow=0)
        blit(sheet, tt.a, x0 + tw - 8, y0 + th + 22, 1.0, ax=tt.pad + tt.ink_w, ay=tt.base)
        fn = Path(m["path"]).name
        fn = fn if len(fn) <= 34 else fn[:31] + "..."
        ft = text_sprite(fb, fn, "mono", 13, MUTED, shadow=0)
        blit(sheet, ft.a, x0 + 6, y0 + th + 40, 1.0, ax=ft.pad, ay=ft.base)
    out = np.clip(sheet * 255, 0, 255).astype(np.uint8)
    cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(out_path))
    return out_path


# ----------------------------------------------------------------------------------------------
# Scene preparation (plate, depth, fx)
# ----------------------------------------------------------------------------------------------
OVERSCAN = 1.14


class PreparedScene:
    pass


def fit_cover(img, w, h):
    sc = max(w / img.shape[1], h / img.shape[0])
    nw, nh = int(math.ceil(img.shape[1] * sc)), int(math.ceil(img.shape[0] * sc))
    up = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4 if sc > 1 else cv2.INTER_AREA)
    x, y = (nw - w) // 2, (nh - h) // 2
    return up[y:y + h, x:x + w]


def prepare_scene(idx, sc, mapping, R, depth_est, depth_cache_dir):
    W, H, S = R.W, R.H, R.S
    m = mapping[sc["scene"]]
    img = imread_rgb(m["path"])
    if m.get("label_box"):
        img = remove_label(img, tuple(m["label_box"]))
    head = find_head(img, m.get("label_box"))
    borrowed = m.get("borrowed")
    if borrowed:
        # re-frame a borrowed image so it reads as a different shot: mirrored, tighter on the subject
        img = cv2.flip(img, 1)
        if head:
            head = (1 - head[0], head[1], head[2])
        hh, ww = img.shape[:2]
        cxn, cyn = (head[0], head[1]) if head else (0.5, 0.55)
        z = 1.28
        cw, ch = int(ww / z), int(hh / z)
        x0 = int(clamp(cxn * ww - cw / 2, 0, ww - cw))
        y0 = int(clamp(cyn * hh - ch / 2, 0, hh - ch))
        img = img[y0:y0 + ch, x0:x0 + cw]
        if head:
            head = ((head[0] * ww - x0) / cw, (head[1] * hh - y0) / ch, head[2] * z)
    PW, PH = int(round(W * OVERSCAN)), int(round(H * OVERSCAN))
    PW += PW % 2
    PH += PH % 2
    # depth (cached)
    dkey = hashlib.md5(f"{m['path']}|{borrowed}|{depth_est.is_model}|{depth_est.mode}|v3".encode()).hexdigest()[:16]
    dpath = Path(depth_cache_dir) / f"depth_{sc['scene']:03d}_{dkey}.png"
    if dpath.exists():
        d = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
    else:
        d = depth_est(img)
        try:
            cv2.imwrite(str(dpath), (np.clip(d, 0, 1) * 65535).astype(np.uint16))
        except Exception:
            pass
    plate = fit_cover(img, PW, PH)
    if plate.shape[1] > img.shape[1] * 1.15:
        # gentle sharpen after upscale keeps ink lines crisp
        bl = cv2.GaussianBlur(plate, (0, 0), 1.2 * S + 0.3)
        plate = cv2.addWeighted(plate, 1.35, bl, -0.35, 0)
    mood = sc.get("mood", "day")
    plate = grade(plate, mood, W)
    ps = PreparedScene()
    ps.idx, ps.sc, ps.mood, ps.plate = idx, sc, mood, plate
    ps.PW, ps.PH = PW, PH
    dq = cv2.resize(fit_cover((np.clip(d, 0, 1) * 65535).astype(np.uint16), PW, PH), (PW // 4, PH // 4),
                    interpolation=cv2.INTER_AREA).astype(np.float32) / 65535.0
    ps.depth_q = dq
    if sc.get("focus"):
        focus = tuple(sc["focus"])
    elif head:
        focus = (head[0], head[1])
    else:
        focus = (0.5, 0.5)
    ps.focus = focus
    ps.cam = Camera(sc.get("camera", {}), 1000 + sc["scene"], focus, 1.0 if depth_est.is_model else 0.65)
    L = LOOKS.get(mood, LOOKS["day"])
    ps.vig_q = cv2.resize(R.vignette(L[8])[..., 0], (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    plate_q = cv2.resize(plate, (PW // 4, PH // 4), interpolation=cv2.INTER_AREA)
    ps.fx = []
    for name in sc.get("fx", []):
        f = make_fx(name, W, H, S, 5000 + sc["scene"] * 17 + len(ps.fx), sc, plate_q, mood)
        if f is not None:
            ps.fx.append(f)
    ps.needs_q = any(getattr(f, "needs_q", False) for f in ps.fx)
    return ps


# ----------------------------------------------------------------------------------------------
# Graphics (motion design layer)
# ----------------------------------------------------------------------------------------------
class Gfx:
    z = 10
    letterbox = 0.0

    def __init__(self, t0, t1):
        self.t0, self.t1 = t0, t1

    def active(self, t):
        return self.t0 <= t <= self.t1

    def bars(self, t):
        return 0.0

    def pre(self, frame, t, R):
        pass

    def draw(self, frame, t, R):
        pass

    def sfx(self):
        return []


def _place_right(frame, spr, x_right, baseline, opacity=1.0, arr=None, dx=0.0, scale=1.0):
    blit(frame, spr.a if arr is None else arr, x_right + dx, baseline, opacity, scale=scale, ax=spr.pad + spr.ink_w, ay=spr.base)


def _place_center(frame, spr, cx, baseline, opacity=1.0, arr=None, scale=1.0, dy=0.0):
    blit(frame, spr.a if arr is None else arr, cx, baseline + dy, opacity, scale=scale, ax=spr.pad + spr.ink_w / 2, ay=spr.base)


def _darken_blur(frame, amt, blur_amt, S):
    if amt <= 0.003 and blur_amt <= 0.003:
        return
    H, W = frame.shape[:2]
    if blur_amt > 0.003:
        sm = cv2.resize(frame, (W // 6, H // 6), interpolation=cv2.INTER_AREA)
        sm = cv2.GaussianBlur(sm, (0, 0), 2.2)
        bl = cv2.resize(sm, (W, H), interpolation=cv2.INTER_LINEAR)
        frame += (bl - frame) * blur_amt
    if amt > 0.003:
        frame *= (1 - amt)


class OpeningTitle(Gfx):
    z = 30

    def __init__(self, t0, t1, film):
        super().__init__(t0, t1)
        self.film = film

    def bars(self, t):
        return 1.0 - ease_in_out_cubic(seg(t, self.t1 - 0.3, self.t1 + 0.9))

    def draw(self, frame, t, R):
        S, W, H = R.S, R.W, R.H
        lt, T = t - self.t0, self.t1 - self.t0
        fb = R.fb
        out = ease_in_cubic(seg(lt, T - 0.9, T))
        vis = 1 - out
        cx, cy = W / 2, H * 0.47
        _darken_blur(frame, 0.30 * vis, 0.0, S)
        k = text_sprite(fb, self.film.get("kicker_ar", ""), "ar_med", 30 * S, MUTED, shadow=0.5)
        ka = ease_out_cubic(seg(lt, 0.4, 1.4))
        _place_center(frame, k, cx, cy - 128 * S, ka * vis, dy=(1 - ka) * 10 * S)
        title = text_sprite(fb, self.film.get("title_ar", ""), "ruqaa", 200 * S, CREAM, shadow=0.6, glow=0.35, glow_color=(1.0, 0.85, 0.6))
        tr = ease_out_expo(seg(lt, 0.9, 2.6))
        arr = blur_sprite(title.a, 16 * S * (1 - ease_out_cubic(seg(lt, 0.9, 2.0))) + 8 * S * out)
        # specular light sweep across the title
        sw = seg(lt, 2.0, 3.6)
        if 0 < sw < 1:
            xs = np.arange(arr.shape[1], dtype=np.float32)
            pos = arr.shape[1] * (1.2 - 1.4 * sw)
            band = np.exp(-((xs - pos) / (arr.shape[1] * 0.08)) ** 2)[None, :, None]
            arr = arr.copy()
            arr[..., :3] *= 1 + 0.55 * band * (arr[..., 3:4] > 0.2)
        _place_center(frame, title, cx, cy + 40 * S, clamp(tr * 1.4) * vis, arr=arr, scale=1.10 - 0.10 * tr + 0.04 * out)
        en = self.film.get("title_en", "")
        letters = []
        trk = 0.62 - 0.14 * ease_out_expo(seg(lt, 1.6, 4.5))
        spr = text_sprite(fb, en, "cinzel", 40 * S, GOLD, shadow=0.5, tracking=trk)
        ea = ease_out_cubic(seg(lt, 1.7, 2.8))
        _place_center(frame, spr, cx, cy + 128 * S, ea * vis)
        # hairlines
        g = ease_out_expo(seg(lt, 1.8, 3.0)) * (1 - ease_in_cubic(seg(lt, T - 0.9, T - 0.2)))
        L = 230 * S * g
        hw = spr.ink_w / 2 + 30 * S
        yl = cy + 116 * S
        draw_line(frame, cx + hw, yl, cx + hw + L, yl, GOLD, max(1, 1.5 * S), 0.85 * vis)
        draw_line(frame, cx - hw, yl, cx - hw - L, yl, GOLD, max(1, 1.5 * S), 0.85 * vis)

    def sfx(self):
        return [("riser", self.t0 + 0.9 - 1.6, 1.6, -16), ("boom", self.t0 + 0.95, 2.8, -9),
                ("shimmer", self.t0 + 1.0, 2.4, -20), ("shimmer", self.t0 + 2.0, 1.6, -26)]


class ChapterCard(Gfx):
    z = 30

    def __init__(self, t0, t1, idx, total, title_ar, title_en):
        super().__init__(t0, t1)
        self.idx, self.total, self.ar, self.en = idx, total, title_ar, title_en

    def bars(self, t):
        return ease_out_expo(seg(t, self.t0 - 0.1, self.t0 + 0.6)) * (1 - ease_in_out_cubic(seg(t, self.t1 - 0.6, self.t1)))

    def draw(self, frame, t, R):
        S, W, H, fb = R.S, R.W, R.H, R.fb
        lt, T = t - self.t0, self.t1 - self.t0
        a_in = ease_out_cubic(seg(lt, 0, 0.6))
        out = ease_in_cubic(seg(lt, T - 0.7, T))
        vis = 1 - out
        _darken_blur(frame, 0.40 * a_in * vis, 0.55 * a_in * vis, S)
        cx, cy = W / 2, H / 2
        lab = text_sprite(fb, f"الفصل {AR_ORD[self.idx - 1] if self.idx <= len(AR_ORD) else ar_num(self.idx)}", "ar_med", 30 * S, ACCENT, shadow=0.5)
        la = ease_out_cubic(seg(lt, 0.15, 0.8))
        _place_center(frame, lab, cx, cy - 122 * S, la * vis, dy=(1 - la) * 12 * S)
        g = ease_out_expo(seg(lt, 0.3, 1.3))
        hw = lab.ink_w / 2 + 22 * S
        yl = cy - 132 * S
        for d in (1, -1):
            draw_line(frame, cx + d * hw, yl, cx + d * (hw + 150 * S * g), yl, ACCENT, max(1, 1.5 * S), 0.9 * vis)
            draw_circle(frame, cx + d * (hw + 150 * S * g + 6 * S), yl, 2.5 * S, ACCENT, opacity=g * vis)
        tt = text_sprite(fb, self.ar, "ruqaa", 128 * S, WHITE, shadow=0.6, glow=0.25, glow_color=(1, 0.9, 0.75))
        tr = ease_out_expo(seg(lt, 0.25, 1.35))
        arr = sprite_wipe(tt.a, tr, 140 * S, rtl=True)
        arr = blur_sprite(arr, 10 * S * (1 - ease_out_cubic(seg(lt, 0.25, 1.0))) + 9 * S * out)
        _place_center(frame, tt, cx, cy + 30 * S - out * 14 * S, vis, arr=arr, scale=1.06 - 0.06 * tr)
        en = f"CHAPTER {ROMAN[self.idx - 1] if self.idx <= len(ROMAN) else self.idx}  ·  {self.en}"
        es = text_sprite(fb, en, "cinzel", 24 * S, MUTED, shadow=0.5, tracking=0.30)
        ea = ease_out_cubic(seg(lt, 0.7, 1.6))
        arr = sprite_wipe(es.a, ea, 80 * S, rtl=False)
        _place_center(frame, es, cx, cy + 102 * S, vis, arr=arr)
        # chapter ticks
        n = self.total
        sp = 22 * S
        x0 = cx - (n - 1) * sp / 2
        for i in range(n):
            ta = ease_out_back(seg(lt, 0.9 + i * 0.04, 1.3 + i * 0.04))
            cur = (n - 1 - i) == (self.idx - 1)  # RTL order: first chapter on the right
            r = (4.2 if cur else 2.4) * S * ta
            draw_circle(frame, x0 + i * sp, cy + 146 * S, r, ACCENT if cur else MUTED, opacity=(1.0 if cur else 0.45) * vis)

    def sfx(self):
        return [("riser", self.t0 + 0.3 - 1.2, 1.2, -20), ("boom", self.t0 + 0.32, 2.4, -12), ("shimmer", self.t0 + 0.4, 1.8, -24)]


class LowerThird(Gfx):
    z = 20

    def __init__(self, t0, t1, n, total, title, chapter_name):
        super().__init__(t0, t1)
        self.n, self.total, self.title, self.ch = n, total, title, chapter_name

    def draw(self, frame, t, R):
        S, W, H, fb = R.S, R.W, R.H, R.fb
        lt, T = t - self.t0, self.t1 - self.t0
        out = ease_in_cubic(seg(lt, T - 0.6, T))
        xr, yb = W - 118 * S, H - 112 * S
        if R.shadow_lt is None:
            R.shadow_lt = soft_shadow_sprite(int(900 * S), int(300 * S), 0.62)
        sh = R.shadow_lt
        blit(frame, sh, xr + 140 * S, yb + 10 * S, 0.9 * ease_out_cubic(seg(lt, 0, 0.6)) * (1 - out), ax=sh.shape[1], ay=sh.shape[0] / 2)
        # accent bar
        g = ease_out_expo(seg(lt, 0.0, 0.55))
        go = ease_in_cubic(seg(lt, T - 0.45, T))
        top, bot = yb - 88 * S, yb + 30 * S
        y_top = bot - (bot - top) * g
        y_bot = bot - (bot - top) * go
        draw_rect(frame, xr + 18 * S, y_top, xr + 22 * S, y_bot, ACCENT, 1.0)
        # chapter label
        cl = text_sprite(fb, self.ch, "ar_med", 24 * S, MUTED, shadow=0.6)
        ca = ease_out_expo(seg(lt, 0.12, 0.8))
        _place_right(frame, cl, xr, yb - 62 * S, 1.0, arr=sprite_wipe(cl.a, ca, 50 * S, True, out))
        # title
        tt = text_sprite(fb, self.title, "ar_black", 58 * S, WHITE, shadow=0.65)
        tr = ease_out_expo(seg(lt, 0.2, 1.05))
        arr = sprite_wipe(tt.a, tr, 90 * S, True, out)
        arr = blur_sprite(arr, 6 * S * (1 - ease_out_cubic(seg(lt, 0.2, 0.75))))
        _place_right(frame, tt, xr, yb, 1.0, arr=arr, dx=(1 - tr) * 26 * S)
        # hairline + index
        lw = tt.ink_w + 34 * S
        lg = ease_out_expo(seg(lt, 0.35, 1.2))
        lo = ease_in_cubic(seg(lt, T - 0.6, T - 0.05))
        yl = yb + 24 * S
        x_end = xr - lw * lg
        x_start = xr - lw * lo
        if x_start > x_end + 0.5:
            draw_line(frame, x_start, yl, x_end, yl, ACCENT, max(1, 1.6 * S), 0.95)
        idx = text_sprite(fb, f"{self.n:03d} / {self.total:03d}", "mono", 18 * S, GOLD, shadow=0.6)
        ia = ease_out_cubic(seg(lt, 0.9, 1.4)) * (1 - out)
        blit(frame, idx.a, xr - lw - 14 * S, yl + 6 * S, ia, ax=idx.pad + idx.ink_w, ay=idx.base)

    def sfx(self):
        return [("whoosh_soft", self.t0 - 0.05, 0.55, -27), ("tick", self.t0 + 0.25, 0.2, -27)]


class Kinetic(Gfx):
    z = 25

    def __init__(self, t0, t1, text, pos="center"):
        super().__init__(t0, t1)
        self.words = text.split()
        self.pos = pos

    def draw(self, frame, t, R):
        S, W, H, fb = R.S, R.W, R.H, R.fb
        lt, T = t - self.t0, self.t1 - self.t0
        out = ease_in_cubic(seg(lt, T - 0.8, T))
        vis = 1 - out
        _darken_blur(frame, 0.22 * ease_out_cubic(seg(lt, 0, 0.8)) * vis, 0.0, S)
        cy = H * (0.30 if self.pos == "top" else 0.50)
        sprs = [text_sprite(fb, w, "ruqaa", 124 * S, WHITE, shadow=0.8, glow=0.18, glow_color=(1, 0.88, 0.7), halo=0.55) for w in self.words]
        gap = 34 * S
        total = sum(s.ink_w for s in sprs) + gap * (len(sprs) - 1)
        x = W / 2 + total / 2
        stagger = 0.34
        for i, s in enumerate(sprs):
            a = ease_out_cubic(seg(lt, 0.15 + i * stagger, 0.85 + i * stagger))
            mv = ease_out_expo(seg(lt, 0.15 + i * stagger, 1.2 + i * stagger))
            arr = blur_sprite(s.a, 12 * S * (1 - a) + 10 * S * out)
            _place_right(frame, s, x, cy + 36 * S * (1 - mv) - 18 * S * out, a * vis, arr=arr, scale=1.10 - 0.10 * mv)
            x -= s.ink_w + gap
        ul = ease_out_expo(seg(lt, 0.4 + len(sprs) * stagger, 1.4 + len(sprs) * stagger))
        if ul > 0.01:
            xr = W / 2 + total / 2
            draw_line(frame, xr, cy + 44 * S, xr - total * ul, cy + 44 * S, ACCENT, max(1, 2.2 * S), 0.95 * vis)

    def sfx(self):
        out = [("whoosh_soft", self.t0 + 0.05, 0.7, -24)]
        for i in range(len(self.words)):
            out.append(("hit", self.t0 + 0.2 + i * 0.34, 1.2, -18))
        return out


class EndCard(Gfx):
    z = 40

    def __init__(self, t0, t1, film):
        super().__init__(t0, t1)
        self.film = film

    def bars(self, t):
        return 1.0

    def draw(self, frame, t, R):
        S, W, H, fb = R.S, R.W, R.H, R.fb
        lt = t - self.t0
        a = ease_out_cubic(seg(lt, 0.2, 1.6))
        tt = text_sprite(fb, self.film.get("title_ar", ""), "ruqaa", 120 * S, CREAM, shadow=0.0, glow=0.3, glow_color=(1, 0.85, 0.6))
        arr = blur_sprite(tt.a, 10 * S * (1 - a))
        _place_center(frame, tt, W / 2, H / 2 + 20 * S, a, arr=arr, scale=1.04 - 0.04 * a)
        g = ease_out_expo(seg(lt, 0.8, 2.2))
        draw_line(frame, W / 2 - 160 * S * g, H / 2 + 62 * S, W / 2 + 160 * S * g, H / 2 + 62 * S, ACCENT, max(1, 1.5 * S), 0.9)
        en = text_sprite(fb, self.film.get("title_en", ""), "cinzel", 22 * S, MUTED, shadow=0, tracking=0.5)
        _place_center(frame, en, W / 2, H / 2 + 104 * S, ease_out_cubic(seg(lt, 1.0, 2.2)))

    def sfx(self):
        return [("boom", self.t0 + 0.25, 3.5, -12), ("shimmer", self.t0 + 0.3, 2.5, -24)]


HUD_ROWS = ["water", "food", "signal", "battery"]


class SurvivalHUD(Gfx):
    z = 22

    def __init__(self, t0, t1, prev, cur, highlight):
        super().__init__(t0, t1)
        self.prev, self.cur, self.hl = prev, cur, set(highlight or [])

    def draw(self, frame, t, R):
        S, W, H, fb = R.S, R.W, R.H, R.fb
        lt, T = t - self.t0, self.t1 - self.t0
        a = ease_out_expo(seg(lt, 0, 0.7))
        out = ease_in_cubic(seg(lt, T - 0.55, T))
        vis = a * (1 - out)
        if vis <= 0.003:
            return
        pw, ph = 392 * S, 262 * S
        x1 = W - 64 * S + (1 - a) * 40 * S + out * 30 * S
        x0, y0 = x1 - pw, 62 * S
        glass(frame, x0, y0, pw, ph, S, opacity=vis)
        pad = 24 * S
        xr = x1 - pad
        h1 = text_sprite(fb, "سجل النجاة", "ar_bold", 27 * S, WHITE, shadow=0.4)
        _place_right(frame, h1, xr, y0 + 46 * S, vis)
        fl = text_sprite(fb, "FIELD LOG", "mono", 15 * S, MUTED, shadow=0)
        blit(frame, fl.a, x0 + pad, y0 + 44 * S, vis * 0.85, ax=fl.pad, ay=fl.base)
        day = self.cur.get("day", 1)
        tod = self.cur.get("time", "الصباح")
        sub = text_sprite(fb, f"اليوم {ar_num(day)} · {tod}", "ar_med", 21 * S,
                          ACCENT if ("day" in self.hl or "time" in self.hl) else MUTED, shadow=0.3)
        _place_right(frame, sub, xr, y0 + 80 * S, vis)
        draw_line(frame, x0 + pad, y0 + 96 * S, x1 - pad, y0 + 96 * S, (1, 1, 1), max(1, S), 0.18 * vis)
        labels = {"water": "الماء", "food": "الطعام", "signal": "الشبكة", "battery": "البطارية"}
        anim = ease_in_out_cubic(seg(lt, 0.9, 2.1))
        for i, key in enumerate(HUD_ROWS):
            ra = ease_out_cubic(seg(lt, 0.15 + i * 0.08, 0.6 + i * 0.08)) * (1 - out)
            yb = y0 + 136 * S + i * 36 * S
            hl = key in self.hl
            if hl:
                fl_a = math.exp(-max(0, lt - 0.8) * 1.6) * (lt > 0.8)
                draw_rect(frame, x0 + 8 * S, yb - 25 * S, x1 - 8 * S, yb + 9 * S, ACCENT, 0.16 * fl_a * ra + 0.05 * ra)
            # icon
            ix, iy = xr - 11 * S, yb - 8 * S
            self._icon(frame, key, ix, iy, S, ra)
            lb = text_sprite(fb, labels[key], "ar_med", 22 * S, WHITE if hl else CREAM, shadow=0.3)
            _place_right(frame, lb, xr - 32 * S, yb, ra)
            bx0, bx1 = x0 + pad, x0 + pad + 170 * S
            v0, v1 = self.prev.get(key, 1.0), self.cur.get(key, 1.0)
            v = v0 + (v1 - v0) * anim if hl else v1
            if key in ("water", "food"):
                col = DANGER if v < 0.2 else (TEAL if key == "water" else GOLD)
                if v < 0.2:
                    col = tuple(c * (0.75 + 0.25 * math.sin(lt * 7)) for c in col)
                draw_rect(frame, bx0, yb - 11 * S, bx1, yb - 5 * S, (1, 1, 1), 0.14 * ra)
                draw_rect(frame, bx0, yb - 11 * S, bx0 + (bx1 - bx0) * v, yb - 5 * S, col, 0.95 * ra)
                if hl and v0 > v1:
                    lost = (v0 - v) * (bx1 - bx0)
                    draw_rect(frame, bx0 + (bx1 - bx0) * v, yb - 11 * S, bx0 + (bx1 - bx0) * v + lost, yb - 5 * S, DANGER, 0.5 * ra * (1 - anim))
            elif key == "signal":
                st = text_sprite(fb, "لا توجد إشارة", "ar_med", 20 * S, DANGER, shadow=0.3)
                blink = 0.65 + 0.35 * math.sin(lt * 5) if hl else 1.0
                blit(frame, st.a, bx0, yb, ra * blink, ax=st.pad, ay=st.base)
            else:
                draw_rect(frame, bx0, yb - 11 * S, bx0 + 110 * S, yb - 5 * S, (1, 1, 1), 0.14 * ra)
                col = DANGER if v < 0.2 else CREAM
                draw_rect(frame, bx0, yb - 11 * S, bx0 + 110 * S * v, yb - 5 * S, col, 0.95 * ra)
                pct = text_sprite(fb, f"{ar_num(int(round(v * 100)))}٪", "ar_med", 20 * S, col, shadow=0.3)
                blit(frame, pct.a, bx0 + 122 * S, yb, ra, ax=pct.pad, ay=pct.base)

    def _icon(self, frame, key, x, y, S, a):
        if a <= 0.01:
            return
        c = CREAM
        if key == "water":
            draw_circle(frame, x, y + 3 * S, 5.5 * S, TEAL, opacity=a)
            pts = np.array([(x - 5.2 * S, y + 2 * S), (x, y - 9 * S), (x + 5.2 * S, y + 2 * S)])
            m = np.zeros((int(30 * S) + 4, int(30 * S) + 4), np.uint8)
            ox, oy = int(x - 15 * S), int(y - 15 * S)
            cv2.fillPoly(m, [np.array([_pt(px - ox, py - oy) for px, py in pts], np.int32)], 255, cv2.LINE_AA, SHIFT)
            fill_mask(frame, m, ox, oy, TEAL, a)
        elif key == "food":
            draw_rect(frame, x - 7 * S, y - 5 * S, x + 7 * S, y + 6 * S, GOLD, a)
            draw_line(frame, x - 7 * S, y - 1 * S, x + 7 * S, y - 1 * S, INK, max(1, S), a * 0.6)
        elif key == "signal":
            for i in range(4):
                hgt = (3 + i * 3) * S
                draw_rect(frame, x - 8 * S + i * 4.5 * S, y + 6 * S - hgt, x - 5.5 * S + i * 4.5 * S, y + 6 * S, c, a * 0.45)
            draw_line(frame, x - 7 * S, y - 6 * S, x + 7 * S, y + 6 * S, DANGER, max(1, 2 * S), a)
            draw_line(frame, x + 7 * S, y - 6 * S, x - 7 * S, y + 6 * S, DANGER, max(1, 2 * S), a)
        else:
            draw_rect(frame, x - 9 * S, y - 5 * S, x + 7 * S, y - 3.8 * S, c, a)
            draw_rect(frame, x - 9 * S, y + 4.8 * S, x + 7 * S, y + 6 * S, c, a)
            draw_rect(frame, x - 9 * S, y - 5 * S, x - 7.8 * S, y + 6 * S, c, a)
            draw_rect(frame, x + 5.8 * S, y - 5 * S, x + 7 * S, y + 6 * S, c, a)
            draw_rect(frame, x + 7 * S, y - 2 * S, x + 9 * S, y + 3 * S, c, a)

    def sfx(self):
        out = [("blips", self.t0 + 0.05, 0.5, -26)]
        if any(self.cur.get(k, 1) < 0.2 and k in self.hl for k in ("water", "food", "battery")):
            out.append(("warn", self.t0 + 0.95, 0.6, -24))
        return out


class SpeciesCard(Gfx):
    z = 21

    def __init__(self, t0, t1, data, num):
        super().__init__(t0, t1)
        self.d, self.num = data, num

    def draw(self, frame, t, R):
        S, W, H, fb = R.S, R.W, R.H, R.fb
        lt, T = t - self.t0, self.t1 - self.t0
        g = ease_out_expo(seg(lt, 0.0, 0.8))
        out = ease_in_cubic(seg(lt, T - 0.6, T))
        vis = 1 - out
        pw, ph = 610 * S, 252 * S
        x1, y1 = W - 84 * S, H - 92 * S
        x0, y0 = x1 - pw, y1 - ph
        glass(frame, x0, y0, pw, ph, S, opacity=vis, clip_w=g)
        # target brackets
        b = ease_out_expo(seg(lt, 0.2, 1.0))
        off = (24 - 16 * b) * S
        L = 22 * S
        ca = b * vis * 0.95
        for sx, sy in ((x0 - off, y0 - off), (x1 + off, y0 - off), (x0 - off, y1 + off), (x1 + off, y1 + off)):
            dx = L if sx < x0 else -L
            dy = L if sy < y0 else -L
            draw_line(frame, sx, sy, sx + dx, sy, ACCENT, max(1, 2 * S), ca)
            draw_line(frame, sx, sy, sx, sy + dy, ACCENT, max(1, 2 * S), ca)
        xr = x1 - 28 * S
        lab = text_sprite(fb, f"ملف كائن · {ar_num(self.num)}", "ar_med", 20 * S, MUTED, shadow=0.2)
        la = ease_out_expo(seg(lt, 0.35, 0.95))
        _place_right(frame, lab, xr, y0 + 40 * S, vis, arr=sprite_wipe(lab.a, la, 40 * S, True))
        sf = text_sprite(fb, "SPECIES FILE", "mono", 14 * S, MUTED, shadow=0)
        blit(frame, sf.a, x0 + 28 * S, y0 + 38 * S, la * vis * 0.8, ax=sf.pad, ay=sf.base)
        nm = text_sprite(fb, self.d["name_ar"], "ar_black", 52 * S, WHITE, shadow=0.5)
        na = ease_out_expo(seg(lt, 0.45, 1.25))
        arr = blur_sprite(sprite_wipe(nm.a, na, 90 * S, True), 5 * S * (1 - na))
        _place_right(frame, nm, xr, y0 + 104 * S, vis, arr=arr, dx=(1 - na) * 22 * S)
        # latin name typewriter
        latin = self.d["latin"]
        k = int(len(latin) * seg(lt, 0.9, 1.9))
        if k > 0:
            ls = text_sprite(fb, latin[:k], "serif_it", 34 * S, GOLD, shadow=0.4)
            full = text_sprite(fb, latin, "serif_it", 34 * S, GOLD, shadow=0.4)
            blit(frame, ls.a, xr - full.ink_w, y0 + 146 * S, vis, ax=ls.pad, ay=ls.base)
            if k < len(latin) or (lt % 0.8) < 0.4:
                cxp = xr - full.ink_w + ls.ink_w + 4 * S
                draw_rect(frame, cxp, y0 + 120 * S, cxp + 2 * S, y0 + 150 * S, GOLD, 0.8 * vis * (lt < 3.0))
        # ruler divider
        rg = ease_out_expo(seg(lt, 0.7, 1.6))
        yl = y0 + 166 * S
        xa = xr - (xr - (x0 + 28 * S)) * rg
        draw_line(frame, xr, yl, xa, yl, (1, 1, 1), max(1, S), 0.35 * vis)
        n = int((xr - (x0 + 28 * S)) / (14 * S))
        for i in range(n + 1):
            xx = xr - i * 14 * S
            if xx < xa:
                break
            hh = (7 if i % 5 == 0 else 3.5) * S
            draw_line(frame, xx, yl, xx, yl + hh, (1, 1, 1), max(1, S), 0.35 * vis)
        for i, fact in enumerate(self.d.get("facts", [])[:2]):
            fa = ease_out_expo(seg(lt, 1.3 + i * 0.25, 2.1 + i * 0.25))
            fs = text_sprite(fb, fact, "ar_med", 22 * S, CREAM, shadow=0.3)
            yb = y0 + 202 * S + i * 32 * S
            draw_circle(frame, xr - 4 * S, yb - 8 * S, 3.4 * S * fa, ACCENT, opacity=vis)
            _place_right(frame, fs, xr - 18 * S, yb, vis, arr=sprite_wipe(fs.a, fa, 60 * S, True))

    def sfx(self):
        return [("whoosh_soft", self.t0 - 0.05, 0.6, -24), ("blips", self.t0 + 0.4, 0.9, -25), ("blips", self.t0 + 1.0, 0.8, -28)]


# --- map data (illustrative, lon/lat) -------------------------------------------------------
SA_COAST = [(-77.9, 7.2), (-77.3, 8.4), (-76.8, 8.8), (-75.6, 9.6), (-75.5, 10.4), (-74.8, 11.0), (-74.2, 11.3), (-73.3, 11.3),
            (-72.2, 11.9), (-71.6, 12.4), (-71.1, 12.0), (-71.3, 11.0), (-71.6, 10.6), (-71.9, 10.0), (-71.4, 9.4), (-71.0, 10.0),
            (-71.0, 10.9), (-70.2, 11.6), (-70.0, 12.2), (-69.6, 11.5), (-68.4, 10.9), (-68.0, 10.6), (-66.9, 10.6), (-66.0, 10.6),
            (-64.5, 10.2), (-63.8, 10.7), (-62.3, 10.6), (-61.8, 10.6), (-62.0, 10.0), (-61.0, 9.0), (-60.0, 8.5), (-59.0, 7.7),
            (-58.3, 7.0), (-57.2, 6.0), (-55.9, 5.9), (-54.8, 6.0), (-53.9, 5.7), (-52.9, 5.3), (-51.9, 4.6), (-51.3, 4.2),
            (-50.8, 2.5), (-50.0, 1.7), (-49.9, 1.0), (-50.6, 0.3), (-49.3, -0.2), (-48.4, -0.7), (-48.0, -0.8), (-46.6, -0.9),
            (-45.3, -1.4), (-44.6, -2.3), (-43.4, -2.4), (-42.2, -2.7), (-41.2, -2.9), (-39.9, -2.9), (-38.5, -3.7), (-37.2, -4.8),
            (-35.5, -5.1), (-35.2, -5.6), (-34.9, -7.2), (-34.9, -8.1), (-35.4, -9.2), (-36.5, -10.5), (-37.4, -11.6), (-38.5, -13.0),
            (-39.0, -14.5), (-39.0, -16.0), (-39.2, -17.6), (-39.7, -19.0), (-40.3, -20.3), (-40.9, -21.6), (-41.9, -22.4),
            (-43.2, -23.0), (-44.7, -23.3), (-46.3, -24.0), (-47.9, -25.2), (-48.6, -26.4), (-48.6, -28.0), (-49.7, -29.2),
            (-50.7, -31.0), (-52.1, -32.3), (-53.4, -33.7), (-54.9, -34.9), (-58.0, -34.6), (-71.6, -33.0), (-71.4, -30.0),
            (-71.4, -28.6), (-70.9, -27.6), (-70.4, -25.0), (-70.4, -23.5), (-70.1, -21.4), (-70.2, -19.9), (-70.3, -18.4),
            (-71.4, -17.6), (-72.5, -16.8), (-74.0, -15.8), (-75.2, -15.2), (-76.0, -14.0), (-76.4, -13.0), (-77.2, -12.0),
            (-77.8, -10.8), (-78.6, -9.2), (-79.4, -7.9), (-80.0, -7.0), (-81.0, -6.0), (-81.2, -5.2), (-81.3, -4.6), (-80.4, -3.6),
            (-79.9, -2.6), (-80.6, -2.3), (-80.9, -1.5), (-80.4, -0.6), (-80.0, 0.4), (-79.4, 1.0), (-78.9, 1.6), (-78.6, 2.2),
            (-77.9, 2.7), (-77.4, 3.5), (-77.3, 4.5), (-77.5, 5.6), (-77.4, 6.6), (-77.9, 7.2)]
PANAMA = [(-77.9, 7.2), (-78.4, 8.3), (-79.6, 9.0), (-80.4, 8.3), (-81.2, 8.0), (-82.0, 8.4), (-83.0, 9.0)]
AMAZON_RIVER = [(-76.2, -6.0), (-75.0, -4.9), (-73.6, -4.5), (-73.25, -3.75), (-72.0, -3.9), (-69.95, -4.2), (-68.0, -3.4),
                (-66.0, -2.6), (-64.7, -3.35), (-63.1, -4.1), (-61.5, -3.6), (-60.0, -3.1), (-58.4, -3.2), (-55.5, -1.9),
                (-54.7, -2.4), (-52.5, -1.6), (-51.0, -0.6), (-50.0, -0.3)]
RIO_NEGRO = [(-60.0, -3.1), (-61.5, -2.2), (-62.9, -0.9), (-64.8, -0.4), (-66.9, 0.6), (-67.5, 1.8)]
MADEIRA = [(-58.8, -3.4), (-60.5, -5.4), (-62.5, -7.3), (-63.9, -8.8), (-65.3, -10.8)]
BASIN = [(-79.0, -1.5), (-77.5, 1.5), (-74.0, 2.5), (-70.5, 3.8), (-66.5, 4.5), (-62.5, 4.2), (-59.5, 3.0), (-56.0, 2.0),
         (-52.5, 1.5), (-50.5, 0.5), (-49.5, -1.5), (-51.5, -3.5), (-52.0, -7.5), (-53.5, -11.5), (-55.0, -14.5), (-58.0, -16.0),
         (-62.0, -17.5), (-65.5, -18.0), (-67.5, -16.5), (-69.5, -14.5), (-72.0, -13.5), (-74.5, -11.5), (-76.5, -9.0),
         (-78.0, -6.5), (-79.3, -4.0)]
LON0, LON1, LAT0, LAT1 = -82.0, -34.0, 14.0, -24.0


class MapCard(Gfx):
    """Explorer's-journal paper map: coastline draw-on, basin wash, Amazon river, unknown position, distance line."""
    z = 18

    def __init__(self, t0, t1, t_phase2):
        super().__init__(t0, t1)
        self.t2 = t_phase2
        self.base = None

    def _build(self, R):
        S = R.S
        self.cw, self.ch = int(700 * S), int(610 * S)
        cw, ch = self.cw, self.ch
        r = np.random.default_rng(77)
        paper = np.ones((ch, cw, 3), np.float32) * np.float32(CREAM)
        blot = fractal_noise(ch, cw, r, 4, 3.0)
        fib = fractal_noise(ch, cw, r, 2, 40.0)
        paper *= (0.93 + 0.07 * blot)[..., None]
        paper *= (0.98 + 0.03 * fib)[..., None]
        yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
        e = np.sqrt(((xx - cw / 2) / (cw / 2)) ** 2 + ((yy - ch / 2) / (ch / 2)) ** 2)
        paper *= (1 - 0.16 * np.clip(e - 0.55, 0, 1) ** 1.5)[..., None]
        paper *= np.float32([1.0, 0.985, 0.95])
        m = rrect_mask(cw, ch, 6 * S).astype(np.float32) / 255.0
        self.mx0, self.my0, self.mx1, self.my1 = 34 * S, 92 * S, cw - 34 * S, ch - 34 * S
        rgb = paper.copy()
        # frame lines
        draw_polyline(rgb, [(14 * S, 14 * S), (cw - 14 * S, 14 * S), (cw - 14 * S, ch - 14 * S), (14 * S, ch - 14 * S)], SEPIA, max(1, 2 * S), 0.8, closed=True)
        draw_polyline(rgb, [(20 * S, 20 * S), (cw - 20 * S, 20 * S), (cw - 20 * S, ch - 20 * S), (20 * S, ch - 20 * S)], SEPIA, max(1, S), 0.5, closed=True)
        # graticule
        for lon in range(-80, -33, 10):
            x = self._x(lon)
            draw_line(rgb, x, self.my0, x, self.my1, SEPIA, max(1, S), 0.16)
            lab = text_sprite(R.fb, f"{abs(lon)}°W", "mono", 12 * S, SEPIA, shadow=0)
            blit(rgb, lab.a, x, self.my1 + 16 * S, 0.7, ax=lab.pad + lab.ink_w / 2, ay=lab.base)
        for lat in range(10, -25, -10):
            y = self._y(lat)
            draw_line(rgb, self.mx0, y, self.mx1, y, SEPIA, max(1, S), 0.16)
            lab = text_sprite(R.fb, f"{abs(lat)}°{'N' if lat > 0 else ('S' if lat < 0 else '')}", "mono", 12 * S, SEPIA, shadow=0)
            blit(rgb, lab.a, self.mx0 - 4 * S, y - 4 * S, 0.7, ax=lab.pad + lab.ink_w, ay=lab.base)
        # compass rose
        ccx, ccy, cr = cw - 80 * S, ch - 92 * S, 34 * S
        draw_circle(rgb, ccx, ccy, cr, SEPIA, thick=max(1, S), opacity=0.55)
        draw_polyline(rgb, [(ccx, ccy - cr * 1.25), (ccx + 6 * S, ccy), (ccx, ccy + cr * 1.25), (ccx - 6 * S, ccy), (ccx, ccy - cr * 1.25)], SEPIA, max(1, S), 0.7)
        nlab = text_sprite(R.fb, "N", "cinzel", 16 * S, SEPIA, shadow=0)
        blit(rgb, nlab.a, ccx, ccy - cr * 1.35, 0.8, ax=nlab.pad + nlab.ink_w / 2, ay=nlab.base)
        tl = text_sprite(R.fb, "حوض الأمازون", "ruqaa", 52 * S, SEPIA, shadow=0)
        blit(rgb, tl.a, cw - 40 * S, 70 * S, 0.92, ax=tl.pad + tl.ink_w, ay=tl.base)
        st = text_sprite(R.fb, "AMAZON BASIN · FIELD MAP", "mono", 13 * S, SEPIA, shadow=0)
        blit(rgb, st.a, 40 * S, 64 * S, 0.7, ax=st.pad, ay=st.base)
        oc = text_sprite(R.fb, "المحيط الأطلسي", "ar_light", 20 * S, (0.25, 0.35, 0.45), shadow=0)
        blit(rgb, oc.a, self._x(-42), self._y(4), 0.8, ax=oc.pad + oc.ink_w / 2, ay=oc.base)
        op = text_sprite(R.fb, "المحيط الهادئ", "ar_light", 20 * S, (0.25, 0.35, 0.45), shadow=0)
        blit(rgb, op.a, self._x(-77), self._y(-17), 0.8, ax=op.pad + op.ink_w / 2, ay=op.base)
        self.static = np.dstack([rgb * m[..., None], m]).astype(np.float32)
        # basin polygon mask (green wash + hatching)
        bm = np.zeros((ch, cw), np.uint8)
        cv2.fillPoly(bm, [np.array([_pt(self._x(a), self._y(b)) for a, b in BASIN], np.int32)], 255, cv2.LINE_AA, SHIFT)
        hatch = np.zeros((ch, cw), np.uint8)
        for k in range(-ch, cw, int(9 * S) + 1):
            cv2.line(hatch, (k, 0), (k + ch, ch), 255, 1, cv2.LINE_AA)
        self.basin = bm.astype(np.float32) / 255.0
        self.hatch = (hatch.astype(np.float32) / 255.0) * self.basin
        self.coast = [(self._x(a), self._y(b)) for a, b in SA_COAST]
        self.panama = [(self._x(a), self._y(b)) for a, b in PANAMA]
        self.river = [(self._x(a), self._y(b)) for a, b in AMAZON_RIVER]
        self.negro = [(self._x(a), self._y(b)) for a, b in RIO_NEGRO]
        self.madeira = [(self._x(a), self._y(b)) for a, b in MADEIRA]
        self.marker = (self._x(-66.0), self._y(-6.2))
        self.help = (self._x(-60.8), self._y(-1.8))
        # shadow sprite
        sh = cv2.GaussianBlur(np.pad(m, int(30 * S)), (0, 0), 14 * S) * 0.55
        self.shadow = np.dstack([np.zeros(sh.shape + (3,), np.float32), sh]).astype(np.float32)
        self.base = True

    def _x(self, lon):
        return self.mx0 + (lon - LON0) / (LON1 - LON0) * (self.mx1 - self.mx0)

    def _y(self, lat):
        return self.my0 + (LAT0 - lat) / (LAT0 - LAT1) * (self.my1 - self.my0)

    def draw(self, frame, t, R):
        if self.base is None:
            self._build(R)
        S, W, H, fb = R.S, R.W, R.H, R.fb
        lt, T = t - self.t0, self.t1 - self.t0
        a_in = ease_out_back(seg(lt, 0.0, 0.9), 1.2)
        out = ease_in_cubic(seg(lt, T - 0.7, T))
        if out >= 1:
            return
        card = self.static.copy()
        rgb = card[..., :3]
        ink = SEPIA
        # basin wash
        ba = ease_out_cubic(seg(lt, 1.2, 2.2))
        if ba > 0:
            col = np.float32((0.33, 0.50, 0.30))
            a = (self.basin * 0.30 + self.hatch * 0.35)[..., None] * ba
            rgb += (col - rgb) * a
        cf = ease_in_out_cubic(seg(lt, 0.3, 2.0))
        draw_polyline(rgb, self.coast, ink, max(1, 2.2 * S), 0.9, frac=cf)
        draw_polyline(rgb, self.panama, ink, max(1, 2.2 * S), 0.9, frac=cf)
        river_c = (0.16, 0.36, 0.55)
        rf = ease_in_out_cubic(seg(lt, 1.5, 3.0))
        draw_polyline(rgb, self.river, river_c, max(1, 3.0 * S), 0.95, frac=rf)
        tf = ease_in_out_cubic(seg(lt, 2.4, 3.4))
        draw_polyline(rgb, self.negro, river_c, max(1, 1.8 * S), 0.85, frac=tf)
        draw_polyline(rgb, self.madeira, river_c, max(1, 1.8 * S), 0.85, frac=tf)
        la = ease_out_cubic(seg(lt, 2.6, 3.4))
        if la > 0:
            rl = text_sprite(fb, "نهر الأمازون", "ar_bold", 20 * S, river_c, shadow=0)
            blit(rgb, rl.a, self._x(-55.0), self._y(-4.2) + 16 * S, la, ax=rl.pad + rl.ink_w / 2, ay=rl.base)
            ar1 = text_sprite(fb, "≈ ٥٫٥ مليون كم²", "ar_black", 30 * S, ink, shadow=0)
            ar2 = text_sprite(fb, "من الغابات المطيرة", "ar_med", 19 * S, ink, shadow=0)
            blit(rgb, ar1.a, self._x(-58.5), self._y(-12.2), la, ax=ar1.pad + ar1.ink_w / 2, ay=ar1.base)
            blit(rgb, ar2.a, self._x(-58.5), self._y(-12.2) + 26 * S, la, ax=ar2.pad + ar2.ink_w / 2, ay=ar2.base)
        # unknown position marker
        ma = ease_out_back(seg(lt, 3.2, 3.8))
        if ma > 0:
            mx, my = self.marker
            pulse = (lt * 0.8) % 1.0
            draw_circle(rgb, mx, my, (8 + 26 * pulse) * S, ACCENT, thick=max(1, 2 * S), opacity=0.8 * (1 - pulse) * ma)
            draw_circle(rgb, mx, my, 7 * S * ma, ACCENT, opacity=1.0)
            q = text_sprite(fb, "؟", "ar_black", 22 * S, (1, 1, 1), shadow=0)
            blit(rgb, q.a, mx, my + 8 * S, ma, ax=q.pad + q.ink_w / 2, ay=q.base)
            ml = text_sprite(fb, "موقعه: مجهول", "ar_black", 21 * S, ACCENT, shadow=0)
            mlx, mly = mx - 4 * S, my + 44 * S
            draw_rect(rgb, mlx - ml.ink_w - 8 * S, mly - 21 * S, mlx + 8 * S, mly + 7 * S, CREAM, 0.85 * ma)
            blit(rgb, ml.a, mlx, mly, ma, ax=ml.pad + ml.ink_w, ay=ml.base)
        # phase 2: distance to help
        p2 = t - self.t2
        if p2 > 0:
            df = ease_in_out_cubic(seg(p2, 0.3, 1.8))
            draw_polyline(rgb, [self.marker, self.help], ACCENT, max(1, 2 * S), 0.9, frac=df, dashed=7 * S)
            ha = ease_out_back(seg(p2, 1.5, 2.1))
            if ha > 0:
                hx, hy = self.help
                draw_circle(rgb, hx, hy, 6 * S * ha, (0.16, 0.36, 0.55), opacity=1.0)
                draw_circle(rgb, hx, hy, 12 * S * ha, (0.16, 0.36, 0.55), thick=max(1, 1.5 * S), opacity=0.8)
                hl = text_sprite(fb, "أقرب مساعدة؟", "ar_black", 21 * S, (0.16, 0.36, 0.55), shadow=0)
                hlx, hly = hx - 2 * S, hy - 22 * S
                draw_rect(rgb, hlx - hl.ink_w / 2 - 8 * S, hly - 21 * S, hlx + hl.ink_w / 2 + 8 * S, hly + 7 * S, CREAM, 0.85 * ha)
                blit(rgb, hl.a, hlx, hly, ha, ax=hl.pad + hl.ink_w / 2, ay=hl.base)
        card[..., :3] = rgb * card[..., 3:4]
        # place with rotation + float
        ang = -2.0 + 2.5 * (1 - a_in) + 0.25 * math.sin(lt * 0.6)
        cx = 150 * S + self.cw / 2 - (1 - a_in) * (self.cw + 200 * S) - out * (self.cw + 260 * S)
        cy = H / 2 + 6 * S * math.sin(lt * 0.5)
        M = cv2.getRotationMatrix2D((self.cw / 2, self.ch / 2), ang, 1.0)
        pad = int(40 * S)
        M[0, 2] += pad
        M[1, 2] += pad
        rot = cv2.warpAffine(card, M, (self.cw + 2 * pad, self.ch + 2 * pad), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        shp = self.shadow
        blit(frame, shp, cx + 10 * S, cy + 16 * S, 0.9, ax=shp.shape[1] / 2, ay=shp.shape[0] / 2)
        blit(frame, rot, cx, cy, 1.0, ax=rot.shape[1] / 2, ay=rot.shape[0] / 2)

    def sfx(self):
        return [("paper", self.t0 - 0.05, 0.7, -20), ("scribble", self.t0 + 0.3, 1.8, -30),
                ("scribble", self.t0 + 1.5, 1.5, -30), ("tick", self.t0 + 3.2, 0.2, -24),
                ("scribble", self.t2 + 0.3, 1.5, -30), ("tick", self.t2 + 1.6, 0.2, -24)]


# ----------------------------------------------------------------------------------------------
# Transitions
# ----------------------------------------------------------------------------------------------
TRANS_DUR = {"cut": 0.0, "dissolve": 0.9, "slow_dissolve": 1.5, "dip_black": 1.4, "dip_night": 1.6,
             "whip_left": 0.46, "whip_right": 0.46, "whip_up": 0.46, "whip_down": 0.46, "zoom_through": 0.56,
             "light_leak": 1.25, "ink": 1.2, "flash": 0.55, "blur_cut": 0.45}
TRANS_SFX = {"whip_left": [("whoosh", -0.30, 0.6, -15)], "whip_right": [("whoosh", -0.30, 0.6, -15)],
             "whip_up": [("whoosh", -0.30, 0.6, -15)], "whip_down": [("whoosh", -0.30, 0.6, -15)],
             "zoom_through": [("whoosh_bright", -0.32, 0.65, -16), ("thump", 0.0, 0.6, -18)],
             "flash": [("impact", -0.01, 1.4, -11)], "ink": [("swell", -0.6, 1.2, -20), ("boom", 0.05, 2.0, -21)],
             "light_leak": [("shimmer", -0.6, 1.6, -24)], "dip_black": [("boom", 0.0, 2.2, -21)],
             "dip_night": [("boom", 0.0, 2.2, -22)]}


class Trans:
    def __init__(self, kind, tc, a, b, seed):
        self.kind, self.tc, self.a, self.b = kind, tc, a, b
        d = TRANS_DUR.get(kind, 0.0)
        if kind == "flash":
            self.w0, self.w1 = tc - 0.04, tc + d
        else:
            self.w0, self.w1 = tc - d / 2, tc + d / 2
        self.seed = seed
        self.mask_noise = None


def whip_offsets(kind, q, W, H):
    """Return (dx_a, dy_a, dx_b, dy_b, velocity_px_per_unit) for a whip at progress q."""
    d = {"whip_left": (1, 0), "whip_right": (-1, 0), "whip_up": (0, 1), "whip_down": (0, -1)}[kind]
    span = (W if d[0] else H) * 0.85
    if q < 0.5:
        e = ease_in_cubic(q / 0.5)
        oa, ob = e * span, -span
    else:
        e = 1 - ease_out_cubic((q - 0.5) / 0.5)
        oa, ob = span, -e * span
    vel = span * 3 * (1 - abs(q - 0.5) * 2) ** 2
    return d[0] * oa, d[1] * oa, d[0] * ob * -1 * -1, d[1] * ob, vel, d


def directional_blur(frame, length, horiz=True):
    if length < 2:
        return frame
    H, W = frame.shape[:2]
    sm = cv2.resize(frame, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    k = max(1, int(length / 2))
    sm = cv2.blur(sm, (k, 1) if horiz else (1, k))
    return cv2.resize(sm, (W, H), interpolation=cv2.INTER_LINEAR)


def radial_blur(frame, amount):
    if amount < 0.004:
        return frame
    H, W = frame.shape[:2]
    sm = cv2.resize(frame, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    acc = sm.copy()
    n = 6
    for i in range(1, n):
        s = 1 + amount * i / n
        M = cv2.getRotationMatrix2D((W / 4, H / 4), 0, s)
        acc += cv2.warpAffine(sm, M, (W // 2, H // 2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.resize(acc / n, (W, H), interpolation=cv2.INTER_LINEAR)


# ----------------------------------------------------------------------------------------------
# Sound design (procedural)
# ----------------------------------------------------------------------------------------------
SR = 48000


def _lp_varying(x, fc):
    """One-pole low-pass with time-varying cutoff (block-wise)."""
    y = np.empty_like(x)
    a = 1 - np.exp(-2 * np.pi * np.clip(fc, 20, SR * 0.45) / SR)
    state = 0.0
    B = 64
    for i in range(0, len(x), B):
        ai = a[i]
        seg_ = x[i:i + B]
        out = np.empty_like(seg_)
        s = state
        for j in range(len(seg_)):
            s += ai * (seg_[j] - s)
            out[j] = s
        y[i:i + B] = out
        state = s
    return y


try:
    from scipy.signal import lfilter as _lfilter

    def _lp_varying(x, fc):  # noqa: F811  (faster block-wise IIR when scipy exists)
        y = np.empty_like(x)
        zi = np.zeros(1)
        B = 256
        for i in range(0, len(x), B):
            a = 1 - math.exp(-2 * math.pi * min(max(float(fc[i]), 20.0), SR * 0.45) / SR)
            y[i:i + B], zi = _lfilter([a], [1, -(1 - a)], x[i:i + B], zi=zi)
        return y
except Exception:  # pragma: no cover
    pass


def _norm(x):
    m = np.max(np.abs(x)) + 1e-9
    return x / m


def synth(kind, dur, rng):
    n = max(16, int(dur * SR))
    t = np.arange(n) / SR
    tt = t / dur
    if kind in ("whoosh", "whoosh_soft", "whoosh_bright"):
        pk = 0.62
        shape = np.where(tt < pk, (tt / pk) ** 2.2, ((1 - tt) / (1 - pk)) ** 1.6)
        bright = {"whoosh": 1.0, "whoosh_soft": 0.45, "whoosh_bright": 1.6}[kind]
        x = rng.standard_normal(n)
        y = _lp_varying(x, 200 + 4200 * bright * shape)
        y = _lp_varying(y, 300 + 5200 * bright * shape)
        y = y - _lp_varying(y, np.full(n, 90.0))
        y = _norm(y) * shape ** 1.1
        pan = np.clip(-0.7 + 1.4 * tt, -1, 1)
        return np.stack([y * np.sqrt((1 - pan) / 2), y * np.sqrt((1 + pan) / 2)], 1)
    if kind in ("boom", "thump", "hit"):
        f0 = {"boom": 52, "thump": 70, "hit": 62}[kind]
        dec = {"boom": 2.2, "thump": 9, "hit": 5}[kind]
        f = f0 * (1 - 0.35 * np.minimum(t / 0.4, 1))
        ph = 2 * np.pi * np.cumsum(f) / SR
        body = np.sin(ph) * np.exp(-t * dec) * (1 - np.exp(-t * 400))
        sub = np.sin(ph * 0.5) * np.exp(-t * dec * 0.8) * 0.5
        click = _lp_varying(rng.standard_normal(n), np.full(n, 1800.0)) * np.exp(-t * 60) * 0.35
        y = np.tanh((body + sub + click) * 1.6)
        return np.stack([y, y], 1)
    if kind == "impact":
        a = synth("boom", dur, rng)
        crack = _lp_varying(rng.standard_normal(n), np.full(n, 6000.0)) * np.exp(-t * 18)
        crack = crack - _lp_varying(crack, np.full(n, 800.0))
        return a + np.stack([crack, crack], 1) * 0.6
    if kind == "riser":
        fc = 300 * (20 ** tt)
        y = _lp_varying(rng.standard_normal(n), fc)
        y = y - _lp_varying(y, fc * 0.4)
        tone = np.sin(2 * np.pi * np.cumsum(180 * (1 + 2.5 * tt ** 2)) / SR) * 0.25
        env = tt ** 2.4 * (1 - np.exp(-(1 - tt) * 60))
        y = (_norm(y) + tone) * env
        return np.stack([y, y], 1)
    if kind == "swell":
        env = np.sin(np.pi * np.clip(tt, 0, 1)) ** 2
        y = _lp_varying(rng.standard_normal(n), 400 + 1600 * env)
        return np.stack([_norm(y) * env, _norm(np.roll(y, 37)) * env], 1)
    if kind == "shimmer":
        y = np.zeros(n)
        for f in rng.uniform(1100, 4200, 6):
            y += np.sin(2 * np.pi * f * t + rng.random() * 6.28) * (0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(3, 7) * t))
        env = np.minimum(tt / 0.55, 1) ** 2 * np.exp(-np.maximum(tt - 0.55, 0) * 6)
        y = _norm(y) * env
        return np.stack([y, np.roll(y, 211)], 1)
    if kind == "tick":
        y = (np.sin(2 * np.pi * 1850 * t) + 0.4 * np.sin(2 * np.pi * 3700 * t)) * np.exp(-t * 110)
        return np.stack([y, y], 1)
    if kind == "blips":
        y = np.zeros(n)
        k = int(dur / 0.07)
        for i in range(k):
            s = int(i * 0.07 * SR + rng.uniform(0, 0.02) * SR)
            if s >= n:
                break
            f = rng.choice([1320, 1560, 1760, 2093, 2349])
            L = min(n - s, int(0.035 * SR))
            tl = np.arange(L) / SR
            y[s:s + L] += np.sin(2 * np.pi * f * tl) * np.exp(-tl * 90) * rng.uniform(0.5, 1)
        return np.stack([y, y], 1)
    if kind == "warn":
        y = np.zeros(n)
        for i, f in enumerate((880, 660)):
            s = int(i * 0.18 * SR)
            L = min(n - s, int(0.16 * SR))
            tl = np.arange(L) / SR
            y[s:s + L] += np.sin(2 * np.pi * f * tl) * np.minimum(tl / 0.01, 1) * np.exp(-tl * 12)
        return np.stack([y, y], 1)
    if kind == "heartbeat":
        y = np.zeros(n)
        for off, g in ((0.0, 1.0), (0.27, 0.65)):
            s = int(off * SR)
            L = min(n - s, int(0.2 * SR))
            tl = np.arange(L) / SR
            y[s:s + L] += np.sin(2 * np.pi * 48 * tl * (1 - 0.3 * tl)) * np.exp(-tl * 22) * g
        y = np.tanh(y * 1.4)
        return np.stack([y, y], 1)
    if kind == "thunder":
        b = np.cumsum(rng.standard_normal(n))
        b = b - _lp_varying(b, np.full(n, 25.0))
        b = _lp_varying(b, np.full(n, 280.0))
        env = (1 - np.exp(-t * 6)) * np.exp(-t * 1.1) * (0.7 + 0.3 * np.sin(2 * np.pi * 1.3 * t) ** 2)
        y = _norm(b) * env
        return np.stack([y, np.roll(y, 480)], 1)
    if kind == "crackle":
        y = np.zeros(n)
        for _ in range(int(dur * 70)):
            s = int(rng.uniform(0, n - 200))
            L = int(rng.uniform(30, 160))
            y[s:s + L] += rng.standard_normal(L) * np.exp(-np.arange(L) / 30)
        y = y - _lp_varying(y, np.full(n, 1500.0))
        return np.stack([y, y], 1)
    if kind == "paper":
        env = np.sin(np.pi * np.clip(tt, 0, 1)) ** 1.5 * (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 9 * t)))
        y = rng.standard_normal(n)
        y = _lp_varying(y, np.full(n, 6000.0))
        y = y - _lp_varying(y, np.full(n, 900.0))
        y = _norm(y) * env
        return np.stack([y, y], 1)
    if kind == "scribble":
        env = (0.4 + 0.6 * np.abs(np.sin(2 * np.pi * 5.5 * t + rng.random()))) * np.sin(np.pi * np.clip(tt, 0, 1))
        y = rng.standard_normal(n)
        y = _lp_varying(y, np.full(n, 5000.0)) - _lp_varying(y, np.full(n, 1800.0))
        y = _norm(y) * env
        return np.stack([y, y], 1)
    return np.zeros((n, 2))


def reverb(x, rng, tail=0.9, wet=0.22):
    L = int(tail * SR)
    ir = rng.standard_normal(L) * np.exp(-np.arange(L) / SR * (6.9 / tail))
    ir = _lp_varying(ir, np.linspace(6000, 1500, L))
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-9
    n = x.shape[0] + L
    nfft = 1 << int(math.ceil(math.log2(n)))
    out = np.zeros((n, 2))
    IR = np.fft.rfft(ir, nfft)
    for c in range(2):
        out[:, c] = np.fft.irfft(np.fft.rfft(x[:, c], nfft) * IR, nfft)[:n]
    dry = np.zeros((n, 2))
    dry[:x.shape[0]] = x
    return dry + out * wet


SFX_REF = {"whoosh": 1.0, "whoosh_soft": 1.0, "whoosh_bright": 1.0, "boom": 1.0, "thump": 1.0, "hit": 1.0,
           "impact": 1.0, "riser": 1.0, "swell": 1.0, "shimmer": 1.0, "tick": 1.0, "blips": 1.0, "warn": 1.0,
           "heartbeat": 1.0, "thunder": 1.0, "crackle": 1.0, "paper": 1.0, "scribble": 1.0}


def build_sfx_bus(cues, total, vo_rms, out_path, db_offset=0.0):
    n = int((total + 1.0) * SR)
    bus = np.zeros((n, 2), np.float32)
    rng = np.random.default_rng(2026)
    for kind, t0, dur, db in cues:
        if t0 + dur < 0 or t0 >= total:
            continue
        x = synth(kind, max(0.05, dur), rng)
        rms = np.sqrt(np.mean(x[x.shape[0] // 8:] ** 2)) + 1e-9
        x = x / rms
        if kind not in ("tick", "blips", "warn"):
            x = reverb(x, rng, tail=1.2 if kind in ("boom", "impact", "hit") else 0.8, wet=0.25)
        g = vo_rms * (10 ** ((db + db_offset) / 20.0))
        s = int(t0 * SR)
        a, b = max(0, s), min(n, s + x.shape[0])
        if b > a:
            bus[a:b] += (x[a - s:b - s] * g).astype(np.float32)
    peak = float(np.max(np.abs(bus))) if bus.size else 0
    if peak > 0.95:
        bus *= 0.95 / peak
    with wave.open(str(out_path), "wb") as wv:
        wv.setnchannels(2)
        wv.setsampwidth(2)
        wv.setframerate(SR)
        wv.writeframes((np.clip(bus, -1, 1) * 32767).astype("<i2").tobytes())
    return out_path


# ----------------------------------------------------------------------------------------------
# FFmpeg helpers
# ----------------------------------------------------------------------------------------------
def run(cmd, what="command"):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise RuntimeError(f"{what} failed:\n" + (p.stderr or p.stdout or "")[-5000:])
    return p.stdout.strip()


def media_duration(path):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "default=noprint_wrappers=1:nokey=1", str(path)], "ffprobe"))


def decode_mono(path, sr=16000):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode:
        raise RuntimeError("Audio decode failed:\n" + p.stderr.decode("utf-8", "replace")[-3000:])
    return np.frombuffer(p.stdout, np.float32)


def vo_analysis(path):
    sr = 16000
    x = decode_mono(path, sr)
    hop = int(0.02 * sr)
    n = len(x) // hop
    fr = x[: n * hop].reshape(n, hop)
    rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    voiced = db > max(np.percentile(db, 60) - 18, -45)
    vo_rms = float(np.sqrt(np.mean(fr[voiced] ** 2))) if voiced.any() else 0.1
    thr = max(np.percentile(db, 20) + 6, -50)
    sil = db < thr
    pauses = []
    i = 0
    while i < n:
        if sil[i]:
            j = i
            while j < n and sil[j]:
                j += 1
            if (j - i) * 0.02 >= 0.14:
                pauses.append((i * 0.02, j * 0.02))
            i = j
        else:
            i += 1
    return vo_rms, pauses


class FFWriter:
    def __init__(self, path, W, H, fps, crf, preset):
        self.path = Path(path)
        self.tmp = self.path.with_name(self.path.stem + ".part.mp4")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
               "-an", "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
               "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
               "-g", str(int(fps * 2)), "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
               "-movflags", "+faststart", str(self.tmp)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def write(self, frame_u8):
        try:
            self.proc.stdin.write(frame_u8.tobytes())
        except BrokenPipeError:
            err = self.proc.stderr.read().decode("utf-8", "replace")
            raise RuntimeError("FFmpeg stopped accepting frames:\n" + err[-4000:])

    def close(self):
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode("utf-8", "replace")
        rc = self.proc.wait()
        if rc:
            raise RuntimeError("FFmpeg encode failed:\n" + err[-4000:])
        os.replace(self.tmp, self.path)

    def abort(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.kill()
        try:
            self.tmp.unlink()
        except Exception:
            pass


# ----------------------------------------------------------------------------------------------
# Renderer
# ----------------------------------------------------------------------------------------------
class Renderer:
    def __init__(self, W, H, fps, fb, preview=False, grain=True, ca=True):
        self.W, self.H, self.fps, self.fb = W, H, fps, fb
        self.S = W / REF_W
        self.preview = preview
        self.shadow_lt = None
        self._vig = {}
        rng = np.random.default_rng(7)
        if grain:
            g = rng.standard_normal((H + 64, W + 64)).astype(np.float32)
            g = cv2.GaussianBlur(g, (0, 0), 0.9 * max(0.6, self.S))
            g /= g.std() + 1e-6
            self.grain = g
        else:
            self.grain = None
        self.ca = ca and not preview and bool(os.environ.get("AMAZON_CA"))
        self.grng = np.random.default_rng(11)
        gh, gw = int(math.ceil(H / 8)), int(math.ceil(W / 8))
        self.gh, self.gw = gh, gw
        self.gu = ((np.arange(gw, dtype=np.float32) + 0.5) * W / gw - 0.5)[None, :].repeat(gh, 0)
        self.gv = ((np.arange(gh, dtype=np.float32) + 0.5) * H / gh - 0.5)[:, None].repeat(gw, 1)
        self.cache = {}

    def vignette(self, strength):
        k = round(strength, 3)
        if k not in self._vig:
            self._vig[k] = vignette_map(self.W, self.H, k)
        return self._vig[k]

    # -- plate ---------------------------------------------------------------------------------
    def plate(self, ps, t, dx=0.0, dy=0.0, zoom=1.0):
        W, H = self.W, self.H
        tl = t - ps.start
        p = tl / ps.dur
        st = ps.cam.state(p, tl)
        du = (self.gu - W / 2) / zoom - dx - st["hx"] * W
        dv = (self.gv - H / 2) / zoom - dy - st["hy"] * W
        c, s = math.cos(st["rot"]), math.sin(st["rot"])
        ru, rv = du * c - dv * s, du * s + dv * c
        PW, PH = ps.PW, ps.PH
        lim_x = 0.5 - 0.5 / (OVERSCAN * min(st["zn"], st["zf"])) - abs(st["px"]) - 0.004
        lim_y = 0.5 - 0.5 / (OVERSCAN * min(st["zn"], st["zf"])) - abs(st["py"]) - 0.004
        cx = clamp(st["cx"], -max(lim_x, 0), max(lim_x, 0)) * PW + PW / 2
        cy = clamp(st["cy"], -max(lim_y, 0), max(lim_y, 0)) * PH + PH / 2
        zm = 0.5 * (st["zn"] + st["zf"])
        x0 = cx + ru / zm
        y0 = cy + rv / zm
        dq = ps.depth_q
        d = cv2.remap(dq, (x0 / 4).astype(np.float32), (y0 / 4).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        sc = st["zf"] + (st["zn"] - st["zf"]) * d
        mx = cx + ru / sc + st["px"] * PW * (d - 0.5)
        my = cy + rv / sc + st["py"] * PH * (d - 0.5)
        ctx = {"tl": tl, "p": p, "dur": ps.dur, "W": W, "H": H, "gu": self.gu, "gv": self.gv, "fps": self.fps,
               "cam_dx": (cx - PW / 2) + st["px"] * PW * 0.5, "cam_dy": (cy - PH / 2) + st["py"] * PH * 0.5}
        for f in ps.fx:
            f.distort(mx, my, ctx)
        mapx = cv2.resize(mx.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        mapy = cv2.resize(my.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        interp = cv2.INTER_LINEAR if self.preview else cv2.INTER_CUBIC
        img = cv2.remap(ps.plate, mapx, mapy, interp, borderMode=cv2.BORDER_REFLECT_101)
        frame = np.multiply(img, np.float32(1.0 / 255.0), dtype=np.float32)
        qw, qh = W // 4, H // 4
        if ps.needs_q:
            qx = cv2.resize(mx.astype(np.float32), (qw, qh), interpolation=cv2.INTER_LINEAR) / 4
            qy = cv2.resize(my.astype(np.float32), (qw, qh), interpolation=cv2.INTER_LINEAR) / 4
            ctx["qmaps"] = (qx, qy)
            ctx["depth_q"] = cv2.remap(dq, qx, qy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        Mq = ps.vig_q.copy()
        Aq = np.zeros((qh, qw, 3), np.float32)
        for f in ps.fx:
            f.lf(ctx, Mq, Aq)
        Aq *= ps.vig_q[..., None]
        frame *= cv2.resize(Mq, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
        if np.any(Aq):
            frame += cv2.resize(Aq, (W, H), interpolation=cv2.INTER_LINEAR)
        for f in ps.fx:
            f.draw(frame, ctx)
        return frame

    # -- transitions -----------------------------------------------------------------------------
    def transition(self, tr, t, A, B):
        W, H = self.W, self.H
        q = seg(t, tr.w0, tr.w1)
        k = tr.kind
        if k in ("dissolve", "slow_dissolve"):
            e = ease_in_out_sine(q)
            fa, fb = self.plate(A, t), self.plate(B, t)
            out = fa * (1 - e) + fb * e
            out *= 1 + 0.035 * math.sin(math.pi * q)
            return out
        if k in ("dip_black", "dip_night"):
            col = np.float32((0.0, 0.0, 0.0)) if k == "dip_black" else np.float32((0.006, 0.012, 0.03))
            if q < 0.5:
                e = ease_in_cubic(q / 0.5)
                return self.plate(A, t) * (1 - e) + col * e
            e = ease_out_cubic((q - 0.5) / 0.5)
            return self.plate(B, t) * e + col * (1 - e)
        if k.startswith("whip"):
            dxa, dya, dxb, dyb, vel, d = whip_offsets(k, q, W, H)
            horiz = d[0] != 0
            if q < 0.5:
                out = self.plate(A, t, dx=dxa, dy=dya)
                if q > 0.38:
                    other = self.plate(B, t, dx=dxb, dy=dyb)
                    w = seg(q, 0.38, 0.5) * 0.5
                    out = out * (1 - w) + other * w
            else:
                out = self.plate(B, t, dx=dxb, dy=dyb)
                if q < 0.62:
                    other = self.plate(A, t, dx=dxa, dy=dya)
                    w = (1 - seg(q, 0.5, 0.62)) * 0.5
                    out = out * (1 - w) + other * w
            blur = vel / self.fps * 0.9
            return directional_blur(out, min(blur, W * 0.25), horiz)
        if k == "zoom_through":
            if q < 0.5:
                e = ease_in_expo(q / 0.5)
                out = self.plate(A, t, zoom=1 + 0.9 * e)
                out = radial_blur(out, 0.25 * e)
            else:
                e = 1 - ease_out_expo((q - 0.5) / 0.5)
                out = self.plate(B, t, zoom=1 + 0.6 * e)
                out = radial_blur(out, 0.25 * e)
            flash = math.exp(-((q - 0.5) / 0.08) ** 2) * 0.25
            return out + (1 - out) * flash
        if k == "light_leak":
            e = ease_in_out_sine(q)
            out = self.plate(A, t) * (1 - e) + self.plate(B, t) * e
            if tr.mask_noise is None:
                r = np.random.default_rng(tr.seed)
                tr.mask_noise = fractal_noise(H // 8, W // 4, r, 3, 2.0)
            n = tr.mask_noise
            xs = np.linspace(0, 1, n.shape[1], dtype=np.float32)[None, :]
            pos = -0.3 + 1.6 * q
            band = np.exp(-((xs - pos) / 0.32) ** 2) * (0.5 + n)
            leak = cv2.resize(band.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
            col = np.float32((1.0, 0.62, 0.26))
            inten = math.sin(math.pi * q) ** 1.2 * 0.95
            out = 1 - (1 - out) * (1 - np.clip(leak * col * inten, 0, 1))
            return out
        if k == "ink":
            if tr.mask_noise is None:
                r = np.random.default_rng(tr.seed)
                h4, w4 = H // 4, W // 4
                n = fractal_noise(h4, w4, r, 5, 3.0)
                yy, xx = np.mgrid[0:h4, 0:w4].astype(np.float32)
                dmin = np.full((h4, w4), 9.0, np.float32)
                for _ in range(4):
                    sx, sy = r.random() * w4, r.random() * h4
                    dmin = np.minimum(dmin, np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2) / w4)
                f = 0.55 * n + 0.45 * (1 - np.clip(dmin / 0.9, 0, 1))
                lo, hi = f.min(), f.max()
                tr.mask_noise = (f - lo) / (hi - lo + 1e-6)
            g = ease_in_out_cubic(q) * 1.12
            thr = 1 - g
            f = tr.mask_noise
            m = np.clip((f - thr) / 0.05, 0, 1)
            band = np.exp(-((f - thr) / 0.022) ** 2)
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
            band = cv2.resize(band, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
            fa = self.plate(A, t) if q < 0.98 else None
            fb_ = self.plate(B, t) if q > 0.02 else None
            if fa is None:
                out = fb_
            elif fb_ is None:
                out = fa
            else:
                out = fa * (1 - m) + fb_ * m
            return out * (1 - band * 0.6 * math.sin(math.pi * min(1, q * 1.1)))
        if k == "flash":
            if t < tr.tc:
                return self.plate(A, t)
            dt = t - tr.tc
            sh = math.exp(-dt * 9) * 14 * self.S
            out = self.plate(B, t, dx=sh * math.sin(dt * 90), dy=sh * math.cos(dt * 77))
            f = math.exp(-dt * 11) * 0.85
            return out + (1 - out) * f
        if k == "blur_cut":
            if q < 0.5:
                e = ease_in_cubic(q / 0.5)
                out = cv2.GaussianBlur(self.plate(A, t), (0, 0), 1 + 14 * e * self.S)
            else:
                e = 1 - ease_out_cubic((q - 0.5) / 0.5)
                out = cv2.GaussianBlur(self.plate(B, t), (0, 0), 1 + 14 * e * self.S)
            return out
        return self.plate(B if t >= tr.tc else A, t)

    # -- finishing -------------------------------------------------------------------------------
    def finish(self, frame, t, events, plan):
        # graphics
        for ev in events:
            ev.draw(frame, t, self)
        # letterbox
        bars = max([ev.bars(t) for ev in plan.bar_events if ev.t0 - 1.5 <= t <= ev.t1 + 1.5] + [0.0])
        if bars > 0.001:
            target = (self.H - self.W / 2.20) / 2
            b = target * bars
            ib = int(b)
            frame[:ib] = 0
            frame[self.H - ib:] = 0
            fr = b - ib
            if ib < self.H // 2:
                frame[ib] *= 1 - fr
                frame[self.H - ib - 1] *= 1 - fr
        if self.grain is not None:
            self._gi = getattr(self, "_gi", -1) + 1
            if self._gi % 2 == 0 or not hasattr(self, "_goff"):  # grain "on twos": filmic and cheaper to encode
                self._goff = (int(self.grng.integers(0, 64)), int(self.grng.integers(0, 64)))
            ox, oy = self._goff
            amt = np.float32(0.009 if self.preview else 0.011)
            frame += (self.grain[oy:oy + self.H, ox:ox + self.W] * amt)[..., None]
        if self.ca:
            c = 0.0011
            for ch, s in ((0, 1 + c), (2, 1 - c)):
                M = cv2.getRotationMatrix2D((self.W / 2, self.H / 2), 0, s)
                frame[..., ch] = cv2.warpAffine(np.ascontiguousarray(frame[..., ch]), M, (self.W, self.H),
                                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        np.maximum(frame, 0, out=frame)
        return cv2.convertScaleAbs(frame, alpha=255.0, beta=0.5)


class Plan:
    pass


def parse_ranges(s, valid):
    out = set()
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return [n for n in valid if n in out]


def build_plan(manifest, audio_len, pauses, fps, args):
    scenes = manifest["scenes"]
    # scene timeline: proportional fit to the VO, cuts snapped to nearby pauses
    if all("start" in s for s in scenes):
        starts = [float(s["start"]) for s in scenes]
        man_len = max(float(s["start"]) + float(s["duration"]) for s in scenes)
    else:
        starts, acc = [], 0.0
        for s in scenes:
            starts.append(acc)
            acc += float(s["duration"])
        man_len = acc
    ratio = audio_len / man_len
    bounds = [st * ratio for st in starts] + [audio_len]
    snapped = 0
    if pauses and not args.no_snap:
        centers = np.array([(a + b) / 2 for a, b in pauses])
        for i in range(1, len(bounds) - 1):
            b = bounds[i]
            j = int(np.argmin(np.abs(centers - b)))
            if abs(centers[j] - b) <= 1.3:
                nb = float(centers[j])
                if nb - bounds[i - 1] > 3.0 and bounds[i + 1] - nb > 3.0:
                    bounds[i] = nb
                    snapped += 1
    plan = Plan()
    plan.fps = fps
    plan.total = audio_len
    plan.ratio = ratio
    plan.snapped = snapped
    plan.starts = bounds[:-1]
    plan.ends = bounds[1:]
    plan.frames = [int(round(b * fps)) for b in bounds]
    plan.frames[-1] = int(round(audio_len * fps))
    n = len(scenes)
    # transitions
    plan.trans = []
    for i in range(n - 1):
        k = scenes[i].get("transition_out", "cut")
        if k in ("cut", "", None) or k not in TRANS_DUR:
            continue
        tr = Trans(k, bounds[i + 1], i, i + 1, 900 + i)
        d = TRANS_DUR[k]
        lim = min(plan.ends[i] - plan.starts[i], plan.ends[i + 1] - plan.starts[i + 1]) * 0.45
        if d / 2 > lim:
            tr.w0, tr.w1 = tr.tc - lim, tr.tc + lim
        plan.trans.append(tr)
    plan.trans_starts = [tr.w0 for tr in plan.trans]
    plan.fade_in = 2.4
    end_d = 4.4
    plan.fade_out = (audio_len - end_d, audio_len - 2.2)
    # graphics events
    ev = []
    chapters = manifest["chapters"]
    ch_first = {c["start_scene"]: i + 1 for i, c in enumerate(chapters)}
    titles = getattr(args, "titles", "all")
    film = manifest.get("film", FILM)
    species = manifest.get("species", {})
    num_idx = {s["scene"]: i for i, s in enumerate(scenes)}
    total_sc = n
    open_t1 = min(7.2, plan.ends[0] - 0.4)
    ev.append(OpeningTitle(0.3, open_t1, film))
    hud_state = {"water": 1.0, "food": 0.4, "signal": 0.0, "battery": 0.62, "day": 1, "time": "الصباح"}
    cards = []
    for i, s in enumerate(scenes):
        t0, t1 = plan.starts[i], plan.ends[i]
        num = s["scene"]
        chi = s.get("chapter", 1)
        ch_name = chapters[chi - 1]["title_ar"] if 0 < chi <= len(chapters) else ""
        if num in ch_first:
            ci = ch_first[num]
            c0 = t0 + 0.25 if i > 0 else open_t1 + 0.25
            c1 = c0 + 3.7
            card = ChapterCard(c0, c1, ci, len(chapters), chapters[ci - 1]["title_ar"], chapters[ci - 1]["title_en"])
            ev.append(card)
            cards.append(card)
        if s.get("kinetic"):
            ev.append(Kinetic(t0 + 0.6, t1 - 0.35, s["kinetic"], s.get("kinetic_pos", "center")))
        elif s.get("species") and s["species"] in species and titles != "none":
            d = species[s["species"]]
            k = list(species.keys()).index(s["species"]) + 1
            ev.append(SpeciesCard(t0 + 0.6, min(t0 + 6.6, t1 - 0.4), d, k))
        elif titles == "all" and i > 0:
            a0 = t0 + 0.55
            for c in cards:
                if c.t0 - 0.3 < a0 + 3.4 and c.t1 > a0:
                    a0 = c.t1 + 0.15
            a1 = min(a0 + 3.6, t1 - 0.35)
            if a1 - a0 >= 2.4:
                ev.append(LowerThird(a0, a1, num, total_sc, s.get("title", ""), ch_name))
        if s.get("hud"):
            prev = dict(hud_state)
            for k2, v in s["hud"].items():
                if k2 != "hl":
                    hud_state[k2] = v
            h0 = t0 + 0.9
            ev.append(SurvivalHUD(h0, min(h0 + 5.2, t1 + 1.5), prev, dict(hud_state), s["hud"].get("hl", [])))
        if s.get("map"):
            if i + 1 < n:
                ev.append(MapCard(t0 + 0.5, plan.ends[i + 1] - 0.6, plan.starts[i + 1]))
            else:
                ev.append(MapCard(t0 + 0.5, t1 - 0.4, t1))
        if s.get("end"):
            ev.append(EndCard(audio_len - 3.6, audio_len, film))
    ev.sort(key=lambda e: e.z)
    plan.events = ev
    plan.bar_events = [e for e in ev if isinstance(e, (OpeningTitle, ChapterCard, EndCard))]
    # sound cues
    cues = []
    for e in ev:
        cues += e.sfx()
    for tr in plan.trans:
        for kind, off, dur, db in TRANS_SFX.get(tr.kind, []):
            cues.append((kind, tr.tc + off, dur, db))
    for i, s in enumerate(scenes):
        t0, dur = plan.starts[i], plan.ends[i] - plan.starts[i]
        fx = [f.split(":")[0] for f in s.get("fx", [])]
        if "lightning" in fx:
            for tp in (0.30, 0.74):
                cues.append(("thunder", t0 + tp * dur + 0.7, 3.0, -15))
        if "pulse" in fx:
            for tp in (0.34, 0.66):
                cues.append(("crackle", t0 + tp * dur, 0.5, -22))
        if "tension" in fx:
            per = 60 / 74.0
            k = 0.0
            while k < dur - 0.3:
                cues.append(("heartbeat", t0 + k, 0.5, -17))
                k += per
    plan.cues = cues
    return plan


def segment_hash(plan, manifest, mapping, i, settings):
    sc = manifest["scenes"]
    nb = [sc[j] for j in (i - 1, i, i + 1) if 0 <= j < len(sc)]
    h = hashlib.md5()
    h.update(json.dumps([VERSION, settings, plan.frames[i], plan.frames[i + 1],
                         [round(x, 4) for x in plan.starts[max(0, i - 1):i + 2]],
                         nb, [mapping[s["scene"]]["path"] for s in nb],
                         manifest.get("film"), manifest.get("chapters"), manifest.get("species")],
                        ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()[:20]


def main():
    ap = argparse.ArgumentParser(description="AMAZON cinematic documentary renderer")
    ap.add_argument("--images", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--output", default="Amazon_Documentary_Final.mp4")
    ap.add_argument("--preview", action="store_true", help="fast 854x480 whole-film preview")
    ap.add_argument("--check", action="store_true", help="only map images -> scenes and write contact sheet")
    ap.add_argument("--scenes", default="", help="render only these scenes, e.g. 1-6,17,66-67")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--crf", type=int, default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--work-dir", default="")
    ap.add_argument("--fonts-dir", default="")
    ap.add_argument("--depth", default="auto", choices=["auto", "model", "heuristic", "off"])
    ap.add_argument("--titles", default="all", choices=["all", "chapters", "none"])
    ap.add_argument("--no-sfx", action="store_true")
    ap.add_argument("--sfx-db", type=float, default=0.0)
    ap.add_argument("--music", default="")
    ap.add_argument("--music-db", type=float, default=-20.0)
    ap.add_argument("--no-snap", action="store_true")
    ap.add_argument("--no-loudnorm", action="store_true", help="keep absolute levels (no loudness normalisation)")
    ap.add_argument("--no-grain", action="store_true")
    ap.add_argument("--export-manifest", default="")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} not found. Install FFmpeg first.")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(args.work_dir) if args.work_dir else out.parent / (out.stem + "_work")
    work.mkdir(parents=True, exist_ok=True)
    log(f"AMAZON cinematic engine {VERSION}")
    manifest = load_manifest(args.manifest)
    if args.export_manifest:
        Path(args.export_manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"Manifest exported: {args.export_manifest}")
    fb = FontBook([args.fonts_dir] if args.fonts_dir else [])
    image_dir, audio = Path(args.images), Path(args.audio)
    if not image_dir.exists():
        raise RuntimeError(f"Image folder not found: {image_dir}")
    if not audio.exists() and not args.check:
        raise RuntimeError(f"Audio file not found: {audio}")

    mapping, extra = map_images(manifest, image_dir, fb, work)
    with open(work / "mapping.csv", "w", encoding="utf-8") as f:
        f.write("scene,title,image,source\n")
        for s in manifest["scenes"]:
            m = mapping[s["scene"]]
            f.write(f"{s['scene']},\"{s.get('title', '')}\",\"{Path(m['path']).name}\",{m['source']}\n")
    sheet = contact_sheet(manifest, mapping, fb, work / "contact_sheet.jpg")
    by_src = {}
    for m in mapping.values():
        k = m["source"].split(":")[0].split(" ")[0]
        by_src[k] = by_src.get(k, 0) + 1
    log(f"Image mapping: {by_src} | reference sheets ignored: {extra['reference']} | unused: {extra['spare']}")
    for s in manifest["scenes"]:
        m = mapping[s["scene"]]
        if m["source"] != "ocr":
            log(f"  scene {s['scene']:03d}: {m['source']}  ->  {Path(m['path']).name}")
    log(f"Contact sheet: {sheet}")
    if args.check:
        return

    if args.preview:
        W, fps, crf, preset = 854, args.fps or 24, args.crf or 26, args.preset or "veryfast"
    else:
        W, fps, crf, preset = args.width or 1920, args.fps or 24, args.crf or 20, args.preset or "fast"
    W -= W % 2
    H = int(round(W * 9 / 16))
    H -= H % 2

    audio_len = media_duration(audio)
    vo_rms, pauses = vo_analysis(audio)
    plan = build_plan(manifest, audio_len, pauses, fps, args)
    log(f"Audio {audio_len:.2f}s | storyboard fit x{plan.ratio:.4f} | cuts snapped to VO pauses: {plan.snapped} | "
        f"{len(manifest['scenes'])} scenes | {len(plan.trans)} styled transitions | {len(plan.events)} graphics | "
        f"{len(plan.cues)} sound cues")
    log(f"Output {W}x{H} @ {fps}fps  crf {crf}  preset {preset}")
    (work / "plan.json").write_text(json.dumps({
        "version": VERSION, "audio": str(audio), "duration": audio_len, "fps": fps, "size": [W, H],
        "scenes": [{"scene": s["scene"], "title": s.get("title"), "start": round(plan.starts[i], 3),
                    "end": round(plan.ends[i], 3), "image": Path(mapping[s["scene"]]["path"]).name,
                    "source": mapping[s["scene"]]["source"]} for i, s in enumerate(manifest["scenes"])]},
        ensure_ascii=False, indent=1), encoding="utf-8")

    R = Renderer(W, H, fps, fb, preview=args.preview, grain=not args.no_grain)
    depth = DepthEstimator(args.depth)
    depth_dir = work / "depth"
    depth_dir.mkdir(exist_ok=True)
    seg_dir = work / (f"segments_{W}x{H}_{fps}")
    seg_dir.mkdir(exist_ok=True)
    scenes = manifest["scenes"]
    nums = [s["scene"] for s in scenes]
    sel = parse_ranges(args.scenes, nums) if args.scenes else nums
    sel_idx = [nums.index(n) for n in sel]
    settings = [W, H, fps, crf, preset, args.preview, args.titles, args.depth, not args.no_grain]

    prepared = {}

    def get_ps(i):
        if i not in prepared:
            t0 = time.time()
            ps = prepare_scene(i, scenes[i], mapping, R, depth, depth_dir)
            ps.start, ps.end = plan.starts[i], plan.ends[i]
            ps.dur = ps.end - ps.start
            prepared[i] = ps
            for k in list(prepared):
                if k < i - 1:
                    del prepared[k]
            log(f"    prepared scene {scenes[i]['scene']:03d} in {time.time() - t0:.1f}s")
        return prepared[i]

    total_frames = sum(plan.frames[i + 1] - plan.frames[i] for i in sel_idx)
    done_frames, t_start = 0, time.time()
    seg_files = []
    for i in sel_idx:
        num = scenes[i]["scene"]
        f0, f1 = plan.frames[i], plan.frames[i + 1]
        hsh = segment_hash(plan, manifest, mapping, i, settings)
        segp = seg_dir / f"seg_{num:03d}_{hsh}.mp4"
        seg_files.append(segp)
        if segp.exists() and segp.stat().st_size > 1000:
            done_frames += f1 - f0
            log(f"Scene {num:03d}/{nums[-1]:03d} already rendered - skipped")
            continue
        for old in seg_dir.glob(f"seg_{num:03d}_*.mp4"):
            try:
                old.unlink()
            except Exception:
                pass
        log(f"Scene {num:03d}/{nums[-1]:03d}: {scenes[i].get('title', '')}  [{plan.starts[i]:.2f}s - {plan.ends[i]:.2f}s]")
        wr = FFWriter(segp, W, H, fps, crf, preset)
        try:
            for f in range(f0, f1):
                t = f / fps
                tr = None
                j = bisect.bisect_right(plan.trans_starts, t) - 1
                if j >= 0 and plan.trans[j].w0 <= t < plan.trans[j].w1:
                    tr = plan.trans[j]
                if tr is not None:
                    frame = R.transition(tr, t, get_ps(tr.a), get_ps(tr.b))
                else:
                    k = bisect.bisect_right(plan.starts, t) - 1
                    k = max(0, min(k, len(scenes) - 1))
                    frame = R.plate(get_ps(k), t)
                if t < plan.fade_in:
                    frame *= ease_in_out_sine(t / plan.fade_in)
                if t > plan.fade_out[0]:
                    frame *= 1 - ease_in_out_sine(seg(t, plan.fade_out[0], plan.fade_out[1]))
                evs = [e for e in plan.events if e.active(t)]
                wr.write(R.finish(frame, t, evs, plan))
                done_frames += 1
                if done_frames % (fps * 5) == 0:
                    el = time.time() - t_start
                    rate = done_frames / max(el, 1e-6)
                    eta = (total_frames - done_frames) / max(rate, 1e-6)
                    log(f"    {done_frames}/{total_frames} frames  {rate:.1f} fps  ETA {eta / 60:.1f} min")
            wr.close()
        except BaseException:
            wr.abort()
            raise

    # audio: VO + SFX (+ music) -> loudness normalised
    log("Building sound design and mix...")
    mix = work / "audio_mix.wav"
    inputs = ["-i", str(audio)]
    fc = "[0:a]aresample=48000,aformat=channel_layouts=stereo,volume=1.0[vo]"
    mixes = ["[vo]"]
    if not args.no_sfx:
        sfx = build_sfx_bus(plan.cues, audio_len, vo_rms, work / "sfx_bus.wav", args.sfx_db)
        inputs += ["-i", str(sfx)]
        fc += ";[1:a]aresample=48000[sfx]"
        mixes.append("[sfx]")
    if args.music:
        mi = len(inputs) // 2
        inputs += ["-stream_loop", "-1", "-i", str(args.music)]
        fc += (f";[{mi}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{audio_len:.3f},"
               f"volume={10 ** (args.music_db / 20):.4f}[mus];[vo]asplit=2[vo1][vosc];"
               f"[mus][vosc]sidechaincompress=threshold=0.04:ratio=6:attack=25:release=450[duck]")
        mixes[0] = "[vo1]"
        mixes.append("[duck]")
    fc += f";{''.join(mixes)}amix=inputs={len(mixes)}:normalize=0:duration=first[pre]"
    meas = subprocess.run(["true"]) if args.no_loudnorm else subprocess.run(["ffmpeg", "-hide_banner", "-nostats"] + inputs +
                          ["-filter_complex", fc + ";[pre]loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json[o]",
                           "-map", "[o]", "-f", "null", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ln = "loudnorm=I=-14:TP=-1.5:LRA=11"
    try:
        js = json.loads(meas.stderr[meas.stderr.rindex("{"):meas.stderr.rindex("}") + 1])
        vals = [float(js[k]) for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")]
        if not all(math.isfinite(v) for v in vals) or vals[0] < -70:
            raise ValueError("silent or unmeasurable audio")
        ln += (f":measured_I={js['input_i']}:measured_TP={js['input_tp']}:measured_LRA={js['input_lra']}"
               f":measured_thresh={js['input_thresh']}:offset={js['target_offset']}:linear=true")
    except Exception:
        pass
    if args.no_loudnorm:
        ln = "alimiter=limit=0.89"
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + inputs +
        ["-filter_complex", fc + f";[pre]{ln},aresample=48000[o]", "-map", "[o]", "-t", f"{audio_len:.3f}",
         "-c:a", "pcm_s16le", str(mix)], "audio mix")

    log("Assembling video...")
    lst = work / "concat.txt"
    with open(lst, "w", encoding="utf-8") as f:
        for p in seg_files:
            f.write("file '" + str(p.resolve()).replace("'", "'\\''") + "'\n")
    if sel_idx == list(range(len(scenes))):
        vid = work / "video_only.mp4"
        run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
             "-c", "copy", str(vid)], "concat")
        run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(vid), "-i", str(mix),
             "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
             "-t", f"{audio_len:.3f}", "-movflags", "+faststart", str(out)], "final mux")
    else:
        parts = []
        for k, i in enumerate(sel_idx):
            a0, a1 = plan.frames[i] / fps, plan.frames[i + 1] / fps
            parts.append(f"[1:a]atrim={a0:.4f}:{a1:.4f},asetpts=PTS-STARTPTS[a{k}]")
        afc = ";".join(parts) + ";" + "".join(f"[a{k}]" for k in range(len(sel_idx))) + f"concat=n={len(sel_idx)}:v=0:a=1[a]"
        run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
             "-i", str(mix), "-filter_complex", afc, "-map", "0:v", "-map", "[a]", "-c:v", "copy",
             "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", str(out)], "selection mux")
    el = time.time() - t_start
    log(f"SUCCESS: {out.resolve()} ({out.stat().st_size / 1024 / 1024:.1f} MB) in {el / 60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted - finished scenes are kept; run again to resume.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nRENDER ERROR: {e}", file=sys.stderr)
        sys.exit(1)
