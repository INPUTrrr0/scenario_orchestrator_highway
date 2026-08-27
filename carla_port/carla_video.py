#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_port/carla_video.py — record the run to an mp4.

Two CARLA RGB cameras, composited side by side and piped to ffmpeg:

    top     a bird's-eye view over the junction, oriented so the script frame's
            +y (North, the N arm) is up in the image. This is the view that
            shows what the orchestrator is doing — the ego, the hero, and every
            background actor at once.
    chase   a following camera behind the ego, which is what shows whether the
            ego policy is actually driving the route.

Frame capture is synchronous. Each camera's `listen` callback pushes into a
queue and the frame for tick N is taken out after `world.tick()` returns, so
the recording is frame-locked to the simulation rather than sampled off a
background thread. A missed frame reuses the previous one instead of shifting
the timeline.

A HUD is drawn over each frame with Pillow: the clock, the ego's speed and
cross-track error, the live D1-D4 verdict with its hero, and the last
intervention the orchestrator applied. That is what makes the mp4 a validation
artifact and not just a pretty picture — the verdict on screen is read from the
same `ds.Verdict` the session snapshots record.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Dict, List, Optional, Tuple

import numpy as np

from .carla_api import carla
# Frame/script types are needed for ANNOTATIONS ONLY (PEP 563: this
# module has `from __future__ import annotations`, so they are never
# evaluated at runtime). Importing them under TYPE_CHECKING keeps this
# module independent of WHICH map frame and WHICH script layer are in
# use, so carla_highway/ can reuse it with a HighwayFrame.
from typing import TYPE_CHECKING

if TYPE_CHECKING:                                # pragma: no cover
    from .carla_map import IntersectionFrame

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:                                   # pragma: no cover
    Image = ImageDraw = ImageFont = None

FONT_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")

TOP = "top"
CHASE = "chase"
VIEWS = (TOP, CHASE, "both")

# HUD colours (RGB)
FG = (238, 240, 246)
DIM = (158, 164, 176)
OK = (110, 210, 130)
BAD = (235, 105, 95)
WARN = (240, 185, 80)


def default_top_span(frame: IntersectionFrame) -> float:
    """Metres to fit across the top view's short axis.

    Wide enough to hold the junction and the whole stretch of approach the
    generated scenarios actually spawn on (`scenarios.build_scenario` places
    actors at most `arm_length - 4` m out, capped at 56), so nothing enters the
    picture from nowhere; but not the full 60 m arm, from whose height CARLA's
    atmospheric fade washes the vehicles out and a car is a dozen pixels.
    """
    return 2.0 * (min(frame.arm_length, 42.0) + 1.5 * frame.lane_width)


def _font(size: int):
    if ImageFont is None:
        return None
    for p in FONT_PATHS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# What the HUD prints
# --------------------------------------------------------------------------- #
@dataclass
class Hud:
    """One frame's worth of annotation. Filled by the runner each step."""
    t: float = 0.0
    mode: str = ""
    seed: int = 0
    town: str = ""
    junction: object = None
    ego_speed: float = 0.0
    ego_mode: str = ""
    xtrack: float = 0.0
    signals: Dict[str, str] = field(default_factory=dict)
    d: Dict[str, bool] = field(default_factory=dict)
    hero: Optional[str] = None
    hero_arm: Optional[str] = None
    ok: bool = True
    interventions: int = 0
    last_intervention: str = ""
    predicted: str = ""
    realized: str = ""
    #: Optional overrides so a backend whose world is not a signalized junction
    #: can fill the same two rows with its own terms. When left None the
    #: intersection rendering above is used unchanged.
    line3: Optional[str] = None           # replaces the "signals ..." row
    badges: Optional[List[Tuple[str, bool]]] = None   # replaces D1..D4
    hero_label: str = "hero"              # what the highlighted actor is called
    #: Optional role-casting card (highway orchestrator panel). `roles` maps
    #: actor id -> assigned intention key; `role_columns` is
    #: [(key, label), ...] in display order (defaults to none|cut-in|block);
    #: `role_scores` is {actor_id: {role_key: 0..1}} for the % beside each light.
    roles: Optional[Dict[str, str]] = None
    role_columns: Optional[List[Tuple[str, str]]] = None
    role_scores: Optional[Dict[str, Dict[str, float]]] = None


def _text_width(dr, text: str, font) -> float:
    """Width of `text` in pixels, across Pillow versions.

    `ImageDraw.textlength` arrived in Pillow 8.0; `textsize` was the way before
    that and is gone in Pillow 10. Supporting both keeps the HUD working on the
    Pillow that ships with a CARLA 0.9.16 / Python 3.8 environment (7.2 here)
    without pinning anything.
    """
    for attempt in (lambda: dr.textlength(text, font=font),
                    lambda: dr.textsize(text, font=font)[0],
                    lambda: font.getsize(text)[0]):
        try:
            return attempt()
        except AttributeError:
            continue
    return len(text) * 7.0


class HudPainter:
    def __init__(self, width: int, height: int):
        self.big = _font(max(15, height // 34))
        self.small = _font(max(13, height // 42))
        self.width = width
        self.height = height

    @staticmethod
    def _clip(dr, text: str, font, budget: float) -> str:
        """Ellipsize `text` to fit `budget` pixels. The intervention line is a
        whole repair plan — several retimes and reroutes with their costs — and
        it can be longer than the frame is wide."""
        if _text_width(dr, text, font) <= budget:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _text_width(dr, text[:mid] + "…", font) <= budget:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo] + "…"

    def paint(self, rgb: "np.ndarray", hud: Hud) -> "np.ndarray":
        if Image is None:
            return rgb
        img = Image.fromarray(rgb)
        dr = ImageDraw.Draw(img, "RGBA")
        pad = 10
        lines_top = [
            (f"t={hud.t:5.2f}s   {hud.mode}   seed {hud.seed}   "
             f"{hud.town} j{hud.junction}", FG),
            (f"ego  {hud.ego_speed:5.2f} m/s   cross-track "
             f"{hud.xtrack:4.2f} m   ({hud.ego_mode})", DIM),
            (hud.line3 if hud.line3 is not None else
             ("signals  " + "  ".join(f"{k}:{v}" for k, v in
                                      sorted(hud.signals.items()))), DIM),
        ]
        h = 4 + len(lines_top) * (self.big.size + 6 if self.big else 20)
        dr.rectangle([0, 0, self.width, h + 6], fill=(14, 16, 22, 190))
        y = 4
        for text, col in lines_top:
            dr.text((pad, y), self._clip(dr, text, self.big, self.width - 2 * pad),
                    font=self.big, fill=col)
            y += (self.big.size + 6) if self.big else 20

        # bottom band: the orchestrator's verdict and its last action
        badge = []
        if hud.badges is not None:
            for label, good in hud.badges:
                badge.append((f"{label}{'+' if good else '-'} ",
                              OK if good else BAD))
        else:
            for k in ("d1", "d2", "d3", "d4"):
                badge.append((f"{k.upper()}{'+' if hud.d.get(k) else '-'} ",
                              OK if hud.d.get(k) else BAD))
        hero = hud.hero or "none"
        if hud.hero and hud.hero_arm:
            hero = f"{hud.hero} (arm {hud.hero_arm})"
        lines_bot = [
            ("verdict  ", DIM), *badge,
            (f"  {hud.hero_label} ", DIM), (hero, WARN if hud.hero else DIM),
            ("  ok ", DIM), ("yes" if hud.ok else "no", OK if hud.ok else BAD),
        ]
        step = (self.small.size + 5) if self.small else 18
        nrows = 3
        top = self.height - (nrows * step + 12)
        dr.rectangle([0, top - 6, self.width, self.height],
                     fill=(14, 16, 22, 190))
        x, y = pad, top
        for text, col in lines_bot:
            dr.text((x, y), text, font=self.small, fill=col)
            x += int(_text_width(dr, text, self.small))
        y += step
        budget = self.width - 2 * pad
        dr.text((pad, y), self._clip(
                    dr, f"last intervention ({hud.interventions})  "
                        f"{hud.last_intervention or '-'}", self.small, budget),
                font=self.small, fill=FG if hud.last_intervention else DIM)
        y += step
        dr.text((pad, y), self._clip(
                    dr, f"collision  predicted {hud.predicted or '-'}   "
                        f"realized {hud.realized or '-'}", self.small, budget),
                font=self.small,
                fill=BAD if (hud.predicted or hud.realized) else DIM)

        if hud.roles:
            # Sit just under the top telemetry band, on the left (top-down
            # half when view=both) — same place the pygame editor puts it.
            self._paint_role_panel(dr, hud, origin=(pad, h + 14))
        return np.asarray(img)

    def _paint_role_panel(self, dr, hud: Hud,
                          origin: Tuple[int, int] = (10, 70)) -> None:
        """Pillow stand-in for highway/cutin_orchestrator.draw_role_panel.

        Drawn into the recorded frame so the mp4 shows the same intention
        matrix the editor's top-left orchestrator card does (green light on
        the assigned role, hollow otherwise, optional score %).
        """
        roles = hud.roles or {}
        cols = hud.role_columns or [
            ("nominal", "none"), ("cutin", "cut-in"), ("block", "block")]
        scores = hud.role_scores or {}
        x, y = origin
        n_cols = max(1, len(cols))
        width = max(332, 142 + n_cols * 72 + 16)
        # Keep the card on the left pane when the frame is a side-by-side
        # top+chase composite (width ≈ 2× one camera).
        max_w = self.width // 2 - 16 if self.width > self.height * 1.4 else \
            self.width - 20
        width = min(width, max_w)
        row_h, header_h, colhdr_h, pad = 22, 24, 18, 6
        col_x0 = x + 110
        col_w = max(40, (width - (col_x0 - x) - 8) // n_cols)
        height = header_h + colhdr_h + max(1, len(roles)) * row_h + pad
        panel_bg = (22, 24, 30, 210)
        banner = (36, 40, 50, 230)
        muted = (150, 156, 168, 255)
        txt = (230, 232, 238, 255)
        on = (72, 190, 100, 255)
        off = (58, 62, 72, 255)

        dr.rectangle([x, y, x + width, y + height], fill=panel_bg,
                     outline=(70, 74, 86, 255))
        dr.rectangle([x, y, x + width, y + header_h], fill=banner)
        dr.text((x + 10, y + 5), "orchestrator", font=self.small, fill=txt)

        hdr_y = y + header_h + 2
        for i, (_, lbl) in enumerate(cols):
            cx = col_x0 + i * col_w + col_w // 2
            tw = _text_width(dr, lbl, self.small)
            dr.text((cx - tw / 2, hdr_y), lbl, font=self.small, fill=muted)
        for i in range(n_cols + 1):
            sx = col_x0 + i * col_w
            dr.line([(sx, y + header_h + colhdr_h - 2),
                     (sx, y + height - pad + 2)],
                    fill=(42, 46, 56, 255), width=1)

        yy = y + header_h + colhdr_h
        for aid, role in roles.items():
            dr.text((x + 10, yy + 4), f"actor {aid}",
                    font=self.small, fill=txt)
            for i, (rkey, _) in enumerate(cols):
                val = scores.get(str(aid), scores.get(aid, {})).get(rkey)
                cx = col_x0 + i * col_w + (14 if val is not None else col_w // 2)
                cy = yy + row_h // 2
                r = 6
                if role == rkey:
                    dr.ellipse([cx - r, cy - r, cx + r, cy + r],
                               fill=on, outline=(20, 22, 28, 255))
                else:
                    dr.ellipse([cx - r, cy - r, cx + r, cy + r],
                               outline=off, width=1)
                if val is not None:
                    vc = on if role == rkey else muted
                    dr.text((cx + 9, cy - 6), f"{val * 100.0:3.0f}%",
                            font=self.small, fill=vc)
            yy += row_h


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #
class CameraRig:
    """One or two RGB cameras, read synchronously."""

    def __init__(self, world, frame: IntersectionFrame, view: str = "both",
                 width: int = 720, height: int = 540, fov: float = 90.0,
                 top_span: Optional[float] = None):
        if view not in VIEWS:
            raise ValueError(f"unknown view {view!r}; expected one of {VIEWS}")
        self.world = world
        self.frame = frame
        self.view = view
        self.width = width
        self.height = height
        self.fov = fov
        self.top_span = top_span or default_top_span(frame)
        self.sensors: List[object] = []
        self.queues: Dict[str, "Queue"] = {}
        self._last: Dict[str, "np.ndarray"] = {}
        self.misses = 0

    # ---- placement ---- #
    def _blueprint(self):
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.width))
        bp.set_attribute("image_size_y", str(self.height))
        bp.set_attribute("fov", str(self.fov))
        bp.set_attribute("sensor_tick", "0.0")     # one image per world tick
        return bp

    def _top_transform(self):
        """Straight down over the junction. For a camera pitched -90 the image's
        up direction is its yaw direction projected on the ground, so yawing it
        along the script's +y axis puts the N arm at the top of the frame — the
        same orientation the pygame view in v4/drivev2.py uses.

        Height is chosen so both arms fit in the SHORT image axis: CARLA's `fov`
        is the horizontal field of view, so a 4:3 frame sees only three quarters
        as much vertically, and the N/S arms are exactly as long as the E/W
        ones."""
        a = self.frame.anchor
        aspect = max(1.0, self.width / max(self.height, 1))
        z = 0.5 * self.top_span * aspect / math.tan(math.radians(0.5 * self.fov))
        return carla.Transform(
            carla.Location(x=a.x, y=a.y, z=a.z + z),
            carla.Rotation(pitch=-90.0, yaw=self.frame.to_carla_yaw(90.0),
                           roll=0.0))

    def attach(self, ego_actor=None) -> int:
        """Spawn the cameras. `ego_actor` is required for the chase view."""
        bp = self._blueprint()
        if self.view in (TOP, "both"):
            cam = self.world.spawn_actor(bp, self._top_transform())
            self._register(TOP, cam)
        if self.view in (CHASE, "both") and ego_actor is not None:
            cam = self.world.spawn_actor(
                bp, carla.Transform(carla.Location(x=-7.0, z=3.4),
                                    carla.Rotation(pitch=-13.0)),
                attach_to=ego_actor,
                attachment_type=carla.AttachmentType.SpringArmGhost)
            self._register(CHASE, cam)
        return len(self.sensors)

    def _register(self, name: str, cam) -> None:
        q: "Queue" = Queue()
        cam.listen(q.put)
        self.sensors.append(cam)
        self.queues[name] = q

    # ---- reading ---- #
    def grab(self, timeout: float = 4.0) -> Optional["np.ndarray"]:
        """The composited RGB frame for the tick that just completed, or None
        before the first frame of every camera has arrived."""
        parts = []
        for name in (TOP, CHASE):
            q = self.queues.get(name)
            if q is None:
                continue
            img = self._one(name, q, timeout)
            if img is None:
                return None
            parts.append(img)
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else np.hstack(parts)

    def _one(self, name: str, q: "Queue", timeout: float):
        try:
            image = q.get(timeout=timeout)
        except Empty:
            self.misses += 1
            return self._last.get(name)     # hold the last frame, keep the clock
        raw = np.frombuffer(image.raw_data, dtype=np.uint8)
        bgra = raw.reshape((image.height, image.width, 4))
        rgb = np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])
        self._last[name] = rgb
        return rgb

    def drain(self) -> None:
        """Discard whatever is queued. Used for the settling ticks before the
        recording starts, so frame 0 of the video is step 0 of the run."""
        for name, q in self.queues.items():
            while True:
                try:
                    q.get_nowait()
                except Empty:
                    break

    def destroy(self) -> None:
        for cam in self.sensors:
            try:
                cam.stop()
            except RuntimeError:
                pass
            try:
                cam.destroy()
            except RuntimeError:
                pass
        self.sensors.clear()
        self.queues.clear()


# --------------------------------------------------------------------------- #
# ffmpeg
# --------------------------------------------------------------------------- #
class VideoWriter:
    """Raw RGB frames -> H.264 mp4, through an ffmpeg pipe."""

    def __init__(self, path: str, width: int, height: int, fps: float,
                 crf: int = 20):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is not on PATH; --no-video to skip "
                               "recording")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.width, self.height = width, height
        self.frames = 0
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
             "-video_size", f"{width}x{height}", "-framerate", f"{fps:g}",
             "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
             "-crf", str(crf), "-preset", "medium", "-movflags", "+faststart",
             "-loglevel", "error", path],
            stdin=subprocess.PIPE)

    def write(self, rgb: "np.ndarray") -> None:
        if self.proc is None:
            return
        if rgb.shape[0] != self.height or rgb.shape[1] != self.width:
            raise ValueError(f"frame {rgb.shape} != {(self.height, self.width)}")
        self.proc.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self) -> Optional[str]:
        if self.proc is None:
            return self.path
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        self.proc.wait()
        rc = self.proc.returncode
        self.proc = None
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited {rc} writing {self.path}")
        return self.path


class Recorder:
    """CameraRig + HudPainter + VideoWriter, as one thing the loop can call."""

    def __init__(self, world, frame: IntersectionFrame, path: str,
                 view: str = "both", width: int = 720, height: int = 540,
                 fps: float = 30.0, sim_fps: float = 60.0, hud: bool = True,
                 top_span: Optional[float] = None):
        self.rig = CameraRig(world, frame, view=view, width=width, height=height,
                             top_span=top_span)
        self.path = path
        self.fps = fps
        self.stride = max(1, int(round(sim_fps / max(fps, 1e-6))))
        self.writer: Optional[VideoWriter] = None
        self.painter: Optional[HudPainter] = None
        self.use_hud = hud and Image is not None
        self.hud_unavailable = hud and Image is None
        self._n = 0

    def attach(self, ego_actor=None) -> int:
        return self.rig.attach(ego_actor)

    def start(self) -> None:
        self.rig.drain()
        self._n = 0

    def capture(self, hud: Hud) -> bool:
        """Take the frame the last `world.tick()` produced. Returns True when a
        frame was written. Must be called once per tick even when it is not a
        recorded one, so the sensor queues cannot run away."""
        rgb = self.rig.grab()
        n, self._n = self._n, self._n + 1
        if rgb is None or n % self.stride:
            return False
        if self.writer is None:
            h, w = rgb.shape[0], rgb.shape[1]
            self.writer = VideoWriter(self.path, w, h, self.fps)
            self.painter = HudPainter(w, h)
        if self.use_hud:
            rgb = self.painter.paint(rgb, hud)
        self.writer.write(rgb)
        return True

    @property
    def frames(self) -> int:
        return self.writer.frames if self.writer else 0

    def close(self) -> Optional[str]:
        self.rig.destroy()
        if self.writer is None:
            return None
        return self.writer.close()
